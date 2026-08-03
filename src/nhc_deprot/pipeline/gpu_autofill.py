"""GPU teacher group queue: global endpoint queue + dynamic GPU claim.

Products: teacher_gpu_g00N/. User-facing: g00N teacher (not “Autofill”).
State dir: gpu_teacher_queue/ (legacy: gpu_autofill/).

Authority: docs/contracts/COMPUTE_DISPATCH_V001.md + RIGID_SMALL_NHC_POOL_V001.
Does not open Final Test. Does not train. Parent backend = gpu4pyscf only.
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.pipeline.live_teacher import LiveParentTeacherEngine
from nhc_deprot.pipeline.teacher_runner import endpoint_charge_mult

G001_ROOTS: frozenset[str] = frozenset(TRAIN_ROOTS) | frozenset(VALIDATION_ROOTS)
DEFAULT_XYZ_SEARCH: tuple[str, ...] = (
    "/home/plab/test/WJW/data/runs/mol_gold/xyz",
    "/home/plab/test/WJW/data/candidates/structures_full/xyz",
    "/home/plab/test/WJW/data/candidates/xyz",
)
GRAD_MAX = 4.5e-4
GRAD_RMS = 3.0e-4


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pool_inchikeys(pool_csv: Path) -> list[str]:
    import csv

    rows = list(csv.DictReader(pool_csv.open(encoding="utf-8")))
    key = "inchikey" if "inchikey" in (rows[0] if rows else {}) else "InChIKey"
    out: list[str] = []
    seen: set[str] = set()
    for r in rows:
        ik = (r.get(key) or "").strip()
        if not ik or ik in seen:
            continue
        seen.add(ik)
        out.append(ik)
    return out


def resolve_xyz_dirs(search: Sequence[str]) -> list[Path]:
    return [Path(p) for p in search if Path(p).is_dir()]


def has_pair(root_id: str, xyz_dirs: Sequence[Path]) -> Path | None:
    for d in xyz_dirs:
        if (d / f"{root_id}_cation.xyz").is_file() and (
            d / f"{root_id}_neutral.xyz"
        ).is_file():
            return d
    return None


def list_busy_gpu_ids() -> set[int]:
    """GPUs currently pinned by NHC0801 parent workers."""
    busy: set[int] = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            cmd = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "nhc0801_pyscf_parent_worker" not in cmd:
            continue
        try:
            env = open(f"/proc/{name}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        for e in env:
            if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                v = e.split(b"=", 1)[1].decode().strip()
                if v.isdigit():
                    busy.add(int(v))
    return busy


def list_free_gpu_ids(all_ids: Sequence[int]) -> list[int]:
    busy = list_busy_gpu_ids()
    return [i for i in all_ids if i not in busy]


def endpoint_done_ok(out_root: Path, root_id: str, endpoint: str) -> bool:
    d = out_root / root_id / endpoint
    man = d / "manifest.json"
    f1 = d / "frame_0001.json"
    if not man.is_file() or not f1.is_file():
        return False
    try:
        m = json.loads(man.read_text(encoding="utf-8"))
        fr = json.loads(f1.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if m.get("live_chemistry") is not True or m.get("dry_run") is True:
        return False
    if (fr.get("functional") or "").lower().replace("_", "-") != "wb97m-d3bj":
        return False
    if fr.get("basis") != "def2-TZVPP":
        return False
    g = fr.get("gradient_hartree_per_bohr") or []
    flat = [abs(float(x)) for row in g for x in row]
    if not flat:
        return False
    gmax = max(flat)
    grms = (sum(x * x for x in flat) / len(flat)) ** 0.5
    return gmax < GRAD_MAX and grms < GRAD_RMS


@dataclass
class AutofillState:
    """Queue state for g00N teacher (kept class name for import stability)."""

    schema: str = "nhc0801-gpu-teacher-queue-v1"
    generation_id: str = "nhc0801-g001"
    pool_csv: str = ""
    xyz_dirs: list[str] = field(default_factory=list)
    exclude_roots: list[str] = field(default_factory=list)
    batch_size_roots: int = 5
    gpu_ids: list[int] = field(default_factory=lambda: list(range(8)))
    next_batch_index: int = 3  # g003 next if 2 done
    queue: list[dict[str, Any]] = field(default_factory=list)
    done: list[dict[str, Any]] = field(default_factory=list)
    failed: list[dict[str, Any]] = field(default_factory=list)
    running: dict[str, dict[str, Any]] = field(default_factory=dict)
    batches: dict[str, dict[str, Any]] = field(default_factory=dict)
    stop_reason: str | None = None
    updated_at_utc: str = ""

    def save(self, path: Path) -> None:
        self.updated_at_utc = _utc()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2, sort_keys=True) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> AutofillState:
        if not path.is_file():
            return cls()
        raw = json.loads(path.read_text(encoding="utf-8"))
        return cls(**{k: raw[k] for k in cls.__dataclass_fields__ if k in raw})


def build_queue(
    *,
    pool: Sequence[str],
    exclude: set[str],
    xyz_dirs: Sequence[Path],
    out_root_for_done: Path | None = None,
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    """Return (queue tasks, skipped_no_xyz, already_done_roots)."""
    queue: list[dict[str, Any]] = []
    no_xyz: list[str] = []
    done_roots: list[str] = []
    for root_id in pool:
        if root_id in exclude or root_id in G001_ROOTS:
            continue
        xyz_dir = has_pair(root_id, xyz_dirs)
        if xyz_dir is None:
            no_xyz.append(root_id)
            continue
        both_done = False
        if out_root_for_done is not None:
            both_done = endpoint_done_ok(
                out_root_for_done, root_id, "cation"
            ) and endpoint_done_ok(out_root_for_done, root_id, "neutral")
        # also scan generation teacher_gpu_* dirs
        if not both_done:
            # check any sibling later
            pass
        if both_done:
            done_roots.append(root_id)
            continue
        for ep in ("cation", "neutral"):
            queue.append(
                {
                    "root_id": root_id,
                    "endpoint": ep,
                    "gold_xyz_dir": str(xyz_dir),
                    "key": f"{root_id}:{ep}",
                }
            )
    return queue, no_xyz, done_roots


def assign_batches(roots_in_order: Sequence[str], batch_size: int, start_index: int) -> dict[str, list[str]]:
    batches: dict[str, list[str]] = {}
    idx = start_index
    for i in range(0, len(roots_in_order), batch_size):
        chunk = list(roots_in_order[i : i + batch_size])
        if not chunk:
            break
        if len(chunk) < batch_size:
            # pad not allowed — leave short final batch as-is only if user allows; here still emit
            pass
        name = f"g{idx:03d}"
        batches[name] = chunk
        idx += 1
    return batches


def run_one_endpoint_job(
    *,
    root_id: str,
    endpoint: str,
    gold_xyz_dir: str,
    out_dir: Path,
    gpu_index: int,
    max_steps: int,
    host_threads: int,
) -> dict[str, Any]:
    for k in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[k] = str(host_threads)
    os.environ.pop("CUDA_VISIBLE_DEVICES", None)
    eng = LiveParentTeacherEngine(
        gold_xyz_dir=Path(gold_xyz_dir),
        max_steps=max_steps,
        backend="gpu",
        cuda_device=int(gpu_index),
        host_threads=host_threads,
    )
    charge, mult = endpoint_charge_mult(endpoint)
    t0 = time.perf_counter()
    result = eng.run_endpoint(
        root_id=root_id,
        endpoint=endpoint,
        charge=charge,
        multiplicity=mult,
        output_dir=out_dir,
    )
    wall = time.perf_counter() - t0
    ok = bool(result.get("converged")) and int(result.get("frame_count") or 0) >= 2
    # post gradient gate
    f1 = out_dir / "frame_0001.json"
    gate_ok = False
    if f1.is_file():
        fr = json.loads(f1.read_text(encoding="utf-8"))
        g = fr.get("gradient_hartree_per_bohr") or []
        flat = [abs(float(x)) for row in g for x in row]
        if flat:
            gmax = max(flat)
            grms = (sum(x * x for x in flat) / len(flat)) ** 0.5
            gate_ok = gmax < GRAD_MAX and grms < GRAD_RMS
    status = "PASS" if ok and gate_ok else ("PARTIAL" if ok else "FAIL")
    return {
        "root_id": root_id,
        "endpoint": endpoint,
        "gpu_index": gpu_index,
        "status": status,
        "wall_seconds": wall,
        "gate_ok": gate_ok,
        "converged": result.get("converged"),
        "frame_count": result.get("frame_count"),
    }


def spawn_endpoint(
    *,
    state_dir: Path,
    nhc0801_root: Path,
    generation_id: str,
    batch_id: str,
    task: dict[str, Any],
    gpu_index: int,
    max_steps: int,
    host_threads: int,
) -> subprocess.Popen[str]:
    """Spawn a detached one-endpoint job; log under state_dir/jobs/."""
    jobs = state_dir / "jobs"
    jobs.mkdir(parents=True, exist_ok=True)
    key = task["key"].replace(":", "_")
    log = jobs / f"{batch_id}_{key}_gpu{gpu_index}.out"
    out_subdir = f"teacher_gpu_{batch_id}"
    py = f"""
import json, sys
from pathlib import Path
sys.path.insert(0, {str(nhc0801_root / "src")!r})
from nhc_deprot.pipeline.gpu_autofill import run_one_endpoint_job
out = Path({str(nhc0801_root)!r}) / "runs" / {generation_id!r} / {out_subdir!r} / {task["root_id"]!r} / {task["endpoint"]!r}
r = run_one_endpoint_job(
    root_id={task["root_id"]!r},
    endpoint={task["endpoint"]!r},
    gold_xyz_dir={task["gold_xyz_dir"]!r},
    out_dir=out,
    gpu_index={gpu_index},
    max_steps={max_steps},
    host_threads={host_threads},
)
print(json.dumps(r))
print("JOB_EXIT", r.get("status"))
"""
    env = os.environ.copy()
    env["PYTHONPATH"] = str(nhc0801_root / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["OMP_NUM_THREADS"] = str(host_threads)
    env["MKL_NUM_THREADS"] = str(host_threads)
    env["OPENBLAS_NUM_THREADS"] = str(host_threads)
    fh = open(log, "w", encoding="utf-8")
    return subprocess.Popen(
        ["python3", "-c", py],
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
        env=env,
        cwd=str(nhc0801_root),
    )
