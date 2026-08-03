#!/usr/bin/env python3
"""GPU g00N teacher wave (gpu4pyscf) — stable entry.

Products: teacher_gpu_g00N/ (default g002). Not pilot Train for g001.


Runs Parent-P01 (wb97m-d3bj / def2-TZVPP) on gold XYZ endpoints using
gpu4pyscf.dft.RKS, one physical GPU per concurrent worker.

Default queue: 10 endpoints from the first 5 complete gold roots that are
NOT the pilot Train/Val set — so this does NOT race the g001 pilot teacher
wave writing under teacher_gpu_g001/.

Outputs under (uniform group layout):
  runs/<gen>/teacher_gpu_g002/<root>/{cation,neutral}/
  runs/<gen>/teacher_gpu_g002/campaign_receipt_live_gpu.json

Notes:
  - g00N inventory compute: NOT auto-added to g001 Train split.
  - Final Test identities never selected.
  - Host OMP kept tiny (default 2) so CPU pool 0-99 stays for CPU teacher.
  - Pin CUDA_VISIBLE_DEVICES to a single id per process (gpu4pyscf multi-GPU unsafe).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.contracts.parent_protocol import (  # noqa: E402
    BASIS,
    FUNCTIONAL,
    PROTOCOL_SHA256,
)
from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.pipeline.live_teacher import LiveParentTeacherEngine  # noqa: E402
from nhc_deprot.pipeline.teacher_runner import endpoint_charge_mult  # noqa: E402

WAVE_LABEL = "g00n_gpu_teacher"
PILOT_ROOTS = set(TRAIN_ROOTS) | set(VALIDATION_ROOTS)


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def list_complete_gold_roots(gold_xyz_dir: Path) -> list[str]:
    roots: list[str] = []
    for p in sorted(gold_xyz_dir.glob("*_cation.xyz")):
        root_id = p.name[: -len("_cation.xyz")]
        if (gold_xyz_dir / f"{root_id}_neutral.xyz").is_file():
            roots.append(root_id)
    return roots


def pick_expansion_roots(
    gold_xyz_dir: Path,
    *,
    n_roots: int,
    exclude: set[str] | None = None,
) -> list[str]:
    """Pick complete gold pairs outside pilot + exclude set (lexicographic)."""
    blocked = set(PILOT_ROOTS) | set(exclude or ())
    out: list[str] = []
    for root_id in list_complete_gold_roots(gold_xyz_dir):
        if root_id in blocked:
            continue
        low = root_id.lower()
        if "final_test" in low or low.startswith("ft_"):
            continue
        out.append(root_id)
        if len(out) >= n_roots:
            break
    if len(out) < n_roots:
        raise SystemExit(
            f"need {n_roots} group roots outside blocked={len(blocked)}; "
            f"found {len(out)} in {gold_xyz_dir}"
        )
    return out


def _run_one_endpoint(
    *,
    nhc0801_root: str,
    generation_id: str,
    root_id: str,
    endpoint: str,
    gold_xyz_dir: str,
    max_steps: int,
    gpu_index: int,
    host_threads: int,
    out_root: str,
) -> dict:
    try:
        # Host threads small; GPU does the heavy lifting.
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            os.environ[k] = str(host_threads)
        # Do NOT pre-pin CUDA here: LiveParentP01Engine sets
        # CUDA_VISIBLE_DEVICES=<physical gpu_index> on the worker subprocess.
        # Pre-setting then overwriting to "0" would collapse all jobs onto GPU0.
        os.environ.pop("CUDA_VISIBLE_DEVICES", None)

        eng = LiveParentTeacherEngine(
            gold_xyz_dir=Path(gold_xyz_dir),
            max_steps=max_steps,
            backend="gpu",
            cuda_device=int(gpu_index),  # physical id → worker env pin
            host_threads=host_threads,
        )

        charge, mult = endpoint_charge_mult(endpoint)
        out_dir = Path(out_root) / root_id / endpoint
        print(
            f"[gpu-teacher] START gpu={gpu_index} {root_id}/{endpoint} "
            f"t_host={host_threads}",
            flush=True,
        )
        result = eng.run_endpoint(
            root_id=root_id,
            endpoint=endpoint,
            charge=charge,
            multiplicity=mult,
            output_dir=out_dir,
        )
        status = (
            "PASS"
            if result.get("converged") and int(result.get("frame_count") or 0) >= 2
            else "PARTIAL"
        )
        print(
            f"[gpu-teacher] END gpu={gpu_index} {root_id}/{endpoint} -> {status} "
            f"frames={result.get('frame_count')} wall={result.get('wall_seconds')}",
            flush=True,
        )
        return {
            "root_id": root_id,
            "endpoint": endpoint,
            "status": status,
            "gpu_index": gpu_index,
            "backend": "gpu4pyscf",
            "frame_count": result.get("frame_count"),
            "converged": result.get("converged"),
            "wall_seconds": result.get("wall_seconds"),
            "output_dir": str(out_dir),
            "live_chemistry": True,
            "dry_run": False,
            "functional": FUNCTIONAL,
            "basis": BASIS,
            "parent_protocol_sha256": PROTOCOL_SHA256,
        }
    except Exception as exc:  # noqa: BLE001
        print(
            f"[gpu-teacher] FAIL gpu={gpu_index} {root_id}/{endpoint}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "root_id": root_id,
            "endpoint": endpoint,
            "status": "FAIL",
            "gpu_index": gpu_index,
            "backend": "gpu4pyscf",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "live_chemistry": True,
            "dry_run": False,
        }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--gold-xyz-dir",
        type=Path,
        default=Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
    )
    p.add_argument("--n-endpoints", type=int, default=10, help="Must be even (cation+neutral)")
    p.add_argument("--max-parallel", type=int, default=8, help="Concurrent GPUs (machine has 8)")
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7", help="Physical GPU ids")
    p.add_argument("--host-threads", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--out-subdir",
        default="teacher_gpu_g002",
        help="Under runs/<gen>/; standard name teacher_gpu_g00N/",
    )
    p.add_argument(
        "--roots",
        default="",
        help="Comma-separated InChIKeys (must be n_endpoints/2). Empty => auto pick.",
    )
    p.add_argument(
        "--exclude-roots",
        default="",
        help="Comma-separated roots to skip when auto-picking (e.g. g002 batch).",
    )
    p.add_argument(
        "--batch-id",
        default="g002_batch1",
        help="Label stored in campaign receipt (g002_batch1 / g003_batch1 / ...)",
    )
    args = p.parse_args(argv)

    if args.n_endpoints % 2 != 0 or args.n_endpoints <= 0:
        raise SystemExit("--n-endpoints must be positive even")
    n_roots = args.n_endpoints // 2
    gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    if not gpu_ids:
        raise SystemExit("no GPU ids")
    max_par = min(args.max_parallel, len(gpu_ids), args.n_endpoints)

    if args.roots.strip():
        roots = [r.strip() for r in args.roots.split(",") if r.strip()]
        if len(roots) != n_roots:
            raise SystemExit(f"--roots must have exactly {n_roots} entries, got {len(roots)}")
    else:
        excl = {r.strip() for r in args.exclude_roots.split(",") if r.strip()}
        roots = pick_expansion_roots(
            args.gold_xyz_dir, n_roots=n_roots, exclude=excl
        )
    queue = [(r, ep) for r in roots for ep in ("cation", "neutral")]
    out_root = (
        Path(args.nhc0801_root) / "runs" / args.generation_id / args.out_subdir
    )
    out_root.mkdir(parents=True, exist_ok=True)
    logs_dir = Path(args.nhc0801_root) / "runs" / args.generation_id / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    print(
        f"[gpu-teacher] wave={WAVE_LABEL} backend=gpu4pyscf "
        f"endpoints={len(queue)} max_parallel={max_par} gpus={gpu_ids}",
        flush=True,
    )
    for r in roots:
        print(f"  root {r}", flush=True)

    results: list[dict] = []
    futs = {}
    with ProcessPoolExecutor(max_workers=max_par) as ex:
        for i, (root_id, endpoint) in enumerate(queue):
            gpu = gpu_ids[i % len(gpu_ids)]
            fut = ex.submit(
                _run_one_endpoint,
                nhc0801_root=str(args.nhc0801_root),
                generation_id=args.generation_id,
                root_id=root_id,
                endpoint=endpoint,
                gold_xyz_dir=str(args.gold_xyz_dir),
                max_steps=args.max_steps,
                gpu_index=gpu,
                host_threads=args.host_threads,
                out_root=str(out_root),
            )
            futs[fut] = (root_id, endpoint, gpu)
        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"[gpu-wave] progress {len(results)}/{len(queue)} "
                f"{res.get('root_id')}/{res.get('endpoint')} -> {res.get('status')}",
                flush=True,
            )

    failed = [r for r in results if r.get("status") == "FAIL"]
    partial = [r for r in results if r.get("status") == "PARTIAL"]
    passed = [r for r in results if r.get("status") == "PASS"]
    if len(results) == len(queue) and not failed and not partial:
        status = "LIVE_TEACHER_GPU_PASS"
    elif passed or partial:
        status = "LIVE_TEACHER_GPU_PARTIAL"
    else:
        status = "LIVE_TEACHER_GPU_FAIL"

    # split role by lexicographic order within this batch (3 train + 2 val when n_roots=5)
    train_roots = list(roots[:3]) if len(roots) >= 5 else list(roots)
    val_roots = list(roots[3:5]) if len(roots) >= 5 else []
    camp = {
        "schema": "nhc0801-teacher-gpu-g00n-campaign-v1",
        "mindmap_step": 2,
        "wave_label": WAVE_LABEL,
        "batch_id": args.batch_id,
        "backend": "gpu4pyscf",
        "dry_run": False,
        "live_chemistry": True,
        "teacher_pyscf_authorized": True,
        "status": status,
        "host_threads": args.host_threads,
        "gpu_ids": gpu_ids,
        "max_parallel": max_par,
        "endpoint_count": len(queue),
        "roots": roots,
        "train_roots_draft": train_roots,
        "val_roots_draft": val_roots,
        "passed_count": len(passed),
        "partial_count": len(partial),
        "failed_count": len(failed),
        "failed_endpoints": [f"{r['root_id']}/{r['endpoint']}" for r in failed],
        "endpoint_results": results,
        "output_root": str(out_root),
        "final_test_payload_read": False,
        "pilot_train_val_excluded": sorted(PILOT_ROOTS),
        "notes": [
            "GPU g00N teacher; roots NOT in pilot TVT split (not g001 train)",
            "Do not mix into Train NPZ without explicit user freeze of roots",
            "Frames=initial+final; xc=wb97m-d3bj; one GPU per concurrent worker",
            "CPU pilot teacher wave remains authoritative for pilot roots",
        ],
        "created_at_utc": _utc(),
    }
    write_json(out_root / "campaign_receipt_live_gpu.json", camp, overwrite=True)
    write_json(logs_dir / f"{Path(args.out_subdir).name}_campaign.json", camp, overwrite=True)
    print(
        json.dumps(
            {
                "status": status,
                "passed": len(passed),
                "partial": len(partial),
                "failed": len(failed),
                "roots": roots,
            },
            indent=2,
        ),
        flush=True,
    )
    print(f"TEACHER_GPU_WAVE_EXIT status={status}", flush=True)
    return 0 if status != "LIVE_TEACHER_GPU_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
