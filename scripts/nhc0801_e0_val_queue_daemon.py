#!/usr/bin/env python3
"""Queue per-batch Epoch-0 jobs (standard names: g001/g002/… Epoch-0).

Policy (user 2026-08-02/03):
  - Task name = **g00N Epoch-0** (NOT "expansion Val e0")
  - g001 Epoch-0: both Val roots — usually via live_orchestrate
  - g00N Epoch-0 (N>=2): only that batch's val_roots (2 per batch)
  - never run e0 on train_roots
  - AIMNet2 GAU only on GPUs without VASP (may share with our gpu4pyscf)
  - parent/handoff = **gpu4pyscf on same physical GPU** (user 2026-08-03; CPU was too slow)
  - disk: epoch0_val_batches/g00N/ holds g00N Epoch-0 receipts

State: runs/<gen>/epoch0_val_queue/state.json
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402

G002_VAL = (
    "HVVRUQBMAZRKPJ-UHFFFAOYSA-N",
    "IPMZWBRHUWBMSP-UHFFFAOYSA-N",
)
TRAIN_SET = frozenset(TRAIN_ROOTS)


def utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, log_path: Path) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def gpus_without_vasp(max_gpu: int = 8) -> list[int]:
    free: list[int] = []
    for i in range(max_gpu):
        try:
            o = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(i),
                    "--query-compute-apps=process_name,used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            )
        except Exception:
            continue
        if "vasp" in o.lower():
            continue
        free.append(i)
    return free


def pick_e0_gpus(n: int, *, exclude: set[int] | None = None) -> list[int]:
    """Pick up to n no-VASP GPUs sorted by lowest used memory."""
    exclude = exclude or set()
    scored: list[tuple[int, int]] = []
    for i in gpus_without_vasp():
        if i in exclude:
            continue
        try:
            o = subprocess.check_output(
                [
                    "nvidia-smi",
                    "-i",
                    str(i),
                    "--query-compute-apps=used_memory",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
        except Exception:
            o = ""
        mem = 0
        for line in o.splitlines():
            if "MiB" in line:
                try:
                    mem += int(line.split()[0])
                except ValueError:
                    pass
        scored.append((mem, i))
    scored.sort()
    return [i for _, i in scored[:n]]


def load_state(path: Path) -> dict:
    if path.is_file():
        return json.loads(path.read_text(encoding="utf-8"))
    return {
        "schema": "nhc0801-e0-val-queue-v1",
        "completed": {},
        "failed": {},
        "running": {},
        "notes": ["val_roots only", "never train", "no-VASP GPUs only"],
    }


def save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = utc()
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def sanitize_val_roots(vals: list[str], trains: set[str] | None = None) -> list[str]:
    ban = set(TRAIN_SET) | set(trains or ())
    return [v for v in vals if v and v not in ban]


def collect_jobs(gen_root: Path) -> list[dict]:
    """Jobs: g001 + g002 + PASS g00N teacher batches' val_roots only."""
    jobs: list[dict] = []
    jobs.append(
        {
            "batch_id": "g001",  # standard name: g001 Epoch-0 → epoch0_val_batches/g001/
            "val_roots": sanitize_val_roots(list(VALIDATION_ROOTS)),
        }
    )
    jobs.append({"batch_id": "g002", "val_roots": sanitize_val_roots(list(G002_VAL))})
    st = gen_root / "gpu_teacher_queue" / "state.json"
    if not st.is_file():
        st = gen_root / "gpu_autofill" / "state.json"  # legacy
    if st.is_file():
        s = json.loads(st.read_text(encoding="utf-8"))
        for bid, meta in sorted((s.get("batches") or {}).items()):
            if meta.get("status") != "PASS":
                continue
            trains = set(meta.get("train_roots") or [])
            vals = sanitize_val_roots(list(meta.get("val_roots") or []), trains)
            if vals:
                jobs.append({"batch_id": bid, "val_roots": vals})
    return jobs


def g001_e0_busy_or_done(logdir: Path) -> str:
    """Detect external g001 Epoch-0 via canonical + legacy log basenames."""
    try:
        from nhc_deprot.generation.artifact_names import (
            EPOCH0_LIVE_LOG,
            EPOCH0_LIVE_LOG_LEGACY,
            resolve_existing,
        )
    except Exception:  # noqa: BLE001
        resolve_existing = None  # type: ignore[assignment]
        EPOCH0_LIVE_LOG = "live_epoch0_g001.out"
        EPOCH0_LIVE_LOG_LEGACY = (
            "live_epoch0_02c.out",
            "live_epoch0.out",
            "g001_epoch0_rerun.out",
        )

    names = (EPOCH0_LIVE_LOG, *EPOCH0_LIVE_LOG_LEGACY)
    texts: list[str] = []
    for name in names:
        p = logdir / name
        if p.is_file():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))
    # also any e0_val_only g001 job log
    for p in sorted(logdir.glob("**/job_g001.out")) + sorted(
        (logdir.parent / "epoch0_val_queue").glob("job_g001.out")
        if (logdir.parent / "epoch0_val_queue").is_dir()
        else []
    ):
        if p.is_file():
            texts.append(p.read_text(encoding="utf-8", errors="replace"))

    for t in texts:
        if "EPOCH0_EXIT" in t or "E0_VAL_EXIT" in t:
            return "done"
    for t in texts:
        if "START_EPOCH0" in t or "starting live route" in t or "g001 Epoch-0" in t:
            for name in os.listdir("/proc"):
                if not name.isdigit():
                    continue
                try:
                    c = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode()
                except OSError:
                    continue
                if "python" in c and (
                    ("nhc0801_live_orchestrate.py" in c and "skip-train-live" in c)
                    or ("e0_val_only" in c and "g001" in c)
                ):
                    return "running"
            return "stale"
    # process-only detection
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            c = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "e0_val_only" in c and "--batch-id g001" in c:
            return "running"
        if "python" in c and "nhc0801_live_orchestrate.py" in c and "skip-train-live" in c:
            return "running"
    return "none"


def launch_job(
    *,
    nhc: Path,
    gen: str,
    batch_id: str,
    val_roots: list[str],
    gpu: int,
    log_path: Path,
    parent_backend: str = "gpu",
) -> subprocess.Popen:
    """Start g00N Epoch-0 in background; returns Popen.

    Parent/handoff default **gpu** (gpu4pyscf on the same physical GPU as AIMNet2).
    User 2026-08-03: CPU parent too slow for multi-batch Epoch-0.
    """
    env = os.environ.copy()
    env["PYTHONPATH"] = str(nhc / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    # Host BLAS small: DFT runs on GPU; leave CPU for others.
    host_t = "2" if parent_backend == "gpu" else "8"
    env["OMP_NUM_THREADS"] = host_t
    env["MKL_NUM_THREADS"] = host_t
    env["OPENBLAS_NUM_THREADS"] = host_t
    roots_csv = ",".join(val_roots)
    parent_flags = f"--parent-backend {parent_backend}"
    if parent_backend == "gpu":
        # physical GPU id (worker sets CUDA_VISIBLE_DEVICES to this id)
        parent_flags += f" --cuda-device {int(gpu)}"
    cmd = (
        f"source /home/plab/test/WJW/env/envs/mlff.sh && "
        f"cd {nhc} && "
        f"python3 -m nhc_deprot.pipeline.e0_val_only "
        f"--nhc0801-root {nhc} "
        f"--generation-id {gen} "
        f"--batch-id {batch_id} "
        f"--val-roots {roots_csv} "
        f"--max-steps 100 "
        f"{parent_flags}"
    )
    log_path.parent.mkdir(parents=True, exist_ok=True)
    fh = log_path.open("w", encoding="utf-8")
    fh.write(
        f"START_E0_VAL {batch_id} Epoch-0 gpu={gpu} parent={parent_backend} "
        f"vals={val_roots} {utc()}\n"
    )
    fh.flush()
    p = subprocess.Popen(
        ["bash", "-lc", cmd],
        stdout=fh,
        stderr=subprocess.STDOUT,
        env=env,
        cwd=str(nhc),
    )
    # keep fh open for child lifetime — attach to process
    p._e0_log_fh = fh  # type: ignore[attr-defined]
    return p


def reap_running(state: dict, state_path: Path, log_path: Path) -> None:
    """Update completed/failed from running pid map."""
    still: dict = {}
    for bid, meta in list((state.get("running") or {}).items()):
        pid = int(meta.get("pid") or 0)
        jlog = Path(meta.get("log") or "")
        alive = False
        if pid > 0:
            try:
                os.kill(pid, 0)
                # zombie?
                with open(f"/proc/{pid}/status", encoding="utf-8") as sf:
                    for line in sf:
                        if line.startswith("State:"):
                            if "Z" in line.split()[1:2] or line.split()[1:2] == ["Z"]:
                                alive = False
                            else:
                                alive = True
                            break
                    else:
                        alive = True
            except OSError:
                alive = False
        if alive:
            still[bid] = meta
            continue
        text = jlog.read_text(encoding="utf-8", errors="replace") if jlog.is_file() else ""
        status = "FAIL"
        for line in reversed(text.splitlines()):
            if line.startswith("E0_VAL_EXIT"):
                # E0_VAL_EXIT LIVE_EPOCH0_PASS | FAIL ... | REFUSED ...
                parts = line.split(maxsplit=1)
                status = parts[1].split()[0] if len(parts) > 1 else "FAIL"
                break
        ok = status not in {"FAIL", "FAILED", "REFUSED"} and (
            "LIVE_EPOCH0" in status
            or status.endswith("PASS")
            or status.endswith("PARTIAL")
            or status == "EXTERNAL_DONE"
        )
        if ok:
            state.setdefault("completed", {})[bid] = {
                "status": status,
                "val_roots": meta.get("val_roots"),
                "gpu": meta.get("gpu"),
                "at": utc(),
                "log": str(jlog),
            }
            log(f"DONE {bid} Epoch-0 status={status}", log_path)
        else:
            fr = state.setdefault("failed", {}).get(bid, {"retries": 0})
            fr["retries"] = int(fr.get("retries") or 0) + 1
            fr["status"] = status
            fr["at"] = utc()
            fr["log"] = str(jlog)
            state["failed"][bid] = fr
            log(f"FAIL {bid} Epoch-0 status={status}", log_path)
    state["running"] = still
    save_state(state_path, state)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    ap.add_argument("--generation-id", default="nhc0801-g001")
    ap.add_argument("--poll-seconds", type=int, default=90)
    ap.add_argument("--skip-g001", action="store_true", help="g001 handled by live_orchestrate")
    ap.add_argument("--max-parallel", type=int, default=3, help="max concurrent g00N Epoch-0 jobs")
    ap.add_argument(
        "--parent-backend",
        choices=("gpu", "cpu"),
        default="gpu",
        help="Parent/handoff backend (default gpu4pyscf on claimed GPU)",
    )
    ap.add_argument("--once", action="store_true")
    args = ap.parse_args(argv)

    nhc = args.nhc0801_root
    gen_root = nhc / "runs" / args.generation_id
    qdir = gen_root / "epoch0_val_queue"
    state_path = qdir / "state.json"
    log_path = gen_root / "logs" / "e0_val_queue.log"
    logdir = gen_root / "logs"

    log(
        f"epoch0-queue start (g00N Epoch-0; Val roots only; never train; "
        f"parent={args.parent_backend}) skip_g001={args.skip_g001} "
        f"max_parallel={args.max_parallel}",
        log_path,
    )
    while True:
        state = load_state(state_path)
        reap_running(state, state_path, log_path)
        state = load_state(state_path)

        jobs = collect_jobs(gen_root)
        running = dict(state.get("running") or {})
        used_gpus = {int(m["gpu"]) for m in running.values() if m.get("gpu") is not None}
        slots = max(0, int(args.max_parallel) - len(running))

        for job in jobs:
            if slots <= 0:
                break
            bid = job["batch_id"]
            vals = sanitize_val_roots(list(job["val_roots"]))
            if not vals:
                log(f"skip {bid}: no val roots after train filter", log_path)
                continue
            if bid == "g001" and args.skip_g001:
                st = g001_e0_busy_or_done(logdir)
                if st == "done":
                    state.setdefault("completed", {})[bid] = {
                        "status": "EXTERNAL_DONE",
                        "val_roots": vals,
                        "at": utc(),
                    }
                    save_state(state_path, state)
                elif st == "running":
                    log("g001 Epoch-0 still running externally (both Val roots); skip enqueue", log_path)
                continue
            if bid in state.get("completed", {}):
                continue
            if bid in running:
                continue
            if bid in state.get("failed", {}) and int(state["failed"][bid].get("retries") or 0) >= 2:
                continue

            picked = pick_e0_gpus(1, exclude=used_gpus)
            if not picked:
                log("no GPU without VASP available; wait", log_path)
                break
            gpu = picked[0]
            jlog = qdir / f"job_{bid}.out"
            log(f"RUN {bid} Epoch-0 gpu={gpu} vals={vals}", log_path)
            p = launch_job(
                nhc=nhc,
                gen=args.generation_id,
                batch_id=bid,
                val_roots=vals,
                gpu=gpu,
                log_path=jlog,
                parent_backend=str(args.parent_backend),
            )
            running[bid] = {
                "pid": p.pid,
                "gpu": gpu,
                "val_roots": vals,
                "log": str(jlog),
                "started_at": utc(),
                "parent_backend": str(args.parent_backend),
            }
            used_gpus.add(gpu)
            slots -= 1
            state["running"] = running
            save_state(state_path, state)

        if args.once and not state.get("running"):
            return 0
        if args.once and state.get("running"):
            # wait for launched
            while state.get("running"):
                time.sleep(30)
                reap_running(state, state_path, log_path)
                state = load_state(state_path)
            return 0
        time.sleep(max(30, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
