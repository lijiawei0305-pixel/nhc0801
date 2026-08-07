#!/usr/bin/env python3
"""Resident GPU **g00N teacher** queue daemon (g003+).

User-facing: **g00N teacher** only.
Stdout tag: ``[gpu-teacher]``.
State: runs/<gen>/gpu_teacher_queue/state.json
  (auto-migrates from legacy gpu_autofill/state.json)

- Pool: docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv
- Product dir: teacher_gpu_g00N/
- Final Test: never
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.gpu_autofill import (  # noqa: E402
    DEFAULT_XYZ_SEARCH,
    AutofillState,
    _utc,
    assign_batches,
    endpoint_done_ok,
    has_pair,
    list_free_gpu_ids,
    load_pool_inchikeys,
    resolve_xyz_dirs,
    spawn_endpoint,
)


def scan_done_across(gen_root: Path, root_id: str, endpoint: str) -> bool:
    for d in sorted(gen_root.glob("teacher_gpu*")):
        if not d.is_dir():
            continue
        if endpoint_done_ok(d, root_id, endpoint):
            return True
    return False


def rebuild_queue(state: AutofillState, gen_root: Path, xyz_dirs: list[Path]) -> None:
    pool = load_pool_inchikeys(Path(state.pool_csv))
    # Exclude pilot Val roots by default (e0 owns Val baseline).
    # Do NOT hard-exclude Train roots — m250 full-traj re-label of Train3 is allowed
    # via state.exclude_roots (or by simply leaving them out of exclude_roots).
    from nhc_deprot.data.paths import TRAIN_ROOTS as _TRAIN_ROOTS
    from nhc_deprot.data.paths import VALIDATION_ROOTS as _VAL_ROOTS

    exclude = set(state.exclude_roots) | set(_VAL_ROOTS)
    train_order = tuple(_TRAIN_ROOTS)
    train_set = set(train_order)

    # roots fully done in any teacher_gpu* dir
    fully_done: list[str] = []
    pending_roots: list[str] = []
    seen: set[str] = set()
    for root_id in pool:
        if root_id in exclude:
            continue
        if has_pair(root_id, xyz_dirs) is None:
            continue
        cat = scan_done_across(gen_root, root_id, "cation")
        neu = scan_done_across(gen_root, root_id, "neutral")
        if cat and neu:
            fully_done.append(root_id)
            seen.add(root_id)
            continue
        pending_roots.append(root_id)
        seen.add(root_id)

    # Ensure generation Train roots stay in the queue even if missing from pool CSV.
    for root_id in train_order:
        if root_id in exclude or root_id in seen:
            continue
        if has_pair(root_id, xyz_dirs) is None:
            continue
        cat = scan_done_across(gen_root, root_id, "cation")
        neu = scan_done_across(gen_root, root_id, "neutral")
        if cat and neu:
            fully_done.append(root_id)
        else:
            pending_roots.append(root_id)

    # Priority: incomplete Train roots first (pool may keep backfilling after).
    train_pending = [r for r in train_order if r in set(pending_roots)]
    rest = [r for r in pending_roots if r not in train_set]
    pending_roots = train_pending + rest

    # build endpoint queue preserving root order; skip endpoints already done
    queue: list[dict] = []
    for root_id in pending_roots:
        xyz_dir = has_pair(root_id, xyz_dirs)
        assert xyz_dir is not None
        for ep in ("cation", "neutral"):
            if scan_done_across(gen_root, root_id, ep):
                continue
            key = f"{root_id}:{ep}"
            if key in state.running:
                continue
            # skip in-flight work from prior waves (has frame0, no manifest yet)
            inflight = False
            for d in gen_root.glob("teacher_gpu*"):
                ep_dir = d / root_id / ep
                if (ep_dir / "frame_0000.json").is_file() and not (
                    ep_dir / "manifest.json"
                ).is_file():
                    inflight = True
                    break
            if inflight:
                continue
            queue.append(
                {
                    "root_id": root_id,
                    "endpoint": ep,
                    "gold_xyz_dir": str(xyz_dir),
                    "key": key,
                }
            )
    state.queue = queue

    # batch map for pending roots only (5 per batch)
    # start index: max existing g00N + 1 or state.next_batch_index
    start = max(3, int(state.next_batch_index))
    existing = [int(k[1:]) for k in state.batches if k.startswith("g") and k[1:].isdigit()]
    if existing:
        start = max(start, max(existing) + 1)
    # only create batches for roots not yet assigned
    assigned: set[str] = set()
    for _b, meta in state.batches.items():
        assigned.update(meta.get("roots") or [])
    unassigned = [r for r in pending_roots if r not in assigned]
    new_batches = assign_batches(unassigned, state.batch_size_roots, start)
    for name, roots in new_batches.items():
        state.batches[name] = {
            "roots": roots,
            "train_roots": roots[:3],
            "val_roots": roots[3:5] if len(roots) >= 5 else roots[3:],
            "created_at_utc": _utc(),
            "status": "PENDING",
        }
        state.next_batch_index = max(state.next_batch_index, int(name[1:]) + 1)

    # annotate each queue task with batch_id
    root_to_batch = {}
    for b, meta in state.batches.items():
        for r in meta.get("roots") or []:
            root_to_batch[r] = b
    for t in state.queue:
        t["batch_id"] = root_to_batch.get(t["root_id"], "g_unbatched")


def root_batch_id(state: AutofillState, root_id: str) -> str:
    for b, meta in state.batches.items():
        if root_id in (meta.get("roots") or []):
            return b
    return "g_unbatched"


def _pid_still_running(pid: int) -> bool:
    """False if missing or zombie (Z).

    os.kill(pid, 0) is true for zombies — must not treat as alive.
    """
    try:
        status = Path(f"/proc/{pid}/status").read_text(encoding="utf-8")
    except OSError:
        return False
    for line in status.splitlines():
        if line.startswith("State:"):
            # e.g. "State:\tZ (zombie)"
            st = line.split()[1] if len(line.split()) > 1 else ""
            if st == "Z":
                return False
            return True
    return False


def _parse_job_log(log: Path) -> tuple[str, dict]:
    status = "FAIL"
    detail: dict = {}
    if not log.is_file():
        return status, detail
    text = log.read_text(encoding="utf-8", errors="replace")
    for line in reversed(text.splitlines()):
        if line.startswith("JOB_EXIT"):
            status = line.split()[-1]
            break
        if line.startswith("{"):
            try:
                detail = json.loads(line)
                status = str(detail.get("status") or status)
            except json.JSONDecodeError:
                pass
            break
    # also accept JOB_EXIT after JSON
    for line in reversed(text.splitlines()):
        if line.startswith("JOB_EXIT"):
            status = line.split()[-1]
            break
    return status, detail


def reap(state: AutofillState, state_path: Path) -> None:
    finished_keys = []
    for key, info in list(state.running.items()):
        pid = int(info["pid"])
        log = Path(info["log"])
        alive = _pid_still_running(pid)
        status, detail = _parse_job_log(log)
        log_done = status in {"PASS", "PARTIAL", "FAIL"} and (
            log.is_file() and "JOB_EXIT" in log.read_text(encoding="utf-8", errors="replace")
        )
        # Reap if process gone/zombie OR job log already finalized (even if wrapper hung)
        if alive and not log_done:
            continue
        if log_done and alive:
            # try soft-kill hung wrapper after success
            try:
                os.kill(pid, signal.SIGTERM)
            except OSError:
                pass
        rec = {
            "key": key,
            "status": status if log_done else ("FAIL" if not alive else status),
            "gpu_index": info.get("gpu_index"),
            "batch_id": info.get("batch_id"),
            "finished_at_utc": _utc(),
            "detail": detail,
            "log": str(log),
            "reaped_zombie_or_log": True,
        }
        # Recover false PARTIAL from mid-trajectory gradient gate: product dir wins.
        if rec["status"] != "PASS":
            root_id = str(info.get("root_id") or key.split(":")[0])
            endpoint = str(info.get("endpoint") or (key.split(":")[1] if ":" in key else ""))
            gen_root = Path(state_path).resolve().parents[1]
            for d in gen_root.glob("teacher_gpu*"):
                if endpoint and endpoint_done_ok(d, root_id, endpoint):
                    rec["status"] = "PASS"
                    rec["recovered_from_product_dir"] = True
                    rec["product_dir"] = str(d / root_id / endpoint)
                    break
        if rec["status"] == "PASS":
            state.done.append(rec)
        else:
            state.failed.append(rec)
        finished_keys.append(key)

    for k in finished_keys:
        state.running.pop(k, None)
    # batch completion
    for _b, meta in state.batches.items():
        roots = meta.get("roots") or []
        if not roots:
            continue
        ok = True
        for r in roots:
            for ep in ("cation", "neutral"):
                # any teacher_gpu dir
                found = False
                for d in Path(state_path).resolve().parents[1].glob("teacher_gpu*"):
                    if endpoint_done_ok(d, r, ep):
                        found = True
                        break
                if not found:
                    ok = False
                    break
            if not ok:
                break
        if ok and meta.get("status") != "PASS":
            meta["status"] = "PASS"
            meta["finished_at_utc"] = _utc()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--pool-csv",
        type=Path,
        default=None,
        help="default: <repo>/docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv",
    )
    p.add_argument("--gpu-ids", default="0,1,2,3,4,5,6,7")
    p.add_argument("--host-threads", type=int, default=2)
    p.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help=(
            "Parent geomeTRIC max steps (GAU/parent contract; default 250). "
            "Do not use 100 for new expansion."
        ),
    )
    p.add_argument("--poll-seconds", type=int, default=20)
    p.add_argument("--batch-size-roots", type=int, default=5)
    p.add_argument(
        "--exclude-roots",
        default="",
        help="extra excludes comma-separated (e.g. incomplete g003 large roots)",
    )
    p.add_argument("--once", action="store_true", help="single scheduling pass then exit")
    args = p.parse_args(argv)

    nhc = args.nhc0801_root
    gen_root = nhc / "runs" / args.generation_id
    # Canonical state dir + one-time migrate from legacy name
    state_dir = gen_root / "gpu_teacher_queue"
    legacy_dir = gen_root / "gpu_autofill"
    state_path = state_dir / "state.json"
    state_dir.mkdir(parents=True, exist_ok=True)
    if not state_path.is_file() and (legacy_dir / "state.json").is_file():
        import shutil

        shutil.copy2(legacy_dir / "state.json", state_path)
        jobs_legacy = legacy_dir / "jobs"
        if jobs_legacy.is_dir() and not (state_dir / "jobs").is_dir():
            shutil.copytree(jobs_legacy, state_dir / "jobs")
        print(f"[gpu-teacher] migrated state from {legacy_dir} -> {state_dir}", flush=True)

    pool_csv = args.pool_csv or (ROOT / "docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv")
    if not pool_csv.is_file():
        # server path
        alt = nhc / "docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv"
        if alt.is_file():
            pool_csv = alt
        else:
            print(f"missing pool csv: {pool_csv}", flush=True)
            return 2

    state = AutofillState.load(state_path)
    if state.schema.startswith("nhc0801-gpu-autofill"):
        state.schema = "nhc0801-gpu-teacher-queue-v1"
    state.generation_id = args.generation_id
    state.pool_csv = str(pool_csv)
    state.gpu_ids = [int(x) for x in args.gpu_ids.split(",") if x.strip() != ""]
    state.batch_size_roots = args.batch_size_roots
    excl = {x.strip() for x in args.exclude_roots.split(",") if x.strip()}
    state.exclude_roots = sorted(set(state.exclude_roots) | excl)

    xyz_dirs = resolve_xyz_dirs(DEFAULT_XYZ_SEARCH)
    state.xyz_dirs = [str(x) for x in xyz_dirs]

    stop = {"flag": False}

    def _sig(_s, _f):
        stop["flag"] = True

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    print(
        f"[gpu-teacher] start {_utc()} pool={pool_csv} "
        f"gpus={state.gpu_ids} xyz_dirs={len(xyz_dirs)}",
        flush=True,
    )

    while not stop["flag"]:
        reap(state, state_path)
        rebuild_queue(state, gen_root, xyz_dirs)
        state.save(state_path)

        if not state.queue and not state.running:
            state.stop_reason = "pool_exhausted_or_all_done"
            state.save(state_path)
            print(
                f"[gpu-teacher] STOP {state.stop_reason} "
                f"done={len(state.done)} failed={len(state.failed)}",
                flush=True,
            )
            return 0

        free = list_free_gpu_ids(state.gpu_ids)
        # do not claim GPUs already in our running map
        claimed = {int(v["gpu_index"]) for v in state.running.values()}
        free = [g for g in free if g not in claimed]
        # If every card already hosts some NHC parent (e0 fill / others) but this
        # daemon still has Train work, co-locate on unused-by-us GPUs rather than
        # stall until e0 drains. Never second-claim a GPU we already use.
        if not free and state.queue:
            free = [int(g) for g in state.gpu_ids if int(g) not in claimed]
            if free:
                print(
                    f"[gpu-teacher] {_utc()} no exclusive free GPU; "
                    f"co-locate on {free} for Train/pool catch-up",
                    flush=True,
                )
        print(
            f"[gpu-teacher] {_utc()} queue={len(state.queue)} "
            f"running={len(state.running)} free_gpus={free}",
            flush=True,
        )

        for gpu in free:
            if not state.queue:
                break
            task = state.queue.pop(0)
            batch_id = task.get("batch_id") or root_batch_id(state, task["root_id"])
            try:
                proc = spawn_endpoint(
                    state_dir=state_dir,
                    nhc0801_root=nhc,
                    generation_id=args.generation_id,
                    batch_id=batch_id,
                    task=task,
                    gpu_index=gpu,
                    max_steps=args.max_steps,
                    host_threads=args.host_threads,
                )
            except Exception as exc:  # noqa: BLE001
                state.failed.append(
                    {"key": task["key"], "status": "SPAWN_FAIL", "error": str(exc), "at": _utc()}
                )
                continue
            log = state_dir / "jobs" / f"{batch_id}_{task['key'].replace(':','_')}_gpu{gpu}.out"
            state.running[task["key"]] = {
                "pid": proc.pid,
                "gpu_index": gpu,
                "batch_id": batch_id,
                "started_at_utc": _utc(),
                "log": str(log),
                "root_id": task["root_id"],
                "endpoint": task["endpoint"],
            }
            print(
                f"[gpu-teacher] CLAIM gpu={gpu} {task['key']} batch={batch_id} pid={proc.pid}",
                flush=True,
            )
            # mark batch running
            if batch_id in state.batches and state.batches[batch_id].get("status") == "PENDING":
                state.batches[batch_id]["status"] = "RUNNING"
                state.batches[batch_id]["started_at_utc"] = _utc()

        state.save(state_path)
        if args.once:
            return 0
        time.sleep(max(5, int(args.poll_seconds)))

    state.stop_reason = "signal"
    state.save(state_path)
    print("[gpu-teacher] stopped by signal", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
