#!/usr/bin/env python3
"""Live g001 teacher wave (mindmap step 2) — stable entry name.

Logs: teacher_wave_g001.out (canonical). Products: teacher_gpu_g001/.

Resource profile: auto_fill_112_t10_r12_v1
  - t=10 threads / endpoint
  - CPU pool 0-99; reserve 100-111 never scheduled
  - up to 10 endpoints concurrent (Train3 + Val2) x (cation+neutral)
  - Parent CPU-only (CUDA_VISIBLE_DEVICES=)
  - continue_queue: one endpoint FAIL does not abort the wave

Does not open Final Test. Does not retrain.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.live_teacher import LiveParentTeacherEngine  # noqa: E402
from nhc_deprot.pipeline.pipeline_status import write_step_status  # noqa: E402
from nhc_deprot.pipeline.teacher_runner import (  # noqa: E402
    endpoint_charge_mult,
)
from nhc_deprot.resources.auto_fill import (  # noqa: E402
    build_auto_fill_plan,
    expand_pool_cpu_ids,
)
from nhc_deprot.resources.host_sampler import expand_cpu_list  # noqa: E402
from nhc_deprot.resources.profiles import (  # noqa: E402
    OFFICIAL_DEFAULT_V002,
    get_profile,
    load_v002_catalog,
    worker_env_for_profile,
)

PROFILE_ID = OFFICIAL_DEFAULT_V002  # auto_fill_112_t10_r12_v1
WAVE_LABEL = "g001_teacher"


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _mem_available_bytes() -> int:
    """Parse MemAvailable from /proc/meminfo (Linux)."""
    try:
        text = Path("/proc/meminfo").read_text(encoding="utf-8")
    except OSError:
        return 200 * 1024**3
    for line in text.splitlines():
        if line.startswith("MemAvailable:"):
            # kB
            parts = line.split()
            return int(parts[1]) * 1024
    return 200 * 1024**3


def _apply_slot_env(*, threads: int, cpu_list: str) -> None:
    prof = get_profile(PROFILE_ID)
    env = worker_env_for_profile(prof, threads=threads)
    for k, v in env.items():
        os.environ[k] = v
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ["NHC0801_CPU_LIST"] = cpu_list
    os.environ["NHC0801_TASKSET"] = cpu_list
    # Hard pin: never schedule on 100-111
    try:
        ids = expand_cpu_list(cpu_list)
        if ids:
            os.sched_setaffinity(0, set(ids))
    except (AttributeError, OSError, ValueError) as exc:
        print(f"[warn] sched_setaffinity failed: {exc}", flush=True)


def _run_one_endpoint(
    *,
    nhc0801_root: str,
    generation_id: str,
    root_id: str,
    endpoint: str,
    gold_xyz_dir: str,
    max_steps: int,
    threads: int,
    cpu_list: str,
    slot_id: int,
) -> dict:
    """Process entry: one root/endpoint teacher optimization."""
    try:
        _apply_slot_env(threads=threads, cpu_list=cpu_list)
        # Refuse reserved CPUs
        reserved = set(expand_cpu_list("100-111"))
        used = set(expand_cpu_list(cpu_list))
        if used & reserved:
            raise RuntimeError(f"slot uses reserved CPUs: {sorted(used & reserved)}")

        layout = resolve_layout(
            generation_id=generation_id, nhc0801_root=Path(nhc0801_root)
        )
        eng = LiveParentTeacherEngine(
            gold_xyz_dir=Path(gold_xyz_dir),
            max_steps=max_steps,
        )
        charge, mult = endpoint_charge_mult(endpoint)
        out_dir = layout.teacher_endpoint_dir(root_id, endpoint)
        print(
            f"[teacher] START slot={slot_id} {root_id}/{endpoint} "
            f"t={threads} cpus={cpu_list}",
            flush=True,
        )
        result = eng.run_endpoint(
            root_id=root_id,
            endpoint=endpoint,
            charge=charge,
            multiplicity=mult,
            output_dir=out_dir,
        )
        status = "PASS" if result.get("converged") and int(result.get("frame_count") or 0) >= 2 else "PARTIAL"
        print(
            f"[teacher] END slot={slot_id} {root_id}/{endpoint} -> {status} "
            f"frames={result.get('frame_count')} wall={result.get('wall_seconds')}",
            flush=True,
        )
        return {
            "root_id": root_id,
            "endpoint": endpoint,
            "status": status,
            "slot_id": slot_id,
            "cpu_list": cpu_list,
            "threads": threads,
            "frame_count": result.get("frame_count"),
            "converged": result.get("converged"),
            "wall_seconds": result.get("wall_seconds"),
            "output_dir": str(out_dir),
            "live_chemistry": True,
            "dry_run": False,
        }
    except Exception as exc:  # noqa: BLE001
        print(
            f"[teacher] FAIL slot={slot_id} {root_id}/{endpoint}: "
            f"{type(exc).__name__}: {exc}",
            flush=True,
        )
        return {
            "root_id": root_id,
            "endpoint": endpoint,
            "status": "FAIL",
            "slot_id": slot_id,
            "cpu_list": cpu_list,
            "threads": threads,
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
            "live_chemistry": True,
            "dry_run": False,
        }


def _select_roots(scope: str) -> list[str]:
    if scope == "train":
        return list(TRAIN_ROOTS)
    if scope == "val":
        return list(VALIDATION_ROOTS)
    if scope == "train+val":
        return list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    raise SystemExit(f"unknown --roots {scope}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--gold-xyz-dir",
        type=Path,
        default=Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
    )
    p.add_argument("--profile", default=PROFILE_ID)
    p.add_argument("--roots", default="train+val", choices=("train", "val", "train+val"))
    p.add_argument("--max-parallel", type=int, default=10, help="Max concurrent endpoints")
    p.add_argument("--threads", type=int, default=0, help="0 => profile default (10)")
    p.add_argument("--max-steps", type=int, default=100)
    p.add_argument(
        "--force-n",
        type=int,
        default=0,
        help="If >0, force N slots from full pool (ignore live idle sample)",
    )
    args = p.parse_args(argv)

    cat = load_v002_catalog()
    if cat.get("revision") and cat.get("revision") not in {WAVE_LABEL, "2026-08-02c"}:
        print(
            f"[warn] catalog revision={cat.get('revision')!r} expected {WAVE_LABEL}",
            flush=True,
        )
    prof = get_profile(args.profile)
    threads = int(args.threads or prof.threads_per_worker)
    if threads != 10 and args.profile == PROFILE_ID:
        print(f"[warn] profile default is 10; using t={threads}", flush=True)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    ensure_generation_tree(layout, exist_ok=True)

    roots = _select_roots(args.roots)
    queue = [(r, ep) for r in roots for ep in ("cation", "neutral")]
    print(
        f"[g001-teacher] wave={WAVE_LABEL} profile={args.profile} "
        f"endpoints={len(queue)} t={threads} max_parallel={args.max_parallel}",
        flush=True,
    )
    for r, ep in queue:
        print(f"  queue {r}/{ep}", flush=True)

    # Build slots from full pool 0-99 (machine expected clean after stop)
    pool_ids = expand_pool_cpu_ids(prof)
    if args.force_n and args.force_n > 0:
        idle_ids = pool_ids
        n_cap = min(args.force_n, args.max_parallel, len(queue))
    else:
        # Prefer full pool when host is idle after stop; capacity still mem-gated
        idle_ids = pool_ids
        n_cap = min(args.max_parallel, len(queue))

    mem = _mem_available_bytes()
    plan = build_auto_fill_plan(
        idle_cpu_ids=idle_ids,
        mem_available_bytes=mem,
        endpoint_queue=queue,
        profile=prof,
        claim_pass=True,
        n_cap=n_cap,
    )
    n_slots = len(plan.slots)
    print(
        f"[wave] capacity N={plan.capacity.n} slots={n_slots} "
        f"n_cpu={plan.capacity.n_cpu} n_mem={plan.capacity.n_mem} "
        f"mem_avail_GiB={mem / 1024**3:.1f}",
        flush=True,
    )
    for s in plan.slots:
        print(f"  slot {s.slot_id}: cpus={s.cpu_list} t={s.threads}", flush=True)

    if n_slots <= 0:
        print("[wave] FAIL: zero slots planned", flush=True)
        return 2

    # Assign endpoints to slots round-robin / queue fill: run min(n_slots, len(queue))
    # at a time; when one finishes, start next (ProcessPool max_workers=n_slots)
    write_step_status(
        layout,
        step=2,
        name="teacher_pyscf",
        status="RUNNING",
        detail={
            "phase": "g001_teacher",
            "profile": args.profile,
            "wave_label": WAVE_LABEL,
            "n_slots": n_slots,
            "n_endpoints": len(queue),
            "threads": threads,
        },
    )

    results: list[dict] = []
    # Static slot assignment: first n_slots tasks get slots; remaining wait in queue
    # ProcessPoolExecutor reuses workers — we submit all tasks with pre-assigned
    # cpu_list via round-robin over slots (continue_queue).
    futs = {}
    with ProcessPoolExecutor(max_workers=n_slots) as ex:
        for i, (root_id, endpoint) in enumerate(queue):
            slot = plan.slots[i % n_slots]
            fut = ex.submit(
                _run_one_endpoint,
                nhc0801_root=str(args.nhc0801_root),
                generation_id=args.generation_id,
                root_id=root_id,
                endpoint=endpoint,
                gold_xyz_dir=str(args.gold_xyz_dir),
                max_steps=args.max_steps,
                threads=threads,
                cpu_list=slot.cpu_list,
                slot_id=slot.slot_id,
            )
            futs[fut] = (root_id, endpoint, slot.slot_id)

        for fut in as_completed(futs):
            res = fut.result()
            results.append(res)
            print(
                f"[wave] progress {len(results)}/{len(queue)} "
                f"{res.get('root_id')}/{res.get('endpoint')} -> {res.get('status')}",
                flush=True,
            )

    failed = [r for r in results if r.get("status") == "FAIL"]
    partial = [r for r in results if r.get("status") == "PARTIAL"]
    passed = [r for r in results if r.get("status") == "PASS"]
    if len(results) == len(queue) and not failed and not partial:
        status = "LIVE_TEACHER_PASS"
    elif passed or partial:
        status = "LIVE_TEACHER_PARTIAL"
    else:
        status = "LIVE_TEACHER_FAIL"

    camp = {
        "schema": "nhc0801-teacher-campaign-receipt-v1",
        "mindmap_step": 2,
        "wave_label": WAVE_LABEL,
        "profile_id": args.profile,
        "dry_run": False,
        "live_chemistry": True,
        "teacher_pyscf_authorized": True,
        "status": status,
        "threads_per_endpoint": threads,
        "cpu_pool": prof.cpu_pool,
        "cpu_reserve_list": prof.cpu_reserve_list,
        "max_parallel": n_slots,
        "endpoint_count": len(queue),
        "passed_count": len(passed),
        "partial_count": len(partial),
        "failed_count": len(failed),
        "failed_endpoints": [
            f"{r['root_id']}/{r['endpoint']}" for r in failed
        ],
        "endpoint_results": results,
        "final_test_payload_read": False,
        "notes": [
            "g001 teacher wave: endpoint-parallel",
            "frames=initial+final only until geomeTRIC step dump wired",
            "Train+Val development roots only; Final Test sealed",
            "continue_queue: single endpoint FAIL does not abort wave",
        ],
        "created_at_utc": _utc(),
    }
    write_json(layout.teacher_dir / "campaign_receipt_live.json", camp, overwrite=True)
    write_json(layout.logs_dir / "teacher_campaign_live_g001.json", camp, overwrite=True)
    write_json(layout.teacher_dir / "campaign_receipt_live.json", camp, overwrite=True)
    write_step_status(
        layout,
        step=2,
        name="teacher_pyscf",
        status=status,
        detail={
            "failed_count": len(failed),
            "passed_count": len(passed),
            "partial_count": len(partial),
            "profile": args.profile,
        },
    )
    print(json.dumps({"status": status, "passed": len(passed), "partial": len(partial), "failed": len(failed)}, indent=2), flush=True)
    print(f"TEACHER_WAVE_EXIT status={status}", flush=True)
    return 0 if status != "LIVE_TEACHER_FAIL" else 1


if __name__ == "__main__":
    raise SystemExit(main())
