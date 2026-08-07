#!/usr/bin/env python3
"""Continuous Val e0 fill: 1 free GPU → 1 pending (root, endpoint).

Real-time scheduling (same spirit as gpu_teacher_daemon / e0_val_queue):
  - Work unit = one endpoint (cation|neutral), not a multi-root wave.
  - Whenever a claimed GPU finishes, immediately launch the next pending endpoint.
  - Prefer one endpoint per physical GPU (gpu4pyscf parent on same card).
  - Never train roots. Never open Final Test. Never kill teacher/VASP.

Target: TVT resplit Val pool (``VALIDATION_ROOTS``) → ``epoch0_val_batches/<batch>/``.

State: ``runs/<gen>/logs/val_e0_fill/state.json``
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.resources.gpu_inventory import inventory_gpus  # noqa: E402

ENDPOINTS = ("cation", "neutral")


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str, log_path: Path) -> None:
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state(path: Path) -> dict[str, Any]:
    if path.is_file():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            pass
    return {
        "schema": "nhc0801-val-e0-fill-v1",
        "running": {},  # key "root:ep" -> {pid,gpu,log,...}
        "done": {},
        "failed": {},
        "notes": [
            "endpoint-level continuous fill",
            "1 free GPU -> 1 pending endpoint",
            "never train roots",
        ],
    }


def save_state(path: Path, state: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    state["updated_at_utc"] = utc()
    path.write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def job_key(root_id: str, endpoint: str) -> str:
    return f"{root_id}:{endpoint}"


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        with open(f"/proc/{pid}/status", encoding="utf-8") as sf:
            for line in sf:
                if line.startswith("State:"):
                    st = line.split()[1:2]
                    return bool(st) and st[0] != "Z"
        return True
    except OSError:
        return False


def root_receipt_status(e0_dir: Path, root_id: str) -> str | None:
    # G1: same resolution as epoch0_campaign_rebuild (root.json then legacy).
    from nhc_deprot.pipeline.epoch0_campaign_rebuild import (
        resolve_epoch0_root_receipt_path,
    )

    rec = resolve_epoch0_root_receipt_path(e0_dir, root_id)
    if rec is None:
        return None
    try:
        return str(json.loads(rec.read_text(encoding="utf-8")).get("status") or "")
    except (OSError, json.JSONDecodeError):
        return None


def endpoint_status(e0_dir: Path, root_id: str, endpoint: str) -> str | None:
    root = e0_dir / root_id
    for name in (f"{endpoint}.json", f"{endpoint}_shard.json"):
        p = root / name
        if p.is_file():
            try:
                return str(json.loads(p.read_text(encoding="utf-8")).get("status") or "")
            except (OSError, json.JSONDecodeError):
                return "UNREADABLE"
    return None


def list_live_e0_gpu_claims() -> dict[int, dict[str, Any]]:
    """Map physical GPU index -> {pid, root, endpoint} for live e0_val_only."""
    claimed: dict[int, dict[str, Any]] = {}
    try:
        names = os.listdir("/proc")
    except OSError:
        return claimed
    for name in names:
        if not name.isdigit():
            continue
        try:
            with open(f"/proc/{name}/cmdline", "rb") as fh:
                cmd = fh.read().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "e0_val_only" not in cmd:
            continue
        # parse --cuda-device / --val-roots / --endpoint
        parts = cmd.split()
        gpu = None
        root = None
        ep = None
        for i, tok in enumerate(parts):
            if tok == "--cuda-device" and i + 1 < len(parts):
                try:
                    gpu = int(parts[i + 1])
                except ValueError:
                    pass
            if tok == "--val-roots" and i + 1 < len(parts):
                root = parts[i + 1].split(",")[0]
            if tok == "--endpoint" and i + 1 < len(parts):
                ep = parts[i + 1]
        if gpu is None:
            # fall back to CUDA_VISIBLE_DEVICES
            try:
                with open(f"/proc/{name}/environ", "rb") as fh:
                    env = fh.read().split(b"\0")
                for e in env:
                    if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                        val = e.decode(errors="replace").split("=", 1)[1]
                        if val and val[0].isdigit():
                            gpu = int(val.split(",")[0])
            except OSError:
                pass
        if gpu is None:
            continue
        claimed[int(gpu)] = {
            "pid": int(name),
            "root_id": root,
            "endpoint": ep,
            "cmd": cmd[:200],
        }
    return claimed


def pending_endpoints(
    *,
    e0_dir: Path,
    val_roots: list[str],
    running_keys: set[str],
    max_fail_retries: int,
    failed: dict[str, Any],
) -> list[tuple[str, str]]:
    """Ordered list of (root, endpoint) still needed."""
    train = frozenset(TRAIN_ROOTS)
    out: list[tuple[str, str]] = []
    for root_id in val_roots:
        if root_id in train:
            continue
        rstat = root_receipt_status(e0_dir, root_id)
        if rstat == "PASS":
            continue
        for ep in ENDPOINTS:
            key = job_key(root_id, ep)
            if key in running_keys:
                continue
            est = endpoint_status(e0_dir, root_id, ep)
            if est == "PASS":
                continue
            # failed endpoint: allow limited retries
            fr = failed.get(key) or {}
            retries = int(fr.get("retries") or 0)
            if est == "FAILED" and retries >= max_fail_retries:
                continue
            out.append((root_id, ep))
    return out


def launch_endpoint(
    *,
    nhc: Path,
    generation_id: str,
    batch_id: str,
    root_id: str,
    endpoint: str,
    gpu: int,
    max_steps: int,
    log_path: Path,
    python_exe: str,
) -> int:
    env = os.environ.copy()
    env["PYTHONPATH"] = str(nhc / "src")
    env["PYTHONUNBUFFERED"] = "1"
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["OMP_NUM_THREADS"] = "2"
    env["MKL_NUM_THREADS"] = "2"
    env["OPENBLAS_NUM_THREADS"] = "2"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        "-u",
        "-m",
        "nhc_deprot.pipeline.e0_val_only",
        "--nhc0801-root",
        str(nhc),
        "--generation-id",
        generation_id,
        "--batch-id",
        batch_id,
        "--val-roots",
        root_id,
        "--endpoint",
        endpoint,
        "--parent-backend",
        "gpu",
        "--cuda-device",
        str(gpu),
        "--max-steps",
        str(int(max_steps)),
    ]
    fh = log_path.open("w", encoding="utf-8")
    fh.write(
        f"START_E0_FILL root={root_id} endpoint={endpoint} gpu={gpu} {utc()}\n"
    )
    fh.flush()
    proc = subprocess.Popen(
        cmd,
        cwd=str(nhc),
        env=env,
        stdout=fh,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    # child keeps log fd
    fh.close()
    return int(proc.pid)


def reap(state: dict[str, Any], e0_dir: Path, log_path: Path) -> None:
    running = dict(state.get("running") or {})
    still: dict[str, Any] = {}
    for key, meta in running.items():
        pid = int(meta.get("pid") or 0)
        if _pid_alive(pid):
            still[key] = meta
            continue
        root_id = str(meta.get("root_id") or key.split(":")[0])
        endpoint = str(meta.get("endpoint") or key.split(":")[-1])
        est = endpoint_status(e0_dir, root_id, endpoint)
        jlog = Path(meta.get("log") or "")
        if est == "PASS":
            state.setdefault("done", {})[key] = {
                "status": "PASS",
                "gpu": meta.get("gpu"),
                "at": utc(),
                "log": str(jlog),
            }
            state.get("failed", {}).pop(key, None)
            log(f"DONE {key} PASS gpu={meta.get('gpu')}", log_path)
        else:
            fr = state.setdefault("failed", {}).get(key, {"retries": 0})
            # count this attempt
            fr["retries"] = int(fr.get("retries") or 0) + 1
            fr["status"] = est or "FAIL"
            fr["at"] = utc()
            fr["log"] = str(jlog)
            state["failed"][key] = fr
            log(
                f"FAIL {key} status={est} retries={fr['retries']} gpu={meta.get('gpu')}",
                log_path,
            )
    state["running"] = still


def free_gpu_ids(
    *,
    max_gpu: int,
    claimed: dict[int, dict[str, Any]],
    allow_vasp_share: bool,
) -> list[int]:
    """GPUs we may place a new e0 endpoint on (not already running our e0)."""
    slots = inventory_gpus(max_gpu=max_gpu)
    free: list[tuple[int, int, int]] = []  # (process_count, used_mib, index)
    for s in slots:
        if s.index in claimed:
            continue
        if s.has_vasp and not allow_vasp_share:
            continue
        free.append((s.process_count, s.used_mib, s.index))
    free.sort()
    return [idx for _, _, idx in free]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    ap.add_argument("--generation-id", default="nhc0801-g001")
    ap.add_argument("--batch-id", default="g001")
    ap.add_argument("--max-steps", type=int, default=250)
    ap.add_argument("--max-gpu", type=int, default=8)
    ap.add_argument("--poll-seconds", type=int, default=20)
    ap.add_argument(
        "--max-running",
        type=int,
        default=8,
        help="Max concurrent e0 endpoint jobs (leave room for teacher Train fill)",
    )
    ap.add_argument("--max-fail-retries", type=int, default=2)
    ap.add_argument(
        "--allow-vasp-share",
        action="store_true",
        default=True,
        help="Co-locate on VASP cards when needed (default true for full 8-fill)",
    )
    ap.add_argument("--no-allow-vasp-share", action="store_true")
    ap.add_argument("--once", action="store_true")
    ap.add_argument(
        "--python-exe",
        default=None,
        help="Interpreter for e0_val_only (default: current)",
    )
    args = ap.parse_args(argv)

    nhc = args.nhc0801_root
    gen_root = nhc / "runs" / args.generation_id
    e0_dir = gen_root / "epoch0_val_batches" / args.batch_id / "epoch0"
    fill_dir = gen_root / "logs" / "val_e0_fill"
    state_path = fill_dir / "state.json"
    log_path = fill_dir / "fill.log"
    allow_vasp = bool(args.allow_vasp_share) and not bool(args.no_allow_vasp_share)
    py = args.python_exe or sys.executable
    val_roots = [r for r in VALIDATION_ROOTS if r not in frozenset(TRAIN_ROOTS)]

    log(
        f"val-e0-fill start batch={args.batch_id} n_val={len(val_roots)} "
        f"max_gpu={args.max_gpu} max_steps={args.max_steps} "
        f"allow_vasp_share={allow_vasp}",
        log_path,
    )

    while True:
        state = load_state(state_path)
        reap(state, e0_dir, log_path)

        # adopt externally started e0_val_only into running map
        claimed = list_live_e0_gpu_claims()
        running = dict(state.get("running") or {})
        for gpu, info in claimed.items():
            root = info.get("root_id")
            ep = info.get("endpoint")
            pid = int(info.get("pid") or 0)
            if not root or not ep or not pid:
                continue
            key = job_key(str(root), str(ep))
            if key not in running:
                running[key] = {
                    "pid": pid,
                    "gpu": int(gpu),
                    "root_id": root,
                    "endpoint": ep,
                    "log": "",
                    "adopted": True,
                    "started_at": utc(),
                }
                log(f"ADOPT {key} pid={pid} gpu={gpu}", log_path)
        state["running"] = running

        pending = pending_endpoints(
            e0_dir=e0_dir,
            val_roots=list(val_roots),
            running_keys=set(running.keys()),
            max_fail_retries=int(args.max_fail_retries),
            failed=dict(state.get("failed") or {}),
        )

        # summary for operators
        n_pass = sum(
            1
            for r in val_roots
            if root_receipt_status(e0_dir, r) == "PASS"
        )
        state["summary"] = {
            "n_val": len(val_roots),
            "n_root_pass": n_pass,
            "n_running": len(running),
            "n_pending_endpoints": len(pending),
            "claimed_gpus": sorted(claimed.keys()),
        }
        save_state(state_path, state)

        free = free_gpu_ids(
            max_gpu=int(args.max_gpu),
            claimed=claimed,
            allow_vasp_share=allow_vasp,
        )
        # Cap concurrent e0 so teacher daemon can claim GPUs for Train roots.
        max_run = max(0, int(args.max_running))
        slots = max(0, max_run - len(running))
        if slots <= 0:
            free = []
        elif len(free) > slots:
            free = free[:slots]
        if not pending:
            log(
                f"idle: all endpoints done or max-retried "
                f"root_pass={n_pass}/{len(val_roots)} running={len(running)}",
                log_path,
            )
            if n_pass == len(val_roots) and not running:
                log("ALL Val roots PASS — fill complete", log_path)
                if args.once:
                    return 0
            if args.once:
                return 0
            time.sleep(max(10, int(args.poll_seconds)))
            continue

        launched = 0
        for gpu in free:
            if not pending:
                break
            root_id, endpoint = pending.pop(0)
            key = job_key(root_id, endpoint)
            tag = f"e0_fill_{args.batch_id}_{root_id[:8]}_{endpoint}_gpu{gpu}"
            jlog = fill_dir / "jobs" / f"{tag}.out"
            try:
                pid = launch_endpoint(
                    nhc=nhc,
                    generation_id=args.generation_id,
                    batch_id=args.batch_id,
                    root_id=root_id,
                    endpoint=endpoint,
                    gpu=int(gpu),
                    max_steps=int(args.max_steps),
                    log_path=jlog,
                    python_exe=py,
                )
            except OSError as exc:
                log(f"LAUNCH_FAIL {key} gpu={gpu}: {exc}", log_path)
                continue
            running[key] = {
                "pid": pid,
                "gpu": int(gpu),
                "root_id": root_id,
                "endpoint": endpoint,
                "log": str(jlog),
                "started_at": utc(),
            }
            claimed[int(gpu)] = {"pid": pid, "root_id": root_id, "endpoint": endpoint}
            launched += 1
            log(f"RUN {key} pid={pid} gpu={gpu} log={jlog}", log_path)

        state["running"] = running
        state["summary"]["n_running"] = len(running)
        state["summary"]["n_pending_endpoints"] = len(pending)
        state["summary"]["last_launched"] = launched
        save_state(state_path, state)

        if launched:
            log(
                f"filled launched={launched} running={len(running)} "
                f"pending_left={len(pending)} root_pass={n_pass}/{len(val_roots)}",
                log_path,
            )

        if args.once:
            return 0
        time.sleep(max(5, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
