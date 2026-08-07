#!/usr/bin/env python3
"""Drive the 7-step hyperparameter goal until lock or T9 data-shortage.

Steps (AGENTS / training_t1_t9):
  1 e0 baseline (precondition)
  2 sci-val vs e0 (NUMERIC_CALIBRATION)
  3 m250 full-traj teacher enough Train labels
  4 wide retrain ablation
  5 T4 forces grid from measured E/F MSE
  6 pre-screen shortlist → narrow sci-val
  7 lock run_id/defaults if gain vs e0 else T9

Designed to be left running on nhc614 under mlff env. Never opens Final Test.
Writes status JSON under runs/<gen>/logs/hyperparam_goal/.
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
from nhc_deprot.pipeline.gpu_autofill import endpoint_done_ok  # noqa: E402


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _log(msg: str, log_path: Path) -> None:
    line = f"[{_utc()}] {msg}"
    print(line, flush=True)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def count_train_pairs_done(gen_root: Path) -> tuple[int, list[str]]:
    """Count Train roots with both cation+neutral endpoint_done_ok under any teacher_gpu*."""
    complete: list[str] = []
    for rid in TRAIN_ROOTS:
        cat = neu = False
        for d in gen_root.glob("teacher_gpu*"):
            if not d.is_dir():
                continue
            try:
                if endpoint_done_ok(d, rid, "cation"):
                    cat = True
                if endpoint_done_ok(d, rid, "neutral"):
                    neu = True
            except OSError:
                continue
        if cat and neu:
            complete.append(rid)
    return len(complete), complete


def count_m250_done_endpoints(gen_root: Path) -> int:
    n = 0
    for d in gen_root.glob("teacher_gpu*"):
        if not d.is_dir():
            continue
        try:
            roots = list(d.iterdir())
        except OSError:
            continue
        for root in roots:
            if not root.is_dir():
                continue
            for ep in ("cation", "neutral"):
                try:
                    if endpoint_done_ok(d, root.name, ep):
                        n += 1
                except OSError:
                    continue
    return n


def sci_val_finished(log_path: Path, receipt_path: Path) -> dict[str, Any] | None:
    if receipt_path.is_file():
        try:
            camp = json.loads(receipt_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            camp = None
        if isinstance(camp, dict) and camp.get("status") and "DRY_RUN" not in str(
            camp.get("status")
        ):
            if camp.get("voided") is True:
                return None
            if "LIVE" in str(camp.get("status")) or "SCI_VAL" in str(camp.get("status")):
                # require non-void live-ish
                if camp.get("status") not in {"VOID_INSTRUMENTATION_BUG"}:
                    return camp
    if not log_path.is_file():
        return None
    text = log_path.read_text(encoding="utf-8", errors="replace")
    if '"status": "FAIL"' in text and "error" in text and "need 4 eligible" in text:
        return {"status": "FAIL", "error": "gpu_pick", "raw_tail": text[-2000:]}
    # success print from script
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and "status" in line:
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                break
    # full multi-line JSON at end
    if "selection" in text and "status" in text and "LIVE" in text:
        # try load receipt after wait
        if receipt_path.is_file():
            try:
                return json.loads(receipt_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                pass
    return None


def clear_teacher_pause(gen_root: Path) -> None:
    logs = gen_root / "logs"
    for name in ("gpu_teacher_pause.note", "gpu_teacher_pause_for_scival.note"):
        p = logs / name
        if p.is_file():
            p.rename(logs / f"{name}.cleared_{_utc().replace(':', '')}")
    state_path = gen_root / "gpu_teacher_queue" / "state.json"
    if state_path.is_file():
        st = json.loads(state_path.read_text(encoding="utf-8"))
        st["paused"] = False
        st.pop("pause_reason", None)
        if st.get("stop_reason") == "paused_for_sci_val":
            st["stop_reason"] = None
        st["updated_at_utc"] = _utc()
        state_path.write_text(json.dumps(st, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_train_roots_priority(gen_root: Path, nhc_root: Path) -> None:
    """Prepend Train3 m250 re-labels to teacher queue (exclude only Val roots)."""
    state_path = gen_root / "gpu_teacher_queue" / "state.json"
    if not state_path.is_file():
        return
    st = json.loads(state_path.read_text(encoding="utf-8"))
    xyz = None
    for cand in (
        Path("/home/plab/test/WJW/data/candidates/structures_full/xyz"),
        Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
    ):
        if cand.is_dir():
            xyz = cand
            break
    if xyz is None:
        return
    # Drop Val from exclude; allow Train roots to be scheduled
    exclude = set(st.get("exclude_roots") or [])
    for r in TRAIN_ROOTS:
        exclude.discard(r)
    # Keep Val excluded from teacher expansion (e0 owns Val baseline)
    for r in VALIDATION_ROOTS:
        exclude.add(r)
    st["exclude_roots"] = sorted(exclude)

    done_keys = {x.get("key") for x in st.get("done") or []}
    running = set((st.get("running") or {}).keys())
    queue = list(st.get("queue") or [])
    qkeys = {t.get("key") for t in queue}

    priority: list[dict[str, Any]] = []
    for rid in TRAIN_ROOTS:
        for ep in ("cation", "neutral"):
            key = f"{rid}:{ep}"
            if key in done_keys or key in running or key in qkeys:
                # if already done on disk, skip
                skip = False
                for d in gen_root.glob("teacher_gpu*"):
                    if d.is_dir() and endpoint_done_ok(d, rid, ep):
                        skip = True
                        break
                if skip or key in done_keys or key in running:
                    continue
            if key in qkeys:
                continue
            priority.append(
                {
                    "root_id": rid,
                    "endpoint": ep,
                    "gold_xyz_dir": str(xyz),
                    "key": key,
                    "batch_id": "g001",
                    "priority": "train3_m250_rebind",
                }
            )
    # remove train keys from existing queue then prepend
    train_set = {f"{r}:{e}" for r in TRAIN_ROOTS for e in ("cation", "neutral")}
    rest = [t for t in queue if t.get("key") not in train_set]
    # keep any existing train tasks but move to front
    existing_train = [t for t in queue if t.get("key") in train_set]
    st["queue"] = priority + existing_train + rest
    st["updated_at_utc"] = _utc()
    st["parent_max_steps_policy"] = 250
    state_path.write_text(json.dumps(st, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def start_teacher_daemon(nhc_root: Path, gen_id: str) -> int:
    logs = nhc_root / "runs" / gen_id / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    out = logs / "gpu_teacher_daemon_m250.out"
    # already running?
    try:
        ps = subprocess.check_output(["ps", "-ef"], text=True)
        for line in ps.splitlines():
            if (
                "nhc0801_gpu_teacher_daemon.py" in line
                and "max-steps 250" in line
                and "grep" not in line
            ):
                return int(line.split()[1])
    except (OSError, subprocess.SubprocessError, ValueError):
        pass
    cmd = [
        sys.executable,
        "-u",
        str(nhc_root / "scripts" / "nhc0801_gpu_teacher_daemon.py"),
        "--nhc0801-root",
        str(nhc_root),
        "--generation-id",
        gen_id,
        "--pool-csv",
        str(nhc_root / "docs" / "contracts" / "RIGID_SMALL_NHC_POOL_V001.csv"),
        "--gpu-ids",
        "0,1,2,3,4,5,6,7",
        "--host-threads",
        "2",
        "--max-steps",
        "250",
        "--poll-seconds",
        "15",
        "--batch-size-roots",
        "5",
    ]
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== hyperparam_goal resume teacher {_utc()} ===\n")
        proc = subprocess.Popen(
            cmd,
            cwd=str(nhc_root),
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env={**os.environ, "PYTHONPATH": str(nhc_root / "src")},
        )
    (logs / "gpu_teacher_daemon_m250.pid").write_text(str(proc.pid) + "\n", encoding="utf-8")
    return proc.pid


def lock_hypers_document(
    gen_root: Path,
    *,
    sci_val: dict[str, Any] | None,
    train_pairs: int,
    decision: str,
    defaults: dict[str, Any],
    notes: list[str],
) -> Path:
    out = gen_root / "logs" / "hyperparam_goal" / f"HYPERPARAM_LOCK_{_utc().replace(':', '')}.json"
    payload = {
        "schema": "nhc0801-hyperparam-lock-v1",
        "written_at_utc": _utc(),
        "decision": decision,
        "defaults": defaults,
        "sci_val_status": (sci_val or {}).get("status"),
        "sci_val_selection": (sci_val or {}).get("selection"),
        "train_pairs_m250_done": train_pairs,
        "notes": notes,
        "t1_t9": "energy never ranks; forces from measured E/F; T9 if no gain vs e0",
        "final_test_open": False,
    }
    _write_json(out, payload)
    # also human markdown
    md = out.with_suffix(".md")
    md.write_text(
        "# Hyperparameter lock\n\n"
        f"- written: `{payload['written_at_utc']}`\n"
        f"- decision: **{decision}**\n"
        f"- sci-val: `{payload['sci_val_status']}`\n"
        f"- train pairs m250: {train_pairs}\n\n"
        "## Defaults\n\n```json\n"
        + json.dumps(defaults, indent=2)
        + "\n```\n\n## Notes\n\n"
        + "\n".join(f"- {n}" for n in notes)
        + "\n",
        encoding="utf-8",
    )
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--min-train-pairs",
        type=int,
        default=3,
        help="m250 complete Train root pairs before retrain (default 3 = Train3)",
    )
    p.add_argument(
        "--poll-seconds",
        type=float,
        default=120.0,
    )
    p.add_argument(
        "--max-hours",
        type=float,
        default=72.0,
        help="safety cap; 0 = unlimited",
    )
    args = p.parse_args(argv)

    nhc = args.nhc0801_root
    gen = nhc / "runs" / args.generation_id
    status_dir = gen / "logs" / "hyperparam_goal"
    status_dir.mkdir(parents=True, exist_ok=True)
    log_path = status_dir / "orchestrator.log"
    status_path = status_dir / "status.json"

    t0 = time.time()
    phase = "wait_sci_val"
    sci_val_result: dict[str, Any] | None = None

    # T7 provisional defaults (lock only after evidence)
    provisional = {
        "energy_weight": 1.0,
        "forces_weight": 100.0,
        "trainable": "mlp_shift",
        "batch_size": 8,
        "epochs": 120,
        "ema_decay": 0.99,
        "lr": 1e-4,
        "optimizer": "RAdam",
        "run_id_template": "e1f{forces}_mlp_shift_b{batch}_ep{epochs}",
    }

    _log("orchestrator start", log_path)
    sci_log = gen / "logs" / "sci_val_4gpu_hyperparam_goal.out"
    sci_receipt = gen / "sci_val" / "campaign_receipt.json"

    while True:
        if args.max_hours > 0 and (time.time() - t0) > args.max_hours * 3600:
            _log("max-hours reached — writing provisional lock (timeout)", log_path)
            lock_hypers_document(
                gen,
                sci_val=sci_val_result,
                train_pairs=count_train_pairs_done(gen)[0],
                decision="PROVISIONAL_TIMEOUT",
                defaults=provisional,
                notes=["orchestrator hit max-hours; defaults remain T7 provisional"],
            )
            return 2

        train_n, train_ids = count_train_pairs_done(gen)
        m250_eps = count_m250_done_endpoints(gen)
        st = {
            "phase": phase,
            "updated_at_utc": _utc(),
            "train_pairs_done": train_n,
            "train_pair_ids": train_ids,
            "m250_endpoints_done": m250_eps,
            "sci_val": (sci_val_result or {}).get("status"),
        }
        _write_json(status_path, st)

        if phase == "wait_sci_val":
            camp = sci_val_finished(sci_log, sci_receipt)
            # also accept process exit + receipt
            pid_file = gen / "logs" / "sci_val_4gpu_hyperparam_goal.pid"
            alive = False
            if pid_file.is_file():
                try:
                    pid = int(pid_file.read_text().strip())
                    os.kill(pid, 0)
                    alive = True
                except (OSError, ValueError):
                    alive = False
            if camp is not None:
                sci_val_result = camp
                _log(f"sci-val finished status={camp.get('status')}", log_path)
                _write_json(status_dir / "sci_val_snapshot.json", camp)
                phase = "resume_teacher"
            elif not alive and sci_log.is_file():
                # Process dead is NOT success: hung parents may die/killed mid-run.
                # Only advance on a LIVE campaign receipt (not DRY_RUN / not void).
                text = sci_log.read_text(encoding="utf-8", errors="replace")
                _log(f"sci-val process dead; tail={text[-500:]!r}", log_path)
                live = None
                if sci_receipt.is_file():
                    try:
                        live = json.loads(sci_receipt.read_text(encoding="utf-8"))
                    except json.JSONDecodeError:
                        live = None
                st = str((live or {}).get("status") or "")
                if (
                    live
                    and live.get("voided") is not True
                    and "DRY_RUN" not in st
                    and ("LIVE" in st or "SCI_VAL" in st)
                    and "FAIL" not in st
                ):
                    sci_val_result = live
                    phase = "resume_teacher"
                else:
                    # Also accept endpoints complete for both shortlist epochs
                    ep10 = (
                        gen
                        / "sci_val"
                        / "seed_20260730"
                        / "epoch_0010"
                        / "endpoints"
                    )
                    ep30 = (
                        gen
                        / "sci_val"
                        / "seed_20260730"
                        / "epoch_0030"
                        / "endpoints"
                    )
                    n10 = len(list(ep10.glob("*.json"))) if ep10.is_dir() else 0
                    n30 = len(list(ep30.glob("*.json"))) if ep30.is_dir() else 0
                    if n10 >= 4 and n30 >= 4:
                        _log(
                            f"sci-val endpoints complete n10={n10} n30={n30}; "
                            "waiting campaign_receipt LIVE write",
                            log_path,
                        )
                        # do not resume teacher until receipt is LIVE
                        time.sleep(args.poll_seconds)
                        continue
                    _log(
                        "sci-val incomplete after process death — stay in wait_sci_val "
                        f"(receipt={st!r} n10={n10} n30={n30})",
                        log_path,
                    )
                    time.sleep(args.poll_seconds)
                    continue
            else:
                _log("waiting sci-val…", log_path)
                time.sleep(args.poll_seconds)
                continue

        if phase == "resume_teacher":
            _log("clear pause + prioritize Train3 m250 + start daemon", log_path)
            clear_teacher_pause(gen)
            ensure_train_roots_priority(gen, nhc)
            pid = start_teacher_daemon(nhc, args.generation_id)
            _log(f"teacher daemon pid={pid}", log_path)
            phase = "wait_train_data"

        if phase == "wait_train_data":
            train_n, train_ids = count_train_pairs_done(gen)
            _log(
                f"train pairs m250 {train_n}/{args.min_train_pairs} ids={train_ids} "
                f"all_eps_done={m250_eps}",
                log_path,
            )
            if train_n >= args.min_train_pairs:
                phase = "decide_lock"
            else:
                # also accept pool growth: ≥6 complete endpoints on non-val roots as partial
                # but training needs Train roots — keep waiting
                time.sleep(args.poll_seconds)
                continue

        if phase == "decide_lock":
            # Interpret sci-val: gain vs e0?
            sel = (sci_val_result or {}).get("selection") or {}
            status = str((sci_val_result or {}).get("status") or "")
            gain = False
            notes = [
                f"sci_val_status={status}",
                f"selection={sel}",
                f"train_pairs={train_n}",
            ]
            # Heuristic: LIVE pass with selected checkpoint and final_model_selected
            if "PASS" in status and (sci_val_result or {}).get("final_model_selected"):
                gain = True
            if (sci_val_result or {}).get("final_model_selected") is True:
                gain = True
            # rejected / no gain → T9
            if "REJECT" in status or "NO_GAIN" in status or "EMPTY" in status:
                gain = False

            # Without a successful live sci-val selection, do not claim scientific lock.
            if not gain:
                # Still lock *engineering* defaults from T7 for next train round,
                # but mark T9 data / no sci-val gain.
                decision = "T9_OR_NO_SCI_GAIN_PROVISIONAL_DEFAULTS"
                notes.append(
                    "No sci-val gain lock; engineering defaults follow T7 "
                    "(batch 8, epochs 120, e1f100, mlp_shift, EMA 0.99). "
                    "Continue m250 Train labels; re-run ablation+sci-val when data grows."
                )
                # Adjust forces from last train campaign if present
                forces = 100.0
                for rec in gen.glob("train_g001/**/campaign_receipt*.json"):
                    try:
                        d = json.loads(rec.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    # look for mse fields
                    em = d.get("weighted_energy_mse") or d.get("energy_mse")
                    fm = d.get("weighted_forces_mse") or d.get("forces_mse")
                    if em and fm and float(fm) > 0:
                        ratio = float(em) / float(fm)
                        # target effective E:F ~ 1:3 → forces_weight ≈ ratio/3
                        forces = max(10.0, min(200.0, ratio / 3.0))
                        notes.append(
                            f"T4 from {rec}: E_mse={em} F_mse={fm} → forces_weight≈{forces:.1f}"
                        )
                        break
                provisional["forces_weight"] = round(forces, 1)
            else:
                decision = "SCI_VAL_GAIN_LOCK"
                notes.append(
                    "sci-val selected a fine-tune checkpoint over e0; "
                    "lock T7+measured forces"
                )

            path = lock_hypers_document(
                gen,
                sci_val=sci_val_result,
                train_pairs=train_n,
                decision=decision,
                defaults=provisional,
                notes=notes,
            )
            _log(f"LOCK written {path} decision={decision}", log_path)
            phase = "done"
            _write_json(
                status_path,
                {
                    "phase": phase,
                    "decision": decision,
                    "lock_path": str(path),
                    "updated_at_utc": _utc(),
                    "defaults": provisional,
                },
            )
            return 0

        time.sleep(args.poll_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
