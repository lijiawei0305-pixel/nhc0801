#!/usr/bin/env python3
"""Path A diagnosis watcher (user charter 2026-08-05).

Phases:
  wait_train3 → ready_retrain → (manual/auto handoff notes)

Does not open Final Test. Train priority is enforced by gpu_teacher_daemon
rebuild_queue (TRAIN_ROOTS first). This process only monitors gates and
writes status for operators / later orchestrator steps.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.pipeline.gpu_autofill import endpoint_done_ok  # noqa: E402


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _write(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def train_pair_status(gen: Path) -> dict[str, dict[str, bool]]:
    out: dict[str, dict[str, bool]] = {}
    for rid in TRAIN_ROOTS:
        cat = neu = False
        for d in gen.glob("teacher_gpu*"):
            if not d.is_dir():
                continue
            try:
                cat = cat or endpoint_done_ok(d, rid, "cation")
                neu = neu or endpoint_done_ok(d, rid, "neutral")
            except OSError:
                continue
        out[rid] = {"cation": cat, "neutral": neu, "pair": cat and neu}
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--poll-seconds", type=int, default=60)
    p.add_argument("--once", action="store_true")
    args = p.parse_args(argv)

    gen = args.nhc0801_root / "runs" / args.generation_id
    status_path = gen / "logs" / "path_a_diagnosis" / "status.json"
    log_path = gen / "logs" / "path_a_diagnosis" / "watch.log"

    def log(msg: str) -> None:
        line = f"[{_utc()}] {msg}"
        print(line, flush=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    charter = {
        "schema": "nhc0801-path-a-diagnosis-v1",
        "path": "A_diagnostic",
        "expand_before_round2_sci_val": False,
        "expand_after_no_gain": {
            "train_frac": 0.8,
            "val_frac": 0.2,
            "eligible": "double_end_m250_PASS_exclude_final_test",
            "e0": "keep_old_val_e0_plus_new_val_roots_only",
        },
        "retrain": {
            "width": "small_wide",
            "run_ids": [
                "e1f1_mlp",
                "e1f100_mlp",
                "e1f1_mlp_shift",
                "e1f100_mlp_shift",
            ],
        },
        "final_test": "SEALED",
        "train_roots": list(TRAIN_ROOTS),
        "val_roots_frozen_until_expand": list(VALIDATION_ROOTS),
    }

    log("path_a watch start")
    while True:
        pairs = train_pair_status(gen)
        n_ok = sum(1 for v in pairs.values() if v["pair"])
        phase = "wait_train3" if n_ok < len(TRAIN_ROOTS) else "ready_retrain"
        payload = {
            "charter": charter,
            "phase": phase,
            "train_pairs_done": n_ok,
            "train_pairs_target": len(TRAIN_ROOTS),
            "train_pair_detail": pairs,
            "next_if_ready": [
                "rebuild_train_weighted_npz_from_m250_teacher",
                "live_ablation_4_run_ids",
                "shortlist_few_epochs",
                "sci_val_4gpu_live_vs_current_e0",
                "if_no_gain: draft_80_20_expand_receipt",
            ],
            "updated_at_utc": _utc(),
        }
        _write(status_path, payload)
        log(f"phase={phase} train_pairs={n_ok}/{len(TRAIN_ROOTS)} detail={pairs}")
        if phase == "ready_retrain":
            ready = gen / "logs" / "path_a_diagnosis" / "READY_RETRAIN.flag"
            ready.write_text(_utc() + "\n", encoding="utf-8")
            log("READY_RETRAIN flag written — hand off to retrain/sci-val pipeline")
            if args.once:
                return 0
            # stay alive reporting; do not exit so ops can see freshness
        if args.once:
            return 0
        time.sleep(max(15, int(args.poll_seconds)))


if __name__ == "__main__":
    raise SystemExit(main())
