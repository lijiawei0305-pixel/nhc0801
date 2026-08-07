#!/usr/bin/env python3
"""Phase A: read-only inventory of double-end teacher PASS roots.

Writes ``eligible_full_pairs.json`` under the generation logs dir (default).
Does not stop teacher, does not train, does not open Final Test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.eligible_full_pairs import (  # noqa: E402
    DEFAULT_TARGET_TRAIN_ROOTS,
    DEFAULT_TARGET_VAL_ROOTS,
    build_inventory,
    write_inventory,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help=(
            "Output JSON path (default: "
            "runs/<gen>/logs/eligible_full_pairs/eligible_full_pairs.json)"
        ),
    )
    p.add_argument("--target-train", type=int, default=DEFAULT_TARGET_TRAIN_ROOTS)
    p.add_argument("--target-val", type=int, default=DEFAULT_TARGET_VAL_ROOTS)
    p.add_argument(
        "--queue-state",
        type=Path,
        default=None,
        help="Optional gpu_teacher_queue/state.json (auto-detected if omitted)",
    )
    args = p.parse_args(argv)

    gen = Path(args.nhc0801_root) / "runs" / args.generation_id
    if not gen.is_dir():
        print(json.dumps({"error": f"missing generation root {gen}"}), flush=True)
        return 2

    out = args.out
    if out is None:
        out = gen / "logs" / "eligible_full_pairs" / "eligible_full_pairs.json"

    payload = build_inventory(
        gen,
        queue_state_path=args.queue_state,
        target_train_roots=int(args.target_train),
        target_val_roots=int(args.target_val),
    )
    path = write_inventory(payload, out)
    c = payload["counts"]
    summary = {
        "status": "OK",
        "inventory_path": str(path),
        "n_full_pairs": c["n_full_pairs"],
        "n_eligible_for_expanded_train": c["n_eligible_for_expanded_train"],
        "gap_to_train_lock_150": c["gap_to_train_lock_150"],
        "train_lock_ready": c["train_lock_ready"],
        "n_incomplete": c["n_incomplete"],
        "suggested_total_before_lock": c["suggested_total_full_pairs_before_split_lock"],
        "gap_suggested_total": c["gap_suggested_total_full_pairs"],
        "queue": payload.get("queue"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
