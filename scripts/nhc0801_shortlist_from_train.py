#!/usr/bin/env python3
"""Mindmap step 7 — aggregate quick shortlists from g001/train seed receipts."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.checkpoint_shortlist import run_shortlist_campaign  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--train-dir", type=Path, default=None, help="Override train dir")
    p.add_argument("--recompute", action="store_true")
    p.add_argument("--max-per-seed", type=int, default=None)
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    camp = run_shortlist_campaign(
        layout=layout,
        maximum_count_per_seed=args.max_per_seed,
        recompute=args.recompute,
        train_dir=args.train_dir,
    )
    print(json.dumps({
        "status": camp["status"],
        "candidate_count": camp["candidate_count"],
        "weights_present_count": camp["weights_present_count"],
        "receipt_path": camp.get("receipt_path"),
        "per_seed_shortlists": {
            str(s["seed"]): s["shortlist_epochs"] for s in camp["per_seed"]
        },
        "final_model_selected": camp["final_model_selected"],
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
