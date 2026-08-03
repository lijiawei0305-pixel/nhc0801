#!/usr/bin/env python3
"""Mindmap steps 8–9 dry-run: sci-val over shortlist + selection (no live DFT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.sci_val_campaign import run_sci_val_campaign  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--shortlist", type=Path, default=None)
    p.add_argument("--max-candidates", type=int, default=None)
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    camp = run_sci_val_campaign(
        layout=layout,
        shortlist_path=args.shortlist,
        dry_run=True,
        max_candidates=args.max_candidates,
    )
    sel = camp.get("selection") or {}
    print(json.dumps({
        "status": camp["status"],
        "candidate_count": camp["candidate_count"],
        "selection_outcome": sel.get("outcome"),
        "selected_epoch": sel.get("selected_epoch"),
        "final_model_selected": camp["final_model_selected"],
        "final_test_authorized": camp["final_test_authorized"],
        "receipt": str(layout.sci_val_dir / "campaign_receipt.json"),
    }, indent=2, sort_keys=True))
    return 0 if str(camp["status"]).endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
