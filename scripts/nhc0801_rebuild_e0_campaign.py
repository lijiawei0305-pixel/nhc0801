#!/usr/bin/env python3
"""Rebuild epoch0 campaign + batch receipt from PASS root receipts (no DFT)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.epoch0_campaign_rebuild import (  # noqa: E402
    rebuild_epoch0_campaign_from_root_receipts,
)
from nhc_deprot.pipeline.epoch0_receipt_audit import (  # noqa: E402
    audit_epoch0_receipts,
    format_audit_summary,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--batch-id", default="g001")
    args = p.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    ensure_generation_tree(layout, exist_ok=True)
    out = rebuild_epoch0_campaign_from_root_receipts(
        layout=layout, batch_id=args.batch_id
    )
    print(json.dumps(out, indent=2, sort_keys=True), flush=True)
    audit = audit_epoch0_receipts(layout=layout, write_report=True)
    print(format_audit_summary(audit), flush=True)
    return 0 if audit.get("audit_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
