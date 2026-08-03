#!/usr/bin/env python3
"""After epoch-0 finishes: audit campaign_receipt + root receipts (read-only)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import resolve_layout  # noqa: E402
from nhc_deprot.pipeline.epoch0_receipt_audit import (  # noqa: E402
    audit_epoch0_receipts,
    format_audit_summary,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nhc0801-root",
        type=Path,
        default=Path("/home/plab/test/WJW/NHC0801"),
    )
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--json", action="store_true")
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    try:
        report = audit_epoch0_receipts(layout=layout, write_report=True)
    except Exception as exc:  # noqa: BLE001
        print(f"EPOCH0_RECEIPT_AUDIT_ERROR: {exc}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(report, indent=2, sort_keys=True))
    else:
        print(format_audit_summary(report))
    return 0 if report.get("audit_pass") else 1


if __name__ == "__main__":
    raise SystemExit(main())
