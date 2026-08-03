#!/usr/bin/env python3
"""Scan or write pipeline_status.json for a generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.pipeline_status import (  # noqa: E402
    scan_generation_status,
    write_pipeline_status,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--write",
        action="store_true",
        help="Write pipeline_status.json (orchestrator path); default is scan-only print",
    )
    p.add_argument("--orchestrator-running", action="store_true")
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    if args.write:
        ensure_generation_tree(layout, exist_ok=True)
        snap = write_pipeline_status(
            layout, orchestrator_running=args.orchestrator_running
        )
    else:
        snap = scan_generation_status(layout)
    print(json.dumps(snap, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
