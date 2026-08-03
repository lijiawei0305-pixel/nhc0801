#!/usr/bin/env python3
"""Read-only SSH terminal TUI for NHC0801 generation progress (30s default)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.dashboard.tui import run_tui_loop  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--nhc0801-root",
        type=Path,
        default=Path("/home/plab/test/WJW/NHC0801"),
    )
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--interval-s", type=float, default=30.0)
    p.add_argument("--once", action="store_true", help="Print one frame and exit")
    args = p.parse_args(argv)
    return run_tui_loop(
        nhc0801_root=args.nhc0801_root,
        generation_id=args.generation_id,
        interval_s=args.interval_s,
        once=args.once,
    )


if __name__ == "__main__":
    raise SystemExit(main())
