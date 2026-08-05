#!/usr/bin/env python3
"""Thin CLI: paired (seed x epoch) recipe contrast over a pre-screen receipt.

Logic lives in ``nhc_deprot.pipeline.paired_recipe_contrast``. Zero DFT, zero
GPU: reads ``screen_campaign.json`` and writes ``paired_recipe_contrast.json``
next to it. Screening only — never final model selection.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from nhc_deprot.pipeline.paired_recipe_contrast import (  # noqa: E402
    PairedContrastError,
    run_paired_contrast_for_screen,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="nhc0801_paired_contrast.py")
    p.add_argument("screen_campaign", type=Path, help="path to screen_campaign.json")
    p.add_argument("--write", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument(
        "--include-hard-gate-failures",
        action="store_true",
        help="keep candidates whose hard gates failed (default: drop them)",
    )
    args = p.parse_args(argv)
    try:
        report = run_paired_contrast_for_screen(
            args.screen_campaign,
            require_hard_gates=not args.include_hard_gate_failures,
            write=bool(args.write),
        )
    except PairedContrastError as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, indent=2))
        return 1
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
