#!/usr/bin/env python3
"""Thin CLI: P5.5 zero-DFT pre-screen (logic in pipeline.ablation_cli)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.ablation_cli import main_pre_screen  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return main_pre_screen(
        argv,
        default_nhc0801_root=ROOT / "runs" / "local_nhc0801",
    )


if __name__ == "__main__":
    raise SystemExit(main())
