#!/usr/bin/env python3
"""Thin CLI: phase-1 multi-seed ablation by run_id (logic in training.ablation_cli)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.training.ablation_cli import main_train_ablation  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    return main_train_ablation(
        argv,
        default_nhc0801_root=ROOT / "runs" / "local_nhc0801",
    )


if __name__ == "__main__":
    raise SystemExit(main())
