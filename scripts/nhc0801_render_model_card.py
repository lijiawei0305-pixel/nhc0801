#!/usr/bin/env python3
"""Render models/vX.Y/card.json + card.svg for a release (or dry demo)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# repo root on PYTHONPATH
_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "src"))

from nhc_deprot.generation.layout import resolve_layout  # noqa: E402
from nhc_deprot.training.model_card import (  # noqa: E402
    ModelCardFeatures,
    card_features_from_info,
    write_model_card,
)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=None)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--version", default="v0.1", help="e.g. v0.1 or 0.1")
    p.add_argument(
        "--from-info",
        type=Path,
        default=None,
        help="Existing models/vX.Y/info.json (else demo features)",
    )
    p.add_argument(
        "--metrics-json",
        type=Path,
        default=None,
        help="Optional JSON with frame/sci metrics for the card",
    )
    p.add_argument("--demo", action="store_true", help="Write a filled demo card")
    args = p.parse_args()

    layout = resolve_layout(
        generation_id=args.generation_id,
        nhc0801_root=args.nhc0801_root,
    )
    metrics: dict = {}
    if args.metrics_json and args.metrics_json.is_file():
        metrics = json.loads(args.metrics_json.read_text(encoding="utf-8"))

    if args.from_info and args.from_info.is_file():
        info = json.loads(args.from_info.read_text(encoding="utf-8"))
        feat = card_features_from_info(info, extras=metrics)
    elif args.demo:
        # Illustrative numbers so bars render; replace on real release
        feat = ModelCardFeatures(
            version=args.version,
            train_batch_id="g001",
            train_roots=3,
            train_frames=120,
            seed=20260730,
            epoch=200,
            energy_mae=0.85,
            energy_rmse=1.2,
            force_mae=0.04,
            force_rmse=0.06,
            deprot_label_mae=1.5,
            vs_epoch0_opt_steps_ratio=0.72,
            vs_epoch0_wall_ratio=0.80,
            handoff_pass_rate=1.0,
            topology_pass_rate=1.0,
            n_val_roots_eval=2,
            weight_sha256_short="a1b2c3d4e5f6…",
            notes=["demo"],
        )
    else:
        feat = ModelCardFeatures(
            version=args.version,
            **{k: v for k, v in metrics.items() if hasattr(ModelCardFeatures, k)},
        )

    paths = write_model_card(layout, feat, overwrite=True)
    print(json.dumps(paths, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
