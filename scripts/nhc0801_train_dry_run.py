#!/usr/bin/env python3
"""Mindmap 4–5 multi-seed trainer dry-run over g001 weighted dataset."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.generation.layout import (  # noqa: E402
    DEFAULT_GENERATION_ID,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.pipeline.d3_projection import run_d3_campaign  # noqa: E402
from nhc_deprot.pipeline.teacher_runner import (  # noqa: E402
    DryRunTeacherEngine,
    run_teacher_campaign,
)
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset  # noqa: E402
from nhc_deprot.resources.profiles import get_profile  # noqa: E402
from nhc_deprot.training.config import TrainingConfig  # noqa: E402
from nhc_deprot.training.multi_seed_trainer import run_multi_seed_training  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=ROOT / "runs" / "local_nhc0801",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Dry-run epoch count")
    parser.add_argument(
        "--bootstrap-data",
        action="store_true",
        help="If weighted dataset missing, run teacher→D3→weighted dry-run first",
    )
    parser.add_argument("--frames-per-endpoint", type=int, default=2)
    parser.add_argument("--live", action="store_true", help="Always fails (no live backend)")
    args = parser.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(
            generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
        )
    else:
        ensure_generation_tree(layout, exist_ok=True)

    if args.live:
        print(
            json.dumps(
                {
                    "error": "live train not available in skeleton",
                    "required": [
                        "aimnet2_train_authorized=true",
                        "non-dry TrainBackend (torch/AIMNet2)",
                        "epoch-0 baseline receipt",
                        "resource claim PASS",
                    ],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    manifest = layout.datasets_dir / "manifest.json"
    if not manifest.is_file():
        if not args.bootstrap_data:
            print(
                json.dumps(
                    {
                        "error": f"missing weighted dataset: {manifest}",
                        "hint": "pass --bootstrap-data or run nhc0801_d3_weighted_dry_run.py",
                    },
                    indent=2,
                ),
                file=sys.stderr,
            )
            return 1
        roots = list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
        run_teacher_campaign(
            layout=layout,
            root_ids=roots,
            profile=get_profile("single_27_physical_v1"),
            engine=DryRunTeacherEngine(frames_per_endpoint=args.frames_per_endpoint),
            dry_run=True,
        )
        run_d3_campaign(layout=layout, root_ids=roots, dry_run=True, overwrite=True)
        assemble_weighted_dataset(
            layout=layout,
            train_roots=list(TRAIN_ROOTS),
            validation_roots=list(VALIDATION_ROOTS),
            dry_run=True,
            overwrite=True,
        )

    cfg = TrainingConfig()
    campaign = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=args.epochs,
        aimnet2_train_authorized=False,
    )
    # Compact stdout
    summary = {
        "status": campaign["status"],
        "generation_id": campaign["generation_id"],
        "epochs_effective": campaign["epochs_effective"],
        "seeds": list(cfg.seeds),
        "checkpoint_count": campaign["checkpoint_count"],
        "failed_seed_count": campaign["failed_seed_count"],
        "final_model_selected": campaign["final_model_selected"],
        "quick_validation_may_select_final_model": False,
        "train_dir": str(layout.train_dir),
        "campaign_receipt": str(layout.train_dir / "campaign_receipt.json"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if campaign["status"] == "DRY_RUN_TRAIN_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
