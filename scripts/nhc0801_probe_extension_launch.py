#!/usr/bin/env python3
"""Launch long-probe / w_f=1 control (plan 20260807_probe_extension_and_matrix).

Thin wrapper: builds TrainingConfig for early-stop or fixed-epoch probes and
calls run_train_ablation. Does not open sci-val / Final Test.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import (  # noqa: E402
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.training.ablation_cli import run_train_ablation  # noqa: E402
from nhc_deprot.training.config import TrainingConfig  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--batch-id", default="g001")
    p.add_argument("--run-id", required=True, help="e1f100_mlp_shift or e1f1_mlp_shift")
    p.add_argument("--seed", type=int, default=20260730)
    p.add_argument("--base-weight", type=Path, required=True)
    p.add_argument("--device", default="cuda")
    p.add_argument(
        "--mode",
        choices=("long_early_stop", "fixed"),
        required=True,
        help="long_early_stop: patience+cap; fixed: plain epochs",
    )
    p.add_argument("--epochs", type=int, default=120, help="fixed mode epochs")
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--cap", type=int, default=480)
    p.add_argument("--checkpoint-interval", type=int, default=1)
    p.add_argument("--aimnet2-train-authorized", action="store_true")
    args = p.parse_args(argv)

    if not args.aimnet2_train_authorized:
        print(json.dumps({"error": "need --aimnet2-train-authorized"}), flush=True)
        return 2
    if not args.base_weight.is_file():
        print(json.dumps({"error": f"missing base weight: {args.base_weight}"}), flush=True)
        return 2

    if args.mode == "long_early_stop":
        base = TrainingConfig(
            seeds=(int(args.seed),),
            epochs=int(args.cap),
            early_stop_patience_epochs=int(args.patience),
            early_stop_max_epochs=int(args.cap),
            early_stop_metric="validation_weighted_loss",
            checkpoint_interval_epochs=int(args.checkpoint_interval),
        )
    else:
        base = TrainingConfig(
            seeds=(int(args.seed),),
            epochs=int(args.epochs),
            early_stop_patience_epochs=None,
            checkpoint_interval_epochs=int(args.checkpoint_interval),
        )

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    else:
        ensure_generation_tree(layout, exist_ok=True)

    print(
        json.dumps(
            {
                "event": "probe_launch",
                "mode": args.mode,
                "run_id": args.run_id,
                "seed": args.seed,
                "config": base.as_dict(),
                "base_weight": str(args.base_weight),
                "device": args.device,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )

    summary = run_train_ablation(
        layout=layout,
        run_ids=[str(args.run_id)],
        train_batch_id=str(args.batch_id),
        dry_run=False,
        aimnet2_train_authorized=True,
        base_config=base,
        base_weight=Path(args.base_weight),
        device=str(args.device),
        require_merge_meta=True,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str), flush=True)
    status = str(summary.get("status", ""))
    return 0 if status.endswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
