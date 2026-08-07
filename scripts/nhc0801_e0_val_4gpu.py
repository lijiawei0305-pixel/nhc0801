#!/usr/bin/env python3
"""Launch Val-only Epoch-0: N roots × cation/neutral across GPUs (分开算).

Thin CLI — logic in ``nhc_deprot.pipeline.e0_val_dispatch``.

Example (on server, mlff env) — full 8-GPU wave (4 roots × 2 endpoints):

  PYTHONPATH=src python scripts/nhc0801_e0_val_4gpu.py \\
    --nhc0801-root $WJW/NHC0801 --batch-id g001 \\
    --val-roots r1,r2,r3,r4 --gpu-ids 0,1,2,3,4,5,6,7 --allow-vasp-share

  # plan only (no spawn):
  PYTHONPATH=src python scripts/nhc0801_e0_val_4gpu.py --nhc0801-root $WJW/NHC0801 \\
    --batch-id g001 --dry-run
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.e0_val_dispatch import (  # noqa: E402
    E0ValDispatchError,
    endpoints_as_table,
    launch_val_e0_4gpu,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--batch-id", default="g001")
    p.add_argument(
        "--val-roots",
        default=None,
        help="Comma-separated Val roots (e.g. 4 roots for 8 endpoints on 8 GPUs)",
    )
    p.add_argument("--parent-backend", choices=("cpu", "gpu"), default="gpu")
    p.add_argument(
        "--max-steps",
        type=int,
        default=250,
        help="Parent geomeTRIC max steps (GAU_LOOSE_V001 / P2 default 250; not AIMNet2 budget)",
    )
    p.add_argument("--max-gpu", type=int, default=8)
    p.add_argument(
        "--exclude-gpu",
        action="append",
        default=None,
        help="GPU index to exclude (repeatable)",
    )
    p.add_argument(
        "--require-free",
        action="store_true",
        help="Only use GPUs with zero compute apps (strict)",
    )
    p.add_argument(
        "--no-shared",
        action="store_true",
        help="Prefer empty GPUs; fail if must share (see pick_gpus)",
    )
    p.add_argument(
        "--allow-vasp-share",
        action="store_true",
        help=(
            "If no VASP-free card exists, co-locate on lowest-mem VASP GPUs "
            "(never kills VASP). Use when machine is fully co-occupied."
        ),
    )
    p.add_argument(
        "--gpu-ids",
        default=None,
        help="Comma-separated GPU indices (skip pick_gpus); e.g. 0,1,2,3,4,5,6,7",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print plan and pick GPUs without spawning",
    )
    args = p.parse_args(argv)

    roots = None
    if args.val_roots:
        roots = [r.strip() for r in str(args.val_roots).split(",") if r.strip()]
    exclude = [int(x) for x in (args.exclude_gpu or [])]
    gpu_ids = None
    if args.gpu_ids:
        gpu_ids = [int(x.strip()) for x in str(args.gpu_ids).split(",") if x.strip()]

    try:
        receipt = launch_val_e0_4gpu(
            nhc0801_root=args.nhc0801_root,
            generation_id=args.generation_id,
            batch_id=args.batch_id,
            val_roots=roots,
            parent_backend=args.parent_backend,
            parent_max_steps=int(args.max_steps),
            max_gpu=int(args.max_gpu),
            exclude_gpus=exclude,
            gpu_ids=gpu_ids,
            require_free=bool(args.require_free),
            allow_shared=not bool(args.no_shared),
            allow_vasp_share=bool(args.allow_vasp_share),
            dry_run=bool(args.dry_run),
        )
    except E0ValDispatchError as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, indent=2))
        return 2

    print(endpoints_as_table(receipt))
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
