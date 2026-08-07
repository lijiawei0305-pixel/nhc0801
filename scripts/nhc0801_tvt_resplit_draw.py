#!/usr/bin/env python3
"""Draw TVT resplit 150/16/3 and write commitment artifacts (scheme A)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.tvt_resplit import (  # noqa: E402
    DEFAULT_FT_N,
    DEFAULT_SEED,
    DEFAULT_TRAIN_N,
    DEFAULT_VAL_N,
    draw_tvt_resplit,
    load_pool_csv,
    write_resplit_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--pool-csv",
        type=Path,
        default=ROOT / "docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv",
    )
    p.add_argument(
        "--usable-roots",
        type=Path,
        required=True,
        help="Text file: one InChIKey per line (frame_count>2 both ends)",
    )
    p.add_argument(
        "--all-teacher-roots",
        type=Path,
        required=True,
        help="Text file: any root with teacher dir presence (for FT exclusion)",
    )
    p.add_argument("--train-n", type=int, default=DEFAULT_TRAIN_N)
    p.add_argument("--val-n", type=int, default=DEFAULT_VAL_N)
    p.add_argument("--ft-n", type=int, default=DEFAULT_FT_N)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=ROOT / "data" / "splits",
    )
    args = p.parse_args(argv)

    pool = load_pool_csv(args.pool_csv)
    usable = [
        ln.strip()
        for ln in args.usable_roots.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    all_t = [
        ln.strip()
        for ln in args.all_teacher_roots.read_text(encoding="utf-8").splitlines()
        if ln.strip()
    ]
    registry = draw_tvt_resplit(
        pool=pool,
        usable_teacher_roots=usable,
        all_teacher_roots=all_t,
        train_n=int(args.train_n),
        val_n=int(args.val_n),
        ft_n=int(args.ft_n),
        seed=int(args.seed),
    )
    paths = write_resplit_artifacts(registry, out_dir=args.out_dir)
    summary = {
        "status": "TVT_RESPLIT_DRAW_OK",
        "seed": args.seed,
        "counts": registry["counts"],
        "audit": registry["audit_split_registry"]["status"],
        "sealed_ft": registry["sealed_final_test_commitment"],
        "locked_val": registry["locked_validation_roots"],
        "paths": {k: str(v) for k, v in paths.items()},
        "train_teacher_ready": sum(1 for r in registry["train_roots"] if r in set(usable)),
        "train_pending_teacher": sum(
            1 for r in registry["train_roots"] if r not in set(usable)
        ),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
