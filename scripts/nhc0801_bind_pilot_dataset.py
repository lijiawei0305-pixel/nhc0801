#!/usr/bin/env python3
"""Bind read-only pilot weighted/D3 products into nhc0801-g001 (no recompute)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.data.weighted_dataset import audit_weighted_dataset  # noqa: E402
from nhc_deprot.generation.layout import (  # noqa: E402
    DEFAULT_GENERATION_ID,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)

DEFAULT_WJW = Path("/home/plab/test/WJW")
DEFAULT_WEIGHTED = DEFAULT_WJW / "data/runs/phase9b_aimnet2_v004_weighted_dataset_v001"
DEFAULT_D3 = DEFAULT_WJW / "data/runs/phase9b_aimnet2_v004_d3_projection_v001"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=DEFAULT_WJW / "NHC0801",
    )
    parser.add_argument("--weighted-src", type=Path, default=DEFAULT_WEIGHTED)
    parser.add_argument("--d3-src", type=Path, default=DEFAULT_D3)
    parser.add_argument(
        "--mode",
        choices=("symlink", "bind-receipt-only"),
        default="symlink",
        help="symlink datasets/weighted to pilot product, or only write binding JSON",
    )
    args = parser.parse_args(argv)

    if not args.weighted_src.is_dir():
        print(json.dumps({"error": f"missing weighted src {args.weighted_src}"}), file=sys.stderr)
        return 2
    if not (args.weighted_src / "manifest.json").is_file():
        print(json.dumps({"error": "weighted src missing manifest.json"}), file=sys.stderr)
        return 2

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(
            generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
        )
    else:
        ensure_generation_tree(layout, exist_ok=True)

    # datasets/weighted may already be a directory from dry-run — require empty or symlink
    target = layout.datasets_dir
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() or target.is_symlink():
        if target.is_symlink():
            target.unlink()
        elif target.is_dir() and not any(target.iterdir()):
            target.rmdir()
        else:
            print(
                json.dumps(
                    {
                        "error": f"datasets dir exists and is non-empty: {target}",
                        "hint": "use a fresh generation or remove dry-run datasets manually",
                    }
                ),
                file=sys.stderr,
            )
            return 3

    if args.mode == "symlink":
        os.symlink(args.weighted_src.resolve(), target, target_is_directory=True)

    # optional d3 symlink
    d3_link = layout.d3_dir
    d3_bound = False
    if args.d3_src.is_dir():
        if d3_link.exists() or d3_link.is_symlink():
            if d3_link.is_symlink():
                d3_link.unlink()
            elif d3_link.is_dir() and not any(d3_link.iterdir()):
                d3_link.rmdir()
            else:
                d3_link = layout.generation_root / "d3_pilot_link"
        if not d3_link.exists():
            os.symlink(args.d3_src.resolve(), d3_link, target_is_directory=True)
            d3_bound = True

    audit = audit_weighted_dataset(
        target if args.mode == "symlink" else args.weighted_src,
        expected_schema="phase9b-aimnet2-development-dataset-v004",
    )
    binding = {
        "schema": "nhc0801-pilot-dataset-binding-v1",
        "created_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation_id": layout.generation_id,
        "mode": args.mode,
        "weighted_src": str(args.weighted_src.resolve()),
        "d3_src": str(args.d3_src.resolve()) if args.d3_src.is_dir() else None,
        "weighted_bind_path": str(target),
        "d3_bound": d3_bound,
        "d3_bind_path": str(d3_link) if d3_bound else None,
        "audit_status": audit.status,
        "frame_count": audit.frame_count,
        "frame_count_by_split": audit.frame_count_by_split,
        "split_weight_sums": audit.split_weight_sums,
        "d3_recomputation_performed": False,
        "final_test_payload_read": False,
        "notes": [
            "read-only reuse of pilot V004 weighted dataset + D3 product",
            "no silent D3 recompute",
            "scope C development roots only",
        ],
    }
    write_json(layout.meta_dir / "pilot_dataset_binding.json", binding, overwrite=True)
    print(json.dumps(binding, indent=2, sort_keys=True))
    return 0 if audit.status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
