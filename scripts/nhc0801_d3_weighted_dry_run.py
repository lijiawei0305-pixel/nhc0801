#!/usr/bin/env python3
"""Mindmap residual path dry-run: teacher (optional) → D3 → weighted NPZ under g001."""

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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=ROOT / "runs" / "local_nhc0801",
    )
    parser.add_argument("--frames-per-endpoint", type=int, default=2)
    parser.add_argument(
        "--skip-teacher",
        action="store_true",
        help="Assume teacher frames already exist under g001/teacher",
    )
    parser.add_argument("--overwrite-d3", action="store_true")
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

    train = list(TRAIN_ROOTS)
    val = list(VALIDATION_ROOTS)
    all_roots = train + val
    report: dict = {"generation_id": layout.generation_id, "dry_run": True}

    if not args.skip_teacher:
        # Fresh sandbox: use overwrite-friendly by using empty teacher tree only once
        teacher = run_teacher_campaign(
            layout=layout,
            root_ids=all_roots,
            profile=get_profile("single_27_physical_v1"),
            engine=DryRunTeacherEngine(frames_per_endpoint=args.frames_per_endpoint),
            dry_run=True,
        )
        report["teacher"] = {
            "status": teacher.status,
            "pool_progress": teacher.pool_progress,
        }
        if teacher.status != "DRY_RUN_COMPLETE":
            print(json.dumps(report, indent=2))
            return 1

    d3 = run_d3_campaign(
        layout=layout,
        root_ids=all_roots,
        dry_run=True,
        overwrite=args.overwrite_d3,
    )
    report["d3"] = {
        "status": d3["status"],
        "frame_count": d3["frame_count"],
    }

    weighted = assemble_weighted_dataset(
        layout=layout,
        train_roots=train,
        validation_roots=val,
        dry_run=True,
        overwrite=True,
        run_audit=True,
    )
    report["weighted"] = weighted
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if weighted.get("status") == "DRY_RUN_WEIGHTED_DATASET_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
