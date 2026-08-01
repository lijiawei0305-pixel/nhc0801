#!/usr/bin/env python3
"""Mindmap step 2 teacher runner CLI (default dry-run; no live PySCF)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import (  # noqa: E402
    DEFAULT_GENERATION_ID,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.pipeline.teacher_runner import (  # noqa: E402
    DryRunTeacherEngine,
    default_pilot_root_queue,
    plan_teacher_paths,
    run_teacher_campaign,
)
from nhc_deprot.resources.profiles import get_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=ROOT / "runs" / "local_nhc0801",
        help="Local sandbox (default ./runs/local_nhc0801); server: $WJW/NHC0801",
    )
    parser.add_argument("--profile", default="single_27_physical_v1")
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print path plan only; do not write frames",
    )
    parser.add_argument(
        "--frames-per-endpoint",
        type=int,
        default=2,
        help="Dry-run synthetic frames per endpoint",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Refuse unless fully authorized (currently always errors — no live engine)",
    )
    args = parser.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(
            generation_id=args.generation_id,
            nhc0801_root=args.nhc0801_root,
        )
    else:
        ensure_generation_tree(layout, exist_ok=True)

    roots = list(default_pilot_root_queue())
    if args.plan_only:
        print(json.dumps(plan_teacher_paths(layout, roots), indent=2, sort_keys=True))
        return 0

    if args.live:
        print(
            json.dumps(
                {
                    "error": "live teacher not available in this skeleton",
                    "hint": "omit --live for dry-run; inject live engine + gates later",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    profile = get_profile(args.profile)
    campaign = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=profile,
        engine=DryRunTeacherEngine(frames_per_endpoint=args.frames_per_endpoint),
        dry_run=True,
        teacher_pyscf_authorized=False,
    )
    print(json.dumps(campaign.as_dict(), indent=2, sort_keys=True))
    return 0 if campaign.status.startswith("DRY_RUN") and "PARTIAL" not in campaign.status else 1


if __name__ == "__main__":
    raise SystemExit(main())
