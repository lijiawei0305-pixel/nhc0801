#!/usr/bin/env python3
"""Mindmap step 3: epoch-0 full-route dry-run under nhc0801-g001/epoch0."""

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
from nhc_deprot.pipeline.epoch0_runner import (  # noqa: E402
    plan_epoch0_paths,
    run_epoch0_campaign,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=ROOT / "runs" / "local_nhc0801",
    )
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument(
        "--live",
        action="store_true",
        help="Always fails in this skeleton (no live engines)",
    )
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

    if args.plan_only:
        print(json.dumps(plan_epoch0_paths(layout), indent=2, sort_keys=True))
        return 0

    if args.live:
        print(
            json.dumps(
                {
                    "error": "live epoch-0 not available in skeleton",
                    "required": [
                        "epoch0_execution=true",
                        "injected AIMNet2+Parent engines",
                        "resource claim PASS",
                        "official weight _0 only",
                    ],
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    campaign = run_epoch0_campaign(layout=layout, dry_run=True, epoch0_execution=False)
    print(json.dumps(campaign, indent=2, sort_keys=True))
    return 0 if str(campaign.get("status", "")).startswith("DRY_RUN_EPOCH0_PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
