#!/usr/bin/env python3
"""Evaluate synthetic or JSON host snapshots against resource claim gates (no SSH)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.resources.claim import (  # noqa: E402
    HostSnapshot,
    evaluate_claim,
    pilot_v002_busy_samples,
)
from nhc_deprot.resources.profiles import get_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", default="single_27_physical_v1")
    parser.add_argument(
        "--demo-v002-busy",
        action="store_true",
        help="Evaluate synthetic V002-like busy CPU samples (expect REJECTED)",
    )
    parser.add_argument(
        "--demo-idle-pass",
        action="store_true",
        help="Evaluate synthetic idle/high-memory samples (expect PASS for single)",
    )
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    if args.demo_v002_busy:
        samples = pilot_v002_busy_samples()
    elif args.demo_idle_pass:
        samples = (
            HostSnapshot(False, 250_000_000_000, 0.0, 0.0, 200_000_000_000, "t0"),
            HostSnapshot(False, 248_000_000_000, 0.0, 0.0, 199_000_000_000, "t1"),
        )
    else:
        parser.error("choose --demo-v002-busy or --demo-idle-pass (live SSH sampler not enabled)")
    result = evaluate_claim(samples=samples, profile=profile)
    print(json.dumps(result.as_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "LIVE_RESOURCE_CLAIM_PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
