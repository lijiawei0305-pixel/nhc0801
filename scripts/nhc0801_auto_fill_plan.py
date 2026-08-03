#!/usr/bin/env python3
"""Plan V002 auto-fill endpoint slots (no chemistry).

Example:
  PYTHONPATH=src python scripts/nhc0801_auto_fill_plan.py \\
    --idle-cpus 0-63 --mem-gib 200 --roots A,B --endpoints both
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.resources.auto_fill import build_auto_fill_plan  # noqa: E402
from nhc_deprot.resources.host_sampler import expand_cpu_list  # noqa: E402
from nhc_deprot.resources.profiles import OFFICIAL_DEFAULT_V002, get_profile  # noqa: E402
from nhc_deprot.resources.profiles import worker_env_for_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--profile", default=OFFICIAL_DEFAULT_V002)
    p.add_argument("--idle-cpus", required=True, help="e.g. 0-111 or 0,2-27")
    p.add_argument("--mem-gib", type=float, required=True)
    p.add_argument("--roots", required=True, help="comma-separated root ids")
    p.add_argument(
        "--endpoints",
        default="both",
        choices=("both", "cation", "neutral"),
    )
    p.add_argument("--claim-pass", action="store_true")
    args = p.parse_args(argv)

    prof = get_profile(args.profile)
    idle = expand_cpu_list(args.idle_cpus)
    roots = [r.strip() for r in args.roots.split(",") if r.strip()]
    eps = ("cation", "neutral") if args.endpoints == "both" else (args.endpoints,)
    queue = [(r, e) for r in roots for e in eps]
    plan = build_auto_fill_plan(
        idle_cpu_ids=idle,
        mem_available_bytes=int(args.mem_gib * (1024**3)),
        endpoint_queue=queue,
        profile=prof,
        claim_pass=args.claim_pass,
    )
    env_example = worker_env_for_profile(prof)
    out = plan.as_dict()
    out["worker_env_template"] = env_example
    print(json.dumps(out, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
