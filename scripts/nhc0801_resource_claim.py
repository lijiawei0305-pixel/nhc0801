#!/usr/bin/env python3
"""Read-only two-sample resource claim (local or SSH). Never starts chemistry."""

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
from nhc_deprot.resources.claim_runner import run_resource_claim  # noqa: E402
from nhc_deprot.resources.host_sampler import (  # noqa: E402
    load_project_root_from_config,
    load_ssh_alias_from_config,
)
from nhc_deprot.resources.profiles import get_profile  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=None,
        help="Local generation root parent (default: ./runs/local_nhc0801 or from config)",
    )
    parser.add_argument("--profile", default="single_27_physical_v1")
    parser.add_argument(
        "--mode",
        choices=("local", "ssh"),
        default="ssh",
        help="ssh = remote probe via BatchMode; local = sample this machine",
    )
    parser.add_argument("--ssh-alias", default=None, help="Override configs/server.local.yaml")
    parser.add_argument(
        "--disk-path",
        default=None,
        help="Filesystem path for free-space check (default: remote project_root or /)",
    )
    parser.add_argument("--interval-s", type=float, default=5.0)
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs" / "server.local.yaml",
    )
    args = parser.parse_args(argv)

    # Validate profile exists early
    profile = get_profile(args.profile)

    nhc_root = args.nhc0801_root
    if nhc_root is None:
        nhc_root = ROOT / "runs" / "local_nhc0801"

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=nhc_root)
    if not layout.generation_meta_path().is_file():
        init_generation(generation_id=args.generation_id, nhc0801_root=nhc_root)
    else:
        ensure_generation_tree(layout, exist_ok=True)

    ssh_alias = args.ssh_alias or load_ssh_alias_from_config(args.config)
    disk_path = args.disk_path
    if disk_path is None:
        if args.mode == "ssh":
            disk_path = load_project_root_from_config(args.config) or "/home/plab/test/WJW/NHC0801"
        else:
            disk_path = str(Path.cwd())

    if args.mode == "ssh" and not ssh_alias:
        print(
            json.dumps(
                {
                    "error": "ssh mode requires --ssh-alias or configs/server.local.yaml",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 2

    try:
        out = run_resource_claim(
            layout=layout,
            profile_id=profile.profile_id,
            mode=args.mode,
            ssh_alias=ssh_alias,
            disk_path=disk_path,
            interval_s=args.interval_s,
            chemistry_authorized=False,
        )
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {
                    "error": str(exc),
                    "type": type(exc).__name__,
                    "hint": "check SSH BatchMode / campus net / CPU list",
                },
                indent=2,
            ),
            file=sys.stderr,
        )
        return 3

    summary = {
        "status": out["status"],
        "claim_id": out["claim_id"],
        "profile_id": out["profile_id"],
        "chemistry_run_allowed": out["chemistry_run_allowed"],
        "reasons": out["reasons"],
        "receipt_path": out["receipt_path"],
        "mode": args.mode,
        "cpu_lists": list(profile.cpu_lists),
        "note": "read-only claim; chemistry gates remain closed",
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if out["status"] == "LIVE_RESOURCE_CLAIM_PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
