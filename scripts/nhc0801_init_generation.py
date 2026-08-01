#!/usr/bin/env python3
"""Create NHC0801 generation tree + generation.json (no chemistry)."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import DEFAULT_GENERATION_ID, init_generation  # noqa: E402


def _git_head() -> str | None:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except OSError:
        return None
    return None


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument(
        "--nhc0801-root",
        type=Path,
        default=ROOT / "runs" / "local_nhc0801",
        help="Local sandbox root (default: ./runs/local_nhc0801). Server: $WJW/NHC0801",
    )
    parser.add_argument("--source-commit", default=None)
    args = parser.parse_args(argv)
    commit = args.source_commit or _git_head()
    layout, meta, receipt = init_generation(
        generation_id=args.generation_id,
        nhc0801_root=args.nhc0801_root,
        source_commit=commit,
    )
    print(
        json.dumps(
            {
                "generation_id": layout.generation_id,
                "generation_root": str(layout.generation_root),
                "meta": meta.as_dict(),
                "receipt": receipt,
                "live_chemistry": False,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
