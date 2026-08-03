#!/usr/bin/env python3
"""Mindmap step 10 — write freeze_manifest under g001/freeze (no Final Test open)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.freeze_package import build_freeze_package  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--repo-root", type=Path, default=ROOT)
    p.add_argument("--require-selection", action="store_true")
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    pkg = build_freeze_package(
        layout=layout,
        repo_root=args.repo_root,
        require_selection=args.require_selection,
    )
    print(json.dumps({
        "status": pkg["status"],
        "model_state": pkg["model_state"],
        "selection_outcome": (pkg.get("selection") or {}).get("outcome"),
        "final_test_ready": pkg["final_test_ready"],
        "receipt_path": pkg.get("receipt_path"),
        "artifacts": list((pkg.get("artifacts") or {}).keys()),
    }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
