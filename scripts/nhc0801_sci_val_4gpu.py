#!/usr/bin/env python3
"""Multi-GPU scientific Validation: 2 roots × 2 endpoints → 4 GPUs per candidate.

Uses gpu4pyscf parent (LiveParentP01Engine backend=gpu). After e0 baseline
exists under epoch0_val_batches/g001 with matching parent_max_steps.

Example:

  PYTHONPATH=src python scripts/nhc0801_sci_val_4gpu.py \\
    --nhc0801-root $WJW/NHC0801 --max-steps 250 --max-candidates 2
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import load_json_object  # noqa: E402
from nhc_deprot.pipeline.sci_val_dispatch import (  # noqa: E402
    SciValDispatchError,
    run_sci_val_campaign_4gpu,
)


def load_pre_screen_shortlist(path: Path, *, max_candidates: int) -> list[dict]:
    payload, _ = load_json_object(path)
    short = payload.get("shortlist")
    if not isinstance(short, list) or not short:
        raise SystemExit(f"no shortlist in {path}")
    out: list[dict] = []
    for c in short[:max_candidates]:
        if not isinstance(c, dict):
            continue
        wp = c.get("weight_path")
        if not wp or not Path(str(wp)).is_file():
            raise SystemExit(f"missing weight for {c.get('checkpoint_id')}: {wp}")
        seed = c.get("seed")
        epoch = c.get("epoch")
        if type(seed) is not int or type(epoch) is not int:
            raise SystemExit(f"bad seed/epoch in {c.get('checkpoint_id')}")
        out.append(
            {
                "seed": seed,
                "epoch": epoch,
                "weight_path": str(wp),
                "checkpoint_id": c.get("checkpoint_id"),
                "run_id": c.get("run_id"),
            }
        )
    if not out:
        raise SystemExit("shortlist empty after filter")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--pre-screen",
        type=Path,
        default=None,
        help="screen_campaign.json (default: live_phase1_v002)",
    )
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=250)
    p.add_argument(
        "--epoch0-max-steps",
        type=int,
        default=None,
        help="Must match --max-steps (default: same as --max-steps)",
    )
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-wait",
        action="store_true",
        help="Launch first candidate shards and exit (debug)",
    )
    args = p.parse_args(argv)

    root = args.nhc0801_root
    pre = args.pre_screen or (
        root
        / "runs"
        / args.generation_id
        / "pre_screen_g001"
        / "live_phase1_v002"
        / "screen_campaign.json"
    )
    candidates = load_pre_screen_shortlist(pre, max_candidates=args.max_candidates)
    print(
        json.dumps(
            {
                "pre_screen": str(pre),
                "candidates": candidates,
                "max_steps": args.max_steps,
                "epoch0_max_steps": args.epoch0_max_steps or args.max_steps,
            },
            indent=2,
        ),
        flush=True,
    )

    try:
        camp = run_sci_val_campaign_4gpu(
            nhc0801_root=root,
            generation_id=args.generation_id,
            candidates=candidates,
            max_steps=int(args.max_steps),
            epoch0_max_steps=(
                int(args.epoch0_max_steps)
                if args.epoch0_max_steps is not None
                else int(args.max_steps)
            ),
            max_candidates=int(args.max_candidates),
            dry_run=bool(args.dry_run),
            wait=not bool(args.no_wait),
        )
    except SciValDispatchError as exc:
        print(json.dumps({"error": str(exc), "status": "FAIL"}, indent=2))
        return 2
    except Exception as exc:  # noqa: BLE001
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"},
                indent=2,
            )
        )
        return 1

    print(
        json.dumps(
            {
                "status": camp.get("status"),
                "selection": camp.get("selection"),
                "candidate_count": camp.get("candidate_count"),
                "final_model_selected": camp.get("final_model_selected"),
                "parent_max_steps": camp.get("parent_max_steps"),
            },
            indent=2,
            default=str,
        ),
        flush=True,
    )
    st = str(camp.get("status") or "")
    return 0 if "PASS" in st or "PLANNED" in st or "LAUNCHED" in st else 1


if __name__ == "__main__":
    raise SystemExit(main())
