#!/usr/bin/env python3
"""Live scientific Validation (mindmap 8–9) vs g001 Epoch-0 baseline.

- Uses pre_screen shortlist (default top 2) with real .pt weights
- Pure labels + e0 baseline from rebuilt root receipts (no pure recompute)
- AIMNet2: LiveCheckpointGauLooseEngine (finetune weights OK)
- Parent: LiveParentP01Engine gpu backend
- Final Test remains sealed

Run under mlff-capable env for AIMNet2; parent worker uses gpupyscf python.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import load_json_object, write_json  # noqa: E402
from nhc_deprot.data.paths import VALIDATION_ROOTS  # noqa: E402
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.e0_val_only import load_geo  # noqa: E402
from nhc_deprot.pipeline.epoch0_campaign_rebuild import (  # noqa: E402
    epoch0_baseline_from_root_receipts,
    load_root_receipts,
    pure_references_from_root_receipts,
)
from nhc_deprot.pipeline.live_epoch0 import LiveParentP01Engine  # noqa: E402
from nhc_deprot.pipeline.live_pre_screen_engine import LiveCheckpointGauLooseEngine  # noqa: E402
from nhc_deprot.pipeline.sci_val_campaign import run_sci_val_campaign  # noqa: E402
from nhc_deprot.resources.gpu_inventory import pick_gpus  # noqa: E402


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_pre_screen_shortlist(
    path: Path, *, max_candidates: int
) -> list[dict]:
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
                "weight_present": True,
                "checkpoint_id": c.get("checkpoint_id"),
                "run_id": c.get("run_id"),
                "pre_screen_mean_rmsd": c.get("mean_rmsd_to_reference_angstrom"),
                "pre_screen_mean_aim_steps": c.get("mean_aimnet2_steps_to_gau_loose"),
                "pre_screen_mean_force_rmse": c.get(
                    "mean_force_rmse_at_reference_ev_per_a"
                ),
            }
        )
    if not out:
        raise SystemExit("shortlist empty after filter")
    return out


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, required=True)
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--batch-id", default="g001")
    p.add_argument(
        "--pre-screen",
        type=Path,
        default=None,
        help="screen_campaign.json (default: pre_screen_g001/live_phase1_v002/...)",
    )
    p.add_argument("--max-candidates", type=int, default=2)
    p.add_argument("--cuda-device", type=int, default=None)
    p.add_argument("--max-steps", type=int, default=None)
    p.add_argument(
        "--epoch0-max-steps",
        type=int,
        default=None,
        help=(
            "Parent max_steps used for the epoch-0 baseline being compared. "
            "Must match --max-steps or campaign raises BASELINE_CONFIG_MISMATCH."
        ),
    )
    p.add_argument(
        "--dry-run-wiring",
        action="store_true",
        help="Only write candidate plan JSON; no live chemistry",
    )
    args = p.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    ensure_generation_tree(layout, exist_ok=True)
    e0_dir = layout.epoch0_batch_dir(args.batch_id)
    roots = list(VALIDATION_ROOTS)
    root_receipts = load_root_receipts(e0_dir, roots)

    pre_path = args.pre_screen or (
        layout.generation_root
        / "pre_screen_g001"
        / "live_phase1_v002"
        / "screen_campaign.json"
    )
    candidates = load_pre_screen_shortlist(pre_path, max_candidates=args.max_candidates)

    plan = {
        "schema": "nhc0801-sci-val-live-plan-v1",
        "started_at_utc": _utc(),
        "generation_id": layout.generation_id,
        "batch_id": args.batch_id,
        "validation_roots": roots,
        "candidates": candidates,
        "pre_screen": str(pre_path),
        "scientific_validation_live": True,
        "final_test_authorized": False,
        "dry_run_wiring": bool(args.dry_run_wiring),
    }
    write_json(layout.sci_val_dir / "live_plan.json", plan, overwrite=True)
    write_json(layout.logs_dir / "sci_val_live_plan.json", plan, overwrite=True)
    print(json.dumps(plan, indent=2), flush=True)

    if args.dry_run_wiring:
        print("dry-run wiring only; exit", flush=True)
        return 0

    golds = (
        Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
        Path("/home/plab/test/WJW/data/candidates/structures_full/xyz"),
        Path("/home/plab/test/WJW/data/candidates/xyz"),
    )
    geos = []
    for rid in roots:
        for ep in ("cation", "neutral"):
            geos.append(load_geo(rid, ep, golds))
    refs = pure_references_from_root_receipts(root_receipts)
    e0_baseline = epoch0_baseline_from_root_receipts(root_receipts)

    cuda = args.cuda_device
    if cuda is None:
        try:
            picked = pick_gpus(n=1, allow_shared=True, require_free=False)
        except Exception as exc:  # noqa: BLE001
            raise SystemExit(f"no GPU for parent/aimnet2: {exc}") from exc
        cuda = int(picked[0])
    print(f"[sci-val-live] parent+aimnet cuda_device={cuda}", flush=True)

    parent_max_steps = int(args.max_steps or 250)
    # Default e0 cap matches candidates only when re-running under matched config.
    # Pass --epoch0-max-steps explicitly when baseline was produced under another cap
    # (e.g. historical 100 vs sci-val 250 → BASELINE_CONFIG_MISMATCH).
    epoch0_max_steps = int(
        args.epoch0_max_steps
        if args.epoch0_max_steps is not None
        else parent_max_steps
    )
    parent = LiveParentP01Engine(
        max_steps=parent_max_steps,
        backend="gpu",
        cuda_device=cuda,
        host_threads=2,
    )

    def aimnet2_factory(cand: dict):
        return LiveCheckpointGauLooseEngine(
            weight_path=Path(str(cand["weight_path"])),
            max_steps=args.max_steps,
            device=f"cuda:{cuda}",
        )

    try:
        camp = run_sci_val_campaign(
            layout=layout,
            dry_run=False,
            scientific_validation_live=True,
            candidates=candidates,
            geometries=geos,
            references=refs,
            parent=parent,
            aimnet2_factory=aimnet2_factory,
            epoch0_baseline=e0_baseline,
            max_candidates=args.max_candidates,
            parent_max_steps=parent_max_steps,
            epoch0_parent_max_steps=epoch0_max_steps,
        )
    except Exception as exc:  # noqa: BLE001
        err = {
            "status": "LIVE_SCI_VAL_FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-3000:],
            "finished_at_utc": _utc(),
        }
        write_json(layout.logs_dir / "sci_val_live_error.json", err, overwrite=True)
        print(json.dumps(err, indent=2), flush=True)
        return 2

    camp["finished_at_utc"] = _utc()
    write_json(layout.logs_dir / "sci_val_live_done.json", camp, overwrite=True)
    print(
        json.dumps(
            {
                "status": camp.get("status"),
                "selection": camp.get("selection"),
                "candidate_count": camp.get("candidate_count"),
                "final_model_selected": camp.get("final_model_selected"),
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
