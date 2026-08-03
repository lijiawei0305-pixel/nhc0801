#!/usr/bin/env python3
"""Resource-aware live orchestration: claim → epoch-0 (optional) → multi-seed train.

Designed to run **on the server** under:
  - GPU train: source mlff.sh, CUDA_VISIBLE_DEVICES set
  - CPU claim: affinity 0,2-27

Epoch-0 full parent DFT is long; use --skip-epoch0-live to only train after
binding pilot pure references later. Default: try live epoch-0 then live train.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.data.paths import (  # noqa: E402
    OFFICIAL_AIMNET2_WEIGHT_SHA256,
    VALIDATION_ROOTS,
)
from nhc_deprot.generation.layout import (  # noqa: E402
    DEFAULT_GENERATION_ID,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.pipeline.epoch0_runner import Epoch0Config, run_epoch0_campaign  # noqa: E402
from nhc_deprot.pipeline.scientific_validation import FrozenEndpointGeometry  # noqa: E402
from nhc_deprot.resources.claim_runner import run_resource_claim  # noqa: E402
from nhc_deprot.training.config import TrainingConfig  # noqa: E402
from nhc_deprot.training.live_aimnet2 import LiveAimnet2TrainBackend  # noqa: E402
from nhc_deprot.training.multi_seed_trainer import run_multi_seed_training  # noqa: E402


def _load_xyz(path: Path) -> FrozenEndpointGeometry:
    from nhc_deprot.pipeline.live_epoch0 import load_xyz
    from nhc_deprot.contracts.parent_protocol import (
        CATION_CHARGE,
        CATION_MULTIPLICITY,
        NEUTRAL_CHARGE,
        NEUTRAL_MULTIPLICITY,
    )

    # path name: KEY_cation.xyz
    name = path.name
    root_id = name.rsplit("_", 1)[0]
    endpoint = "cation" if name.endswith("_cation.xyz") else "neutral"
    elements, coords = load_xyz(path)
    charge, mult = (
        (CATION_CHARGE, CATION_MULTIPLICITY)
        if endpoint == "cation"
        else (NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY)
    )
    return FrozenEndpointGeometry(
        root_id=root_id,
        endpoint=endpoint,
        elements=tuple(elements),
        coordinates=tuple(tuple(c) for c in coords),
        charge=charge,
        multiplicity=mult,
        geometry_sha256="",
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    parser.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument(
        "--base-weight",
        type=Path,
        default=Path("/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt"),
    )
    parser.add_argument(
        "--gold-xyz-dir",
        type=Path,
        default=Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
    )
    parser.add_argument("--skip-epoch0-live", action="store_true")
    parser.add_argument("--skip-train-live", action="store_true")
    parser.add_argument("--train-epochs", type=int, default=None, help="Override epochs (default 200)")
    parser.add_argument("--seeds", type=str, default="20260730,20260731,20260732")
    parser.add_argument("--claim-interval-s", type=float, default=3.0)
    parser.add_argument(
        "--allow-train-without-cpu-claim",
        action="store_true",
        help="GPU train may proceed even if CPU affinity claim is busy (epoch0 uses CPUs)",
    )
    parser.add_argument(
        "--allow-epoch0-without-cpu-claim",
        action="store_true",
        help="Proceed with live epoch-0 even if affinity claim is busy (use carefully)",
    )
    parser.add_argument(
        "--pyscf-python",
        type=Path,
        default=Path("/home/plab/test/WJW/env/conda/gpupyscf/bin/python"),
        help="Python for Parent-P01 worker (gpupyscf or molenv)",
    )
    parser.add_argument(
        "--epoch0-max-steps",
        type=int,
        default=100,
        help="Parent geometric maxsteps per endpoint",
    )
    args = parser.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    else:
        ensure_generation_tree(layout, exist_ok=True)

    report: dict = {
        "started_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "generation_id": layout.generation_id,
        "gpu_index": args.gpu_index,
    }

    # 1) resource claim
    claim = run_resource_claim(
        layout=layout,
        profile_id="single_27_physical_v1",
        mode="local",
        disk_path=str(args.nhc0801_root),
        interval_s=args.claim_interval_s,
        chemistry_authorized=True,
    )
    report["claim"] = {
        "status": claim["status"],
        "chemistry_run_allowed": claim["chemistry_run_allowed"],
        "reasons": claim["reasons"],
        "receipt_path": claim["receipt_path"],
    }
    claim_pass = claim["status"] == "LIVE_RESOURCE_CLAIM_PASS"
    if not claim_pass:
        # GPU-only train can continue if explicitly allowed (CPU may be used by epoch0)
        if args.skip_epoch0_live and args.allow_train_without_cpu_claim and not args.skip_train_live:
            report["claim_note"] = "CPU claim failed; proceeding GPU train only by flag"
        elif (
            args.skip_train_live
            and not args.skip_epoch0_live
            and args.allow_epoch0_without_cpu_claim
        ):
            report["claim_note"] = "CPU claim failed; proceeding epoch-0 only by flag"
        elif args.skip_train_live and not args.skip_epoch0_live:
            write_json(layout.logs_dir / "live_orchestrate_report.json", report, overwrite=True)
            print(json.dumps(report, indent=2))
            return 2
        elif not args.allow_train_without_cpu_claim and not (
            args.allow_epoch0_without_cpu_claim and not args.skip_epoch0_live
        ):
            write_json(layout.logs_dir / "live_orchestrate_report.json", report, overwrite=True)
            print(json.dumps(report, indent=2))
            return 2

    # 2) live epoch-0
    if not args.skip_epoch0_live:
        try:
            from nhc_deprot.pipeline.live_epoch0 import (
                LiveAimnet2GauLooseEngine,
                LiveParentP01Engine,
            )

            geos = []
            for root in VALIDATION_ROOTS:
                for ep in ("cation", "neutral"):
                    xyz = args.gold_xyz_dir / f"{root}_{ep}.xyz"
                    if not xyz.is_file():
                        raise FileNotFoundError(xyz)
                    geos.append(_load_xyz(xyz))
            # g001 Epoch-0 disk = epoch0_val_batches/g001/ (same pattern as g002…)
            from dataclasses import replace as _dc_replace

            e0_layout = _dc_replace(
                layout,
                epoch0_dir=layout.epoch0_batch_dir("g001"),
                logs_dir=layout.epoch0_batch_root("g001") / "logs",
            )
            e0_layout.epoch0_dir.mkdir(parents=True, exist_ok=True)
            e0_layout.logs_dir.mkdir(parents=True, exist_ok=True)
            print(
                f"[epoch0] g001 Epoch-0 starting live route roots={list(VALIDATION_ROOTS)} "
                f"endpoints=4 max_steps={args.epoch0_max_steps} "
                f"epoch0_dir={e0_layout.epoch0_dir}",
                flush=True,
            )
            aim = LiveAimnet2GauLooseEngine(weight_path=args.base_weight)
            parent = LiveParentP01Engine(
                max_steps=int(args.epoch0_max_steps),
                pyscf_python=str(args.pyscf_python),
            )
            # Pure reference uses same parent engine from frozen gold (no AIMNet2)
            e0 = run_epoch0_campaign(
                layout=e0_layout,
                config=Epoch0Config(validation_roots=VALIDATION_ROOTS),
                dry_run=False,
                epoch0_execution=True,
                aimnet2=aim,
                parent=parent,
                pure_parent=parent,
                geometries=geos,
            )
            report["epoch0"] = {
                "status": e0.get("status"),
                "failed_root_count": e0.get("failed_root_count"),
            }
            print(f"[epoch0] finished status={e0.get('status')}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report["epoch0"] = {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
            write_json(layout.logs_dir / "live_orchestrate_report.json", report, overwrite=True)
            print(json.dumps(report, indent=2), flush=True)
            # Continue to train only if user allowed skip; else abort train when e0 fails
            if not args.skip_train_live:
                # mindmap: epoch-0 before train — fail closed
                return 3
    else:
        report["epoch0"] = {"status": "SKIPPED"}

    # 3) live multi-seed train
    if not args.skip_train_live:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
        cfg = TrainingConfig(seeds=seeds)
        if args.train_epochs is not None:
            # frozen dataclass: rebuild
            cfg = TrainingConfig(seeds=seeds, epochs=int(args.train_epochs))
        try:
            # Custom multi-seed loop using live backend per seed
            seed_results = []
            for seed in cfg.seeds:
                backend = LiveAimnet2TrainBackend(
                    dataset_root=layout.datasets_dir,
                    base_weight=args.base_weight,
                    config=cfg,
                    seed=seed,
                    device="cuda",
                    batches_per_epoch=-1,
                )
                # monkey-compatible path: use run_one_seed with dummy batches
                from nhc_deprot.training.multi_seed_trainer import run_one_seed

                dummy_batches: list = []  # backend ignores and uses SizeGrouped loaders
                one = run_one_seed(
                    layout=layout,
                    seed=seed,
                    config=cfg,
                    train_batches=dummy_batches,
                    val_batches=dummy_batches,
                    train_frame_count=backend.train_frame_count,
                    backend=backend,
                    epochs=cfg.epochs,
                    dry_run=False,
                )
                # write real weights for last checkpoint
                if one.status == "PASS" and one.checkpoints:
                    last = one.checkpoints[-1]
                    pt_path = layout.train_dir / f"seed_{seed}" / f"epoch_{last['epoch']:04d}.pt"
                    export = backend.export_checkpoint(pt_path)
                    last["live_weights_written"] = True
                    last["weight_export"] = export
                seed_results.append(one.as_dict())
            failed = sum(1 for s in seed_results if s.get("status") != "PASS")
            campaign = {
                "schema": "nhc0801-multi-seed-train-campaign-v1",
                "dry_run": False,
                "live_chemistry": True,
                "aimnet2_train_authorized": True,
                "status": "LIVE_TRAIN_PASS" if failed == 0 else "LIVE_TRAIN_PARTIAL",
                "seed_results": seed_results,
                "failed_seed_count": failed,
                "final_model_selected": False,
                "quick_validation_may_select_final_model": False,
                "official_base_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
                "gpu_index": args.gpu_index,
            }
            write_json(layout.train_dir / "campaign_receipt_live.json", campaign, overwrite=True)
            report["train"] = {
                "status": campaign["status"],
                "failed_seed_count": failed,
                "seeds": list(seeds),
                "epochs": cfg.epochs,
            }
        except Exception as exc:  # noqa: BLE001
            report["train"] = {
                "status": "FAILED",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
            write_json(layout.logs_dir / "live_orchestrate_report.json", report, overwrite=True)
            print(json.dumps(report, indent=2))
            return 4
    else:
        report["train"] = {"status": "SKIPPED"}

    report["finished_at_utc"] = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    write_json(layout.logs_dir / "live_orchestrate_report.json", report, overwrite=True)
    print(json.dumps(report, indent=2, sort_keys=True))
    e0_ok = report.get("epoch0", {}).get("status") in {
        "DRY_RUN_EPOCH0_PASS",
        "LIVE_EPOCH0_PASS",
        "SKIPPED",
    } or str(report.get("epoch0", {}).get("status", "")).endswith("PASS")
    tr_ok = report.get("train", {}).get("status") in {"LIVE_TRAIN_PASS", "SKIPPED"}
    return 0 if e0_ok and tr_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
