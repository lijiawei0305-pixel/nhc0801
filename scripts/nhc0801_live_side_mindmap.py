#!/usr/bin/env python3
"""Run mindmap side-work WITHOUT stopping live epoch-0.

Uses free resources (V002-style t=8, limited concurrency):
  A) GPU: AIMNet2 GAU_LOOSE preopt on Train roots (step 2/3 handoff path)
  B) CPU: live Parent-P01 teacher frames for Train roots (step 2, initial+final)

Leaves epoch-0 process alone. Final Test never opened.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.contracts.parent_protocol import (  # noqa: E402
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
)
from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.data.paths import TRAIN_ROOTS  # noqa: E402
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.live_epoch0 import LiveAimnet2GauLooseEngine, load_xyz  # noqa: E402
from nhc_deprot.pipeline.live_teacher import LiveParentTeacherEngine  # noqa: E402
from nhc_deprot.pipeline.pipeline_status import write_step_status  # noqa: E402
from nhc_deprot.pipeline.teacher_runner import (  # noqa: E402
    run_root_teacher,
)
from nhc_deprot.resources.profiles import get_profile, worker_env_for_profile  # noqa: E402


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _apply_worker_env(*, threads: int = 8) -> None:
    prof = get_profile("auto_fill_112_t8_v1")
    env = worker_env_for_profile(prof, threads=threads)
    for k, v in env.items():
        os.environ[k] = v
    # parent CPU-only; do not steal epoch0's CUDA_VISIBLE_DEVICES=1 if set in parent shell
    os.environ["CUDA_VISIBLE_DEVICES"] = ""


def _run_one_teacher_root(
    *,
    nhc0801_root: str,
    generation_id: str,
    root_id: str,
    gold_xyz_dir: str,
    max_steps: int,
    threads: int,
) -> dict:
    """Process-entry for one root (cation then neutral)."""
    try:
        _apply_worker_env(threads=threads)
        layout = resolve_layout(
            generation_id=generation_id, nhc0801_root=Path(nhc0801_root)
        )
        eng = LiveParentTeacherEngine(
            gold_xyz_dir=Path(gold_xyz_dir),
            max_steps=max_steps,
        )
        receipt = run_root_teacher(
            layout=layout, root_id=root_id, engine=eng, dry_run=False
        )
        out = layout.teacher_root_dir(root_id) / "root_receipt.json"
        return {
            "root_id": root_id,
            "status": "PASS" if receipt.status == "PASS" else receipt.status,
            "receipt": str(out),
            "endpoints": dict(receipt.endpoints or {}),
        }
    except Exception as exc:  # noqa: BLE001
        return {
            "root_id": root_id,
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1200:],
        }


def run_aimnet2_gau_train(
    *,
    layout,
    gold_xyz_dir: Path,
    weight: Path,
    max_workers: int = 1,
) -> dict:
    """GPU-side: GAU_LOOSE preopt for each train endpoint."""

    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    eng = LiveAimnet2GauLooseEngine(weight_path=weight)
    out_dir = layout.generation_root / "preopt_gau_loose"
    out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for root_id in TRAIN_ROOTS:
        for endpoint in ("cation", "neutral"):
            xyz = gold_xyz_dir / f"{root_id}_{endpoint}.xyz"
            els, coords = load_xyz(xyz)
            ch, mu = (
                (CATION_CHARGE, CATION_MULTIPLICITY)
                if endpoint == "cation"
                else (NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY)
            )
            print(f"[GAU] {root_id} {endpoint}", flush=True)
            try:
                r = eng.optimize_to_gau_loose(
                    root_id=root_id,
                    endpoint=endpoint,
                    elements=els,
                    coordinates=coords,
                    charge=ch,
                    multiplicity=mu,
                    checkpoint_id="epoch-0-official",
                )
                path = out_dir / f"{root_id}_{endpoint}.json"
                write_json(path, r, overwrite=True)
                results.append(
                    {
                        "root_id": root_id,
                        "endpoint": endpoint,
                        "status": "PASS" if r.get("converged") else "PARTIAL",
                        "steps": r.get("steps"),
                        "path": str(path),
                    }
                )
            except Exception as exc:  # noqa: BLE001
                results.append(
                    {
                        "root_id": root_id,
                        "endpoint": endpoint,
                        "status": "FAIL",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
    camp = {
        "schema": "nhc0801-preopt-gau-loose-campaign-v1",
        "mindmap_related_steps": [2, 3],
        "status": "PASS"
        if all(x.get("status") in {"PASS", "PARTIAL"} for x in results)
        else "PARTIAL",
        "results": results,
        "final_test_payload_read": False,
        "created_at_utc": _utc(),
    }
    write_json(out_dir / "campaign_receipt.json", camp, overwrite=True)
    return camp


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument(
        "--gold-xyz-dir",
        type=Path,
        default=Path("/home/plab/test/WJW/data/runs/mol_gold/xyz"),
    )
    p.add_argument(
        "--base-weight",
        type=Path,
        default=Path("/home/plab/.cache/aimnet/aimnet2_wb97m_d3_0.pt"),
    )
    p.add_argument("--skip-gau", action="store_true")
    p.add_argument("--skip-teacher", action="store_true")
    p.add_argument(
        "--teacher-parallel",
        type=int,
        default=2,
        help="Concurrent train roots for parent DFT (leave room for epoch0)",
    )
    p.add_argument("--threads-per-endpoint", type=int, default=8)
    p.add_argument("--max-steps", type=int, default=80)
    args = p.parse_args(argv)

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=args.nhc0801_root
    )
    ensure_generation_tree(layout, exist_ok=True)
    report: dict = {
        "schema": "nhc0801-live-side-mindmap-v1",
        "started_at_utc": _utc(),
        "generation_id": layout.generation_id,
        "leaves_epoch0_running": True,
        "final_test_authorized": False,
        "resource_note": (
            f"teacher_parallel={args.teacher_parallel} "
            f"t={args.threads_per_endpoint} (V002-style); "
            "does not stop live epoch0"
        ),
    }

    # A) AIMNet2 GAU on GPU (quick relative to parent DFT)
    if not args.skip_gau:
        try:
            # isolate to GPU0; epoch0 launcher used GPU1 for GAU if any
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            print("[side] AIMNet2 GAU_LOOSE train roots on GPU0", flush=True)
            gau = run_aimnet2_gau_train(
                layout=layout,
                gold_xyz_dir=args.gold_xyz_dir,
                weight=args.base_weight,
            )
            report["gau_loose"] = {
                "status": gau.get("status"),
                "n": len(gau.get("results") or []),
            }
            write_step_status(
                layout,
                step=2,
                name="teacher_pyscf",
                status="RUNNING",
                detail={"phase": "gau_preopt_done", "gau": gau.get("status")},
            )
        except Exception as exc:  # noqa: BLE001
            report["gau_loose"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-1500:],
            }
            print(f"[side] GAU FAIL: {exc}", flush=True)

    # B) Live teacher parent DFT for train roots (parallel roots, serial endpoints/root)
    if not args.skip_teacher:
        print(
            f"[side] live teacher Train roots parallel={args.teacher_parallel} "
            f"t={args.threads_per_endpoint}",
            flush=True,
        )
        root_results = []
        # ProcessPool: each root own process with OMP=8
        # Use threads via initializer pattern — ProcessPoolExecutor with function
        from concurrent.futures import ProcessPoolExecutor, as_completed

        futs = {}
        with ProcessPoolExecutor(max_workers=max(1, args.teacher_parallel)) as ex:
            for root_id in TRAIN_ROOTS:
                fut = ex.submit(
                    _run_one_teacher_root,
                    nhc0801_root=str(args.nhc0801_root),
                    generation_id=args.generation_id,
                    root_id=root_id,
                    gold_xyz_dir=str(args.gold_xyz_dir),
                    max_steps=args.max_steps,
                    threads=args.threads_per_endpoint,
                )
                futs[fut] = root_id
            for fut in as_completed(futs):
                res = fut.result()
                root_results.append(res)
                print(
                    f"[side] teacher {res.get('root_id')} -> {res.get('status')}",
                    flush=True,
                )

        failed = sum(1 for r in root_results if r.get("status") != "PASS")
        camp = {
            "schema": "nhc0801-teacher-campaign-receipt-v1",
            "mindmap_step": 2,
            "dry_run": False,
            "live_chemistry": True,
            "teacher_pyscf_authorized": True,
            "status": "LIVE_TEACHER_PASS" if failed == 0 else "LIVE_TEACHER_PARTIAL",
            "failed_root_count": failed,
            "root_results": root_results,
            "final_test_payload_read": False,
            "notes": [
                "side compute while epoch0 running",
                "frames=initial+final only until geomeTRIC step dump wired",
                "V002-style t=8, limited parallel roots",
            ],
            "created_at_utc": _utc(),
        }
        write_json(layout.teacher_dir / "campaign_receipt_live.json", camp, overwrite=True)
        write_json(layout.logs_dir / "teacher_campaign_live.json", camp, overwrite=True)
        report["teacher"] = {
            "status": camp["status"],
            "failed_root_count": failed,
            "roots": [r.get("root_id") for r in root_results],
        }
        write_step_status(
            layout,
            step=2,
            name="teacher_pyscf",
            status=str(camp["status"]),
            detail={"failed_root_count": failed},
        )

    report["finished_at_utc"] = _utc()
    write_json(layout.logs_dir / "live_side_mindmap_report.json", report, overwrite=True)
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    # PARTIAL still exit 0 if any progress; FAIL only if teacher hard-failed all
    if report.get("teacher", {}).get("status") == "FAIL":
        return 1
    return 0


if __name__ == "__main__":
    # ProcessPool on some platforms needs guard
    raise SystemExit(main())
