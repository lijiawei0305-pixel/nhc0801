#!/usr/bin/env python3
"""After live epoch-0 finishes: audit receipts then advance mindmap steps 7–10.

Does NOT open Final Test (steps 11–12).
Default sci-val is dry-run; pass --live-sci-val-weights-only for live route
on shortlist candidates that already have .pt weights.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.data.io_util import write_json  # noqa: E402
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout  # noqa: E402
from nhc_deprot.pipeline.checkpoint_shortlist import run_shortlist_campaign  # noqa: E402
from nhc_deprot.pipeline.epoch0_receipt_audit import (  # noqa: E402
    audit_epoch0_receipts,
    format_audit_summary,
)
from nhc_deprot.pipeline.freeze_package import build_freeze_package  # noqa: E402
from nhc_deprot.pipeline.pipeline_status import (  # noqa: E402
    write_pipeline_status,
    write_step_status,
)
from nhc_deprot.pipeline.sci_val_campaign import run_sci_val_campaign  # noqa: E402


def _utc() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--nhc0801-root", type=Path, default=Path("/home/plab/test/WJW/NHC0801"))
    p.add_argument("--generation-id", default="nhc0801-g001")
    p.add_argument("--repo-root", type=Path, default=None)
    p.add_argument(
        "--skip-sci-val",
        action="store_true",
        help="Only audit + shortlist + freeze + pipeline status",
    )
    args = p.parse_args(argv)

    layout = resolve_layout(generation_id=args.generation_id, nhc0801_root=args.nhc0801_root)
    ensure_generation_tree(layout, exist_ok=True)
    report: dict = {
        "schema": "nhc0801-post-epoch0-continue-v1",
        "started_at_utc": _utc(),
        "generation_id": layout.generation_id,
        "final_test_authorized": False,
        "steps": {},
    }

    # --- step 3 audit ---
    try:
        audit = audit_epoch0_receipts(layout=layout, write_report=True)
        report["steps"]["epoch0_audit"] = {
            "status": audit.get("status"),
            "audit_pass": audit.get("audit_pass"),
            "campaign_status": audit.get("campaign_status"),
            "missing_roots": audit.get("missing_roots"),
            "failed_roots": audit.get("failed_roots"),
            "baseline_metrics": audit.get("baseline_metrics"),
        }
        write_step_status(
            layout,
            step=3,
            name="epoch0",
            status=str(audit.get("campaign_status") or audit.get("status")),
            detail={"audit_pass": audit.get("audit_pass")},
        )
        print(format_audit_summary(audit), flush=True)
        if not audit.get("audit_pass"):
            # still continue shortlist/freeze with what we have, but mark fail closed for sci-val live
            report["epoch0_audit_failed"] = True
    except Exception as exc:  # noqa: BLE001
        report["steps"]["epoch0_audit"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc()[-1500:],
        }
        write_json(layout.logs_dir / "post_epoch0_continue.json", report, overwrite=True)
        print(json.dumps(report, indent=2), flush=True)
        return 2

    # --- step 7 shortlist ---
    try:
        short = run_shortlist_campaign(layout=layout, recompute=False)
        report["steps"]["shortlist"] = {
            "status": short.get("status"),
            "candidate_count": short.get("candidate_count"),
            "weights_present_count": short.get("weights_present_count"),
        }
        write_step_status(
            layout,
            step=7,
            name="shortlist",
            status=str(short.get("status")),
            detail={"candidate_count": short.get("candidate_count")},
        )
        print(
            f"[7] shortlist {short.get('status')} "
            f"candidates={short.get('candidate_count')} "
            f"weights={short.get('weights_present_count')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        report["steps"]["shortlist"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[7] shortlist FAIL: {exc}", flush=True)

    # --- steps 8–9 sci-val (dry-run: wires selection; live finetuned GAU needs separate auth/engine) ---
    if not args.skip_sci_val:
        try:
            print("[8-9] dry-run sci-val over shortlist (Final Test sealed)", flush=True)
            sci = run_sci_val_campaign(layout=layout, dry_run=True)
            report["steps"]["sci_val"] = {
                "status": sci.get("status"),
                "mode": "dry_run",
                "selection": sci.get("selection"),
                "candidate_count": sci.get("candidate_count"),
                "note": (
                    "dry_run after e0 audit; live sci-val for finetuned .pt "
                    "requires next authorization + SHA-aware AIMNet2 loader"
                ),
            }
            write_step_status(
                layout,
                step=8,
                name="sci_val",
                status=str(sci.get("status")),
                detail={"mode": "dry_run"},
            )
            sel = (sci.get("selection") or {}).get("outcome")
            write_step_status(
                layout,
                step=9,
                name="select",
                status=str(sel or "UNKNOWN"),
                detail=sci.get("selection") or {},
            )
            print(f"[8-9] sci_val {sci.get('status')} selection={sel}", flush=True)
        except Exception as exc:  # noqa: BLE001
            report["steps"]["sci_val"] = {
                "status": "FAIL",
                "error": f"{type(exc).__name__}: {exc}",
                "traceback": traceback.format_exc()[-2000:],
            }
            print(f"[8-9] sci_val FAIL: {exc}", flush=True)

    # --- step 10 freeze ---
    try:
        repo = args.repo_root or args.nhc0801_root
        freeze = build_freeze_package(layout=layout, repo_root=repo)
        report["steps"]["freeze"] = {
            "status": freeze.get("status"),
            "model_state": freeze.get("model_state"),
            "selection_outcome": (freeze.get("selection") or {}).get("outcome"),
            "final_test_ready": freeze.get("final_test_ready"),
        }
        write_step_status(
            layout,
            step=10,
            name="freeze",
            status=str(freeze.get("status")),
            detail={"model_state": freeze.get("model_state")},
        )
        print(
            f"[10] freeze {freeze.get('status')} model_state={freeze.get('model_state')} "
            f"FT_ready={freeze.get('final_test_ready')}",
            flush=True,
        )
    except Exception as exc:  # noqa: BLE001
        report["steps"]["freeze"] = {
            "status": "FAIL",
            "error": f"{type(exc).__name__}: {exc}",
        }
        print(f"[10] freeze FAIL: {exc}", flush=True)

    # --- pipeline status ---
    try:
        pipe = write_pipeline_status(layout, orchestrator_running=False)
        report["steps"]["pipeline_status"] = {
            "status": "WRITTEN",
            "problems": pipe.get("problems"),
        }
    except Exception as exc:  # noqa: BLE001
        report["steps"]["pipeline_status"] = {"status": "FAIL", "error": str(exc)}

    report["finished_at_utc"] = _utc()
    report["next_manual"] = [
        "Final Test (11–12) remains sealed — human confirm required",
        "If sci_val was dry_run only, authorize live sci-val for weight-present checkpoints",
    ]
    write_json(layout.logs_dir / "post_epoch0_continue.json", report, overwrite=True)
    write_json(
        layout.generation_root / "pipeline" / "post_epoch0_continue.json",
        report,
        overwrite=True,
    )
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)

    audit_ok = report.get("steps", {}).get("epoch0_audit", {}).get("audit_pass") is True
    return 0 if audit_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
