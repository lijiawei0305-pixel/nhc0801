"""Post-epoch0 audit: read campaign_receipt + per-root receipts (mindmap step 3).

Safe to call after live epoch-0 finishes. Does not start chemistry.
"""

from __future__ import annotations

from typing import Any, Final

from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout

AUDIT_SCHEMA: Final = "nhc0801-epoch0-receipt-audit-v1"


class Epoch0ReceiptAuditError(RuntimeError):
    """Epoch-0 receipt audit failed closed."""


def audit_epoch0_receipts(
    *,
    layout: GenerationLayout,
    expected_roots: tuple[str, ...] | None = None,
    write_report: bool = True,
) -> dict[str, Any]:
    """Load and summarize g001/epoch0 campaign + root receipts."""

    roots = expected_roots or VALIDATION_ROOTS
    campaign_path = layout.epoch0_dir / "campaign_receipt.json"
    if not campaign_path.is_file():
        raise Epoch0ReceiptAuditError(f"missing campaign_receipt: {campaign_path}")

    campaign, _camp_raw = load_json_object(campaign_path)
    root_reports: list[dict[str, Any]] = []
    missing: list[str] = []
    failed: list[str] = []

    for root_id in roots:
        rp = layout.epoch0_dir / root_id / "epoch0_root_receipt.json"
        if not rp.is_file():
            missing.append(root_id)
            root_reports.append(
                {
                    "root_id": root_id,
                    "present": False,
                    "path": str(rp),
                }
            )
            continue
        payload, _root_raw = load_json_object(rp)
        status = payload.get("status")
        if status not in {"PASS", "DRY_RUN_PASS"} and not str(status).endswith("PASS"):
            failed.append(root_id)
        comparison = payload.get("comparison") or {}
        root_reports.append(
            {
                "root_id": root_id,
                "present": True,
                "path": str(rp),
                "status": status,
                "absolute_label_error_kcal": comparison.get("absolute_label_error_kcal"),
                "epoch0_parent_opt_steps": comparison.get("epoch0_parent_opt_steps"),
                "pure_parent_opt_steps": comparison.get("pure_parent_opt_steps"),
                "epoch0_label_kcal": comparison.get("epoch0_label_kcal"),
                "pure_label_kcal": comparison.get("pure_label_kcal"),
                "sha256_bytes": rp.stat().st_size,
            }
        )

    camp_status = campaign.get("status")
    ok = (
        bool(
            str(camp_status).endswith("PASS")
            or camp_status in {"LIVE_EPOCH0_PASS", "DRY_RUN_EPOCH0_PASS"}
        )
        and not missing
        and not failed
        and int(campaign.get("failed_root_count") or 0) == 0
    )

    report = {
        "schema": AUDIT_SCHEMA,
        "generation_id": layout.generation_id,
        "campaign_path": str(campaign_path),
        "campaign_status": camp_status,
        "campaign_failed_root_count": campaign.get("failed_root_count"),
        "campaign_live_chemistry": campaign.get("live_chemistry"),
        "campaign_dry_run": campaign.get("dry_run"),
        "baseline_metrics": campaign.get("baseline_metrics"),
        "root_reports": root_reports,
        "missing_roots": missing,
        "failed_roots": failed,
        "audit_pass": ok,
        "status": "EPOCH0_RECEIPT_AUDIT_PASS" if ok else "EPOCH0_RECEIPT_AUDIT_FAIL",
        "notes": [
            "read-only audit of written receipts",
            "does not re-run DFT or open Final Test",
        ],
    }

    if write_report:
        layout.logs_dir.mkdir(parents=True, exist_ok=True)
        write_json(layout.logs_dir / "epoch0_receipt_audit.json", report, overwrite=True)
        write_json(layout.epoch0_dir / "receipt_audit.json", report, overwrite=True)
    return report


def format_audit_summary(report: dict[str, Any]) -> str:
    lines = [
        f"epoch0 audit: {report.get('status')}",
        f"  campaign_status={report.get('campaign_status')}",
        f"  failed_root_count={report.get('campaign_failed_root_count')}",
        f"  missing={report.get('missing_roots')}",
        f"  failed={report.get('failed_roots')}",
    ]
    metrics = report.get("baseline_metrics") or {}
    if metrics:
        lines.append(f"  mae={metrics.get('mean_absolute_label_error_kcal_mol')}")
        lines.append(f"  max_ae={metrics.get('maximum_absolute_label_error_kcal_mol')}")
    for rr in report.get("root_reports") or []:
        if not rr.get("present"):
            lines.append(f"  - {rr.get('root_id')}: MISSING")
            continue
        lines.append(
            f"  - {rr.get('root_id')}: status={rr.get('status')} "
            f"|ae|={rr.get('absolute_label_error_kcal')} "
            f"steps e0/pure={rr.get('epoch0_parent_opt_steps')}/{rr.get('pure_parent_opt_steps')}"
        )
    return "\n".join(lines)
