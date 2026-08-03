"""Mindmap pipeline status aggregator (write + scan).

Orchestrator may write RUNNING / step receipts under runs/<gen>/pipeline/.
TUI is read-only and consumes these files + other campaign receipts.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.generation.layout import GenerationLayout

PIPELINE_STATUS_SCHEMA: Final = "nhc0801-pipeline-status-v1"
STEP_DEFS: Final = (
    (0, "freeze_roots"),
    (1, "tvt_split"),
    (2, "teacher_pyscf"),
    (3, "epoch0"),
    (4, "train"),
    (5, "checkpoints"),
    (6, "quick_val"),
    (7, "shortlist"),
    (8, "sci_val"),
    (9, "select"),
    (10, "freeze"),
    (11, "final_test"),
    (12, "no_post_select"),
)


def _utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def pipeline_dir(layout: GenerationLayout) -> Path:
    return layout.generation_root / "pipeline"


def pipeline_status_path(layout: GenerationLayout) -> Path:
    return pipeline_dir(layout) / "pipeline_status.json"


def _safe_load(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        payload, _ = load_json_object(path)
        return payload
    except Exception:  # noqa: BLE001
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return raw if isinstance(raw, dict) else None
        except Exception:  # noqa: BLE001
            return None


def _status_from_payload(payload: dict[str, Any] | None, *, present: bool) -> str:
    if not present or payload is None:
        return "NOT_STARTED"
    st = payload.get("status")
    if st is None:
        return "UNKNOWN"
    s = str(st)
    if s in {"RUNNING", "IN_PROGRESS"}:
        return "RUNNING"
    if "REJECT" in s.upper():
        return "REJECTED"
    if "PARTIAL" in s.upper():
        return "PARTIAL"
    if "FAIL" in s.upper():
        return "FAIL"
    if s.endswith("PASS") or s in {
        "PASS",
        "SHORTLIST_PASS",
        "VALIDATION_SELECTED",
        "FROZEN",
        "PROVISIONAL",
        "LIVE_TRAIN_PASS",
        "LIVE_EPOCH0_PASS",
        "DRY_RUN_EPOCH0_PASS",
        "DRY_RUN_SCI_VAL_PASS",
        "EPOCH0_RECEIPT_AUDIT_PASS",
    }:
        return "PASS" if s != "PROVISIONAL" else "PROVISIONAL"
    if "SELECTED" in s.upper():
        return "PASS"
    return s


def scan_generation_status(layout: GenerationLayout) -> dict[str, Any]:
    """Read-only scan of generation artifacts → pipeline status snapshot."""

    steps: list[dict[str, Any]] = []
    problems: list[dict[str, Any]] = []

    # Prefer explicit step files if present
    pdir = pipeline_dir(layout)
    explicit: dict[int, dict[str, Any]] = {}
    if pdir.is_dir():
        for path in pdir.glob("step_*.json"):
            payload = _safe_load(path)
            if not payload:
                continue
            try:
                idx = int(payload.get("step", path.name.split("_")[1]))
            except (TypeError, ValueError):
                continue
            explicit[idx] = payload

    # Canonical product paths first; trailing entries are LEGACY_READONLY only
    # (symlink teacher->teacher_gpu_g001 may still exist on server).
    artifact_map: dict[int, list[Path]] = {
        2: [
            layout.teacher_batch_dir("g001") / "campaign_receipt.json",
            layout.teacher_batch_dir("g001") / "campaign_receipt_live.json",
            layout.teacher_dir / "campaign_receipt.json",  # == teacher_gpu_g001 via layout
            layout.teacher_dir / "campaign_receipt_live.json",
            layout.logs_dir / "teacher_campaign_receipt.json",
            layout.logs_dir / "teacher_campaign_live_g001.json",
            # LEGACY_READONLY:
            layout.generation_root / "teacher" / "campaign_receipt.json",
            layout.logs_dir / "teacher_campaign_live_02c.json",
        ],
        3: [
            layout.epoch0_batch_dir("g001") / "campaign_receipt.json",
            layout.epoch0_dir / "campaign_receipt.json",  # same as batch g001 when layout current
            layout.logs_dir / "epoch0_campaign_receipt.json",
            layout.generation_root / "epoch0_val_batches" / "g001_epoch0_val_receipt.json",
            # LEGACY_READONLY top-level epoch0/:
            layout.generation_root / "epoch0" / "campaign_receipt.json",
        ],
        4: [
            layout.train_campaign_receipt_path("g001"),
            layout.logs_dir / "train_g001_result.json",
            layout.train_batch_dir("g001") / "campaign_receipt.json",  # obsolete name
            layout.train_dir / "campaign_receipt_live.json",  # legacy pilot
            layout.train_dir / "campaign_receipt.json",
        ],
        5: [
            layout.train_campaign_receipt_path("g001"),
            layout.logs_dir / "train_g001_result.json",
            layout.train_batch_dir("g001") / "campaign_receipt.json",
            layout.train_dir / "campaign_receipt_live.json",
            layout.train_dir / "campaign_receipt.json",
        ],
        7: [
            layout.sci_val_dir / "shortlist_campaign.json",
            layout.logs_dir / "shortlist_campaign.json",
        ],
        8: [
            layout.sci_val_dir / "campaign_receipt.json",
            layout.logs_dir / "sci_val_campaign_receipt.json",
        ],
        9: [layout.sci_val_dir / "selection_receipt.json"],
        10: [layout.freeze_dir / "freeze_manifest.json", layout.logs_dir / "freeze_manifest.json"],
    }

    train_metrics: list[dict[str, Any]] = []
    train_scan = layout.resolve_train_batch_dir_for_read("g001")
    for seed_dir in sorted(train_scan.glob("seed_*")) if train_scan.is_dir() else []:
        rec = _safe_load(seed_dir / "seed_result.json") or _safe_load(
            seed_dir / "seed_receipt.json"
        )
        if not rec:
            continue
        logs = rec.get("epoch_logs") or []
        last = logs[-1] if logs else {}
        qv = (last.get("quick_validation") or {}) if isinstance(last, dict) else {}
        tr = (last.get("train") or {}) if isinstance(last, dict) else {}
        train_metrics.append(
            {
                "batch_id": rec.get("batch_id") or "g001",
                "seed": rec.get("seed"),
                "status": rec.get("status"),
                "epochs_run": rec.get("epochs_run"),
                "shortlist_epochs": rec.get("shortlist_epochs"),
                "last_epoch": last.get("epoch") if isinstance(last, dict) else None,
                "train_weighted_loss": tr.get("train_weighted_loss"),
                "validation_weighted_loss": qv.get("validation_weighted_loss"),
                "has_pt": any(seed_dir.glob("*.pt")),
            }
        )

    sci_metrics: dict[str, Any] = {}
    e0 = _safe_load(layout.epoch0_dir / "campaign_receipt.json")
    if not e0:
        # legacy top-level epoch0/ (pre-unified layout)
        e0 = _safe_load(layout.generation_root / "epoch0" / "campaign_receipt.json")
    if e0:
        sci_metrics["epoch0"] = {
            "status": e0.get("status"),
            "baseline_metrics": e0.get("baseline_metrics"),
            "failed_root_count": e0.get("failed_root_count"),
        }
    sci = _safe_load(layout.sci_val_dir / "campaign_receipt.json")
    if sci:
        sci_metrics["sci_val"] = {
            "status": sci.get("status"),
            "selection": sci.get("selection"),
            "candidate_count": sci.get("candidate_count"),
            "dry_run": sci.get("dry_run"),
        }
    short = _safe_load(layout.sci_val_dir / "shortlist_campaign.json")
    if short:
        sci_metrics["shortlist"] = {
            "status": short.get("status"),
            "candidate_count": short.get("candidate_count"),
            "weights_present_count": short.get("weights_present_count"),
        }

    for idx, name in STEP_DEFS:
        payload = explicit.get(idx)
        source = "pipeline_step_file" if payload else None
        if payload is None:
            for art in artifact_map.get(idx, []):
                payload = _safe_load(art)
                if payload:
                    source = str(art)
                    break
        # step 6 is embedded in train
        if idx == 6 and train_metrics:
            payload = {
                "status": "PASS"
                if all(m.get("status") == "PASS" for m in train_metrics)
                else "PARTIAL",
                "note": "quick_val embedded in train seed receipts",
            }
            source = "train/seed_*/seed_receipt.json"
        if idx == 11:
            payload = payload or {
                "status": "SEALED",
                "note": "Final Test requires human confirm; never auto",
            }
            source = source or "policy"
        if idx == 12:
            payload = payload or {
                "status": "POLICY",
                "note": "no post-Test selection",
            }
            source = source or "policy"
        if idx in {0, 1} and payload is None:
            meta = _safe_load(layout.generation_meta_path())
            if meta:
                payload = {"status": "PASS", "note": "generation meta present"}
                source = str(layout.generation_meta_path())

        present = payload is not None
        status = _status_from_payload(payload, present=present)
        if idx == 11 and status == "PASS" and payload and payload.get("status") == "SEALED":
            status = "SEALED"
        step = {
            "step": idx,
            "name": name,
            "status": status,
            "source": source,
            "detail": {
                k: payload.get(k)
                for k in ("status", "error", "failed_root_count", "failed_seed_count", "outcome")
                if payload and k in payload
            }
            if payload
            else {},
        }
        steps.append(step)
        if status in {"FAIL", "REJECTED", "PARTIAL", "FAILED"}:
            problems.append(
                {
                    "step": idx,
                    "name": name,
                    "status": status,
                    "detail": step["detail"],
                    "source": source,
                }
            )

    existing = _safe_load(pipeline_status_path(layout))
    orchestrator_running = bool(existing and existing.get("orchestrator_running"))

    return {
        "schema": PIPELINE_STATUS_SCHEMA,
        "generation_id": layout.generation_id,
        "updated_at_utc": _utc_now(),
        "orchestrator_running": orchestrator_running,
        "final_test_sealed": True,
        "final_test_auto_start": False,
        "steps": steps,
        "problems": problems,
        "train_metrics": train_metrics,
        "scientific_metrics": sci_metrics,
        "notes": [
            "TUI/readers are read-only; orchestrator may write this file",
            "Final Test never auto-starts",
        ],
    }


def write_pipeline_status(
    layout: GenerationLayout,
    *,
    orchestrator_running: bool | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Scan + write pipeline_status.json (orchestrator path)."""

    snap = scan_generation_status(layout)
    if orchestrator_running is not None:
        snap["orchestrator_running"] = bool(orchestrator_running)
    if extra:
        snap["extra"] = dict(extra)
    pipeline_dir(layout).mkdir(parents=True, exist_ok=True)
    write_json(pipeline_status_path(layout), snap, overwrite=True)
    write_json(layout.logs_dir / "pipeline_status.json", snap, overwrite=True)
    snap["receipt_path"] = str(pipeline_status_path(layout))
    return snap


def write_step_status(
    layout: GenerationLayout,
    *,
    step: int,
    name: str,
    status: str,
    detail: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one step file then refresh aggregate status."""

    payload = {
        "schema": "nhc0801-pipeline-step-v1",
        "step": step,
        "name": name,
        "status": status,
        "updated_at_utc": _utc_now(),
        "detail": dict(detail or {}),
        "final_test_auto_start": False,
    }
    pipeline_dir(layout).mkdir(parents=True, exist_ok=True)
    path = pipeline_dir(layout) / f"step_{step:02d}_{name}.json"
    write_json(path, payload, overwrite=True)
    write_pipeline_status(layout)
    return payload
