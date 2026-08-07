"""Rebuild g00N Epoch-0 campaign/batch receipts from existing root receipts.

No DFT. Used after sharded e0_val_only finishes both roots so
``campaign_receipt.json`` matches per-root PASS status.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256, TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.epoch0_runner import (
    CHECKPOINT_ID,
    EPOCH0_CAMPAIGN_SCHEMA,
    MINDMAP_STEP,
)
from nhc_deprot.pipeline.scientific_validation import (
    EndpointRouteReceipt,
    RootRouteReceipt,
    aggregate_checkpoint_validation,
)

REBUILD_SCHEMA: Final = "nhc0801-epoch0-campaign-rebuild-v1"


class Epoch0CampaignRebuildError(RuntimeError):
    """Campaign rebuild failed closed."""


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def endpoint_receipt_from_dict(raw: dict[str, Any]) -> EndpointRouteReceipt:
    return EndpointRouteReceipt(
        root_id=str(raw["root_id"]),
        endpoint=str(raw["endpoint"]),
        route_kind=str(raw["route_kind"]),
        checkpoint_id=str(raw["checkpoint_id"]),
        stages_completed=list(raw.get("stages_completed") or []),
        aimnet2_converged=bool(raw.get("aimnet2_converged", False)),
        aimnet2_steps=int(raw.get("aimnet2_steps") or 0),
        handoff_classification=raw.get("handoff_classification"),
        continue_parent_optimization=bool(raw.get("continue_parent_optimization", False)),
        parent_geometry_converged=bool(raw.get("parent_geometry_converged", False)),
        parent_final_sp_converged=bool(raw.get("parent_final_sp_converged", False)),
        parent_final_state=raw.get("parent_final_state"),
        parent_energy_hartree=(
            float(raw["parent_energy_hartree"])
            if raw.get("parent_energy_hartree") is not None
            else None
        ),
        parent_opt_steps=int(raw.get("parent_opt_steps") or 0),
        # Absent key ⇒ unmeasured (maxcap); do not invent burden from config constants.
        parent_opt_steps_is_maxcap=bool(raw.get("parent_opt_steps_is_maxcap", True)),
        parent_scf_cycles=int(raw.get("parent_scf_cycles") or 0),
        wall_seconds=float(raw.get("wall_seconds") or 0.0),
        identity_and_structure_ok=bool(raw.get("identity_and_structure_ok", False)),
        catastrophic=bool(raw.get("catastrophic", False)),
        catastrophic_reasons=list(raw.get("catastrophic_reasons") or []),
        aimnet2_energy_used_in_label=bool(raw.get("aimnet2_energy_used_in_label", False)),
        single_point_only=bool(raw.get("single_point_only", False)),
        notes=list(raw.get("notes") or []),
    )


def root_route_from_dict(raw: dict[str, Any]) -> RootRouteReceipt:
    cat = raw.get("cation")
    neu = raw.get("neutral")
    return RootRouteReceipt(
        root_id=str(raw["root_id"]),
        route_kind=str(raw.get("route_kind") or "epoch_zero"),
        checkpoint_id=str(raw.get("checkpoint_id") or CHECKPOINT_ID),
        cation=endpoint_receipt_from_dict(cat) if isinstance(cat, dict) else None,
        neutral=endpoint_receipt_from_dict(neu) if isinstance(neu, dict) else None,
        label_kcal=(
            float(raw["label_kcal"]) if raw.get("label_kcal") is not None else None
        ),
        reference_label_kcal=(
            float(raw["reference_label_kcal"])
            if raw.get("reference_label_kcal") is not None
            else None
        ),
        absolute_label_error_kcal=(
            float(raw["absolute_label_error_kcal"])
            if raw.get("absolute_label_error_kcal") is not None
            else None
        ),
        signed_label_error_kcal=(
            float(raw["signed_label_error_kcal"])
            if raw.get("signed_label_error_kcal") is not None
            else None
        ),
        all_identity_and_structure_hard_gates=bool(
            raw.get("all_identity_and_structure_hard_gates", False)
        ),
        catastrophic_failure=bool(raw.get("catastrophic_failure", False)),
    )


def resolve_epoch0_root_receipt_path(
    epoch0_dir: Path, root_id: str
) -> Path | None:
    """Prefer ``root.json``, fall back to legacy ``epoch0_root_receipt.json``.

    Shared by campaign rebuild and Val e0 status readers — do not reimplement.
    """

    preferred = epoch0_dir / root_id / "root.json"
    if preferred.is_file():
        return preferred
    legacy = epoch0_dir / root_id / "epoch0_root_receipt.json"
    if legacy.is_file():
        return legacy
    return None


def load_root_receipts(
    epoch0_dir: Path,
    roots: Sequence[str],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    missing: list[str] = []
    for root_id in roots:
        if root_id in TRAIN_ROOTS:
            raise Epoch0CampaignRebuildError(
                f"refused train root in e0 rebuild: {root_id}"
            )
        path = resolve_epoch0_root_receipt_path(epoch0_dir, root_id)
        if path is None:
            missing.append(root_id)
            continue
        payload, _ = load_json_object(path)
        if not isinstance(payload, dict):
            raise Epoch0CampaignRebuildError(f"invalid root receipt: {path}")
        payload = dict(payload)
        payload["_receipt_path"] = str(path)
        out.append(payload)
    if missing:
        raise Epoch0CampaignRebuildError(f"missing root receipts: {missing}")
    return out


def rebuild_epoch0_campaign_from_root_receipts(
    *,
    layout: GenerationLayout,
    batch_id: str = "g001",
    validation_roots: Sequence[str] | None = None,
    require_all_pass: bool = True,
) -> dict[str, Any]:
    """Write campaign_receipt + batch val receipt from on-disk root receipts only."""

    roots = list(validation_roots or VALIDATION_ROOTS)
    epoch0_dir = layout.epoch0_batch_dir(batch_id)
    if not epoch0_dir.is_dir():
        raise Epoch0CampaignRebuildError(f"missing epoch0 dir: {epoch0_dir}")

    root_receipts = load_root_receipts(epoch0_dir, roots)
    failed = [r["root_id"] for r in root_receipts if r.get("status") != "PASS"]
    if require_all_pass and failed:
        raise Epoch0CampaignRebuildError(
            f"root receipts not all PASS (refusing rebuild): {failed}"
        )

    pure_roots: list[RootRouteReceipt] = []
    e0_roots: list[RootRouteReceipt] = []
    clean_roots: list[dict[str, Any]] = []
    for r in root_receipts:
        pure_raw = r.get("pure_pyscf_reference")
        e0_raw = r.get("epoch0_route")
        if not isinstance(pure_raw, dict) or not isinstance(e0_raw, dict):
            raise Epoch0CampaignRebuildError(
                f"root {r.get('root_id')} missing pure/epoch0 route dicts"
            )
        pure_roots.append(root_route_from_dict(pure_raw))
        e0_roots.append(root_route_from_dict(e0_raw))
        clean = {k: v for k, v in r.items() if k != "_receipt_path"}
        clean_roots.append(clean)

    pure_agg = aggregate_checkpoint_validation(
        epoch=0,
        checkpoint_id="pure-pyscf-reference",
        checkpoint_sha256="0" * 64,
        route_kind="pure_pyscf_reference",
        root_receipts=pure_roots,
        live_chemistry_executed=True,
    )
    e0_agg = aggregate_checkpoint_validation(
        epoch=0,
        checkpoint_id=str(
            root_receipts[0].get("checkpoint_id") or CHECKPOINT_ID
        ),
        checkpoint_sha256=str(
            root_receipts[0].get("official_weight_sha256")
            or OFFICIAL_AIMNET2_WEIGHT_SHA256
        ),
        route_kind="epoch_zero",
        root_receipts=e0_roots,
        live_chemistry_executed=True,
    )

    failed_n = len(failed)
    campaign: dict[str, Any] = {
        "schema": EPOCH0_CAMPAIGN_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "batch_id": batch_id,
        "dry_run": False,
        "live_chemistry": True,
        "epoch0_execution": True,
        "official_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
        "checkpoint_id": CHECKPOINT_ID,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "validation_roots": list(roots),
        "root_count": len(roots),
        "failed_root_count": failed_n,
        "status": "LIVE_EPOCH0_PASS" if failed_n == 0 else "LIVE_EPOCH0_PARTIAL",
        "pure_aggregate": pure_agg.as_dict(),
        "epoch0_aggregate": e0_agg.as_dict(),
        "root_receipts": clean_roots,
        "baseline_metrics": {
            "mean_absolute_label_error_kcal_mol": e0_agg.mean_absolute_label_error_kcal_mol,
            "maximum_absolute_label_error_kcal_mol": (
                e0_agg.maximum_absolute_label_error_kcal_mol
            ),
            "catastrophic_failure_count": e0_agg.catastrophic_failure_count,
            "pyscf_geometry_work_reduction_fraction": (
                e0_agg.pyscf_geometry_work_reduction_fraction
            ),
        },
        "notes": [
            "rebuilt from per-root epoch0_root_receipt.json (no DFT)",
            "after e0_val_only endpoint shards + root merge",
            "AIMNet2 energy never enters labels",
            "Final Test not accessed",
        ],
        "rebuilt_from_shards": True,
        "rebuilt_at_utc": _utc(),
        "rebuild_schema": REBUILD_SCHEMA,
        "final_test_payload_read": False,
        "training_started": False,
    }

    # primary paths under batch epoch0/
    write_json(epoch0_dir / "campaign_receipt.json", campaign, overwrite=True)
    logs_dir = layout.epoch0_batch_root(batch_id) / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(logs_dir / "epoch0_campaign_receipt.json", campaign, overwrite=True)
    # generation logs mirror
    layout.logs_dir.mkdir(parents=True, exist_ok=True)
    write_json(
        layout.logs_dir / f"epoch0_{batch_id}_campaign_rebuilt.json",
        campaign,
        overwrite=True,
    )

    batch_receipt = {
        "schema": "nhc0801-epoch0-val-only-receipt-v1",
        "mindmap_step": 3,
        "batch_id": batch_id,
        "standard_name": f"{batch_id} Epoch-0",
        "val_roots_only": True,
        "train_roots_forbidden": True,
        "val_roots": list(roots),
        "refused_train_roots": list(TRAIN_ROOTS),
        "status": campaign["status"],
        "failed_root_count": failed_n,
        "live_chemistry": True,
        "dry_run": False,
        "disk_layout": "epoch0_val_batches/<batch_id>/",
        "epoch0_dir": str(epoch0_dir),
        "batch_root": str(layout.epoch0_batch_root(batch_id)),
        "created_at_utc": _utc(),
        "rebuilt_from_shards": True,
        "campaign": campaign,
    }
    side = layout.generation_root / "epoch0_val_batches"
    side.mkdir(parents=True, exist_ok=True)
    write_json(side / f"{batch_id}_epoch0_val_receipt.json", batch_receipt, overwrite=True)
    write_json(layout.logs_dir / f"epoch0_{batch_id}.json", batch_receipt, overwrite=True)

    return {
        "status": campaign["status"],
        "failed_root_count": failed_n,
        "campaign_path": str(epoch0_dir / "campaign_receipt.json"),
        "batch_receipt_path": str(side / f"{batch_id}_epoch0_val_receipt.json"),
        "baseline_metrics": campaign["baseline_metrics"],
        "validation_roots": list(roots),
    }


def pure_references_from_root_receipts(
    root_receipts: Sequence[dict[str, Any]],
) -> dict[str, Any]:
    """Map root_id → PureReferenceLabel fields (for sci-val)."""
    from nhc_deprot.pipeline.scientific_validation import PureReferenceLabel

    out: dict[str, PureReferenceLabel] = {}
    for r in root_receipts:
        pure = r.get("pure_pyscf_reference") or {}
        cat = pure.get("cation") or {}
        neu = pure.get("neutral") or {}
        e_c = cat.get("parent_energy_hartree")
        e_n = neu.get("parent_energy_hartree")
        if e_c is None or e_n is None:
            raise Epoch0CampaignRebuildError(
                f"root {r.get('root_id')} pure energies missing"
            )
        rid = str(r["root_id"])
        label = pure.get("label_kcal")
        out[rid] = PureReferenceLabel(
            root_id=rid,
            e_cation_hartree=float(e_c),
            e_neutral_hartree=float(e_n),
            label_kcal=float(label) if label is not None else float("nan"),
            protocol_sha256=PROTOCOL_SHA256,
        )
    return out


def epoch0_baseline_from_root_receipts(
    root_receipts: Sequence[dict[str, Any]],
) -> Any:
    """Build CheckpointScientificValidation for epoch-0 from root receipts."""
    e0_roots = [
        root_route_from_dict(r["epoch0_route"])
        for r in root_receipts
    ]
    sha = str(
        root_receipts[0].get("official_weight_sha256") or OFFICIAL_AIMNET2_WEIGHT_SHA256
    )
    return aggregate_checkpoint_validation(
        epoch=0,
        checkpoint_id=str(root_receipts[0].get("checkpoint_id") or CHECKPOINT_ID),
        checkpoint_sha256=sha if len(sha) == 64 else OFFICIAL_AIMNET2_WEIGHT_SHA256,
        route_kind="epoch_zero",
        root_receipts=e0_roots,
        live_chemistry_executed=True,
    )
