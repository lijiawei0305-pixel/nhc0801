"""Rebuild epoch0 campaign from root receipts (no DFT)."""

from __future__ import annotations

from pathlib import Path

from nhc_deprot.data.io_util import write_json
from nhc_deprot.data.paths import VALIDATION_ROOTS
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout
from nhc_deprot.pipeline.epoch0_campaign_rebuild import (
    Epoch0CampaignRebuildError,
    rebuild_epoch0_campaign_from_root_receipts,
)
from nhc_deprot.pipeline.epoch0_receipt_audit import audit_epoch0_receipts


def _minimal_endpoint(root_id: str, endpoint: str, *, energy: float, steps: int) -> dict:
    return {
        "root_id": root_id,
        "endpoint": endpoint,
        "route_kind": "epoch_zero",
        "checkpoint_id": "epoch-0-official",
        "stages_completed": ["parent_full_gau_optimization"],
        "aimnet2_converged": True,
        "aimnet2_steps": 10,
        "handoff_classification": "OK",
        "continue_parent_optimization": True,
        "parent_geometry_converged": True,
        "parent_final_sp_converged": True,
        "parent_final_state": "FINAL_PARENT_GAU_CONVERGED",
        "parent_energy_hartree": energy,
        "parent_opt_steps": steps,
        "parent_scf_cycles": 8,
        "wall_seconds": 1.0,
        "identity_and_structure_ok": True,
        "catastrophic": False,
        "catastrophic_reasons": [],
        "aimnet2_energy_used_in_label": False,
        "single_point_only": False,
        "notes": [],
    }


def _root_receipt(root_id: str, *, e_c: float, e_n: float) -> dict:
    pure_c = _minimal_endpoint(root_id, "cation", energy=e_c, steps=100)
    pure_c["route_kind"] = "pure_pyscf_reference"
    pure_c["checkpoint_id"] = "pure-pyscf-reference"
    pure_n = _minimal_endpoint(root_id, "neutral", energy=e_n, steps=100)
    pure_n["route_kind"] = "pure_pyscf_reference"
    pure_n["checkpoint_id"] = "pure-pyscf-reference"
    e0_c = _minimal_endpoint(root_id, "cation", energy=e_c + 1e-4, steps=100)
    e0_n = _minimal_endpoint(root_id, "neutral", energy=e_n + 1e-4, steps=100)
    pure_label = 100.0
    e0_label = 100.5
    return {
        "schema": "nhc0801-epoch0-root-receipt-v1",
        "mindmap_step": 3,
        "root_id": root_id,
        "dry_run": False,
        "live_chemistry": True,
        "official_weight_sha256": "a" * 64,
        "checkpoint_id": "epoch-0-official",
        "parent_protocol_sha256": "b" * 64,
        "single_point_only": False,
        "aimnet2_energy_enters_label": False,
        "pure_pyscf_reference": {
            "root_id": root_id,
            "route_kind": "pure_pyscf_reference",
            "checkpoint_id": "pure-pyscf-reference",
            "cation": pure_c,
            "neutral": pure_n,
            "label_kcal": pure_label,
            "reference_label_kcal": None,
            "absolute_label_error_kcal": None,
            "signed_label_error_kcal": None,
            "all_identity_and_structure_hard_gates": True,
            "catastrophic_failure": False,
        },
        "epoch0_route": {
            "root_id": root_id,
            "route_kind": "epoch_zero",
            "checkpoint_id": "epoch-0-official",
            "cation": e0_c,
            "neutral": e0_n,
            "label_kcal": e0_label,
            "reference_label_kcal": pure_label,
            "absolute_label_error_kcal": 0.5,
            "signed_label_error_kcal": 0.5,
            "all_identity_and_structure_hard_gates": True,
            "catastrophic_failure": False,
        },
        "comparison": {
            "pure_label_kcal": pure_label,
            "epoch0_label_kcal": e0_label,
            "absolute_label_error_kcal": 0.5,
            "signed_label_error_kcal": 0.5,
            "pure_parent_opt_steps": 200,
            "epoch0_parent_opt_steps": 200,
            "parent_opt_step_reduction_fraction": 0.0,
        },
        "status": "PASS",
        "merged_from_endpoints": True,
    }


def test_rebuild_campaign_pass_and_audit(tmp_path: Path) -> None:
    layout = resolve_layout(generation_id="nhc0801-g001", nhc0801_root=tmp_path)
    ensure_generation_tree(layout, exist_ok=True)
    roots = list(VALIDATION_ROOTS)
    e0 = layout.epoch0_batch_dir("g001")
    for rid in roots:
        d = e0 / rid
        d.mkdir(parents=True, exist_ok=True)
        write_json(
            d / "epoch0_root_receipt.json",
            _root_receipt(rid, e_c=-100.0, e_n=-99.5),
            overwrite=True,
        )

    # stale partial campaign
    write_json(
        e0 / "campaign_receipt.json",
        {"status": "LIVE_EPOCH0_PARTIAL", "failed_root_count": 2},
        overwrite=True,
    )

    out = rebuild_epoch0_campaign_from_root_receipts(
        layout=layout, batch_id="g001", validation_roots=roots
    )
    assert out["status"] == "LIVE_EPOCH0_PASS"
    assert out["failed_root_count"] == 0
    assert Path(out["campaign_path"]).is_file()
    assert Path(out["batch_receipt_path"]).is_file()

    audit = audit_epoch0_receipts(layout=layout, expected_roots=tuple(roots))
    assert audit["audit_pass"] is True
    assert audit["campaign_status"] == "LIVE_EPOCH0_PASS"


def test_rebuild_refuses_failed_root(tmp_path: Path) -> None:
    layout = resolve_layout(generation_id="nhc0801-g001", nhc0801_root=tmp_path)
    ensure_generation_tree(layout, exist_ok=True)
    roots = list(VALIDATION_ROOTS)
    e0 = layout.epoch0_batch_dir("g001")
    for i, rid in enumerate(roots):
        d = e0 / rid
        d.mkdir(parents=True, exist_ok=True)
        rec = _root_receipt(rid, e_c=-100.0, e_n=-99.5)
        if i == 0:
            rec["status"] = "FAILED"
        write_json(d / "epoch0_root_receipt.json", rec, overwrite=True)

    try:
        rebuild_epoch0_campaign_from_root_receipts(
            layout=layout, batch_id="g001", validation_roots=roots
        )
        raise AssertionError("expected fail")
    except Epoch0CampaignRebuildError as exc:
        assert "not all PASS" in str(exc)
