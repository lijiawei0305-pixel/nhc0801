"""P1 instrumentation fixes for sci-val (20260804 plan).

- burden unmeasured fails closed (maxcap / None)
- non-regression uses NUMERIC_CALIBRATION signed_bias_tolerance (1.5)
- baseline parent_max_steps mismatch rejected
- wall_seconds measured on sim/live paths where injectable
"""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256, deprotonation_electronic_kcal
from nhc_deprot.contracts.tvt_gates import select_scientific_checkpoint
from nhc_deprot.generation.layout import ensure_generation_tree, resolve_layout
from nhc_deprot.pipeline.sci_val_campaign import SciValCampaignError, run_sci_val_campaign
from nhc_deprot.pipeline.scientific_validation import (
    EndpointRouteReceipt,
    FrozenEndpointGeometry,
    PureReferenceLabel,
    RootRouteReceipt,
    SimulatedAimnet2Engine,
    SimulatedParentEngine,
    aggregate_checkpoint_validation,
    run_scientific_validation_for_checkpoint,
    select_after_scientific_validation,
)
from nhc_deprot.pipeline.training_blockers import load_numeric_calibration


def _geom(root: str, endpoint: str, n_atoms: int = 3) -> FrozenEndpointGeometry:
    elements = tuple(["C"] * n_atoms)
    coords = tuple((float(i), 0.0, 0.0) for i in range(n_atoms))
    charge = 1 if endpoint == "cation" else 0
    return FrozenEndpointGeometry(
        root_id=root,
        endpoint=endpoint,
        elements=elements,
        coordinates=coords,
        charge=charge,
        multiplicity=1,
        geometry_sha256="a" * 64,
    )


def _ref(root: str, e_c: float = -100.0, e_n: float = -99.5) -> PureReferenceLabel:
    return PureReferenceLabel(
        root_id=root,
        e_cation_hartree=e_c,
        e_neutral_hartree=e_n,
        label_kcal=deprotonation_electronic_kcal(e_n, e_c),
        protocol_sha256=PROTOCOL_SHA256,
    )


def _endpoint(
    *,
    root: str = "R",
    endpoint: str = "cation",
    opt_steps: int = 20,
    is_maxcap: bool = False,
    scf: int = 10,
    wall: float = 1.0,
    energy: float = -100.0,
) -> EndpointRouteReceipt:
    return EndpointRouteReceipt(
        root_id=root,
        endpoint=endpoint,
        route_kind="finetuned_checkpoint",
        checkpoint_id="ck",
        parent_geometry_converged=True,
        parent_final_sp_converged=True,
        parent_energy_hartree=energy,
        parent_opt_steps=opt_steps,
        parent_opt_steps_is_maxcap=is_maxcap,
        parent_scf_cycles=scf,
        wall_seconds=wall,
        identity_and_structure_ok=True,
    )


def _root(
    *,
    root: str = "R",
    mae_signed: float = 0.0,
    is_maxcap: bool = False,
    opt_steps: int = 20,
    e_c: float = -100.0,
    e_n: float = -99.5,
) -> RootRouteReceipt:
    cat = _endpoint(
        root=root, endpoint="cation", opt_steps=opt_steps, is_maxcap=is_maxcap, energy=e_c
    )
    neu = _endpoint(
        root=root, endpoint="neutral", opt_steps=opt_steps, is_maxcap=is_maxcap, energy=e_n
    )
    label = deprotonation_electronic_kcal(e_n, e_c)
    ref = label  # perfect labels
    return RootRouteReceipt(
        root_id=root,
        route_kind="finetuned_checkpoint",
        checkpoint_id="ck",
        cation=cat,
        neutral=neu,
        label_kcal=label + mae_signed,
        reference_label_kcal=ref,
        absolute_label_error_kcal=abs(mae_signed),
        signed_label_error_kcal=mae_signed,
        all_identity_and_structure_hard_gates=True,
        catastrophic_failure=False,
    )


def _addendum() -> dict[str, object]:
    return load_numeric_calibration()


def test_burden_unmeasured_fails_closed() -> None:
    """Any endpoint with parent_opt_steps_is_maxcap → burden None + reject code."""

    root = _root(is_maxcap=True, opt_steps=250)
    agg = aggregate_checkpoint_validation(
        epoch=10,
        checkpoint_id="ck",
        checkpoint_sha256="b" * 64,
        route_kind="finetuned_checkpoint",
        root_receipts=[root],
        epoch0_mae=1.0,
        epoch0_mean_parent_steps=100.0,
        signed_bias_tolerance_kcal_mol=1.5,
    )
    assert agg.parent_opt_steps_unmeasured is True
    assert agg.pyscf_geometry_work_reduction_fraction is None

    payload = agg.selection_payload()
    # Make selection hard gates pass so burden gate is the only reject
    payload["all_identity_and_structure_hard_gates"] = True
    payload["catastrophic_failure_count"] = 0
    payload["maximum_absolute_label_error_kcal_mol"] = 0.1
    payload["critical_endpoint_non_regression_vs_epoch_zero"] = True
    payload["parent_gradient_reduction_fraction"] = 0.0
    payload["cumulative_scf_cycle_reduction_fraction"] = 0.0
    payload["end_to_end_wall_reduction_fraction"] = 0.0

    sel = select_scientific_checkpoint([payload], numeric_addendum=_addendum())
    assert sel["outcome"] == "VALIDATION_REJECTED"
    reasons = sel["rejected"][0]["reason_codes"]
    assert "BURDEN_METRIC_UNMEASURED" in reasons
    assert "PYSCF_BURDEN_REDUCTION_FAILED" not in reasons


def test_non_regression_uses_contract_tolerance() -> None:
    """mae = e0 + 1.4 PASS; +1.6 EPOCH_ZERO_REGRESSION (tol=1.5 from addendum)."""

    tol = float(_addendum()["signed_bias_tolerance_kcal_mol"])
    assert tol == pytest.approx(1.5)

    e0_mae = 1.0
    pass_root = _root(mae_signed=e0_mae + 1.4)
    fail_root = _root(mae_signed=e0_mae + 1.6)

    pass_agg = aggregate_checkpoint_validation(
        epoch=10,
        checkpoint_id="pass",
        checkpoint_sha256="c" * 64,
        route_kind="finetuned_checkpoint",
        root_receipts=[pass_root],
        epoch0_mae=e0_mae,
        epoch0_mean_parent_steps=40.0,
        signed_bias_tolerance_kcal_mol=tol,
    )
    fail_agg = aggregate_checkpoint_validation(
        epoch=11,
        checkpoint_id="fail",
        checkpoint_sha256="d" * 64,
        route_kind="finetuned_checkpoint",
        root_receipts=[fail_root],
        epoch0_mae=e0_mae,
        epoch0_mean_parent_steps=40.0,
        signed_bias_tolerance_kcal_mol=tol,
    )
    assert pass_agg.critical_endpoint_non_regression_vs_epoch_zero is True
    assert fail_agg.critical_endpoint_non_regression_vs_epoch_zero is False

    # Measured burden for both so selection reaches non-regression gate
    for agg in (pass_agg, fail_agg):
        assert agg.pyscf_geometry_work_reduction_fraction is not None

    sel_pass = select_after_scientific_validation([pass_agg])
    assert sel_pass["outcome"] == "VALIDATION_SELECTED"

    sel_fail = select_after_scientific_validation([fail_agg])
    assert sel_fail["outcome"] == "VALIDATION_REJECTED"
    assert "EPOCH_ZERO_REGRESSION" in sel_fail["rejected"][0]["reason_codes"]


def test_baseline_config_mismatch_rejected(tmp_path: Path) -> None:
    layout = resolve_layout(generation_id="nhc0801-g001", nhc0801_root=tmp_path)
    ensure_generation_tree(layout, exist_ok=True)
    with pytest.raises(SciValCampaignError, match="BASELINE_CONFIG_MISMATCH"):
        run_sci_val_campaign(
            layout=layout,
            dry_run=True,
            candidates=[{"seed": 1, "epoch": 10}],
            parent_max_steps=250,
            epoch0_parent_max_steps=100,
        )


def test_wall_seconds_is_measured() -> None:
    """Simulated parent returns positive wall; route aggregates it."""

    root = "VALROOT"
    result = run_scientific_validation_for_checkpoint(
        epoch=5,
        checkpoint_id="ckpt",
        checkpoint_sha256="e" * 64,
        route_kind="finetuned_checkpoint",
        geometries=[_geom(root, "cation"), _geom(root, "neutral")],
        references={root: _ref(root)},
        aimnet2=SimulatedAimnet2Engine(),
        parent=SimulatedParentEngine(),
        live=False,
    )
    # SimulatedParentEngine hardcodes wall_seconds=2.0 per endpoint
    assert result.root_receipts[0].cation is not None
    assert result.root_receipts[0].cation.wall_seconds > 0
    assert result.root_receipts[0].cation.parent_opt_steps_is_maxcap is False
    assert result.pyscf_geometry_work_reduction_fraction is not None


def test_measured_burden_hard_gate_still_applies() -> None:
    """Conservative hard gate: measured burden < 0.0 rejects (limit is 0.0)."""

    root = _root(is_maxcap=False, opt_steps=250)
    agg = aggregate_checkpoint_validation(
        epoch=12,
        checkpoint_id="ck",
        checkpoint_sha256="f" * 64,
        route_kind="finetuned_checkpoint",
        root_receipts=[root],
        epoch0_mae=1.0,
        epoch0_mean_parent_steps=100.0,  # reduction = (100-250)/100 = -1.5
        signed_bias_tolerance_kcal_mol=1.5,
    )
    assert agg.pyscf_geometry_work_reduction_fraction == pytest.approx(-1.5)
    payload = agg.selection_payload()
    payload["all_identity_and_structure_hard_gates"] = True
    payload["catastrophic_failure_count"] = 0
    payload["maximum_absolute_label_error_kcal_mol"] = 0.1
    payload["critical_endpoint_non_regression_vs_epoch_zero"] = True
    payload["parent_gradient_reduction_fraction"] = 0.0
    payload["cumulative_scf_cycle_reduction_fraction"] = 0.0
    payload["end_to_end_wall_reduction_fraction"] = 0.0
    sel = select_scientific_checkpoint([payload], numeric_addendum=_addendum())
    assert sel["outcome"] == "VALIDATION_REJECTED"
    assert "PYSCF_BURDEN_REDUCTION_FAILED" in sel["rejected"][0]["reason_codes"]


