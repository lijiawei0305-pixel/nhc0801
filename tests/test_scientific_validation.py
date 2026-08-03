"""Scientific Validation writer tests (simulated engines — no live chemistry)."""

from __future__ import annotations

import pytest

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256, deprotonation_electronic_kcal
from nhc_deprot.pipeline.parent_handoff import (
    FAILED_PARENT_HANDOFF,
    FINAL_PARENT_GAU_CONVERGED,
    HANDOFF_CALIBRATION_MISS,
    HANDOFF_CALIBRATION_PASS,
)
from nhc_deprot.pipeline.scientific_validation import (
    FrozenEndpointGeometry,
    PureReferenceLabel,
    ScientificValidationError,
    SimulatedAimnet2Engine,
    SimulatedParentEngine,
    exact_byte_handoff_payload,
    route_contract_summary,
    run_scientific_validation_for_checkpoint,
    select_after_scientific_validation,
    writer_is_implemented,
)


def _geom(root: str, endpoint: str, n_atoms: int = 3) -> FrozenEndpointGeometry:
    elements = tuple(["C"] * n_atoms)
    coords = tuple((float(i), 0.0, 0.0) for i in range(n_atoms))
    charge = 1 if endpoint == "cation" else 0
    mult = 1
    return FrozenEndpointGeometry(
        root_id=root,
        endpoint=endpoint,
        elements=elements,
        coordinates=coords,
        charge=charge,
        multiplicity=mult,
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


def test_writer_implemented_flag() -> None:
    assert writer_is_implemented() is True
    summary = route_contract_summary()
    assert summary["single_point_only"] is False
    assert summary["aimnet2_energy_enters_label"] is False
    assert summary["implemented"] is True


def test_exact_byte_handoff_binds_geometry() -> None:
    payload = exact_byte_handoff_payload(
        elements=["C", "N"],
        coordinates=((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
        charge=1,
        multiplicity=1,
        checkpoint_id="epoch-10",
        root_id="ROOT",
        endpoint="cation",
    )
    assert payload["exact_bytes"] is True
    assert payload["single_point_only"] is False
    assert len(payload["handoff_sha256"]) == 64


def test_full_route_sim_pass_and_select() -> None:
    root = "VALROOT"
    geoms = [_geom(root, "cation"), _geom(root, "neutral")]
    refs = {root: _ref(root)}
    aim = SimulatedAimnet2Engine(converge=True, steps=8)
    parent = SimulatedParentEngine(
        energy_cation=-100.0, energy_neutral=-99.5, opt_steps=15, handoff_pass=True
    )
    result = run_scientific_validation_for_checkpoint(
        epoch=20,
        checkpoint_id="ckpt-20",
        checkpoint_sha256="b" * 64,
        route_kind="finetuned_checkpoint",
        geometries=geoms,
        references=refs,
        aimnet2=aim,
        parent=parent,
        live=False,
    )
    assert result.catastrophic_failure_count == 0
    assert result.all_identity_and_structure_hard_gates is True
    assert result.maximum_absolute_label_error_kcal_mol == pytest.approx(0.0, abs=1e-9)
    assert result.single_point_only is False
    assert result.live_chemistry_executed is False
    assert result.root_receipts[0].cation is not None
    assert result.root_receipts[0].cation.handoff_classification == HANDOFF_CALIBRATION_PASS
    assert result.root_receipts[0].cation.parent_final_state == FINAL_PARENT_GAU_CONVERGED

    # Second candidate with larger parent burden but same labels still selectable
    parent_slow = SimulatedParentEngine(
        energy_cation=-100.0, energy_neutral=-99.5, opt_steps=40, scf_cycles=200
    )
    worse = run_scientific_validation_for_checkpoint(
        epoch=10,
        checkpoint_id="ckpt-10",
        checkpoint_sha256="c" * 64,
        route_kind="finetuned_checkpoint",
        geometries=geoms,
        references=refs,
        aimnet2=aim,
        parent=parent_slow,
        epoch0_baseline=result,
        live=False,
    )
    selection = select_after_scientific_validation([result, worse])
    assert selection["outcome"] == "VALIDATION_SELECTED"
    assert selection["selected_epoch"] == 20
    assert selection["test_authorized"] is False


def test_handoff_miss_still_continues_parent() -> None:
    root = "VALROOT"
    geoms = [_geom(root, "cation"), _geom(root, "neutral")]
    refs = {root: _ref(root)}
    result = run_scientific_validation_for_checkpoint(
        epoch=5,
        checkpoint_id="ckpt-5",
        checkpoint_sha256="d" * 64,
        route_kind="finetuned_checkpoint",
        geometries=geoms,
        references=refs,
        aimnet2=SimulatedAimnet2Engine(),
        parent=SimulatedParentEngine(handoff_pass=False),
        live=False,
    )
    assert result.root_receipts[0].cation is not None
    assert result.root_receipts[0].cation.handoff_classification == HANDOFF_CALIBRATION_MISS
    assert result.root_receipts[0].cation.continue_parent_optimization is True
    assert result.root_receipts[0].cation.parent_final_state == FINAL_PARENT_GAU_CONVERGED
    assert result.catastrophic_failure_count == 0


def test_failed_parent_handoff_is_catastrophic() -> None:
    root = "VALROOT"
    geoms = [_geom(root, "cation"), _geom(root, "neutral")]
    refs = {root: _ref(root)}
    result = run_scientific_validation_for_checkpoint(
        epoch=5,
        checkpoint_id="ckpt-fail",
        checkpoint_sha256="e" * 64,
        route_kind="finetuned_checkpoint",
        geometries=geoms,
        references=refs,
        aimnet2=SimulatedAimnet2Engine(),
        parent=SimulatedParentEngine(fail_scf=True),
        live=False,
    )
    assert result.catastrophic_failure_count == 1
    assert result.all_identity_and_structure_hard_gates is False
    assert result.root_receipts[0].cation is not None
    assert result.root_receipts[0].cation.handoff_classification == FAILED_PARENT_HANDOFF


def test_live_requires_gate() -> None:
    root = "VALROOT"
    with pytest.raises(ScientificValidationError, match="scientific_validation_live"):
        run_scientific_validation_for_checkpoint(
            epoch=1,
            checkpoint_id="x",
            checkpoint_sha256="f" * 64,
            route_kind="finetuned_checkpoint",
            geometries=[_geom(root, "cation"), _geom(root, "neutral")],
            references={root: _ref(root)},
            aimnet2=SimulatedAimnet2Engine(),
            parent=SimulatedParentEngine(),
            live=True,
            scientific_validation_live=False,
        )


def test_label_never_uses_aimnet2_energy() -> None:
    """Simulated AIMNet2 returns energy_ev; label must equal parent-only formula."""

    root = "VALROOT"
    e_c, e_n = -123.0, -122.4
    parent = SimulatedParentEngine(energy_cation=e_c, energy_neutral=e_n)
    result = run_scientific_validation_for_checkpoint(
        epoch=3,
        checkpoint_id="ckpt",
        checkpoint_sha256="1" * 64,
        route_kind="finetuned_checkpoint",
        geometries=[_geom(root, "cation"), _geom(root, "neutral")],
        references={root: _ref(root, e_c=e_c, e_n=e_n)},
        aimnet2=SimulatedAimnet2Engine(),
        parent=parent,
        live=False,
    )
    expected = deprotonation_electronic_kcal(e_n, e_c)
    assert result.root_receipts[0].label_kcal == pytest.approx(expected)
    assert result.mean_absolute_label_error_kcal_mol == pytest.approx(0.0, abs=1e-12)
    assert not any(
        ep.aimnet2_energy_used_in_label
        for r in result.root_receipts
        for ep in (r.cation, r.neutral)
        if ep is not None
    )
