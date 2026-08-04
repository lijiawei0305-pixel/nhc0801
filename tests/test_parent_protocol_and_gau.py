"""Contract smoke tests for parent protocol and GAU_LOOSE profile."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.contracts.parent_protocol import (  # noqa: E402
    BASIS,
    FUNCTIONAL,
    PROTOCOL_SHA256,
    deprotonation_electronic_kcal,
)
from nhc_deprot.pipeline.parent_handoff import (  # noqa: E402
    FAILED_PARENT_HANDOFF,
    HANDOFF_CALIBRATION_MISS,
    HANDOFF_CALIBRATION_PASS,
    classify_first_parent_gradient,
    load_gau_loose_profile,
)


def test_parent_protocol_is_p01_not_b3lyp_svp() -> None:
    assert FUNCTIONAL == "wb97m-d3bj"
    assert BASIS == "def2-TZVPP"
    assert PROTOCOL_SHA256.startswith("227c22a5")


def test_label_formula() -> None:
    label = deprotonation_electronic_kcal(-100.0, -100.4)
    expected = (0.4 * 627.509474) - 6.28
    assert abs(label - expected) < 1e-9


def test_load_default_gau_loose() -> None:
    profile = load_gau_loose_profile()
    assert profile.gradient_rms_eh_bohr == 1.7e-3
    assert profile.gradient_max_eh_bohr == 2.5e-3
    assert profile.ase_fmax_ev_angstrom == 0.10
    assert profile.maximum_steps == 250  # GAU_LOOSE_V002 budget


def test_gau_loose_v001_preserved_at_100() -> None:
    from nhc_deprot.pipeline.parent_handoff import GAU_LOOSE_V001_CONTRACT

    v001 = load_gau_loose_profile(GAU_LOOSE_V001_CONTRACT)
    assert v001.maximum_steps == 100
    assert v001.ase_fmax_ev_angstrom == 0.10
    assert v001.gradient_max_eh_bohr == 2.5e-3


def _base_ok(**overrides):
    profile = load_gau_loose_profile()
    kwargs = dict(
        profile=profile,
        scf_converged=True,
        energy_hartree=-1000.0,
        gradient_hartree_bohr=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        coordinates_finite=True,
        atom_identity_preserved=True,
        charge_multiplicity_preserved=True,
        topology_valid=True,
    )
    kwargs.update(overrides)
    return classify_first_parent_gradient(**kwargs)


def test_handoff_pass_continues() -> None:
    result = _base_ok()
    assert result["classification"] == HANDOFF_CALIBRATION_PASS
    assert result.get("continue_same_parent_optimization") is True


def test_handoff_miss_still_continues() -> None:
    # large gradient -> MISS but still continue
    big = ((1.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    result = _base_ok(gradient_hartree_bohr=big)
    assert result["classification"] == HANDOFF_CALIBRATION_MISS
    assert result.get("continue_same_parent_optimization") is True


def test_handoff_failed_stops() -> None:
    result = _base_ok(scf_converged=False)
    assert result["classification"] == FAILED_PARENT_HANDOFF


def test_analytic_gradient_unavailable_when_missing() -> None:
    result = _base_ok(gradient_hartree_bohr=None)
    assert result["classification"] == FAILED_PARENT_HANDOFF
    assert "ANALYTIC_GRADIENT_UNAVAILABLE" in (result.get("failure_types") or [])


def test_worker_gradient_hartree_per_bohr_alias_is_usable() -> None:
    """Live worker emits gradient_hartree_per_bohr; sci-val must accept it.

    g002 Epoch-0 failed with ANALYTIC_GRADIENT_UNAVAILABLE solely because the
    reader only looked for gradient_hartree_bohr (simulated-engine key).
    """
    worker_payload = {
        "scf_converged": True,
        "energy_hartree": -1000.0,
        "gradient_hartree_per_bohr": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        # deliberately omit gradient_hartree_bohr
    }
    grad = worker_payload.get("gradient_hartree_bohr")
    if grad is None:
        grad = worker_payload.get("gradient_hartree_per_bohr")
    result = _base_ok(gradient_hartree_bohr=grad)
    assert result["classification"] == HANDOFF_CALIBRATION_PASS
    assert result.get("continue_same_parent_optimization") is True
