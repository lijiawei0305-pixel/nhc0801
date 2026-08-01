"""Training readiness / numeric calibration / forbidden stack tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.contracts.forbidden_stacks import (
    ForbiddenStackError,
    assert_parent_protocol_allowed,
    assert_quick_val_not_final_selector,
)
from nhc_deprot.pipeline.mindmap_orchestrator import preflight, run_dry
from nhc_deprot.pipeline.training_blockers import (
    REASON_NUMERIC,
    assess_training_readiness,
    load_numeric_calibration,
)

REPO = Path(__file__).resolve().parents[1]


def test_numeric_calibration_frozen_and_valid() -> None:
    payload = load_numeric_calibration()
    assert payload["status"] == "FROZEN"
    assert payload["chosen_before_final_test"] is True
    assert payload["label_error_tolerance_kcal_mol"] > 0


def test_numeric_blocker_resolved() -> None:
    readiness = assess_training_readiness(repo_root=REPO)
    numeric = next(b for b in readiness.blockers if b.code == REASON_NUMERIC)
    assert numeric.status == "RESOLVED"
    assert readiness.state == "BLOCKED_BEFORE_TRAINING"
    assert REASON_NUMERIC not in readiness.open_hard


def test_scientific_validation_writer_blocker_resolved() -> None:
    from nhc_deprot.pipeline.training_blockers import REASON_SCI_VAL

    readiness = assess_training_readiness(repo_root=REPO)
    sci = next(b for b in readiness.blockers if b.code == REASON_SCI_VAL)
    assert sci.status == "RESOLVED"
    assert REASON_SCI_VAL not in readiness.open_hard


def test_forbidden_b3lyp_parent() -> None:
    with pytest.raises(ForbiddenStackError, match="B3LYP"):
        assert_parent_protocol_allowed({"functional": "B3LYP-D3(BJ)", "basis": "def2-SVP"})


def test_forbidden_quick_val_final_select() -> None:
    with pytest.raises(ForbiddenStackError, match="quick validation"):
        assert_quick_val_not_final_selector({"quick_validation_may_select_final_model": True})


def test_orchestrator_preflight_dry() -> None:
    report = preflight(repo_root=REPO)
    assert report.preflight_ok is True
    assert report.split_summary["train_count"] == 3
    assert report.split_summary["validation_count"] == 2
    assert report.split_summary["final_test_identities_loaded"] is False
    assert len(report.steps) == 13
    # Live train steps blocked by default
    train_step = next(s for s in report.steps if s.step == 4)
    assert train_step.action == "blocked"
    text = run_dry(repo_root=REPO)
    assert "BLOCKED_BEFORE_TRAINING" in text
    assert "NUMERIC_CALIBRATION" in text
