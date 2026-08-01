"""Epoch-0 dry-run tests (mindmap step 3)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256, VALIDATION_ROOTS
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.epoch0_runner import (
    Epoch0Error,
    plan_epoch0_paths,
    run_epoch0_campaign,
)
from nhc_deprot.pipeline.parent_handoff import (
    FINAL_PARENT_GAU_CONVERGED,
    HANDOFF_CALIBRATION_PASS,
)
from nhc_deprot.pipeline.scientific_validation import SimulatedAimnet2Engine


def test_plan_epoch0_paths(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    plan = plan_epoch0_paths(layout)
    assert plan["mindmap_step"] == 3
    assert len(plan["roots"]) == len(VALIDATION_ROOTS)
    assert plan["official_weight_sha256"] == OFFICIAL_AIMNET2_WEIGHT_SHA256


def test_dry_run_epoch0_campaign(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    camp = run_epoch0_campaign(layout=layout, dry_run=True)
    assert camp["status"] == "DRY_RUN_EPOCH0_PASS"
    assert camp["dry_run"] is True
    assert camp["live_chemistry"] is False
    assert camp["final_test_payload_read"] is False
    assert camp["training_started"] is False
    assert camp["official_weight_sha256"] == OFFICIAL_AIMNET2_WEIGHT_SHA256
    assert camp["parent_protocol_sha256"] == PROTOCOL_SHA256
    assert camp["failed_root_count"] == 0
    assert camp["root_count"] == 2

    # aggregate label error should be ~0 with matching synthetic energies
    assert camp["baseline_metrics"]["mean_absolute_label_error_kcal_mol"] == pytest.approx(
        0.0, abs=1e-9
    )

    for root_id in VALIDATION_ROOTS:
        path = layout.epoch0_dir / root_id / "epoch0_root_receipt.json"
        assert path.is_file()
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload["status"] == "PASS"
        assert payload["comparison"]["absolute_label_error_kcal"] == pytest.approx(0.0, abs=1e-9)
        e0 = payload["epoch0_route"]
        assert e0["cation"]["handoff_classification"] == HANDOFF_CALIBRATION_PASS
        assert e0["cation"]["parent_final_state"] == FINAL_PARENT_GAU_CONVERGED
        assert e0["cation"]["aimnet2_energy_used_in_label"] is False
        # epoch-0 should use fewer parent steps than pure in our synthetic setup
        assert payload["comparison"]["epoch0_parent_opt_steps"] < payload["comparison"][
            "pure_parent_opt_steps"
        ]

    assert (layout.epoch0_dir / "campaign_receipt.json").is_file()


def test_live_without_gate_fails(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    with pytest.raises(Epoch0Error, match="epoch0_execution"):
        run_epoch0_campaign(layout=layout, dry_run=False, epoch0_execution=False)


def test_live_rejects_simulated_engines(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    with pytest.raises(Epoch0Error, match="Simulated|injected"):
        run_epoch0_campaign(
            layout=layout,
            dry_run=False,
            epoch0_execution=True,
            aimnet2=SimulatedAimnet2Engine(),
            parent=None,
        )
