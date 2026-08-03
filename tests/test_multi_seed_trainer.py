"""Multi-seed trainer dry-run tests (mindmap 4–5)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.d3_projection import run_d3_campaign
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset
from nhc_deprot.resources.profiles import get_profile
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.multi_seed_trainer import (
    TrainerError,
    run_multi_seed_training,
)


def _bootstrap_dataset(tmp_path: Path):
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=2),
        dry_run=True,
    )
    run_d3_campaign(layout=layout, root_ids=roots, dry_run=True, overwrite=True)
    assemble_weighted_dataset(
        layout=layout,
        train_roots=list(TRAIN_ROOTS),
        validation_roots=list(VALIDATION_ROOTS),
        dry_run=True,
        overwrite=True,
    )
    return layout


def test_multi_seed_dry_run_retains_all_outcomes(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    cfg = TrainingConfig(
        seeds=(20260730, 20260731),
        epochs=200,
        checkpoint_interval_epochs=2,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=4,
    )
    assert camp["status"] == "DRY_RUN_TRAIN_PASS"
    assert camp["final_model_selected"] is False
    assert camp["quick_validation_may_select_final_model"] is False
    assert camp["scientific_validation_required_before_final_selection"] is True
    assert camp["epochs_effective"] == 4
    assert camp["failed_seed_count"] == 0
    assert len(camp["seed_results"]) == 2

    for seed_res in camp["seed_results"]:
        assert seed_res["status"] == "PASS"
        assert seed_res["epochs_run"] == 4
        assert len(seed_res["epoch_logs"]) == 4
        # checkpoints at epochs 2 and 4
        epochs = {c["epoch"] for c in seed_res["checkpoints"]}
        assert epochs == {2, 4}
        assert seed_res["shortlist_epochs"]  # non-empty shortlist
        for ckpt in seed_res["checkpoints"]:
            assert ckpt["checkpoint_selection_permitted"] is False
            assert ckpt["live_weights_written"] is False
            meta_path = Path(ckpt["path"])
            assert meta_path.is_file()

    run_id = cfg.run_id
    run_dir = layout.train_batch_run_dir("g001", run_id)
    receipt = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    assert receipt["final_model_selected"] is False
    assert receipt["batch_id"] == "g001"
    assert receipt["run_id"] == run_id
    assert receipt["product_rel"] == f"train_g001/runs/{run_id}"
    # products under train_g001/runs/<run_id>/seed_*/ — not legacy seed_* or bare train/
    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    assert seed_dir.is_dir()
    assert (seed_dir / "seed_result.json").is_file()
    assert not layout.train_seed_dir("g001", 20260730).exists()
    assert not (layout.train_dir / "campaign_receipt.json").exists()


def test_live_without_auth_fails(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    with pytest.raises(TrainerError, match="aimnet2_train_authorized"):
        run_multi_seed_training(
            layout=layout,
            dry_run=False,
            aimnet2_train_authorized=False,
            dry_run_epochs=1,
        )


def test_config_rejects_quick_val_final_select() -> None:
    with pytest.raises(ValueError, match="quick_validation_may_select_final_model"):
        TrainingConfig(quick_validation_may_select_final_model=True).assert_policy()
