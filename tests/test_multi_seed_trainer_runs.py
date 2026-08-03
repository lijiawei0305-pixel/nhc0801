"""M8: multi_seed trainer writes under train_g00N/runs/<run_id>/seed_*/."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from nhc_deprot.data.io_util import sha256_file
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.d3_projection import run_d3_campaign
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset
from nhc_deprot.resources.profiles import get_profile
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.live_aimnet2 import train_config_digest
from nhc_deprot.training.multi_seed_trainer import (
    DryRunTrainBackend,
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


class _RecordingSchedulerBackend(DryRunTrainBackend):
    """Dry backend that records step_scheduler calls (M7/M8 contract)."""

    def __init__(self) -> None:
        self.scheduler_steps: list[float] = []

    def step_scheduler(self, val_loss: float) -> None:
        self.scheduler_steps.append(float(val_loss))


def test_train_products_land_under_run_id_not_legacy_seed(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    run_id = "e1f100_mlp_shift"
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=4,
        checkpoint_interval_epochs=2,
        run_id=run_id,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=4,
    )
    assert camp["status"] == "DRY_RUN_TRAIN_PASS"
    assert camp["run_id"] == run_id

    run_dir = layout.train_batch_run_dir("g001", run_id)
    assert run_dir.is_dir()
    assert camp["product_dir"] == str(run_dir)
    assert (run_dir / "train_info.json").is_file()
    assert (run_dir / "train_result.json").is_file()

    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    assert seed_dir.is_dir()
    assert (seed_dir / "seed_result.json").is_file()
    assert (seed_dir / "epoch_0002.meta.json").is_file()
    assert (seed_dir / "epoch_0004.meta.json").is_file()

    # New writes must not use legacy train_g00N/seed_* (no run_id layer)
    legacy_seed = layout.train_seed_dir("g001", 20260730)
    assert not legacy_seed.exists()
    assert not (layout.train_batch_dir("g001") / "train_result.json").exists()
    assert not (layout.train_batch_dir("g001") / "train_info.json").exists()

    for ckpt in camp["seed_results"][0]["checkpoints"]:
        meta_path = Path(ckpt["path"])
        assert run_id in meta_path.parts
        assert meta_path.is_relative_to(run_dir)
        assert ckpt.get("run_id") == run_id


def test_train_info_has_run_id_digest_and_manifest_sha(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=2,
        checkpoint_interval_epochs=1,
        run_id="e1f1_mlp",
        forces_weight=100.0,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=2,
    )
    run_dir = layout.train_batch_run_dir("g001", cfg.run_id)
    info: dict[str, Any] = json.loads(
        (run_dir / "train_info.json").read_text(encoding="utf-8")
    )
    assert info["run_id"] == cfg.run_id
    assert info["batch_id"] == "g001"
    expected_digest = train_config_digest(cfg)
    assert info["train_config_digest"] == expected_digest
    assert camp["train_config_digest"] == expected_digest

    manifest_path = layout.datasets_dir / "manifest.json"
    expected_sha = sha256_file(manifest_path)
    assert info["dataset_manifest_sha256"] == expected_sha
    assert camp["dataset_manifest_sha256"] == expected_sha
    assert len(info["dataset_manifest_sha256"]) == 64


def test_run_id_kwarg_overrides_config(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    cfg = TrainingConfig(seeds=(20260730,), epochs=2, run_id="e1f1_mlp")
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=2,
        run_id="e1f100_mlp",
    )
    assert camp["run_id"] == "e1f100_mlp"
    assert layout.train_batch_run_dir("g001", "e1f100_mlp").is_dir()
    assert not layout.train_batch_run_dir("g001", "e1f1_mlp").exists()


def test_step_scheduler_called_once_per_epoch(tmp_path: Path) -> None:
    layout = _bootstrap_dataset(tmp_path)
    backend = _RecordingSchedulerBackend()
    cfg = TrainingConfig(
        seeds=(20260730, 20260731),
        epochs=3,
        checkpoint_interval_epochs=10,
        run_id="e1f1_mlp",
        quick_validation_each_epoch=True,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=3,
        backend=backend,
    )
    assert camp["status"] == "DRY_RUN_TRAIN_PASS"
    # 2 seeds × 3 epochs
    assert len(backend.scheduler_steps) == 6
    for v in backend.scheduler_steps:
        assert isinstance(v, float)
        assert v == v  # not NaN


def test_dry_run_regression_still_passes_with_default_run_id(tmp_path: Path) -> None:
    """Default TrainingConfig.run_id lands products under runs/<run_id>/."""
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
    assert camp["epochs_effective"] == 4
    assert camp["failed_seed_count"] == 0
    assert camp["run_id"] == cfg.run_id
    assert len(camp["seed_results"]) == 2

    run_dir = layout.train_batch_run_dir("g001", cfg.run_id)
    receipt = json.loads((run_dir / "train_result.json").read_text(encoding="utf-8"))
    assert receipt["final_model_selected"] is False
    assert receipt["batch_id"] == "g001"
    assert receipt["run_id"] == cfg.run_id

    for seed_res in camp["seed_results"]:
        assert seed_res["status"] == "PASS"
        assert seed_res["epochs_run"] == 4
        epochs = {c["epoch"] for c in seed_res["checkpoints"]}
        assert epochs == {2, 4}
        assert seed_res["shortlist_epochs"]
