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


class _MockLiveExportBackend:
    """Non-dry backend with export_checkpoint — never loads torch/AIMNet2.

    Records every export path so tests can assert checkpoint interval + last epoch.
    """

    def __init__(self, *, train_config_digest: str = "deadbeef" * 8) -> None:
        self._inner = DryRunTrainBackend()
        self.export_paths: list[Path] = []
        self.train_config_digest = train_config_digest
        self.scheduler_steps: list[float] = []

    def train_epoch(
        self,
        batches: Any,
        *,
        split_frame_count: int,
        energy_weight: float,
        forces_weight: float,
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        return self._inner.train_epoch(
            batches,
            split_frame_count=split_frame_count,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            seed=seed,
            epoch=epoch,
        )

    def evaluate(
        self,
        batches: Any,
        *,
        energy_weight: float,
        forces_weight: float,
        energy_bias: float = 0.0,
    ) -> dict[str, Any]:
        return self._inner.evaluate(
            batches,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            energy_bias=energy_bias,
        )

    def step_scheduler(self, val_loss: float) -> None:
        self.scheduler_steps.append(float(val_loss))

    def export_checkpoint(self, path: Path) -> dict[str, Any]:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # Minimal fake weight file (not a real AIMNet2 bundle)
        path.write_bytes(b"NHC0801_MOCK_CHECKPOINT\n")
        self.export_paths.append(path)
        raw = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": "0" * 64,
            "live_weights_written": True,
            "train_config_digest": self.train_config_digest,
        }


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
        # Dry-run must not claim live weights or write .pt
        for ckpt in seed_res["checkpoints"]:
            assert ckpt["live_weights_written"] is False
            assert not Path(ckpt["weight_path"]).is_file()


def test_live_export_checkpoint_writes_pt_under_run_seed(tmp_path: Path) -> None:
    """M14fix-pt: live + export_checkpoint → epoch_NNNN.pt at each interval + last."""
    layout = _bootstrap_dataset(tmp_path)
    run_id = "e1f100_mlp_shift"
    digest = "a" * 64
    backend = _MockLiveExportBackend(train_config_digest=digest)
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=5,
        checkpoint_interval_epochs=2,
        run_id=run_id,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=False,
        aimnet2_train_authorized=True,
        backend=backend,
        require_merge_meta=False,
    )
    assert camp["status"] == "LIVE_TRAIN_PASS"
    assert camp["failed_seed_count"] == 0

    # interval 2 → epochs 2,4; always last → 5
    assert [p.name for p in backend.export_paths] == [
        "epoch_0002.pt",
        "epoch_0004.pt",
        "epoch_0005.pt",
    ]
    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    for pt in backend.export_paths:
        assert pt.is_file()
        assert pt.parent == seed_dir
        assert run_id in pt.parts
        assert "train_g001" in pt.parts

    seed_res = camp["seed_results"][0]
    assert seed_res["status"] == "PASS"
    epochs = {c["epoch"] for c in seed_res["checkpoints"]}
    assert epochs == {2, 4, 5}
    for ckpt in seed_res["checkpoints"]:
        assert ckpt["live_weights_written"] is True
        assert ckpt["train_config_digest"] == digest
        assert Path(ckpt["weight_path"]).is_file()
        assert Path(ckpt["path"]).is_file()
        meta = json.loads(Path(ckpt["path"]).read_text(encoding="utf-8"))
        assert meta["live_weights_written"] is True
        assert meta["train_config_digest"] == digest
        assert meta["weight_export"]["live_weights_written"] is True


def test_live_without_export_checkpoint_writes_meta_only(tmp_path: Path) -> None:
    """Live backend without export_checkpoint still retains meta; no .pt required."""
    layout = _bootstrap_dataset(tmp_path)
    # RecordingSchedulerBackend is DryRunTrainBackend subclass — not allowed live.
    # Use a minimal non-dry backend without export_checkpoint.
    class _NoExportBackend:
        def __init__(self) -> None:
            self._inner = DryRunTrainBackend()

        def train_epoch(self, *a: Any, **k: Any) -> dict[str, Any]:
            return self._inner.train_epoch(*a, **k)

        def evaluate(self, *a: Any, **k: Any) -> dict[str, Any]:
            return self._inner.evaluate(*a, **k)

        def step_scheduler(self, val_loss: float) -> None:
            self._inner.step_scheduler(val_loss)

    run_id = "e1f1_mlp"
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=2,
        checkpoint_interval_epochs=1,
        run_id=run_id,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=False,
        aimnet2_train_authorized=True,
        backend=_NoExportBackend(),
        require_merge_meta=False,
    )
    assert camp["status"] == "LIVE_TRAIN_PASS"
    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    for ckpt in camp["seed_results"][0]["checkpoints"]:
        assert ckpt["live_weights_written"] is False
        assert Path(ckpt["path"]).is_file()
        assert not Path(ckpt["weight_path"]).is_file()
    assert list(seed_dir.glob("*.pt")) == []


class _MockAuditingBackend(_MockLiveExportBackend):
    """Live backend that also exposes ``export_raw_audit_sibling`` (T7 dual export).

    Mirrors :class:`LiveAimnet2TrainBackend` without torch: writes the raw
    sibling next to the exported ``.pt`` and returns an audit payload.
    """

    def __init__(self, *, train_config_digest: str = "b" * 64) -> None:
        super().__init__(train_config_digest=train_config_digest)
        self.audited_paths: list[Path] = []

    def export_raw_audit_sibling(self, ema_path: Path) -> dict[str, Any]:
        ema_path = Path(ema_path)
        raw_path = ema_path.with_name(f"{ema_path.stem}.raw{ema_path.suffix}")
        raw_path.write_bytes(b"NHC0801_MOCK_RAW_CHECKPOINT\n")
        self.audited_paths.append(ema_path)
        return {
            "status": "EMA_EXPORT_AUDIT_PASS",
            "ema_decay": 0.99,
            "ema_enabled": True,
            "weights_diverged": True,
            "total_l2": 0.125,
            "raw_weight_path": str(raw_path),
            "exported_weight_path": str(ema_path),
        }


def test_dual_export_audit_runs_on_last_epoch_only(tmp_path: Path) -> None:
    """T7: raw sibling + on-disk EMA audit at the final checkpoint, not each interval."""
    layout = _bootstrap_dataset(tmp_path)
    run_id = "e1f100_mlp_shift"
    backend = _MockAuditingBackend()
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=5,
        checkpoint_interval_epochs=2,
        run_id=run_id,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=False,
        aimnet2_train_authorized=True,
        backend=backend,
        require_merge_meta=False,
    )
    assert camp["status"] == "LIVE_TRAIN_PASS"
    assert [p.name for p in backend.audited_paths] == ["epoch_0005.pt"]

    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    assert (seed_dir / "epoch_0005.raw.pt").is_file()
    # interval checkpoints must not get a raw sibling
    assert not (seed_dir / "epoch_0002.raw.pt").exists()
    assert not (seed_dir / "epoch_0004.raw.pt").exists()

    ckpts = {c["epoch"]: c for c in camp["seed_results"][0]["checkpoints"]}
    assert "ema_export_audit" not in ckpts[2]
    assert "ema_export_audit" not in ckpts[4]
    audit = ckpts[5]["ema_export_audit"]
    assert audit["status"] == "EMA_EXPORT_AUDIT_PASS"
    assert audit["weights_diverged"] is True
    # the audit must reach the on-disk meta, not just the in-memory receipt
    meta = json.loads(Path(ckpts[5]["path"]).read_text(encoding="utf-8"))
    assert meta["ema_export_audit"]["status"] == "EMA_EXPORT_AUDIT_PASS"


def test_backend_without_audit_method_is_unaffected(tmp_path: Path) -> None:
    """Backwards compatible: backends lacking the method still pass, no raw sibling."""
    layout = _bootstrap_dataset(tmp_path)
    run_id = "e1f1_mlp"
    backend = _MockLiveExportBackend()
    cfg = TrainingConfig(
        seeds=(20260730,),
        epochs=3,
        checkpoint_interval_epochs=2,
        run_id=run_id,
    )
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=False,
        aimnet2_train_authorized=True,
        backend=backend,
        require_merge_meta=False,
    )
    assert camp["status"] == "LIVE_TRAIN_PASS"
    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    assert not list(seed_dir.glob("*.raw.pt"))
    for ckpt in camp["seed_results"][0]["checkpoints"]:
        assert "ema_export_audit" not in ckpt


def test_dry_run_never_dual_exports(tmp_path: Path) -> None:
    """Dry-run must not write weights or run the audit even if the backend can."""
    layout = _bootstrap_dataset(tmp_path)
    run_id = "e1f1_mlp"
    cfg = TrainingConfig(seeds=(20260730,), epochs=3, run_id=run_id)
    camp = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=3,
    )
    seed_dir = layout.train_run_seed_dir("g001", run_id, 20260730)
    assert not list(seed_dir.glob("*.raw.pt"))
    for ckpt in camp["seed_results"][0]["checkpoints"]:
        assert "ema_export_audit" not in ckpt
