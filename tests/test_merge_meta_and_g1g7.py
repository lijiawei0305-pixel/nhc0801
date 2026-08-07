"""G1–G7 gates for first full train chain (2026-08-07 plan)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256, TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import ensure_generation_tree, init_generation, resolve_layout
from nhc_deprot.pipeline.d3_projection import run_d3_campaign
from nhc_deprot.pipeline.epoch0_campaign_rebuild import resolve_epoch0_root_receipt_path
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import assemble_weighted_dataset
from nhc_deprot.resources.profiles import get_profile
from nhc_deprot.training.ablation_cli import run_train_ablation
from nhc_deprot.training.config import TRAINABLE_MLP_SHIFT, TrainingConfig
from nhc_deprot.training.merge_meta import (
    MERGE_META_SCHEMA,
    MergeMetaError,
    assert_merge_meta_ready,
    build_merge_meta,
    write_merge_meta,
)
from nhc_deprot.training.multi_seed_trainer import (
    DryRunTrainBackend,
    TrainerError,
    run_multi_seed_training,
)


def test_g1_resolve_root_receipt_prefers_root_json(tmp_path: Path) -> None:
    e0 = tmp_path / "epoch0" / "RID"
    e0.mkdir(parents=True)
    (e0 / "root.json").write_text('{"status":"PASS","via":"root"}', encoding="utf-8")
    (e0 / "epoch0_root_receipt.json").write_text(
        '{"status":"FAILED","via":"legacy"}', encoding="utf-8"
    )
    p = resolve_epoch0_root_receipt_path(tmp_path / "epoch0", "RID")
    assert p is not None
    assert p.name == "root.json"
    assert json.loads(p.read_text())["via"] == "root"


def test_g1_resolve_root_receipt_falls_back_to_legacy(tmp_path: Path) -> None:
    e0 = tmp_path / "epoch0" / "RID"
    e0.mkdir(parents=True)
    (e0 / "epoch0_root_receipt.json").write_text(
        '{"status":"PASS"}', encoding="utf-8"
    )
    p = resolve_epoch0_root_receipt_path(tmp_path / "epoch0", "RID")
    assert p is not None
    assert p.name == "epoch0_root_receipt.json"


def test_g2_merge_meta_disjoint_asserted() -> None:
    with pytest.raises(MergeMetaError, match="overlap"):
        build_merge_meta(
            merge_group_id="g001",
            archive_path="/tmp/arch",
            merged_from_groups=["teacher_gpu_g001"],
            train_roots=["A", "B"],
            val_roots=["B", "C"],
        )


def test_g2_write_and_assert_merge_meta(tmp_path: Path) -> None:
    train_dir = tmp_path / "train_g001"
    train_dir.mkdir()
    meta = build_merge_meta(
        merge_group_id="g001",
        archive_path=str(tmp_path / "_archive_g001_pre_merge_x"),
        merged_from_groups=["teacher_gpu_g001", "teacher_gpu_g374"],
        train_roots=list(TRAIN_ROOTS)[:3],
        val_roots=list(VALIDATION_ROOTS)[:2],
    )
    assert meta["train_val_disjoint"] is True
    assert meta["schema"] == MERGE_META_SCHEMA
    path = write_merge_meta(train_dir, meta)
    assert path.is_file()
    loaded = assert_merge_meta_ready(train_dir)
    assert loaded["train_val_disjoint"] is True


def test_g2_live_train_refuses_missing_merge_meta(tmp_path: Path) -> None:
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
    # no merge_meta written
    class _Mock:
        def train_epoch(self, *a, **k):
            return {
                "train_weighted_loss": 1.0,
                "energy_bias": 0.0,
                "weighted_energy_mse": 1.0,
                "weighted_forces_mse": 1.0,
            }

        def evaluate(self, *a, **k):
            return {
                "validation_weighted_loss": 1.0,
                "checkpoint_selection_permitted": False,
            }

        def step_scheduler(self, *a, **k):
            return None

    with pytest.raises(TrainerError, match="merge_meta"):
        run_multi_seed_training(
            layout=layout,
            config=TrainingConfig(seeds=(20260730,), epochs=1),
            dry_run=False,
            aimnet2_train_authorized=True,
            backend=_Mock(),  # type: ignore[arg-type]
            skip_dataset_audit=True,
        )


def test_g5_no_teacher_gpu_g0_star_glob_in_scanners() -> None:
    """Regression: teacher_gpu_g0* silently drops g100+ product dirs."""

    repo = Path(__file__).resolve().parents[1]
    bad: list[str] = []
    for path in (repo / "src").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if 'glob("teacher_gpu_g0' in text or "glob('teacher_gpu_g0" in text:
            bad.append(str(path.relative_to(repo)))
    for path in (repo / "scripts").rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        if 'glob("teacher_gpu_g0' in text or "glob('teacher_gpu_g0" in text:
            bad.append(str(path.relative_to(repo)))
    assert bad == [], f"forbidden teacher_gpu_g0* glob in: {bad}"


def test_g7_backend_factory_called_per_seed(tmp_path: Path) -> None:
    """Each seed gets a fresh backend (official weight reload), not chained state."""

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
    train_dir = layout.train_batch_dir("g001")
    train_dir.mkdir(parents=True, exist_ok=True)
    write_merge_meta(
        train_dir,
        build_merge_meta(
            merge_group_id="g001",
            archive_path=str(tmp_path / "arch"),
            merged_from_groups=["teacher_gpu_g001"],
            train_roots=list(TRAIN_ROOTS),
            val_roots=list(VALIDATION_ROOTS),
        ),
    )

    built_seeds: list[int] = []
    factory_base_weights: list[str] = []
    # §7.4: factory must be invoked once per seed with the same official path.
    # Mock path uses a sentinel file whose SHA we stamp as the "base weight".
    base_weight_path = tmp_path / "fake_official.pt"
    base_weight_path.write_bytes(b"official-aimnet2-mock-bytes")
    expected_base = str(base_weight_path.resolve())

    class _PerSeed:
        def __init__(self, seed: int, *, base_weight: Path) -> None:
            self.seed = seed
            self.base_weight = Path(base_weight)
            self._inner = DryRunTrainBackend()
            built_seeds.append(seed)
            factory_base_weights.append(str(self.base_weight.resolve()))

        def train_epoch(self, *a, **k):
            return self._inner.train_epoch(*a, **k)

        def evaluate(self, *a, **k):
            return self._inner.evaluate(*a, **k)

        def step_scheduler(self, *a, **k):
            return self._inner.step_scheduler(*a, **k)

    seeds = (20260730, 20260731, 20260732)

    def _factory(s: int) -> _PerSeed:
        return _PerSeed(s, base_weight=base_weight_path)

    camp = run_multi_seed_training(
        layout=layout,
        config=TrainingConfig(seeds=seeds, epochs=1, checkpoint_interval_epochs=1),
        dry_run=False,
        aimnet2_train_authorized=True,
        backend_factory=_factory,  # type: ignore[arg-type,return-value]
        skip_dataset_audit=True,
    )
    assert camp["status"] == "LIVE_TRAIN_PASS"
    assert built_seeds == list(seeds)
    assert factory_base_weights == [expected_base] * len(seeds)
    # Campaign still stamps the official constant (policy identity).
    assert camp["official_base_weight_sha256"] == OFFICIAL_AIMNET2_WEIGHT_SHA256


def test_g7_mlp_shift_regex_tuple_both_patterns() -> None:
    """T8: mlp_shift must apply both energy_mlp and atomic_shift regexes."""

    assert len(TRAINABLE_MLP_SHIFT) == 2
    joined = " ".join(TRAINABLE_MLP_SHIFT)
    assert "energy_mlp" in joined
    assert "atomic_shift" in joined


def test_g7_ablation_factory_path_writes_merge_gate(tmp_path: Path) -> None:
    """run_train_ablation live with factory still enforces merge_meta."""

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
    # missing merge_meta → fail closed even with injected factory via base_weight path
    # (we inject backend to avoid torch)
    class _Mock:
        def train_epoch(self, *a, **k):
            return {
                "train_weighted_loss": 1.0,
                "energy_bias": 0.0,
                "weighted_energy_mse": 1.0,
                "weighted_forces_mse": 1.0,
            }

        def evaluate(self, *a, **k):
            return {
                "validation_weighted_loss": 1.0,
                "checkpoint_selection_permitted": False,
            }

        def step_scheduler(self, *a, **k):
            return None

    with pytest.raises(TrainerError, match="merge_meta"):
        run_train_ablation(
            layout=layout,
            run_ids=("e1f1_mlp",),
            dry_run=False,
            aimnet2_train_authorized=True,
            base_config=TrainingConfig(seeds=(20260730,), epochs=1),
            backend=_Mock(),  # type: ignore[arg-type]
        )
