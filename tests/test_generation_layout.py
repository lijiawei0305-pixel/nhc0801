"""Generation layout tests (local sandbox only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import (
    DEFAULT_GENERATION_ID,
    GenerationError,
    default_model_version_for_train_batch,
    init_generation,
    load_generation_meta,
    normalize_model_version,
    resolve_layout,
)
from nhc_deprot.training.model_versions import (
    ModelVersionError,
    list_model_versions,
    register_model_version,
)


def test_init_generation_tree(tmp_path: Path) -> None:
    layout, meta, receipt = init_generation(
        generation_id=DEFAULT_GENERATION_ID,
        nhc0801_root=tmp_path / "NHC0801",
        source_commit="a" * 40,
    )
    assert layout.generation_root.is_dir()
    assert (layout.meta_dir / "generation.json").is_file()
    assert layout.resources_dir.is_dir()
    assert layout.teacher_dir.is_dir()
    assert receipt["wrote"] is True
    assert meta.scope == "C"
    assert meta.parallel_strategy == "S"
    assert meta.live_chemistry_authorized is False
    assert meta.final_test_identities_exposed is False
    assert meta.train_roots == list(TRAIN_ROOTS)
    assert meta.validation_roots == list(VALIDATION_ROOTS)
    loaded = load_generation_meta(layout.generation_meta_path())
    assert loaded.generation_id == DEFAULT_GENERATION_ID
    assert loaded.parent_protocol_sha256.startswith("227c22a5")


def test_no_overwrite_divergent_meta(tmp_path: Path) -> None:
    root = tmp_path / "NHC0801"
    init_generation(nhc0801_root=root, source_commit="b" * 40)
    with pytest.raises(GenerationError, match="differs"):
        init_generation(nhc0801_root=root, source_commit="c" * 40)


def test_teacher_paths(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    p = layout.teacher_endpoint_dir("ROOT-A", "cation")
    assert p.parts[-2:] == ("ROOT-A", "cation")
    # uniform group dirs: teacher_gpu_g001 / teacher_gpu_g002 / …
    assert layout.teacher_dir.name == "teacher_gpu_g001"
    assert layout.teacher_batch_dir("g001") == layout.teacher_dir
    assert layout.teacher_batch_dir("g002").name == "teacher_gpu_g002"
    assert layout.teacher_batch_dir("g003").name == "teacher_gpu_g003"


def test_train_batch_paths(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    assert layout.train_batch_dir("g001") == layout.generation_root / "train_g001"
    assert layout.train_batch_dir("g002").name == "train_g002"
    assert layout.train_seed_dir("g001", 20260730).name == "seed_20260730"
    assert layout.train_checkpoint_weight_path("g001", 20260730, 200).name == "epoch_0200.pt"
    assert layout.train_checkpoint_meta_path("g001", 20260730, 5).name == "epoch_0005.meta.json"
    assert layout.train_campaign_receipt_path("g003").name == "train_result.json"
    assert layout.train_seed_receipt_path("g001", 1).name == "seed_result.json"
    assert layout.train_batch_logs_dir("g001").name == "logs"
    # g001 read: empty scaffold ignored; legacy train/ with seed_* wins
    layout.train_batch_dir("g001").mkdir(parents=True)
    layout.train_dir.mkdir(parents=True)
    (layout.train_dir / "seed_1").mkdir()
    assert layout.resolve_train_batch_dir_for_read("g001") == layout.train_dir
    # canonical train_g001 with seed_* preferred
    (layout.train_batch_dir("g001") / "seed_2").mkdir()
    assert layout.resolve_train_batch_dir_for_read("g001") == layout.train_batch_dir("g001")


def test_model_version_paths_and_register(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    assert normalize_model_version("0.1") == "v0.1"
    assert normalize_model_version("v0.2") == "v0.2"
    # fixed order: train_g00N → v0.N
    assert default_model_version_for_train_batch("g001") == "v0.1"
    assert default_model_version_for_train_batch("g002") == "v0.2"
    assert default_model_version_for_train_batch("g010") == "v0.10"
    with pytest.raises(GenerationError):
        normalize_model_version("aimnet2_finetuned_best")

    assert layout.models_dir.name == "models"
    assert layout.model_version_dir("0.1").name == "v0.1"
    assert layout.model_weight_path("v0.1").name == "model.pt"
    assert layout.model_info_path("0.1").name == "info.json"

    src = tmp_path / "src.pt"
    src.write_bytes(b"fake-weights")
    # omit version → default from train_batch_id (g001 → v0.1)
    info = register_model_version(
        layout=layout,
        source_weight=src,
        train_batch_id="g001",
        seed=20260730,
        epoch=200,
        notes=["demo"],
    )
    assert info["version"] == "v0.1"
    assert info["weight_basename"] == "model.pt"
    assert layout.model_weight_path("v0.1").is_file()
    assert layout.model_weight_path("v0.1").read_bytes() == b"fake-weights"
    assert list_model_versions(layout) == ["v0.1"]
    assert layout.model_version_dir("v0.1").joinpath("card.svg").is_file()
    assert layout.model_version_dir("v0.1").joinpath("card.json").is_file()
    assert "v0.1" in layout.model_version_dir("v0.1").joinpath("card.svg").read_text(
        encoding="utf-8"
    )

    # g001 must not publish as v0.2
    with pytest.raises(ModelVersionError, match="v0.1"):
        register_model_version(
            layout=layout,
            version="v0.2",
            source_weight=src,
            train_batch_id="g001",
            overwrite=True,
        )
