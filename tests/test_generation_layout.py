"""Generation layout tests (local sandbox only)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import (
    DEFAULT_GENERATION_ID,
    GenerationError,
    init_generation,
    load_generation_meta,
    resolve_layout,
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
