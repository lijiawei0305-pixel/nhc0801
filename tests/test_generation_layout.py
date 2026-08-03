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
