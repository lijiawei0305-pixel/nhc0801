"""Val-only epoch-0 policy: never train roots; uniform disk layout."""

from __future__ import annotations

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import resolve_layout
from nhc_deprot.pipeline.e0_val_only import (
    E0ValOnlyError,
    normalize_epoch0_batch_id,
    refuse_train_roots,
)


def test_refuse_train_roots_ok():
    got = refuse_train_roots(list(VALIDATION_ROOTS))
    assert got == list(VALIDATION_ROOTS)


def test_refuse_train_roots_blocks_train():
    with pytest.raises(E0ValOnlyError, match="REFUSED train"):
        refuse_train_roots([TRAIN_ROOTS[0], VALIDATION_ROOTS[0]])


def test_refuse_empty():
    with pytest.raises(E0ValOnlyError, match="empty"):
        refuse_train_roots([])


def test_normalize_batch_id_g001_aliases():
    assert normalize_epoch0_batch_id("g001") == "g001"
    assert normalize_epoch0_batch_id("g001_pilot") == "g001"
    assert normalize_epoch0_batch_id("g002") == "g002"


def test_g001_epoch0_disk_same_pattern_as_g002(tmp_path):
    layout = resolve_layout(generation_id="nhc0801-g001", nhc0801_root=tmp_path)
    assert layout.epoch0_dir == layout.generation_root / "epoch0_val_batches" / "g001" / "epoch0"
    assert layout.epoch0_batch_dir("g002") == (
        layout.generation_root / "epoch0_val_batches" / "g002" / "epoch0"
    )
    assert layout.epoch0_batch_dir("g001") == layout.epoch0_dir
