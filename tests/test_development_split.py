"""Development split loader tests (synthetic + packaged V004 extract)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.data.development_split import (
    SPLIT_SCHEMA,
    load_development_split,
    load_packaged_v004_day1_split,
)
from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.paths import (
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)

REPO = Path(__file__).resolve().parents[1]


def _profile(candidate: str) -> dict[str, object]:
    return {
        "candidate": candidate,
        "root_id": candidate,
        "canonical_identity": candidate,
        "electron_count": 10,
        "cation_atom_count": 2,
        "neutral_atom_count": 1,
        "cation_sha256": "a" * 64,
        "neutral_sha256": "b" * 64,
    }


def test_load_synthetic_split(tmp_path: Path) -> None:
    payload = {
        "schema": SPLIT_SCHEMA,
        "train": [_profile("TRAIN-A"), _profile("TRAIN-B")],
        "validation": [_profile("VALID-A")],
        "sealed_final_test_commitment": {"sha256": "c" * 64, "root_count": 2},
        "not_admitted_today": [{"candidate": "SKIP-ME", "reason_code": "INCOMPLETE"}],
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    split = load_development_split(path)
    assert split.train_roots == ("TRAIN-A", "TRAIN-B")
    assert split.validation_roots == ("VALID-A",)
    assert split.candidate_count("train") == 2
    assert split.candidate_count("validation") == 1
    assert split.sealed_final_test.sha256 == "c" * 64
    assert split.sealed_final_test.root_count == 2
    assert split.not_admitted == ("SKIP-ME",)
    assert len(split.split_sha256) == 64


def test_rejects_final_test_identity_surface(tmp_path: Path) -> None:
    payload = {
        "schema": SPLIT_SCHEMA,
        "train": [_profile("TRAIN-A")],
        "validation": [_profile("VALID-A")],
        "final_test": [_profile("SECRET")],
        "sealed_final_test_commitment": {"sha256": "c" * 64, "root_count": 2},
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetError, match="Final Test"):
        load_development_split(path)


def test_rejects_train_val_overlap(tmp_path: Path) -> None:
    payload = {
        "schema": SPLIT_SCHEMA,
        "train": [_profile("SAME")],
        "validation": [_profile("SAME")],
        "sealed_final_test_commitment": {"sha256": "c" * 64, "root_count": 2},
    }
    path = tmp_path / "split.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(DatasetError, match="duplicate"):
        load_development_split(path)


def test_packaged_v004_day1_split() -> None:
    split = load_packaged_v004_day1_split(repo_root=REPO, require_v004_pilot_roots=True)
    assert split.train_roots == TRAIN_ROOTS
    assert split.validation_roots == VALIDATION_ROOTS
    assert split.sealed_final_test.sha256 == SEALED_FINAL_TEST_COMMITMENT_SHA256
    assert split.sealed_final_test.root_count == 2
    # Pilot evidence: 3+2 roots — counts come from file, not a magic global
    assert split.candidate_count("train") == 3
    assert split.candidate_count("validation") == 2
