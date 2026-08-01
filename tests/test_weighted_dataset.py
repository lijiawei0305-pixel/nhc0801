"""Synthetic weighted-dataset fixture: schema + sample_weight sum = 1 (no HPC)."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.io_util import sha256_bytes
from nhc_deprot.data.weighted_dataset import (
    MANIFEST_SCHEMA,
    REQUIRED_ARRAYS,
    audit_public_weighted_result,
    audit_weighted_dataset,
    load_split_sample_weights,
)
from nhc_deprot.training.weighted_loss import SAMPLE_WEIGHT_KEY

REPO = Path(__file__).resolve().parents[1]


def _make_npz(
    *,
    candidate: str,
    endpoint: str,
    n_frames: int,
    atom_count: int,
    per_frame_weight: float,
    frame_index_start: int = 0,
) -> bytes:
    """Build one atom-count group NPZ matching V004 REQUIRED_ARRAYS."""

    coords = np.zeros((n_frames, atom_count, 3), dtype=np.float32)
    numbers = np.full((n_frames, atom_count), 6, dtype=np.int64)
    charge = np.full(n_frames, 1.0 if endpoint == "cation" else 0.0, dtype=np.float32)
    # residual short-range targets
    energy = np.full(n_frames, -10.0, dtype=np.float64)
    forces = np.zeros((n_frames, atom_count, 3), dtype=np.float32)
    d3_energy = np.full(n_frames, -0.1, dtype=np.float64)
    d3_forces = np.zeros((n_frames, atom_count, 3), dtype=np.float32)
    total_energy = energy + d3_energy
    total_forces = forces + d3_forces
    arrays: dict[str, Any] = {
        "coord": coords,
        "numbers": numbers,
        "charge": charge,
        "energy": energy,
        "forces": forces,
        "total_energy": total_energy,
        "total_forces": total_forces,
        "d3_energy": d3_energy,
        "d3_forces": d3_forces,
        "candidate": np.asarray([candidate] * n_frames),
        "endpoint": np.asarray([endpoint] * n_frames),
        "frame_index": np.arange(frame_index_start, frame_index_start + n_frames, dtype=np.int64),
        SAMPLE_WEIGHT_KEY: np.full(n_frames, per_frame_weight, dtype=np.float64),
    }
    assert set(arrays) == REQUIRED_ARRAYS
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def _write_receipt(root: Path, relative: str, raw: bytes, *, atom_count: int, frame_count: int) -> dict[str, object]:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(raw)
    return {
        "path": str(path),
        "bytes": len(raw),
        "sha256": sha256_bytes(raw),
        "atom_count": atom_count,
        "frame_count": frame_count,
    }


def _build_synthetic_dataset(tmp_path: Path) -> Path:
    """2 train roots + 1 val root; unequal frame counts; weight sums = 1 per split."""

    root = tmp_path / "weighted_dev"
    root.mkdir()
    # train: 2 candidates → each 0.5; endpoint 0.25
    # A: cation 2 frames, neutral 3 frames
    # B: cation 1 frame,  neutral 1 frame
    train_receipts = [
        _write_receipt(
            root,
            "train/002.npz",
            _make_npz(
                candidate="A",
                endpoint="cation",
                n_frames=2,
                atom_count=2,
                per_frame_weight=0.25 / 2,
            ),
            atom_count=2,
            frame_count=2,
        ),
        _write_receipt(
            root,
            "train/003.npz",
            _make_npz(
                candidate="A",
                endpoint="neutral",
                n_frames=3,
                atom_count=3,
                per_frame_weight=0.25 / 3,
            ),
            atom_count=3,
            frame_count=3,
        ),
        _write_receipt(
            root,
            "train/004.npz",
            # B both endpoints same atom count → stack into one file per endpoint
            _make_npz(
                candidate="B",
                endpoint="cation",
                n_frames=1,
                atom_count=4,
                per_frame_weight=0.25,
            ),
            atom_count=4,
            frame_count=1,
        ),
        _write_receipt(
            root,
            "train/005.npz",
            _make_npz(
                candidate="B",
                endpoint="neutral",
                n_frames=1,
                atom_count=5,
                per_frame_weight=0.25,
            ),
            atom_count=5,
            frame_count=1,
        ),
    ]
    # validation: 1 candidate → mass 1.0; endpoint 0.5
    val_receipts = [
        _write_receipt(
            root,
            "validation/002.npz",
            _make_npz(
                candidate="V",
                endpoint="cation",
                n_frames=2,
                atom_count=2,
                per_frame_weight=0.5 / 2,
            ),
            atom_count=2,
            frame_count=2,
        ),
        _write_receipt(
            root,
            "validation/003.npz",
            _make_npz(
                candidate="V",
                endpoint="neutral",
                n_frames=4,
                atom_count=3,
                per_frame_weight=0.5 / 4,
            ),
            atom_count=3,
            frame_count=4,
        ),
    ]
    train_frames = sum(int(r["frame_count"]) for r in train_receipts)
    val_frames = sum(int(r["frame_count"]) for r in val_receipts)
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "scope": "development",
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "split_sha256": "d" * 64,
        "split_unit": "molecular_root",
        "candidate_count": 3,
        "frame_count_by_split": {"train": train_frames, "validation": val_frames},
        "files": {"train": train_receipts, "validation": val_receipts},
        "target_definition": {
            "energy": "P01_total_energy_minus_frozen_two_body_D3_BJ",
            "forces": "P01_total_forces_minus_frozen_two_body_D3_BJ_forces",
            "external_d3_required_at_inference": True,
            "d3_functional": "wb97m",
            "d3_damping": "d3bj",
            "atm": False,
            "d3_recomputation_performed": False,
        },
        "frozen_d3_projection_binding": {
            "result_sha256": "e" * 64,
            "result_outcome": "D3_PROJECTION_PASS",
            "receipt_count": train_frames + val_frames,
            "exact_receipt_bytes_preserved": True,
        },
        "candidate_endpoint_weighting": {
            "policy": "equal_candidate_then_equal_endpoint_then_uniform_frames",
            "storage_key": "sample_weight",
        },
        "sealed_final_test_commitment": {"sha256": "f" * 64, "root_count": 2},
        "final_test_identity_accessed": False,
        "final_test_payload_present": False,
        "final_test_used_for_training": False,
        "training_started": False,
        "production_accepted": False,
    }
    (root / "manifest.json").write_text(
        json.dumps(manifest, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    return root


def test_audit_synthetic_weighted_dataset(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    audit = audit_weighted_dataset(root)
    assert audit.status == "PASS"
    assert audit.frame_count_by_split["train"] == 7
    assert audit.frame_count_by_split["validation"] == 6
    assert audit.frame_count == 13  # parameterized from content, not 235
    assert audit.candidate_count_by_split == {"train": 2, "validation": 1}
    assert audit.split_weight_sums["train"] == pytest.approx(1.0)
    assert audit.split_weight_sums["validation"] == pytest.approx(1.0)
    assert audit.d3_recomputation_performed is False
    assert audit.final_test_payload_present is False
    assert audit.parent_protocol_sha256 == PROTOCOL_SHA256


def test_expected_counts_optional_binding(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    with pytest.raises(DatasetError, match="frame_count"):
        audit_weighted_dataset(root, expected_total_frame_count=235)


def test_rejects_d3_recompute_flag(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    manifest_path = root / "manifest.json"
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    payload["target_definition"]["d3_recomputation_performed"] = True
    manifest_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    with pytest.raises(DatasetError, match="D3 recomputation"):
        audit_weighted_dataset(root)


def test_rejects_final_test_directory(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    (root / "final_test").mkdir()
    with pytest.raises(DatasetError, match="Final Test"):
        audit_weighted_dataset(root)


def test_load_split_sample_weights(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    train = load_split_sample_weights(root, "train")
    assert train["frame_count"] == 7
    assert train["weight_sum"] == pytest.approx(1.0)
    assert set(train["candidate"].tolist()) == {"A", "B"}


def test_path_schema_only_mode(tmp_path: Path) -> None:
    root = _build_synthetic_dataset(tmp_path)
    audit = audit_weighted_dataset(root, check_path_exists_only=True)
    assert audit.status == "PATH_SCHEMA_PASS"
    assert audit.frame_count == 13


def test_public_result_extract() -> None:
    path = REPO / "docs" / "evidence" / "pilot_day1" / "WEIGHTED_DATASET_RESULT.json"
    if not path.is_file():
        path = (
            REPO
            / "docs"
            / "extracted"
            / "v004"
            / "PHASE9B_AIMNET2_V004_WEIGHTED_DATASET_RESULT.json"
        )
    out = audit_public_weighted_result(
        path,
        expected_generation_id="phase9b-aimnet2-nhc-p01-tvt-20260801-v001",
    )
    assert out["status"] == "PASS"
    # Counts come from the evidence file (pilot 235 is evidence, not a code constant)
    assert out["frame_count"] == 235
    assert out["train_frame_count"] == 123
    assert out["validation_frame_count"] == 112
    assert out["train_split_weight_sum"] == 1.0
    assert out["validation_split_weight_sum"] == 1.0
