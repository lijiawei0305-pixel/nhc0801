"""Parameterized reader / auditor for V004-style weighted development NPZ datasets.

Goals (mindmap training data layer, no live train):
  - resolve weighted-dataset root (server path convention)
  - load manifest.json and audit schema / scope / Final Test seals
  - load NPZ groups and verify required keys + length consistency
  - verify sample_weight sums to 1.0 per split (policy-aware)
  - refuse silent D3 recomputation (frozen-receipt mode only)
  - never open Final Test identities

Frame counts are **derived** from the dataset under audit. Pilot 235 is not a
global constant — pass optional expected counts only for binding checks.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping, cast

import numpy as np

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.io_util import load_json_object, sha256_bytes
from nhc_deprot.data.paths import DEFAULT_WJW, V004_WEIGHTED_DATASET
from nhc_deprot.data.weight_policy import ENDPOINTS, audit_split_weight_sums
from nhc_deprot.training.weighted_loss import SAMPLE_WEIGHT_KEY, WEIGHTING_POLICY

# V004 weighted development output schema (evidence binding)
MANIFEST_SCHEMA: Final = "phase9b-aimnet2-development-dataset-v004"
PUBLIC_RESULT_SCHEMA: Final = "phase9b-aimnet2-v004-weighted-dataset-public-result-v1"
INTERNAL_RESULT_SCHEMA: Final = "phase9b-aimnet2-v004-weighted-dataset-result-v1"

REQUIRED_ARRAYS: Final = frozenset(
    {
        "coord",
        "numbers",
        "charge",
        "energy",
        "forces",
        "total_energy",
        "total_forces",
        "d3_energy",
        "d3_forces",
        "candidate",
        "endpoint",
        "frame_index",
        SAMPLE_WEIGHT_KEY,
    }
)

DEVELOPMENT_SPLITS: Final = ("train", "validation")
ENERGY_RECON_ATOL: Final = 1e-10
FORCE_RECON_ATOL: Final = 1e-5
WEIGHT_ABS_TOL: Final = 1e-12


@dataclass(frozen=True, slots=True)
class NpzGroupAudit:
    path: Path
    frame_count: int
    atom_count: int | None
    sha256: str
    bytes: int


@dataclass
class SplitAudit:
    frame_count: int = 0
    candidate_count: int = 0
    weight_sum: float = 0.0
    groups: list[NpzGroupAudit] = field(default_factory=list)
    weight_rows: list[dict[str, object]] = field(default_factory=list)


@dataclass
class WeightedDatasetAudit:
    dataset_root: Path
    manifest: dict[str, Any]
    manifest_sha256: str
    status: str
    frame_count_by_split: dict[str, int]
    frame_count: int
    candidate_count_by_split: dict[str, int]
    split_weight_sums: dict[str, float]
    splits: dict[str, SplitAudit]
    d3_recomputation_performed: bool
    final_test_payload_present: bool
    training_started: bool
    parent_protocol_sha256: str
    notes: list[str] = field(default_factory=list)


def default_v004_weighted_dataset_root(wjw: Path | None = None) -> Path:
    """Server path for the frozen V004 weighted development product (read-only)."""

    return (wjw or DEFAULT_WJW) / V004_WEIGHTED_DATASET


def load_manifest(dataset_root: Path) -> tuple[dict[str, Any], bytes]:
    path = dataset_root / "manifest.json"
    if not path.is_file():
        raise DatasetError(f"weighted dataset missing manifest.json: {dataset_root}")
    return load_json_object(path)


def _require_bool_false(payload: Mapping[str, Any], key: str) -> None:
    if payload.get(key) is not False:
        raise DatasetError(f"manifest requires {key}=false")


def _audit_manifest_identity(
    manifest: Mapping[str, Any],
    *,
    expected_schema: str = MANIFEST_SCHEMA,
    expected_parent_protocol_sha256: str = PROTOCOL_SHA256,
) -> None:
    if manifest.get("schema") != expected_schema:
        raise DatasetError(
            f"manifest schema mismatch: expected {expected_schema!r}, "
            f"got {manifest.get('schema')!r}"
        )
    if manifest.get("scope") != "development":
        raise DatasetError("weighted dataset scope must be development")
    if manifest.get("parent_protocol_sha256") != expected_parent_protocol_sha256:
        raise DatasetError("parent protocol SHA256 mismatch")
    for key in ("final_test", "test"):
        if key in manifest:
            raise DatasetError("manifest exposes Final Test identities")
    _require_bool_false(manifest, "final_test_identity_accessed")
    _require_bool_false(manifest, "final_test_payload_present")
    _require_bool_false(manifest, "final_test_used_for_training")
    _require_bool_false(manifest, "training_started")

    target = manifest.get("target_definition")
    if not isinstance(target, dict):
        raise DatasetError("manifest target_definition missing")
    if target.get("d3_recomputation_performed") is not False:
        raise DatasetError("D3 recomputation must be false (frozen receipts only)")
    if target.get("external_d3_required_at_inference") is not True:
        raise DatasetError("external D3 required at inference must be true")

    binding = manifest.get("frozen_d3_projection_binding")
    if not isinstance(binding, dict):
        raise DatasetError("manifest frozen_d3_projection_binding missing")
    if binding.get("exact_receipt_bytes_preserved") is not True:
        raise DatasetError("frozen D3 receipts must be byte-preserved")

    weighting = manifest.get("candidate_endpoint_weighting")
    if isinstance(weighting, dict) and weighting.get("policy") not in (None, WEIGHTING_POLICY):
        raise DatasetError(f"unexpected weighting policy: {weighting.get('policy')!r}")
    if isinstance(weighting, dict) and weighting.get("storage_key") not in (
        None,
        SAMPLE_WEIGHT_KEY,
    ):
        raise DatasetError(f"unexpected weight storage key: {weighting.get('storage_key')!r}")

    commitment = manifest.get("sealed_final_test_commitment")
    if not isinstance(commitment, dict) or set(commitment) != {"sha256", "root_count"}:
        raise DatasetError("manifest sealed Final Test commitment invalid")


def _verify_file_receipt(
    receipt: Mapping[str, Any],
    *,
    dataset_root: Path,
) -> tuple[Path, bytes]:
    raw_path = receipt.get("path")
    if not isinstance(raw_path, str) or not raw_path:
        raise DatasetError("NPZ receipt missing path")
    path = Path(raw_path)
    if not path.is_absolute():
        path = dataset_root / path
    path = path.resolve()
    root = dataset_root.resolve()
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise DatasetError(f"NPZ path escaped dataset root: {path}") from exc
    if not path.is_file():
        raise DatasetError(f"NPZ missing: {path}")
    raw = path.read_bytes()
    expected_sha = receipt.get("sha256")
    expected_bytes = receipt.get("bytes")
    if expected_sha is not None and sha256_bytes(raw) != expected_sha:
        raise DatasetError(f"NPZ SHA256 mismatch: {path.name}")
    if expected_bytes is not None and len(raw) != expected_bytes:
        raise DatasetError(f"NPZ size mismatch: {path.name}")
    return path, raw


def _load_npz_arrays(raw: bytes) -> dict[str, np.ndarray]:
    archive = np.load(io.BytesIO(raw), allow_pickle=False)
    try:
        keys = set(archive.files)
        if keys != REQUIRED_ARRAYS:
            missing = sorted(REQUIRED_ARRAYS - keys)
            extra = sorted(keys - REQUIRED_ARRAYS)
            raise DatasetError(
                f"NPZ array key set mismatch missing={missing} extra={extra}"
            )
        arrays = {key: archive[key] for key in REQUIRED_ARRAYS}
    finally:
        archive.close()
    return arrays


def _audit_npz_arrays(arrays: Mapping[str, np.ndarray]) -> int:
    count = int(len(arrays[SAMPLE_WEIGHT_KEY]))
    if count <= 0:
        raise DatasetError("NPZ has zero frames")
    for key in REQUIRED_ARRAYS:
        if len(arrays[key]) != count:
            raise DatasetError(f"NPZ array length mismatch on {key}")
    weights = np.asarray(arrays[SAMPLE_WEIGHT_KEY], dtype=np.float64)
    if not np.isfinite(weights).all() or (weights <= 0).any():
        raise DatasetError("sample_weight must be finite and positive")
    if not np.allclose(
        arrays["energy"] + arrays["d3_energy"],
        arrays["total_energy"],
        rtol=0.0,
        atol=ENERGY_RECON_ATOL,
    ):
        raise DatasetError("energy reconstruction failed (E_short + E_D3 != E_total)")
    if not np.allclose(
        arrays["forces"] + arrays["d3_forces"],
        arrays["total_forces"],
        rtol=0.0,
        atol=FORCE_RECON_ATOL,
    ):
        raise DatasetError("force reconstruction failed (F_short + F_D3 != F_total)")
    return count


def audit_weighted_dataset(
    dataset_root: Path,
    *,
    expected_schema: str = MANIFEST_SCHEMA,
    expected_parent_protocol_sha256: str = PROTOCOL_SHA256,
    expected_frame_count_by_split: Mapping[str, int] | None = None,
    expected_candidate_count_by_split: Mapping[str, int] | None = None,
    expected_total_frame_count: int | None = None,
    check_path_exists_only: bool = False,
) -> WeightedDatasetAudit:
    """Audit a weighted development dataset root.

    When ``check_path_exists_only`` is True, only verifies root + manifest presence
    and identity fields (useful when NPZ files are remote-not-mounted). Default
    mode loads every NPZ listed in the manifest.
    """

    root = dataset_root.resolve()
    if not root.is_dir():
        raise DatasetError(f"weighted dataset root does not exist: {root}")
    if (root / "final_test").exists() or (root / "test").exists():
        raise DatasetError("weighted dataset directory exposes Final Test")

    manifest, manifest_raw = load_manifest(root)
    _audit_manifest_identity(
        manifest,
        expected_schema=expected_schema,
        expected_parent_protocol_sha256=expected_parent_protocol_sha256,
    )

    notes: list[str] = []
    split_audits: dict[str, SplitAudit] = {split: SplitAudit() for split in DEVELOPMENT_SPLITS}
    frame_count_by_split: dict[str, int] = {split: 0 for split in DEVELOPMENT_SPLITS}
    candidate_count_by_split: dict[str, int] = {split: 0 for split in DEVELOPMENT_SPLITS}
    split_weight_sums: dict[str, float] = {split: 0.0 for split in DEVELOPMENT_SPLITS}

    files = manifest.get("files")
    if not isinstance(files, dict) or set(files) != set(DEVELOPMENT_SPLITS):
        raise DatasetError("manifest files must contain exactly train and validation")

    if check_path_exists_only:
        notes.append("NPZ content not loaded (path/schema-only mode)")
        for split in DEVELOPMENT_SPLITS:
            receipts = files[split]
            if not isinstance(receipts, list) or not receipts:
                raise DatasetError(f"manifest split empty: {split}")
            for value in receipts:
                if not isinstance(value, dict):
                    raise DatasetError("manifest receipt is not an object")
                path, _ = _verify_file_receipt(cast(dict[str, Any], value), dataset_root=root)
                atom_count = value.get("atom_count")
                frame_count = int(value.get("frame_count") or 0)
                split_audits[split].groups.append(
                    NpzGroupAudit(
                        path=path,
                        frame_count=frame_count,
                        atom_count=int(atom_count) if type(atom_count) is int else None,
                        sha256=str(value.get("sha256") or ""),
                        bytes=int(value.get("bytes") or path.stat().st_size),
                    )
                )
                frame_count_by_split[split] += frame_count
        # Prefer explicit counts from manifest when present
        declared = manifest.get("frame_count_by_split")
        if isinstance(declared, dict):
            for split in DEVELOPMENT_SPLITS:
                if split in declared:
                    frame_count_by_split[split] = int(declared[split])
        total = sum(frame_count_by_split.values())
        if expected_frame_count_by_split is not None:
            for split, expected in expected_frame_count_by_split.items():
                if frame_count_by_split.get(split) != expected:
                    raise DatasetError(
                        f"frame_count_by_split[{split}]={frame_count_by_split.get(split)} "
                        f"!= expected {expected}"
                    )
        if expected_total_frame_count is not None and total != expected_total_frame_count:
            raise DatasetError(
                f"total frame_count={total} != expected {expected_total_frame_count}"
            )
        return WeightedDatasetAudit(
            dataset_root=root,
            manifest=cast(dict[str, Any], manifest),
            manifest_sha256=sha256_bytes(manifest_raw),
            status="PATH_SCHEMA_PASS",
            frame_count_by_split=frame_count_by_split,
            frame_count=total,
            candidate_count_by_split=candidate_count_by_split,
            split_weight_sums=split_weight_sums,
            splits=split_audits,
            d3_recomputation_performed=False,
            final_test_payload_present=False,
            training_started=False,
            parent_protocol_sha256=str(manifest["parent_protocol_sha256"]),
            notes=notes,
        )

    for split in DEVELOPMENT_SPLITS:
        receipts = files[split]
        if not isinstance(receipts, list) or not receipts:
            raise DatasetError(f"manifest split empty: {split}")
        weights: dict[str, dict[str, float]] = defaultdict(lambda: defaultdict(float))
        for value in receipts:
            if not isinstance(value, dict):
                raise DatasetError("manifest receipt is not an object")
            path, raw = _verify_file_receipt(cast(dict[str, Any], value), dataset_root=root)
            arrays = _load_npz_arrays(raw)
            count = _audit_npz_arrays(arrays)
            atom_count = value.get("atom_count")
            if type(atom_count) is not int:
                # Derive from numbers if not declared
                atom_count = int(np.asarray(arrays["numbers"]).shape[-1])
            split_audits[split].groups.append(
                NpzGroupAudit(
                    path=path,
                    frame_count=count,
                    atom_count=atom_count,
                    sha256=sha256_bytes(raw),
                    bytes=len(raw),
                )
            )
            frame_count_by_split[split] += count
            for candidate, endpoint, weight in zip(
                arrays["candidate"],
                arrays["endpoint"],
                arrays[SAMPLE_WEIGHT_KEY],
                strict=True,
            ):
                endpoint_text = str(endpoint)
                if endpoint_text not in ENDPOINTS:
                    raise DatasetError(f"invalid endpoint in NPZ: {endpoint_text}")
                weights[str(candidate)][endpoint_text] += float(weight)

        candidate_count = len(weights)
        if expected_candidate_count_by_split is not None:
            expected_c = expected_candidate_count_by_split.get(split)
            if expected_c is not None and candidate_count != expected_c:
                raise DatasetError(
                    f"candidate_count[{split}]={candidate_count} != expected {expected_c}"
                )
        weight_audit = audit_split_weight_sums(
            weights_by_candidate_endpoint={
                candidate: dict(endpoint_map) for candidate, endpoint_map in weights.items()
            },
            candidate_count=candidate_count,
            abs_tol=WEIGHT_ABS_TOL,
        )
        candidate_count_by_split[split] = candidate_count
        split_weight_sums[split] = float(weight_audit["split_weight_sum"])
        split_audits[split].frame_count = frame_count_by_split[split]
        split_audits[split].candidate_count = candidate_count
        split_audits[split].weight_sum = split_weight_sums[split]
        split_audits[split].weight_rows = cast(list[dict[str, object]], weight_audit["rows"])

    # Cross-check declared manifest counts when present (parameterized, not hardcoded)
    declared = manifest.get("frame_count_by_split")
    if isinstance(declared, dict):
        for split in DEVELOPMENT_SPLITS:
            if split in declared and int(declared[split]) != frame_count_by_split[split]:
                raise DatasetError(
                    f"manifest frame_count_by_split[{split}] drifted vs NPZ content"
                )

    if expected_frame_count_by_split is not None:
        for split, expected in expected_frame_count_by_split.items():
            if frame_count_by_split.get(split) != expected:
                raise DatasetError(
                    f"frame_count_by_split[{split}]={frame_count_by_split.get(split)} "
                    f"!= expected {expected}"
                )

    total = sum(frame_count_by_split.values())
    if expected_total_frame_count is not None and total != expected_total_frame_count:
        raise DatasetError(
            f"total frame_count={total} != expected {expected_total_frame_count}"
        )

    for split, weight_sum in split_weight_sums.items():
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=WEIGHT_ABS_TOL):
            raise DatasetError(f"split {split} weight sum is {weight_sum}, expected 1.0")

    return WeightedDatasetAudit(
        dataset_root=root,
        manifest=cast(dict[str, Any], manifest),
        manifest_sha256=sha256_bytes(manifest_raw),
        status="PASS",
        frame_count_by_split=frame_count_by_split,
        frame_count=total,
        candidate_count_by_split=candidate_count_by_split,
        split_weight_sums=split_weight_sums,
        splits=split_audits,
        d3_recomputation_performed=False,
        final_test_payload_present=False,
        training_started=False,
        parent_protocol_sha256=str(manifest["parent_protocol_sha256"]),
        notes=notes,
    )


def load_split_sample_weights(
    dataset_root: Path,
    split: str,
) -> dict[str, Any]:
    """Load all sample_weight / identity columns for one split (for trainer wiring)."""

    if split not in DEVELOPMENT_SPLITS:
        raise DatasetError(f"invalid split: {split}")
    audit = audit_weighted_dataset(dataset_root)
    if audit.status != "PASS":
        raise DatasetError(f"dataset audit status {audit.status}")
    # Re-load arrays for the split (audit already validated)
    manifest = audit.manifest
    files = cast(dict[str, list[dict[str, Any]]], manifest["files"])
    candidates: list[str] = []
    endpoints: list[str] = []
    frame_indices: list[int] = []
    weights: list[float] = []
    for receipt in files[split]:
        path, raw = _verify_file_receipt(receipt, dataset_root=dataset_root.resolve())
        arrays = _load_npz_arrays(raw)
        for c, e, fi, w in zip(
            arrays["candidate"],
            arrays["endpoint"],
            arrays["frame_index"],
            arrays[SAMPLE_WEIGHT_KEY],
            strict=True,
        ):
            candidates.append(str(c))
            endpoints.append(str(e))
            frame_indices.append(int(fi))
            weights.append(float(w))
    return {
        "split": split,
        "frame_count": len(weights),
        "sample_weight": np.asarray(weights, dtype=np.float64),
        "candidate": np.asarray(candidates),
        "endpoint": np.asarray(endpoints),
        "frame_index": np.asarray(frame_indices, dtype=np.int64),
        "weight_sum": float(np.sum(weights)),
    }


def audit_public_weighted_result(
    path: Path,
    *,
    expected_generation_id: str | None = None,
) -> dict[str, Any]:
    """Audit the public weighted-dataset result JSON (counts from file, not hardcoded)."""

    payload, _ = load_json_object(path)
    schema = payload.get("schema")
    if schema not in {PUBLIC_RESULT_SCHEMA, INTERNAL_RESULT_SCHEMA}:
        raise DatasetError(f"unexpected weighted result schema: {schema!r}")
    outcome = payload.get("final_outcome")
    if outcome not in {
        "WEIGHTED_DEVELOPMENT_DATASET_PASS",
        "PASS",
    }:
        raise DatasetError(f"unexpected final_outcome: {outcome!r}")
    if payload.get("final_test_payload_read") is not False:
        raise DatasetError("result reports Final Test payload was read")
    if payload.get("training_started") is not False:
        raise DatasetError("result reports training_started")
    frozen = payload.get("frozen_d3_consumption") or payload.get("frozen_d3_projection_binding")
    if isinstance(frozen, dict) and frozen.get("d3_recomputation_performed") is not False:
        raise DatasetError("result reports D3 recomputation")
    if expected_generation_id is not None:
        if payload.get("generation_id") != expected_generation_id:
            raise DatasetError("generation_id mismatch")
    scope = payload.get("scope")
    if not isinstance(scope, dict):
        raise DatasetError("result scope missing")
    weighting = payload.get("weighting") if isinstance(payload.get("weighting"), dict) else {}
    return {
        "status": "PASS",
        "schema": schema,
        "generation_id": payload.get("generation_id"),
        "frame_count": scope.get("frame_count"),
        "train_frame_count": scope.get("train_frame_count"),
        "validation_frame_count": scope.get("validation_frame_count"),
        "train_root_count": scope.get("train_root_count"),
        "validation_root_count": scope.get("validation_root_count"),
        "weighting_policy": weighting.get("policy"),
        "train_split_weight_sum": weighting.get("train_split_weight_sum"),
        "validation_split_weight_sum": weighting.get("validation_split_weight_sum"),
    }
