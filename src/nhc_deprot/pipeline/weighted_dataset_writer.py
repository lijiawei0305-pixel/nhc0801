"""Build weighted development NPZ from teacher + frozen D3 receipts (dry-run OK).

Mindmap training path after teacher frames:
  teacher frames + frozen D3 → residual E/F → sample_weight → NPZ groups
  under generation datasets/weighted/.

No live training. Final Test identities never included.
"""

from __future__ import annotations

import io
import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

import numpy as np

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.io_util import load_json_object, sha256_bytes, write_json
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.data.weight_policy import assign_candidate_endpoint_weights
from nhc_deprot.data.weighted_dataset import REQUIRED_ARRAYS, audit_weighted_dataset
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.d3_projection import D3_RECEIPT_SCHEMA
from nhc_deprot.pipeline.teacher_runner import FRAME_SCHEMA
from nhc_deprot.training.weighted_loss import SAMPLE_WEIGHT_KEY, WEIGHTING_POLICY

OUTPUT_MANIFEST_SCHEMA: Final = "nhc0801-development-dataset-v1"
CAMPAIGN_SCHEMA: Final = "nhc0801-weighted-dataset-campaign-v1"
ENDPOINTS: Final = ("cation", "neutral")
SPLITS: Final = ("train", "validation")

# Optional unit conversion for model-facing arrays (hartree → eV style scale)
HARTREE_TO_EV: Final = 27.211386245988
BOHR_TO_ANGSTROM: Final = 0.529177210903
FORCE_H_PER_B_TO_EV_PER_A: Final = HARTREE_TO_EV / BOHR_TO_ANGSTROM

ATOMIC_NUMBERS: Final = {
    "H": 1,
    "B": 5,
    "C": 6,
    "N": 7,
    "O": 8,
    "F": 9,
    "Si": 14,
    "P": 15,
    "S": 16,
    "Cl": 17,
}


class WeightedDatasetWriterError(RuntimeError):
    """Weighted dataset assembly failed closed."""


@dataclass
class FrameRecord:
    candidate: str
    endpoint: str
    frame_index: int
    split: str
    coord: np.ndarray
    numbers: np.ndarray
    charge: float
    energy: float  # short-range (training target)
    forces: np.ndarray
    total_energy: float
    total_forces: np.ndarray
    d3_energy: float
    d3_forces: np.ndarray
    sample_weight: float = 0.0


def _element_numbers(elements: Sequence[str]) -> np.ndarray:
    nums: list[int] = []
    for el in elements:
        if el not in ATOMIC_NUMBERS:
            raise WeightedDatasetWriterError(f"unsupported element for dry-run NPZ: {el}")
        nums.append(ATOMIC_NUMBERS[el])
    return np.asarray(nums, dtype=np.int64)


def load_frame_record(
    *,
    teacher_path: Path,
    d3_path: Path,
    split: str,
) -> FrameRecord:
    teacher, teacher_raw = load_json_object(teacher_path)
    d3, _ = load_json_object(d3_path)
    if teacher.get("schema") != FRAME_SCHEMA:
        raise WeightedDatasetWriterError(f"bad teacher schema: {teacher_path.name}")
    if d3.get("schema") != D3_RECEIPT_SCHEMA:
        raise WeightedDatasetWriterError(f"bad D3 schema: {d3_path.name}")
    if d3.get("d3_recomputation_performed") is not False:
        raise WeightedDatasetWriterError("D3 receipt must not recompute")
    if d3.get("external_d3_required_at_inference") is not True:
        raise WeightedDatasetWriterError("external D3 required flag missing")
    if d3.get("parent_protocol_sha256") != PROTOCOL_SHA256:
        raise WeightedDatasetWriterError("D3 protocol mismatch")
    if d3.get("source_frame_sha256") != sha256_bytes(teacher_raw):
        raise WeightedDatasetWriterError("D3 source_frame_sha256 binding mismatch")

    root_id = str(teacher["root_id"])
    endpoint = str(teacher["endpoint"])
    if d3.get("candidate") != root_id or d3.get("endpoint") != endpoint:
        raise WeightedDatasetWriterError("D3/teacher identity mismatch")
    if int(d3["frame_index"]) != int(teacher["frame_index"]):
        raise WeightedDatasetWriterError("frame_index mismatch")

    elements = cast(list[str], teacher["elements"])
    coord = np.asarray(teacher["coordinates_angstrom"], dtype=np.float32)
    numbers = _element_numbers(elements)
    if coord.shape[0] != numbers.shape[0]:
        raise WeightedDatasetWriterError("coord/numbers length mismatch")

    # Residual short-range targets from frozen D3 receipt
    short_e = float(d3["short_range_energy_hartree"]) * HARTREE_TO_EV
    total_e = float(d3["total_energy_hartree"]) * HARTREE_TO_EV
    d3_e = float(d3["d3_energy_hartree"]) * HARTREE_TO_EV
    short_f = (
        -np.asarray(d3["short_range_gradient_hartree_per_bohr"], dtype=np.float64)
        * FORCE_H_PER_B_TO_EV_PER_A
    ).astype(np.float32)
    total_g = np.asarray(d3["total_gradient_hartree_per_bohr"], dtype=np.float64)
    d3_g = np.asarray(d3["d3_gradient_hartree_per_bohr"], dtype=np.float64)
    total_f = (-total_g * FORCE_H_PER_B_TO_EV_PER_A).astype(np.float32)
    d3_f = (-d3_g * FORCE_H_PER_B_TO_EV_PER_A).astype(np.float32)

    if not math.isclose(short_e + d3_e, total_e, rel_tol=0.0, abs_tol=1e-8):
        raise WeightedDatasetWriterError("energy reconstruction failed in eV convert")

    return FrameRecord(
        candidate=root_id,
        endpoint=endpoint,
        frame_index=int(teacher["frame_index"]),
        split=split,
        coord=coord,
        numbers=numbers,
        charge=float(teacher["charge"]),
        energy=short_e,
        forces=short_f,
        total_energy=total_e,
        total_forces=total_f,
        d3_energy=d3_e,
        d3_forces=d3_f,
    )


def collect_split_records(
    layout: GenerationLayout,
    *,
    split: str,
    root_ids: Sequence[str],
) -> list[dict[str, Any]]:
    """Load mutable dict records for weight assignment."""

    if split not in SPLITS:
        raise WeightedDatasetWriterError(f"invalid split: {split}")
    records: list[dict[str, Any]] = []
    for root_id in root_ids:
        for endpoint in ENDPOINTS:
            t_dir = layout.teacher_endpoint_dir(root_id, endpoint)
            d_dir = layout.d3_dir / root_id / endpoint
            t_frames = sorted(t_dir.glob("frame_*.json"))
            if not t_frames:
                raise WeightedDatasetWriterError(f"no teacher frames: {t_dir}")
            for t_path in t_frames:
                d_path = d_dir / t_path.name
                if not d_path.is_file():
                    raise WeightedDatasetWriterError(f"missing D3 receipt: {d_path}")
                fr = load_frame_record(teacher_path=t_path, d3_path=d_path, split=split)
                records.append(
                    {
                        "candidate": fr.candidate,
                        "endpoint": fr.endpoint,
                        "frame_index": fr.frame_index,
                        "split": fr.split,
                        "coord": fr.coord,
                        "numbers": fr.numbers,
                        "charge": fr.charge,
                        "energy": fr.energy,
                        "forces": fr.forces,
                        "total_energy": fr.total_energy,
                        "total_forces": fr.total_forces,
                        "d3_energy": fr.d3_energy,
                        "d3_forces": fr.d3_forces,
                    }
                )
    return records


def _records_to_npz(records: Sequence[Mapping[str, Any]]) -> bytes:
    arrays: dict[str, Any] = {
        "coord": np.stack([cast(np.ndarray, r["coord"]) for r in records]).astype(np.float32),
        "numbers": np.stack([cast(np.ndarray, r["numbers"]) for r in records]).astype(np.int64),
        "charge": np.asarray([r["charge"] for r in records], dtype=np.float32),
        "energy": np.asarray([r["energy"] for r in records], dtype=np.float64),
        "forces": np.stack([cast(np.ndarray, r["forces"]) for r in records]).astype(np.float32),
        "total_energy": np.asarray([r["total_energy"] for r in records], dtype=np.float64),
        "total_forces": np.stack([cast(np.ndarray, r["total_forces"]) for r in records]).astype(
            np.float32
        ),
        "d3_energy": np.asarray([r["d3_energy"] for r in records], dtype=np.float64),
        "d3_forces": np.stack([cast(np.ndarray, r["d3_forces"]) for r in records]).astype(
            np.float32
        ),
        "candidate": np.asarray([r["candidate"] for r in records]),
        "endpoint": np.asarray([r["endpoint"] for r in records]),
        "frame_index": np.asarray([r["frame_index"] for r in records], dtype=np.int64),
        SAMPLE_WEIGHT_KEY: np.asarray(
            [r[SAMPLE_WEIGHT_KEY] for r in records], dtype=np.float64
        ),
    }
    if set(arrays) != REQUIRED_ARRAYS:
        raise WeightedDatasetWriterError(
            f"NPZ key set mismatch: {sorted(set(arrays) ^ REQUIRED_ARRAYS)}"
        )
    buf = io.BytesIO()
    np.savez_compressed(buf, **arrays)
    return buf.getvalue()


def assemble_weighted_dataset(
    *,
    layout: GenerationLayout,
    train_roots: Sequence[str] | None = None,
    validation_roots: Sequence[str] | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
    run_audit: bool = True,
) -> dict[str, Any]:
    """Write train/validation NPZ groups + manifest under layout.datasets_dir."""

    train = list(train_roots or TRAIN_ROOTS)
    val = list(validation_roots or VALIDATION_ROOTS)
    if set(train) & set(val):
        raise WeightedDatasetWriterError("train/validation root overlap")

    split_roots = {"train": train, "validation": val}
    files: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}
    frame_count_by_split: dict[str, int] = {"train": 0, "validation": 0}
    weight_evidence: dict[str, list[dict[str, object]]] = {"train": [], "validation": []}

    layout.datasets_dir.mkdir(parents=True, exist_ok=True)

    for split, roots in split_roots.items():
        records = collect_split_records(layout, split=split, root_ids=roots)
        # assign weights per candidate
        by_cand: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            by_cand[str(rec["candidate"])].append(rec)
        cand_count = len(roots)
        if len(by_cand) != cand_count:
            raise WeightedDatasetWriterError(
                f"{split}: expected {cand_count} candidates, got {len(by_cand)}"
            )
        for cand in roots:
            if cand not in by_cand:
                raise WeightedDatasetWriterError(f"{split}: missing candidate frames {cand}")
            evidence = assign_candidate_endpoint_weights(
                by_cand[cand], candidate_count=cand_count
            )
            weight_evidence[split].append(cast(dict[str, object], evidence))

        # group by atom count
        groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
        for rec in records:
            n = int(cast(np.ndarray, rec["numbers"]).shape[0])
            groups[n].append(rec)

        split_dir = layout.datasets_dir / split
        split_dir.mkdir(parents=True, exist_ok=True)
        for atom_count, group in sorted(groups.items()):
            raw = _records_to_npz(group)
            rel_name = f"{atom_count:03d}.npz"
            path = split_dir / rel_name
            if path.exists() and not overwrite:
                existing = path.read_bytes()
                if existing != raw:
                    raise WeightedDatasetWriterError(f"NPZ exists and differs: {path}")
            else:
                path.write_bytes(raw)
            # Manifest paths must be relative to datasets_dir (audit joins root + path)
            rel_posix = f"{split}/{rel_name}"
            files[split].append(
                {
                    "path": rel_posix,
                    "bytes": len(raw),
                    "sha256": sha256_bytes(raw),
                    "atom_count": atom_count,
                    "frame_count": len(group),
                }
            )
            frame_count_by_split[split] += len(group)

    manifest = {
        "schema": OUTPUT_MANIFEST_SCHEMA,
        "scope": "development",
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "split_unit": "molecular_root",
        "candidate_count": len(train) + len(val),
        "frame_count_by_split": frame_count_by_split,
        "files": files,
        "model_input_units": {
            "coord": "Angstrom",
            "energy": "eV",
            "forces": "eV/Angstrom",
            "charge": "elementary_charge",
        },
        "training_keys": {
            "x": ["coord", "numbers", "charge"],
            "y": ["energy", "forces"],
            "weight": SAMPLE_WEIGHT_KEY,
        },
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
            "d3_root": str(layout.d3_dir),
            "exact_receipt_bytes_preserved": True,
            "result_outcome": (
                "DRY_RUN_D3_PASS" if dry_run else "LIVE_D3_PASS"
            ),
            "receipt_count": sum(frame_count_by_split.values()),
        },
        "candidate_endpoint_weighting": {
            "policy": WEIGHTING_POLICY,
            "storage_key": SAMPLE_WEIGHT_KEY,
            "storage_status": "PASS",
            "splits": weight_evidence,
        },
        "sealed_final_test_commitment": {
            "sha256": "834f973954064565aa857e8d8c563d110d0f6256c99e54fc3283dc428efa6975",
            "root_count": 2,
        },
        "final_test_identity_accessed": False,
        "final_test_payload_present": False,
        "final_test_used_for_training": False,
        "training_started": False,
        "production_accepted": False,
        "train_roots": train,
        "validation_roots": val,
    }
    write_json(layout.datasets_dir / "manifest.json", manifest, overwrite=True)

    audit = None
    if run_audit:
        audit_obj = audit_weighted_dataset(
            layout.datasets_dir,
            expected_schema=OUTPUT_MANIFEST_SCHEMA,
            expected_frame_count_by_split=frame_count_by_split,
        )
        audit = {
            "status": audit_obj.status,
            "frame_count": audit_obj.frame_count,
            "frame_count_by_split": audit_obj.frame_count_by_split,
            "split_weight_sums": audit_obj.split_weight_sums,
            "candidate_count_by_split": audit_obj.candidate_count_by_split,
        }

    campaign = {
        "schema": CAMPAIGN_SCHEMA,
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "status": (
            "DRY_RUN_WEIGHTED_DATASET_PASS"
            if dry_run
            else "LIVE_WEIGHTED_DATASET_PASS"
        ),
        "frame_count_by_split": frame_count_by_split,
        "frame_count": sum(frame_count_by_split.values()),
        "d3_recomputation_performed": False,
        "training_started": False,
        "audit": audit,
        "manifest_path": str(layout.datasets_dir / "manifest.json"),
    }
    write_json(layout.datasets_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "weighted_dataset_campaign_receipt.json", campaign, overwrite=True)
    return campaign
