"""Load mindmap/V004 development Train+Validation split without Final Test identities.

Authority:
  - mindmap.md steps 0–1 (molecular_root split; Train ∩ Val ∩ Test = ∅)
  - V004 day1 development split schema (opaque sealed Final Test commitment only)

Does not open Final Test identities or payloads.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, cast

from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.io_util import load_json_object, sha256_bytes
from nhc_deprot.data.paths import (
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    SEALED_FINAL_TEST_ROOT_COUNT,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)

# V004 day1 frozen schema id (evidence binding; not a science free variable)
SPLIT_SCHEMA: Final = "phase9b-aimnet2-tvt-day1-development-split-v001"
DEVELOPMENT_SPLITS: Final = ("train", "validation")
FORBIDDEN_IDENTITY_KEYS: Final = ("final_test", "test")


@dataclass(frozen=True, slots=True)
class SealedFinalTestCommitment:
    sha256: str
    root_count: int


@dataclass(frozen=True, slots=True)
class DevelopmentSplit:
    """Development-visible Train/Validation roots only."""

    schema: str
    assignments: dict[str, str]  # candidate -> train|validation
    profiles: dict[str, dict[str, Any]]
    split_sha256: str
    sealed_final_test: SealedFinalTestCommitment
    not_admitted: tuple[str, ...]
    source_path: Path | None = None

    @property
    def train_roots(self) -> tuple[str, ...]:
        return tuple(
            candidate for candidate, split in self.assignments.items() if split == "train"
        )

    @property
    def validation_roots(self) -> tuple[str, ...]:
        return tuple(
            candidate for candidate, split in self.assignments.items() if split == "validation"
        )

    def candidate_count(self, split: str) -> int:
        if split not in DEVELOPMENT_SPLITS:
            raise DatasetError(f"unknown development split: {split}")
        return sum(1 for owner in self.assignments.values() if owner == split)

    def assert_disjoint(self) -> None:
        train = set(self.train_roots)
        val = set(self.validation_roots)
        if train & val:
            raise DatasetError(f"train/validation root overlap: {sorted(train & val)}")
        if not train or not val:
            raise DatasetError("development split requires non-empty train and validation")


def _hex_sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise DatasetError(f"{label} is not a SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise DatasetError(f"{label} is not hexadecimal") from exc
    return value.lower()


def _parse_commitment(payload: Mapping[str, Any]) -> SealedFinalTestCommitment:
    commitment = payload.get("sealed_final_test_commitment")
    if not isinstance(commitment, dict) or set(commitment) != {"sha256", "root_count"}:
        raise DatasetError("Final Test commitment must be {sha256, root_count} only")
    digest = _hex_sha256(commitment.get("sha256"), label="sealed_final_test_commitment.sha256")
    count = commitment.get("root_count")
    if type(count) is not int or count <= 0:
        raise DatasetError("Final Test root_count must be a positive int")
    return SealedFinalTestCommitment(sha256=digest, root_count=count)


def load_development_split(
    path: Path,
    *,
    expected_schema: str = SPLIT_SCHEMA,
    require_v004_pilot_roots: bool = False,
) -> DevelopmentSplit:
    """Load Train/Validation identities; reject any Final Test identity surface.

    Counts and root lists are taken from the file (parameterized). Optional
    ``require_v004_pilot_roots`` only checks against the known day1 pilot set
    for binding audits — it is not a global science constant.
    """

    payload, raw = load_json_object(path)
    if payload.get("schema") != expected_schema:
        raise DatasetError(
            f"development split schema mismatch: expected {expected_schema!r}, "
            f"got {payload.get('schema')!r}"
        )
    for key in FORBIDDEN_IDENTITY_KEYS:
        if key in payload:
            raise DatasetError("development split exposes Final Test identities")

    assignments: dict[str, str] = {}
    profiles: dict[str, dict[str, Any]] = {}
    for split in DEVELOPMENT_SPLITS:
        candidates = payload.get(split)
        if not isinstance(candidates, list) or not candidates:
            raise DatasetError(f"development split is empty: {split}")
        for value in candidates:
            if not isinstance(value, dict):
                raise DatasetError(f"{split} profile is not an object")
            profile = cast(dict[str, Any], value)
            candidate = profile.get("candidate")
            if not isinstance(candidate, str) or not candidate:
                raise DatasetError(f"{split} candidate is invalid")
            if candidate in assignments:
                raise DatasetError(f"duplicate candidate across splits: {candidate}")
            assignments[candidate] = split
            profiles[candidate] = profile

    not_admitted: list[str] = []
    raw_not = payload.get("not_admitted_today", [])
    if raw_not is None:
        raw_not = []
    if not isinstance(raw_not, list):
        raise DatasetError("not_admitted_today must be a list")
    for item in raw_not:
        if isinstance(item, dict) and isinstance(item.get("candidate"), str):
            not_admitted.append(str(item["candidate"]))
        elif isinstance(item, str):
            not_admitted.append(item)

    result = DevelopmentSplit(
        schema=str(payload["schema"]),
        assignments=assignments,
        profiles=profiles,
        split_sha256=sha256_bytes(raw),
        sealed_final_test=_parse_commitment(payload),
        not_admitted=tuple(not_admitted),
        source_path=path.resolve(),
    )
    result.assert_disjoint()

    if require_v004_pilot_roots:
        if tuple(result.train_roots) != TRAIN_ROOTS:
            raise DatasetError("V004 pilot train roots mismatch")
        if tuple(result.validation_roots) != VALIDATION_ROOTS:
            raise DatasetError("V004 pilot validation roots mismatch")
        if result.sealed_final_test.sha256 != SEALED_FINAL_TEST_COMMITMENT_SHA256:
            raise DatasetError("V004 sealed Final Test commitment SHA256 mismatch")
        if result.sealed_final_test.root_count != SEALED_FINAL_TEST_ROOT_COUNT:
            raise DatasetError("V004 sealed Final Test root_count mismatch")

    return result


def load_packaged_v004_day1_split(
    *,
    repo_root: Path | None = None,
    require_v004_pilot_roots: bool = True,
) -> DevelopmentSplit:
    """Load the pilot day1 development split (NHC0801 evidence path preferred)."""

    root = repo_root or Path(__file__).resolve().parents[3]
    candidates = (
        root / "docs" / "evidence" / "pilot_day1" / "DEVELOPMENT_SPLIT.json",
        root
        / "docs"
        / "extracted"
        / "v004"
        / "PHASE9B_AIMNET2_TVT_DAY1_DEVELOPMENT_SPLIT_V001.json",
    )
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        raise DatasetError(
            "packaged day1 split missing (tried docs/evidence/pilot_day1/ and extracted/v004/)"
        )
    return load_development_split(path, require_v004_pilot_roots=require_v004_pilot_roots)
