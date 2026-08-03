"""Teacher-frame path conventions for **legacy V004** Pure-PySCF P01 frames.

These helpers point at historical ``$WJW/data/runs/autofill_*_v001`` trees
(read-only inventory). New NHC0801 teacher products use
``runs/nhc0801-g001/teacher_gpu_g00N/`` via generation layout — not this module.


Read-only path resolution against $WJW runs. Does not load chemistry or run DFT.
Frame *counts* are never assumed (no 235 magic number).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Final, Iterable

from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.paths import (
    DEFAULT_WJW,
    PARENT_PROTOCOL_SHA256,
    autofill_run_dir,
    frame_path,
)

ENDPOINTS: Final = ("cation", "neutral")
FRAME_GLOB: Final = "frame_*.json"


@dataclass(frozen=True, slots=True)
class TeacherFrameRef:
    candidate: str
    endpoint: str
    frame_index: int
    path: Path


def runs_root(wjw: Path | None = None) -> Path:
    return (wjw or DEFAULT_WJW) / "data" / "runs"


def list_endpoint_frame_paths(
    runs_root_path: Path,
    candidate: str,
    endpoint: str,
) -> list[Path]:
    if endpoint not in ENDPOINTS:
        raise DatasetError(f"invalid endpoint: {endpoint}")
    endpoint_dir = autofill_run_dir(runs_root_path, candidate) / "training_data" / endpoint
    if not endpoint_dir.is_dir():
        return []
    return sorted(endpoint_dir.glob(FRAME_GLOB))


def list_candidate_frame_refs(
    runs_root_path: Path,
    candidate: str,
) -> list[TeacherFrameRef]:
    """Enumerate existing frame_*.json for both endpoints (path inventory only)."""

    refs: list[TeacherFrameRef] = []
    for endpoint in ENDPOINTS:
        for path in list_endpoint_frame_paths(runs_root_path, candidate, endpoint):
            stem = path.stem  # frame_0000
            if not stem.startswith("frame_"):
                continue
            try:
                index = int(stem.split("_", 1)[1])
            except ValueError as exc:
                raise DatasetError(f"non-numeric frame name: {path.name}") from exc
            refs.append(
                TeacherFrameRef(
                    candidate=candidate,
                    endpoint=endpoint,
                    frame_index=index,
                    path=path,
                )
            )
    return refs


def expected_frame_path(
    runs_root_path: Path,
    candidate: str,
    endpoint: str,
    index: int,
) -> Path:
    return frame_path(runs_root_path, candidate, endpoint, index)


def inventory_candidates(
    runs_root_path: Path,
    candidates: Iterable[str],
) -> dict[str, dict[str, object]]:
    """Path-level inventory: counts and existence only (no JSON chemistry parse)."""

    report: dict[str, dict[str, object]] = {}
    for candidate in candidates:
        run_dir = autofill_run_dir(runs_root_path, candidate)
        refs = list_candidate_frame_refs(runs_root_path, candidate)
        by_endpoint = {
            endpoint: sum(1 for ref in refs if ref.endpoint == endpoint)
            for endpoint in ENDPOINTS
        }
        report[candidate] = {
            "run_dir": str(run_dir),
            "run_dir_exists": run_dir.is_dir(),
            "frame_count": len(refs),
            "frame_count_by_endpoint": by_endpoint,
            "parent_protocol_sha256_expected": PARENT_PROTOCOL_SHA256,
        }
    return report


def d3_receipt_path(
    projection_root: Path,
    candidate: str,
    endpoint: str,
    frame_index: int,
) -> Path:
    """Frozen D3 receipt layout under a projection product root."""

    if endpoint not in ENDPOINTS:
        raise DatasetError(f"invalid endpoint: {endpoint}")
    return projection_root / candidate / endpoint / f"frame_{frame_index:04d}.json"
