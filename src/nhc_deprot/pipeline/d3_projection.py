"""Frozen two-body D3(BJ) projection over teacher frames (mindmap step 2 residual path).

Dry-run default: derive synthetic D3 components from teacher frames and write
immutable-style receipts under ``generation/d3/``. Never silently recompute on
read — consumers must load these receipts.

Live PySCF dispersion backend is not wired; ``live=True`` fails closed without
an injected projector.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.io_util import canonical_json, load_json_object, sha256_bytes, write_json
from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.teacher_runner import FRAME_SCHEMA

D3_RECEIPT_SCHEMA: Final = "nhc0801-training-d3-projection-v1"
D3_CAMPAIGN_SCHEMA: Final = "nhc0801-d3-projection-campaign-v1"
ENDPOINTS: Final = ("cation", "neutral")

# Dry-run synthetic two-body D3 scale (not scientific)
DRY_RUN_D3_FRACTION: Final = 0.01


class D3ProjectionError(RuntimeError):
    """D3 projection failed closed."""


class D3Projector(Protocol):
    def project_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return d3_energy_hartree and d3_gradient_hartree_per_bohr."""
        ...


@dataclass
class DryRunD3Projector:
    """Synthetic D3: fraction of total energy + scaled gradient (not chemistry)."""

    fraction: float = DRY_RUN_D3_FRACTION

    def project_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        total_e = float(frame["energy_hartree"])
        grad = frame.get("gradient_hartree_per_bohr")
        if not isinstance(grad, list) or not grad:
            raise D3ProjectionError("teacher frame missing gradient")
        d3_e = total_e * self.fraction
        d3_g = [[float(c) * self.fraction for c in row] for row in grad]
        return {
            "d3_energy_hartree": d3_e,
            "d3_gradient_hartree_per_bohr": d3_g,
            "dispersion_identity": {
                "atm": False,
                "damping": "d3bj",
                "functional": "wb97m",
                "dry_run": True,
            },
        }


def _list_teacher_frames(endpoint_dir: Path) -> list[Path]:
    return sorted(endpoint_dir.glob("frame_*.json"))


def project_endpoint(
    *,
    layout: GenerationLayout,
    root_id: str,
    endpoint: str,
    projector: D3Projector,
    dry_run: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read teacher frames → write D3 receipts under layout.d3_dir."""

    if endpoint not in ENDPOINTS:
        raise D3ProjectionError(f"invalid endpoint: {endpoint}")
    teacher_dir = layout.teacher_endpoint_dir(root_id, endpoint)
    if not teacher_dir.is_dir():
        raise D3ProjectionError(f"missing teacher endpoint dir: {teacher_dir}")
    frames = _list_teacher_frames(teacher_dir)
    if not frames:
        raise D3ProjectionError(f"no teacher frames in {teacher_dir}")

    out_dir = layout.d3_dir / root_id / endpoint
    out_dir.mkdir(parents=True, exist_ok=True)
    receipts: list[dict[str, object]] = []

    for path in frames:
        payload, raw = load_json_object(path)
        if payload.get("schema") != FRAME_SCHEMA:
            raise D3ProjectionError(f"unexpected teacher frame schema: {path.name}")
        if payload.get("parent_protocol_sha256") != PROTOCOL_SHA256:
            raise D3ProjectionError("teacher frame protocol SHA mismatch")
        if int(payload.get("frame_index", -1)) < 0:
            raise D3ProjectionError("invalid frame_index")

        proj = projector.project_frame(payload)
        d3_e = float(proj["d3_energy_hartree"])
        d3_g = proj["d3_gradient_hartree_per_bohr"]
        total_e = float(payload["energy_hartree"])
        total_g = payload["gradient_hartree_per_bohr"]
        if not isinstance(total_g, list):
            raise D3ProjectionError("teacher gradient missing")
        short_e = total_e - d3_e
        short_g = [
            [float(t) - float(d) for t, d in zip(trow, drow, strict=True)]
            for trow, drow in zip(total_g, d3_g, strict=True)
        ]
        short_f = [[-c for c in row] for row in short_g]
        if not math.isclose(short_e + d3_e, total_e, rel_tol=0.0, abs_tol=1e-12):
            raise D3ProjectionError("energy reconstruction failed")

        source_sha = sha256_bytes(raw)
        receipt_body = {
            "schema": D3_RECEIPT_SCHEMA,
            "dry_run": dry_run,
            "live_chemistry": not dry_run,
            "d3_recomputation_performed": False,
            "external_d3_required_at_inference": True,
            "candidate": root_id,
            "endpoint": endpoint,
            "frame_index": int(payload["frame_index"]),
            "parent_protocol_sha256": PROTOCOL_SHA256,
            "source_frame_sha256": source_sha,
            "geometry_sha256": payload.get("geometry_sha256")
            or sha256_bytes(
                canonical_json(
                    {
                        "elements": payload.get("elements"),
                        "coordinates": payload.get("coordinates_angstrom"),
                    }
                )
            ),
            "total_energy_hartree": total_e,
            "d3_energy_hartree": d3_e,
            "short_range_energy_hartree": short_e,
            "total_gradient_hartree_per_bohr": total_g,
            "d3_gradient_hartree_per_bohr": d3_g,
            "short_range_gradient_hartree_per_bohr": short_g,
            "short_range_forces_hartree_per_bohr": short_f,
            "dispersion_identity": proj.get("dispersion_identity"),
            "atom_count": len(total_g),
        }
        # canonical_sha256 over body without that field
        receipt_body["canonical_sha256"] = sha256_bytes(canonical_json(receipt_body))
        out_path = out_dir / f"frame_{int(payload['frame_index']):04d}.json"
        meta = write_json(out_path, receipt_body, overwrite=overwrite)
        receipts.append(
            {
                "frame_index": int(payload["frame_index"]),
                "path": str(out_path),
                "sha256": meta["sha256"],
            }
        )

    endpoint_manifest = {
        "schema": "nhc0801-d3-endpoint-manifest-v1",
        "candidate": root_id,
        "endpoint": endpoint,
        "frame_count": len(receipts),
        "dry_run": dry_run,
        "d3_recomputation_performed": False,
        "receipts": receipts,
    }
    write_json(out_dir / "manifest.json", endpoint_manifest, overwrite=overwrite)
    return {
        "root_id": root_id,
        "endpoint": endpoint,
        "frame_count": len(receipts),
        "receipts": receipts,
    }


def run_d3_campaign(
    *,
    layout: GenerationLayout,
    root_ids: Sequence[str] | None = None,
    train_roots: Sequence[str] | None = None,
    validation_roots: Sequence[str] | None = None,
    projector: D3Projector | None = None,
    dry_run: bool = True,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Project all endpoints for listed roots (default pilot train+val)."""

    if not dry_run and projector is None:
        raise D3ProjectionError("live D3 requires an injected projector (PySCF not wired)")
    if not dry_run:
        raise D3ProjectionError(
            "live D3 projection not authorized in this skeleton; use dry_run=True"
        )

    roots = list(
        root_ids
        or (list(train_roots or TRAIN_ROOTS) + list(validation_roots or VALIDATION_ROOTS))
    )
    eng: D3Projector = projector or DryRunD3Projector()
    endpoint_rows: list[dict[str, Any]] = []
    total_frames = 0
    for root_id in roots:
        for endpoint in ENDPOINTS:
            row = project_endpoint(
                layout=layout,
                root_id=root_id,
                endpoint=endpoint,
                projector=eng,
                dry_run=dry_run,
                overwrite=overwrite,
            )
            endpoint_rows.append(row)
            total_frames += int(row["frame_count"])

    campaign = {
        "schema": D3_CAMPAIGN_SCHEMA,
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "live_chemistry": False,
        "d3_recomputation_performed": False,
        "external_d3_required_at_inference": True,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "root_count": len(roots),
        "endpoint_count": len(endpoint_rows),
        "frame_count": total_frames,
        "endpoints": endpoint_rows,
        "status": "DRY_RUN_D3_PASS",
        "notes": [
            "synthetic D3 for path/contract tests only",
            "consumers must set d3_recomputation_performed=false",
        ],
    }
    write_json(layout.d3_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "d3_campaign_receipt.json", campaign, overwrite=True)
    return campaign
