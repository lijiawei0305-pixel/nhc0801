"""Frozen two-body D3(BJ) projection over teacher frames (mindmap step 2 residual path).

Dry-run default: derive synthetic D3 components from teacher frames and write
immutable-style receipts under ``generation/d3/``. Never silently recompute on
read — consumers must load these receipts.

Live path: inject a ``D3Projector`` (typically :class:`Dftd3Projector` via
simple-dftd3). ``dry_run=False`` without an injected projector fails closed.

**D3(BJ) two-body terms are pure geometry analytic functions.** Any teacher frame
(including intermediate geomeTRIC evaluations from full-trajectory capture) can
be projected at **zero DFT cost** — no SCF, no parent energy recompute.
``d3_recomputation_performed`` stays ``false`` (parent total energy is not
recomputed; we only evaluate the two-body D3 component from geometry).
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

# Frozen dispersion_identity (must match parent P01 / weighted_dataset target_definition)
D3_FUNCTIONAL: Final = "wb97m"
D3_DAMPING: Final = "d3bj"
D3_ATM: Final = False
D3_BACKEND_NAME: Final = "simple-dftd3"

# simple-dftd3 Structure positions are Bohr; teacher frames store Angstrom.
BOHR_TO_ANGSTROM: Final = 0.529177210903
ANGSTROM_TO_BOHR: Final = 1.0 / BOHR_TO_ANGSTROM

# NHC-relevant Z table (same coverage as weighted_dataset_writer)
_ATOMIC_NUMBERS: Final = {
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


class D3ProjectionError(RuntimeError):
    """D3 projection failed closed."""


class D3Projector(Protocol):
    def project_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        """Return d3_energy_hartree and d3_gradient_hartree_per_bohr."""
        ...


def _dftd3_backend_version() -> str:
    try:
        from importlib.metadata import version as pkg_version

        return str(pkg_version("dftd3"))
    except Exception:
        try:
            import dftd3

            return str(getattr(dftd3, "__version__", "unknown"))
        except Exception:
            return "unknown"


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
                "atm": D3_ATM,
                "damping": D3_DAMPING,
                "functional": D3_FUNCTIONAL,
                "dry_run": True,
            },
            "d3_two_body_computed_by": "dry_run_synthetic",
            "d3_backend_version": "n/a",
        }


@dataclass
class Dftd3Projector:
    """Live two-body D3(BJ) projector via simple-dftd3 / ``dftd3`` Python API.

    Parameters are pinned to frozen ``dispersion_identity``:
    ``functional="wb97m"``, ``damping="d3bj"``, ``atm=False``.

    D3(BJ) two-body is a pure geometry analytic function — any frame can be
    projected at zero DFT cost (no parent SCF / no parent total recompute).
    """

    functional: str = D3_FUNCTIONAL
    damping: str = D3_DAMPING
    atm: bool = D3_ATM

    def project_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        if (
            self.functional != D3_FUNCTIONAL
            or self.damping != D3_DAMPING
            or self.atm is not D3_ATM
        ):
            raise D3ProjectionError(
                "dispersion_identity mismatch: require "
                f"functional={D3_FUNCTIONAL!r} damping={D3_DAMPING!r} atm={D3_ATM}, "
                f"got functional={self.functional!r} damping={self.damping!r} "
                f"atm={self.atm}"
            )
        try:
            import numpy as np
            from dftd3.interface import DispersionModel, RationalDampingParam
        except ImportError as exc:
            raise D3ProjectionError(
                "simple-dftd3 (package dftd3) is required for live D3 projection"
            ) from exc

        elements = frame.get("elements")
        coords = frame.get("coordinates_angstrom")
        if not isinstance(elements, list) or not elements:
            raise D3ProjectionError("teacher frame missing elements")
        if not isinstance(coords, list) or not coords:
            raise D3ProjectionError("teacher frame missing coordinates_angstrom")
        if len(elements) != len(coords):
            raise D3ProjectionError("elements/coordinates length mismatch")

        numbers: list[int] = []
        for el in elements:
            z = _ATOMIC_NUMBERS.get(str(el))
            if z is None:
                raise D3ProjectionError(f"unsupported element for D3: {el}")
            numbers.append(z)

        positions_bohr = np.asarray(coords, dtype=float) * ANGSTROM_TO_BOHR
        numbers_arr = np.asarray(numbers, dtype="i4")
        # RationalDampingParam == D3(BJ); atm=False keeps two-body only.
        model = DispersionModel(numbers=numbers_arr, positions=positions_bohr)
        param = RationalDampingParam(method=self.functional, atm=self.atm)
        result = model.get_dispersion(param, grad=True)

        energy = float(np.asarray(result["energy"]).reshape(-1)[0])
        gradient = np.asarray(result["gradient"], dtype=float).reshape(-1, 3)
        if gradient.shape[0] != len(numbers):
            raise D3ProjectionError("D3 gradient atom count mismatch")

        return {
            "d3_energy_hartree": energy,
            "d3_gradient_hartree_per_bohr": gradient.tolist(),
            "dispersion_identity": {
                "atm": self.atm,
                "damping": self.damping,
                "functional": self.functional,
                "dry_run": False,
            },
            "d3_two_body_computed_by": D3_BACKEND_NAME,
            "d3_backend_version": _dftd3_backend_version(),
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
    provenance_by: str | None = None
    provenance_ver: str | None = None

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

        computed_by = proj.get("d3_two_body_computed_by")
        backend_ver = proj.get("d3_backend_version")
        if provenance_by is None and computed_by is not None:
            provenance_by = str(computed_by)
            provenance_ver = str(backend_ver) if backend_ver is not None else None

        source_sha = sha256_bytes(raw)
        receipt_body = {
            "schema": D3_RECEIPT_SCHEMA,
            "dry_run": dry_run,
            "live_chemistry": not dry_run,
            # Parent total energy is never recomputed; only two-body D3 from geometry.
            "d3_recomputation_performed": False,
            "external_d3_required_at_inference": True,
            "d3_two_body_computed_by": computed_by,
            "d3_backend_version": backend_ver,
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
        "d3_two_body_computed_by": provenance_by,
        "d3_backend_version": provenance_ver,
        "receipts": receipts,
    }
    write_json(out_dir / "manifest.json", endpoint_manifest, overwrite=overwrite)
    return {
        "root_id": root_id,
        "endpoint": endpoint,
        "frame_count": len(receipts),
        "receipts": receipts,
        "d3_two_body_computed_by": provenance_by,
        "d3_backend_version": provenance_ver,
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
    """Project all endpoints for listed roots (default pilot train+val).

    Live chemistry (``dry_run=False``) requires an injected projector — typically
    :class:`Dftd3Projector`. Real dftd3 is not invoked unless that projector runs.
    """

    if not dry_run and projector is None:
        raise D3ProjectionError(
            "live D3 requires an injected projector "
            "(use Dftd3Projector for simple-dftd3 / dftd3)"
        )

    roots = list(
        root_ids
        or (list(train_roots or TRAIN_ROOTS) + list(validation_roots or VALIDATION_ROOTS))
    )
    eng: D3Projector = projector or DryRunD3Projector()
    endpoint_rows: list[dict[str, Any]] = []
    total_frames = 0
    campaign_by: str | None = None
    campaign_ver: str | None = None
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
            if campaign_by is None:
                raw_by = row.get("d3_two_body_computed_by")
                raw_ver = row.get("d3_backend_version")
                campaign_by = str(raw_by) if raw_by is not None else None
                campaign_ver = str(raw_ver) if raw_ver is not None else None
            endpoint_rows.append(row)
            total_frames += int(row["frame_count"])

    if dry_run:
        status = "DRY_RUN_D3_PASS"
        notes = [
            "synthetic D3 for path/contract tests only",
            "consumers must set d3_recomputation_performed=false",
        ]
    else:
        status = "LIVE_D3_PASS"
        notes = [
            "two-body D3(BJ) via injected projector (simple-dftd3 when Dftd3Projector)",
            "d3_recomputation_performed=false (parent total energy not recomputed)",
            "D3(BJ) two-body is pure geometry analytic; zero DFT cost per frame",
        ]

    campaign = {
        "schema": D3_CAMPAIGN_SCHEMA,
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "live_chemistry": not dry_run,
        "d3_recomputation_performed": False,
        "external_d3_required_at_inference": True,
        "d3_two_body_computed_by": campaign_by,
        "d3_backend_version": campaign_ver,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "root_count": len(roots),
        "endpoint_count": len(endpoint_rows),
        "frame_count": total_frames,
        "endpoints": endpoint_rows,
        "status": status,
        "notes": notes,
    }
    write_json(layout.d3_dir / "campaign_receipt.json", campaign, overwrite=True)
    write_json(layout.logs_dir / "d3_campaign_receipt.json", campaign, overwrite=True)
    return campaign
