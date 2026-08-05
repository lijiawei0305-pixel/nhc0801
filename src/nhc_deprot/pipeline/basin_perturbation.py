"""Starting-geometry perturbation for basin-stability pre-screen tests.

Pure functions only — does **not** change production ranking in
``pre_screen.py``. Perturbs ``TeacherEndpointReference.start_coordinates``
in memory (frozen dataclass → ``dataclasses.replace``); never writes teacher
files on disk.

Used by the multi-start basin experiment (see
``docs/plans/20260805_grok_task_basin_perturbation_test.md``).
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

import numpy as np

from nhc_deprot.pipeline.pre_screen import TeacherEndpointReference

__all__ = [
    "PerturbationResult",
    "apply_rigid_translation",
    "perturb_start_geometry",
    "rms_displacement",
]


@dataclass(frozen=True, slots=True)
class PerturbationResult:
    """Perturbed reference plus the realized RMS atomic displacement (Å)."""

    reference: TeacherEndpointReference
    applied_rms_displacement: float
    epsilon_angstrom: float
    rng_seed: int


def rms_displacement(
    coords_a: Any,
    coords_b: Any,
) -> float:
    """RMS of per-atom Euclidean displacements (Å), no Kabsch."""

    a = np.asarray(coords_a, dtype=np.float64)
    b = np.asarray(coords_b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 2 or a.shape[1] != 3:
        raise ValueError(f"coord shape mismatch: {a.shape} vs {b.shape}")
    if a.shape[0] == 0:
        return 0.0
    d = a - b
    return float(np.sqrt(np.mean(np.sum(d * d, axis=1))))


def _as_coord_tuples(arr: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    return tuple(
        (float(row[0]), float(row[1]), float(row[2])) for row in arr
    )


def perturb_start_geometry(
    reference: TeacherEndpointReference,
    *,
    epsilon_angstrom: float,
    rng_seed: int,
) -> PerturbationResult:
    """Add isotropic Gaussian noise to **every** start atom; scale to RMS = ε.

    - Does not modify ``reference`` (frozen + ``replace``).
    - Does **not** perturb reference (terminal) coordinates or forces.
    - ε = 0 → bit-identical start coordinates, ``applied_rms_displacement=0``.
    - Same ``rng_seed`` is bit-reproducible.

    The displacement is **not** a rigid transform: Kabsch would cancel pure
    translations/rotations, so per-atom random noise is required for a real
    basin probe.
    """

    eps = float(epsilon_angstrom)
    if eps < 0.0:
        raise ValueError(f"epsilon_angstrom must be >= 0, got {eps}")
    seed = int(rng_seed)

    start = np.asarray(reference.start_coordinates_angstrom, dtype=np.float64)
    if start.ndim != 2 or start.shape[1] != 3:
        raise ValueError(f"bad start coords shape {start.shape}")

    if eps == 0.0 or start.shape[0] == 0:
        return PerturbationResult(
            reference=reference,
            applied_rms_displacement=0.0,
            epsilon_angstrom=eps,
            rng_seed=seed,
        )

    rng = np.random.default_rng(seed)
    noise = rng.standard_normal(size=start.shape)
    # Current RMS of per-atom Euclidean norms
    atom_norm_sq = np.sum(noise * noise, axis=1)
    current_rms = float(np.sqrt(np.mean(atom_norm_sq)))
    if current_rms < 1e-30:
        # degenerate draw — reseed once
        noise = rng.standard_normal(size=start.shape)
        atom_norm_sq = np.sum(noise * noise, axis=1)
        current_rms = float(np.sqrt(np.mean(atom_norm_sq)))
    if current_rms < 1e-30:
        raise RuntimeError("failed to draw non-zero Gaussian noise")

    scaled = noise * (eps / current_rms)
    new_start = start + scaled
    applied = rms_displacement(new_start, start)

    new_ref = replace(
        reference,
        start_coordinates_angstrom=_as_coord_tuples(new_start),
    )
    return PerturbationResult(
        reference=new_ref,
        applied_rms_displacement=float(applied),
        epsilon_angstrom=eps,
        rng_seed=seed,
    )


def apply_rigid_translation(
    reference: TeacherEndpointReference,
    *,
    shift_angstrom: tuple[float, float, float],
) -> TeacherEndpointReference:
    """Translate all start atoms by a constant vector (rigid; Kabsch → ~0 RMSD).

    Test / control helper only — not used as the basin probe.
    """

    start = np.asarray(reference.start_coordinates_angstrom, dtype=np.float64)
    shift = np.asarray(shift_angstrom, dtype=np.float64).reshape(1, 3)
    new_start = start + shift
    return replace(
        reference,
        start_coordinates_angstrom=_as_coord_tuples(new_start),
    )
