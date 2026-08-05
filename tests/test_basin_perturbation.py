"""Unit tests for starting-geometry basin perturbation (no live AIMNet2)."""

from __future__ import annotations

import numpy as np

from nhc_deprot.pipeline.basin_perturbation import (
    apply_rigid_translation,
    perturb_start_geometry,
    rms_displacement,
)
from nhc_deprot.pipeline.pre_screen import (
    TeacherEndpointReference,
    kabsch_rmsd,
)


def _ref(n: int = 5, seed: int = 0) -> TeacherEndpointReference:
    rng = np.random.default_rng(seed)
    start = rng.normal(size=(n, 3))
    # terminal slightly different so Kabsch tests are well-defined
    term = start + rng.normal(scale=0.01, size=(n, 3))
    forces = np.zeros((n, 3))
    return TeacherEndpointReference(
        root_id="ROOTA",
        endpoint="cation",
        elements=tuple(["C"] * n),
        start_coordinates_angstrom=tuple(
            (float(r[0]), float(r[1]), float(r[2])) for r in start
        ),
        reference_coordinates_angstrom=tuple(
            (float(r[0]), float(r[1]), float(r[2])) for r in term
        ),
        reference_forces_ev_per_a=tuple(
            (float(r[0]), float(r[1]), float(r[2])) for r in forces
        ),
        charge=1,
        multiplicity=1,
        start_frame_index=0,
        reference_frame_index=1,
    )


def test_applied_rms_matches_epsilon() -> None:
    ref = _ref(n=12, seed=1)
    eps = 1e-3
    out = perturb_start_geometry(ref, epsilon_angstrom=eps, rng_seed=42)
    rel = abs(out.applied_rms_displacement - eps) / eps
    assert rel < 0.01, (out.applied_rms_displacement, eps, rel)
    # cross-check with independent RMS helper
    got = rms_displacement(
        out.reference.start_coordinates_angstrom,
        ref.start_coordinates_angstrom,
    )
    assert abs(got - out.applied_rms_displacement) < 1e-12


def test_same_seed_reproducible_different_seed_differs() -> None:
    ref = _ref(n=8, seed=2)
    a = perturb_start_geometry(ref, epsilon_angstrom=1e-2, rng_seed=7)
    b = perturb_start_geometry(ref, epsilon_angstrom=1e-2, rng_seed=7)
    c = perturb_start_geometry(ref, epsilon_angstrom=1e-2, rng_seed=8)
    assert a.reference.start_coordinates_angstrom == (
        b.reference.start_coordinates_angstrom
    )
    assert a.applied_rms_displacement == b.applied_rms_displacement
    assert a.reference.start_coordinates_angstrom != (
        c.reference.start_coordinates_angstrom
    )


def test_original_reference_unmodified() -> None:
    ref = _ref(n=6, seed=3)
    before = ref.start_coordinates_angstrom
    _ = perturb_start_geometry(ref, epsilon_angstrom=0.05, rng_seed=99)
    assert ref.start_coordinates_angstrom is before
    assert ref.start_coordinates_angstrom == before


def test_rigid_translation_kabsch_rmsd_near_zero() -> None:
    """Why we cannot use pure translation as a basin probe."""

    ref = _ref(n=10, seed=4)
    shifted = apply_rigid_translation(
        ref, shift_angstrom=(0.5, -0.3, 0.2)
    )
    k = kabsch_rmsd(
        shifted.start_coordinates_angstrom,
        ref.start_coordinates_angstrom,
    )
    assert k < 1e-10, k
    # but raw RMS displacement is the full translation magnitude
    raw = rms_displacement(
        shifted.start_coordinates_angstrom,
        ref.start_coordinates_angstrom,
    )
    assert abs(raw - np.linalg.norm([0.5, -0.3, 0.2])) < 1e-9


def test_epsilon_zero_bit_identical() -> None:
    ref = _ref(n=7, seed=5)
    out = perturb_start_geometry(ref, epsilon_angstrom=0.0, rng_seed=123)
    assert out.applied_rms_displacement == 0.0
    assert out.reference.start_coordinates_angstrom == (
        ref.start_coordinates_angstrom
    )
    # terminal / forces untouched
    assert out.reference.reference_coordinates_angstrom == (
        ref.reference_coordinates_angstrom
    )
    assert out.reference.reference_forces_ev_per_a == (
        ref.reference_forces_ev_per_a
    )


def test_all_atoms_perturbed_when_eps_positive() -> None:
    ref = _ref(n=9, seed=6)
    out = perturb_start_geometry(ref, epsilon_angstrom=1e-2, rng_seed=11)
    a = np.asarray(ref.start_coordinates_angstrom)
    b = np.asarray(out.reference.start_coordinates_angstrom)
    per_atom = np.linalg.norm(a - b, axis=1)
    assert np.all(per_atom > 0.0)
