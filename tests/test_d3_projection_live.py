"""Live D3 projection wiring (M4): injectable projector; no real dftd3 in CI."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pytest

from nhc_deprot.data.io_util import load_json_object
from nhc_deprot.data.paths import TRAIN_ROOTS
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.d3_projection import (
    D3ProjectionError,
    Dftd3Projector,
    run_d3_campaign,
)
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.resources.profiles import get_profile


class _FixedFractionProjector:
    """Injected projector: fixed fraction of total E/G (not chemistry; no dftd3)."""

    fraction: float = 0.02

    def project_frame(self, frame: Mapping[str, Any]) -> Mapping[str, Any]:
        total_e = float(frame["energy_hartree"])
        grad = frame["gradient_hartree_per_bohr"]
        d3_e = total_e * self.fraction
        d3_g = [[float(c) * self.fraction for c in row] for row in grad]
        return {
            "d3_energy_hartree": d3_e,
            "d3_gradient_hartree_per_bohr": d3_g,
            "dispersion_identity": {
                "atm": False,
                "damping": "d3bj",
                "functional": "wb97m",
                "dry_run": False,
            },
            "d3_two_body_computed_by": "injected_test_projector",
            "d3_backend_version": "test-0",
        }


def _prepare_teacher(tmp_path: Path, roots: list[str], n_frames: int = 2):
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    camp = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=n_frames),
        dry_run=True,
    )
    assert camp.status == "DRY_RUN_COMPLETE"
    return layout


def test_live_refused_without_projector(tmp_path: Path) -> None:
    layout = _prepare_teacher(tmp_path, list(TRAIN_ROOTS[:1]), n_frames=1)
    with pytest.raises(D3ProjectionError, match="live D3 requires an injected projector"):
        run_d3_campaign(layout=layout, root_ids=list(TRAIN_ROOTS[:1]), dry_run=False)


def test_live_with_injected_projector_energy_reconstruction(tmp_path: Path) -> None:
    roots = list(TRAIN_ROOTS[:1])
    layout = _prepare_teacher(tmp_path, roots, n_frames=3)
    camp = run_d3_campaign(
        layout=layout,
        root_ids=roots,
        projector=_FixedFractionProjector(),
        dry_run=False,
        overwrite=True,
    )
    assert camp["dry_run"] is False
    assert camp["live_chemistry"] is True
    assert camp["d3_recomputation_performed"] is False
    assert camp["d3_two_body_computed_by"] == "injected_test_projector"
    assert camp["d3_backend_version"] == "test-0"
    assert camp["status"] == "LIVE_D3_PASS"
    assert camp["frame_count"] == 1 * 2 * 3  # root × endpoints × frames

    receipt_path = layout.d3_dir / roots[0] / "cation" / "frame_0001.json"
    body, _ = load_json_object(receipt_path)
    assert body["d3_recomputation_performed"] is False
    assert body["live_chemistry"] is True
    assert body["dry_run"] is False
    assert body["d3_two_body_computed_by"] == "injected_test_projector"
    assert body["d3_backend_version"] == "test-0"
    total = float(body["total_energy_hartree"])
    d3 = float(body["d3_energy_hartree"])
    short = float(body["short_range_energy_hartree"])
    assert short + d3 == pytest.approx(total, abs=1e-12)
    assert body["dispersion_identity"]["functional"] == "wb97m"
    assert body["dispersion_identity"]["damping"] == "d3bj"
    assert body["dispersion_identity"]["atm"] is False

    # Gradient reconstruction: short + d3 == total
    total_g = body["total_gradient_hartree_per_bohr"]
    d3_g = body["d3_gradient_hartree_per_bohr"]
    short_g = body["short_range_gradient_hartree_per_bohr"]
    for trow, drow, srow in zip(total_g, d3_g, short_g, strict=True):
        for t, d, s in zip(trow, drow, srow, strict=True):
            assert float(s) + float(d) == pytest.approx(float(t), abs=1e-12)


def test_dftd3_projector_identity_defaults() -> None:
    p = Dftd3Projector()
    assert p.functional == "wb97m"
    assert p.damping == "d3bj"
    assert p.atm is False


def test_dftd3_projector_rejects_wrong_identity() -> None:
    p = Dftd3Projector(functional="b3lyp")
    with pytest.raises(D3ProjectionError, match="dispersion_identity"):
        p.project_frame(
            {
                "elements": ["C", "H"],
                "coordinates_angstrom": [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                "energy_hartree": -1.0,
                "gradient_hartree_per_bohr": [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            }
        )


def test_dry_run_campaign_still_passes(tmp_path: Path) -> None:
    roots = list(TRAIN_ROOTS[:1])
    layout = _prepare_teacher(tmp_path, roots, n_frames=2)
    camp = run_d3_campaign(layout=layout, root_ids=roots, dry_run=True)
    assert camp["status"] == "DRY_RUN_D3_PASS"
    assert camp["d3_recomputation_performed"] is False
    assert camp["live_chemistry"] is False
    assert camp["d3_two_body_computed_by"] == "dry_run_synthetic"
