"""endpoint_done_ok must gate on final/terminal gradients, not mid-trajectory frame_0001."""

from __future__ import annotations

import json
from pathlib import Path

from nhc_deprot.pipeline.gpu_autofill import (
    GRAD_MAX,
    GRAD_RMS,
    endpoint_done_ok,
    endpoint_grad_gate_ok,
    load_terminal_frame,
)


def _write_frame(
    path: Path,
    *,
    frame_index: int,
    gmax: float,
    is_terminal: bool,
    functional: str = "wb97m-d3bj",
    basis: str = "def2-TZVPP",
) -> None:
    # 3 atoms × 3: one component = gmax, rest tiny
    grad = [[gmax, 1e-9, 1e-9], [1e-9, 1e-9, 1e-9], [1e-9, 1e-9, 1e-9]]
    path.write_text(
        json.dumps(
            {
                "frame_index": frame_index,
                "is_terminal": is_terminal,
                "functional": functional,
                "basis": basis,
                "gradient_hartree_per_bohr": grad,
                "energy_hartree": -100.0,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def _write_manifest(
    path: Path,
    *,
    frame_count: int = 3,
    complete: bool = True,
    final_gmax: float = 1e-5,
    final_grms: float = 1e-5,
    live: bool = True,
    dry_run: bool = False,
) -> None:
    path.write_text(
        json.dumps(
            {
                "live_chemistry": live,
                "dry_run": dry_run,
                "complete_geometry_optimization": complete,
                "frame_count": frame_count,
                "final_grad_max_eh_bohr": final_gmax,
                "final_grad_rms_eh_bohr": final_grms,
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_frame_0001_large_grad_but_terminal_ok_is_done(tmp_path: Path) -> None:
    """Regression: complete traj with large early grads must not be re-CLAIMED."""
    root = "ROOTA-UHFFFAOYSA-N"
    ep = "cation"
    d = tmp_path / root / ep
    d.mkdir(parents=True)
    _write_frame(d / "frame_0000.json", frame_index=0, gmax=0.1, is_terminal=False)
    # early step: far above gate (this is what the old code inspected)
    assert 0.05 > GRAD_MAX
    _write_frame(d / "frame_0001.json", frame_index=1, gmax=0.05, is_terminal=False)
    _write_frame(d / "frame_0002.json", frame_index=2, gmax=1e-5, is_terminal=True)
    _write_manifest(d / "manifest.json", frame_count=3, final_gmax=1e-5, final_grms=5e-6)

    assert load_terminal_frame(d)["frame_index"] == 2
    assert endpoint_grad_gate_ok(d) is True
    assert endpoint_done_ok(tmp_path, root, ep) is True


def test_manifest_final_grad_fails_not_done(tmp_path: Path) -> None:
    root = "ROOTB-UHFFFAOYSA-N"
    ep = "neutral"
    d = tmp_path / root / ep
    d.mkdir(parents=True)
    _write_frame(d / "frame_0000.json", frame_index=0, gmax=0.1, is_terminal=False)
    _write_frame(d / "frame_0001.json", frame_index=1, gmax=1e-5, is_terminal=True)
    _write_manifest(
        d / "manifest.json",
        frame_count=2,
        final_gmax=float(GRAD_MAX) * 2,
        final_grms=float(GRAD_RMS) * 2,
    )
    assert endpoint_done_ok(tmp_path, root, ep) is False


def test_incomplete_geometry_not_done(tmp_path: Path) -> None:
    root = "ROOTC-UHFFFAOYSA-N"
    ep = "cation"
    d = tmp_path / root / ep
    d.mkdir(parents=True)
    _write_frame(d / "frame_0000.json", frame_index=0, gmax=1e-5, is_terminal=False)
    _write_frame(d / "frame_0001.json", frame_index=1, gmax=1e-5, is_terminal=True)
    _write_manifest(d / "manifest.json", complete=False, final_gmax=1e-5, final_grms=1e-5)
    assert endpoint_done_ok(tmp_path, root, ep) is False


def test_gate_falls_back_to_terminal_when_manifest_missing_final(tmp_path: Path) -> None:
    root = "ROOTD-UHFFFAOYSA-N"
    ep = "cation"
    d = tmp_path / root / ep
    d.mkdir(parents=True)
    _write_frame(d / "frame_0000.json", frame_index=0, gmax=0.2, is_terminal=False)
    _write_frame(d / "frame_0001.json", frame_index=1, gmax=0.1, is_terminal=False)
    _write_frame(d / "frame_0005.json", frame_index=5, gmax=1e-5, is_terminal=True)
    # manifest without final_grad fields
    (d / "manifest.json").write_text(
        json.dumps(
            {
                "live_chemistry": True,
                "dry_run": False,
                "complete_geometry_optimization": True,
                "frame_count": 6,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert endpoint_grad_gate_ok(d) is True
    assert endpoint_done_ok(tmp_path, root, ep) is True
