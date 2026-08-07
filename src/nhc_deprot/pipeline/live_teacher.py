"""Live TeacherEngine (mindmap step 2): Parent-P01 via pyscf worker.

Writes frozen initial frame plus full geomeTRIC evaluation dump when
``capture_trajectory=True`` (default). Trajectory rows include rejected
line-search trial steps; each row records ``cycle``. Does not open Final Test.
Resource affinity is the caller's responsibility.
"""

from __future__ import annotations

import json
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from nhc_deprot.contracts.parent_protocol import (
    BASIS,
    FUNCTIONAL,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
)
from nhc_deprot.data.io_util import write_json
from nhc_deprot.pipeline.live_epoch0 import LiveParentP01Engine, load_xyz
from nhc_deprot.pipeline.teacher_runner import (
    ENDPOINT_MANIFEST_SCHEMA,
    FRAME_SCHEMA,
    MINDMAP_STEP,
    TeacherRunnerError,
)

_TRAJ_NOTE = (
    "live parent teacher: full geomeTRIC evaluation dump "
    "(cycle recorded; includes rejected trial steps)"
)
_LEGACY_NOTE = (
    "live parent teacher: initial+final frames (not full geomeTRIC step dump)"
)


def _read_trajectory_jsonl(path: Path) -> list[dict[str, Any]]:
    """Parse worker trajectory JSONL (one evaluation per line)."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if not text:
                continue
            row = json.loads(text)
            if not isinstance(row, dict):
                raise TeacherRunnerError(f"trajectory JSONL row is not an object: {path}")
            rows.append(row)
    return rows


class LiveParentTeacherEngine:
    """TeacherEngine: gold XYZ → parent first_grad + optimize (+ trajectory frames)."""

    def __init__(
        self,
        *,
        gold_xyz_dir: Path,
        max_steps: int = 250,
        pyscf_python: str = "/home/plab/test/WJW/env/conda/gpupyscf/bin/python",
        worker_script: str | None = None,
        backend: str = "cpu",
        cuda_device: int | None = None,
        host_threads: int = 2,
        capture_trajectory: bool = True,
    ) -> None:
        self.gold_xyz_dir = Path(gold_xyz_dir)
        self.backend = str(backend).lower()
        self.cuda_device = cuda_device
        self.capture_trajectory = bool(capture_trajectory)
        self.parent = LiveParentP01Engine(
            max_steps=max_steps,
            pyscf_python=pyscf_python,
            worker_script=worker_script,
            backend=self.backend,
            cuda_device=cuda_device,
            host_threads=host_threads,
        )

    def _optimize_to_final_gau(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        trajectory_out_path: str | None,
    ) -> dict[str, Any]:
        """Call parent optimize; forward trajectory path when supported.

        Prefer ``trajectory_out_path`` kwarg (test fakes / future live_epoch0).
        If the parent rejects the kwarg, bridge via ``_call`` so the worker still
        receives the field without editing ``live_epoch0`` (M2 ownership).
        """
        base_kwargs: dict[str, Any] = {
            "root_id": root_id,
            "endpoint": endpoint,
            "elements": elements,
            "coordinates": coordinates,
            "charge": charge,
            "multiplicity": multiplicity,
            "continue_from_handoff": False,
        }
        if trajectory_out_path is None:
            return dict(self.parent.optimize_to_final_gau(**base_kwargs))

        # Prefer explicit kwarg (test fakes; future live_epoch0). Dynamic call
        # so mypy does not require LiveParentP01Engine to accept the field yet.
        optimize_fn: Any = self.parent.optimize_to_final_gau
        try:
            return dict(
                optimize_fn(**base_kwargs, trajectory_out_path=trajectory_out_path)
            )
        except TypeError as exc:
            if "trajectory_out_path" not in str(exc):
                raise

        call = getattr(self.parent, "_call", None)
        if call is None:
            raise TeacherRunnerError(
                "parent engine rejects trajectory_out_path and has no _call bridge"
            )
        parent = self.parent

        def _call_with_traj(payload: dict[str, Any]) -> dict[str, Any]:
            body = dict(payload)
            if body.get("op") == "optimize_to_final_gau":
                body["trajectory_out_path"] = trajectory_out_path
            return call(body)

        parent._call = _call_with_traj  # type: ignore[method-assign]
        try:
            return dict(parent.optimize_to_final_gau(**base_kwargs))
        finally:
            parent._call = call  # type: ignore[method-assign]

    def run_endpoint(
        self,
        *,
        root_id: str,
        endpoint: str,
        charge: int,
        multiplicity: int,
        output_dir: Path,
    ) -> Mapping[str, Any]:
        xyz = self.gold_xyz_dir / f"{root_id}_{endpoint}.xyz"
        if not xyz.is_file():
            raise TeacherRunnerError(f"missing gold xyz: {xyz}")
        elements, coords0 = load_xyz(xyz)
        output_dir.mkdir(parents=True, exist_ok=True)
        t0 = time.perf_counter()
        notes: list[str] = [
            _TRAJ_NOTE if self.capture_trajectory else _LEGACY_NOTE,
            f"functional={FUNCTIONAL}",
            f"basis={BASIS}",
            f"backend={self.backend}",
        ]
        if self.backend == "gpu" and self.cuda_device is not None:
            notes.append(f"cuda_device={self.cuda_device}")

        # frame 0: first gradient at frozen geometry
        g0 = self.parent.first_gradient(
            root_id=root_id,
            endpoint=endpoint,
            elements=elements,
            coordinates=coords0,
            charge=charge,
            multiplicity=multiplicity,
        )
        if not g0.get("scf_converged"):
            raise TeacherRunnerError(f"{root_id}/{endpoint}: initial SCF failed")
        frames: list[dict[str, Any]] = []
        paths: list[str] = []

        def _write_frame(
            index: int,
            *,
            coordinates: Sequence[Sequence[float]],
            energy: float,
            gradient: Sequence[Sequence[float]] | None,
            is_terminal: bool,
            optimizer_step: int,
            source: str,
        ) -> None:
            grad = gradient or [[0.0, 0.0, 0.0] for _ in elements]
            frame = {
                "schema": FRAME_SCHEMA,
                "dry_run": False,
                "live_chemistry": True,
                "root_id": root_id,
                "endpoint": endpoint,
                "frame_index": index,
                "parent_protocol_id": PROTOCOL_ID,
                "parent_protocol_sha256": PROTOCOL_SHA256,
                "functional": FUNCTIONAL,
                "basis": BASIS,
                "charge": charge,
                "multiplicity": multiplicity,
                "elements": list(elements),
                "coordinates_angstrom": [list(map(float, row)) for row in coordinates],
                "energy_hartree": float(energy),
                "gradient_hartree_per_bohr": [list(map(float, row)) for row in grad],
                "forces_hartree_per_bohr": [
                    [-float(row[0]), -float(row[1]), -float(row[2])] for row in grad
                ],
                "optimizer_step": optimizer_step,
                "is_terminal": is_terminal,
                "lineage": {
                    "mindmap_step": MINDMAP_STEP,
                    "engine": "LiveParentTeacherEngine",
                    "single_point_only": False,
                    "source_xyz": str(xyz),
                    "source": source,
                },
            }
            path = output_dir / f"frame_{index:04d}.json"
            write_json(path, frame, overwrite=True)
            frames.append(frame)
            paths.append(str(path))

        _write_frame(
            0,
            coordinates=coords0,
            energy=float(g0["energy_hartree"]),
            gradient=g0.get("gradient_hartree_per_bohr"),
            is_terminal=False,
            optimizer_step=0,
            source="first_gradient",
        )

        traj_path: Path | None = None
        traj_path_str: str | None = None
        if self.capture_trajectory:
            traj_path = output_dir / "trajectory.jsonl"
            traj_path_str = str(traj_path)

        opt = self._optimize_to_final_gau(
            root_id=root_id,
            endpoint=endpoint,
            elements=elements,
            coordinates=coords0,
            charge=charge,
            multiplicity=multiplicity,
            trajectory_out_path=traj_path_str,
        )
        if not opt.get("geometry_converged") or not opt.get("final_single_point_converged"):
            notes.append("optimize flags incomplete; see energy / gradient gates")
        if opt.get("final_grad_max_eh_bohr") is not None:
            notes.append(
                f"final_grad_max={opt.get('final_grad_max_eh_bohr')} "
                f"rms={opt.get('final_grad_rms_eh_bohr')} "
                f"gate_max={opt.get('grad_gate_max')} gate_rms={opt.get('grad_gate_rms')}"
            )

        traj_rows: list[dict[str, Any]] = []
        if traj_path is not None:
            traj_rows = _read_trajectory_jsonl(traj_path)

        coords_f = opt.get("coordinates") or coords0
        final_scf_ok = bool(opt.get("final_single_point_converged"))

        if self.capture_trajectory and traj_rows:
            # Variable-length frames from every geomeTRIC evaluation (incl. rejected trials).
            # No second first_gradient (T2): each row already carries parent E/F.
            n_traj = len(traj_rows)
            for i, row in enumerate(traj_rows):
                is_last = i == n_traj - 1
                _write_frame(
                    i + 1,
                    coordinates=row["coordinates_angstrom"],
                    energy=float(row["energy_hartree"]),
                    gradient=row.get("gradient_hartree_per_bohr"),
                    is_terminal=is_last,
                    optimizer_step=int(row["cycle"]),
                    source="geometric_callback",
                )
            evaluation_count = int(
                opt.get("trajectory_frame_count")
                or opt.get("opt_steps")
                or n_traj
            )
            trajectory_captured = True
        elif self.capture_trajectory:
            # Capture requested but empty JSONL: terminal from worker final SP + grad (T2).
            final_grad = opt.get("final_gradient_hartree_per_bohr")
            if final_grad is None:
                raise TeacherRunnerError(
                    f"{root_id}/{endpoint}: capture_trajectory set but no trajectory "
                    "rows and worker did not return final_gradient_hartree_per_bohr"
                )
            _write_frame(
                1,
                coordinates=coords_f,
                energy=float(opt.get("energy_hartree") or 0.0),
                gradient=final_grad,
                is_terminal=True,
                optimizer_step=int(opt.get("opt_steps") or 1),
                source="geometric_callback",
            )
            evaluation_count = int(opt.get("trajectory_frame_count") or opt.get("opt_steps") or 0)
            trajectory_captured = False
        else:
            # Legacy two-frame path (no trajectory dump; worker omits final_gradient).
            g1 = self.parent.first_gradient(
                root_id=root_id,
                endpoint=endpoint,
                elements=elements,
                coordinates=coords_f,
                charge=charge,
                multiplicity=multiplicity,
            )
            final_scf_ok = bool(g1.get("scf_converged"))
            _write_frame(
                1,
                coordinates=coords_f,
                energy=float(opt.get("energy_hartree") or g1.get("energy_hartree") or 0.0),
                gradient=g1.get("gradient_hartree_per_bohr"),
                is_terminal=True,
                optimizer_step=int(opt.get("opt_steps") or 1),
                source="first_gradient",
            )
            evaluation_count = 0
            trajectory_captured = False

        wall = time.perf_counter() - t0
        geom_ok = bool(opt.get("geometry_converged"))
        manifest = {
            "schema": ENDPOINT_MANIFEST_SCHEMA,
            "root_id": root_id,
            "endpoint": endpoint,
            "frame_count": len(frames),
            "complete_geometry_optimization": geom_ok,
            "dry_run": False,
            "live_chemistry": True,
            "parent_protocol_sha256": PROTOCOL_SHA256,
            "wall_seconds": wall,
            "final_energy_hartree": frames[-1]["energy_hartree"],
            "final_grad_max_eh_bohr": opt.get("final_grad_max_eh_bohr"),
            "final_grad_rms_eh_bohr": opt.get("final_grad_rms_eh_bohr"),
            "trajectory_captured": trajectory_captured,
            "evaluation_count": evaluation_count,
            "frames": [
                {"frame_index": i, "path": f"frame_{i:04d}.json"} for i in range(len(frames))
            ],
            "notes": notes,
        }
        write_json(output_dir / "manifest.json", manifest, overwrite=True)
        return {
            "frame_count": len(frames),
            "converged": bool(geom_ok and final_scf_ok),
            "frame_paths": paths,
            "notes": notes,
            "wall_seconds": wall,
            "final_energy_hartree": frames[-1]["energy_hartree"],
            "trajectory_captured": trajectory_captured,
            "evaluation_count": evaluation_count,
        }
