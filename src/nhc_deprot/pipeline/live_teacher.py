"""Live TeacherEngine (mindmap step 2): Parent-P01 via pyscf worker.

Writes initial + final frames (full geomeTRIC step dump can be added later).
Does not open Final Test. Resource affinity is the caller's responsibility.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

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


class LiveParentTeacherEngine:
    """TeacherEngine: gold XYZ → parent first_grad + optimize + final grad frames."""

    def __init__(
        self,
        *,
        gold_xyz_dir: Path,
        max_steps: int = 100,
        pyscf_python: str = "/home/plab/test/WJW/env/conda/gpupyscf/bin/python",
        worker_script: str | None = None,
        backend: str = "cpu",
        cuda_device: int | None = None,
        host_threads: int = 2,
    ) -> None:
        self.gold_xyz_dir = Path(gold_xyz_dir)
        self.backend = str(backend).lower()
        self.cuda_device = cuda_device
        self.parent = LiveParentP01Engine(
            max_steps=max_steps,
            pyscf_python=pyscf_python,
            worker_script=worker_script,
            backend=self.backend,
            cuda_device=cuda_device,
            host_threads=host_threads,
        )

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
            "live parent teacher: initial+final frames (not full geomeTRIC step dump)",
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
        )

        opt = self.parent.optimize_to_final_gau(
            root_id=root_id,
            endpoint=endpoint,
            elements=elements,
            coordinates=coords0,
            charge=charge,
            multiplicity=multiplicity,
            continue_from_handoff=False,
        )
        if not opt.get("geometry_converged") or not opt.get("final_single_point_converged"):
            notes.append("optimize flags incomplete; see energy / gradient gates")
        if opt.get("final_grad_max_eh_bohr") is not None:
            notes.append(
                f"final_grad_max={opt.get('final_grad_max_eh_bohr')} "
                f"rms={opt.get('final_grad_rms_eh_bohr')} "
                f"gate_max={opt.get('grad_gate_max')} gate_rms={opt.get('grad_gate_rms')}"
            )
        coords_f = opt.get("coordinates") or coords0
        # Prefer gradient already computed at optimized geometry inside worker when present.
        g1 = self.parent.first_gradient(
            root_id=root_id,
            endpoint=endpoint,
            elements=elements,
            coordinates=coords_f,
            charge=charge,
            multiplicity=multiplicity,
        )
        _write_frame(
            1,
            coordinates=coords_f,
            energy=float(opt.get("energy_hartree") or g1.get("energy_hartree") or 0.0),
            gradient=g1.get("gradient_hartree_per_bohr"),
            is_terminal=True,
            optimizer_step=int(opt.get("opt_steps") or 1),
        )

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
            "frames": [
                {"frame_index": i, "path": f"frame_{i:04d}.json"} for i in range(len(frames))
            ],
            "notes": notes,
        }
        write_json(output_dir / "manifest.json", manifest, overwrite=True)
        return {
            "frame_count": len(frames),
            "converged": bool(opt.get("geometry_converged") and g1.get("scf_converged")),
            "frame_paths": paths,
            "notes": notes,
            "wall_seconds": wall,
            "final_energy_hartree": frames[-1]["energy_hartree"],
        }
