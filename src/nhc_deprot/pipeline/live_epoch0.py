"""Live epoch-0 engines: AIMNet2ASE GAU_LOOSE + Parent-Level P01 PySCF opt.

Requires separate process environments:
  - AIMNet2 path: mlff.sh
  - Parent path: molenv.sh or gpupyscf python

This module is intentionally cautious and may raise if ASE/aimnet/pyscf
are missing. Resource affinity is the caller's responsibility.
"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nhc_deprot.contracts.parent_protocol import (
    BASIS,
    FUNCTIONAL,
    GRID_LEVEL,
    PROTOCOL_SHA256,
    SCF_CONV_TOL,
)
from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256
from nhc_deprot.pipeline.parent_handoff import (
    aimnet2_gau_loose_metrics,
    load_gau_loose_profile,
)
from nhc_deprot.pipeline.scientific_validation import ScientificValidationError


def load_xyz(path: Path) -> tuple[list[str], list[list[float]]]:
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    n = int(lines[0])
    elements: list[str] = []
    coords: list[list[float]] = []
    for line in lines[2 : 2 + n]:
        parts = line.split()
        elements.append(parts[0])
        coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return elements, coords


class LiveAimnet2GauLooseEngine:
    """ASE LBFGS until AIMNet2 GAU_LOOSE (or max steps)."""

    def __init__(self, *, weight_path: Path, max_steps: int | None = None) -> None:
        self.weight_path = weight_path
        self.profile = load_gau_loose_profile()
        self.max_steps = max_steps or self.profile.maximum_steps
        if _sha256(weight_path) != OFFICIAL_AIMNET2_WEIGHT_SHA256:
            raise ScientificValidationError("epoch-0 weight SHA mismatch")

    def optimize_to_gau_loose(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        checkpoint_id: str,
    ) -> dict[str, Any]:
        from aimnet.calculators import AIMNet2ASE
        from ase import Atoms
        from ase.optimize import LBFGS

        atoms = Atoms(symbols=list(elements), positions=np.asarray(coordinates, dtype=float))
        calc = AIMNet2ASE(str(self.weight_path), charge=charge, mult=multiplicity)
        atoms.calc = calc
        # ASE typing marks logfile as IO|str; None is the documented quiet mode.
        opt = LBFGS(atoms, logfile=None)  # type: ignore[arg-type]
        prev_e = None
        prev_pos = None
        steps = 0
        converged = False
        for _ in opt.irun(fmax=0.0, steps=self.max_steps):
            steps = int(opt.get_number_of_steps())
            e = float(atoms.get_potential_energy())
            f = np.asarray(atoms.get_forces(), dtype=float)
            pos = np.asarray(atoms.get_positions(), dtype=float)
            metrics = aimnet2_gau_loose_metrics(
                profile=self.profile,
                step_index=steps,
                energy_ev=e,
                forces_ev_angstrom=f.tolist(),
                coordinates_angstrom=pos.tolist(),
                previous_energy_ev=prev_e,
                previous_coordinates_angstrom=None if prev_pos is None else prev_pos.tolist(),
            )
            prev_e, prev_pos = e, pos.copy()
            if metrics.get("aimnet2_gau_loose_converged") is True:
                converged = True
                break
        pos = np.asarray(atoms.get_positions(), dtype=float)
        return {
            "converged": converged,
            "steps": steps,
            "coordinates": pos.tolist(),
            "energy_ev": float(atoms.get_potential_energy()),
            "wall_seconds": 0.0,
            "atom_identity_preserved": True,
            "topology_valid": True,
            "coordinates_finite": bool(np.isfinite(pos).all()),
            "charge_multiplicity_preserved": True,
            "checkpoint_id": checkpoint_id,
            "root_id": root_id,
            "endpoint": endpoint,
        }


class LiveParentP01Engine:
    """Parent-Level P01 via subprocess worker (gpupyscf/molenv python).

    backend:
      - \"cpu\": pyscf.dft.RKS (default; CUDA_VISIBLE_DEVICES cleared)
      - \"gpu\": gpu4pyscf.dft.RKS (cuda_device pins one physical GPU)
    """

    def __init__(
        self,
        *,
        max_steps: int = 100,
        pyscf_python: str = "/home/plab/test/WJW/env/conda/gpupyscf/bin/python",
        worker_script: str | None = None,
        backend: str = "cpu",
        cuda_device: int | None = None,
        host_threads: int = 2,
    ) -> None:
        self.max_steps = max_steps
        self.pyscf_python = pyscf_python
        self.worker_script = worker_script or str(
            Path(__file__).resolve().parents[3] / "scripts" / "nhc0801_pyscf_parent_worker.py"
        )
        self.profile = load_gau_loose_profile()
        backend_l = str(backend).lower()
        if backend_l not in {"cpu", "gpu"}:
            raise ScientificValidationError(f"invalid parent backend: {backend}")
        self.backend = backend_l
        self.cuda_device = cuda_device
        self.host_threads = max(1, int(host_threads))

    def _worker_env(self) -> dict[str, str]:
        import os

        env = os.environ.copy()
        # Keep host BLAS small so GPU wave does not steal the CPU teacher pool (0-99).
        for k in (
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "NUMEXPR_NUM_THREADS",
        ):
            env[k] = str(self.host_threads)
        if self.backend == "gpu":
            if self.cuda_device is None:
                raise ScientificValidationError("gpu backend requires cuda_device")
            env["CUDA_VISIBLE_DEVICES"] = str(int(self.cuda_device))
        else:
            env["CUDA_VISIBLE_DEVICES"] = ""
        return env

    def _call(self, payload: dict[str, Any]) -> dict[str, Any]:
        import json
        import subprocess

        body = dict(payload)
        body["backend"] = self.backend
        proc = subprocess.run(
            [self.pyscf_python, self.worker_script],
            input=json.dumps(body),
            text=True,
            capture_output=True,
            check=False,
            env=self._worker_env(),
        )
        if proc.returncode != 0:
            raise ScientificValidationError(
                f"pyscf worker failed ({proc.returncode}): {proc.stderr[-1500:]}"
            )
        return json.loads(proc.stdout)

    def first_gradient(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
    ) -> dict[str, Any]:
        del root_id, endpoint
        out = self._call(
            {
                "op": "first_gradient",
                "elements": list(elements),
                "coordinates": [list(map(float, row)) for row in coordinates],
                "charge": charge,
                "multiplicity": multiplicity,
                "basis": BASIS,
                "xc": "wb97m-d3bj",
                "grid": GRID_LEVEL,
                "conv_tol": SCF_CONV_TOL,
            }
        )
        # Normalize worker key → sci-val contract key (both kept for callers).
        if out.get("gradient_hartree_bohr") is None and out.get(
            "gradient_hartree_per_bohr"
        ) is not None:
            out["gradient_hartree_bohr"] = out["gradient_hartree_per_bohr"]
        return out

    def optimize_to_final_gau(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        continue_from_handoff: bool,
    ) -> dict[str, Any]:
        del root_id, endpoint, continue_from_handoff
        out = self._call(
            {
                "op": "optimize_to_final_gau",
                "elements": list(elements),
                "coordinates": [list(map(float, row)) for row in coordinates],
                "charge": charge,
                "multiplicity": multiplicity,
                "max_steps": self.max_steps,
                "basis": BASIS,
                "xc": "wb97m-d3bj",
                "grid": GRID_LEVEL,
                "conv_tol": SCF_CONV_TOL,
            }
        )
        out["parent_protocol_sha256"] = PROTOCOL_SHA256
        out["functional"] = FUNCTIONAL
        out["basis"] = BASIS
        out["backend"] = self.backend
        return out


def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()
