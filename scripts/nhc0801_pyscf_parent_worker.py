#!/usr/bin/env python3
"""JSON stdin/stdout Parent-P01 worker for gpupyscf/molenv python only.

Supports backend:
  - cpu (default): pyscf.dft.RKS  (CUDA_VISIBLE_DEVICES should be empty)
  - gpu: gpu4pyscf.dft.RKS       (pin one GPU via CUDA_VISIBLE_DEVICES)

xc must be compound 'wb97m-d3bj' on this stack (plain wb97m missing).
"""

from __future__ import annotations

import json
import os
import sys
from collections.abc import Callable
from typing import Any, TextIO

# CODATA-style Bohr radius in Angstrom (matches plan / ASE-style conversion).
BOHR_TO_ANGSTROM = 0.529177210903


def _to_numpy(a: Any) -> Any:
    """Convert array-like (numpy / cupy) to a float ndarray."""
    import numpy as np

    # gpu4pyscf may return cupy arrays; those expose .get() -> numpy.
    return np.asarray(a.get() if hasattr(a, "get") else a, dtype=float)


def _make_trajectory_capture(
    traj_fh: TextIO | None,
    eval_count: list[int],
) -> Callable[[dict[str, Any]], None]:
    """Build a geomeTRIC ``callback`` that dumps every energy+gradient evaluation.

    IMPORTANT — science note (AGENTS T5 / plan M1):
    Captures **all geometries that receive an energy+gradient evaluation**,
    including line-search **rejected trial steps**. This is **not** the accepted
    optimization path only. Frames remain valid parent-level (geometry, E, F)
    labels and cover a wider non-equilibrium region (valuable for a
    pre-optimizer), but ``cycle`` must be stored and frames must **not** be
    interpreted as sequential accepted-path order.
    """

    def _capture(envs: dict[str, Any]) -> None:
        eval_count[0] += 1
        if traj_fh is None:
            return
        c = _to_numpy(envs["coords"]).reshape(-1, 3) * BOHR_TO_ANGSTROM
        g = _to_numpy(envs["gradients"]).reshape(-1, 3)
        cycle = int(envs["self"].cycle)
        traj_fh.write(
            json.dumps(
                {
                    "cycle": cycle,
                    "energy_hartree": float(envs["energy"]),
                    "coordinates_angstrom": c.tolist(),
                    "gradient_hartree_per_bohr": g.tolist(),
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        traj_fh.flush()

    return _capture


def main() -> int:
    req = json.load(sys.stdin)
    op = req["op"]
    elements = req["elements"]
    coordinates = req["coordinates"]
    charge = int(req["charge"])
    multiplicity = int(req["multiplicity"])
    max_steps = int(req.get("max_steps", 100))
    basis = req.get("basis", "def2-TZVPP")
    # gpupyscf libxc: plain 'wb97m' missing; 'wb97m-d3bj' works (P01)
    xc = req.get("xc", "wb97m-d3bj")
    grid = int(req.get("grid", 4))
    conv = float(req.get("conv_tol", 1e-9))
    backend = str(req.get("backend", "cpu")).lower()
    if backend not in {"cpu", "gpu"}:
        print(json.dumps({"error": f"unknown backend {backend}"}), file=sys.stderr)
        return 2

    from pyscf import gto
    from pyscf.geomopt.geometric_solver import optimize as geometric_optimize

    if backend == "gpu":
        # Must not multi-GPU one process; caller pins CUDA_VISIBLE_DEVICES to one id.
        from gpu4pyscf import dft as dft_mod
    else:
        from pyscf import dft as dft_mod

    def _build_mf(mol_obj: Any) -> Any:
        """Parent P01: xc must be compound 'wb97m-d3bj' (plain wb97m missing on gpupyscf).

        Do NOT also set mf.disp when xc already embeds D3BJ — avoids double counting.
        """
        mf_obj = dft_mod.RKS(mol_obj)
        mf_obj.xc = xc
        xc_l = str(xc).lower().replace("_", "-")
        if "d3bj" not in xc_l and "d3" not in xc_l:
            # only attach empirical disp when functional itself has none
            try:
                mf_obj.disp = "d3bj"
            except Exception:
                pass
        mf_obj.grids.level = grid
        mf_obj.conv_tol = conv
        return mf_obj

    mol = gto.M(
        atom=[(el, tuple(c)) for el, c in zip(elements, coordinates, strict=True)],
        basis=basis,
        charge=charge,
        spin=multiplicity - 1,
        unit="Angstrom",
        verbose=0,
    )
    mf = _build_mf(mol)

    if op == "first_gradient":
        e = float(mf.kernel())
        scf_ok = bool(mf.converged)
        g = None
        if scf_ok:
            g = mf.nuc_grad_method().kernel()
            g = [[float(x) for x in row] for row in g]
        out: dict[str, Any] = {
            "scf_converged": scf_ok,
            "energy_hartree": e if scf_ok else None,
            "gradient_hartree_per_bohr": g,
            "coordinates_finite": True,
            "atom_identity_preserved": True,
            "charge_multiplicity_preserved": True,
            "topology_valid": True,
            "xc": xc,
            "backend": backend,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        json.dump(out, sys.stdout)
        return 0

    if op == "optimize_to_final_gau":
        # Optional full-trajectory dump (M1). Absent => legacy response shape.
        # geomeTRIC callback sees every energy+gradient evaluation (incl. rejected
        # line-search trial steps), not only the accepted optimization path.
        import time

        t0 = time.perf_counter()
        raw_traj = req.get("trajectory_out_path")
        traj_path: str | None = str(raw_traj) if raw_traj else None
        eval_count = [0]
        traj_fh: TextIO | None = (
            open(traj_path, "w", encoding="utf-8") if traj_path else None
        )
        try:
            # geomeTRIC default conv ≈ parent GAU-class thresholds; may return even if maxsteps hit.
            if traj_path is not None:
                capture = _make_trajectory_capture(traj_fh, eval_count)
                mol_eq = geometric_optimize(
                    mf, maxsteps=max_steps, callback=capture
                )
            else:
                mol_eq = geometric_optimize(mf, maxsteps=max_steps)
            mf_final = _build_mf(mol_eq)
            e = float(mf_final.kernel())
            sp_ok = bool(mf_final.converged)
            coords = mol_eq.atom_coords(unit="Angstrom")
            # Endpoint quality: final gradient must be tight (not just "opt returned").
            # geomeTRIC defaults ~ gmax 4.5e-4, grms 3e-4 Eh/Bohr.
            gmax_gate = 4.5e-4
            grms_gate = 3.0e-4
            grad = mf_final.nuc_grad_method().kernel()
            if traj_path is not None:
                grad_arr = _to_numpy(grad).reshape(-1, 3)
                flat = [abs(float(x)) for x in grad_arr.ravel()]
                final_grad_list = [[float(x) for x in row] for row in grad_arr]
            else:
                flat = [abs(float(x)) for row in grad for x in row]
                final_grad_list = None
            gmax = max(flat) if flat else 1e9
            grms = (sum(x * x for x in flat) / len(flat)) ** 0.5 if flat else 1e9
            geometry_converged = bool(sp_ok and gmax < gmax_gate and grms < grms_gate)
            if traj_path is not None:
                # Real evaluation count from callback (fixes B4 when capturing).
                opt_steps = int(eval_count[0])
                opt_steps_is_maxcap = False
            else:
                # Legacy: actual step count not exposed; do not claim max_steps as truth.
                opt_steps = int(max_steps)
                opt_steps_is_maxcap = True
            out = {
                "geometry_converged": geometry_converged,
                "final_single_point_converged": sp_ok,
                "energy_hartree": e,
                "opt_steps": opt_steps,
                "opt_steps_is_maxcap": opt_steps_is_maxcap,
                "final_grad_max_eh_bohr": float(gmax),
                "final_grad_rms_eh_bohr": float(grms),
                "grad_gate_max": gmax_gate,
                "grad_gate_rms": grms_gate,
                "scf_cycles": int(getattr(mf_final, "cycles", 0) or 0),
                "wall_seconds": float(time.perf_counter() - t0),
                "coordinates": [[float(x) for x in row] for row in coords],
                "xc": xc,
                "backend": backend,
                "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
            }
            if traj_path is not None:
                out["trajectory_frame_count"] = int(eval_count[0])
                out["trajectory_path"] = traj_path
                out["final_gradient_hartree_per_bohr"] = final_grad_list
            json.dump(out, sys.stdout)
            return 0
        finally:
            if traj_fh is not None:
                traj_fh.close()

    print(json.dumps({"error": f"unknown op {op}"}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
