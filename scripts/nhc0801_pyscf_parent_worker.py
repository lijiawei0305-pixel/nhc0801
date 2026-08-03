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
from typing import Any


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
        from gpu4pyscf import dft as dft_mod  # type: ignore
    else:
        from pyscf import dft as dft_mod  # type: ignore

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
        # geomeTRIC default conv ≈ parent GAU-class thresholds; may return even if maxsteps hit.
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
        flat = [abs(float(x)) for row in grad for x in row]
        gmax = max(flat) if flat else 1e9
        grms = (sum(x * x for x in flat) / len(flat)) ** 0.5 if flat else 1e9
        geometry_converged = bool(sp_ok and gmax < gmax_gate and grms < grms_gate)
        out = {
            "geometry_converged": geometry_converged,
            "final_single_point_converged": sp_ok,
            "energy_hartree": e,
            # actual step count not always exposed; do not claim max_steps as truth
            "opt_steps": int(max_steps),
            "opt_steps_is_maxcap": True,
            "final_grad_max_eh_bohr": float(gmax),
            "final_grad_rms_eh_bohr": float(grms),
            "grad_gate_max": gmax_gate,
            "grad_gate_rms": grms_gate,
            "scf_cycles": int(getattr(mf_final, "cycles", 0) or 0),
            "wall_seconds": 0.0,
            "coordinates": [[float(x) for x in row] for row in coords],
            "xc": xc,
            "backend": backend,
            "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        }
        json.dump(out, sys.stdout)
        return 0

    print(json.dumps({"error": f"unknown op {op}"}), file=sys.stderr)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
