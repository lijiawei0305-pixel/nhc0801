"""M1: parent worker full-trajectory callback (no real PySCF)."""

from __future__ import annotations

import importlib.util
import io
import json
import sys
import types
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
WORKER_PATH = ROOT / "scripts" / "nhc0801_pyscf_parent_worker.py"
BOHR = 0.529177210903


def _load_worker():
    """Load parent worker module by path (scripts/ is not a package)."""
    name = "nhc0801_pyscf_parent_worker"
    # Drop cached module so each test can reinstall pyscf mocks if needed.
    if name in sys.modules:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, WORKER_PATH)
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


class _CupyLike:
    """Minimal cupy-like object: has .get() returning a numpy array."""

    def __init__(self, arr: Any) -> None:
        self._arr = np.asarray(arr, dtype=float)

    def get(self) -> np.ndarray:
        return self._arr.copy()


class _FakeEngine:
    def __init__(self, cycle: int) -> None:
        self.cycle = cycle


def _install_fake_pyscf(
    *,
    geometric_optimize: Any,
    final_energy: float = -100.0,
    final_grad: Any | None = None,
    final_coords_angstrom: Any | None = None,
    scf_converged: bool = True,
) -> dict[str, Any]:
    """Install fake pyscf / gpu4pyscf modules into sys.modules."""
    if final_grad is None:
        final_grad = np.zeros((3, 3), dtype=float)
    if final_coords_angstrom is None:
        final_coords_angstrom = np.array(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0]],
            dtype=float,
        )

    mol_eq = MagicMock(name="mol_eq")
    mol_eq.atom_coords = MagicMock(return_value=final_coords_angstrom)

    grad_method = MagicMock()
    grad_method.kernel = MagicMock(return_value=final_grad)

    mf_final = MagicMock(name="mf_final")
    mf_final.kernel = MagicMock(return_value=final_energy)
    mf_final.converged = scf_converged
    mf_final.cycles = 3
    mf_final.nuc_grad_method = MagicMock(return_value=grad_method)
    mf_final.grids = MagicMock()
    mf_final.conv_tol = 1e-9
    mf_final.xc = "wb97m-d3bj"

    mf0 = MagicMock(name="mf0")
    mf0.grids = MagicMock()
    mf0.conv_tol = 1e-9
    mf0.xc = "wb97m-d3bj"

    rks_calls: list[Any] = []

    def _rks(mol_obj: Any) -> Any:
        rks_calls.append(mol_obj)
        # first build is pre-opt mf; subsequent is final SP
        if len(rks_calls) == 1:
            return mf0
        return mf_final

    dft_mod = types.ModuleType("pyscf.dft")
    dft_mod.RKS = _rks  # type: ignore[attr-defined]

    gto_mod = types.ModuleType("pyscf.gto")
    gto_mod.M = MagicMock(return_value=MagicMock(name="mol0"))  # type: ignore[attr-defined]

    geom_solver = types.ModuleType("pyscf.geomopt.geometric_solver")
    geom_solver.optimize = geometric_optimize  # type: ignore[attr-defined]

    geomopt = types.ModuleType("pyscf.geomopt")
    geomopt.geometric_solver = geom_solver  # type: ignore[attr-defined]

    pyscf = types.ModuleType("pyscf")
    pyscf.gto = gto_mod  # type: ignore[attr-defined]
    pyscf.dft = dft_mod  # type: ignore[attr-defined]
    pyscf.geomopt = geomopt  # type: ignore[attr-defined]

    sys.modules["pyscf"] = pyscf
    sys.modules["pyscf.gto"] = gto_mod
    sys.modules["pyscf.dft"] = dft_mod
    sys.modules["pyscf.geomopt"] = geomopt
    sys.modules["pyscf.geomopt.geometric_solver"] = geom_solver
    # Ensure gpu backend path is importable if ever requested
    gpu4 = types.ModuleType("gpu4pyscf")
    gpu4.dft = dft_mod  # type: ignore[attr-defined]
    sys.modules["gpu4pyscf"] = gpu4
    sys.modules["gpu4pyscf.dft"] = dft_mod

    return {
        "mol_eq": mol_eq,
        "mf0": mf0,
        "mf_final": mf_final,
        "rks_calls": rks_calls,
        "final_grad": final_grad,
        "final_coords_angstrom": final_coords_angstrom,
    }


def _base_optimize_request(**overrides: Any) -> dict[str, Any]:
    req: dict[str, Any] = {
        "op": "optimize_to_final_gau",
        "elements": ["O", "H", "H"],
        "coordinates": [
            [0.0, 0.0, 0.0],
            [0.96, 0.0, 0.0],
            [-0.24, 0.93, 0.0],
        ],
        "charge": 0,
        "multiplicity": 1,
        "max_steps": 100,
        "backend": "cpu",
    }
    req.update(overrides)
    return req


def _run_main(worker: Any, req: dict[str, Any]) -> dict[str, Any]:
    stdin = io.StringIO(json.dumps(req))
    stdout = io.StringIO()
    old_in, old_out = sys.stdin, sys.stdout
    try:
        sys.stdin = stdin
        sys.stdout = stdout
        rc = worker.main()
    finally:
        sys.stdin = old_in
        sys.stdout = old_out
    assert rc == 0, f"main returned {rc}; stdout={stdout.getvalue()!r}"
    return json.loads(stdout.getvalue())


# ---------------------------------------------------------------------------
# Helpers (unit)
# ---------------------------------------------------------------------------


def test_to_numpy_plain_and_cupy_like() -> None:
    worker = _load_worker()
    plain = np.array([[1.0, 2.0, 3.0]], dtype=float)
    out = worker._to_numpy(plain)
    assert isinstance(out, np.ndarray)
    np.testing.assert_allclose(out, plain)

    cupy_like = _CupyLike([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])
    out2 = worker._to_numpy(cupy_like)
    np.testing.assert_allclose(out2, [[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]])


def test_trajectory_callback_unit_conversion_and_jsonl(tmp_path: Path) -> None:
    worker = _load_worker()
    traj = tmp_path / "trajectory.jsonl"
    eval_count = [0]
    with traj.open("w", encoding="utf-8") as fh:
        capture = worker._make_trajectory_capture(fh, eval_count)
        # coords in Bohr; one atom at (1, 0, 0) Bohr -> BOHR Angstrom
        coords_bohr = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]])
        grads = np.array([[0.1, 0.0, 0.0], [0.0, -0.2, 0.0], [0.0, 0.0, 0.3]])
        capture(
            {
                "coords": coords_bohr,
                "energy": -99.5,
                "gradients": grads,
                "self": _FakeEngine(7),
            }
        )
        # cupy-like second eval (rejected trial step etc.)
        capture(
            {
                "coords": _CupyLike(coords_bohr * 1.1),
                "energy": -99.4,
                "gradients": _CupyLike(grads * 0.5),
                "self": _FakeEngine(8),
            }
        )

    assert eval_count[0] == 2
    lines = traj.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    row0 = json.loads(lines[0])
    assert row0["cycle"] == 7
    assert row0["energy_hartree"] == pytest.approx(-99.5)
    np.testing.assert_allclose(
        row0["coordinates_angstrom"],
        (coords_bohr * BOHR).tolist(),
    )
    np.testing.assert_allclose(row0["gradient_hartree_per_bohr"], grads.tolist())
    row1 = json.loads(lines[1])
    assert row1["cycle"] == 8
    np.testing.assert_allclose(
        row1["coordinates_angstrom"],
        (coords_bohr * 1.1 * BOHR).tolist(),
    )


# ---------------------------------------------------------------------------
# Full request path with mocked geometric_optimize
# ---------------------------------------------------------------------------


def test_optimize_with_trajectory_out_path(tmp_path: Path) -> None:
    """Mock geometric_optimize injects N callback calls; JSONL + response fields."""
    n_evals = 4
    coords_seq = [
        np.array([[1.0 + 0.1 * i, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
        for i in range(n_evals)
    ]
    grads_seq = [
        np.array([[0.01 * (i + 1), 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
        for i in range(n_evals)
    ]
    energies = [-100.0 - 0.01 * i for i in range(n_evals)]
    captured_kwargs: dict[str, Any] = {}

    def fake_optimize(mf: Any, maxsteps: int = 100, callback: Any = None, **kwargs: Any):
        captured_kwargs["maxsteps"] = maxsteps
        captured_kwargs["callback"] = callback
        captured_kwargs["kwargs"] = kwargs
        assert callback is not None
        for i in range(n_evals):
            callback(
                {
                    "coords": coords_seq[i],
                    "energy": energies[i],
                    "gradients": grads_seq[i],
                    "self": _FakeEngine(i + 1),
                }
            )
        return MagicMock(
            atom_coords=MagicMock(
                return_value=np.array(
                    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
                )
            )
        )

    final_grad = np.array(
        [[1e-5, 0.0, 0.0], [0.0, 1e-5, 0.0], [0.0, 0.0, 1e-5]], dtype=float
    )
    _install_fake_pyscf(geometric_optimize=fake_optimize, final_grad=final_grad)
    worker = _load_worker()

    traj_path = tmp_path / "traj.jsonl"
    out = _run_main(
        worker,
        _base_optimize_request(trajectory_out_path=str(traj_path), max_steps=50),
    )

    assert traj_path.is_file()
    lines = traj_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == n_evals
    for i, line in enumerate(lines):
        row = json.loads(line)
        assert row["cycle"] == i + 1
        assert row["energy_hartree"] == pytest.approx(energies[i])
        np.testing.assert_allclose(
            row["coordinates_angstrom"],
            (coords_seq[i] * BOHR).tolist(),
        )
        np.testing.assert_allclose(
            row["gradient_hartree_per_bohr"],
            grads_seq[i].tolist(),
        )

    assert out["opt_steps"] == n_evals
    assert out["opt_steps_is_maxcap"] is False
    assert out["trajectory_frame_count"] == n_evals
    assert out["trajectory_path"] == str(traj_path)
    assert "final_gradient_hartree_per_bohr" in out
    np.testing.assert_allclose(out["final_gradient_hartree_per_bohr"], final_grad)
    assert out["geometry_converged"] is True
    assert out["energy_hartree"] == pytest.approx(-100.0)
    assert captured_kwargs["maxsteps"] == 50


def test_optimize_without_trajectory_matches_legacy_semantics(tmp_path: Path) -> None:
    """trajectory_out_path absent: opt_steps=max_steps, is_maxcap True, no new keys."""

    def fake_optimize(mf: Any, maxsteps: int = 100, callback: Any = None, **kwargs: Any):
        # Legacy call must not require callback; if present, still fine if unused
        assert callback is None
        return MagicMock(
            atom_coords=MagicMock(
                return_value=np.array(
                    [[0.0, 0.0, 0.0], [0.96, 0.0, 0.0], [-0.24, 0.93, 0.0]]
                )
            )
        )

    final_grad = np.zeros((3, 3), dtype=float)
    _install_fake_pyscf(geometric_optimize=fake_optimize, final_grad=final_grad)
    worker = _load_worker()

    out = _run_main(worker, _base_optimize_request(max_steps=100))

    assert out["opt_steps"] == 100
    assert out["opt_steps_is_maxcap"] is True
    assert "trajectory_frame_count" not in out
    assert "trajectory_path" not in out
    assert "final_gradient_hartree_per_bohr" not in out
    # Legacy keys still present
    for key in (
        "geometry_converged",
        "final_single_point_converged",
        "energy_hartree",
        "final_grad_max_eh_bohr",
        "final_grad_rms_eh_bohr",
        "grad_gate_max",
        "grad_gate_rms",
        "scf_cycles",
        "wall_seconds",
        "coordinates",
        "xc",
        "backend",
        "cuda_visible_devices",
    ):
        assert key in out


def test_trajectory_callback_docstring_mentions_rejected_trials() -> None:
    worker = _load_worker()
    # Implementation must document: all evaluated geometries, not accepted path only.
    src = WORKER_PATH.read_text(encoding="utf-8")
    assert "线搜索" in src or "rejected" in src.lower() or "试探" in src
    assert "不是" in src or "not" in src.lower()
    # ensure factory has a docstring that states the science note
    doc = worker._make_trajectory_capture.__doc__
    assert doc is not None
    assert (
        "rejected" in doc.lower()
        or "试探" in doc
        or "所有被求值" in doc
    )
