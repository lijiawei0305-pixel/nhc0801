"""Live AIMNet2 GAU_LOOSE engine for zero-DFT pre-screen (mindmap step 7).

Unlike :class:`~nhc_deprot.pipeline.live_epoch0.LiveAimnet2GauLooseEngine`,
this engine accepts **fine-tune** checkpoints (``nhc0801_live_finetune``
bundles) as well as the official base weight. It does **not** enforce the
official base SHA (that check is epoch-0 specific).

Protocol surface for :mod:`nhc_deprot.pipeline.pre_screen`:
  - ``optimize_to_gau_loose(...)`` — ASE LBFGS until GAU_LOOSE / max steps
  - ``forces_at_geometry(...)`` — single-point forces at an arbitrary geometry
    (used for force RMSE at the teacher terminal reference)

Energy may appear in return dicts but is never used for pre-screen ranking
(AGENTS T1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from nhc_deprot.pipeline.parent_handoff import (
    aimnet2_gau_loose_metrics,
    load_gau_loose_profile,
)
from nhc_deprot.pipeline.pre_screen import CheckpointCandidate


class LivePreScreenEngineError(RuntimeError):
    """Live pre-screen engine failed closed."""


def load_aimnet2_weight_bundle(
    path: Path,
    *,
    map_location: str = "cpu",
) -> dict[str, Any]:
    """Load a ``.pt`` AIMNet2 export bundle (base or fine-tune).

    Requires ``torch`` at call time (live path only). Validates that the payload
    is a dict with ``state_dict`` and ``model_yaml`` — the keys AIMNet2ASE needs.
    Does **not** check the official base SHA.
    """

    if not path.is_file():
        raise LivePreScreenEngineError(f"weight file missing: {path}")
    try:
        import torch
    except ImportError as exc:  # pragma: no cover - live env has torch
        raise LivePreScreenEngineError(
            "torch is required to inspect AIMNet2 weight bundles"
        ) from exc
    try:
        raw = torch.load(path, map_location=map_location, weights_only=False)
    except Exception as exc:  # noqa: BLE001
        raise LivePreScreenEngineError(
            f"failed to load weight bundle {path}: {type(exc).__name__}: {exc}"
        ) from exc
    if not isinstance(raw, dict):
        raise LivePreScreenEngineError(
            f"weight is not a dict bundle: {path} (got {type(raw).__name__})"
        )
    if "state_dict" not in raw or "model_yaml" not in raw:
        raise LivePreScreenEngineError(
            f"weight missing state_dict/model_yaml: {path}"
        )
    return raw


def validate_checkpoint_weight(
    path: Path | str,
    *,
    bundle_loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Validate weight path + AIMNet2 bundle keys; return metadata.

    ``bundle_loader`` is injectable so unit tests can avoid torch/AIMNet2.
    """

    weight = Path(path)
    if not weight.is_file():
        raise LivePreScreenEngineError(f"weight file missing: {weight}")
    loader = bundle_loader or load_aimnet2_weight_bundle
    bundle = loader(weight)
    if not isinstance(bundle, Mapping):
        raise LivePreScreenEngineError(
            f"weight is not a mapping bundle: {weight}"
        )
    if "state_dict" not in bundle or "model_yaml" not in bundle:
        raise LivePreScreenEngineError(
            f"weight missing state_dict/model_yaml: {weight}"
        )
    return {
        "path": str(weight.resolve()),
        "is_finetune": bool(bundle.get("nhc0801_live_finetune")),
        "has_state_dict": True,
        "has_model_yaml": True,
        "run_id": bundle.get("run_id"),
        "train_config_digest": bundle.get("train_config_digest"),
        "base_sha256": bundle.get("base_sha256"),
    }


class LiveCheckpointGauLooseEngine:
    """ASE LBFGS GAU_LOOSE + single-point forces for fine-tune or base weights.

    Accepts official base ``aimnet2_wb97m_d3_0.pt`` **or**
    ``nhc0801_live_finetune`` epoch checkpoints. No official-SHA gate.
    """

    def __init__(
        self,
        *,
        weight_path: Path | str,
        max_steps: int | None = None,
        device: str | None = None,
        bundle_loader: Callable[[Path], Mapping[str, Any]] | None = None,
        skip_bundle_validation: bool = False,
    ) -> None:
        self.weight_path = Path(weight_path)
        self.device = device
        self.profile = load_gau_loose_profile()
        self.max_steps = int(max_steps) if max_steps is not None else self.profile.maximum_steps
        self.bundle_meta: dict[str, Any]
        if skip_bundle_validation:
            if not self.weight_path.is_file():
                raise LivePreScreenEngineError(
                    f"weight file missing: {self.weight_path}"
                )
            self.bundle_meta = {
                "path": str(self.weight_path.resolve()),
                "is_finetune": None,
                "has_state_dict": None,
                "has_model_yaml": None,
            }
        else:
            self.bundle_meta = validate_checkpoint_weight(
                self.weight_path, bundle_loader=bundle_loader
            )

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
        """Relax from start geometry until GAU_LOOSE five-criteria or max steps.

        Same LBFGS + ``aimnet2_gau_loose_metrics`` loop as live_epoch0.
        """

        from ase import Atoms
        from ase.optimize import LBFGS

        atoms = Atoms(
            symbols=list(elements),
            positions=np.asarray(coordinates, dtype=float),
        )
        calc = self._build_calculator(charge=charge, multiplicity=multiplicity)
        atoms.calc = calc
        # ASE typing marks logfile as IO|str; None is the documented quiet mode.
        opt = LBFGS(atoms, logfile=None)  # type: ignore[arg-type]
        prev_e: float | None = None
        prev_pos: np.ndarray | None = None
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
                previous_coordinates_angstrom=(
                    None if prev_pos is None else prev_pos.tolist()
                ),
            )
            prev_e, prev_pos = e, pos.copy()
            if metrics.get("aimnet2_gau_loose_converged") is True:
                converged = True
                break
        pos = np.asarray(atoms.get_positions(), dtype=float)
        energy_ev = float(atoms.get_potential_energy())
        return {
            "converged": converged,
            "steps": steps,
            "coordinates": pos.tolist(),
            "energy_ev": energy_ev,  # never ranked by pre_screen
            "wall_seconds": 0.0,
            "atom_identity_preserved": True,
            "topology_valid": True,
            "coordinates_finite": bool(np.isfinite(pos).all()),
            "charge_multiplicity_preserved": True,
            "checkpoint_id": checkpoint_id,
            "root_id": root_id,
            "endpoint": endpoint,
            "weight_path": str(self.weight_path),
            "device": self.device,
        }

    def forces_at_geometry(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        checkpoint_id: str | None = None,
    ) -> dict[str, Any]:
        """Single-point model forces (eV/Å) at the given geometry.

        Used by :func:`pre_screen._extract_model_forces_at_reference` for force
        RMSE at the teacher terminal — not at the relaxed geometry.
        """

        from ase import Atoms

        atoms = Atoms(
            symbols=list(elements),
            positions=np.asarray(coordinates, dtype=float),
        )
        calc = self._build_calculator(charge=charge, multiplicity=multiplicity)
        atoms.calc = calc
        forces = np.asarray(atoms.get_forces(), dtype=float)
        energy_ev = float(atoms.get_potential_energy())
        if not np.isfinite(forces).all():
            raise LivePreScreenEngineError(
                f"non-finite forces at geometry root={root_id} endpoint={endpoint}"
            )
        return {
            "forces_ev_angstrom": forces.tolist(),
            "energy_ev": energy_ev,  # never ranked
            "root_id": root_id,
            "endpoint": endpoint,
            "checkpoint_id": checkpoint_id,
            "weight_path": str(self.weight_path),
            "n_atoms": int(forces.shape[0]),
        }

    def _build_calculator(self, *, charge: int, multiplicity: int) -> Any:
        from aimnet.calculators import AIMNet2ASE

        # device is a soft hint for receipts; AIMNet2ASE manages CUDA itself.
        return AIMNet2ASE(str(self.weight_path), charge=charge, mult=multiplicity)


def live_checkpoint_engine_factory(
    candidate: CheckpointCandidate,
    *,
    max_steps: int | None = None,
    device: str | None = None,
    bundle_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    skip_bundle_validation: bool = False,
) -> LiveCheckpointGauLooseEngine:
    """Build a live engine for one pre-screen candidate.

    Fail closed if ``candidate.weight_path`` is missing.
    """

    if not candidate.weight_path:
        raise LivePreScreenEngineError(
            f"candidate {candidate.checkpoint_id!r} missing weight_path "
            "(live pre-screen requires a loadable .pt per checkpoint)"
        )
    return LiveCheckpointGauLooseEngine(
        weight_path=candidate.weight_path,
        max_steps=max_steps,
        device=device,
        bundle_loader=bundle_loader,
        skip_bundle_validation=skip_bundle_validation,
    )


def make_engine_factory(
    *,
    max_steps: int | None = None,
    device: str | None = None,
    bundle_loader: Callable[[Path], Mapping[str, Any]] | None = None,
    skip_bundle_validation: bool = False,
) -> Callable[[CheckpointCandidate], LiveCheckpointGauLooseEngine]:
    """Return ``engine_factory(candidate) -> LiveCheckpointGauLooseEngine``."""

    def _factory(candidate: CheckpointCandidate) -> LiveCheckpointGauLooseEngine:
        return live_checkpoint_engine_factory(
            candidate,
            max_steps=max_steps,
            device=device,
            bundle_loader=bundle_loader,
            skip_bundle_validation=skip_bundle_validation,
        )

    return _factory
