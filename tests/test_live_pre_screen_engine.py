"""Unit tests for live pre-screen AIMNet2 engine (mocked; no real weights)."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from nhc_deprot.pipeline.live_pre_screen_engine import (
    LiveCheckpointGauLooseEngine,
    LivePreScreenEngineError,
    live_checkpoint_engine_factory,
    make_engine_factory,
    validate_checkpoint_weight,
)
from nhc_deprot.pipeline.pre_screen import (
    CheckpointCandidate,
    TeacherEndpointReference,
    evaluate_endpoint,
    run_pre_screen_campaign,
)


def _fake_bundle(
    *,
    finetune: bool = True,
    run_id: str = "e1f100_mlp_shift",
) -> dict[str, Any]:
    return {
        "state_dict": {"dummy": 1},
        "model_yaml": "name: fake\n",
        "nhc0801_live_finetune": finetune,
        "run_id": run_id,
        "train_config_digest": "abc123",
        "base_sha256": "0" * 64,
    }


def _bundle_loader_ok(path: Path) -> Mapping[str, Any]:
    del path
    return _fake_bundle()


def _write_weight(tmp: Path, name: str = "epoch_0060.pt") -> Path:
    path = tmp / name
    path.write_bytes(b"fake-pt-bytes")
    return path


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def test_validate_checkpoint_weight_ok(tmp_path: Path) -> None:
    w = _write_weight(tmp_path)
    meta = validate_checkpoint_weight(w, bundle_loader=_bundle_loader_ok)
    assert meta["has_state_dict"] is True
    assert meta["has_model_yaml"] is True
    assert meta["is_finetune"] is True
    assert meta["path"].endswith("epoch_0060.pt")


def test_validate_checkpoint_weight_missing_file(tmp_path: Path) -> None:
    with pytest.raises(LivePreScreenEngineError, match="missing"):
        validate_checkpoint_weight(tmp_path / "nope.pt")


def test_validate_checkpoint_weight_missing_keys(tmp_path: Path) -> None:
    w = _write_weight(tmp_path)

    def bad_loader(path: Path) -> Mapping[str, Any]:
        del path
        return {"not_state": True}

    with pytest.raises(LivePreScreenEngineError, match="state_dict/model_yaml"):
        validate_checkpoint_weight(w, bundle_loader=bad_loader)


def test_engine_accepts_finetune_without_official_sha(tmp_path: Path) -> None:
    """Unlike LiveAimnet2GauLooseEngine, fine-tune weights must not raise SHA error."""

    w = _write_weight(tmp_path, "ft_seed_epoch_0120.pt")
    eng = LiveCheckpointGauLooseEngine(
        weight_path=w, bundle_loader=_bundle_loader_ok, max_steps=5
    )
    assert eng.weight_path == w
    assert eng.max_steps == 5
    assert eng.bundle_meta["is_finetune"] is True


def test_engine_skip_bundle_validation_still_requires_file(tmp_path: Path) -> None:
    with pytest.raises(LivePreScreenEngineError, match="missing"):
        LiveCheckpointGauLooseEngine(
            weight_path=tmp_path / "absent.pt",
            skip_bundle_validation=True,
        )
    w = _write_weight(tmp_path)
    eng = LiveCheckpointGauLooseEngine(
        weight_path=w, skip_bundle_validation=True, max_steps=3
    )
    assert eng.max_steps == 3


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def test_factory_requires_weight_path() -> None:
    cand = CheckpointCandidate(
        checkpoint_id="c1",
        run_id="e1f1_mlp",
        seed=1,
        epoch=10,
        weight_path=None,
    )
    with pytest.raises(LivePreScreenEngineError, match="weight_path"):
        live_checkpoint_engine_factory(cand, skip_bundle_validation=True)


def test_factory_builds_per_weight_path(tmp_path: Path) -> None:
    w1 = _write_weight(tmp_path, "a.pt")
    w2 = _write_weight(tmp_path, "b.pt")
    factory = make_engine_factory(
        max_steps=7,
        device="cpu",
        bundle_loader=_bundle_loader_ok,
    )
    e1 = factory(
        CheckpointCandidate(
            checkpoint_id="c1",
            run_id="r",
            seed=1,
            epoch=10,
            weight_path=str(w1),
        )
    )
    e2 = factory(
        CheckpointCandidate(
            checkpoint_id="c2",
            run_id="r",
            seed=1,
            epoch=20,
            weight_path=str(w2),
        )
    )
    assert e1.weight_path == w1
    assert e2.weight_path == w2
    assert e1 is not e2
    assert e1.max_steps == 7
    assert e1.device == "cpu"


# ---------------------------------------------------------------------------
# Mocked AIMNet2ASE / LBFGS
# ---------------------------------------------------------------------------


class _FakeOpt:
    """Minimal ASE optimizer stand-in for irun loop."""

    def __init__(self, atoms: Any, logfile: Any = None) -> None:
        del logfile
        self.atoms = atoms
        self._steps = 0

    def irun(self, fmax: float = 0.0, steps: int = 100):  # noqa: ARG002
        self._steps = 1
        yield None

    def get_number_of_steps(self) -> int:
        return self._steps


def _atoms_cls_from(instance: MagicMock) -> MagicMock:
    cls = MagicMock(name="Atoms")
    cls.return_value = instance
    return cls


def _install_fake_aimnet_ase(
    *,
    calc_cls: MagicMock,
    atoms_cls: MagicMock,
    include_lbfgs: bool = False,
):
    """Patch sys.modules so local imports inside the engine resolve to mocks."""

    import sys
    import types

    aimnet_mod = types.ModuleType("aimnet")
    calc_mod = types.ModuleType("aimnet.calculators")
    calc_mod.AIMNet2ASE = calc_cls  # type: ignore[attr-defined]
    aimnet_mod.calculators = calc_mod  # type: ignore[attr-defined]
    ase_mod = types.ModuleType("ase")
    ase_mod.Atoms = atoms_cls  # type: ignore[attr-defined]
    modules: dict[str, Any] = {
        "aimnet": aimnet_mod,
        "aimnet.calculators": calc_mod,
        "ase": ase_mod,
    }
    if include_lbfgs:
        opt_mod = types.ModuleType("ase.optimize")
        opt_mod.LBFGS = _FakeOpt  # type: ignore[attr-defined]
        ase_mod.optimize = opt_mod  # type: ignore[attr-defined]
        modules["ase.optimize"] = opt_mod
    return patch.dict(sys.modules, modules)


def test_optimize_to_gau_loose_mocked(tmp_path: Path) -> None:
    w = _write_weight(tmp_path)
    eng = LiveCheckpointGauLooseEngine(
        weight_path=w, bundle_loader=_bundle_loader_ok, max_steps=10
    )
    positions = np.array(
        [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=float
    )
    forces = np.array(
        [[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float
    )
    fake_atoms = MagicMock(name="atoms_instance")
    fake_atoms.get_positions.return_value = positions
    fake_atoms.get_forces.return_value = forces
    fake_atoms.get_potential_energy.return_value = -100.0
    atoms_cls = _atoms_cls_from(fake_atoms)
    calc_cls = MagicMock(name="AIMNet2ASE")

    with _install_fake_aimnet_ase(
        calc_cls=calc_cls, atoms_cls=atoms_cls, include_lbfgs=True
    ):
        out = eng.optimize_to_gau_loose(
            root_id="ROOTA",
            endpoint="cation",
            elements=("C", "N", "H"),
            coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.0, 1.0, 0.0)),
            charge=1,
            multiplicity=1,
            checkpoint_id="ck1",
        )
    assert out["checkpoint_id"] == "ck1"
    assert out["root_id"] == "ROOTA"
    assert out["endpoint"] == "cation"
    assert out["steps"] == 1
    assert "coordinates" in out
    assert out["coordinates_finite"] is True
    assert out["atom_identity_preserved"] is True
    # Energy may be present but is never a ranking field
    assert "energy_ev" in out
    calc_cls.assert_called()
    assert str(w) in str(calc_cls.call_args)


def test_forces_at_geometry_mocked(tmp_path: Path) -> None:
    w = _write_weight(tmp_path)
    eng = LiveCheckpointGauLooseEngine(
        weight_path=w, bundle_loader=_bundle_loader_ok, max_steps=5
    )
    forces = np.array(
        [[0.1, 0.0, 0.0], [-0.1, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float
    )
    fake_atoms = MagicMock()
    fake_atoms.get_forces.return_value = forces
    fake_atoms.get_potential_energy.return_value = -42.0
    atoms_cls = _atoms_cls_from(fake_atoms)
    calc_cls = MagicMock(name="AIMNet2ASE")

    with _install_fake_aimnet_ase(calc_cls=calc_cls, atoms_cls=atoms_cls):
        out = eng.forces_at_geometry(
            root_id="ROOTA",
            endpoint="neutral",
            elements=("C", "N", "H"),
            coordinates=((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.0, 1.0, 0.0)),
            charge=0,
            multiplicity=1,
            checkpoint_id="ck_force",
        )
    assert out["forces_ev_angstrom"] == forces.tolist()
    assert out["n_atoms"] == 3
    assert out["checkpoint_id"] == "ck_force"
    calc_cls.assert_called_once()
    call = calc_cls.call_args
    assert call.kwargs.get("charge") == 0
    assert call.kwargs.get("mult") == 1


def test_evaluate_endpoint_calls_forces_at_geometry(tmp_path: Path) -> None:
    """pre_screen.evaluate_endpoint must use forces_at_geometry when present."""

    w = _write_weight(tmp_path)
    eng = LiveCheckpointGauLooseEngine(
        weight_path=w, bundle_loader=_bundle_loader_ok, max_steps=5
    )

    coords = ((0.0, 0.0, 0.0), (1.4, 0.0, 0.0), (0.0, 1.0, 0.0))
    forces = ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
    ref = TeacherEndpointReference(
        root_id="R1",
        endpoint="cation",
        elements=("C", "N", "H"),
        start_coordinates_angstrom=coords,
        reference_coordinates_angstrom=coords,
        reference_forces_ev_per_a=forces,
        charge=1,
        multiplicity=1,
    )

    eng.optimize_to_gau_loose = MagicMock(  # type: ignore[method-assign]
        return_value={
            "converged": True,
            "steps": 4,
            "coordinates": [list(r) for r in coords],
            "atom_identity_preserved": True,
            "topology_valid": True,
            "coordinates_finite": True,
            "charge_multiplicity_preserved": True,
        }
    )
    eng.forces_at_geometry = MagicMock(  # type: ignore[method-assign]
        return_value={"forces_ev_angstrom": [list(r) for r in forces]}
    )

    metrics = evaluate_endpoint(eng, reference=ref, checkpoint_id="ck_eval")
    assert metrics.hard_gates_passed is True
    assert metrics.force_rmse_at_reference_ev_per_a == pytest.approx(0.0)
    eng.forces_at_geometry.assert_called_once()
    call_kw = eng.forces_at_geometry.call_args.kwargs
    assert call_kw["coordinates"] == coords
    assert call_kw["checkpoint_id"] == "ck_eval"


def test_campaign_engine_factory_per_candidate(tmp_path: Path) -> None:
    w1 = _write_weight(tmp_path, "e1.pt")
    w2 = _write_weight(tmp_path, "e2.pt")
    built_paths: list[str] = []

    def factory(cand: CheckpointCandidate) -> Any:
        assert cand.weight_path is not None
        built_paths.append(cand.weight_path)
        eng = MagicMock()
        coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.0, 0.0]]
        eng.optimize_to_gau_loose.return_value = {
            "converged": True,
            "steps": 2,
            "coordinates": coords,
            "atom_identity_preserved": True,
            "topology_valid": True,
            "coordinates_finite": True,
            "charge_multiplicity_preserved": True,
            "forces_at_reference_ev_per_a": [[0.0, 0.0, 0.0]] * 3,
        }
        return eng

    refs = [
        TeacherEndpointReference(
            root_id="R1",
            endpoint="cation",
            elements=("C", "N", "H"),
            start_coordinates_angstrom=(
                (0.0, 0.0, 0.0),
                (1.4, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            reference_coordinates_angstrom=(
                (0.0, 0.0, 0.0),
                (1.4, 0.0, 0.0),
                (0.0, 1.0, 0.0),
            ),
            reference_forces_ev_per_a=(
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
            charge=1,
            multiplicity=1,
        )
    ]
    candidates = [
        CheckpointCandidate(
            checkpoint_id="c1",
            run_id="e1f100_mlp",
            seed=1,
            epoch=10,
            weight_path=str(w1),
        ),
        CheckpointCandidate(
            checkpoint_id="c2",
            run_id="e1f100_mlp",
            seed=1,
            epoch=20,
            weight_path=str(w2),
        ),
    ]
    campaign = run_pre_screen_campaign(
        candidates=candidates,
        references=refs,
        engine_factory=factory,
        write=False,
    )
    assert campaign["final_model_selected"] is False
    assert campaign["energy_loss_used_for_ranking"] is False
    assert set(built_paths) == {str(w1), str(w2)}


# ---------------------------------------------------------------------------
# CLI wiring (live path without real AIMNet2)
# ---------------------------------------------------------------------------


def test_run_pre_screen_cli_live_fail_closed_missing_weight(
    tmp_path: Path,
) -> None:
    from nhc_deprot.generation.layout import init_generation
    from nhc_deprot.pipeline.ablation_cli import PreScreenCliError, run_pre_screen_cli

    layout, _meta, _receipt = init_generation(nhc0801_root=tmp_path / "NHC0801")
    cands = [
        CheckpointCandidate(
            checkpoint_id="no_weight",
            run_id="e1f1_mlp",
            seed=1,
            epoch=5,
            weight_path=None,
        )
    ]
    with pytest.raises(PreScreenCliError, match="weight_path"):
        run_pre_screen_cli(
            layout=layout,
            batch_id="g001",
            run_ids=["e1f1_mlp"],
            candidates=cands,
            root_ids=["FAKEROOT"],
            dry_run=False,
            write=False,
        )


def test_run_pre_screen_cli_live_uses_factory(tmp_path: Path) -> None:
    from nhc_deprot.generation.layout import init_generation
    from nhc_deprot.pipeline import ablation_cli as acl

    layout, _meta, _receipt = init_generation(nhc0801_root=tmp_path / "NHC0801")
    w = _write_weight(tmp_path, "live.pt")
    cands = [
        CheckpointCandidate(
            checkpoint_id="with_w",
            run_id="e1f100_mlp_shift",
            seed=20260730,
            epoch=60,
            weight_path=str(w),
        )
    ]

    mock_eng = MagicMock()
    coords = [[0.0, 0.0, 0.0], [1.4, 0.0, 0.0], [0.0, 1.0, 0.0]]
    mock_eng.optimize_to_gau_loose.return_value = {
        "converged": True,
        "steps": 3,
        "coordinates": coords,
        "atom_identity_preserved": True,
        "topology_valid": True,
        "coordinates_finite": True,
        "charge_multiplicity_preserved": True,
        "forces_at_reference_ev_per_a": [[0.0, 0.0, 0.0]] * 3,
    }

    factory_calls: list[CheckpointCandidate] = []

    def fake_make_factory(**kwargs: Any):
        del kwargs

        def _f(cand: CheckpointCandidate) -> Any:
            factory_calls.append(cand)
            return mock_eng

        return _f

    with (
        patch.object(acl, "resolve_references", return_value=[
            TeacherEndpointReference(
                root_id="R1",
                endpoint="cation",
                elements=("C", "N", "H"),
                start_coordinates_angstrom=(
                    (0.0, 0.0, 0.0),
                    (1.4, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                reference_coordinates_angstrom=(
                    (0.0, 0.0, 0.0),
                    (1.4, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                reference_forces_ev_per_a=(
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
                charge=1,
                multiplicity=1,
            )
        ]),
        patch(
            "nhc_deprot.pipeline.live_pre_screen_engine.make_engine_factory",
            side_effect=fake_make_factory,
        ),
    ):
        campaign = acl.run_pre_screen_cli(
            layout=layout,
            batch_id="g001",
            run_ids=["e1f100_mlp_shift"],
            candidates=cands,
            root_ids=["R1"],
            dry_run=False,
            write=False,
            device="cuda",
        )
    assert campaign["final_model_selected"] is False
    assert len(factory_calls) == 1
    assert factory_calls[0].weight_path == str(w)
    mock_eng.optimize_to_gau_loose.assert_called()


def test_pre_screen_parser_live_and_device() -> None:
    from nhc_deprot.pipeline.ablation_cli import build_pre_screen_parser

    p = build_pre_screen_parser()
    args = p.parse_args(["--live", "--device", "cuda", "--max-steps", "50"])
    assert args.live is True
    assert args.device == "cuda"
    assert args.max_steps == 50
    help_text = p.format_help()
    assert "--live" in help_text
    assert "--device" in help_text
