"""Zero-DFT pre-screen tests — Kabsch + deterministic ranking (no live chemistry)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.pre_screen import (
    CAMPAIGN_SCHEMA,
    ROUTE_KIND_EPOCH_ZERO,
    ROUTE_KIND_FINETUNED,
    SELECTION_AUTHORITY,
    CheckpointCandidate,
    PreScreenError,
    SimulatedPreScreenEngine,
    TeacherEndpointReference,
    forces_hartree_bohr_to_ev_angstrom,
    heavy_atom_kabsch_rmsd,
    kabsch_rmsd,
    load_teacher_endpoint_reference,
    rank_candidates,
    run_pre_screen_campaign,
    screen_checkpoint,
)


def _rot_z(theta: float) -> np.ndarray:
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def test_kabsch_rmsd_identity_zero() -> None:
    pts = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
    assert kabsch_rmsd(pts, pts) == pytest.approx(0.0, abs=1e-12)


def test_kabsch_rmsd_translation_invariant() -> None:
    a = np.array(
        [[0.0, 0.0, 0.0], [1.2, 0.1, 0.0], [0.3, 1.1, 0.2], [0.0, 0.0, 1.5]],
        dtype=np.float64,
    )
    b = a + np.array([3.0, -2.5, 7.0])
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)


def test_kabsch_rmsd_rotation_invariant() -> None:
    a = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8, 0.0], [0.2, 0.1, 0.9]],
        dtype=np.float64,
    )
    r = _rot_z(0.7)
    b = a @ r.T
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-10)


def test_kabsch_rmsd_rotation_and_translation() -> None:
    a = np.array(
        [[1.0, 2.0, 3.0], [1.5, 2.1, 3.2], [0.8, 2.4, 2.9], [1.1, 1.7, 3.5]],
        dtype=np.float64,
    )
    r = _rot_z(-1.1) @ _rot_z(0.3)
    b = (a @ r.T) + np.array([-10.0, 4.0, 0.5])
    assert kabsch_rmsd(a, b) == pytest.approx(0.0, abs=1e-9)


def test_kabsch_rmsd_detects_real_difference() -> None:
    a = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    b = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.5, 0.0]]
    rmsd = kabsch_rmsd(a, b)
    assert rmsd > 0.1


def test_heavy_atom_kabsch_ignores_hydrogen_displacement() -> None:
    # Heavy skeleton identical; only H moves → heavy RMSD ~ 0
    elements = ["C", "N", "H", "H"]
    a = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, -1.0, 0.0],
    ]
    b = [
        [0.0, 0.0, 0.0],
        [1.4, 0.0, 0.0],
        [0.0, 2.5, 0.0],  # H moved far
        [0.0, -2.5, 0.0],
    ]
    assert heavy_atom_kabsch_rmsd(a, b, elements) == pytest.approx(0.0, abs=1e-12)


def test_forces_unit_conversion() -> None:
    from nhc_deprot.pipeline.pre_screen import FORCE_H_PER_B_TO_EV_PER_A

    f = forces_hartree_bohr_to_ev_angstrom([[1.0, 0.0, 0.0]])
    assert f[0][0] == pytest.approx(FORCE_H_PER_B_TO_EV_PER_A)


def _ref(
    root: str = "ROOTA",
    endpoint: str = "cation",
    n: int = 3,
    ref_shift: float = 0.0,
) -> TeacherEndpointReference:
    elements = tuple(["C"] * n)
    start = tuple((float(i), 0.0, 0.0) for i in range(n))
    ref = tuple((float(i) + ref_shift, 0.0, 0.0) for i in range(n))
    forces = tuple((0.0, 0.0, 0.0) for _ in range(n))
    charge = 1 if endpoint == "cation" else 0
    return TeacherEndpointReference(
        root_id=root,
        endpoint=endpoint,
        elements=elements,
        start_coordinates_angstrom=start,
        reference_coordinates_angstrom=ref,
        reference_forces_ev_per_a=forces,
        charge=charge,
        multiplicity=1,
        start_frame_index=0,
        reference_frame_index=1,
    )


def test_rank_order_hard_gate_then_force_then_steps_then_rmsd() -> None:
    """Deterministic sort: hard pass → force RMSE ↑ → steps ↑ → RMSD ↑.

    Energy is present on the fake engine but must not affect order.
    """

    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]

    # Fixture metrics (ref forces are zero → force RMSE scales with |F_pred|):
    #   ckpt_E: hard, force~0.01, steps=10, atom0_dx=0.01
    #   ckpt_D: hard, force~0.1,  steps=10, atom0_dx=0.01
    #   ckpt_B: hard, force~0.1,  steps=40, atom0_dx=0.01
    #   ckpt_A: hard, force~0.2,  steps=5,  atom0_dx=0.05
    #   ckpt_C: hard-fail (not converged) despite best geometry/forces
    # Hand rank under force → steps → rmsd:
    #   E (best force) → D (same force as B, fewer steps) → B → A (worst force)
    #   → C (hard fail last). Energy must not reorder.

    # atom0_dx = non-rigid deformation (Kabsch kills pure COM translation)
    outcomes = {
        "ckpt_A": {
            "atom0_dx": 0.05,
            "steps": 5,
            "forces_at_reference_ev_per_a": [
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
                [0.2, 0.0, 0.0],
            ],
            "energy_ev": -999.0,  # "best" energy — must not win
            "converged": True,
        },
        "ckpt_B": {
            "atom0_dx": 0.01,
            "steps": 40,
            "forces_at_reference_ev_per_a": [
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            "energy_ev": 0.0,
            "converged": True,
        },
        "ckpt_C": {
            "atom0_dx": 0.0,
            "steps": 1,
            "forces_at_reference_ev_per_a": [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0],
            ],
            "energy_ev": -1e6,
            "converged": False,
        },
        "ckpt_D": {
            "atom0_dx": 0.01,
            "steps": 10,
            "forces_at_reference_ev_per_a": [
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            "energy_ev": 100.0,
            "converged": True,
        },
        "ckpt_E": {
            "atom0_dx": 0.01,
            "steps": 10,
            "forces_at_reference_ev_per_a": [
                [0.01, 0.0, 0.0],
                [0.01, 0.0, 0.0],
                [0.01, 0.0, 0.0],
            ],
            "energy_ev": 50.0,
            "converged": True,
        },
    }
    engine = SimulatedPreScreenEngine(outcomes=outcomes)

    candidates = [
        CheckpointCandidate("ckpt_A", "run_x", 1, 10),
        CheckpointCandidate("ckpt_B", "run_x", 1, 20),
        CheckpointCandidate("ckpt_C", "run_x", 1, 30),
        CheckpointCandidate("ckpt_D", "run_x", 1, 40),
        CheckpointCandidate("ckpt_E", "run_x", 1, 50),
    ]
    results = [screen_checkpoint(engine, c, refs) for c in candidates]
    ranked = rank_candidates(results)
    order = [r.candidate.checkpoint_id for r in ranked]

    # Hand-derived under hard → force ↑ → steps ↑ → rmsd ↑:
    # E (force 0.01) → D (force 0.1, steps 10) → B (force 0.1, steps 40) →
    # A (force 0.2) → C (hard fail)
    assert order == ["ckpt_E", "ckpt_D", "ckpt_B", "ckpt_A", "ckpt_C"]
    assert ranked[0].hard_gates_passed is True
    assert ranked[-1].hard_gates_passed is False
    # Energy must not appear in rank key
    for r in ranked:
        assert "energy" not in str(r.rank_key).lower()


def test_topology_fail_fails_hard_gate() -> None:
    refs = [_ref()]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "bad": {
                "topology_valid": False,
                "converged": True,
                "atom0_dx": 0.0,
            }
        }
    )
    cand = CheckpointCandidate("bad", "r", 1, 1)
    result = screen_checkpoint(engine, cand, refs)
    assert result.topology_preserved is False
    assert result.hard_gates_passed is False


def test_campaign_receipt_schema_and_authority(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "good": {"atom0_dx": 0.0, "steps": 8, "converged": True},
            "ok": {"atom0_dx": 0.02, "steps": 12, "converged": True},
            "fail": {"converged": False, "atom0_dx": 0.0, "steps": 1},
        }
    )
    candidates = [
        CheckpointCandidate("good", "e1f100_mlp_shift", 20260730, 60),
        CheckpointCandidate("ok", "e1f100_mlp_shift", 20260730, 120),
        CheckpointCandidate("fail", "e1f100_mlp_shift", 20260730, 200),
    ]
    campaign = run_pre_screen_campaign(
        layout=layout,
        batch_id="g001",
        screen_id="e1f100_mlp_shift",
        candidates=candidates,
        references=refs,
        engine=engine,
        shortlist_count=2,
        write=True,
    )
    assert campaign["schema"] == CAMPAIGN_SCHEMA
    assert campaign["final_model_selected"] is False
    assert campaign["selection_authority"] == SELECTION_AUTHORITY
    assert campaign["energy_loss_used_for_ranking"] is False
    assert campaign["shortlist_checkpoint_ids"] == ["good", "ok"]
    assert "fail" not in campaign["shortlist_checkpoint_ids"]

    receipt = Path(campaign["receipt_path"])
    assert receipt.name == "screen_campaign.json"
    assert "pre_screen_g001" in str(receipt)
    assert "e1f100_mlp_shift" in str(receipt)
    loaded = json.loads(receipt.read_text(encoding="utf-8"))
    assert loaded["final_model_selected"] is False
    assert loaded["selection_authority"] == SELECTION_AUTHORITY
    assert loaded["schema"] == CAMPAIGN_SCHEMA


def test_load_teacher_endpoint_reference(tmp_path: Path) -> None:
    ep = tmp_path / "ROOTX" / "cation"
    ep.mkdir(parents=True)
    start = {
        "root_id": "ROOTX",
        "endpoint": "cation",
        "elements": ["C", "N", "H"],
        "charge": 1,
        "multiplicity": 1,
        "coordinates_angstrom": [
            [0.0, 0.0, 0.0],
            [1.4, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ],
        "is_terminal": False,
    }
    term = {
        "root_id": "ROOTX",
        "endpoint": "cation",
        "elements": ["C", "N", "H"],
        "charge": 1,
        "multiplicity": 1,
        "coordinates_angstrom": [
            [0.01, 0.0, 0.0],
            [1.41, 0.0, 0.0],
            [0.0, 1.01, 0.0],
        ],
        "forces_hartree_per_bohr": [
            [1.0e-5, 0.0, 0.0],
            [-0.5e-5, 0.0, 0.0],
            [-0.5e-5, 0.0, 0.0],
        ],
        "is_terminal": True,
    }
    (ep / "frame_0000.json").write_text(json.dumps(start), encoding="utf-8")
    (ep / "frame_0001.json").write_text(json.dumps(term), encoding="utf-8")

    ref = load_teacher_endpoint_reference(ep)
    assert ref.root_id == "ROOTX"
    assert ref.endpoint == "cation"
    assert ref.reference_frame_index == 1
    assert len(ref.elements) == 3
    assert ref.reference_forces_ev_per_a[0][0] != 0.0


def test_shortlist_excludes_hard_fail_even_if_top_k_not_filled() -> None:
    refs = [_ref()]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "only_good": {"converged": True, "coord_nudge_x": 0.0},
            "bad1": {"converged": False},
            "bad2": {"topology_valid": False},
        }
    )
    campaign = run_pre_screen_campaign(
        candidates=[
            CheckpointCandidate("only_good", "r", 1, 1),
            CheckpointCandidate("bad1", "r", 1, 2),
            CheckpointCandidate("bad2", "r", 1, 3),
        ],
        references=refs,
        engine=engine,
        shortlist_count=3,
        write=False,
    )
    assert campaign["shortlist_checkpoint_ids"] == ["only_good"]
    assert campaign["final_model_selected"] is False


def test_rank_prefers_force_over_rmsd_when_the_two_disagree() -> None:
    """Discriminating case: this ordering is only correct under the new rule.

    The main ranking fixture happens to give the same order under both the old
    (rmsd -> steps -> force) and new (force -> steps -> rmsd) keys, so it cannot
    catch a regression of the key order. Here the two keys point opposite ways:

      ckpt_force_best: force 0.01 (best), rmsd ~0.20 (worst)
      ckpt_rmsd_best : force 0.30 (worst), rmsd ~0.01 (best)

    Old key ranks rmsd_best first; new key ranks force_best first (T1 wording +
    cross-device stability, T9_OPERATIONAL §3).
    """

    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    outcomes = {
        "ckpt_force_best": {
            "atom0_dx": 0.20,
            "steps": 10,
            "forces_at_reference_ev_per_a": [[0.01, 0.0, 0.0]] * 3,
            "converged": True,
        },
        "ckpt_rmsd_best": {
            "atom0_dx": 0.01,
            "steps": 10,
            "forces_at_reference_ev_per_a": [[0.30, 0.0, 0.0]] * 3,
            "converged": True,
        },
    }
    engine = SimulatedPreScreenEngine(outcomes=outcomes)
    candidates = [
        CheckpointCandidate("ckpt_rmsd_best", "run_x", 1, 10),
        CheckpointCandidate("ckpt_force_best", "run_x", 1, 20),
    ]
    results = [screen_checkpoint(engine, c, refs) for c in candidates]
    ranked = rank_candidates(results)

    assert [r.candidate.checkpoint_id for r in ranked] == [
        "ckpt_force_best",
        "ckpt_rmsd_best",
    ]
    # Guard the premise: the two keys really do disagree on this fixture.
    winner, loser = ranked[0], ranked[1]
    assert winner.mean_force_rmse_at_reference_ev_per_a < (
        loser.mean_force_rmse_at_reference_ev_per_a
    )
    assert winner.mean_rmsd_to_reference_angstrom > (
        loser.mean_rmsd_to_reference_angstrom
    )


# --- epoch-zero baseline must not consume a shortlist slot -------------------


def _passing_outcome(force: float, dx: float = 0.01, steps: int = 10) -> dict:
    return {
        "atom0_dx": dx,
        "steps": steps,
        "forces_at_reference_ev_per_a": [[force, 0.0, 0.0]] * 3,
        "converged": True,
    }


def test_epoch_zero_candidate_ranks_but_takes_no_shortlist_slot(tmp_path: Path) -> None:
    """e0 is the yardstick, not a competitor (20260804 sci-val plan P0-2).

    The contract's epoch_zero_non_regression_rule needs e0's numbers, so it must
    stay visible in `ranked` and get its own receipt section — but it must not
    displace a fine-tuned candidate from the shortlist that feeds sci-val.
    """

    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    # e0 has the best force, so under the new key it would otherwise rank first.
    outcomes = {
        "epoch_zero": _passing_outcome(0.01),
        "ft_good": _passing_outcome(0.02),
        "ft_ok": _passing_outcome(0.03),
        "ft_meh": _passing_outcome(0.04),
    }
    candidates = [
        CheckpointCandidate(
            "epoch_zero", "epoch_zero", 0, 0, route_kind=ROUTE_KIND_EPOCH_ZERO
        ),
        CheckpointCandidate("ft_good", "run_x", 1, 10),
        CheckpointCandidate("ft_ok", "run_x", 1, 20),
        CheckpointCandidate("ft_meh", "run_x", 1, 30),
    ]
    campaign = run_pre_screen_campaign(
        candidates=candidates,
        references=refs,
        engine=SimulatedPreScreenEngine(outcomes=outcomes),
        layout=layout,
        shortlist_count=2,
    )

    # ranked keeps e0 and it does come first on merit
    assert campaign["ranked"][0]["checkpoint_id"] == "epoch_zero"
    assert campaign["ranked"][0]["route_kind"] == ROUTE_KIND_EPOCH_ZERO
    # but the shortlist is fine-tuned candidates only, and still gets 2 of them
    assert campaign["shortlist_checkpoint_ids"] == ["ft_good", "ft_ok"]
    assert all(r["route_kind"] == ROUTE_KIND_FINETUNED for r in campaign["shortlist"])
    # e0 is reported separately as the baseline
    assert campaign["epoch_zero_baseline"]["checkpoint_id"] == "epoch_zero"
    assert campaign["epoch_zero_baseline"]["rank"] == 1
    assert campaign["epoch_zero_excluded_from_shortlist"] is True


def test_route_kind_defaults_to_finetuned() -> None:
    c = CheckpointCandidate("x", "run_x", 1, 10)
    assert c.route_kind == ROUTE_KIND_FINETUNED


def test_campaign_without_epoch_zero_reports_null_baseline(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    refs = [_ref("R1", "cation")]
    campaign = run_pre_screen_campaign(
        candidates=[CheckpointCandidate("ft_only", "run_x", 1, 10)],
        references=refs,
        engine=SimulatedPreScreenEngine(outcomes={"ft_only": _passing_outcome(0.02)}),
        layout=layout,
        shortlist_count=2,
    )
    assert campaign["epoch_zero_baseline"] is None
    assert campaign["epoch_zero_excluded_from_shortlist"] is False
    assert campaign["shortlist_checkpoint_ids"] == ["ft_only"]


def test_epoch_zero_only_campaign_yields_empty_shortlist(tmp_path: Path) -> None:
    """A baseline-only screen must not produce a sci-val shortlist at all."""

    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    refs = [_ref("R1", "cation")]
    campaign = run_pre_screen_campaign(
        candidates=[
            CheckpointCandidate(
                "epoch_zero", "epoch_zero", 0, 0, route_kind=ROUTE_KIND_EPOCH_ZERO
            )
        ],
        references=refs,
        engine=SimulatedPreScreenEngine(outcomes={"epoch_zero": _passing_outcome(0.01)}),
        layout=layout,
        shortlist_count=3,
    )
    assert campaign["shortlist_checkpoint_ids"] == []
    assert campaign["status"] == "PRE_SCREEN_EMPTY_SHORTLIST"
    assert campaign["epoch_zero_baseline"]["checkpoint_id"] == "epoch_zero"


def test_candidates_json_carries_route_kind(tmp_path: Path) -> None:
    from nhc_deprot.pipeline.ablation_cli import candidates_from_json_file

    p = tmp_path / "cands.json"
    p.write_text(
        json.dumps(
            {
                "candidates": [
                    {
                        "checkpoint_id": "epoch_zero",
                        "run_id": "epoch_zero",
                        "seed": 0,
                        "epoch": 0,
                        "route_kind": ROUTE_KIND_EPOCH_ZERO,
                    },
                    {"checkpoint_id": "ft", "run_id": "run_x", "seed": 1, "epoch": 10},
                ]
            }
        ),
        encoding="utf-8",
    )
    got = {c.checkpoint_id: c.route_kind for c in candidates_from_json_file(p)}
    assert got == {
        "epoch_zero": ROUTE_KIND_EPOCH_ZERO,
        "ft": ROUTE_KIND_FINETUNED,
    }


def test_unknown_route_kind_fails_closed() -> None:
    with pytest.raises(PreScreenError, match="route_kind"):
        CheckpointCandidate("x", "run_x", 1, 10, route_kind="something_else").validate()
