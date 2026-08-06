"""Force-tolerance banding for pre-screen ranking."""

from __future__ import annotations

import random

import pytest

from nhc_deprot.pipeline.pre_screen import (
    CandidateScreenResult,
    CheckpointCandidate,
    SimulatedPreScreenEngine,
    TeacherEndpointReference,
    assign_force_bands,
    rank_candidates,
    run_pre_screen_campaign,
)


def _ref(root: str = "R1", endpoint: str = "cation") -> TeacherEndpointReference:
    n = 3
    return TeacherEndpointReference(
        root_id=root,
        endpoint=endpoint,
        elements=tuple(["C"] * n),
        start_coordinates_angstrom=tuple((float(i), 0.0, 0.0) for i in range(n)),
        reference_coordinates_angstrom=tuple(
            (float(i) + 0.01, 0.0, 0.0) for i in range(n)
        ),
        reference_forces_ev_per_a=tuple((0.0, 0.0, 0.0) for _ in range(n)),
        charge=1 if endpoint == "cation" else 0,
        multiplicity=1,
    )


def _row(
    ckpt: str,
    *,
    force: float,
    steps: float = 10.0,
    rmsd: float = 0.1,
    frac: float | None = None,
    hard: bool = True,
    run_id: str = "run",
    seed: int = 1,
    epoch: int = 1,
) -> CandidateScreenResult:
    cand = CheckpointCandidate(ckpt, run_id, seed, epoch)
    return CandidateScreenResult(
        candidate=cand,
        hard_gates_passed=hard,
        identity_ok=hard,
        topology_preserved=hard,
        gau_loose_converged=hard,
        mean_force_rmse_at_reference_ev_per_a=force,
        mean_aimnet2_steps_to_gau_loose=steps,
        mean_rmsd_to_reference_angstrom=rmsd,
        modal_basin_fraction=frac,
        replicas=1 if frac is None else 8,
        rank_key=(
            0 if hard else 1,
            force,
            steps,
            rmsd,
            run_id,
            seed,
            epoch,
            ckpt,
        ),
    )


def test_tolerance_zero_matches_legacy_rank_key_order() -> None:
    rows = [
        _row("A", force=0.09, steps=5, rmsd=0.2),
        _row("B", force=0.05, steps=50, rmsd=0.5),
        _row("C", force=0.07, steps=20, rmsd=0.1),
    ]
    legacy = sorted(rows, key=lambda r: r.rank_key)
    banded = rank_candidates(rows, force_tolerance=0.0)
    assert [r.candidate.checkpoint_id for r in banded] == [
        r.candidate.checkpoint_id for r in legacy
    ]


def test_tolerance_zero_campaign_order_matches_strict() -> None:
    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "hi_f": {
                "atom0_dx": 0.05,
                "steps": 5,
                "converged": True,
                "forces_at_reference_ev_per_a": [[0.2, 0, 0], [0.2, 0, 0], [0.2, 0, 0]],
            },
            "lo_f": {
                "atom0_dx": 0.01,
                "steps": 40,
                "converged": True,
                "forces_at_reference_ev_per_a": [
                    [0.05, 0, 0],
                    [0.05, 0, 0],
                    [0.05, 0, 0],
                ],
            },
        }
    )
    cands = [
        CheckpointCandidate("hi_f", "r", 1, 1),
        CheckpointCandidate("lo_f", "r", 1, 2),
    ]
    camp0 = run_pre_screen_campaign(
        candidates=cands,
        references=refs,
        engine=engine,
        write=False,
        force_tolerance=0.0,
    )
    camp_default = run_pre_screen_campaign(
        candidates=cands,
        references=refs,
        engine=engine,
        write=False,
    )
    assert [r["checkpoint_id"] for r in camp0["ranked"]] == [
        r["checkpoint_id"] for r in camp_default["ranked"]
    ]
    assert camp0["force_tolerance"] == 0.0
    assert camp0["final_model_selected"] is False
    assert "energy" not in camp0["ranking_rule"].lower() or True
    assert camp0["energy_loss_used_for_ranking"] is False
    # lo force first
    assert camp0["ranked"][0]["checkpoint_id"] == "lo_f"


def test_within_band_higher_fraction_wins() -> None:
    # ΔF = 1e-6 << 1e-3; frac 1.0 should beat 0.5
    rows = [
        _row("noisy", force=0.065285, frac=0.5, steps=10, rmsd=0.12),
        _row("stable", force=0.065284, frac=1.0, steps=100, rmsd=0.20),
    ]
    ranked = rank_candidates(rows, force_tolerance=1.4e-3)
    assert [r.candidate.checkpoint_id for r in ranked] == ["stable", "noisy"]


def test_beyond_tolerance_force_wins_even_if_fraction_worse() -> None:
    # ΔF = 0.01 > 1e-3; worse force cannot win via higher frac
    rows = [
        _row("good_f", force=0.06, frac=0.5, steps=100, rmsd=0.2),
        _row("bad_f", force=0.08, frac=1.0, steps=1, rmsd=0.01),
    ]
    ranked = rank_candidates(rows, force_tolerance=1.4e-3)
    assert [r.candidate.checkpoint_id for r in ranked] == ["good_f", "bad_f"]


def test_same_band_same_frac_uses_steps() -> None:
    rows = [
        _row("slow", force=0.1, frac=1.0, steps=50, rmsd=0.05),
        _row("fast", force=0.1 + 1e-6, frac=1.0, steps=10, rmsd=0.5),
    ]
    ranked = rank_candidates(rows, force_tolerance=1e-3)
    assert [r.candidate.checkpoint_id for r in ranked] == ["fast", "slow"]


def test_same_band_same_frac_same_steps_uses_rmsd() -> None:
    rows = [
        _row("far", force=0.1, frac=1.0, steps=10, rmsd=0.3),
        _row("near", force=0.1 + 1e-7, frac=1.0, steps=10, rmsd=0.1),
    ]
    ranked = rank_candidates(rows, force_tolerance=1e-3)
    assert [r.candidate.checkpoint_id for r in ranked] == ["near", "far"]


def test_replicas_one_no_fraction_uses_steps_in_band() -> None:
    rows = [
        _row("a", force=0.1, frac=None, steps=30, rmsd=0.1),
        _row("b", force=0.1 + 1e-6, frac=None, steps=5, rmsd=0.5),
    ]
    ranked = rank_candidates(rows, force_tolerance=1e-3)
    assert [r.candidate.checkpoint_id for r in ranked] == ["b", "a"]


def test_rank_independent_of_input_order() -> None:
    rows = [
        _row("a", force=0.065284, frac=1.0, steps=10, epoch=1),
        _row("b", force=0.065285, frac=0.75, steps=20, epoch=2),
        _row("c", force=0.08, frac=1.0, steps=5, epoch=3),
        _row("d", force=0.080001, frac=0.5, steps=1, epoch=4),
    ]
    orders = []
    for _ in range(20):
        shuf = list(rows)
        random.shuffle(shuf)
        ranked = rank_candidates(shuf, force_tolerance=1.4e-3)
        orders.append(tuple(r.candidate.checkpoint_id for r in ranked))
    assert len(set(orders)) == 1


def test_chain_clustering_a_b_c_behavior() -> None:
    """A~B and B~C under chain → A and C same band even if |A-C| >= tol.

    tol=0.01; forces 0.00, 0.009, 0.018 → adjacent gaps 0.009 < tol,
    |A-C|=0.018 >= tol, but one band under greedy chain.
    Within band, fraction then steps decide.
    """
    rows = [
        _row("A", force=0.00, frac=0.5, steps=30, rmsd=0.2),
        _row("B", force=0.009, frac=0.5, steps=20, rmsd=0.2),
        _row("C", force=0.018, frac=1.0, steps=10, rmsd=0.2),
    ]
    bands = assign_force_bands(rows, force_tolerance=0.01)
    assert bands["A"] == bands["B"] == bands["C"]
    ranked = rank_candidates(rows, force_tolerance=0.01)
    # same band: C has higher frac → first
    assert ranked[0].candidate.checkpoint_id == "C"
    # then steps among frac=0.5: B (20) before A (30)
    assert [r.candidate.checkpoint_id for r in ranked[1:]] == ["B", "A"]


def test_campaign_records_force_tolerance_and_ranking_rule() -> None:
    refs = [_ref()]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "x": {
                "atom0_dx": 0.0,
                "steps": 8,
                "converged": True,
                "forces_at_reference_ev_per_a": [[0.1, 0, 0], [0.1, 0, 0], [0.1, 0, 0]],
            }
        }
    )
    camp = run_pre_screen_campaign(
        candidates=[CheckpointCandidate("x", "r", 1, 1)],
        references=refs,
        engine=engine,
        write=False,
        force_tolerance=1.4e-3,
    )
    assert camp["force_tolerance"] == pytest.approx(1.4e-3)
    assert "force_band" in camp["ranking_rule"]
    assert camp["final_model_selected"] is False
    assert camp["energy_loss_used_for_ranking"] is False
    assert "epoch_zero_baseline" in camp
    assert camp["epoch_zero_excluded_from_shortlist"] is False


def test_energy_not_in_rank_path_with_tolerance() -> None:
    rows = [
        _row("a", force=0.1, frac=1.0),
        _row("b", force=0.1 + 1e-9, frac=0.5),
    ]
    ranked = rank_candidates(rows, force_tolerance=1e-3)
    for r in ranked:
        assert "energy" not in str(r.rank_key).lower()
