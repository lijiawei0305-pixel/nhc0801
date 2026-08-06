"""Multi-replica pre-screen measurement (median steps/RMSD + basin stats)."""

from __future__ import annotations

import pytest

from nhc_deprot.pipeline.pre_screen import (
    CandidateScreenResult,
    CheckpointCandidate,
    PreScreenError,
    SimulatedPreScreenEngine,
    TeacherEndpointReference,
    aggregate_replica_results,
    basin_statistics,
    rank_candidates,
    replica_rng_seed,
    run_pre_screen_campaign,
    screen_checkpoint,
    screen_checkpoint_replicas,
)


def _ref(
    root: str = "R1",
    endpoint: str = "cation",
    n: int = 3,
) -> TeacherEndpointReference:
    elements = tuple(["C"] * n)
    start = tuple((float(i), 0.0, 0.0) for i in range(n))
    ref = tuple((float(i) + 0.01, 0.0, 0.0) for i in range(n))
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


def _shot(
    cand: CheckpointCandidate,
    *,
    rmsd: float,
    steps: float,
    force: float,
    hard: bool = True,
) -> CandidateScreenResult:
    return CandidateScreenResult(
        candidate=cand,
        hard_gates_passed=hard,
        identity_ok=hard,
        topology_preserved=hard,
        gau_loose_converged=hard,
        mean_rmsd_to_reference_angstrom=rmsd,
        mean_aimnet2_steps_to_gau_loose=steps,
        mean_force_rmse_at_reference_ev_per_a=force,
        rank_key=(
            0 if hard else 1,
            force,
            steps,
            rmsd,
            cand.run_id,
            cand.seed,
            cand.epoch,
            cand.checkpoint_id,
        ),
    )


def test_replicas_one_matches_single_shot_as_dict() -> None:
    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    engine = SimulatedPreScreenEngine(
        outcomes={
            "ckpt": {
                "atom0_dx": 0.02,
                "steps": 12,
                "converged": True,
                "forces_at_reference_ev_per_a": [
                    [0.05, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
                ],
            }
        }
    )
    cand = CheckpointCandidate("ckpt", "run_a", 20260730, 10)
    single = screen_checkpoint(engine, cand, refs)
    multi = screen_checkpoint_replicas(engine, cand, refs, replicas=1)
    assert single.as_dict() == multi.as_dict()
    # campaign-level path
    camp1 = run_pre_screen_campaign(
        candidates=[cand],
        references=refs,
        engine=engine,
        write=False,
        shortlist_count=1,
    )
    camp2 = run_pre_screen_campaign(
        candidates=[cand],
        references=refs,
        engine=engine,
        write=False,
        shortlist_count=1,
        replicas=1,
    )
    assert camp1["ranked"][0] == camp2["ranked"][0]
    assert camp1["shortlist_checkpoint_ids"] == camp2["shortlist_checkpoint_ids"]
    assert camp1["final_model_selected"] is False
    assert camp1["energy_loss_used_for_ranking"] is False


def test_bimodal_basin_statistics() -> None:
    # 4 low + 2 high → modal fraction 4/6
    rmsds = [0.12, 0.121, 0.122, 0.123, 0.18, 0.181]
    stats = basin_statistics(rmsds, gap=0.01)
    assert stats["basin_count"] == 2
    assert abs(stats["modal_basin_fraction"] - 4 / 6) < 1e-12
    assert stats["deterministic"] is False


def test_unimodal_deterministic() -> None:
    rmsds = [0.12, 0.121, 0.1205, 0.122]
    stats = basin_statistics(rmsds, gap=0.01)
    assert stats["basin_count"] == 1
    assert stats["modal_basin_fraction"] == 1.0
    assert stats["deterministic"] is True


def test_force_spread_fail_closed() -> None:
    cand = CheckpointCandidate("bad_force", "run_x", 1, 1)
    reps = [
        _shot(cand, rmsd=0.12, steps=10, force=0.05),
        _shot(cand, rmsd=0.12, steps=11, force=0.06),  # different force
    ]
    with pytest.raises(PreScreenError) as ei:
        aggregate_replica_results(
            cand,
            reps,
            replica_epsilon_angstrom=1e-4,
        )
    msg = str(ei.value)
    assert "bad_force" in msg
    assert "force RMSE" in msg


def test_aggregate_median_and_rank_key_order() -> None:
    cand = CheckpointCandidate("c1", "run_x", 7, 10)
    # rmsds: 0.10, 0.20, 0.30 → median 0.20; steps 10,20,30 → 20
    reps = [
        _shot(cand, rmsd=0.10, steps=10, force=0.05),
        _shot(cand, rmsd=0.20, steps=20, force=0.05),
        _shot(cand, rmsd=0.30, steps=30, force=0.05),
    ]
    agg = aggregate_replica_results(
        cand, reps, replica_epsilon_angstrom=1e-4, basin_gap_angstrom=0.05
    )
    assert agg.mean_rmsd_to_reference_angstrom == pytest.approx(0.20)
    assert agg.mean_aimnet2_steps_to_gau_loose == pytest.approx(20.0)
    assert agg.mean_force_rmse_at_reference_ev_per_a == pytest.approx(0.05)
    assert agg.replicas == 3
    assert agg.basin_count == 3  # gaps 0.10 > 0.05
    # rank_key: hard, force, steps, rmsd, ...
    assert agg.rank_key[0] == 0
    assert agg.rank_key[1] == pytest.approx(0.05)
    assert agg.rank_key[2] == pytest.approx(20.0)
    assert agg.rank_key[3] == pytest.approx(0.20)
    assert "energy" not in str(agg.rank_key).lower()
    d = agg.as_dict()
    assert d["final_model_selected"] is False
    assert d["replicas"] == 3
    assert d["modal_basin_fraction"] is not None


def test_replica_rng_seed_reproducible() -> None:
    cand = CheckpointCandidate("c1", "run_x", 20260730, 10)
    a = replica_rng_seed(cand, replica_index=0, base_seed=0)
    b = replica_rng_seed(cand, replica_index=0, base_seed=0)
    c = replica_rng_seed(cand, replica_index=1, base_seed=0)
    assert a == b
    assert a != c


def test_multi_replica_campaign_rank_order_and_epoch_zero() -> None:
    refs = [_ref("R1", "cation"), _ref("R1", "neutral")]
    # Same force; different atom0_dx → different RMSD under simulated engine.
    # For multi-replica with perturbation, RMSD will vary; force fixed.
    engine = SimulatedPreScreenEngine(
        outcomes={
            "ft": {
                "atom0_dx": 0.03,
                "steps": 15,
                "converged": True,
                "forces_at_reference_ev_per_a": [
                    [0.1, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                    [0.1, 0.0, 0.0],
                ],
            },
            "e0": {
                "atom0_dx": 0.01,
                "steps": 20,
                "converged": True,
                "forces_at_reference_ev_per_a": [
                    [0.05, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
                    [0.05, 0.0, 0.0],
                ],
            },
        }
    )
    ft = CheckpointCandidate("ft", "run_ft", 1, 10)
    e0 = CheckpointCandidate(
        "e0",
        "epoch_zero_official",
        0,
        0,
        route_kind="epoch_zero",
    )
    campaign = run_pre_screen_campaign(
        candidates=[ft, e0],
        references=refs,
        engine=engine,
        write=False,
        shortlist_count=1,
        replicas=3,
        replica_epsilon_angstrom=1e-4,
    )
    assert campaign["final_model_selected"] is False
    assert campaign["replicas"] == 3
    assert campaign["epoch_zero_excluded_from_shortlist"] is True
    assert "e0" not in campaign["shortlist_checkpoint_ids"]
    # force better on e0 → e0 ranks first but not shortlisted
    assert campaign["ranked"][0]["checkpoint_id"] == "e0"
    assert campaign["shortlist_checkpoint_ids"] == ["ft"]
    for row in campaign["ranked"]:
        assert "energy" not in str(row.get("rank_key", "")).lower()
        assert row["final_model_selected"] is False
        # multi-replica fields present
        assert row["replicas"] == 3
        assert row["deterministic"] is not None


def test_rank_key_order_force_then_steps_then_rmsd() -> None:
    c_a = CheckpointCandidate("A", "r", 1, 1)
    c_b = CheckpointCandidate("B", "r", 1, 2)
    # A worse force, B better force
    ra = _shot(c_a, rmsd=0.10, steps=5, force=0.2)
    rb = _shot(c_b, rmsd=0.50, steps=50, force=0.05)
    # re-aggregate as multi to set rank_key properly
    aa = aggregate_replica_results(c_a, [ra, ra], replica_epsilon_angstrom=1e-4)
    bb = aggregate_replica_results(c_b, [rb, rb], replica_epsilon_angstrom=1e-4)
    ranked = rank_candidates([aa, bb])
    assert [r.candidate.checkpoint_id for r in ranked] == ["B", "A"]
