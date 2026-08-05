"""Paired (within seed×epoch) recipe contrast over a pre-screen receipt.

Global ranking is dominated by the seed block factor, so recipe signal has to be
recovered by pairing. Pure functions over the receipt dict — no GPU, no torch,
no DFT. Energy is structurally forbidden as a metric (T1).
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.pipeline.paired_recipe_contrast import (  # noqa: E402
    PRE_SCREEN_T1_METRIC_KEYS,
    PairedContrastError,
    contrast_pair,
    index_blocks,
    paired_recipe_contrast,
    variance_decomposition,
)

_RMSD = "mean_rmsd_to_reference_angstrom"
_STEPS = "mean_aimnet2_steps_to_gau_loose"
_FRMSE = "mean_force_rmse_at_reference_ev_per_a"


def _row(run_id: str, seed: int, epoch: int, rmsd: float, steps: float = 100.0,
         frmse: float = 0.1) -> dict[str, Any]:
    return {
        "run_id": run_id,
        "seed": seed,
        "epoch": epoch,
        "hard_gates_passed": True,
        _RMSD: rmsd,
        _STEPS: steps,
        _FRMSE: frmse,
    }


def _two_recipe_grid(delta: float, seed_offsets: dict[int, float]) -> list[dict[str, Any]]:
    """A grid where recipe B is uniformly ``delta`` worse, on top of a seed offset.

    The seed offset is the block effect that global ranking cannot see through.
    """

    rows: list[dict[str, Any]] = []
    for seed, off in seed_offsets.items():
        for epoch in (10, 30, 70, 120):
            rows.append(_row("recipe_a", seed, epoch, 0.12 + off))
            rows.append(_row("recipe_b", seed, epoch, 0.12 + off + delta))
    return rows


# --- index_blocks ------------------------------------------------------------


def test_index_blocks_groups_by_seed_and_epoch() -> None:
    rows = _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    assert len(blocks) == 8  # 2 seeds x 4 epochs
    assert (20260730, 10) in blocks
    assert sorted(blocks[(20260730, 10)]) == ["recipe_a", "recipe_b"]
    assert blocks[(20260730, 10)]["recipe_a"][_RMSD] == pytest.approx(0.12)


def test_index_blocks_rejects_duplicate_cells() -> None:
    rows = [_row("recipe_a", 1, 10, 0.1), _row("recipe_a", 1, 10, 0.2)]
    with pytest.raises(PairedContrastError, match="duplicate"):
        index_blocks(rows, metric_keys=(_RMSD,))


def test_index_blocks_rejects_non_finite_metric() -> None:
    rows = [_row("recipe_a", 1, 10, math.inf)]
    with pytest.raises(PairedContrastError, match="non-finite"):
        index_blocks(rows, metric_keys=(_RMSD,))


def test_index_blocks_rejects_missing_metric() -> None:
    row = _row("recipe_a", 1, 10, 0.1)
    del row[_RMSD]
    with pytest.raises(PairedContrastError, match="missing metric"):
        index_blocks([row], metric_keys=(_RMSD,))


def test_index_blocks_refuses_energy_metrics() -> None:
    """T1: frame energy loss must never rank or contrast checkpoints."""

    rows = [{"run_id": "a", "seed": 1, "epoch": 10, "weighted_energy_mse": 8.5}]
    with pytest.raises(PairedContrastError, match="energy"):
        index_blocks(rows, metric_keys=("weighted_energy_mse",))


def test_index_blocks_can_drop_hard_gate_failures() -> None:
    rows = _two_recipe_grid(0.005, {1: 0.0})
    rows[0]["hard_gates_passed"] = False
    kept = index_blocks(rows, metric_keys=(_RMSD,), require_hard_gates=True)
    # the (1, 10) cell now has only recipe_b
    assert sorted(kept[(1, 10)]) == ["recipe_b"]


# --- contrast_pair -----------------------------------------------------------


def test_contrast_pair_recovers_uniform_effect_through_seed_offset() -> None:
    """The whole point: a +0.005 recipe effect hidden under a +0.06 seed offset."""

    rows = _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06, 20260732: 0.03})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    assert out["paired_block_count"] == 12
    assert out["mean_delta"] == pytest.approx(-0.005)
    assert out["a_better_block_count"] == 12
    assert out["b_better_block_count"] == 0
    assert out["sign_consistency"] == 1.0
    assert out["better_run_id"] == "recipe_a"
    # 12 blocks all one way -> two-sided sign test well below 0.05
    assert out["sign_test_p_two_sided"] < 0.001


def test_contrast_pair_reports_no_effect_when_signs_scatter() -> None:
    rows: list[dict[str, Any]] = []
    for i, epoch in enumerate((10, 30, 70, 120)):
        rows.append(_row("recipe_a", 1, epoch, 0.12))
        rows.append(_row("recipe_b", 1, epoch, 0.12 + (0.004 if i % 2 else -0.004)))
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    assert out["mean_delta"] == pytest.approx(0.0)
    assert out["sign_consistency"] == 0.5
    assert out["sign_test_p_two_sided"] == pytest.approx(1.0)
    assert out["better_run_id"] is None


def test_contrast_pair_uses_only_blocks_holding_both_recipes() -> None:
    """Ragged epoch grids are normal; unpaired cells must be dropped, not imputed."""

    rows = [
        _row("recipe_a", 1, 10, 0.12),
        _row("recipe_b", 1, 10, 0.13),
        _row("recipe_a", 1, 30, 0.11),  # no recipe_b at epoch 30
        _row("recipe_b", 1, 70, 0.14),  # no recipe_a at epoch 70
    ]
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    assert out["paired_block_count"] == 1
    assert out["unpaired_block_count"] == 2


def test_contrast_pair_relative_delta_is_percent_of_block_mean() -> None:
    rows = [_row("recipe_a", 1, 10, 0.10), _row("recipe_b", 1, 10, 0.12)]
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    # delta -0.02 over block mean 0.11 -> -18.18%
    assert out["mean_relative_delta_percent"] == pytest.approx(-100 * 0.02 / 0.11)


def test_contrast_pair_fails_closed_with_no_shared_blocks() -> None:
    rows = [_row("recipe_a", 1, 10, 0.12), _row("recipe_b", 2, 30, 0.13)]
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    with pytest.raises(PairedContrastError, match="no paired blocks"):
        contrast_pair(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)


# --- variance_decomposition --------------------------------------------------


def test_variance_decomposition_detects_block_dominated_design() -> None:
    """Seed offsets 0/0.06 dwarf a 0.005 recipe effect -> ratio >> 1."""

    rows = _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = variance_decomposition(blocks, metric_key=_RMSD)
    assert out["mean_within_block_spread"] == pytest.approx(0.005)
    assert out["between_block_spread"] == pytest.approx(0.06)
    assert out["between_over_within_ratio"] == pytest.approx(12.0)
    assert out["block_dominated"] is True


def test_variance_decomposition_flags_recipe_dominated_design() -> None:
    rows = _two_recipe_grid(0.05, {20260730: 0.0, 20260731: 0.001})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = variance_decomposition(blocks, metric_key=_RMSD)
    assert out["between_over_within_ratio"] < 1.0
    assert out["block_dominated"] is False


# --- paired_recipe_contrast (top level) --------------------------------------


def test_paired_recipe_contrast_covers_all_pairs_and_metrics() -> None:
    rows = _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06})
    report = paired_recipe_contrast(rows)
    assert report["run_ids"] == ["recipe_a", "recipe_b"]
    assert report["metric_keys"] == list(PRE_SCREEN_T1_METRIC_KEYS)
    # one pair x three metrics
    assert len(report["contrasts"]) == 3
    assert report["final_model_selected"] is False
    assert report["energy_loss_used_for_ranking"] is False


def test_paired_recipe_contrast_orders_metrics_by_t1_priority() -> None:
    assert list(PRE_SCREEN_T1_METRIC_KEYS) == [_RMSD, _STEPS, _FRMSE]
    rows = _two_recipe_grid(0.005, {1: 0.0})
    report = paired_recipe_contrast(rows)
    assert [c["metric_key"] for c in report["contrasts"]] == [_RMSD, _STEPS, _FRMSE]


def test_paired_recipe_contrast_never_selects_a_model() -> None:
    rows = _two_recipe_grid(0.005, {1: 0.0, 2: 0.06})
    report = paired_recipe_contrast(rows)
    assert report["final_model_selected"] is False
    assert report["selection_authority"].endswith("not_final")
    for c in report["contrasts"]:
        assert "final_model_selected" not in c or c["final_model_selected"] is False


def test_paired_recipe_contrast_requires_two_recipes() -> None:
    rows = [_row("only_one", 1, 10, 0.12)]
    with pytest.raises(PairedContrastError, match="at least two run_ids"):
        paired_recipe_contrast(rows)


# --- receipt I/O wrapper -----------------------------------------------------


def test_run_paired_contrast_writes_receipt_beside_campaign(tmp_path: Path) -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import run_paired_contrast_for_screen

    d = tmp_path / "live_phase1_v002"
    d.mkdir()
    (d / "screen_campaign.json").write_text(
        json.dumps(
            {
                "screen_id": "live_phase1_v002",
                "batch_id": "g001",
                "status": "PRE_SCREEN_PASS",
                "candidate_count": 8,
                "ranked": _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06}),
            }
        ),
        encoding="utf-8",
    )
    report = run_paired_contrast_for_screen(d / "screen_campaign.json")
    out = d / "paired_recipe_contrast.json"
    assert out.is_file()
    assert report["screen_id"] == "live_phase1_v002"
    assert report["final_model_selected"] is False
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["schema"] == "nhc0801-paired-recipe-contrast-v1"


def test_run_paired_contrast_refuses_dry_run_campaign(tmp_path: Path) -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import run_paired_contrast_for_screen

    d = tmp_path / "dry_campaign_v1"
    d.mkdir()
    (d / "screen_campaign.json").write_text(
        json.dumps(
            {
                "screen_id": "dry_campaign_v1",
                "ranked": _two_recipe_grid(0.005, {1: 0.0}),
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(PairedContrastError, match="dry-run"):
        run_paired_contrast_for_screen(d / "screen_campaign.json")


def test_run_paired_contrast_fails_closed_on_missing_file(tmp_path: Path) -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import run_paired_contrast_for_screen

    with pytest.raises(PairedContrastError, match="missing screen campaign"):
        run_paired_contrast_for_screen(tmp_path / "nope.json")


# --- seed-level unit (block-level p is inflated when a seed is flat) ---------


def test_seed_level_contrast_collapses_epochs_within_a_seed() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import contrast_pair_by_seed

    rows = _two_recipe_grid(0.005, {20260730: 0.0, 20260731: 0.06, 20260732: 0.03})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair_by_seed(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    assert out["independent_unit"] == "seed"
    assert out["seed_count"] == 3  # not 12 blocks
    assert out["a_better_seed_count"] == 3
    assert out["mean_delta"] == pytest.approx(-0.005)
    assert out["better_run_id"] == "recipe_a"


def test_seed_level_sign_test_floors_at_quarter_for_three_seeds() -> None:
    """Unanimous at n=3 still cannot reach p<0.05 — the report must say so."""

    from nhc_deprot.pipeline.paired_recipe_contrast import contrast_pair_by_seed

    rows = _two_recipe_grid(0.005, {1: 0.0, 2: 0.06, 3: 0.03})
    blocks = index_blocks(rows, metric_keys=(_RMSD,))
    out = contrast_pair_by_seed(blocks, "recipe_a", "recipe_b", metric_key=_RMSD)
    assert out["sign_consistency"] == 1.0
    assert out["sign_test_p_two_sided"] == pytest.approx(0.25)
    assert out["sign_test_p_floor"] == pytest.approx(0.25)


def test_report_flags_a_seed_whose_metric_is_flat_across_epochs() -> None:
    rows = [
        _row("recipe_a", 1, 10, 0.120), _row("recipe_b", 1, 10, 0.125),
        _row("recipe_a", 1, 120, 0.180), _row("recipe_b", 1, 120, 0.185),
        # seed 2 is frozen: identical at both epochs
        _row("recipe_a", 2, 10, 0.180), _row("recipe_b", 2, 10, 0.185),
        _row("recipe_a", 2, 120, 0.180), _row("recipe_b", 2, 120, 0.185),
    ]
    report = paired_recipe_contrast(rows, metric_keys=(_RMSD,))
    assert report["seed_count"] == 2
    spread = {(r["seed"], r["run_id"]): r for r in report["per_seed_epoch_spread"]}
    assert spread[(1, "recipe_a")][f"{_RMSD}__epoch_spread"] == pytest.approx(0.06)
    assert spread[(2, "recipe_a")][f"{_RMSD}__epoch_spread"] == pytest.approx(0.0)


def test_report_carries_both_block_and_seed_level_contrasts() -> None:
    rows = _two_recipe_grid(0.005, {1: 0.0, 2: 0.06})
    report = paired_recipe_contrast(rows)
    assert len(report["contrasts"]) == 3
    assert len(report["seed_level_contrasts"]) == 3
    for c in report["seed_level_contrasts"]:
        assert c["independent_unit"] == "seed"
    assert any("floors at p=0.25" in n for n in report["notes"])


# --- epoch curve -------------------------------------------------------------


def test_epoch_curve_finds_interior_minimum() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import epoch_curve

    rows = [
        _row("recipe_a", 1, 10, 0.20),
        _row("recipe_a", 1, 20, 0.12),  # interior minimum
        _row("recipe_a", 1, 30, 0.18),
        _row("recipe_a", 1, 40, 0.22),
    ]
    out = epoch_curve(rows, metric_keys=(_RMSD,))
    c = out["curves"][0][_RMSD]
    assert c["argmin_epoch"] == 20
    assert c["minimum_at_first_sampled_epoch"] is False
    assert c["monotonic_increasing"] is False


def test_epoch_curve_flags_monotonic_degradation() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import epoch_curve

    rows = [_row("recipe_a", 1, e, 0.10 + 0.01 * i)
            for i, e in enumerate((10, 20, 30, 40))]
    out = epoch_curve(rows, metric_keys=(_RMSD,))
    c = out["curves"][0][_RMSD]
    assert c["argmin_epoch"] == 10
    assert c["minimum_at_first_sampled_epoch"] is True
    assert c["monotonic_increasing"] is True
    assert c["spread"] == pytest.approx(0.03)


def test_epoch_curve_separates_run_and_seed() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import epoch_curve

    rows = [
        _row("recipe_a", 1, 10, 0.10), _row("recipe_a", 1, 20, 0.11),
        _row("recipe_a", 2, 10, 0.20), _row("recipe_a", 2, 20, 0.19),
        _row("recipe_b", 1, 10, 0.30), _row("recipe_b", 1, 20, 0.29),
    ]
    out = epoch_curve(rows, metric_keys=(_RMSD,))
    assert out["curve_count"] == 3
    keyed = {(c["run_id"], c["seed"]): c for c in out["curves"]}
    assert keyed[("recipe_a", 1)][_RMSD]["monotonic_increasing"] is True
    assert keyed[("recipe_a", 2)][_RMSD]["monotonic_decreasing"] is True


def test_epoch_curve_rejects_duplicate_epoch() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import epoch_curve

    rows = [_row("recipe_a", 1, 10, 0.1), _row("recipe_a", 1, 10, 0.2)]
    with pytest.raises(PairedContrastError, match="duplicate epochs"):
        epoch_curve(rows, metric_keys=(_RMSD,))


def test_epoch_curve_refuses_energy_metric() -> None:
    from nhc_deprot.pipeline.paired_recipe_contrast import epoch_curve

    with pytest.raises(PairedContrastError, match="energy"):
        epoch_curve([_row("a", 1, 10, 0.1)], metric_keys=("weighted_energy_mse",))
