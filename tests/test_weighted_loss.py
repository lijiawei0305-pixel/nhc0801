"""Smoke tests for ported weighted loss (no torch / no chemistry)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.training.weighted_loss import (  # noqa: E402
    WeightedEvaluationAccumulator,
    loader_keys,
    scaled_training_loss,
    weighted_batch_terms,
)


def test_loader_keys_appends_sample_weight() -> None:
    x, y = loader_keys(["coord"], ["energy", "forces"])
    assert x == ["coord"]
    assert y == ["energy", "forces", "sample_weight"]


def test_uniform_weights_match_batch_mean_scale() -> None:
    pred = {
        "energy": np.array([1.0, 2.0]),
        "forces": np.zeros((2, 3)),
        "_natom": np.array([3.0, 3.0]),
    }
    truth = {
        "energy": np.array([1.0, 2.0]),
        "forces": np.zeros((2, 3)),
        "sample_weight": np.array([0.5, 0.5]),
    }
    terms = weighted_batch_terms(pred, truth)
    out = scaled_training_loss(
        terms, split_frame_count=2, energy_weight=1.0, forces_weight=1.0
    )
    assert math.isclose(float(out["loss"]), 0.0, abs_tol=1e-12)


def test_accumulator_global_weighted_mean() -> None:
    acc = WeightedEvaluationAccumulator()
    acc.update(
        energy_numerator=0.2,
        forces_numerator=0.4,
        sample_weight_sum=0.5,
        batch_size=1,
    )
    acc.update(
        energy_numerator=0.1,
        forces_numerator=0.2,
        sample_weight_sum=0.5,
        batch_size=1,
    )
    result = acc.finalize(energy_weight=1.0, forces_weight=1.0)
    assert math.isclose(result["sample_weight_sum"], 1.0)
    assert math.isclose(result["weighted_energy_mse"], 0.3)
    assert math.isclose(result["weighted_forces_mse"], 0.6)


def test_single_sample_batch_force_matches_multi_sample() -> None:
    """B=1 and B=2 must apply the same natom/component_count force scaling.

    size-grouped sampling can emit singleton batches; the old B=1 branch omitted
    the padding correction and disagreed with multi-sample batches by ~natom factor.
    """
    # Flat force components: 2 atoms × 3 = 6 slots (sample 1 has 1 real atom; pad zeros).
    forces_pred = np.array(
        [
            [1.0, 0.0, 0.0, 0.5, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    forces_true = np.zeros_like(forces_pred)
    natom = np.array([2.0, 1.0], dtype=np.float64)
    energy = np.array([0.0, 0.0], dtype=np.float64)
    weights = np.array([0.25, 0.75], dtype=np.float64)

    multi = weighted_batch_terms(
        {"energy": energy, "forces": forces_pred, "_natom": natom},
        {"energy": energy, "forces": forces_true, "sample_weight": weights},
    )

    single_force_nums: list[float] = []
    for i in range(2):
        terms = weighted_batch_terms(
            {
                "energy": energy[i : i + 1],
                "forces": forces_pred[i : i + 1],
                "_natom": natom[i : i + 1],
            },
            {
                "energy": energy[i : i + 1],
                "forces": forces_true[i : i + 1],
                "sample_weight": weights[i : i + 1],
            },
        )
        single_force_nums.append(float(terms["forces_numerator"]))

    assert math.isclose(
        single_force_nums[0] + single_force_nums[1],
        float(multi["forces_numerator"]),
        rel_tol=0.0,
        abs_tol=1e-12,
    )

    # Per-sample: B=1 must match B=2 with one-hot sample weights.
    for i in range(2):
        one_hot = np.zeros(2, dtype=np.float64)
        one_hot[i] = weights[i]
        multi_i = weighted_batch_terms(
            {"energy": energy, "forces": forces_pred, "_natom": natom},
            {"energy": energy, "forces": forces_true, "sample_weight": one_hot},
        )
        assert math.isclose(
            single_force_nums[i],
            float(multi_i["forces_numerator"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        )

    # Non-trivial: scaling must actually fire (not zero / not bare mean).
    component_count = forces_pred.shape[-1]
    mean_sq_0 = float(np.mean(forces_pred[0] ** 2))
    scale_0 = float(natom[0]) / component_count
    expected_0 = mean_sq_0 * scale_0 * float(weights[0])
    assert math.isclose(single_force_nums[0], expected_0, rel_tol=0.0, abs_tol=1e-12)
    assert single_force_nums[0] != mean_sq_0 * float(weights[0])
