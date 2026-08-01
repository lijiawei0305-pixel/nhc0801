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
