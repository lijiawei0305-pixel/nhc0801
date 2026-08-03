"""M7 pure helpers: multi-regex union, EMA math, train_config_digest.

Does **not** load real AIMNet2 weights or import torch.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.training.config import (  # noqa: E402
    TRAINABLE_MLP,
    TRAINABLE_MLP_SHIFT,
    TrainingConfig,
)
from nhc_deprot.training.live_aimnet2 import (  # noqa: E402
    ema_blend_scalar,
    ema_update_arrays,
    match_trainable_parameter_names,
    temporary_array_swap,
    train_config_digest,
)

# Synthetic AIMNet2-like parameter names (no real module).
_FAKE_PARAM_NAMES = (
    "outputs.energy_mlp.0.weight",
    "outputs.energy_mlp.0.bias",
    "outputs.atomic_shift.shifts",
    "outputs.atomic_shift.scale",
    "aev.radial.centers",
    "outputs.srcoulomb.rc",
)


def test_match_trainable_single_regex_mlp_only() -> None:
    matched = match_trainable_parameter_names(_FAKE_PARAM_NAMES, TRAINABLE_MLP)
    assert matched == [
        "outputs.energy_mlp.0.weight",
        "outputs.energy_mlp.0.bias",
    ]
    assert "outputs.atomic_shift.shifts" not in matched


def test_match_trainable_multi_regex_union_larger_than_single() -> None:
    """B3: full tuple must unfreeze atomic_shift as well as energy_mlp."""

    single = match_trainable_parameter_names(_FAKE_PARAM_NAMES, TRAINABLE_MLP)
    multi = match_trainable_parameter_names(_FAKE_PARAM_NAMES, TRAINABLE_MLP_SHIFT)
    assert len(multi) > len(single)
    assert set(single).issubset(set(multi))
    assert "outputs.atomic_shift.shifts" in multi
    assert "outputs.atomic_shift.scale" in multi
    # non-trainable stay frozen under both presets
    assert "aev.radial.centers" not in multi
    assert "outputs.srcoulomb.rc" not in multi


def test_match_trainable_empty_regexes() -> None:
    assert match_trainable_parameter_names(_FAKE_PARAM_NAMES, ()) == []


def test_match_trainable_preserves_input_order() -> None:
    names = (
        "outputs.atomic_shift.shifts",
        "outputs.energy_mlp.0.weight",
        "other",
    )
    matched = match_trainable_parameter_names(names, TRAINABLE_MLP_SHIFT)
    assert matched == [
        "outputs.atomic_shift.shifts",
        "outputs.energy_mlp.0.weight",
    ]


def test_ema_blend_scalar_math() -> None:
    # shadow=0, param=1, decay=0.9 → 0.1
    assert math.isclose(ema_blend_scalar(0.0, 1.0, 0.9), 0.1, rel_tol=0, abs_tol=1e-12)
    # identity when param == shadow
    assert math.isclose(ema_blend_scalar(2.5, 2.5, 0.99), 2.5, rel_tol=0, abs_tol=1e-12)
    # two steps accumulate toward param
    s = 0.0
    for _ in range(3):
        s = ema_blend_scalar(s, 10.0, 0.5)
    # 0 → 5 → 7.5 → 8.75
    assert math.isclose(s, 8.75, rel_tol=0, abs_tol=1e-12)


def test_ema_blend_rejects_bad_decay() -> None:
    import pytest

    with pytest.raises(ValueError, match="ema decay"):
        ema_blend_scalar(0.0, 1.0, 0.0)
    with pytest.raises(ValueError, match="ema decay"):
        ema_blend_scalar(0.0, 1.0, 1.0)


def test_ema_update_arrays_first_and_second_step() -> None:
    shadow: dict[str, np.ndarray] = {}
    current = {
        "w": np.array([1.0, 2.0], dtype=np.float64),
        "b": np.array([0.0], dtype=np.float64),
    }
    ema_update_arrays(shadow, current, decay=0.9)
    # first observation initializes as copy
    np.testing.assert_allclose(shadow["w"], [1.0, 2.0])
    np.testing.assert_allclose(shadow["b"], [0.0])

    next_params = {
        "w": np.array([11.0, 12.0], dtype=np.float64),
        "b": np.array([10.0], dtype=np.float64),
    }
    ema_update_arrays(shadow, next_params, decay=0.9)
    # 0.9 * prev + 0.1 * new
    np.testing.assert_allclose(shadow["w"], [0.9 * 1.0 + 0.1 * 11.0, 0.9 * 2.0 + 0.1 * 12.0])
    np.testing.assert_allclose(shadow["b"], [0.9 * 0.0 + 0.1 * 10.0])


def test_temporary_array_swap_restores() -> None:
    live = {
        "a": np.array([1.0, 2.0]),
        "b": np.array([3.0]),
    }
    shadow = {
        "a": np.array([9.0, 8.0]),
        "b": np.array([7.0]),
    }
    with temporary_array_swap(live, shadow):
        np.testing.assert_allclose(live["a"], [9.0, 8.0])
        np.testing.assert_allclose(live["b"], [7.0])
    np.testing.assert_allclose(live["a"], [1.0, 2.0])
    np.testing.assert_allclose(live["b"], [3.0])


def test_temporary_array_swap_restores_on_exception() -> None:
    live = {"a": np.array([1.0])}
    shadow = {"a": np.array([99.0])}
    try:
        with temporary_array_swap(live, shadow):
            np.testing.assert_allclose(live["a"], [99.0])
            raise RuntimeError("boom")
    except RuntimeError:
        pass
    np.testing.assert_allclose(live["a"], [1.0])


def test_train_config_digest_stable() -> None:
    cfg = TrainingConfig(
        run_id="e1f100_mlp_shift",
        energy_weight=1.0,
        forces_weight=100.0,
        trainable_parameter_regex=TRAINABLE_MLP_SHIFT,
        ema_decay=0.99,
        batch_size=8,
        epochs=120,
    )
    d1 = train_config_digest(cfg)
    d2 = train_config_digest(cfg)
    assert d1 == d2
    assert len(d1) == 64
    assert all(c in "0123456789abcdef" for c in d1)


def test_train_config_digest_changes_with_recipe() -> None:
    base = TrainingConfig(run_id="e1f1_mlp", forces_weight=1.0)
    alt_force = TrainingConfig(run_id="e1f1_mlp", forces_weight=100.0)
    alt_regex = TrainingConfig(
        run_id="e1f1_mlp_shift",
        trainable_parameter_regex=TRAINABLE_MLP_SHIFT,
    )
    alt_ema = TrainingConfig(run_id="e1f1_mlp", ema_decay=None)
    digests = {
        train_config_digest(base),
        train_config_digest(alt_force),
        train_config_digest(alt_regex),
        train_config_digest(alt_ema),
    }
    assert len(digests) == 4


def test_train_config_digest_independent_of_unhashed_fields() -> None:
    """Seeds / learning_rate are not part of the recipe digest keys."""

    a = TrainingConfig(seeds=(1,), learning_rate=1.0e-4)
    b = TrainingConfig(seeds=(2, 3), learning_rate=9.9e-4)
    assert train_config_digest(a) == train_config_digest(b)


def test_export_payload_fields_documented_by_digest_keys() -> None:
    """Sanity: digest helper and TrainingConfig expose the T8 fields."""

    cfg = TrainingConfig()
    d = train_config_digest(cfg)
    # recompute manually to lock canonical shape
    from nhc_deprot.data.io_util import canonical_json, sha256_bytes

    body = {
        "batch_size": int(cfg.batch_size),
        "ema_decay": cfg.ema_decay,
        "energy_weight": float(cfg.energy_weight),
        "epochs": int(cfg.epochs),
        "forces_weight": float(cfg.forces_weight),
        "run_id": cfg.run_id,
        "trainable_parameter_regex": list(cfg.trainable_parameter_regex),
    }
    # canonical_json sorts keys — body key order above is for readability only
    assert sha256_bytes(canonical_json(body)) == d
