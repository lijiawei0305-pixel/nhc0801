"""EMA export audit: prove the ``.pt`` on disk really holds EMA weights.

Metadata (``ema_decay`` / ``ema_enabled``) only echoes config and can never
prove the export succeeded — see ``ema_export_audit``. These tests are pure
numpy: they never import torch or load AIMNet2.
"""

from __future__ import annotations

import math
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from nhc_deprot.generation.artifact_names import (  # noqa: E402
    train_checkpoint_raw_weight_name,
    train_checkpoint_weight_name,
)
from nhc_deprot.training.live_aimnet2 import (  # noqa: E402
    ema_export_audit,
    state_dict_l2_divergence,
)
from nhc_deprot.training.multi_seed_trainer import TrainerError  # noqa: E402

_TRAINED = ("outputs.energy_mlp.0.weight", "outputs.energy_mlp.0.bias")


def _state(weight: list[float], bias: list[float]) -> dict[str, np.ndarray]:
    """A state dict with two trained tensors plus one frozen tensor."""

    return {
        "outputs.energy_mlp.0.weight": np.array(weight, dtype=np.float64),
        "outputs.energy_mlp.0.bias": np.array(bias, dtype=np.float64),
        # frozen — identical in every export, must not be compared
        "aev.radial.centers": np.array([0.5, 1.5, 2.5], dtype=np.float64),
    }


# --- state_dict_l2_divergence ------------------------------------------------


def test_divergence_is_zero_for_identical_states() -> None:
    a = _state([1.0, 2.0], [0.5])
    b = _state([1.0, 2.0], [0.5])
    out = state_dict_l2_divergence(a, b, parameter_names=_TRAINED)
    assert out["total_l2"] == 0.0
    assert out["max_abs_delta"] == 0.0
    assert out["compared_parameter_count"] == 2


def test_divergence_l2_value_is_euclidean_over_named_parameters() -> None:
    a = _state([1.0, 2.0], [0.0])
    b = _state([4.0, 6.0], [0.0])
    out = state_dict_l2_divergence(a, b, parameter_names=_TRAINED)
    # weight delta (3, 4) -> 5 ; bias delta 0 -> total sqrt(25 + 0)
    assert math.isclose(out["total_l2"], 5.0, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(out["max_abs_delta"], 4.0, rel_tol=0, abs_tol=1e-12)
    per = out["per_parameter_l2"]
    assert math.isclose(per["outputs.energy_mlp.0.weight"], 5.0, rel_tol=0, abs_tol=1e-12)
    assert per["outputs.energy_mlp.0.bias"] == 0.0


def test_divergence_ignores_frozen_parameters() -> None:
    """A frozen tensor that differs must not leak into the trained-scope L2."""

    a = _state([1.0, 2.0], [0.5])
    b = _state([1.0, 2.0], [0.5])
    b["aev.radial.centers"] = np.array([99.0, 99.0, 99.0], dtype=np.float64)
    out = state_dict_l2_divergence(a, b, parameter_names=_TRAINED)
    assert out["total_l2"] == 0.0
    assert "aev.radial.centers" not in out["per_parameter_l2"]


def test_divergence_fails_closed_on_missing_parameter() -> None:
    a = _state([1.0, 2.0], [0.5])
    b = _state([1.0, 2.0], [0.5])
    del b["outputs.energy_mlp.0.bias"]
    with pytest.raises(TrainerError, match="missing from"):
        state_dict_l2_divergence(a, b, parameter_names=_TRAINED)


def test_divergence_fails_closed_on_shape_mismatch() -> None:
    a = _state([1.0, 2.0], [0.5])
    b = _state([1.0, 2.0, 3.0], [0.5])
    with pytest.raises(TrainerError, match="shape"):
        state_dict_l2_divergence(a, b, parameter_names=_TRAINED)


def test_divergence_requires_parameter_names() -> None:
    a = _state([1.0], [0.5])
    with pytest.raises(TrainerError, match="parameter_names"):
        state_dict_l2_divergence(a, a, parameter_names=())


# --- ema_export_audit --------------------------------------------------------


def test_audit_passes_when_ema_on_and_exports_differ() -> None:
    ema = _state([1.0, 2.0], [0.5])
    raw = _state([1.4, 2.3], [0.5])
    audit = ema_export_audit(ema, raw, parameter_names=_TRAINED, ema_decay=0.99)
    assert audit["status"] == "EMA_EXPORT_AUDIT_PASS"
    assert audit["ema_enabled"] is True
    assert audit["weights_diverged"] is True
    assert audit["total_l2"] > 0.0


def test_audit_fails_closed_when_ema_on_but_export_equals_raw() -> None:
    """The aliasing bug's signature: metadata says EMA, bytes say raw."""

    same = _state([1.0, 2.0], [0.5])
    with pytest.raises(TrainerError, match="EMA_EXPORT_IS_RAW"):
        ema_export_audit(same, _state([1.0, 2.0], [0.5]), parameter_names=_TRAINED, ema_decay=0.99)


def test_audit_passes_when_ema_off_and_exports_match() -> None:
    same = _state([1.0, 2.0], [0.5])
    audit = ema_export_audit(
        same, _state([1.0, 2.0], [0.5]), parameter_names=_TRAINED, ema_decay=None
    )
    assert audit["status"] == "EMA_EXPORT_AUDIT_PASS"
    assert audit["ema_enabled"] is False
    assert audit["weights_diverged"] is False
    assert audit["total_l2"] == 0.0


def test_audit_fails_closed_when_ema_off_but_exports_differ() -> None:
    """EMA disabled means the swap is a no-op; any divergence is a wiring bug."""

    ema = _state([1.0, 2.0], [0.5])
    raw = _state([9.0, 2.0], [0.5])
    with pytest.raises(TrainerError, match="EMA_EXPORT_UNEXPECTED_DIVERGENCE"):
        ema_export_audit(ema, raw, parameter_names=_TRAINED, ema_decay=None)


def test_audit_reports_decay_and_compared_count() -> None:
    ema = _state([1.0, 2.0], [0.5])
    raw = _state([1.1, 2.1], [0.6])
    audit = ema_export_audit(ema, raw, parameter_names=_TRAINED, ema_decay=0.99)
    assert audit["ema_decay"] == 0.99
    assert audit["compared_parameter_count"] == 2
    assert audit["max_abs_delta"] > 0.0


# --- regression: the snapshot must not alias live parameter storage ----------
#
# ``Tensor.cpu()`` returns *self* when the tensor is already on CPU, so
# ``v.detach().cpu()`` aliases the live parameter. ``_use_ema_weights`` restores
# with ``param.data.copy_(saved)`` — an **in-place** write into that same
# storage — so the aliased snapshot silently becomes raw weights before
# ``torch.save`` runs. ``temporary_array_swap`` rebinds dict entries instead of
# writing in place, so it does *not* reproduce this; the helper below mirrors
# the ``copy_`` semantics with numpy and needs no torch.


@contextmanager
def _inplace_weight_swap(
    live: dict[str, np.ndarray], shadow: dict[str, np.ndarray]
) -> Iterator[None]:
    """numpy mirror of ``_use_ema_weights`` (in-place ``param.data.copy_``)."""

    saved = {k: live[k].copy() for k in shadow if k in live}
    for k, v in shadow.items():
        if k in live:
            np.copyto(live[k], v)
    try:
        yield
    finally:
        for k, v in saved.items():
            np.copyto(live[k], v)


def test_aliasing_snapshot_is_clobbered_by_restore() -> None:
    """Documents the failure mode: a non-copying snapshot loses the EMA values."""

    live = {"w": np.array([1.0, 2.0])}
    shadow = {"w": np.array([9.0, 8.0])}
    aliased: dict[str, np.ndarray] = {}
    with _inplace_weight_swap(live, shadow):
        aliased["w"] = live["w"]  # no copy — the ".cpu() on CPU" case
        np.testing.assert_allclose(aliased["w"], [9.0, 8.0])
    # restore ran in place; the "snapshot" now holds raw weights, not EMA
    np.testing.assert_allclose(aliased["w"], [1.0, 2.0])


def test_copied_snapshot_survives_restore() -> None:
    """The fix: copy out of the swap window (``.clone()`` in the torch path)."""

    live = {"w": np.array([1.0, 2.0])}
    shadow = {"w": np.array([9.0, 8.0])}
    copied: dict[str, np.ndarray] = {}
    with _inplace_weight_swap(live, shadow):
        copied["w"] = live["w"].copy()
    np.testing.assert_allclose(copied["w"], [9.0, 8.0])
    np.testing.assert_allclose(live["w"], [1.0, 2.0])


def test_audit_catches_the_aliased_snapshot() -> None:
    """End-to-end: an aliased export is exactly what the audit must reject."""

    live = {
        "outputs.energy_mlp.0.weight": np.array([1.0, 2.0]),
        "outputs.energy_mlp.0.bias": np.array([0.5]),
    }
    shadow = {
        "outputs.energy_mlp.0.weight": np.array([9.0, 8.0]),
        "outputs.energy_mlp.0.bias": np.array([7.0]),
    }
    on_disk: dict[str, np.ndarray] = {}
    with _inplace_weight_swap(live, shadow):
        on_disk.update(live)  # aliased, i.e. the bug
    raw_snapshot = {k: v.copy() for k, v in live.items()}
    with pytest.raises(TrainerError, match="EMA_EXPORT_IS_RAW"):
        ema_export_audit(on_disk, raw_snapshot, parameter_names=_TRAINED, ema_decay=0.99)


def test_audit_passes_on_a_correctly_copied_export() -> None:
    """Same flow with ``.copy()`` (the ``.clone()`` fix) must audit clean."""

    live = {
        "outputs.energy_mlp.0.weight": np.array([1.0, 2.0]),
        "outputs.energy_mlp.0.bias": np.array([0.5]),
    }
    shadow = {
        "outputs.energy_mlp.0.weight": np.array([9.0, 8.0]),
        "outputs.energy_mlp.0.bias": np.array([7.0]),
    }
    with _inplace_weight_swap(live, shadow):
        on_disk = {k: v.copy() for k, v in live.items()}
    raw_snapshot = {k: v.copy() for k, v in live.items()}
    audit = ema_export_audit(
        on_disk, raw_snapshot, parameter_names=_TRAINED, ema_decay=0.99
    )
    assert audit["status"] == "EMA_EXPORT_AUDIT_PASS"
    assert audit["weights_diverged"] is True


# --- artifact naming ---------------------------------------------------------


def test_raw_sibling_name_matches_weight_name_stem() -> None:
    assert train_checkpoint_weight_name(120) == "epoch_0120.pt"
    assert train_checkpoint_raw_weight_name(120) == "epoch_0120.raw.pt"
    assert train_checkpoint_raw_weight_name(7) == "epoch_0007.raw.pt"
