"""TrainingConfig defaults and assert_policy (mindmap 4–5 / M5 knobs)."""

from __future__ import annotations

import pytest

from nhc_deprot.training.config import (
    TRAINABLE_MLP,
    TRAINABLE_MLP_SHIFT,
    TrainingConfig,
)


def test_default_knobs() -> None:
    cfg = TrainingConfig()
    assert cfg.run_id == "e1f1_mlp"
    assert cfg.ema_decay == 0.99
    assert cfg.batch_size == 8
    assert cfg.epochs == 120
    # MLP-only remains the default trainable scope
    assert cfg.trainable_parameter_regex == TRAINABLE_MLP
    assert cfg.trainable_parameter_regex == (r"^outputs\.energy_mlp\.",)
    # Untouched frozen knobs (T7 / plan M5)
    assert cfg.learning_rate == 1.0e-4
    assert cfg.optimizer == "torch.optim.RAdam"
    assert cfg.weight_decay == 1.0e-8
    assert cfg.gradient_clip_value == 0.4
    assert cfg.scheduler_type == "ReduceLROnPlateau"
    assert cfg.quick_validation_may_select_final_model is False
    assert cfg.energy_weight == 1.0
    assert cfg.forces_weight == 1.0
    cfg.assert_policy()


def test_trainable_regex_presets() -> None:
    assert TRAINABLE_MLP == (r"^outputs\.energy_mlp\.",)
    assert TRAINABLE_MLP_SHIFT == (
        r"^outputs\.energy_mlp\.",
        r"^outputs\.atomic_shift",
    )
    cfg = TrainingConfig(trainable_parameter_regex=TRAINABLE_MLP_SHIFT)
    assert len(cfg.trainable_parameter_regex) == 2
    cfg.assert_policy()


def test_assert_policy_rejects_illegal_run_id() -> None:
    with pytest.raises(ValueError, match="run_id"):
        TrainingConfig(run_id="E1F1-MLP").assert_policy()
    with pytest.raises(ValueError, match="run_id"):
        TrainingConfig(run_id="").assert_policy()
    with pytest.raises(ValueError, match="run_id"):
        TrainingConfig(run_id="e1f1 mlp").assert_policy()
    with pytest.raises(ValueError, match="run_id"):
        TrainingConfig(run_id="e1f1/mlp").assert_policy()


def test_assert_policy_rejects_illegal_ema_decay() -> None:
    with pytest.raises(ValueError, match="ema_decay"):
        TrainingConfig(ema_decay=0.0).assert_policy()
    with pytest.raises(ValueError, match="ema_decay"):
        TrainingConfig(ema_decay=1.0).assert_policy()
    with pytest.raises(ValueError, match="ema_decay"):
        TrainingConfig(ema_decay=-0.1).assert_policy()
    with pytest.raises(ValueError, match="ema_decay"):
        TrainingConfig(ema_decay=1.5).assert_policy()


def test_assert_policy_accepts_ema_none_and_open_interval() -> None:
    TrainingConfig(ema_decay=None).assert_policy()
    TrainingConfig(ema_decay=0.99).assert_policy()
    TrainingConfig(ema_decay=1.0e-6).assert_policy()
    TrainingConfig(ema_decay=0.99999).assert_policy()


def test_assert_policy_still_rejects_quick_val_final_select() -> None:
    with pytest.raises(ValueError, match="quick_validation_may_select_final_model"):
        TrainingConfig(quick_validation_may_select_final_model=True).assert_policy()


def test_as_dict_includes_new_fields() -> None:
    d = TrainingConfig().as_dict()
    assert d["run_id"] == "e1f1_mlp"
    assert d["ema_decay"] == 0.99
    assert d["batch_size"] == 8
    assert d["epochs"] == 120
