"""Frozen multi-seed training hyper-parameters (mindmap steps 4–5).

Matches pilot generation config training block; not a live-train authorization.

M5 knobs (AGENTS T3/T4/T7/T8): run_id, ema_decay, smaller batch/epochs.
Trainable scope presets (_mlp / _mlp_shift) are caller-selected; default is MLP-only.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Any, Final

DEFAULT_SEEDS: Final = (20260730, 20260731, 20260732)

# Trainable-parameter regex presets (caller chooses; default remains MLP-only).
# _mlp_shift unfreezes per-element E0 (atomic_shift) — see AGENTS T3.
TRAINABLE_MLP: Final[tuple[str, ...]] = (r"^outputs\.energy_mlp\.",)
TRAINABLE_MLP_SHIFT: Final[tuple[str, ...]] = (
    r"^outputs\.energy_mlp\.",
    r"^outputs\.atomic_shift",
)

_RUN_ID_RE: Final = re.compile(r"^[a-z0-9_]+$")


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = 120
    # Dry-run skeleton may use a tiny epoch count without changing the frozen default
    # when callers pass dry_run_epochs override.
    run_id: str = "e1f1_mlp"
    ema_decay: float | None = 0.99
    optimizer: str = "torch.optim.RAdam"
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-8
    batch_size: int = 8
    batch_mode: str = "molecules"
    gradient_clip_value: float = 0.4
    trainable_parameter_regex: tuple[str, ...] = TRAINABLE_MLP
    energy_weight: float = 1.0
    forces_weight: float = 1.0
    energy_normalization: str = "sqrt_atom_count"
    force_normalization: str = "per_atom"
    scheduler_type: str = "ReduceLROnPlateau"
    scheduler_factor: float = 0.5
    scheduler_patience_epochs: int = 15
    scheduler_min_lr: float = 1.0e-7
    checkpoint_interval_epochs: int = 10
    quick_validation_each_epoch: bool = True
    quick_validation_may_select_final_model: bool = False
    all_seed_and_checkpoint_outcomes_retained: bool = True
    quick_checkpoint_maximum_count_per_seed: int = 4
    official_base_weight_required: bool = True

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)

    def assert_policy(self) -> None:
        if self.quick_validation_may_select_final_model:
            raise ValueError("quick_validation_may_select_final_model must be false")
        if not self.all_seed_and_checkpoint_outcomes_retained:
            raise ValueError("all outcomes must be retained")
        if self.epochs <= 0 or not self.seeds:
            raise ValueError("epochs and seeds must be positive/non-empty")
        if not self.run_id or not _RUN_ID_RE.fullmatch(self.run_id):
            raise ValueError(
                f"run_id must match ^[a-z0-9_]+$ (got {self.run_id!r})"
            )
        if self.ema_decay is not None and not (0.0 < self.ema_decay < 1.0):
            raise ValueError(
                f"ema_decay must be None or in (0, 1) (got {self.ema_decay!r})"
            )
