"""Frozen multi-seed training hyper-parameters (mindmap steps 4–5).

Matches pilot generation config training block; not a live-train authorization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final

DEFAULT_SEEDS: Final = (20260730, 20260731, 20260732)


@dataclass(frozen=True, slots=True)
class TrainingConfig:
    seeds: tuple[int, ...] = DEFAULT_SEEDS
    epochs: int = 200
    # Dry-run skeleton may use a tiny epoch count without changing the frozen default
    # when callers pass dry_run_epochs override.
    optimizer: str = "torch.optim.RAdam"
    learning_rate: float = 1.0e-4
    weight_decay: float = 1.0e-8
    batch_size: int = 32
    batch_mode: str = "molecules"
    gradient_clip_value: float = 0.4
    trainable_parameter_regex: tuple[str, ...] = (r"^outputs\.energy_mlp\.",)
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
