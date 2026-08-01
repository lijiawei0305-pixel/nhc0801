"""Ported for NHC0801 / nhc-deprot.

Source: nhc-deprot-ranker-science-pilot (agent/phase9b-science-pilot, dirty V004 worktree).
Authority for science: /Users/cc/nhc-deprot/mindmap.md first; V004 contracts second.
Do not import production two_endpoint B3LYP/def2-SVP or fmax=0.05 preopt as parent protocol.
"""


from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Final

SAMPLE_WEIGHT_KEY: Final = "sample_weight"
WEIGHTING_POLICY: Final = "equal_candidate_then_equal_endpoint_then_uniform_frames"


class WeightedLossError(RuntimeError):
    """The weighted-loss contract is internally inconsistent."""


def loader_keys(x_keys: list[str], y_keys: list[str]) -> tuple[list[str], list[str]]:
    """Expose sample_weight to y while preserving the frozen model inputs and targets."""

    if SAMPLE_WEIGHT_KEY in x_keys:
        raise WeightedLossError("sample_weight must not be a model input")
    if SAMPLE_WEIGHT_KEY in y_keys:
        raise WeightedLossError("sample_weight was added more than once")
    return list(x_keys), [*y_keys, SAMPLE_WEIGHT_KEY]


def weighted_batch_terms(predicted: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
    """Return globally weighted energy/force numerators for one size-grouped batch.

    Operations intentionally use the common NumPy/Torch tensor surface so this
    module can be tested without importing Torch or constructing an AIMNet2 model.
    """

    required_predictions = {"energy", "forces", "_natom"}
    required_truth = {"energy", "forces", SAMPLE_WEIGHT_KEY}
    if not required_predictions.issubset(predicted) or not required_truth.issubset(truth):
        raise WeightedLossError("weighted batch omitted a required tensor")
    energy_error = predicted["energy"] - truth["energy"]
    natom = predicted["_natom"]
    weights = truth[SAMPLE_WEIGHT_KEY].reshape(-1)
    energy_per_sample = ((energy_error**2) / (natom**0.5)).reshape(-1)
    batch_size = int(energy_per_sample.shape[0])
    if batch_size <= 0 or int(weights.shape[0]) != batch_size:
        raise WeightedLossError("sample_weight shape does not match the batch")
    force_error_squared = ((predicted["forces"] - truth["forces"]) ** 2).reshape(batch_size, -1)
    if batch_size == 1:
        force_per_sample = force_error_squared.mean(-1)
    else:
        component_count = int(force_error_squared.shape[-1])
        force_per_sample = force_error_squared.mean(-1) * (natom.reshape(-1) / component_count)
    return {
        "energy_numerator": (weights * energy_per_sample).sum(),
        "forces_numerator": (weights * force_per_sample).sum(),
        "sample_weight_sum": weights.sum(),
        "batch_size": batch_size,
    }


def scaled_training_loss(
    terms: dict[str, Any],
    *,
    split_frame_count: int,
    energy_weight: float,
    forces_weight: float,
) -> dict[str, Any]:
    """Build an unbiased minibatch estimate of the global weighted objective.

    N/B scaling makes uniform weights exactly reproduce the historical batch
    mean, while non-uniform weights retain their frozen global coefficients.
    """

    batch_size = int(terms["batch_size"])
    if split_frame_count <= 0 or batch_size <= 0:
        raise WeightedLossError("weighted loss requires positive split and batch sizes")
    component_weight_sum = energy_weight + forces_weight
    if (
        not math.isfinite(energy_weight)
        or not math.isfinite(forces_weight)
        or energy_weight < 0.0
        or forces_weight < 0.0
        or component_weight_sum <= 0.0
    ):
        raise WeightedLossError("multi-task loss weights are invalid")
    scale = split_frame_count / batch_size
    energy = terms["energy_numerator"] * scale
    forces = terms["forces_numerator"] * scale
    loss = energy * (energy_weight / component_weight_sum) + forces * (
        forces_weight / component_weight_sum
    )
    return {
        "loss": loss,
        "energy": energy,
        "forces": forces,
        "sample_weight_sum": terms["sample_weight_sum"],
        "batch_size": batch_size,
        "scale": scale,
    }


@dataclass
class WeightedEvaluationAccumulator:
    """Accumulate exact split-level weighted quick-validation loss."""

    energy_numerator: float = 0.0
    forces_numerator: float = 0.0
    sample_weight_sum: float = 0.0
    sample_count: int = 0

    def update(
        self,
        *,
        energy_numerator: float,
        forces_numerator: float,
        sample_weight_sum: float,
        batch_size: int,
    ) -> None:
        values = (energy_numerator, forces_numerator, sample_weight_sum)
        if not all(math.isfinite(value) for value in values):
            raise WeightedLossError("weighted evaluation term is non-finite")
        if sample_weight_sum <= 0.0 or batch_size <= 0:
            raise WeightedLossError("weighted evaluation batch is empty")
        self.energy_numerator += energy_numerator
        self.forces_numerator += forces_numerator
        self.sample_weight_sum += sample_weight_sum
        self.sample_count += batch_size

    def finalize(self, *, energy_weight: float, forces_weight: float) -> dict[str, float]:
        if self.sample_count <= 0 or self.sample_weight_sum <= 0.0:
            raise WeightedLossError("weighted evaluation dataset is empty")
        component_weight_sum = energy_weight + forces_weight
        if component_weight_sum <= 0.0:
            raise WeightedLossError("multi-task loss weights are invalid")
        energy = self.energy_numerator / self.sample_weight_sum
        forces = self.forces_numerator / self.sample_weight_sum
        return {
            "weighted_loss": (
                energy * (energy_weight / component_weight_sum)
                + forces * (forces_weight / component_weight_sum)
            ),
            "weighted_energy_mse": energy,
            "weighted_forces_mse": forces,
            "sample_weight_sum": self.sample_weight_sum,
            "sample_count": float(self.sample_count),
        }
