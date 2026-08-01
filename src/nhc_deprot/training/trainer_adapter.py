"""Ported for NHC0801 / nhc-deprot.

Source: nhc-deprot-ranker-science-pilot (agent/phase9b-science-pilot, dirty V004 worktree).
Authority for science: /Users/cc/nhc-deprot/mindmap.md first; V004 contracts second.
Do not import production two_endpoint B3LYP/def2-SVP or fmax=0.05 preopt as parent protocol.

Weighted trainer adapter only — no live training entrypoint."""

from __future__ import annotations

import math
import sys
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

SCRIPT_ROOT = Path(__file__).resolve().parent
if str(SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPT_ROOT))

import nhc_deprot.training.weighted_loss as weighted_loss  # noqa: E402

WRITER_SCHEMA: Final = "nhc0801-weighted-training-writer-v1"
# Frame counts are parameterized from the weighted dataset audit — never hardcode pilot 235.


class TrainingWriterError(RuntimeError):
    """Weighted trainer integration failed closed (no live train entrypoint)."""


# Back-compat alias for older pilot imports
V004TrainingWriterError = TrainingWriterError


def _as_float(value: Any) -> float:
    """Convert a scalar NumPy/Torch-like value without importing a runtime."""

    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    item = getattr(current, "item", None)
    number = float(item() if callable(item) else current)
    if not math.isfinite(number):
        raise TrainingWriterError("weighted trainer produced a non-finite scalar")
    return number


def dataset_keys(x_keys: list[str], y_keys: list[str]) -> list[str]:
    """Return the exact keys a V004 SizeGroupedDataset must expose."""

    weighted_x, weighted_y = weighted_loss.loader_keys(x_keys, y_keys)
    return [*weighted_x, *weighted_y]


def build_weighted_loader(
    dataset: Any,
    sampler: Any,
    *,
    x_keys: list[str],
    y_keys: list[str],
    num_workers: int = 0,
    pin_memory: bool = True,
) -> Any:
    """Build a loader whose loss-target mapping includes ``sample_weight``."""

    weighted_x, weighted_y = weighted_loss.loader_keys(x_keys, y_keys)
    loader = dataset.get_loader(
        sampler,
        weighted_x,
        weighted_y,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    if weighted_loss.SAMPLE_WEIGHT_KEY not in weighted_y:
        raise TrainingWriterError("sample_weight did not reach loader targets")
    return loader


@dataclass(frozen=True)
class WeightedStep:
    """One differentiable weighted loss and its non-mutating audit receipt."""

    loss: Any
    receipt: dict[str, float | int | str | bool]


def weighted_training_step(
    model: Callable[[dict[str, Any]], dict[str, Any]],
    batch: tuple[dict[str, Any], dict[str, Any]],
    *,
    prepare_batch: Callable[..., dict[str, Any]],
    split_frame_count: int,
    energy_weight: float,
    forces_weight: float,
    device: str,
) -> WeightedStep:
    """Calculate the V004 training loss without mutating model parameters.

    The returned loss preserves the tensor graph when tensor inputs are used.
    A future authorized trainer is responsible for backward/optimizer actions.
    """

    x, y = batch
    prepared_x = prepare_batch(x, device=device, non_blocking=True)
    prepared_y = prepare_batch(y, device=device, non_blocking=True)
    if weighted_loss.SAMPLE_WEIGHT_KEY not in prepared_y:
        raise TrainingWriterError("training batch omitted sample_weight")
    if weighted_loss.SAMPLE_WEIGHT_KEY in prepared_x:
        raise TrainingWriterError("sample_weight leaked into model inputs")
    predicted = model(prepared_x)
    terms = weighted_loss.weighted_batch_terms(predicted, prepared_y)
    scaled = weighted_loss.scaled_training_loss(
        terms,
        split_frame_count=split_frame_count,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
    )
    return WeightedStep(
        loss=scaled["loss"],
        receipt={
            "schema": WRITER_SCHEMA,
            "stage": "training_step_loss_only",
            "sample_weight_consumed": True,
            "sample_weight_sum": _as_float(scaled["sample_weight_sum"]),
            "batch_size": int(scaled["batch_size"]),
            "split_frame_count": split_frame_count,
            "n_over_b_scale": float(scaled["scale"]),
            "weighted_energy_term": _as_float(scaled["energy"]),
            "weighted_forces_term": _as_float(scaled["forces"]),
            "weighted_loss": _as_float(scaled["loss"]),
            "backward_called": False,
            "optimizer_step_called": False,
        },
    )


def quick_validate(
    model: Callable[[dict[str, Any]], dict[str, Any]],
    loader: Iterable[tuple[dict[str, Any], dict[str, Any]]],
    *,
    prepare_batch: Callable[..., dict[str, Any]],
    energy_weight: float,
    forces_weight: float,
    device: str,
) -> dict[str, float | int | str | bool]:
    """Evaluate exact split-level weighted loss without parameter updates."""

    accumulator = weighted_loss.WeightedEvaluationAccumulator()
    batch_count = 0
    for x, y in loader:
        prepared_x = prepare_batch(x, device=device, non_blocking=True)
        prepared_y = prepare_batch(y, device=device, non_blocking=True)
        if weighted_loss.SAMPLE_WEIGHT_KEY not in prepared_y:
            raise TrainingWriterError("quick-validation batch omitted sample_weight")
        if weighted_loss.SAMPLE_WEIGHT_KEY in prepared_x:
            raise TrainingWriterError("sample_weight leaked into model inputs")
        predicted = model(prepared_x)
        terms = weighted_loss.weighted_batch_terms(predicted, prepared_y)
        accumulator.update(
            energy_numerator=_as_float(terms["energy_numerator"]),
            forces_numerator=_as_float(terms["forces_numerator"]),
            sample_weight_sum=_as_float(terms["sample_weight_sum"]),
            batch_size=int(terms["batch_size"]),
        )
        batch_count += 1
    result = accumulator.finalize(energy_weight=energy_weight, forces_weight=forces_weight)
    return {
        "schema": WRITER_SCHEMA,
        "stage": "quick_validation",
        "sample_weight_consumed": True,
        "weighted_loss": result["weighted_loss"],
        "weighted_energy_mse": result["weighted_energy_mse"],
        "weighted_forces_mse": result["weighted_forces_mse"],
        "sample_weight_sum": result["sample_weight_sum"],
        "sample_count": int(result["sample_count"]),
        "batch_count": batch_count,
        "backward_called": False,
        "optimizer_step_called": False,
        "checkpoint_selection_permitted": False,
    }
