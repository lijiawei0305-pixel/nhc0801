"""Mindmap step 7 — paired recipe contrast over a pre-screen receipt.

Global pre-screen ranking sorts by ``mean_rmsd`` across every candidate, so when
the seed effect is larger than the recipe effect the ranking stratifies by seed
and the recipe signal is invisible (2026-08-04: top 9 of 49 were all one seed).
Seed is a block factor; the fix is to pair recipes **within** a
``(seed, epoch)`` cell and average the paired differences, which cancels the
block effect exactly.

Pure functions over the receipt dict: no torch, no GPU, no DFT. Screening only —
never final selection (mindmap steps 8–9 keep that authority).
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping, Sequence
from itertools import combinations
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.io_util import load_json_object, write_json

PAIRED_CONTRAST_SCHEMA: Final = "nhc0801-paired-recipe-contrast-v1"
MINDMAP_STEP: Final = 7
SELECTION_AUTHORITY: Final = "pre_screen_paired_contrast_only_not_final"

# T1 ranking order: geometry first, then parent burden, then forces.
PRE_SCREEN_T1_METRIC_KEYS: Final[tuple[str, ...]] = (
    "mean_rmsd_to_reference_angstrom",
    "mean_aimnet2_steps_to_gau_loose",
    "mean_force_rmse_at_reference_ev_per_a",
)

# All T1 metrics are "lower is better"; a negative delta means run_a wins.
_LOWER_IS_BETTER: Final = True

# Ratio above which the design is called block(seed)-dominated.
BLOCK_DOMINANCE_RATIO: Final = 1.0

BlockKey = tuple[int, int]
BlockTable = dict[BlockKey, dict[str, dict[str, float]]]


class PairedContrastError(RuntimeError):
    """Paired contrast failed closed."""


def _reject_energy_keys(metric_keys: Iterable[str]) -> None:
    """T1: frame energy loss must never rank or contrast checkpoints."""

    for key in metric_keys:
        if "energy" in str(key).lower():
            raise PairedContrastError(
                f"metric {key!r} is energy-derived; T1 forbids it for ranking"
            )


def index_blocks(
    ranked: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = PRE_SCREEN_T1_METRIC_KEYS,
    require_hard_gates: bool = False,
) -> BlockTable:
    """Group receipt rows into ``(seed, epoch) -> run_id -> {metric: value}``.

    ``require_hard_gates`` drops rows whose ``hard_gates_passed`` is not True.
    Fails closed on duplicate cells and on missing / non-finite metrics: a
    silently dropped candidate would bias the paired means.
    """

    keys = [str(k) for k in metric_keys]
    if not keys:
        raise PairedContrastError("metric_keys must be non-empty")
    _reject_energy_keys(keys)

    blocks: BlockTable = {}
    for row in ranked:
        if require_hard_gates and row.get("hard_gates_passed") is not True:
            continue
        run_id = row.get("run_id")
        seed = row.get("seed")
        epoch = row.get("epoch")
        if not isinstance(run_id, str) or not run_id:
            raise PairedContrastError(f"row missing run_id: {row!r}")
        if type(seed) is not int or type(epoch) is not int:
            raise PairedContrastError(
                f"row needs int seed and epoch (run_id={run_id!r})"
            )
        cell = blocks.setdefault((seed, epoch), {})
        if run_id in cell:
            raise PairedContrastError(
                f"duplicate cell: run_id={run_id!r} seed={seed} epoch={epoch}"
            )
        values: dict[str, float] = {}
        for key in keys:
            if key not in row:
                raise PairedContrastError(
                    f"row missing metric {key!r} (run_id={run_id!r} seed={seed} "
                    f"epoch={epoch})"
                )
            value = float(row[key])
            if not math.isfinite(value):
                raise PairedContrastError(
                    f"non-finite metric {key!r} (run_id={run_id!r} seed={seed} "
                    f"epoch={epoch})"
                )
            values[key] = value
        cell[run_id] = values
    if not blocks:
        raise PairedContrastError("no rows survived indexing")
    return blocks


def _sign_test_p_two_sided(successes: int, trials: int) -> float:
    """Exact two-sided sign test. Assumption-light and valid at tiny n."""

    if trials <= 0:
        return 1.0
    k = max(int(successes), trials - int(successes))
    tail = sum(math.comb(trials, i) for i in range(k, trials + 1)) / (2**trials)
    return min(1.0, 2.0 * tail)


def contrast_pair(
    blocks: BlockTable,
    run_a: str,
    run_b: str,
    *,
    metric_key: str,
) -> dict[str, Any]:
    """Paired ``run_a - run_b`` contrast for one metric across shared blocks.

    Only ``(seed, epoch)`` cells holding **both** run_ids contribute; ragged
    epoch grids are normal and unpaired cells are dropped, never imputed.
    Reports the mean difference **and** the sign consistency, because at this
    block count a single outlying cell can carry the mean on its own.
    """

    _reject_energy_keys([metric_key])
    if run_a == run_b:
        raise PairedContrastError("run_a and run_b must differ")

    deltas: list[float] = []
    relative: list[float] = []
    paired_keys: list[BlockKey] = []
    unpaired = 0
    for key in sorted(blocks):
        cell = blocks[key]
        if run_a not in cell or run_b not in cell:
            if run_a in cell or run_b in cell:
                unpaired += 1
            continue
        a = cell[run_a][metric_key]
        b = cell[run_b][metric_key]
        delta = a - b
        deltas.append(delta)
        block_mean = (a + b) / 2.0
        if block_mean != 0.0:
            relative.append(100.0 * delta / block_mean)
        paired_keys.append(key)

    n = len(deltas)
    if n == 0:
        raise PairedContrastError(
            f"no paired blocks for {run_a!r} vs {run_b!r} on {metric_key!r}"
        )

    a_better = sum(1 for d in deltas if d < 0.0) if _LOWER_IS_BETTER else 0
    b_better = sum(1 for d in deltas if d > 0.0) if _LOWER_IS_BETTER else 0
    tied = n - a_better - b_better
    decided = a_better + b_better
    mean_delta = sum(deltas) / n
    ordered = sorted(deltas)
    mid = n // 2
    median_delta = (
        ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    )
    sign_consistency = (max(a_better, b_better) / n) if n else 0.0

    better: str | None = None
    if a_better > b_better:
        better = run_a
    elif b_better > a_better:
        better = run_b

    return {
        "run_a": run_a,
        "run_b": run_b,
        "metric_key": metric_key,
        "lower_is_better": _LOWER_IS_BETTER,
        "paired_block_count": n,
        "unpaired_block_count": unpaired,
        "paired_blocks": [{"seed": s, "epoch": e} for s, e in paired_keys],
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "mean_relative_delta_percent": (
            sum(relative) / len(relative) if relative else 0.0
        ),
        "a_better_block_count": a_better,
        "b_better_block_count": b_better,
        "tied_block_count": tied,
        "sign_consistency": sign_consistency,
        "sign_test_p_two_sided": _sign_test_p_two_sided(a_better, decided),
        "better_run_id": better,
    }


def variance_decomposition(
    blocks: BlockTable, *, metric_key: str
) -> dict[str, Any]:
    """Compare recipe spread **inside** a block against spread **across** blocks.

    ``between_over_within_ratio`` > 1 means the ``(seed, epoch)`` block factor
    moves the metric more than the recipe does — i.e. a global ranking over all
    candidates is measuring the block, not the recipe.
    """

    _reject_energy_keys([metric_key])
    within: list[float] = []
    block_means: list[float] = []
    for key in sorted(blocks):
        values = [m[metric_key] for m in blocks[key].values()]
        if not values:
            continue
        block_means.append(sum(values) / len(values))
        if len(values) > 1:
            within.append(max(values) - min(values))
    if not block_means:
        raise PairedContrastError(f"no blocks carry metric {metric_key!r}")

    mean_within = sum(within) / len(within) if within else 0.0
    between = max(block_means) - min(block_means)
    ratio = (between / mean_within) if mean_within > 0.0 else math.inf
    return {
        "metric_key": metric_key,
        "block_count": len(block_means),
        "blocks_with_multiple_recipes": len(within),
        "mean_within_block_spread": mean_within,
        "between_block_spread": between,
        "between_over_within_ratio": ratio,
        "block_dominated": bool(ratio > BLOCK_DOMINANCE_RATIO),
    }


def _per_seed_epoch_spread(
    blocks: BlockTable, metric_keys: Sequence[str]
) -> list[dict[str, Any]]:
    """Per (seed, run_id), how much each metric moves across epochs.

    A spread near zero means training stopped changing anything the pre-screen
    can see, so that seed's blocks carry no independent information.
    """

    gathered: dict[tuple[int, str], dict[str, list[float]]] = {}
    for (seed, _epoch), cell in blocks.items():
        for run_id, values in cell.items():
            slot = gathered.setdefault((seed, run_id), {})
            for key in metric_keys:
                slot.setdefault(key, []).append(values[key])
    out: list[dict[str, Any]] = []
    for (seed, run_id), slot in sorted(gathered.items()):
        row: dict[str, Any] = {
            "seed": seed,
            "run_id": run_id,
            "epoch_count": len(next(iter(slot.values()))) if slot else 0,
        }
        for key, vals in slot.items():
            row[f"{key}__epoch_spread"] = max(vals) - min(vals)
        out.append(row)
    return out


def contrast_pair_by_seed(
    blocks: BlockTable,
    run_a: str,
    run_b: str,
    *,
    metric_key: str,
) -> dict[str, Any]:
    """Same contrast, but with the **seed** as the independent unit.

    ``contrast_pair`` treats every ``(seed, epoch)`` block as independent. When
    a seed's metric barely moves across epochs (2026-08-05: seeds 731/732 span
    <0.001 Å from epoch 10 to 120), its blocks are near-duplicates and the
    block-level sign test overstates the evidence. Averaging over epochs within
    a seed first gives one observation per seed — far fewer, but defensible.

    With 3 seeds the two-sided sign test floors at p=0.25, so a unanimous result
    can never reach p<0.05. Read the direction and the effect size, not the p.
    """

    _reject_energy_keys([metric_key])
    if run_a == run_b:
        raise PairedContrastError("run_a and run_b must differ")

    per_seed: dict[int, list[float]] = {}
    for (seed, _epoch), cell in blocks.items():
        if run_a in cell and run_b in cell:
            delta = cell[run_a][metric_key] - cell[run_b][metric_key]
            per_seed.setdefault(seed, []).append(delta)
    if not per_seed:
        raise PairedContrastError(
            f"no paired seeds for {run_a!r} vs {run_b!r} on {metric_key!r}"
        )

    seed_deltas = {s: sum(v) / len(v) for s, v in sorted(per_seed.items())}
    n = len(seed_deltas)
    a_better = sum(1 for d in seed_deltas.values() if d < 0.0)
    b_better = sum(1 for d in seed_deltas.values() if d > 0.0)
    better: str | None = None
    if a_better > b_better:
        better = run_a
    elif b_better > a_better:
        better = run_b
    return {
        "run_a": run_a,
        "run_b": run_b,
        "metric_key": metric_key,
        "independent_unit": "seed",
        "seed_count": n,
        "per_seed_delta": {str(s): d for s, d in seed_deltas.items()},
        "mean_delta": sum(seed_deltas.values()) / n,
        "a_better_seed_count": a_better,
        "b_better_seed_count": b_better,
        "sign_consistency": max(a_better, b_better) / n,
        "sign_test_p_two_sided": _sign_test_p_two_sided(a_better, a_better + b_better),
        "sign_test_p_floor": _sign_test_p_two_sided(0, n),
        "better_run_id": better,
    }


def paired_recipe_contrast(
    ranked: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = PRE_SCREEN_T1_METRIC_KEYS,
    require_hard_gates: bool = False,
) -> dict[str, Any]:
    """Full report: every recipe pair × every T1 metric, plus the block check.

    Reports both the block-level and the seed-level contrast. The seed-level
    one is the defensible unit; the block-level p-value assumes independent
    blocks, which does not hold when a seed's metric is flat across epochs.

    Screening only. The result must never be read as a model selection.
    """

    keys = [str(k) for k in metric_keys]
    blocks = index_blocks(
        ranked, metric_keys=keys, require_hard_gates=require_hard_gates
    )
    run_ids = sorted({rid for cell in blocks.values() for rid in cell})
    if len(run_ids) < 2:
        raise PairedContrastError("need at least two run_ids to contrast")

    contrasts: list[dict[str, Any]] = []
    seed_contrasts: list[dict[str, Any]] = []
    for key in keys:  # T1 priority order is the caller's key order
        for run_a, run_b in combinations(run_ids, 2):
            try:
                contrasts.append(contrast_pair(blocks, run_a, run_b, metric_key=key))
                seed_contrasts.append(
                    contrast_pair_by_seed(blocks, run_a, run_b, metric_key=key)
                )
            except PairedContrastError as exc:
                contrasts.append(
                    {
                        "run_a": run_a,
                        "run_b": run_b,
                        "metric_key": key,
                        "paired_block_count": 0,
                        "skipped_reason": str(exc),
                    }
                )

    seed_count = len({seed for seed, _ in blocks})
    epoch_spread = _per_seed_epoch_spread(blocks, keys)

    return {
        "schema": PAIRED_CONTRAST_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "run_ids": run_ids,
        "metric_keys": keys,
        "block_key": ["seed", "epoch"],
        "block_count": len(blocks),
        "candidate_count": sum(len(cell) for cell in blocks.values()),
        "seed_count": seed_count,
        "per_seed_epoch_spread": epoch_spread,
        "variance_decomposition": [
            variance_decomposition(blocks, metric_key=k) for k in keys
        ],
        "contrasts": contrasts,
        "seed_level_contrasts": seed_contrasts,
        "final_model_selected": False,
        "energy_loss_used_for_ranking": False,
        "selection_authority": SELECTION_AUTHORITY,
        "scientific_validation_required_before_final_selection": True,
        "notes": [
            "seed is a block factor; global pre-screen ranking stratifies by it",
            "paired deltas cancel the block effect; report sign consistency too",
            "block-level p assumes independent blocks — check per_seed_epoch_spread; "
            "a near-zero spread means that seed's blocks are near-duplicates",
            "seed_level_contrasts uses seed as the independent unit (defensible); "
            "with 3 seeds the two-sided sign test floors at p=0.25",
            "screening only — mindmap steps 8–9 keep selection authority",
        ],
    }


PAIRED_CONTRAST_RECEIPT_NAME: Final = "paired_recipe_contrast.json"


def run_paired_contrast_for_screen(
    screen_campaign_path: Path | str,
    *,
    metric_keys: Sequence[str] = PRE_SCREEN_T1_METRIC_KEYS,
    require_hard_gates: bool = True,
    write: bool = True,
) -> dict[str, Any]:
    """Read a ``screen_campaign.json`` and write the paired contrast beside it.

    Reads only; the pre-screen receipt is never modified. Refuses dry-run
    campaigns — a simulated engine cannot support a recipe conclusion.
    """

    path = Path(screen_campaign_path)
    if not path.is_file():
        raise PairedContrastError(f"missing screen campaign: {path}")
    campaign, _raw = load_json_object(path)
    ranked = campaign.get("ranked")
    if not isinstance(ranked, list) or not ranked:
        raise PairedContrastError(f"screen campaign has no ranked rows: {path}")
    screen_id = str(campaign.get("screen_id") or path.parent.name)
    if "dry" in screen_id.lower():
        raise PairedContrastError(
            f"refusing dry-run campaign {screen_id!r}: simulated engine cannot "
            "support a recipe conclusion"
        )

    report = paired_recipe_contrast(
        ranked, metric_keys=metric_keys, require_hard_gates=require_hard_gates
    )
    report["source_screen_campaign"] = str(path)
    report["screen_id"] = screen_id
    report["batch_id"] = campaign.get("batch_id")
    report["source_status"] = campaign.get("status")
    report["source_candidate_count"] = campaign.get("candidate_count")
    report["source_ranking_rule"] = campaign.get("ranking_rule")

    if write:
        out = path.parent / PAIRED_CONTRAST_RECEIPT_NAME
        write_json(out, report, overwrite=True)
        report["receipt_path"] = str(out)
    return report


def epoch_curve(
    ranked: Sequence[Mapping[str, Any]],
    *,
    metric_keys: Sequence[str] = PRE_SCREEN_T1_METRIC_KEYS,
    require_hard_gates: bool = False,
) -> dict[str, Any]:
    """Metric-vs-epoch curve per ``(run_id, seed)``, with the minimum located.

    Answers one question: does the pre-screen metric fall to an **interior**
    minimum at some epoch, or is the best checkpoint simply the earliest one
    sampled (i.e. fine-tuning monotonically degrades the pre-optimizer)?

    ``minimum_at_first_sampled_epoch`` being true does not prove monotonicity —
    it means the sampled window never turned around, so the true optimum may sit
    below the earliest epoch on the grid.
    """

    keys = [str(k) for k in metric_keys]
    _reject_energy_keys(keys)
    series: dict[tuple[str, int], list[tuple[int, dict[str, float]]]] = {}
    for row in ranked:
        if require_hard_gates and row.get("hard_gates_passed") is not True:
            continue
        run_id = row.get("run_id")
        seed = row.get("seed")
        epoch = row.get("epoch")
        if not isinstance(run_id, str) or type(seed) is not int or type(epoch) is not int:
            raise PairedContrastError(f"row needs run_id/seed/epoch: {row!r}")
        values: dict[str, float] = {}
        for key in keys:
            if key not in row:
                raise PairedContrastError(f"row missing metric {key!r}")
            value = float(row[key])
            if not math.isfinite(value):
                raise PairedContrastError(f"non-finite metric {key!r}")
            values[key] = value
        series.setdefault((run_id, seed), []).append((epoch, values))
    if not series:
        raise PairedContrastError("no rows survived epoch-curve indexing")

    curves: list[dict[str, Any]] = []
    for (run_id, seed), points in sorted(series.items()):
        points.sort(key=lambda p: p[0])
        epochs = [e for e, _ in points]
        if len(set(epochs)) != len(epochs):
            raise PairedContrastError(
                f"duplicate epochs for run_id={run_id!r} seed={seed}"
            )
        curve: dict[str, Any] = {
            "run_id": run_id,
            "seed": seed,
            "epochs": epochs,
            "point_count": len(points),
        }
        for key in keys:
            vals = [v[key] for _, v in points]
            best = min(range(len(vals)), key=lambda i: vals[i])
            deltas = [b - a for a, b in zip(vals, vals[1:], strict=False)]
            curve[key] = {
                "values": vals,
                "argmin_epoch": epochs[best],
                "min_value": vals[best],
                "max_value": max(vals),
                "spread": max(vals) - min(vals),
                "minimum_at_first_sampled_epoch": best == 0,
                "monotonic_increasing": all(d >= 0.0 for d in deltas),
                "monotonic_decreasing": all(d <= 0.0 for d in deltas),
            }
        curves.append(curve)

    return {
        "schema": "nhc0801-pre-screen-epoch-curve-v1",
        "mindmap_step": MINDMAP_STEP,
        "metric_keys": keys,
        "curve_count": len(curves),
        "curves": curves,
        "final_model_selected": False,
        "energy_loss_used_for_ranking": False,
        "selection_authority": SELECTION_AUTHORITY,
    }
