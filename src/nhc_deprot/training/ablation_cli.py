"""Ablation matrix + train-ablation CLI helpers (mindmap 4–5 / plan M11).

Logic lives here; ``scripts/nhc0801_train_ablation.py`` is a thin argparse wrapper.

Phase-1 matrix (plan §4 / AGENTS T4): forces_weight **100** (not 10) is the
force-dominant axis (effective E:F enters the force-led regime). That is a
loss-weight fact only — **force-dominant ≠ better on T1 pre-screen metrics**;
on g001 pilot, T1 force RMSE / RMSD favored f1 over f100 (see
``docs/science/T9_OPERATIONAL_20260805_no_gain_vs_epoch0.md`` §2.3). Shared
frozen knobs come from :class:`TrainingConfig`.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal

from nhc_deprot.generation.layout import (
    DEFAULT_GENERATION_ID,
    GenerationLayout,
    ensure_generation_tree,
    init_generation,
    resolve_layout,
)
from nhc_deprot.training.config import (
    TRAINABLE_MLP,
    TRAINABLE_MLP_SHIFT,
    TrainingConfig,
)
from nhc_deprot.training.multi_seed_trainer import run_multi_seed_training

if TYPE_CHECKING:
    from nhc_deprot.training.multi_seed_trainer import TrainBackend

TrainableScope = Literal["mlp", "mlp_shift"]

# Plan §4 phase-1 matrix — do not substitute forces_weight=10 (AGENTS T4).
DEFAULT_ABLATION_RUN_IDS: Final[tuple[str, ...]] = (
    "e1f1_mlp",
    "e1f100_mlp",
    "e1f1_mlp_shift",
    "e1f100_mlp_shift",
)


@dataclass(frozen=True, slots=True)
class AblationRecipe:
    """One row of the phase-1 ablation matrix."""

    run_id: str
    energy_weight: float
    forces_weight: float
    trainable_scope: TrainableScope

    def as_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "energy_weight": self.energy_weight,
            "forces_weight": self.forces_weight,
            "trainable_scope": self.trainable_scope,
            "trainable_parameter_regex": list(
                TRAINABLE_MLP_SHIFT
                if self.trainable_scope == "mlp_shift"
                else TRAINABLE_MLP
            ),
        }


# Explicit rows (forces 1 vs 100 × mlp vs mlp_shift). No forces_weight=10.
DEFAULT_ABLATION_MATRIX: Final[tuple[AblationRecipe, ...]] = (
    AblationRecipe("e1f1_mlp", 1.0, 1.0, "mlp"),
    AblationRecipe("e1f100_mlp", 1.0, 100.0, "mlp"),
    AblationRecipe("e1f1_mlp_shift", 1.0, 1.0, "mlp_shift"),
    AblationRecipe("e1f100_mlp_shift", 1.0, 100.0, "mlp_shift"),
)

_MATRIX_BY_RUN_ID: Final[dict[str, AblationRecipe]] = {
    r.run_id: r for r in DEFAULT_ABLATION_MATRIX
}


class AblationCliError(RuntimeError):
    """Ablation CLI / matrix failed closed."""


def default_ablation_matrix() -> tuple[AblationRecipe, ...]:
    """Return the frozen phase-1 recipe rows (copy-safe tuple)."""

    return DEFAULT_ABLATION_MATRIX


def recipe_for_run_id(run_id: str) -> AblationRecipe:
    """Look up a known matrix row; raise if unknown."""

    rid = str(run_id).strip()
    if rid not in _MATRIX_BY_RUN_ID:
        known = ", ".join(DEFAULT_ABLATION_RUN_IDS)
        raise AblationCliError(
            f"unknown ablation run_id={rid!r}; known: {known}"
        )
    return _MATRIX_BY_RUN_ID[rid]


def training_config_for_run_id(
    run_id: str,
    *,
    base: TrainingConfig | None = None,
) -> TrainingConfig:
    """Build a :class:`TrainingConfig` for one matrix ``run_id``.

    Shared knobs (batch_size, epochs, ema_decay, seeds, lr, …) come from
    ``base`` or defaults. Recipe-specific: run_id, E/F weights, trainable regex.
    """

    recipe = recipe_for_run_id(run_id)
    regex = (
        TRAINABLE_MLP_SHIFT
        if recipe.trainable_scope == "mlp_shift"
        else TRAINABLE_MLP
    )
    if base is None:
        return TrainingConfig(
            run_id=recipe.run_id,
            energy_weight=recipe.energy_weight,
            forces_weight=recipe.forces_weight,
            trainable_parameter_regex=regex,
        )
    d = base.as_dict()
    d["run_id"] = recipe.run_id
    d["energy_weight"] = recipe.energy_weight
    d["forces_weight"] = recipe.forces_weight
    d["trainable_parameter_regex"] = regex
    return TrainingConfig(**d)


def parse_run_id_list(
    values: Sequence[str] | None,
    *,
    default: Sequence[str] = DEFAULT_ABLATION_RUN_IDS,
) -> tuple[str, ...]:
    """Normalize CLI ``--run-id`` values (repeatable and/or comma-separated)."""

    if not values:
        out = tuple(str(x).strip() for x in default if str(x).strip())
    else:
        parts: list[str] = []
        for v in values:
            for piece in str(v).split(","):
                p = piece.strip()
                if p:
                    parts.append(p)
        out = tuple(parts)
    if not out:
        raise AblationCliError("run_id list is empty")
    # Validate each against matrix
    for rid in out:
        recipe_for_run_id(rid)
    return out


def _build_live_aimnet2_backend(
    *,
    layout: GenerationLayout,
    config: TrainingConfig,
    base_weight: Path,
    seed: int,
    device: str = "cuda",
) -> TrainBackend:
    """Construct :class:`LiveAimnet2TrainBackend` for one seed (live only).

    G7 / mindmap step 4: every seed reloads official epoch-0 weights from
    ``base_weight`` — never continue from a previous seed's trained state.
    """

    # Lazy import: keep dry-run / unit tests free of torch + AIMNet2.
    from nhc_deprot.training.live_aimnet2 import LiveAimnet2TrainBackend

    weight = Path(base_weight)
    if not weight.is_file():
        raise AblationCliError(f"missing --base-weight file: {weight}")
    return LiveAimnet2TrainBackend(
        dataset_root=layout.datasets_dir,
        base_weight=weight,
        config=config,
        seed=int(seed),
        device=device,
    )


def run_train_ablation(
    *,
    layout: GenerationLayout,
    run_ids: Sequence[str],
    train_batch_id: str = "g001",
    dry_run: bool = True,
    dry_run_epochs: int | None = 5,
    aimnet2_train_authorized: bool = False,
    base_config: TrainingConfig | None = None,
    backend: TrainBackend | None = None,
    base_weight: Path | None = None,
    device: str = "cuda",
    require_merge_meta: bool | None = None,
) -> dict[str, Any]:
    """Run multi-seed training once per ``run_id`` (sequential).

    Live train (``dry_run=False``) requires ``aimnet2_train_authorized`` and a
    non-dry :class:`TrainBackend`. Callers may inject ``backend`` (tests /
    custom), or pass ``base_weight`` so **each seed** builds a fresh
    :class:`LiveAimnet2TrainBackend` from the official weight (G7). Dry-run
    ignores both and keeps the multi-seed default :class:`DryRunTrainBackend`.
    """

    campaigns: list[dict[str, Any]] = []
    for rid in run_ids:
        cfg = training_config_for_run_id(rid, base=base_config)
        cfg.assert_policy()
        active_backend: TrainBackend | None = backend
        backend_factory = None
        if not dry_run and active_backend is None:
            if base_weight is None:
                raise AblationCliError(
                    "live ablation requires backend= or base_weight= "
                    "to supply LiveAimnet2TrainBackend"
                )
            weight = Path(base_weight)

            def _factory(
                seed: int,
                *,
                _layout: GenerationLayout = layout,
                _cfg: TrainingConfig = cfg,
                _weight: Path = weight,
                _device: str = device,
            ) -> TrainBackend:
                return _build_live_aimnet2_backend(
                    layout=_layout,
                    config=_cfg,
                    base_weight=_weight,
                    seed=seed,
                    device=_device,
                )

            backend_factory = _factory
        camp = run_multi_seed_training(
            layout=layout,
            config=cfg,
            dry_run=dry_run,
            dry_run_epochs=dry_run_epochs if dry_run else None,
            aimnet2_train_authorized=aimnet2_train_authorized,
            train_batch_id=train_batch_id,
            run_id=rid,
            backend=active_backend,
            backend_factory=backend_factory,
            require_merge_meta=require_merge_meta,
        )
        campaigns.append(
            {
                "run_id": rid,
                "status": camp.get("status"),
                "product_dir": camp.get("product_dir"),
                "checkpoint_count": camp.get("checkpoint_count"),
                "failed_seed_count": camp.get("failed_seed_count"),
                "final_model_selected": camp.get("final_model_selected", False),
                "train_config_digest": camp.get("train_config_digest"),
                "forces_weight": cfg.forces_weight,
                "energy_weight": cfg.energy_weight,
                "trainable_parameter_regex": list(cfg.trainable_parameter_regex),
            }
        )

    failed = sum(
        1
        for c in campaigns
        if not str(c.get("status", "")).endswith("PASS")
    )
    return {
        "schema": "nhc0801-train-ablation-campaign-v1",
        "batch_id": train_batch_id,
        "generation_id": layout.generation_id,
        "dry_run": dry_run,
        "run_ids": list(run_ids),
        "matrix": [recipe_for_run_id(r).as_dict() for r in run_ids],
        "campaigns": campaigns,
        "failed_run_count": failed,
        "final_model_selected": False,
        "status": (
            "DRY_RUN_ABLATION_PASS"
            if dry_run and failed == 0
            else (
                "DRY_RUN_ABLATION_PARTIAL"
                if dry_run
                else (
                    "LIVE_ABLATION_PASS" if failed == 0 else "LIVE_ABLATION_PARTIAL"
                )
            )
        ),
        "notes": [
            "phase-1 matrix: e1f1_mlp / e1f100_mlp / e1f1_mlp_shift / e1f100_mlp_shift",
            "forces_weight=100 is the force-dominant axis (AGENTS T4; not 10)",
            "force-dominant ≠ better on T1 pre-screen metrics (T9_OPERATIONAL §2.3)",
            "quick-val never final-selects; sci-val / pre-screen still required",
        ],
    }


def build_train_ablation_parser() -> argparse.ArgumentParser:
    """Argparse for ``nhc0801_train_ablation.py``."""

    p = argparse.ArgumentParser(
        prog="nhc0801_train_ablation.py",
        description=(
            "Run phase-1 AIMNet2 finetune ablation by run_id "
            "(e1f1_mlp / e1f100_mlp / e1f1_mlp_shift / e1f100_mlp_shift). "
            "Default is dry-run; live train needs authorization + backend."
        ),
    )
    p.add_argument("--generation-id", default=DEFAULT_GENERATION_ID)
    p.add_argument(
        "--nhc0801-root",
        type=Path,
        default=None,
        help="NHC0801 write root (default: <repo>/runs/local_nhc0801)",
    )
    p.add_argument("--batch-id", default="g001", help="Molecular group g00N")
    p.add_argument(
        "--run-id",
        action="append",
        default=None,
        dest="run_ids",
        help=(
            "Recipe run_id (repeatable or comma-separated). "
            "Default: all four phase-1 matrix rows."
        ),
    )
    p.add_argument(
        "--list-matrix",
        action="store_true",
        help="Print the default ablation matrix as JSON and exit",
    )
    p.add_argument(
        "--dry-run",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Dry-run multi-seed trainer (default: true)",
    )
    p.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Dry-run epoch cap (ignored for live unless config epochs smaller)",
    )
    p.add_argument(
        "--live",
        action="store_true",
        help="Request live train (fails closed without authorized backend)",
    )
    p.add_argument(
        "--aimnet2-train-authorized",
        action="store_true",
        help="Gate flag required for live train",
    )
    p.add_argument(
        "--base-weight",
        type=Path,
        default=None,
        help=(
            "Path to official AIMNet2 base weight (.pt). "
            "Required for --live; used to construct LiveAimnet2TrainBackend."
        ),
    )
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device for live LiveAimnet2TrainBackend (default: cuda)",
    )
    return p


def main_train_ablation(
    argv: Sequence[str] | None = None,
    *,
    default_nhc0801_root: Path | None = None,
) -> int:
    """Entry used by the thin script (and unit tests)."""

    parser = build_train_ablation_parser()
    args = parser.parse_args(list(argv) if argv is not None else None)

    if args.list_matrix:
        payload = {
            "matrix": [r.as_dict() for r in DEFAULT_ABLATION_MATRIX],
            "run_ids": list(DEFAULT_ABLATION_RUN_IDS),
            "notes": [
                "forces_weight=100 (not 10) for force-dominant recipes (T4)",
                "force-dominant ≠ better on T1 pre-screen metrics (T9_OPERATIONAL §2.3)",
            ],
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0

    try:
        run_ids = parse_run_id_list(args.run_ids)
    except AblationCliError as exc:
        print(json.dumps({"error": str(exc)}, indent=2), flush=True)
        return 2

    root = args.nhc0801_root
    if root is None:
        root = default_nhc0801_root
    if root is None:
        print(
            json.dumps(
                {
                    "error": "missing --nhc0801-root",
                    "hint": "pass --nhc0801-root or use the scripts/ wrapper",
                },
                indent=2,
            ),
            flush=True,
        )
        return 2

    dry_run = bool(args.dry_run) and not bool(args.live)
    if args.live and not args.aimnet2_train_authorized:
        print(
            json.dumps(
                {
                    "error": "live train requires --aimnet2-train-authorized",
                    "required": [
                        "aimnet2_train_authorized",
                        "non-dry TrainBackend (torch/AIMNet2)",
                        "--base-weight",
                        "resource claim PASS",
                    ],
                },
                indent=2,
            ),
            flush=True,
        )
        return 2

    base_weight: Path | None = None
    if not dry_run:
        if args.base_weight is None:
            print(
                json.dumps(
                    {
                        "error": "live train requires --base-weight",
                        "hint": (
                            "path to aimnet2_wb97m_d3_0.pt "
                            "(constructs LiveAimnet2TrainBackend per recipe)"
                        ),
                        "required": [
                            "aimnet2_train_authorized",
                            "--base-weight",
                            "non-dry TrainBackend (LiveAimnet2TrainBackend)",
                        ],
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 2
        base_weight = Path(args.base_weight)
        # Fail closed before multi-seed: weight path must exist (no silent DryRun).
        # LiveAimnet2TrainBackend is constructed per recipe inside run_train_ablation
        # (see _build_live_aimnet2_backend; matches live_orchestrate base_weight wiring).
        if not base_weight.is_file():
            print(
                json.dumps(
                    {
                        "error": f"missing --base-weight file: {base_weight}",
                        "status": "FAIL",
                    },
                    indent=2,
                ),
                flush=True,
            )
            return 2

    layout = resolve_layout(
        generation_id=args.generation_id, nhc0801_root=root
    )
    if not layout.generation_meta_path().is_file():
        init_generation(generation_id=args.generation_id, nhc0801_root=root)
    else:
        ensure_generation_tree(layout, exist_ok=True)

    try:
        summary = run_train_ablation(
            layout=layout,
            run_ids=run_ids,
            train_batch_id=str(args.batch_id),
            dry_run=dry_run,
            dry_run_epochs=int(args.epochs) if dry_run else None,
            aimnet2_train_authorized=bool(args.aimnet2_train_authorized),
            base_weight=base_weight,
            device=str(args.device),
        )
    except Exception as exc:  # noqa: BLE001 — CLI surface
        print(
            json.dumps(
                {"error": f"{type(exc).__name__}: {exc}", "status": "FAIL"},
                indent=2,
            ),
            flush=True,
        )
        return 1

    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0 if str(summary.get("status", "")).endswith("PASS") else 1
