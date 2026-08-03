"""Mindmap steps 4–5 — multi-seed trainer skeleton (dry-run by default).

- Reads weighted NPZ under generation datasets/weighted
- Multi-seed × multi-epoch loop; **retains every outcome**
- Quick validation each epoch; **never final-selects** on frame loss
- Shortlist via tvt_gates.quick_checkpoint_shortlist (screening only)
- Live torch/AIMNet2 train requires aimnet2_train_authorized + injected backend
- Products under ``train_g00N/runs/<run_id>/seed_*/`` (AGENTS T8 / M8)

Dry-run uses a SimulatedResidualModel (numpy) so tests need no torch.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from nhc_deprot.contracts.forbidden_stacks import assert_quick_val_not_final_selector
from nhc_deprot.contracts.tvt_gates import quick_checkpoint_shortlist
from nhc_deprot.data.io_util import sha256_file, write_json
from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256
from nhc_deprot.data.weighted_dataset import REQUIRED_ARRAYS, audit_weighted_dataset
from nhc_deprot.generation import artifact_names as anames
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.weighted_loss import (
    SAMPLE_WEIGHT_KEY,
    WeightedEvaluationAccumulator,
    scaled_training_loss,
    weighted_batch_terms,
)

TRAIN_CAMPAIGN_SCHEMA: Final = "nhc0801-multi-seed-train-campaign-v1"
TRAIN_INFO_SCHEMA: Final = "nhc0801-train-info-v1"
SEED_RECEIPT_SCHEMA: Final = "nhc0801-train-seed-receipt-v1"
CKPT_META_SCHEMA: Final = "nhc0801-checkpoint-meta-v1"
MINDMAP_STEPS: Final = (4, 5)
DEFAULT_TRAIN_BATCH_ID: Final = "g001"


class TrainerError(RuntimeError):
    """Trainer skeleton failed closed."""


class TrainBackend(Protocol):
    """One training step and one eval pass (live torch or dry numpy)."""

    def train_epoch(
        self,
        batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *,
        split_frame_count: int,
        energy_weight: float,
        forces_weight: float,
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        ...

    def evaluate(
        self,
        batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *,
        energy_weight: float,
        forces_weight: float,
    ) -> dict[str, Any]:
        ...

    def step_scheduler(self, val_loss: float) -> None:
        """Advance LR scheduler once per epoch (caller-owned; M7 B2 / M8)."""
        ...


@dataclass
class SimulatedResidualModel:
    """Numpy model: predict label + small seed/epoch-dependent bias (not AIMNet2)."""

    energy_bias: float = 0.0
    force_scale: float = 1.0

    def __call__(self, x: dict[str, Any], truth: dict[str, Any]) -> dict[str, Any]:
        energy = np.asarray(truth["energy"], dtype=np.float64) + self.energy_bias
        forces = np.asarray(truth["forces"], dtype=np.float64) * self.force_scale
        numbers = x.get("numbers")
        if numbers is None:
            raise TrainerError("batch missing numbers for _natom")
        # numbers shape (B, natom) or (B,)
        arr = np.asarray(numbers)
        if arr.ndim == 2:
            natom = np.sum(arr != 0, axis=-1).astype(np.float64)
            # if zeros not used for padding, use full width
            if float(natom.min()) <= 0:
                natom = np.full(arr.shape[0], arr.shape[1], dtype=np.float64)
        else:
            natom = np.ones(energy.shape[0], dtype=np.float64)
        return {"energy": energy, "forces": forces, "_natom": natom}


@dataclass
class DryRunTrainBackend:
    """In-memory dry-run backend using SimulatedResidualModel + weighted_loss."""

    def train_epoch(
        self,
        batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *,
        split_frame_count: int,
        energy_weight: float,
        forces_weight: float,
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        # Bias shrinks with epoch to simulate learning
        bias = 0.05 * math.exp(-0.15 * epoch) * (1.0 + 0.01 * (seed % 7))
        model = SimulatedResidualModel(energy_bias=bias, force_scale=1.0)
        numer_e = 0.0
        numer_f = 0.0
        wsum = 0.0
        n_batches = 0
        last_scale = 0.0
        for x, y in batches:
            pred = model(x, y)
            terms = weighted_batch_terms(pred, y)
            scaled = scaled_training_loss(
                terms,
                split_frame_count=split_frame_count,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
            )
            numer_e += float(terms["energy_numerator"])
            numer_f += float(terms["forces_numerator"])
            wsum += float(terms["sample_weight_sum"])
            last_scale = float(scaled["scale"])
            n_batches += 1
        # Approximate epoch loss as mean of scaled batch losses reconstructed from sums
        if n_batches <= 0:
            raise TrainerError("empty training epoch")
        # Unbiased-ish: use last scale with total numerators / batch count
        # Better: average scaled losses
        losses = []
        model = SimulatedResidualModel(energy_bias=bias, force_scale=1.0)
        for x, y in batches:
            pred = model(x, y)
            terms = weighted_batch_terms(pred, y)
            scaled = scaled_training_loss(
                terms,
                split_frame_count=split_frame_count,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
            )
            losses.append(float(scaled["loss"]))
        return {
            "train_weighted_loss": float(sum(losses) / len(losses)),
            "batch_count": n_batches,
            "sample_weight_sum": wsum,
            "energy_bias": bias,
            "backward_called": False,
            "optimizer_step_called": False,
            "live_parameters_updated": False,
            "n_over_b_scale_example": last_scale,
        }

    def evaluate(
        self,
        batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
        *,
        energy_weight: float,
        forces_weight: float,
        energy_bias: float = 0.0,
    ) -> dict[str, Any]:
        model = SimulatedResidualModel(energy_bias=energy_bias, force_scale=1.0)
        acc = WeightedEvaluationAccumulator()
        for x, y in batches:
            pred = model(x, y)
            terms = weighted_batch_terms(pred, y)
            acc.update(
                energy_numerator=float(terms["energy_numerator"]),
                forces_numerator=float(terms["forces_numerator"]),
                sample_weight_sum=float(terms["sample_weight_sum"]),
                batch_size=int(terms["batch_size"]),
            )
        out = acc.finalize(energy_weight=energy_weight, forces_weight=forces_weight)
        return {
            "validation_weighted_loss": float(out["weighted_loss"]),
            "weighted_energy_mse": float(out["weighted_energy_mse"]),
            "weighted_forces_mse": float(out["weighted_forces_mse"]),
            "sample_weight_sum": float(out["sample_weight_sum"]),
            "sample_count": int(out["sample_count"]),
            "checkpoint_selection_permitted": False,
            "backward_called": False,
        }

    def step_scheduler(self, val_loss: float) -> None:
        """No-op in dry-run (no ReduceLROnPlateau); still called once per epoch."""
        del val_loss


def iter_batches_from_npz_dir(
    datasets_dir: Path,
    split: str,
    *,
    batch_size: int,
) -> tuple[list[tuple[dict[str, Any], dict[str, Any]]], int]:
    """Load NPZ groups; yield simple contiguous batches within each atom-count group."""

    split_dir = datasets_dir / split
    npz_files = sorted(split_dir.glob("*.npz"))
    if not npz_files:
        raise TrainerError(f"no NPZ in {split_dir}")
    batches: list[tuple[dict[str, Any], dict[str, Any]]] = []
    total = 0
    for path in npz_files:
        data = np.load(path, allow_pickle=False)
        try:
            if set(data.files) != REQUIRED_ARRAYS:
                raise TrainerError(f"NPZ key mismatch: {path.name}")
            n = len(data[SAMPLE_WEIGHT_KEY])
            total += n
            for start in range(0, n, batch_size):
                end = min(start + batch_size, n)
                x = {
                    "coord": data["coord"][start:end],
                    "numbers": data["numbers"][start:end],
                    "charge": data["charge"][start:end],
                }
                y = {
                    "energy": data["energy"][start:end],
                    "forces": data["forces"][start:end],
                    SAMPLE_WEIGHT_KEY: data[SAMPLE_WEIGHT_KEY][start:end],
                }
                batches.append((x, y))
        finally:
            data.close()
    if total <= 0:
        raise TrainerError(f"empty split: {split}")
    return batches, total


@dataclass
class SeedTrainResult:
    seed: int
    epochs_run: int
    epoch_logs: list[dict[str, Any]] = field(default_factory=list)
    checkpoints: list[dict[str, Any]] = field(default_factory=list)
    shortlist_epochs: tuple[int, ...] = ()
    status: str = "PASS"
    failure_reason: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def run_one_seed(
    *,
    layout: GenerationLayout,
    seed: int,
    config: TrainingConfig,
    train_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    val_batches: Sequence[tuple[dict[str, Any], dict[str, Any]]],
    train_frame_count: int,
    backend: TrainBackend,
    epochs: int,
    dry_run: bool,
    train_batch_id: str = DEFAULT_TRAIN_BATCH_ID,
    run_id: str | None = None,
    train_config_digest: str | None = None,
) -> SeedTrainResult:
    """Train one seed; products under ``train_g00N/runs/<run_id>/seed_<seed>/``."""
    effective_run_id = run_id if run_id is not None else config.run_id
    seed_dir = layout.train_run_seed_dir(train_batch_id, effective_run_id, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    result = SeedTrainResult(seed=seed, epochs_run=0)
    last_bias = 0.05

    try:
        for epoch in range(1, epochs + 1):
            train_out = backend.train_epoch(
                train_batches,
                split_frame_count=train_frame_count,
                energy_weight=config.energy_weight,
                forces_weight=config.forces_weight,
                seed=seed,
                epoch=epoch,
            )
            last_bias = float(train_out.get("energy_bias", last_bias))
            val_out: dict[str, Any] = {
                "validation_weighted_loss": float("nan"),
                "checkpoint_selection_permitted": False,
            }
            if config.quick_validation_each_epoch:
                if isinstance(backend, DryRunTrainBackend):
                    val_out = backend.evaluate(
                        val_batches,
                        energy_weight=config.energy_weight,
                        forces_weight=config.forces_weight,
                        energy_bias=last_bias,
                    )
                else:
                    val_out = backend.evaluate(
                        val_batches,
                        energy_weight=config.energy_weight,
                        forces_weight=config.forces_weight,
                    )
                if val_out.get("checkpoint_selection_permitted") is True:
                    raise TrainerError("quick validation must not permit final selection")

            # M7 B2 / M8: scheduler is caller-owned — once per epoch after quick-val
            val_loss_raw = val_out.get("validation_weighted_loss")
            if val_loss_raw is not None and math.isfinite(float(val_loss_raw)):
                backend.step_scheduler(float(val_loss_raw))

            log = {
                "epoch": epoch,
                "seed": seed,
                "batch_id": train_batch_id,
                "run_id": effective_run_id,
                "train": train_out,
                "quick_validation": val_out,
                "quick_validation_may_select_final_model": False,
            }
            result.epoch_logs.append(log)
            result.epochs_run = epoch

            # Retain checkpoint meta every interval and always last epoch
            if epoch % config.checkpoint_interval_epochs == 0 or epoch == epochs:
                meta_path = seed_dir / anames.train_checkpoint_meta_name(epoch)
                weight_path = seed_dir / anames.train_checkpoint_weight_name(epoch)
                ckpt: dict[str, Any] = {
                    "schema": CKPT_META_SCHEMA,
                    "batch_id": train_batch_id,
                    "run_id": effective_run_id,
                    "seed": seed,
                    "epoch": epoch,
                    "dry_run": dry_run,
                    "live_weights_written": False,
                    "official_base_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
                    "validation_weighted_loss": val_out.get("validation_weighted_loss"),
                    "train_weighted_loss": train_out.get("train_weighted_loss"),
                    "checkpoint_selection_permitted": False,
                    "path": str(meta_path),
                    "weight_path": str(weight_path),
                    "weight_basename": anames.train_checkpoint_weight_name(epoch),
                }
                if train_config_digest is not None:
                    ckpt["train_config_digest"] = train_config_digest
                write_json(meta_path, ckpt, overwrite=True)
                result.checkpoints.append(ckpt)

        # Shortlist from retained checkpoints (screening only)
        shortlist_input = [
            {
                "epoch": int(c["epoch"]),
                "validation_weighted_loss": float(c["validation_weighted_loss"]),
            }
            for c in result.checkpoints
            if c.get("validation_weighted_loss") is not None
            and math.isfinite(float(c["validation_weighted_loss"]))
        ]
        if shortlist_input:
            result.shortlist_epochs = quick_checkpoint_shortlist(
                shortlist_input,
                maximum_count=config.quick_checkpoint_maximum_count_per_seed,
            )
        result.status = "PASS"
    except Exception as exc:  # noqa: BLE001
        result.status = "FAILED"
        result.failure_reason = f"{type(exc).__name__}: {exc}"

    write_json(
        seed_dir / anames.TRAIN_SEED_RESULT_JSON,
        {
            "schema": SEED_RECEIPT_SCHEMA,
            "batch_id": train_batch_id,
            "run_id": effective_run_id,
            "mindmap_steps": list(MINDMAP_STEPS),
            **result.as_dict(),
            "final_model_selected": False,
            "selection_authority": "quick_validation_shortlist_only_not_final",
        },
        overwrite=True,
    )
    return result


def run_multi_seed_training(
    *,
    layout: GenerationLayout,
    config: TrainingConfig | None = None,
    dry_run: bool = True,
    aimnet2_train_authorized: bool = False,
    dry_run_epochs: int | None = 5,
    backend: TrainBackend | None = None,
    skip_dataset_audit: bool = False,
    train_batch_id: str = DEFAULT_TRAIN_BATCH_ID,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Run multi-seed training skeleton over generation weighted dataset.

    Products land under ``train_<batch>/runs/<run_id>/`` (human: g00N train recipe).
    Legacy ``train_g00N/seed_*`` is not written.
    """
    # Lazy import avoids circular import with live_aimnet2 (which imports TrainerError).
    from nhc_deprot.training.live_aimnet2 import train_config_digest as _train_config_digest

    cfg = config or TrainingConfig()
    effective_run_id = run_id if run_id is not None else cfg.run_id
    # Policy check uses cfg.run_id; if caller overrides run_id, validate the override too.
    if run_id is not None and run_id != cfg.run_id:
        # Temporary config view for digest identity: digest follows the *effective* recipe
        # fields on cfg, but path segment is effective_run_id. Mirror override into a
        # digest-bearing config so train_info self-identifies the disk run_id.
        cfg = TrainingConfig(
            **{
                **cfg.as_dict(),
                "run_id": effective_run_id,
            }
        )
    cfg.assert_policy()
    assert_quick_val_not_final_selector(
        {"quick_validation_may_select_final_model": cfg.quick_validation_may_select_final_model}
    )

    if not dry_run:
        if not aimnet2_train_authorized:
            raise TrainerError("live training requires aimnet2_train_authorized=true")
        if backend is None or isinstance(backend, DryRunTrainBackend):
            raise TrainerError("live training requires a non-dry TrainBackend")
    else:
        backend = backend or DryRunTrainBackend()

    datasets_dir = layout.datasets_dir
    manifest_path = datasets_dir / "manifest.json"
    if not manifest_path.is_file():
        raise TrainerError(
            f"weighted dataset missing at {datasets_dir}; run D3+weighted dry-run first"
        )
    dataset_manifest_sha256 = sha256_file(manifest_path)
    digest = _train_config_digest(cfg)

    if not skip_dataset_audit:
        # Accept NHC0801 writer schema or pilot V004 development schema (reuse path)
        last_err: Exception | None = None
        audit = None
        for schema in (
            "nhc0801-development-dataset-v1",
            "phase9b-aimnet2-development-dataset-v004",
        ):
            try:
                audit = audit_weighted_dataset(
                    datasets_dir,
                    expected_schema=schema,
                )
                last_err = None
                break
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        if audit is None:
            raise TrainerError(f"dataset audit failed: {last_err}")
        if audit.status != "PASS":
            raise TrainerError(f"dataset audit not PASS: {audit.status}")
        if audit.training_started:
            raise TrainerError("dataset manifest claims training_started")

    train_batches, train_n = iter_batches_from_npz_dir(
        datasets_dir, "train", batch_size=cfg.batch_size
    )
    val_batches, val_n = iter_batches_from_npz_dir(
        datasets_dir, "validation", batch_size=cfg.batch_size
    )

    epochs = cfg.epochs
    if dry_run and dry_run_epochs is not None:
        epochs = min(cfg.epochs, max(1, int(dry_run_epochs)))

    run_dir = layout.train_batch_run_dir(train_batch_id, effective_run_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    seed_results: list[dict[str, Any]] = []
    all_checkpoints: list[dict[str, Any]] = []

    for seed in cfg.seeds:
        one = run_one_seed(
            layout=layout,
            seed=seed,
            config=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            train_frame_count=train_n,
            backend=backend,
            epochs=epochs,
            dry_run=dry_run,
            train_batch_id=train_batch_id,
            run_id=effective_run_id,
            train_config_digest=digest,
        )
        seed_results.append(one.as_dict())
        for ckpt in one.checkpoints:
            all_checkpoints.append({**ckpt, "seed": seed})

    failed = sum(1 for s in seed_results if s.get("status") != "PASS")
    train_info: dict[str, Any] = {
        "schema": TRAIN_INFO_SCHEMA,
        "mindmap_steps": list(MINDMAP_STEPS),
        "generation_id": layout.generation_id,
        "batch_id": train_batch_id,
        "run_id": effective_run_id,
        "train_config_digest": digest,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "dataset_manifest_path": str(manifest_path),
        "config": cfg.as_dict(),
        "epochs_effective": epochs,
        "train_frame_count": train_n,
        "validation_frame_count": val_n,
        "seeds": list(cfg.seeds),
        "dry_run": dry_run,
        "official_base_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
        "product_dir": str(run_dir),
    }
    write_json(run_dir / anames.TRAIN_INFO_JSON, train_info, overwrite=True)

    campaign = {
        "schema": TRAIN_CAMPAIGN_SCHEMA,
        "mindmap_steps": list(MINDMAP_STEPS),
        "generation_id": layout.generation_id,
        "batch_id": train_batch_id,
        "run_id": effective_run_id,
        "train_config_digest": digest,
        "dataset_manifest_sha256": dataset_manifest_sha256,
        "product_dir": str(run_dir),
        "product_rel": (
            f"{anames.train_product_dirname(train_batch_id)}/runs/{effective_run_id}"
        ),
        "dry_run": dry_run,
        "live_chemistry": not dry_run,
        "aimnet2_train_authorized": aimnet2_train_authorized,
        "config": cfg.as_dict(),
        "epochs_effective": epochs,
        "train_frame_count": train_n,
        "validation_frame_count": val_n,
        "official_base_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
        "quick_validation_may_select_final_model": False,
        "all_seed_and_checkpoint_outcomes_retained": True,
        "final_model_selected": False,
        "final_model_selection_authority": None,
        "scientific_validation_required_before_final_selection": True,
        "seed_results": seed_results,
        "checkpoint_count": len(all_checkpoints),
        "failed_seed_count": failed,
        "status": (
            "DRY_RUN_TRAIN_PASS"
            if dry_run and failed == 0
            else (
                "DRY_RUN_TRAIN_PARTIAL"
                if dry_run
                else ("LIVE_TRAIN_PASS" if failed == 0 else "LIVE_TRAIN_PARTIAL")
            )
        ),
        "notes": [
            "products under train_g00N/runs/<run_id>/ (not bare seed_* or train/)",
            "quick-val loss only shortlists; never final model",
            "retain all seeds/epochs/failures",
            "dry-run SimulatedResidualModel is not AIMNet2",
            "live train needs authorized backend + claim + frozen epoch-0 baseline",
            "step_scheduler called once per epoch after quick-val (M7 B2)",
        ],
        "final_test_payload_read": False,
    }
    write_json(run_dir / anames.TRAIN_RESULT_JSON, campaign, overwrite=True)
    # Copy under generation logs/ (batch+run scoped basename)
    write_json(
        layout.logs_dir / f"train_{train_batch_id}_{effective_run_id}_result.json",
        campaign,
        overwrite=True,
    )
    return campaign
