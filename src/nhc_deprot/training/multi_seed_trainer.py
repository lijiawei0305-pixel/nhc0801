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
from collections.abc import Callable, Sequence
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
    """One training step and one eval pass (live torch or dry numpy).

    Live backends that can persist weights should also implement
    ``export_checkpoint(path) -> dict``; ``run_one_seed`` calls it at each
    checkpoint interval when ``dry_run=False``. Dry-run backends omit it.
    """

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
    """In-memory dry-run backend using SimulatedResidualModel + weighted_loss.

    Carries a small mutable ``theta`` so resume export/load can be acceptance-
    tested on CPU without torch (20-ep straight vs 10+resume10).
    """

    theta: float = 0.0
    last_completed_epoch: int = 0
    _rng_counter: int = 0

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
        # Fresh seed starts at epoch 1: reset so multi-seed shared backends
        # do not leak theta across seeds. Resume restores theta then continues
        # at epoch > 1 without hitting this reset.
        if int(epoch) == 1:
            self.theta = 0.0
            self._rng_counter = 0
        # Bias shrinks with epoch; theta accumulates so resume must restore it.
        self._rng_counter += 1
        self.theta = float(self.theta) * 0.97 + 0.001 * float(epoch) + 1e-6 * (
            seed % 7
        )
        bias = (
            0.05 * math.exp(-0.15 * epoch) * (1.0 + 0.01 * (seed % 7))
            + 0.01 * float(self.theta)
        )
        self.last_completed_epoch = int(epoch)
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
            "theta": float(self.theta),
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

    def export_resume_checkpoint(
        self,
        path: Path,
        *,
        epoch: int,
        best_epoch: int | None = None,
        best_validation_weighted_loss: float | None = None,
        epochs_since_best: int = 0,
        train_config_digest: str | None = None,
    ) -> dict[str, Any]:
        """CPU resume payload (theta + counter); mirrors live sibling contract."""

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload: dict[str, Any] = {
            "schema": "nhc0801-train-resume-v1",
            "epoch": int(epoch),
            "theta": float(self.theta),
            "rng_counter": int(self._rng_counter),
            "best_epoch": best_epoch,
            "best_validation_weighted_loss": best_validation_weighted_loss,
            "epochs_since_best": int(epochs_since_best),
            "train_config_digest": train_config_digest,
            "backend": "DryRunTrainBackend",
        }
        # JSON text under .resume.pt path (no torch in dry-run).
        path.write_text(
            __import__("json").dumps(payload, sort_keys=True, separators=(",", ":")),
            encoding="utf-8",
        )
        return {
            "path": str(path),
            "bytes": path.stat().st_size,
            "live_resume_written": True,
            "epoch": int(epoch),
        }

    def load_resume_checkpoint(self, path: Path) -> dict[str, Any]:
        """Restore theta/counter; return epoch bookkeeping for run_one_seed."""

        path = Path(path)
        if not path.is_file():
            raise TrainerError(f"missing resume checkpoint: {path}")
        payload = __import__("json").loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict) or payload.get("schema") != "nhc0801-train-resume-v1":
            raise TrainerError(f"bad resume schema: {path}")
        self.theta = float(payload["theta"])
        self._rng_counter = int(payload.get("rng_counter", 0))
        self.last_completed_epoch = int(payload["epoch"])
        return {
            "epoch": int(payload["epoch"]),
            "best_epoch": payload.get("best_epoch"),
            "best_validation_weighted_loss": payload.get(
                "best_validation_weighted_loss"
            ),
            "epochs_since_best": int(payload.get("epochs_since_best", 0)),
        }


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
    early_stop_triggered: bool = False
    early_stop_reason: str | None = None  # "PATIENCE" | "CAP" | None
    right_censored: bool = False
    best_epoch: int | None = None
    best_validation_weighted_loss: float | None = None
    resumed_from_epoch: int | None = None
    resume_merged: bool = False

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_epoch_cap(config: TrainingConfig, *, epochs: int) -> int:
    """Loop upper bound: early-stop max when patience is set, else ``epochs``."""

    if config.early_stop_patience_epochs is not None:
        return int(config.early_stop_max_epochs)
    return int(epochs)


def stopping_policy_provenance(
    config: TrainingConfig,
    *,
    epochs: int,
    actual_epochs_run: int,
) -> dict[str, Any]:
    """P5-B provenance: stopping policy lives **outside** train_config_digest.

    ``epochs`` still appears in the digest today (historical accident). These
    fields make the contract explicit: digest does **not** self-certify training
    length or early-stop policy. Option D (remove epochs from digest) waits for
    a clean seam before the 11-run matrix.
    """

    return {
        "effective_epoch_cap": resolve_epoch_cap(config, epochs=epochs),
        "actual_epochs_run": int(actual_epochs_run),
        "early_stop_patience_epochs": config.early_stop_patience_epochs,
        "early_stop_max_epochs": int(config.early_stop_max_epochs),
        # Contract flag: readers must not treat digest as encoding stop policy.
        "digest_excludes_stopping_policy": True,
    }


def early_stop_update(
    *,
    metric: float,
    best_metric: float,
    epochs_since_best: int,
    patience: int | None,
) -> tuple[float, int, bool]:
    """Classic early-stop bookkeeping on one epoch's metric.

    Returns ``(new_best, new_epochs_since_best, stop_now)``.
    Improvement is strict ``metric < best`` (float compare).
    """

    if not math.isfinite(metric):
        raise TrainerError(f"early-stop metric is non-finite: {metric!r}")
    if patience is None:
        if metric < best_metric:
            return metric, 0, False
        return best_metric, epochs_since_best + 1, False
    if metric < best_metric:
        return metric, 0, False
    since = epochs_since_best + 1
    return best_metric, since, since >= int(patience)


def _merge_prior_seed_receipt(
    *,
    seed_dir: Path,
    start_epoch: int,
    result: "SeedTrainResult",
) -> None:
    """Prepend prior epoch_logs/checkpoints when resuming in the same seed_dir.

    Fail-closed if the prior receipt is missing or does not contain
    ``start_epoch - 1`` (would leave a hole in the curve).
    """

    prior_path = seed_dir / anames.TRAIN_SEED_RESULT_JSON
    if not prior_path.is_file():
        raise TrainerError(
            f"resume merge requires existing {prior_path.name} with epochs "
            f"covering up to {start_epoch - 1}; file missing"
        )
    prior = __import__("json").loads(prior_path.read_text(encoding="utf-8"))
    if not isinstance(prior, dict):
        raise TrainerError(f"prior seed_result is not a dict: {prior_path}")
    prior_logs = list(prior.get("epoch_logs") or [])
    prior_ckpts = list(prior.get("checkpoints") or [])
    need = start_epoch - 1
    prior_epochs = {
        int(e["epoch"]) for e in prior_logs if e.get("epoch") is not None
    }
    if need not in prior_epochs:
        raise TrainerError(
            f"resume merge: prior seed_result missing epoch {need} "
            f"(have {sorted(prior_epochs)[:8]}... n={len(prior_epochs)})"
        )
    # Keep only epochs strictly before the resumed segment
    keep_logs = [e for e in prior_logs if int(e["epoch"]) < start_epoch]
    keep_ckpts = [c for c in prior_ckpts if int(c["epoch"]) < start_epoch]
    # De-dupe by epoch, prefer later list entries (should be unique)
    def _dedupe(items: list[dict[str, Any]], key: str = "epoch") -> list[dict[str, Any]]:
        by: dict[int, dict[str, Any]] = {}
        for it in items:
            by[int(it[key])] = it
        return [by[k] for k in sorted(by)]

    result.epoch_logs = _dedupe(keep_logs) + list(result.epoch_logs)
    result.checkpoints = _dedupe(keep_ckpts) + list(result.checkpoints)
    result.resumed_from_epoch = need
    result.resume_merged = True


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
    resume_from: Path | None = None,
) -> SeedTrainResult:
    """Train one seed; products under ``train_g00N/runs/<run_id>/seed_<seed>/``."""
    effective_run_id = run_id if run_id is not None else config.run_id
    seed_dir = layout.train_run_seed_dir(train_batch_id, effective_run_id, seed)
    seed_dir.mkdir(parents=True, exist_ok=True)
    result = SeedTrainResult(seed=seed, epochs_run=0)
    last_bias = 0.05
    start_epoch = 1
    patience = config.early_stop_patience_epochs
    epoch_cap = resolve_epoch_cap(config, epochs=epochs)
    best_metric = float("inf")
    best_epoch = 0
    epochs_since_best = 0
    stop_reason: str | None = None
    resume_interval = (
        int(config.resume_checkpoint_interval_epochs)
        if config.resume_checkpoint_interval_epochs is not None
        else int(config.checkpoint_interval_epochs)
    )

    try:
        if patience is not None and not config.quick_validation_each_epoch:
            raise TrainerError(
                "early_stop_patience_epochs requires quick_validation_each_epoch=True"
            )

        if resume_from is not None:
            load_fn = getattr(backend, "load_resume_checkpoint", None)
            if not callable(load_fn):
                raise TrainerError(
                    "resume_from set but backend has no load_resume_checkpoint"
                )
            loaded = load_fn(Path(resume_from))
            if isinstance(loaded, dict):
                last_done = int(loaded.get("epoch", 0))
                if loaded.get("best_validation_weighted_loss") is not None:
                    best_metric = float(loaded["best_validation_weighted_loss"])
                    best_epoch = int(loaded.get("best_epoch", last_done))
                    epochs_since_best = int(loaded.get("epochs_since_best", 0))
            else:
                last_done = int(loaded)
            if last_done < 0:
                raise TrainerError(f"resume epoch invalid: {last_done}")
            start_epoch = last_done + 1
            # P2: merge prior receipt from the same seed_dir before writing.
            _merge_prior_seed_receipt(
                seed_dir=seed_dir,
                start_epoch=start_epoch,
                result=result,
            )

        for epoch in range(start_epoch, epoch_cap + 1):
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

            metric_name = config.early_stop_metric
            metric_val = val_out.get(metric_name)
            stop_now = False
            if patience is not None:
                # P3: fail-closed — missing / non-finite metric must not no-op
                if metric_val is None:
                    raise TrainerError(
                        f"early-stop metric {metric_name!r} missing at epoch {epoch}"
                    )
                try:
                    metric_f = float(metric_val)
                except (TypeError, ValueError) as exc:
                    raise TrainerError(
                        f"early-stop metric {metric_name!r} not floatable at "
                        f"epoch {epoch}: {metric_val!r}"
                    ) from exc
                if not math.isfinite(metric_f):
                    raise TrainerError(
                        f"early-stop metric {metric_name!r} non-finite at "
                        f"epoch {epoch}: {metric_f!r}"
                    )
                best_metric, epochs_since_best, stop_now = early_stop_update(
                    metric=metric_f,
                    best_metric=best_metric,
                    epochs_since_best=epochs_since_best,
                    patience=patience,
                )
                if epochs_since_best == 0:
                    best_epoch = epoch
            elif metric_val is not None and math.isfinite(float(metric_val)):
                # Fixed-epochs path: still track best for receipts, no stop
                best_metric, epochs_since_best, _ = early_stop_update(
                    metric=float(metric_val),
                    best_metric=best_metric,
                    epochs_since_best=epochs_since_best,
                    patience=None,
                )
                if epochs_since_best == 0:
                    best_epoch = epoch

            log = {
                "epoch": epoch,
                "seed": seed,
                "batch_id": train_batch_id,
                "run_id": effective_run_id,
                "train": train_out,
                "quick_validation": val_out,
                "quick_validation_may_select_final_model": False,
                "best_epoch_so_far": best_epoch if best_epoch > 0 else None,
                "epochs_since_best": epochs_since_best,
            }
            result.epoch_logs.append(log)
            result.epochs_run = epoch

            is_last = epoch == epoch_cap
            write_weight = (
                epoch % config.checkpoint_interval_epochs == 0
                or is_last
                or stop_now
            )
            write_resume = (
                epoch % resume_interval == 0
                or is_last
                or stop_now
            )
            if write_weight or write_resume:
                meta_path = seed_dir / anames.train_checkpoint_meta_name(epoch)
                weight_path = seed_dir / anames.train_checkpoint_weight_name(epoch)
                live_weights_written = False
                weight_export: dict[str, Any] | None = None
                export_fn = getattr(backend, "export_checkpoint", None)
                if write_weight and (not dry_run) and callable(export_fn):
                    weight_export = export_fn(weight_path)
                    live_weights_written = bool(
                        weight_export is not None
                        and weight_export.get("live_weights_written", True)
                    )
                    if not weight_path.is_file():
                        raise TrainerError(
                            f"export_checkpoint did not create weight file: {weight_path}"
                        )
                resume_export: dict[str, Any] | None = None
                resume_fn = getattr(backend, "export_resume_checkpoint", None)
                resume_path = seed_dir / anames.train_checkpoint_resume_name(epoch)
                if write_resume and callable(resume_fn):
                    resume_export = resume_fn(
                        resume_path,
                        epoch=epoch,
                        best_epoch=best_epoch if best_epoch > 0 else epoch,
                        best_validation_weighted_loss=(
                            best_metric if math.isfinite(best_metric) else None
                        ),
                        epochs_since_best=epochs_since_best,
                        train_config_digest=train_config_digest,
                    )
                # Meta is written whenever either artifact is written so receipts
                # stay aligned with on-disk siblings.
                if write_weight or resume_export is not None:
                    ckpt: dict[str, Any] = {
                        "schema": CKPT_META_SCHEMA,
                        "batch_id": train_batch_id,
                        "run_id": effective_run_id,
                        "seed": seed,
                        "epoch": epoch,
                        "dry_run": dry_run,
                        "live_weights_written": live_weights_written,
                        "official_base_weight_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
                        "validation_weighted_loss": val_out.get(
                            "validation_weighted_loss"
                        ),
                        "train_weighted_loss": train_out.get("train_weighted_loss"),
                        "checkpoint_selection_permitted": False,
                        "path": str(meta_path),
                        "weight_path": str(weight_path),
                        "weight_basename": anames.train_checkpoint_weight_name(epoch),
                        "resume_path": str(resume_path) if resume_export else None,
                    }
                    if train_config_digest is not None:
                        ckpt["train_config_digest"] = train_config_digest
                    if weight_export is not None:
                        ckpt["weight_export"] = weight_export
                        if weight_export.get("train_config_digest") is not None:
                            ckpt["train_config_digest"] = weight_export[
                                "train_config_digest"
                            ]
                        if weight_export.get("weight_kind") is not None:
                            ckpt["weight_kind"] = weight_export["weight_kind"]
                    if resume_export is not None:
                        ckpt["resume_export"] = resume_export
                    if (is_last or stop_now) and not dry_run and live_weights_written:
                        audit_fn = getattr(backend, "export_raw_audit_sibling", None)
                        if callable(audit_fn) and weight_path.is_file():
                            ckpt["ema_export_audit"] = audit_fn(weight_path)
                    write_json(meta_path, ckpt, overwrite=True)
                    # Shortlist uses weight-bearing checkpoints; include any meta
                    # with a finite val loss so resume-only epochs still contribute
                    # when they also wrote weight, or when dry-run has no weights.
                    result.checkpoints.append(ckpt)

            if stop_now:
                stop_reason = "PATIENCE"
                result.early_stop_triggered = True
                result.early_stop_reason = "PATIENCE"
                result.right_censored = False
                break
        else:
            # Natural exit at cap with early-stop enabled → RIGHT_CENSORED
            if patience is not None:
                stop_reason = "CAP"
                result.early_stop_triggered = False  # P4: cap is not early-stop
                result.early_stop_reason = "CAP"
                result.right_censored = True

        if best_epoch > 0:
            result.best_epoch = best_epoch
            result.best_validation_weighted_loss = (
                best_metric if math.isfinite(best_metric) else None
            )

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
            "early_stop_triggered": result.early_stop_triggered,
            "early_stop_reason": result.early_stop_reason,
            "right_censored": result.right_censored,
            "best_epoch": result.best_epoch,
            "best_validation_weighted_loss": result.best_validation_weighted_loss,
            "resumed_from_epoch": result.resumed_from_epoch,
            "resume_merged": result.resume_merged,
            **stopping_policy_provenance(
                config,
                epochs=epochs,
                actual_epochs_run=int(result.epochs_run),
            ),
            "final_model_selected": False,
            "selection_authority": "quick_validation_shortlist_only_not_final",
            "stop_reason_note": stop_reason,
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
    backend_factory: Callable[[int], TrainBackend] | None = None,
    skip_dataset_audit: bool = False,
    train_batch_id: str = DEFAULT_TRAIN_BATCH_ID,
    run_id: str | None = None,
    require_merge_meta: bool | None = None,
) -> dict[str, Any]:
    """Run multi-seed training skeleton over generation weighted dataset.

    Products land under ``train_<batch>/runs/<run_id>/`` (human: g00N train recipe).
    Legacy ``train_g00N/seed_*`` is not written.

    Live training:
      - requires ``merge_meta.json`` with ``train_val_disjoint=true`` (unless
        ``require_merge_meta=False`` for tests that inject a backend).
      - prefer ``backend_factory(seed)`` so **each seed** reloads the official
        epoch-0 weight (G7 / mindmap step 4). A single shared ``backend`` is
        still accepted for tests / dry-run, but live multi-seed must not chain
        seeds on the same trained weights.
    """
    # Lazy import avoids circular import with live_aimnet2 (which imports TrainerError).
    from nhc_deprot.training.live_aimnet2 import train_config_digest as _train_config_digest
    from nhc_deprot.training.merge_meta import MergeMetaError, assert_merge_meta_ready

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
        if backend_factory is None and (
            backend is None or isinstance(backend, DryRunTrainBackend)
        ):
            raise TrainerError(
                "live training requires backend_factory= or a non-dry TrainBackend"
            )
        # G7: multi-seed live must not chain seeds on one shared backend.
        if backend_factory is None and len(cfg.seeds) > 1:
            raise TrainerError(
                "live multi-seed training requires backend_factory= so each seed "
                "reloads official epoch-0 weights (G7 / mindmap step 4); "
                f"got shared backend with {len(cfg.seeds)} seeds"
            )
        # COMPUTE_DISPATCH §1.4.4: no merge_meta → refuse live train labels.
        # Default: required for live; tests may set require_merge_meta=False.
        need_meta = True if require_merge_meta is None else bool(require_merge_meta)
        if need_meta:
            train_dir = layout.train_batch_dir(train_batch_id)
            try:
                assert_merge_meta_ready(train_dir)
            except MergeMetaError as exc:
                raise TrainerError(f"merge_meta gate: {exc}") from exc
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
        # G7: each seed must start from official epoch-0 weights when a factory
        # is provided (live ablation path). Shared backend is for dry-run/tests.
        if backend_factory is not None:
            seed_backend: TrainBackend = backend_factory(int(seed))
        else:
            assert backend is not None
            seed_backend = backend
        one = run_one_seed(
            layout=layout,
            seed=seed,
            config=cfg,
            train_batches=train_batches,
            val_batches=val_batches,
            train_frame_count=train_n,
            backend=seed_backend,
            epochs=epochs,
            dry_run=dry_run,
            train_batch_id=train_batch_id,
            run_id=effective_run_id,
            train_config_digest=digest,
        )
        seed_payload = {
            **one.as_dict(),
            **stopping_policy_provenance(
                cfg, epochs=epochs, actual_epochs_run=int(one.epochs_run)
            ),
        }
        seed_results.append(seed_payload)
        for ckpt in one.checkpoints:
            all_checkpoints.append({**ckpt, "seed": seed})

    failed = sum(1 for s in seed_results if s.get("status") != "PASS")
    # Campaign-level actual_epochs_run = max across seeds (each seed may early-stop).
    max_actual = max(
        (int(s.get("epochs_run") or 0) for s in seed_results),
        default=0,
    )
    stop_prov = stopping_policy_provenance(
        cfg, epochs=epochs, actual_epochs_run=max_actual
    )
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
        "right_censored": any(bool(s.get("right_censored")) for s in seed_results),
        "early_stop_triggered": any(
            bool(s.get("early_stop_triggered")) for s in seed_results
        ),
        **stop_prov,
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
        "right_censored": any(bool(s.get("right_censored")) for s in seed_results),
        "early_stop_triggered": any(
            bool(s.get("early_stop_triggered")) for s in seed_results
        ),
        **stop_prov,
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
            "digest_excludes_stopping_policy: train_config_digest does not encode "
            "early-stop / epoch-cap (P5-B; see stopping provenance fields)",
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
