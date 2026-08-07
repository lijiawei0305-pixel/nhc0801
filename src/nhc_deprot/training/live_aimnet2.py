"""Live AIMNet2 fine-tune backend (GPU). Requires mlff env + aimnet + torch.

Not imported by default dry-run paths. Resource control is the caller's job
(CUDA_VISIBLE_DEVICES, claim, single-process).

M7: multi-regex trainable union (B3), EMA shadows, scheduler stepped by caller (B2),
train_config_digest on every checkpoint payload.
"""

from __future__ import annotations

import copy
import hashlib
import math
import random
import re
from collections.abc import Iterator, Mapping, MutableMapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Final

import numpy as np

from nhc_deprot.data.io_util import canonical_json, sha256_bytes
from nhc_deprot.data.paths import OFFICIAL_AIMNET2_WEIGHT_SHA256
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.multi_seed_trainer import TrainerError
from nhc_deprot.training.weighted_loss import (
    SAMPLE_WEIGHT_KEY,
    WeightedEvaluationAccumulator,
    scaled_training_loss,
    weighted_batch_terms,
)

SHORT_RANGE_KEY = "outputs.srcoulomb.rc"
LONG_RANGE_KEY = "outputs.lrcoulomb.rc"

# Fields hashed into every checkpoint's train_config_digest (AGENTS T8).
_TRAIN_CONFIG_DIGEST_KEYS: tuple[str, ...] = (
    "run_id",
    "energy_weight",
    "forces_weight",
    "trainable_parameter_regex",
    "ema_decay",
    "batch_size",
    "epochs",
)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def train_config_digest(config: TrainingConfig) -> str:
    """Canonical SHA256 of the recipe identity fields (T8).

    Pure / no torch. Any ``.pt`` payload should carry this so checkpoints
    self-identify the training recipe without re-deriving hyperparams.
    """

    payload: dict[str, Any] = {
        "run_id": config.run_id,
        "energy_weight": float(config.energy_weight),
        "forces_weight": float(config.forces_weight),
        # list (not tuple) so JSON is stable and sort_keys is well-defined
        "trainable_parameter_regex": list(config.trainable_parameter_regex),
        "ema_decay": config.ema_decay,
        "batch_size": int(config.batch_size),
        "epochs": int(config.epochs),
    }
    # keep only the frozen key set (defensive if config grows)
    body = {k: payload[k] for k in _TRAIN_CONFIG_DIGEST_KEYS}
    return sha256_bytes(canonical_json(body))


def match_trainable_parameter_names(
    parameter_names: Sequence[str],
    trainable_regexes: Sequence[str],
) -> list[str]:
    """Return names matching any of the regexes (union). Order follows input.

    Pure helper for multi-regex freeze/unfreeze (B3 / T3 / T8). Does not load
    AIMNet2 weights.
    """

    if not trainable_regexes:
        return []
    patterns = [re.compile(rx) for rx in trainable_regexes]
    matched: list[str] = []
    for name in parameter_names:
        if any(p.search(name) for p in patterns):
            matched.append(name)
    return matched


def ema_blend_scalar(shadow: float, param: float, decay: float) -> float:
    """One EMA step: ``decay * shadow + (1 - decay) * param`` (pure math)."""

    if not (0.0 < decay < 1.0):
        raise ValueError(f"ema decay must be in (0, 1), got {decay!r}")
    return float(decay) * float(shadow) + (1.0 - float(decay)) * float(param)


def ema_update_arrays(
    shadow: MutableMapping[str, np.ndarray],
    current: Mapping[str, np.ndarray],
    decay: float,
) -> None:
    """In-place EMA update over named float arrays (no torch).

    First observation for a name initializes the shadow as a copy of current.
    """

    if not (0.0 < decay < 1.0):
        raise ValueError(f"ema decay must be in (0, 1), got {decay!r}")
    alpha = 1.0 - float(decay)
    d = float(decay)
    for name, value in current.items():
        arr = np.asarray(value, dtype=np.float64)
        if name not in shadow:
            shadow[name] = arr.copy()
        else:
            shadow[name] = d * shadow[name] + alpha * arr


def state_dict_l2_divergence(
    state_a: Mapping[str, Any],
    state_b: Mapping[str, Any],
    *,
    parameter_names: Sequence[str],
) -> dict[str, Any]:
    """L2 distance between two state dicts over ``parameter_names`` (no torch).

    Accepts anything ``np.asarray`` handles, so the same helper compares numpy
    fixtures in tests and CPU torch tensors in the live export path. Frozen
    parameters are excluded by construction: only the named entries are read.
    """

    names = [str(n) for n in parameter_names]
    if not names:
        raise TrainerError("state_dict_l2_divergence needs non-empty parameter_names")
    per_parameter: dict[str, float] = {}
    total_square = 0.0
    max_abs = 0.0
    for name in names:
        if name not in state_a or name not in state_b:
            missing = "state_a" if name not in state_a else "state_b"
            raise TrainerError(f"parameter {name!r} missing from {missing}")
        a = np.asarray(state_a[name], dtype=np.float64)
        b = np.asarray(state_b[name], dtype=np.float64)
        if a.shape != b.shape:
            raise TrainerError(
                f"parameter {name!r} shape mismatch: {a.shape} != {b.shape}"
            )
        delta = a - b
        square = float(np.sum(delta * delta))
        per_parameter[name] = math.sqrt(square)
        total_square += square
        if delta.size:
            max_abs = max(max_abs, float(np.max(np.abs(delta))))
    return {
        "total_l2": math.sqrt(total_square),
        "max_abs_delta": max_abs,
        "compared_parameter_count": len(names),
        "per_parameter_l2": per_parameter,
    }


def ema_export_audit(
    exported_state: Mapping[str, Any],
    raw_state: Mapping[str, Any],
    *,
    parameter_names: Sequence[str],
    ema_decay: float | None,
) -> dict[str, Any]:
    """Assert the exported weights match what ``ema_decay`` promises (T7).

    ``ema_decay`` / ``ema_enabled`` in a checkpoint payload only echo the config:
    they stay true even when the export silently wrote raw weights. The one
    check that cannot be faked is comparing the exported bytes against the live
    (non-EMA) parameters, so this raises both ways:

    * EMA on but the two are identical → ``EMA_EXPORT_IS_RAW``
    * EMA off but the two differ → ``EMA_EXPORT_UNEXPECTED_DIVERGENCE``

    Returns the audit payload to attach to the checkpoint receipt.
    """

    divergence = state_dict_l2_divergence(
        exported_state, raw_state, parameter_names=parameter_names
    )
    total_l2 = float(divergence["total_l2"])
    ema_enabled = ema_decay is not None
    diverged = total_l2 > 0.0
    if ema_enabled and not diverged:
        raise TrainerError(
            "EMA_EXPORT_IS_RAW: ema_decay="
            f"{ema_decay!r} but exported weights equal the live parameters "
            f"over {divergence['compared_parameter_count']} trained tensors"
        )
    if not ema_enabled and diverged:
        raise TrainerError(
            "EMA_EXPORT_UNEXPECTED_DIVERGENCE: ema_decay is None but exported "
            f"weights differ from the live parameters (l2={total_l2:.6g})"
        )
    return {
        "status": "EMA_EXPORT_AUDIT_PASS",
        "ema_decay": ema_decay,
        "ema_enabled": ema_enabled,
        "weights_diverged": diverged,
        "total_l2": total_l2,
        "max_abs_delta": float(divergence["max_abs_delta"]),
        "compared_parameter_count": int(divergence["compared_parameter_count"]),
        "per_parameter_l2": divergence["per_parameter_l2"],
        "audit_reads_exported_file": True,
    }


@contextmanager
def temporary_array_swap(
    live: MutableMapping[str, np.ndarray],
    shadow: Mapping[str, np.ndarray],
) -> Iterator[None]:
    """Temporarily replace live arrays with shadow copies; restore on exit."""

    saved: dict[str, np.ndarray] = {
        k: np.asarray(live[k]).copy() for k in shadow if k in live
    }
    for k, v in shadow.items():
        if k in live:
            live[k] = np.asarray(v, dtype=np.float64).copy()
    try:
        yield
    finally:
        for k, v in saved.items():
            live[k] = v


def load_aimnet2_forces_model(
    *,
    base_weight: Path,
    device: str = "cuda",
    trainable_regexes: Sequence[str] = (r"^outputs\.energy_mlp\.",),
) -> tuple[Any, Any, dict[str, Any], list[str]]:
    """Load official bundle → Forces model with union of regexes trainable.

    Returns ``(model, core, base_bundle, matched_parameter_names)``.
    """

    import torch
    import yaml
    from aimnet.config import build_module
    from aimnet.modules import Forces

    if not base_weight.is_file():
        raise TrainerError(f"missing base weight: {base_weight}")
    digest = _sha256_file(base_weight)
    if digest != OFFICIAL_AIMNET2_WEIGHT_SHA256:
        raise TrainerError(
            f"base weight SHA mismatch: {digest} != {OFFICIAL_AIMNET2_WEIGHT_SHA256}"
        )
    base = torch.load(base_weight, map_location="cpu", weights_only=False)
    if not isinstance(base, dict) or "state_dict" not in base or "model_yaml" not in base:
        raise TrainerError("base weight is not an AIMNet2 export bundle")
    sd = dict(base["state_dict"])
    # Bundle ships srcoulomb; keep as-is (no forced rename)
    cfg = yaml.safe_load(base["model_yaml"])
    core = build_module(copy.deepcopy(cfg))
    if hasattr(core, "outputs") and hasattr(core.outputs, "atomic_shift"):
        core.outputs.atomic_shift.double()
    core.load_state_dict(sd, strict=True)

    all_names = [name for name, _ in core.named_parameters()]
    matched = match_trainable_parameter_names(all_names, trainable_regexes)
    matched_set = set(matched)
    n_train = 0
    for name, param in core.named_parameters():
        enabled = name in matched_set
        param.requires_grad_(enabled)
        n_train += int(enabled)
    if n_train <= 0:
        raise TrainerError(
            "no trainable parameters matched trainable_regexes="
            f"{list(trainable_regexes)!r}"
        )
    model = Forces(core).to(device)
    return model, core, base, matched


def _natom_from_numbers(numbers: Any) -> Any:
    import torch

    natom = (numbers != 0).sum(dim=-1).to(dtype=torch.float64)
    return torch.clamp(natom, min=1.0)


RESUME_SCHEMA: Final = "nhc0801-train-resume-v1"


def capture_rng_state(torch_mod: Any) -> dict[str, Any]:
    """Snapshot python / numpy / torch / torch_cuda RNG for resume payloads."""

    rng_state: dict[str, Any] = {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch_mod.get_rng_state(),
    }
    if torch_mod.cuda.is_available():
        try:
            rng_state["torch_cuda"] = torch_mod.cuda.get_rng_state_all()
        except Exception:  # noqa: BLE001
            rng_state["torch_cuda"] = None
    else:
        rng_state["torch_cuda"] = None
    return rng_state


def restore_rng_state(torch_mod: Any, rng: Mapping[str, Any] | None) -> None:
    """Restore RNG from a resume payload's ``rng_state`` (best-effort on CUDA)."""

    if not rng:
        return
    if rng.get("python") is not None:
        random.setstate(rng["python"])
    if rng.get("numpy") is not None:
        np.random.set_state(rng["numpy"])
    if rng.get("torch") is not None:
        torch_mod.set_rng_state(rng["torch"])
    if rng.get("torch_cuda") is not None and torch_mod.cuda.is_available():
        try:
            torch_mod.cuda.set_rng_state_all(rng["torch_cuda"])
        except Exception:  # noqa: BLE001
            pass


def build_resume_payload(
    *,
    live_state_dict: Mapping[str, Any],
    ema_shadow: Mapping[str, Any],
    optimizer_state_dict: Mapping[str, Any],
    scheduler_state_dict: Mapping[str, Any],
    rng_state: Mapping[str, Any],
    epoch: int,
    train_config_digest: str,
    seed: int,
    run_id: str,
    best_epoch: int | None,
    best_validation_weighted_loss: float | None,
    epochs_since_best: int,
    ema_decay: float | None,
    base_sha256: str = OFFICIAL_AIMNET2_WEIGHT_SHA256,
) -> dict[str, Any]:
    """Pure pack of true-resume state (no I/O). Used by live export + tests.

    Caller must pass **live** (non-EMA) ``live_state_dict`` — never snapshot
    inside an EMA weight-swap context.

    Optimizer / scheduler / tensor maps are **deep-copied** so a later
    ``optimizer.step()`` cannot mutate an already-built payload in place
    (torch ``state_dict()`` returns live tensor refs).
    """

    return {
        "schema": RESUME_SCHEMA,
        "epoch": int(epoch),
        "live_state_dict": copy.deepcopy(dict(live_state_dict)),
        "ema_shadow": copy.deepcopy(dict(ema_shadow)),
        "optimizer_state_dict": copy.deepcopy(optimizer_state_dict),
        "scheduler_state_dict": copy.deepcopy(scheduler_state_dict),
        "rng_state": copy.deepcopy(dict(rng_state)),
        "train_config_digest": train_config_digest,
        "seed": int(seed),
        "run_id": run_id,
        "best_epoch": best_epoch,
        "best_validation_weighted_loss": best_validation_weighted_loss,
        "epochs_since_best": int(epochs_since_best),
        "ema_decay": ema_decay,
        "base_sha256": base_sha256,
    }


def apply_resume_payload(
    payload: Mapping[str, Any],
    *,
    torch_mod: Any,
    core: Any,
    optimizer: Any,
    scheduler: Any,
    ema_shadow_out: MutableMapping[str, Any],
    expected_digest: str,
    ema_decay: float | None,
) -> dict[str, Any]:
    """Restore live params / EMA / optimizer / scheduler / RNG. Fail-closed.

    Mutates ``core``, ``optimizer``, ``scheduler``, and ``ema_shadow_out``.
    Returns bookkeeping for ``run_one_seed`` (``epoch`` = last completed).
    """

    if not isinstance(payload, dict) or payload.get("schema") != RESUME_SCHEMA:
        raise TrainerError(
            f"bad resume schema: expected {RESUME_SCHEMA!r}, "
            f"got {payload.get('schema') if isinstance(payload, dict) else type(payload)}"
        )
    digest = payload.get("train_config_digest")
    if digest != expected_digest:
        raise TrainerError(
            "resume train_config_digest mismatch: "
            f"{digest!r} != {expected_digest!r}"
        )
    live_sd = payload.get("live_state_dict")
    if not isinstance(live_sd, dict):
        raise TrainerError("resume missing live_state_dict")
    core.load_state_dict(live_sd, strict=True)

    ema_shadow_out.clear()
    ema_in = payload.get("ema_shadow")
    if ema_decay is not None:
        if not isinstance(ema_in, dict):
            raise TrainerError("resume missing ema_shadow")
        for name, param in core.named_parameters():
            if not param.requires_grad:
                continue
            if name not in ema_in:
                raise TrainerError(f"resume missing ema_shadow for {name!r}")
            t = ema_in[name]
            if not isinstance(t, torch_mod.Tensor):
                t = torch_mod.as_tensor(t)
            ema_shadow_out[name] = t.detach().to(device=param.device).clone()

    opt_sd = payload.get("optimizer_state_dict")
    if not isinstance(opt_sd, dict):
        raise TrainerError("resume missing optimizer_state_dict")
    optimizer.load_state_dict(opt_sd)

    sch_sd = payload.get("scheduler_state_dict")
    if not isinstance(sch_sd, dict):
        raise TrainerError("resume missing scheduler_state_dict")
    scheduler.load_state_dict(sch_sd)

    restore_rng_state(torch_mod, payload.get("rng_state"))

    epoch = int(payload["epoch"])
    return {
        "epoch": epoch,
        "best_epoch": payload.get("best_epoch"),
        "best_validation_weighted_loss": payload.get("best_validation_weighted_loss"),
        "epochs_since_best": int(payload.get("epochs_since_best", 0)),
    }


class LiveAimnet2TrainBackend:
    """One-seed live backend: train_epoch / evaluate with sample_weight.

    Scheduler is **not** stepped inside evaluate (B2); callers must call
    ``step_scheduler(val_loss)`` once per epoch (wired by multi_seed_trainer / M8).
    """

    def __init__(
        self,
        *,
        dataset_root: Path,
        base_weight: Path,
        config: TrainingConfig,
        seed: int,
        device: str = "cuda",
        batches_per_epoch: int = -1,
    ) -> None:
        import torch
        from aimnet.data import SizeGroupedDataset, SizeGroupedSampler
        from aimnet.train.utils import prepare_batch

        self.torch = torch
        self.prepare_batch = prepare_batch
        self.config = config
        self.seed = seed
        self.device = device
        self.dataset_root = dataset_root
        self.ema_decay: float | None = config.ema_decay

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        # B3: pass the full regex tuple (not [0]) so _mlp_shift actually unfreezes E0
        (
            self.model,
            self.core,
            self.base_bundle,
            self.trainable_parameter_names,
        ) = load_aimnet2_forces_model(
            base_weight=base_weight,
            device=device,
            trainable_regexes=config.trainable_parameter_regex,
        )
        self.train_config_digest = train_config_digest(config)

        keys = ["coord", "numbers", "charge", "energy", "forces", SAMPLE_WEIGHT_KEY]
        self.train_ds = SizeGroupedDataset(str(dataset_root / "train"), keys=keys)
        self.val_ds = SizeGroupedDataset(str(dataset_root / "validation"), keys=keys)
        if len(self.train_ds) <= 0 or len(self.val_ds) <= 0:
            raise TrainerError("empty train or validation SizeGroupedDataset")
        self.train_frame_count = len(self.train_ds)
        self.val_frame_count = len(self.val_ds)

        train_sampler = SizeGroupedSampler(
            self.train_ds,
            batch_size=config.batch_size,
            batch_mode=config.batch_mode,
            shuffle=True,
            batches_per_epoch=batches_per_epoch,
            seed=seed,
        )
        val_sampler = SizeGroupedSampler(
            self.val_ds,
            batch_size=config.batch_size,
            batch_mode=config.batch_mode,
            shuffle=False,
            batches_per_epoch=-1,
            seed=seed,
        )
        x_keys = ["coord", "numbers", "charge"]
        y_keys = ["energy", "forces", SAMPLE_WEIGHT_KEY]
        self.train_loader = self.train_ds.get_loader(
            train_sampler, x_keys, y_keys, num_workers=0, pin_memory=True
        )
        self.val_loader = self.val_ds.get_loader(
            val_sampler, x_keys, y_keys, num_workers=0, pin_memory=True
        )
        params = [p for p in self.model.parameters() if p.requires_grad]
        self.optimizer = torch.optim.RAdam(
            params, lr=config.learning_rate, weight_decay=config.weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            factor=config.scheduler_factor,
            patience=config.scheduler_patience_epochs,
        )

        # EMA shadow over requires_grad core parameters (H4 / T7)
        self._ema_shadow: dict[str, Any] = {}
        if self.ema_decay is not None:
            for name, param in self.core.named_parameters():
                if param.requires_grad:
                    self._ema_shadow[name] = param.detach().clone()

    def _update_ema(self) -> None:
        """After optimizer.step: shadow = d * shadow + (1-d) * param."""

        decay = self.ema_decay
        if decay is None or not self._ema_shadow:
            return
        d = float(decay)
        alpha = 1.0 - d
        for name, param in self.core.named_parameters():
            shadow = self._ema_shadow.get(name)
            if shadow is None:
                continue
            shadow.mul_(d).add_(param.data, alpha=alpha)

    @contextmanager
    def _use_ema_weights(self) -> Iterator[None]:
        """Temporarily swap EMA weights into core; restore on exit.

        No-op when ``ema_decay is None`` or no shadow entries exist.
        """

        if self.ema_decay is None or not self._ema_shadow:
            yield
            return
        saved: dict[str, Any] = {}
        for name, param in self.core.named_parameters():
            shadow = self._ema_shadow.get(name)
            if shadow is None:
                continue
            saved[name] = param.data.detach().clone()
            param.data.copy_(shadow)
        try:
            yield
        finally:
            for name, param in self.core.named_parameters():
                if name in saved:
                    param.data.copy_(saved[name])

    def step_scheduler(self, val_loss: float) -> None:
        """Advance ReduceLROnPlateau from caller (once per epoch; B2 / M8)."""

        self.scheduler.step(float(val_loss))

    def train_epoch(
        self,
        batches: Any = None,
        *,
        split_frame_count: int,
        energy_weight: float,
        forces_weight: float,
        seed: int,
        epoch: int,
    ) -> dict[str, Any]:
        del batches, seed  # use internal loader
        self.model.train()
        losses: list[float] = []
        wsum = 0.0
        n_batches = 0
        for x, y in self.train_loader:
            x = self.prepare_batch(x, device=self.device, non_blocking=True)
            y = self.prepare_batch(y, device=self.device, non_blocking=True)
            if SAMPLE_WEIGHT_KEY not in y:
                raise TrainerError("sample_weight missing from batch")
            self.optimizer.zero_grad(set_to_none=True)
            pred = self.model(x)
            pred = {**pred, "_natom": _natom_from_numbers(x["numbers"])}
            terms = weighted_batch_terms(pred, y)
            scaled = scaled_training_loss(
                terms,
                split_frame_count=split_frame_count or self.train_frame_count,
                energy_weight=energy_weight,
                forces_weight=forces_weight,
            )
            loss = scaled["loss"]
            if not math.isfinite(float(loss.detach().cpu())):
                raise TrainerError("non-finite training loss")
            loss.backward()
            self.torch.nn.utils.clip_grad_value_(
                self.model.parameters(), self.config.gradient_clip_value
            )
            self.optimizer.step()
            self._update_ema()
            losses.append(float(loss.detach().cpu()))
            wsum += float(terms["sample_weight_sum"].detach().cpu())
            n_batches += 1
        if n_batches <= 0:
            raise TrainerError("empty training epoch")
        return {
            "train_weighted_loss": float(sum(losses) / len(losses)),
            "batch_count": n_batches,
            "sample_weight_sum": wsum,
            "backward_called": True,
            "optimizer_step_called": True,
            "live_parameters_updated": True,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "ema_enabled": self.ema_decay is not None,
        }

    def evaluate(
        self,
        batches: Any = None,
        *,
        energy_weight: float,
        forces_weight: float,
        energy_bias: float = 0.0,
    ) -> dict[str, Any]:
        del batches, energy_bias
        self.model.eval()
        acc = WeightedEvaluationAccumulator()
        # Use EMA weights for validation when enabled (FT tutorial practice).
        # Scheduler is intentionally NOT stepped here (B2) — call step_scheduler.
        with self._use_ema_weights(), self.torch.no_grad():
            for x, y in self.val_loader:
                x = self.prepare_batch(x, device=self.device, non_blocking=True)
                y = self.prepare_batch(y, device=self.device, non_blocking=True)
                pred = self.model(x)
                pred = {**pred, "_natom": _natom_from_numbers(x["numbers"])}
                terms = weighted_batch_terms(pred, y)
                acc.update(
                    energy_numerator=float(terms["energy_numerator"].detach().cpu()),
                    forces_numerator=float(terms["forces_numerator"].detach().cpu()),
                    sample_weight_sum=float(terms["sample_weight_sum"].detach().cpu()),
                    batch_size=int(terms["batch_size"]),
                )
        out = acc.finalize(energy_weight=energy_weight, forces_weight=forces_weight)
        val_loss = float(out["weighted_loss"])
        return {
            "validation_weighted_loss": val_loss,
            "weighted_energy_mse": float(out["weighted_energy_mse"]),
            "weighted_forces_mse": float(out["weighted_forces_mse"]),
            "sample_weight_sum": float(out["sample_weight_sum"]),
            "sample_count": int(out["sample_count"]),
            "checkpoint_selection_permitted": False,
            "backward_called": False,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
            "ema_enabled": self.ema_decay is not None,
        }

    def _weight_kind(self) -> str:
        """What the exported ``.pt`` actually holds: "ema" or "raw"."""

        return "ema" if self.ema_decay is not None else "raw"

    def _core_state_snapshot(self) -> dict[str, Any]:
        """Detached host copy of ``core.state_dict()``.

        ``.clone()`` is load-bearing: ``Tensor.cpu()`` returns *self* when the
        tensor is already on CPU, so without it the snapshot would alias the
        live parameters and ``_use_ema_weights``' in-place restore would
        overwrite the EMA values before ``torch.save`` runs (silently exporting
        raw weights under EMA metadata). Costs one host copy on CUDA.
        """

        return {k: v.detach().cpu().clone() for k, v in self.core.state_dict().items()}

    def _checkpoint_payload(
        self, state: Mapping[str, Any], *, weight_kind: str
    ) -> dict[str, Any]:
        """Export bundle for ``state``; ``weight_kind`` is "ema" or "raw" (T7/T8)."""

        payload: dict[str, Any] = {
            "format_version": self.base_bundle.get("format_version"),
            "model_yaml": self.base_bundle.get("model_yaml"),
            "state_dict": dict(state),
            "seed": self.seed,
            "base_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
            "nhc0801_live_finetune": True,
            "train_config_digest": self.train_config_digest,
            "run_id": self.config.run_id,
            "trainable_parameter_regex": list(self.config.trainable_parameter_regex),
            "trainable_parameter_names": list(self.trainable_parameter_names),
            "ema_decay": self.ema_decay,
            "weight_kind": weight_kind,
            "energy_weight": float(self.config.energy_weight),
            "forces_weight": float(self.config.forces_weight),
            "batch_size": int(self.config.batch_size),
            "epochs": int(self.config.epochs),
        }
        # preserve immutable runtime metadata fields when present (do not alter cutoff)
        for key in (
            "cutoff",
            "needs_coulomb",
            "needs_dispersion",
            "coulomb_mode",
            "coulomb_sr_rc",
            "coulomb_sr_envelope",
            "d3_params",
            "has_embedded_lr",
            "implemented_species",
        ):
            if key in self.base_bundle:
                payload[key] = self.base_bundle[key]
        return payload

    def export_checkpoint(self, path: Path) -> dict[str, Any]:
        """Write state_dict + frozen metadata (EMA weights when enabled)."""

        path.parent.mkdir(parents=True, exist_ok=True)
        with self._use_ema_weights():
            state = self._core_state_snapshot()
        self.torch.save(self._checkpoint_payload(state, weight_kind=self._weight_kind()), path)
        raw = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "live_weights_written": True,
            "train_config_digest": self.train_config_digest,
            "trainable_parameter_count": len(self.trainable_parameter_names),
            "ema_enabled": self.ema_decay is not None,
            "weight_kind": self._weight_kind(),
        }

    def export_raw_audit_sibling(self, ema_path: Path) -> dict[str, Any]:
        """Write ``epoch_NNNN.raw.pt`` and audit it against the exported ``.pt``.

        Call right after :meth:`export_checkpoint`. The audit **re-reads the
        exported file** rather than the in-memory snapshot — that is the only
        way to prove the bytes on disk carry EMA weights, which no amount of
        ``ema_decay`` metadata can establish. Fails closed via
        :func:`ema_export_audit`.
        """

        ema_path = Path(ema_path)
        if not ema_path.is_file():
            raise TrainerError(f"missing exported checkpoint to audit: {ema_path}")
        on_disk = self.torch.load(ema_path, map_location="cpu", weights_only=False)
        if not isinstance(on_disk, dict) or "state_dict" not in on_disk:
            raise TrainerError(f"exported checkpoint is not a bundle: {ema_path}")
        # Live (non-EMA) parameters: snapshot outside any _use_ema_weights window.
        raw_state = self._core_state_snapshot()
        audit = ema_export_audit(
            on_disk["state_dict"],
            raw_state,
            parameter_names=self.trainable_parameter_names,
            ema_decay=self.ema_decay,
        )
        raw_path = ema_path.with_name(f"{ema_path.stem}.raw{ema_path.suffix}")
        self.torch.save(self._checkpoint_payload(raw_state, weight_kind="raw"), raw_path)
        audit["exported_weight_path"] = str(ema_path)
        audit["raw_weight_path"] = str(raw_path)
        audit["raw_weight_sha256"] = hashlib.sha256(raw_path.read_bytes()).hexdigest()
        audit["exported_weight_kind"] = on_disk.get("weight_kind")
        return audit

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
        """Write ``epoch_NNNN.resume.pt`` — full train state for true resume.

        Does **not** change ``epoch_NNNN.pt`` semantics (EMA weights for
        pre-screen / sci-val). Resume payload holds live (non-EMA) params,
        EMA shadows, optimizer, scheduler, and RNG states.

        Live state is snapshotted **outside** ``_use_ema_weights()`` (do not
        move this call into an EMA swap window).
        """

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # LOAD-BEARING: outside _use_ema_weights — live params, not EMA.
        live_state = self._core_state_snapshot()
        digest = (
            train_config_digest
            if train_config_digest is not None
            else self.train_config_digest
        )
        payload = build_resume_payload(
            live_state_dict=live_state,
            ema_shadow={
                name: t.detach().cpu().clone() for name, t in self._ema_shadow.items()
            },
            optimizer_state_dict=self.optimizer.state_dict(),
            scheduler_state_dict=self.scheduler.state_dict(),
            rng_state=capture_rng_state(self.torch),
            epoch=int(epoch),
            train_config_digest=digest,
            seed=int(self.seed),
            run_id=self.config.run_id,
            best_epoch=best_epoch,
            best_validation_weighted_loss=best_validation_weighted_loss,
            epochs_since_best=int(epochs_since_best),
            ema_decay=self.ema_decay,
        )
        self.torch.save(payload, path)
        raw = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "live_resume_written": True,
            "epoch": int(epoch),
            "train_config_digest": digest,
        }

    def load_resume_checkpoint(self, path: Path) -> dict[str, Any]:
        """Restore live params / EMA / optimizer / scheduler / RNG from resume file.

        Fail-closed if ``train_config_digest`` mismatches this backend's recipe.
        Returns bookkeeping dict consumed by ``run_one_seed`` (epoch = last done).
        """

        path = Path(path)
        if not path.is_file():
            raise TrainerError(f"missing resume checkpoint: {path}")
        payload = self.torch.load(path, map_location="cpu", weights_only=False)
        return apply_resume_payload(
            payload,
            torch_mod=self.torch,
            core=self.core,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            ema_shadow_out=self._ema_shadow,
            expected_digest=self.train_config_digest,
            ema_decay=self.ema_decay,
        )
