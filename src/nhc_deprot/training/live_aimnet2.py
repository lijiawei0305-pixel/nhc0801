"""Live AIMNet2 fine-tune backend (GPU). Requires mlff env + aimnet + torch.

Not imported by default dry-run paths. Resource control is the caller's job
(CUDA_VISIBLE_DEVICES, claim, single-process).
"""

from __future__ import annotations

import copy
import hashlib
import io
import math
import random
import re
from pathlib import Path
from typing import Any

import numpy as np

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


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_aimnet2_forces_model(
    *,
    base_weight: Path,
    device: str = "cuda",
    trainable_regex: str = r"^outputs\.energy_mlp\.",
) -> tuple[Any, Any, dict[str, Any]]:
    """Load official bundle → Forces model with only energy_mlp trainable."""

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
    pattern = re.compile(trainable_regex)
    n_train = 0
    for name, param in core.named_parameters():
        enabled = bool(pattern.search(name))
        param.requires_grad_(enabled)
        n_train += int(enabled)
    if n_train <= 0:
        raise TrainerError("no trainable parameters matched energy_mlp regex")
    model = Forces(core).to(device)
    return model, core, base


def _natom_from_numbers(numbers: Any) -> Any:
    import torch

    natom = (numbers != 0).sum(dim=-1).to(dtype=torch.float64)
    return torch.clamp(natom, min=1.0)


class LiveAimnet2TrainBackend:
    """One-seed live backend: train_epoch / evaluate with sample_weight."""

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

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False

        self.model, self.core, self.base_bundle = load_aimnet2_forces_model(
            base_weight=base_weight,
            device=device,
            trainable_regex=config.trainable_parameter_regex[0],
        )
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
        with self.torch.no_grad():
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
        # scheduler step from outside using val loss
        val_loss = float(out["weighted_loss"])
        self.scheduler.step(val_loss)
        return {
            "validation_weighted_loss": val_loss,
            "weighted_energy_mse": float(out["weighted_energy_mse"]),
            "weighted_forces_mse": float(out["weighted_forces_mse"]),
            "sample_weight_sum": float(out["sample_weight_sum"]),
            "sample_count": int(out["sample_count"]),
            "checkpoint_selection_permitted": False,
            "backward_called": False,
            "learning_rate": float(self.optimizer.param_groups[0]["lr"]),
        }

    def export_checkpoint(self, path: Path) -> dict[str, Any]:
        """Write state_dict + frozen metadata (not full export bundle yet)."""

        path.parent.mkdir(parents=True, exist_ok=True)
        state = {k: v.detach().cpu() for k, v in self.core.state_dict().items()}
        payload = {
            "format_version": self.base_bundle.get("format_version"),
            "model_yaml": self.base_bundle.get("model_yaml"),
            "state_dict": state,
            "seed": self.seed,
            "base_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
            "nhc0801_live_finetune": True,
        }
        # preserve immutable runtime metadata fields when present
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
        self.torch.save(payload, path)
        raw = path.read_bytes()
        return {
            "path": str(path),
            "bytes": len(raw),
            "sha256": hashlib.sha256(raw).hexdigest(),
            "live_weights_written": True,
        }
