"""Short released model versions (v0.1, v0.2, …) under runs/<gen>/models/.

Training dumps live in train_g00N/seed_*/epoch_NNNN.pt.
After scientific selection, promote one checkpoint to a short version tag:

    models/v0.1/model.pt
    models/v0.1/info.json

Do not invent long English weight basenames for releases.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any

from nhc_deprot.data.io_util import sha256_file, write_json
from nhc_deprot.generation import artifact_names as anames
from nhc_deprot.generation.layout import GenerationError, GenerationLayout, normalize_model_version

MODEL_INFO_SCHEMA = "nhc0801-model-version-info-v1"


class ModelVersionError(RuntimeError):
    """Model version register/read failed closed."""


def register_model_version(
    *,
    layout: GenerationLayout,
    version: str,
    source_weight: Path,
    train_batch_id: str | None = None,
    seed: int | None = None,
    epoch: int | None = None,
    notes: list[str] | None = None,
    overwrite: bool = False,
    copy_weights: bool = True,
) -> dict[str, Any]:
    """Promote a training checkpoint to models/vX.Y/model.pt + info.json.

    ``version`` may be ``0.1`` or ``v0.1``. Weight product name is always ``model.pt``.
    """
    ver = normalize_model_version(version)
    src = Path(source_weight)
    if not src.is_file():
        raise ModelVersionError(f"source weight missing: {src}")

    out_dir = layout.model_version_dir(ver)
    weight_path = layout.model_weight_path(ver)
    info_path = layout.model_info_path(ver)

    if weight_path.exists() and not overwrite:
        raise ModelVersionError(
            f"version {ver} already has {anames.MODEL_WEIGHT_BASENAME}; "
            "pass overwrite=True to replace"
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    if copy_weights:
        shutil.copy2(src, weight_path)
    elif not weight_path.exists():
        raise ModelVersionError("copy_weights=False but model.pt missing")

    info: dict[str, Any] = {
        "schema": MODEL_INFO_SCHEMA,
        "version": ver,
        "human_name": ver,  # say "v0.1" — not a long English product string
        "weight_basename": anames.MODEL_WEIGHT_BASENAME,
        "weight_path": str(weight_path),
        "weight_sha256": sha256_file(weight_path),
        "source_weight": str(src.resolve()),
        "train_batch_id": train_batch_id,
        "seed": seed,
        "epoch": epoch,
        "generation_id": layout.generation_id,
        "notes": list(notes or []),
    }
    write_json(info_path, info, overwrite=True)
    return info


def list_model_versions(layout: GenerationLayout) -> list[str]:
    """Return sorted version tags that have model.pt present."""
    root = layout.models_dir
    if not root.is_dir():
        return []
    found: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        try:
            ver = normalize_model_version(p.name)
        except GenerationError:
            continue
        if (p / anames.MODEL_WEIGHT_BASENAME).is_file():
            found.append(ver)
    return found
