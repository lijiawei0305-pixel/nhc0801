"""Canonical experimental artifact names under runs/<generation>/.

Policy (AGENTS.md Experimental data naming):
  - Product dirs: teacher_gpu_g00N/, epoch0_val_batches/g00N/, train_g00N/, models/v0.1/
  - Released weights: models/vX.Y/model.pt (short version tag; no long English stems)
  - Log basenames: prefer group-scoped stable names; accept legacy *02c* as read aliases
  - Engineering module names (gpu_autofill) may differ from user-facing g00N labels

Writers of *new* logs should use the CANONICAL names. Readers must use
``resolve_existing`` so old jobs keep working until they exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, Sequence

# --- Fine-tune / AIMNet2 train (group-scoped, flat like teacher_gpu_g00N) ---
# Human: "g001 的训练" → train_g001/
TRAIN_DIR_PREFIX: Final = "train_"  # train_g001, train_g002, …
TRAIN_INFO_JSON: Final = "train_info.json"  # 这次训了谁、用什么设定
TRAIN_RESULT_JSON: Final = "train_result.json"  # 多种子训练总结果
TRAIN_SEED_RESULT_JSON: Final = "seed_result.json"  # 某一个随机种子的结果
# obsolete / legacy (read only)
TRAIN_BATCHES_DIR_LEGACY: Final = "train_batches"
TRAIN_CAMPAIGN_RECEIPT_LEGACY: Final = "campaign_receipt.json"
TRAIN_SEED_RECEIPT_LEGACY: Final = "seed_receipt.json"
TRAIN_DIR_LEGACY: Final = "train"  # pilot flat train/; do not write new runs here


def train_product_dirname(batch_id: str) -> str:
    """Directory name for one group fine-tune, e.g. train_g001."""
    bid = str(batch_id).strip()
    return f"train_{bid}"


def train_log_basename(batch_id: str) -> str:
    """Stdout log for one train group, e.g. train_g001.out."""
    return f"{train_product_dirname(batch_id)}.out"


def train_seed_dirname(seed: int) -> str:
    """Random-seed folder, e.g. seed_20260730 (number is RNG seed, not a calendar date)."""
    return f"seed_{int(seed)}"


def train_checkpoint_stem(epoch: int) -> str:
    """Shared stem for weights + meta: epoch_NNNN (training round index)."""
    return f"epoch_{int(epoch):04d}"


def train_checkpoint_weight_name(epoch: int) -> str:
    return f"{train_checkpoint_stem(epoch)}.pt"


def train_checkpoint_meta_name(epoch: int) -> str:
    return f"{train_checkpoint_stem(epoch)}.meta.json"


# --- Released AIMNet2 versions (selected after train + scientific selection) ---
# Human: "v0.1" / "v0.2"  →  models/v0.1/model.pt
MODELS_DIR: Final = "models"
MODEL_WEIGHT_BASENAME: Final = "model.pt"  # always this name inside a version folder
MODEL_INFO_BASENAME: Final = "info.json"


def model_version_dirname(version: str) -> str:
    """Folder name under models/, e.g. v0.1 (caller should normalize first)."""
    s = str(version).strip()
    if not s.startswith("v"):
        s = f"v{s}"
    return s.lower()


# --- Teacher wave (g001 pilot CPU wave historically trial 02c) ---
TEACHER_WAVE_LOG: Final = "teacher_wave_g001.out"
TEACHER_WAVE_LOG_LEGACY: Final = (
    "teacher_wave_02c.out",
)
TEACHER_CAMPAIGN_JSON: Final = "teacher_campaign_live_g001.json"
TEACHER_CAMPAIGN_JSON_LEGACY: Final = (
    "teacher_campaign_live_02c.json",
    "teacher_campaign_receipt.json",
)

# --- Epoch-0 live (g001) ---
EPOCH0_LIVE_LOG: Final = "live_epoch0_g001.out"
EPOCH0_LIVE_LOG_LEGACY: Final = (
    "live_epoch0_02c.out",
    "live_epoch0.out",
    "start_epoch0_now_nohup.out",
    "g001_epoch0_rerun.out",
)
EPOCH0_START_MARKER: Final = "START_EPOCH0_G001"
EPOCH0_START_MARKER_LEGACY: Final = (
    "START_EPOCH0_02c",
    "START_EPOCH0",
)

# --- GPU teacher queue (engineering state; products are teacher_gpu_g00N/) ---
GPU_TEACHER_STATE_DIR: Final = "gpu_teacher_queue"
GPU_TEACHER_STATE_DIR_LEGACY: Final = "gpu_autofill"  # migrate → gpu_teacher_queue
GPU_TEACHER_LOG_TAG: Final = "gpu-teacher"  # stdout tag for humans

# Banned product names for new fine-tune writes (read of legacy is OK)
TRAIN_BANNED_WRITE_NAMES: Final = (
    "train",  # bare generation-level train/ as new campaign root
    "train_batches",  # obsolete nested form
    "finetune",
    "ft",
    "ckpts",
    "checkpoints",
    "best.pt",
    "latest.pt",
)

# Released model basenames must stay short — ban long English stems as the product name
MODEL_BANNED_BASENAMES: Final = (
    "best.pt",
    "latest.pt",
    "final.pt",
    "selected.pt",
    "finetuned.pt",
    "aimnet2_finetuned.pt",
)


def resolve_existing(logs_dir: Path, canonical: str, legacy: Sequence[str] = ()) -> Path | None:
    """Return first existing path among canonical then legacy; else None."""
    candidates = (canonical, *legacy)
    for name in candidates:
        p = logs_dir / name
        if p.is_file():
            return p
    return None


def resolve_for_write(logs_dir: Path, canonical: str) -> Path:
    """Path to open for append/write (always canonical name)."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    return logs_dir / canonical


def all_log_candidates(canonical: str, legacy: Iterable[str] = ()) -> tuple[str, ...]:
    return (canonical, *tuple(legacy))
