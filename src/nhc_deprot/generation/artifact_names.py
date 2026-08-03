"""Canonical experimental artifact names under runs/<generation>/.

Policy (AGENTS.md Experimental data naming):
  - Product dirs: teacher_gpu_g00N/, epoch0_val_batches/g00N/
  - Log basenames: prefer group-scoped stable names; accept legacy *02c* as read aliases
  - Engineering module names (gpu_autofill) may differ from user-facing g00N labels

Writers of *new* logs should use the CANONICAL names. Readers must use
``resolve_existing`` so old jobs keep working until they exit.
"""

from __future__ import annotations

from pathlib import Path
from typing import Final, Iterable, Sequence

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
