"""Read-only GPU inventory for dispatch (no chemistry, no kills).

Shared by teacher autofill, e0 Val 4-GPU fan-out, and future dispatchers.
Policy (AGENTS / COMPUTE_DISPATCH):
  - Prefer GPUs with **no** compute apps (truly free).
  - Never place on GPUs running **VASP** (other users).
  - Prefer lowest used memory among remaining cards.
  - One process pins one physical GPU via CUDA_VISIBLE_DEVICES.
"""

from __future__ import annotations

import subprocess
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any


class GpuInventoryError(RuntimeError):
    """nvidia-smi / inventory failed closed."""


@dataclass(frozen=True, slots=True)
class GpuSlot:
    index: int
    used_mib: int
    util_percent: int | None
    has_vasp: bool
    process_count: int
    process_names: tuple[str, ...]


def _nvidia_smi_csv(query: str, *, gpu_index: int | None = None) -> str:
    cmd = ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"]
    if gpu_index is not None:
        cmd[1:1] = ["-i", str(int(gpu_index))]
    try:
        return subprocess.check_output(cmd, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError) as exc:
        raise GpuInventoryError(f"nvidia-smi failed: {exc}") from exc


def _compute_apps(gpu_index: int) -> list[tuple[str, int]]:
    """Return [(process_name, used_mib), ...] for one GPU."""
    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "-i",
                str(int(gpu_index)),
                "--query-compute-apps=process_name,used_memory",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            timeout=30,
        )
    except (OSError, subprocess.SubprocessError):
        return []
    apps: list[tuple[str, int]] = []
    for line in out.splitlines():
        line = line.strip()
        if not line or line.lower() == "n/a":
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        name = parts[0]
        try:
            mem = int(float(parts[1]))
        except ValueError:
            mem = 0
        apps.append((name, mem))
    return apps


def inventory_gpus(max_gpu: int = 8) -> list[GpuSlot]:
    """Snapshot all GPUs 0..max_gpu-1."""
    raw = _nvidia_smi_csv("index,memory.used,utilization.gpu")
    slots: list[GpuSlot] = []
    for line in raw.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        try:
            idx = int(parts[0])
            used = int(float(parts[1]))
        except ValueError:
            continue
        util: int | None
        try:
            util = int(float(parts[2])) if len(parts) > 2 else None
        except ValueError:
            util = None
        if idx >= max_gpu:
            continue
        apps = _compute_apps(idx)
        names = tuple(a[0] for a in apps)
        has_vasp = any("vasp" in n.lower() for n in names)
        slots.append(
            GpuSlot(
                index=idx,
                used_mib=used,
                util_percent=util,
                has_vasp=has_vasp,
                process_count=len(apps),
                process_names=names,
            )
        )
    return sorted(slots, key=lambda s: s.index)


def pick_gpus(
    n: int,
    *,
    max_gpu: int = 8,
    exclude: Sequence[int] | None = None,
    allow_shared: bool = True,
    require_free: bool = False,
) -> list[int]:
    """Pick up to *n* GPUs for new NHC jobs.

    Order of preference:
      1. No VASP.
      2. If require_free: process_count == 0 only.
      3. Else if not allow_shared: process_count == 0 preferred, then lowest used_mib.
      4. Sort by (process_count, used_mib, index).

    Raises GpuInventoryError if fewer than *n* eligible GPUs.
    """
    if n <= 0:
        return []
    ban = {int(x) for x in (exclude or ())}
    slots = inventory_gpus(max_gpu=max_gpu)
    eligible = [s for s in slots if s.index not in ban and not s.has_vasp]
    if require_free:
        eligible = [s for s in eligible if s.process_count == 0]
    elif not allow_shared:
        free_only = [s for s in eligible if s.process_count == 0]
        if len(free_only) >= n:
            eligible = free_only
    eligible.sort(key=lambda s: (s.process_count, s.used_mib, s.index))
    picked = [s.index for s in eligible[:n]]
    if len(picked) < n:
        raise GpuInventoryError(
            f"need {n} eligible GPUs, only {len(picked)} available "
            f"(no-VASP, exclude={sorted(ban)}, require_free={require_free}); "
            f"inventory={[ (s.index, s.used_mib, s.process_count, s.has_vasp) for s in slots ]}"
        )
    return picked


def inventory_as_dict(max_gpu: int = 8) -> dict[str, Any]:
    slots = inventory_gpus(max_gpu=max_gpu)
    return {
        "schema": "nhc0801-gpu-inventory-v1",
        "gpus": [
            {
                "index": s.index,
                "used_mib": s.used_mib,
                "util_percent": s.util_percent,
                "has_vasp": s.has_vasp,
                "process_count": s.process_count,
                "process_names": list(s.process_names),
            }
            for s in slots
        ],
    }
