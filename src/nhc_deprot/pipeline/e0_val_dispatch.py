"""Dispatch Val-only Epoch-0: 2 roots × 2 endpoints → 4 GPUs.

Hard rule (AGENTS): never pin all four endpoints of a Val batch to one GPU.
Uses :mod:`nhc_deprot.resources.gpu_inventory` (no-VASP, free-first).
Launches ``python -m nhc_deprot.pipeline.e0_val_only --endpoint …`` per shard.
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.resources.gpu_inventory import (
    GpuInventoryError,
    inventory_as_dict,
    pick_gpus,
)

ENDPOINTS: Final = ("cation", "neutral")


class E0ValDispatchError(RuntimeError):
    """Val e0 multi-GPU dispatch failed closed."""


@dataclass(frozen=True, slots=True)
class EndpointJob:
    """One (root, cation|neutral) job on one GPU."""

    root_id: str
    endpoint: str
    gpu_index: int
    log_path: Path

    @property
    def key(self) -> str:
        return f"{self.root_id}:{self.endpoint}"


# Deprecated alias — do not use in new user-facing text.
EndpointShard = EndpointJob


def _utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def default_val_roots_for_batch(batch_id: str) -> tuple[str, ...]:
    """g001 pilot Val2; other batches must pass roots explicitly for now."""
    bid = str(batch_id).strip()
    if bid in {"g001", "g001_pilot", "pilot", "nhc0801-g001"}:
        return tuple(VALIDATION_ROOTS)
    raise E0ValDispatchError(
        f"no built-in Val roots for batch_id={batch_id!r}; pass --val-roots"
    )


def plan_val_endpoint_jobs(
    val_roots: Sequence[str],
    *,
    gpu_ids: Sequence[int],
    log_dir: Path,
    batch_id: str,
) -> list[EndpointJob]:
    """Map N Val roots × 2 endpoints onto GPU ids (cycle if fewer GPUs).

    Legacy pilot used N=2 with exactly 4 GPUs (one endpoint per card). Larger
    Val sets (TVT resplit) still produce 2N jobs; GPUs are assigned round-robin
    over ``gpu_ids``.
    """
    roots = [r.strip() for r in val_roots if r and r.strip()]
    if len(roots) < 1:
        raise E0ValDispatchError(
            f"Val batch e0 needs >= 1 root, got {len(roots)}: {roots}"
        )
    train = frozenset(TRAIN_ROOTS)
    bad = [r for r in roots if r in train]
    if bad:
        raise E0ValDispatchError(f"REFUSED train roots in Val e0 dispatch: {bad}")
    gpus = [int(g) for g in gpu_ids]
    if not gpus:
        raise E0ValDispatchError(f"need >= 1 GPU id, got {list(gpu_ids)}")

    log_dir.mkdir(parents=True, exist_ok=True)
    jobs: list[EndpointJob] = []
    k = 0
    for root_id in roots:
        for ep in ENDPOINTS:
            gpu = gpus[k % len(gpus)]
            tag = f"e0_{batch_id}_{root_id[:8]}_{ep}_gpu{gpu}"
            jobs.append(
                EndpointJob(
                    root_id=root_id,
                    endpoint=ep,
                    gpu_index=gpu,
                    log_path=log_dir / f"{tag}.out",
                )
            )
            k += 1
    return jobs


# Deprecated alias
plan_val_endpoint_shards = plan_val_endpoint_jobs


def launch_val_e0_4gpu(
    *,
    nhc0801_root: Path,
    generation_id: str,
    batch_id: str,
    val_roots: Sequence[str] | None = None,
    parent_backend: str = "gpu",
    parent_max_steps: int = 250,
    max_gpu: int = 8,
    exclude_gpus: Sequence[int] | None = None,
    gpu_ids: Sequence[int] | None = None,
    require_free: bool = False,
    allow_shared: bool = True,
    allow_vasp_share: bool = False,
    dry_run: bool = False,
    python_exe: str | None = None,
) -> dict[str, Any]:
    """Spawn Val e0 endpoint jobs across GPUs (cation/neutral 分开算).

    One process per (root, endpoint). GPUs are assigned round-robin over
    ``gpu_ids`` (or auto-picked). Prefer ``len(gpu_ids) == n_endpoints`` so
    each endpoint gets its own card (e.g. 4 roots × 2 endpoints → 8 GPUs).

    If *gpu_ids* is provided (length >= 1), use it as-is. Otherwise pick
    ``min(n_endpoints, max_gpu)`` cards via :func:`pick_gpus`.

    ``allow_vasp_share``: only when the machine is fully VASP-occupied; co-locate
    without killing VASP (see ``gpu_inventory.pick_gpus``).
    """
    roots = (
        list(val_roots)
        if val_roots is not None
        else list(default_val_roots_for_batch(batch_id))
    )
    n_endpoints = len(roots) * len(ENDPOINTS)
    if n_endpoints < 1:
        raise E0ValDispatchError("no endpoints to launch (empty val_roots)")
    inv = inventory_as_dict(max_gpu=max_gpu)
    if gpu_ids is not None:
        gpus = [int(x) for x in gpu_ids]
        if len(gpus) < 1:
            raise E0ValDispatchError(f"gpu_ids must be non-empty, got {gpus}")
    else:
        n_pick = min(int(n_endpoints), int(max_gpu))
        try:
            gpus = pick_gpus(
                n_pick,
                max_gpu=max_gpu,
                exclude=exclude_gpus,
                allow_shared=allow_shared,
                require_free=require_free,
                allow_vasp_share=allow_vasp_share,
            )
        except GpuInventoryError as exc:
            raise E0ValDispatchError(str(exc)) from exc

    gen_root = Path(nhc0801_root) / "runs" / generation_id
    log_dir = gen_root / "logs" / "e0_val_4gpu"
    jobs = plan_val_endpoint_jobs(
        roots, gpu_ids=gpus, log_dir=log_dir, batch_id=batch_id
    )

    py = python_exe or sys.executable
    env_base = os.environ.copy()
    env_base["PYTHONPATH"] = str(Path(nhc0801_root) / "src")
    env_base["PYTHONUNBUFFERED"] = "1"

    launched: list[dict[str, Any]] = []
    for job in jobs:
        cmd = [
            py,
            "-u",
            "-m",
            "nhc_deprot.pipeline.e0_val_only",
            "--nhc0801-root",
            str(nhc0801_root),
            "--generation-id",
            generation_id,
            "--batch-id",
            batch_id,
            "--val-roots",
            job.root_id,
            "--endpoint",
            job.endpoint,
            "--parent-backend",
            parent_backend,
            "--cuda-device",
            str(job.gpu_index),
            "--max-steps",
            str(int(parent_max_steps)),
        ]
        entry: dict[str, Any] = {
            "key": job.key,
            "root_id": job.root_id,
            "endpoint": job.endpoint,
            "gpu_index": job.gpu_index,
            "log_path": str(job.log_path),
            "cmd": cmd,
        }
        if dry_run:
            entry["pid"] = None
            entry["status"] = "DRY_RUN_PLANNED"
        else:
            env = env_base.copy()
            env["CUDA_VISIBLE_DEVICES"] = str(job.gpu_index)
            job.log_path.parent.mkdir(parents=True, exist_ok=True)
            log_fh = job.log_path.open("w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd,
                cwd=str(nhc0801_root),
                env=env,
                stdout=log_fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            entry["pid"] = int(proc.pid)
            entry["status"] = "LAUNCHED"
            # parent closes; child keeps fd
            log_fh.close()
        launched.append(entry)

    receipt = {
        "schema": "nhc0801-e0-val-4gpu-dispatch-v1",
        "created_at_utc": _utc(),
        "batch_id": batch_id,
        "generation_id": generation_id,
        "val_roots": roots,
        "n_endpoints": len(jobs),
        "gpu_ids": gpus,
        "parent_backend": parent_backend,
        "parent_max_steps": parent_max_steps,
        "require_free": require_free,
        "allow_shared": allow_shared,
        "allow_vasp_share": allow_vasp_share,
        "dry_run": dry_run,
        "inventory": inv,
        "endpoints": launched,
        # backward key for older daemons
        "shards": launched,
        "notes": [
            "Val e0: N roots × cation/neutral 分开算 → round-robin over gpu_ids",
            "Prefer 1 endpoint per GPU (e.g. 4 roots → 8 GPUs when max_gpu=8)",
            "GPU pick: no-VASP, free/low-mem first (gpu_inventory.pick_gpus)",
            "Does not kill daemons or other users' jobs",
            "AIMNet2 GAU_LOOSE budget from GAU_LOOSE_V001.yaml (not parent --max-steps)",
        ],
    }
    plan_path = log_dir / f"dispatch_{batch_id}_{int(time.time())}.json"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        __import__("json").dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    receipt["plan_path"] = str(plan_path)
    return receipt


def endpoints_as_table(receipt: dict[str, Any]) -> str:
    lines = [
        "root_id | endpoint | gpu | pid | log",
        "--- | --- | ---: | ---: | ---",
    ]
    rows = receipt.get("endpoints") or receipt.get("shards") or []
    for s in rows:
        lines.append(
            f"{s.get('root_id')} | {s.get('endpoint')} | {s.get('gpu_index')} | "
            f"{s.get('pid')} | {s.get('log_path')}"
        )
    return "\n".join(lines)


# Deprecated alias
shards_as_table = endpoints_as_table
