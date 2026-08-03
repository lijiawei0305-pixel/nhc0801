"""Root-level worker slot model (parallel strategy S).

Coordinates N in-process logical workers with disjoint CPU list labels.
Does not spawn processes or run chemistry — claim/release + assignment only.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from nhc_deprot.resources.profiles import ResourceProfile, assert_profile_allowed_for_chemistry


class WorkerPoolError(RuntimeError):
    """Worker pool assignment failed closed."""


@dataclass
class WorkerSlot:
    worker_id: int
    cpu_list: str
    threads: int
    status: str = "idle"  # idle | busy | failed
    claimed_root: str | None = None


@dataclass
class RootTask:
    root_id: str
    status: str = "ready"  # ready | claimed | done | failed
    worker_id: int | None = None
    failure_reason: str | None = None


@dataclass
class WorkerPool:
    profile_id: str
    slots: list[WorkerSlot] = field(default_factory=list)
    tasks: dict[str, RootTask] = field(default_factory=dict)
    claim_pass: bool = False
    selection_receipt_present: bool = False
    live_dispatch_enabled: bool = False  # always false until explicit gate

    def as_dict(self) -> dict[str, Any]:
        return {
            "profile_id": self.profile_id,
            "claim_pass": self.claim_pass,
            "selection_receipt_present": self.selection_receipt_present,
            "live_dispatch_enabled": self.live_dispatch_enabled,
            "slots": [asdict(s) for s in self.slots],
            "tasks": {k: asdict(v) for k, v in self.tasks.items()},
        }


def build_pool(
    profile: ResourceProfile,
    root_ids: Sequence[str],
    *,
    claim_pass: bool = False,
    selection_receipt_present: bool = False,
) -> WorkerPool:
    if profile.retry or profile.fallback:
        raise WorkerPoolError("retry/fallback not allowed")
    if len(set(root_ids)) != len(root_ids):
        raise WorkerPoolError("duplicate root_id in queue")
    threads = profile.threads_per_worker
    if isinstance(threads, int):
        thread_list = [threads] * profile.worker_count
    else:
        thread_list = list(threads)
        if len(thread_list) != profile.worker_count:
            raise WorkerPoolError("threads_per_worker length mismatch")

    slots = [
        WorkerSlot(worker_id=i, cpu_list=profile.cpu_lists[i], threads=thread_list[i])
        for i in range(profile.worker_count)
    ]
    tasks = {root_id: RootTask(root_id=root_id) for root_id in root_ids}
    return WorkerPool(
        profile_id=profile.profile_id,
        slots=slots,
        tasks=tasks,
        claim_pass=claim_pass,
        selection_receipt_present=selection_receipt_present,
        live_dispatch_enabled=False,
    )


def assert_ready_for_live_dispatch(pool: WorkerPool, profile: ResourceProfile) -> None:
    assert_profile_allowed_for_chemistry(
        profile,
        claim_pass=pool.claim_pass,
        selection_receipt_present=pool.selection_receipt_present,
    )
    if not pool.live_dispatch_enabled:
        raise WorkerPoolError(
            "live_dispatch_enabled=false (default); refuse process spawn / chemistry"
        )


def claim_next_root(pool: WorkerPool, worker_id: int) -> str | None:
    """Atomically (in-process) assign next ready root to an idle worker."""

    if worker_id < 0 or worker_id >= len(pool.slots):
        raise WorkerPoolError(f"invalid worker_id: {worker_id}")
    slot = pool.slots[worker_id]
    if slot.status != "idle" or slot.claimed_root is not None:
        raise WorkerPoolError(f"worker {worker_id} is not idle")
    for root_id, task in pool.tasks.items():
        if task.status == "ready":
            task.status = "claimed"
            task.worker_id = worker_id
            slot.status = "busy"
            slot.claimed_root = root_id
            return root_id
    return None


def complete_root(
    pool: WorkerPool, root_id: str, *, success: bool, reason: str | None = None
) -> None:
    task = pool.tasks.get(root_id)
    if task is None:
        raise WorkerPoolError(f"unknown root: {root_id}")
    if task.status != "claimed" or task.worker_id is None:
        raise WorkerPoolError(f"root {root_id} is not claimed")
    worker_id = task.worker_id
    slot = pool.slots[worker_id]
    if success:
        task.status = "done"
        task.failure_reason = None
    else:
        task.status = "failed"
        task.failure_reason = reason or "FAILED"
        # strategy S: no automatic retry
    slot.status = "idle"
    slot.claimed_root = None
    task.worker_id = None


def progress_summary(pool: WorkerPool) -> dict[str, int]:
    counts = {"ready": 0, "claimed": 0, "done": 0, "failed": 0}
    for task in pool.tasks.values():
        counts[task.status] = counts.get(task.status, 0) + 1
    return counts
