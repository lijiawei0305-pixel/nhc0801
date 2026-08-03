"""V002 auto-fill capacity planner and endpoint CPU slot allocator.

Does not spawn chemistry. Pure planning given idle CPU counts and memory.
Authority: docs/contracts/RESOURCE_SCHEDULING_V001.md + RESOURCE_PROFILES_V002.yaml
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from nhc_deprot.resources.host_sampler import expand_cpu_list
from nhc_deprot.resources.profiles import (
    OFFICIAL_DEFAULT_V002,
    ResourceProfile,
    get_profile,
    worker_env_for_profile,
)

GIB: int = 1024**3


class AutoFillError(RuntimeError):
    """Auto-fill planning failed closed."""


@dataclass(frozen=True, slots=True)
class AutoFillCapacity:
    """Result of N = min(N_cpu, N_mem)."""

    profile_id: str
    threads_per_endpoint: int
    idle_logical_cpus: int
    mem_available_bytes: int
    host_memory_reserve_bytes: int
    memory_per_endpoint_bytes: int
    n_cpu: int
    n_mem: int
    n: int
    n_cap: int | None
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class EndpointSlot:
    slot_id: int
    cpu_ids: list[int]
    cpu_list: str
    threads: int
    status: str = "idle"  # idle | reserved | busy
    root_id: str | None = None
    endpoint: str | None = None

    def env(self, profile: ResourceProfile) -> dict[str, str]:
        env = worker_env_for_profile(profile, threads=self.threads)
        env["NHC0801_CPU_LIST"] = self.cpu_list
        env["NHC0801_TASKSET"] = self.cpu_list
        return env

    def as_dict(self) -> dict[str, Any]:
        return {
            "slot_id": self.slot_id,
            "cpu_ids": list(self.cpu_ids),
            "cpu_list": self.cpu_list,
            "threads": self.threads,
            "status": self.status,
            "root_id": self.root_id,
            "endpoint": self.endpoint,
        }


@dataclass
class EndpointTask:
    root_id: str
    endpoint: str  # cation | neutral
    status: str = "ready"  # ready | claimed | done | failed
    slot_id: int | None = None
    failure_reason: str | None = None

    @property
    def key(self) -> str:
        return f"{self.root_id}:{self.endpoint}"

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class AutoFillPlan:
    """Concrete slot layout for up to N concurrent endpoint workers."""

    capacity: AutoFillCapacity
    slots: list[EndpointSlot] = field(default_factory=list)
    tasks: dict[str, EndpointTask] = field(default_factory=dict)
    free_cpu_ids: list[int] = field(default_factory=list)
    claim_pass: bool = False
    live_dispatch_enabled: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "capacity": self.capacity.as_dict(),
            "slots": [s.as_dict() for s in self.slots],
            "tasks": {k: v.as_dict() for k, v in self.tasks.items()},
            "free_cpu_ids": list(self.free_cpu_ids),
            "claim_pass": self.claim_pass,
            "live_dispatch_enabled": self.live_dispatch_enabled,
            "progress": progress_endpoints(self),
        }


def compute_capacity(
    *,
    idle_logical_cpus: int,
    mem_available_bytes: int,
    profile: ResourceProfile | None = None,
    profile_id: str = OFFICIAL_DEFAULT_V002,
    n_cap: int | None = None,
) -> AutoFillCapacity:
    prof = profile or get_profile(profile_id)
    if not isinstance(prof.threads_per_worker, int):
        raise AutoFillError("auto-fill requires scalar threads_per_worker")
    t = int(prof.threads_per_worker)
    if t <= 0:
        raise AutoFillError("threads_per_worker must be positive")
    idle = max(0, int(idle_logical_cpus))
    mem = max(0, int(mem_available_bytes))
    reserve = int(prof.host_memory_reserve_mb) * 1024 * 1024
    per = int(prof.memory_per_endpoint_mb or prof.pyscf_max_memory_mb_per_worker) * 1024 * 1024
    if per <= 0:
        raise AutoFillError("memory_per_endpoint must be positive")

    n_cpu = idle // t
    n_mem = max(0, (mem - reserve) // per) if mem > reserve else 0
    n = min(n_cpu, n_mem)
    if n_cap is not None:
        n = min(n, int(n_cap))
    notes = [
        f"t={t}",
        f"N=min(N_cpu={n_cpu}, N_mem={n_mem}"
        + (f", N_cap={n_cap}" if n_cap is not None else "")
        + f")={n}",
        "parent_pyscf_cpu_only" if prof.parent_pyscf_cpu_only else "gpu_ok_for_parent",
    ]
    return AutoFillCapacity(
        profile_id=prof.profile_id,
        threads_per_endpoint=t,
        idle_logical_cpus=idle,
        mem_available_bytes=mem,
        host_memory_reserve_bytes=reserve,
        memory_per_endpoint_bytes=per,
        n_cpu=n_cpu,
        n_mem=n_mem,
        n=n,
        n_cap=n_cap,
        notes=tuple(notes),
    )


def _cpu_list_str(ids: Sequence[int]) -> str:
    if not ids:
        return ""
    ids_s = sorted(int(x) for x in ids)
    # compact ranges
    parts: list[str] = []
    start = prev = ids_s[0]
    for x in ids_s[1:]:
        if x == prev + 1:
            prev = x
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = x
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return ",".join(parts)


def build_auto_fill_plan(
    *,
    idle_cpu_ids: Sequence[int],
    mem_available_bytes: int,
    endpoint_queue: Sequence[tuple[str, str]],
    profile: ResourceProfile | None = None,
    profile_id: str = OFFICIAL_DEFAULT_V002,
    claim_pass: bool = False,
    n_cap: int | None = None,
) -> AutoFillPlan:
    """Build N endpoint slots from idle CPU ids + memory; queue all endpoints as tasks."""

    prof = profile or get_profile(profile_id)
    t = int(prof.threads_per_worker) if isinstance(prof.threads_per_worker, int) else 8
    idle_ids = sorted({int(x) for x in idle_cpu_ids})
    cap = compute_capacity(
        idle_logical_cpus=len(idle_ids),
        mem_available_bytes=mem_available_bytes,
        profile=prof,
        n_cap=n_cap,
    )
    n = cap.n
    slots: list[EndpointSlot] = []
    remaining = list(idle_ids)
    for i in range(n):
        if len(remaining) < t:
            break
        chunk = remaining[:t]
        remaining = remaining[t:]
        slots.append(
            EndpointSlot(
                slot_id=i,
                cpu_ids=list(chunk),
                cpu_list=_cpu_list_str(chunk),
                threads=t,
            )
        )
    # If memory limited n below what CPUs allow, slots already capped by n
    tasks: dict[str, EndpointTask] = {}
    for root_id, endpoint in endpoint_queue:
        if endpoint not in {"cation", "neutral"}:
            raise AutoFillError(f"invalid endpoint: {endpoint}")
        task = EndpointTask(root_id=root_id, endpoint=endpoint)
        if task.key in tasks:
            raise AutoFillError(f"duplicate endpoint task: {task.key}")
        tasks[task.key] = task
    return AutoFillPlan(
        capacity=cap,
        slots=slots,
        tasks=tasks,
        free_cpu_ids=remaining,
        claim_pass=claim_pass,
        live_dispatch_enabled=False,
    )


def claim_next_endpoint(plan: AutoFillPlan, slot_id: int) -> EndpointTask | None:
    if slot_id < 0 or slot_id >= len(plan.slots):
        raise AutoFillError(f"invalid slot_id: {slot_id}")
    slot = plan.slots[slot_id]
    if slot.status != "idle":
        raise AutoFillError(f"slot {slot_id} is not idle")
    for task in plan.tasks.values():
        if task.status == "ready":
            task.status = "claimed"
            task.slot_id = slot_id
            slot.status = "busy"
            slot.root_id = task.root_id
            slot.endpoint = task.endpoint
            return task
    return None


def complete_endpoint(
    plan: AutoFillPlan,
    root_id: str,
    endpoint: str,
    *,
    success: bool,
    reason: str | None = None,
) -> None:
    key = f"{root_id}:{endpoint}"
    task = plan.tasks.get(key)
    if task is None:
        raise AutoFillError(f"unknown task: {key}")
    if task.status != "claimed" or task.slot_id is None:
        raise AutoFillError(f"task {key} is not claimed")
    slot = plan.slots[task.slot_id]
    if success:
        task.status = "done"
        task.failure_reason = None
    else:
        task.status = "failed"
        task.failure_reason = reason or "FAILED"
    slot.status = "idle"
    slot.root_id = None
    slot.endpoint = None
    task.slot_id = None


def progress_endpoints(plan: AutoFillPlan) -> dict[str, int]:
    counts = {"ready": 0, "claimed": 0, "done": 0, "failed": 0}
    for task in plan.tasks.values():
        counts[task.status] = counts.get(task.status, 0) + 1
    return counts


def expand_pool_cpu_ids(profile: ResourceProfile) -> list[int]:
    """CPUs eligible for auto-fill (excludes reserve list when present)."""

    if profile.cpu_pool:
        pool = set(expand_cpu_list(profile.cpu_pool))
    elif profile.cpu_lists:
        pool = set()
        for spec in profile.cpu_lists:
            pool.update(expand_cpu_list(spec))
    else:
        pool = set(range(112))
    if profile.cpu_reserve_list:
        pool -= set(expand_cpu_list(profile.cpu_reserve_list))
    return sorted(pool)


def plan_from_idle_mask(
    *,
    pool_cpu_ids: Sequence[int],
    busy_cpu_ids: Sequence[int],
    mem_available_bytes: int,
    endpoint_queue: Sequence[tuple[str, str]],
    profile: ResourceProfile | None = None,
    claim_pass: bool = False,
) -> AutoFillPlan:
    """Convenience: idle = pool − busy."""

    busy = set(int(x) for x in busy_cpu_ids)
    idle = [c for c in pool_cpu_ids if c not in busy]
    return build_auto_fill_plan(
        idle_cpu_ids=idle,
        mem_available_bytes=mem_available_bytes,
        endpoint_queue=endpoint_queue,
        profile=profile,
        claim_pass=claim_pass,
    )
