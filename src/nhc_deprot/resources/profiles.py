"""Load frozen resource profiles (parallel strategy S)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

DEFAULT_PROFILES_PATH: Final = (
    Path(__file__).resolve().parents[3] / "docs" / "contracts" / "RESOURCE_PROFILES_V001.yaml"
)

OFFICIAL_DEFAULT: Final = "single_27_physical_v1"
DUAL_CANDIDATE: Final = "dual_14_13_physical_v1"


class ResourceProfileError(RuntimeError):
    """Resource profile catalog is invalid."""


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    profile_id: str
    status: str
    worker_count: int
    root_concurrency: int
    endpoint_concurrency: int
    threads_per_worker: int | tuple[int, ...]
    smt: bool
    cpu_lists: tuple[str, ...]
    pyscf_max_memory_mb_per_worker: int
    aggregate_memory_budget_mb: int
    host_memory_reserve_mb: int
    numa_local_required: bool
    retry: bool
    fallback: bool
    requires_isolated_benchmark_receipt: bool
    minimum_throughput_improvement_vs_single: float | None
    raw: dict[str, Any]


def load_profile_catalog(path: Path | None = None) -> dict[str, Any]:
    catalog_path = path or DEFAULT_PROFILES_PATH
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("schema") != "nhc0801-resource-profiles-v001":
        raise ResourceProfileError("resource profiles schema mismatch")
    if payload.get("parallel_strategy") != "S":
        raise ResourceProfileError("catalog parallel_strategy must be S for this code path")
    return payload


def get_profile(profile_id: str, *, path: Path | None = None) -> ResourceProfile:
    catalog = load_profile_catalog(path)
    profiles = catalog.get("profiles")
    if not isinstance(profiles, dict) or profile_id not in profiles:
        raise ResourceProfileError(f"unknown profile: {profile_id}")
    raw = profiles[profile_id]
    if not isinstance(raw, dict):
        raise ResourceProfileError(f"profile {profile_id} is not a mapping")
    threads = raw["threads_per_worker"]
    if isinstance(threads, list):
        threads_t: int | tuple[int, ...] = tuple(int(x) for x in threads)
    else:
        threads_t = int(threads)
    cpu_lists = tuple(str(x) for x in (raw.get("cpu_lists") or []))
    if not cpu_lists:
        raise ResourceProfileError(f"profile {profile_id} missing cpu_lists")
    worker_count = int(raw["worker_count"])
    if worker_count != len(cpu_lists):
        raise ResourceProfileError(
            f"profile {profile_id}: worker_count {worker_count} != len(cpu_lists)"
        )
    return ResourceProfile(
        profile_id=profile_id,
        status=str(raw.get("status", "")),
        worker_count=worker_count,
        root_concurrency=int(raw["root_concurrency"]),
        endpoint_concurrency=int(raw["endpoint_concurrency"]),
        threads_per_worker=threads_t,
        smt=bool(raw.get("smt", False)),
        cpu_lists=cpu_lists,
        pyscf_max_memory_mb_per_worker=int(raw["pyscf_max_memory_mb_per_worker"]),
        aggregate_memory_budget_mb=int(raw["aggregate_memory_budget_mb"]),
        host_memory_reserve_mb=int(raw["host_memory_reserve_mb"]),
        numa_local_required=bool(raw.get("numa_local_required", True)),
        retry=bool(raw.get("retry", False)),
        fallback=bool(raw.get("fallback", False)),
        requires_isolated_benchmark_receipt=bool(
            raw.get("requires_isolated_benchmark_receipt", False)
        ),
        minimum_throughput_improvement_vs_single=(
            float(raw["minimum_throughput_improvement_vs_single"])
            if raw.get("minimum_throughput_improvement_vs_single") is not None
            else None
        ),
        raw=dict(raw),
    )


def default_collection_profile_id(path: Path | None = None) -> str:
    catalog = load_profile_catalog(path)
    tcp = catalog.get("throughput_collection") or {}
    return str(tcp.get("default_profile", OFFICIAL_DEFAULT))


def assert_profile_allowed_for_chemistry(
    profile: ResourceProfile,
    *,
    claim_pass: bool,
    selection_receipt_present: bool,
) -> None:
    """Fail closed before any live chemistry dispatch."""

    if profile.retry or profile.fallback:
        raise ResourceProfileError("retry/fallback profiles are forbidden")
    if profile.smt:
        raise ResourceProfileError("SMT-enabled profile is not default-allowed")
    if not claim_pass:
        raise ResourceProfileError("live resource claim has not PASSed")
    if profile.requires_isolated_benchmark_receipt and not selection_receipt_present:
        raise ResourceProfileError(
            f"profile {profile.profile_id} requires isolated benchmark selection receipt"
        )
    if profile.worker_count > 2:
        raise ResourceProfileError(
            "N>2 workers require a new RESOURCE_PROFILES catalog version (strategy S)"
        )
