"""Load frozen resource profiles (V001 strategy S + V002 auto-fill)."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

_DOCS = Path(__file__).resolve().parents[3] / "docs" / "contracts"
DEFAULT_PROFILES_V001: Final = _DOCS / "RESOURCE_PROFILES_V001.yaml"
DEFAULT_PROFILES_V002: Final = _DOCS / "RESOURCE_PROFILES_V002.yaml"

# V001 defaults (historical)
OFFICIAL_DEFAULT: Final = "single_27_physical_v1"
DUAL_CANDIDATE: Final = "dual_14_13_physical_v1"
# V002 default (trial 2026-08-02c: t=10, reserve 12 CPUs)
OFFICIAL_DEFAULT_V002: Final = "auto_fill_112_t10_r12_v1"
# Backward-compatible alias still present in V002 catalog
V002_LEGACY_ALIAS: Final = "auto_fill_112_t8_v1"

SCHEMA_V001: Final = "nhc0801-resource-profiles-v001"
SCHEMA_V002: Final = "nhc0801-resource-profiles-v002"


class ResourceProfileError(RuntimeError):
    """Resource profile catalog is invalid."""


@dataclass(frozen=True, slots=True)
class ResourceProfile:
    profile_id: str
    status: str
    worker_count: int  # 0 => dynamic (auto-fill)
    root_concurrency: int  # 0 => dynamic
    endpoint_concurrency: int  # 0 => dynamic
    threads_per_worker: int | tuple[int, ...]
    smt: bool
    cpu_lists: tuple[str, ...]
    pyscf_max_memory_mb_per_worker: int
    aggregate_memory_budget_mb: int  # 0 => derived at plan time
    host_memory_reserve_mb: int
    numa_local_required: bool
    retry: bool
    fallback: bool
    requires_isolated_benchmark_receipt: bool
    minimum_throughput_improvement_vs_single: float | None
    raw: dict[str, Any]
    catalog_schema: str = SCHEMA_V001
    dynamic: bool = False
    cpu_pool: str | None = None
    cpu_reserve_list: str | None = None
    idle_cpu_util_threshold_pct: float = 15.0
    memory_per_endpoint_mb: int = 0
    parent_pyscf_cpu_only: bool = False

    @property
    def is_auto_fill(self) -> bool:
        return self.dynamic or self.profile_id.startswith("auto_fill_")


def load_profile_catalog(path: Path | None = None) -> dict[str, Any]:
    """Load a catalog. Default remains V001 for backward-compatible callers."""

    catalog_path = path or DEFAULT_PROFILES_V001
    payload = yaml.safe_load(catalog_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ResourceProfileError("resource profiles root must be a mapping")
    schema = payload.get("schema")
    if schema not in {SCHEMA_V001, SCHEMA_V002}:
        raise ResourceProfileError(f"resource profiles schema mismatch: {schema}")
    if schema == SCHEMA_V001 and payload.get("parallel_strategy") != "S":
        raise ResourceProfileError("V001 catalog parallel_strategy must be S")
    if schema == SCHEMA_V002 and payload.get("parallel_strategy") != "auto_fill":
        raise ResourceProfileError("V002 catalog parallel_strategy must be auto_fill")
    return payload


def load_v002_catalog(path: Path | None = None) -> dict[str, Any]:
    return load_profile_catalog(path or DEFAULT_PROFILES_V002)


def _as_int_or_dynamic(value: object, *, field: str) -> int:
    if value == "dynamic" or value is None:
        return 0
    if type(value) is int:
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    raise ResourceProfileError(f"{field} must be int or 'dynamic'")


def _parse_profile(profile_id: str, raw: dict[str, Any], *, catalog_schema: str) -> ResourceProfile:
    threads = raw.get("threads_per_worker", 8)
    if isinstance(threads, list):
        threads_t: int | tuple[int, ...] = tuple(int(x) for x in threads)
    else:
        threads_t = int(threads)

    worker_count = _as_int_or_dynamic(raw.get("worker_count", 1), field="worker_count")
    root_conc = _as_int_or_dynamic(raw.get("root_concurrency", 1), field="root_concurrency")
    ep_conc = _as_int_or_dynamic(
        raw.get("endpoint_concurrency", 1), field="endpoint_concurrency"
    )
    dynamic = worker_count == 0 or str(raw.get("worker_count")) == "dynamic"

    cpu_lists = tuple(str(x) for x in (raw.get("cpu_lists") or []))
    cpu_pool = raw.get("cpu_pool")
    if cpu_pool is not None:
        cpu_pool = str(cpu_pool)
    if not dynamic and not cpu_lists:
        raise ResourceProfileError(f"profile {profile_id} missing cpu_lists")
    if not dynamic and worker_count != len(cpu_lists):
        raise ResourceProfileError(
            f"profile {profile_id}: worker_count {worker_count} != len(cpu_lists)"
        )

    agg = raw.get("aggregate_memory_budget_mb")
    agg_mb = 0 if agg is None else int(agg)

    smt_raw = raw.get("smt", False)
    if isinstance(smt_raw, str):
        smt = smt_raw.lower() in {"true", "allowed_in_pool", "yes", "1"}
    else:
        smt = bool(smt_raw)

    host_reserve = int(raw.get("host_memory_reserve_mb", 40960 if dynamic else 32000))
    mem_per = int(raw.get("pyscf_max_memory_mb_per_worker", 30000 if dynamic else 64000))
    # V002: 8 GiB/endpoint (measured HWM~4.5 GiB); not 32 GiB
    memory_per_endpoint_mb = int(
        raw.get("memory_per_endpoint_mb")
        or (8 * 1024 if dynamic else mem_per)
    )
    cpu_reserve_list = raw.get("cpu_reserve_list")
    if cpu_reserve_list is not None:
        cpu_reserve_list = str(cpu_reserve_list)

    return ResourceProfile(
        profile_id=profile_id,
        status=str(raw.get("status", "")),
        worker_count=worker_count,
        root_concurrency=root_conc,
        endpoint_concurrency=ep_conc,
        threads_per_worker=threads_t,
        smt=smt,
        cpu_lists=cpu_lists,
        pyscf_max_memory_mb_per_worker=mem_per,
        aggregate_memory_budget_mb=agg_mb,
        host_memory_reserve_mb=host_reserve,
        numa_local_required=bool(raw.get("numa_local_required", not dynamic)),
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
        catalog_schema=catalog_schema,
        dynamic=dynamic,
        cpu_pool=cpu_pool,
        cpu_reserve_list=cpu_reserve_list if dynamic else None,
        idle_cpu_util_threshold_pct=float(raw.get("idle_cpu_util_threshold_pct", 15.0)),
        memory_per_endpoint_mb=memory_per_endpoint_mb,
        parent_pyscf_cpu_only=bool(raw.get("parent_pyscf_cpu_only", dynamic)),
    )


def get_profile(
    profile_id: str,
    *,
    path: Path | None = None,
    prefer_v002: bool | None = None,
) -> ResourceProfile:
    """Resolve profile from V002 and/or V001 catalogs.

    Search order:
      - if path given: that catalog only
      - if profile looks like auto_fill / legacy bridge / prefer_v002: V002 then V001
      - else: V001 then V002
    """

    if path is not None:
        catalog = load_profile_catalog(path)
        profiles = catalog.get("profiles")
        if not isinstance(profiles, dict) or profile_id not in profiles:
            raise ResourceProfileError(f"unknown profile: {profile_id}")
        raw = profiles[profile_id]
        if not isinstance(raw, dict):
            raise ResourceProfileError(f"profile {profile_id} is not a mapping")
        return _parse_profile(profile_id, raw, catalog_schema=str(catalog["schema"]))

    v002_first = prefer_v002
    if v002_first is None:
        v002_first = (
            profile_id.startswith("auto_fill_")
            or profile_id.startswith("legacy_")
            or profile_id == OFFICIAL_DEFAULT_V002
        )
    order = (
        (DEFAULT_PROFILES_V002, DEFAULT_PROFILES_V001)
        if v002_first
        else (DEFAULT_PROFILES_V001, DEFAULT_PROFILES_V002)
    )
    last_err: Exception | None = None
    for cat_path in order:
        if not cat_path.is_file():
            continue
        try:
            catalog = load_profile_catalog(cat_path)
            profiles = catalog.get("profiles")
            if isinstance(profiles, dict) and profile_id in profiles:
                raw = profiles[profile_id]
                if not isinstance(raw, dict):
                    raise ResourceProfileError(f"profile {profile_id} is not a mapping")
                return _parse_profile(
                    profile_id, raw, catalog_schema=str(catalog["schema"])
                )
        except Exception as exc:  # noqa: BLE001
            last_err = exc
            continue
    if last_err:
        raise ResourceProfileError(f"unknown profile: {profile_id} ({last_err})")
    raise ResourceProfileError(f"unknown profile: {profile_id}")


def default_collection_profile_id(path: Path | None = None) -> str:
    catalog = load_profile_catalog(path or DEFAULT_PROFILES_V001)
    if catalog.get("schema") == SCHEMA_V002:
        return OFFICIAL_DEFAULT_V002
    tcp = catalog.get("throughput_collection") or {}
    return str(tcp.get("default_profile", OFFICIAL_DEFAULT))


def default_v002_profile_id() -> str:
    return OFFICIAL_DEFAULT_V002


def assert_profile_allowed_for_chemistry(
    profile: ResourceProfile,
    *,
    claim_pass: bool,
    selection_receipt_present: bool,
) -> None:
    """Fail closed before any live chemistry dispatch."""

    if profile.retry or profile.fallback:
        raise ResourceProfileError("retry/fallback profiles are forbidden")
    # V001 forbade SMT; V002 auto-fill explicitly allows SMT siblings in the pool
    if profile.smt and not profile.is_auto_fill:
        raise ResourceProfileError("SMT-enabled profile is not default-allowed")
    if not claim_pass:
        raise ResourceProfileError("live resource claim has not PASSed")
    if profile.requires_isolated_benchmark_receipt and not selection_receipt_present:
        raise ResourceProfileError(
            f"profile {profile.profile_id} requires isolated benchmark selection receipt"
        )
    if not profile.is_auto_fill and profile.worker_count > 2:
        raise ResourceProfileError(
            "N>2 workers require a new RESOURCE_PROFILES catalog version (strategy S)"
        )


def worker_env_for_profile(
    profile: ResourceProfile, *, threads: int | None = None
) -> dict[str, str]:
    """Environment variables every Parent-P01 / teacher worker must set."""

    t = int(threads if threads is not None else (
        profile.threads_per_worker
        if isinstance(profile.threads_per_worker, int)
        else profile.threads_per_worker[0]
    ))
    env = {
        "OMP_NUM_THREADS": str(t),
        "MKL_NUM_THREADS": str(t),
        "OPENBLAS_NUM_THREADS": str(t),
        "NUMEXPR_NUM_THREADS": str(t),
    }
    if profile.parent_pyscf_cpu_only or profile.is_auto_fill:
        env["CUDA_VISIBLE_DEVICES"] = ""
    return env
