"""Resource claim evaluation (pure logic).

Live host sampling is injected as snapshots — this module never SSHes or
starts chemistry. Strategy S: evaluate gates for a selected profile.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Final, Mapping, Sequence

from nhc_deprot.resources.profiles import ResourceProfile, get_profile

CLAIM_SCHEMA: Final = "nhc0801-live-resource-claim-v1"


class ResourceClaimError(RuntimeError):
    """Resource claim payload is invalid."""


@dataclass(frozen=True, slots=True)
class HostSnapshot:
    """One sample of host resource state (units: bytes / fraction 0–1)."""

    selected_cpus_busy: bool
    mem_available_bytes: int
    memory_psi_avg10: float
    io_psi_avg10: float
    disk_free_bytes: int
    timestamp_utc: str | None = None
    notes: tuple[str, ...] = ()


@dataclass
class ClaimGates:
    two_sample_required: bool = True
    selected_cpus_must_be_idle: bool = True
    memory_psi_avg10_max: float = 0.01
    io_psi_avg10_max: float = 0.05
    min_mem_available_bytes: int = 100_000_000_000
    min_disk_free_bytes: int = 50_000_000_000


@dataclass
class ClaimResult:
    schema: str = CLAIM_SCHEMA
    status: str = "LIVE_RESOURCE_CLAIM_REJECTED"  # or PASS
    profile_id: str = ""
    sample_count: int = 0
    reasons: list[str] = field(default_factory=list)
    chemistry_permitted: bool = False
    dual_escalation_permitted: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def gates_from_catalog(catalog: Mapping[str, Any]) -> ClaimGates:
    raw = catalog.get("claim_gates") or {}
    return ClaimGates(
        two_sample_required=bool(raw.get("two_sample_required", True)),
        selected_cpus_must_be_idle=bool(raw.get("selected_cpus_must_be_idle", True)),
        memory_psi_avg10_max=float(raw.get("memory_psi_avg10_max", 0.01)),
        io_psi_avg10_max=float(raw.get("io_psi_avg10_max", 0.05)),
        min_mem_available_bytes=int(raw.get("min_mem_available_bytes", 100_000_000_000)),
        min_disk_free_bytes=int(raw.get("min_disk_free_bytes", 50_000_000_000)),
    )


def _eval_one(snapshot: HostSnapshot, gates: ClaimGates, profile: ResourceProfile) -> list[str]:
    reasons: list[str] = []
    if gates.selected_cpus_must_be_idle and snapshot.selected_cpus_busy:
        reasons.append("SELECTED_CPU_BUNDLE_BUSY")
    # Profile-aware memory floor: catalog min and aggregate budget + reserve
    if profile.is_auto_fill:
        # Need reserve + at least one endpoint budget
        per = profile.memory_per_endpoint_mb or profile.pyscf_max_memory_mb_per_worker
        need = max(
            gates.min_mem_available_bytes,
            (profile.host_memory_reserve_mb + per) * 1024 * 1024,
        )
    else:
        need = max(
            gates.min_mem_available_bytes,
            (profile.aggregate_memory_budget_mb + profile.host_memory_reserve_mb)
            * 1024
            * 1024,
        )
    if snapshot.mem_available_bytes < need:
        reasons.append("MEM_AVAILABLE_BELOW_FLOOR")
    if snapshot.memory_psi_avg10 > gates.memory_psi_avg10_max:
        reasons.append("MEMORY_PSI_TOO_HIGH")
    if snapshot.io_psi_avg10 > gates.io_psi_avg10_max:
        reasons.append("IO_PSI_TOO_HIGH")
    if snapshot.disk_free_bytes < gates.min_disk_free_bytes:
        reasons.append("DISK_FREE_BELOW_FLOOR")
    return reasons


def evaluate_claim(
    *,
    samples: Sequence[HostSnapshot],
    profile_id: str = "single_27_physical_v1",
    gates: ClaimGates | None = None,
    profile: ResourceProfile | None = None,
) -> ClaimResult:
    """Evaluate one or two host snapshots against claim gates (no side effects)."""

    prof = profile or get_profile(profile_id)
    g = gates or ClaimGates()
    if g.two_sample_required and len(samples) < 2:
        return ClaimResult(
            status="LIVE_RESOURCE_CLAIM_REJECTED",
            profile_id=prof.profile_id,
            sample_count=len(samples),
            reasons=["TWO_SAMPLE_REQUIRED"],
            chemistry_permitted=False,
            dual_escalation_permitted=False,
        )
    if not samples:
        return ClaimResult(
            status="LIVE_RESOURCE_CLAIM_REJECTED",
            profile_id=prof.profile_id,
            sample_count=0,
            reasons=["NO_SAMPLES"],
            chemistry_permitted=False,
        )

    all_reasons: list[str] = []
    for index, sample in enumerate(samples):
        reasons = _eval_one(sample, g, prof)
        for reason in reasons:
            tagged = f"sample{index}:{reason}"
            if tagged not in all_reasons:
                all_reasons.append(tagged)

    ok = not all_reasons
    # Dual still needs separate calibration receipt; claim PASS only unlocks single by default
    dual_ok = ok and prof.profile_id == "dual_14_13_physical_v1"
    chem_ok = ok and (
        prof.profile_id == "single_27_physical_v1"
        or prof.is_auto_fill
        or prof.profile_id.startswith("legacy_")
    )
    return ClaimResult(
        status="LIVE_RESOURCE_CLAIM_PASS" if ok else "LIVE_RESOURCE_CLAIM_REJECTED",
        profile_id=prof.profile_id,
        sample_count=len(samples),
        reasons=all_reasons,
        chemistry_permitted=chem_ok,
        dual_escalation_permitted=dual_ok,
        notes=[
            "chemistry_permitted applies only after user opens teacher/epoch0 gates",
            "dual requires isolated_benchmark selection receipt even if claim PASS",
            "auto_fill V002: N=min(idle/t, mem) planned after claim; does not spawn chemistry",
        ],
    )


def pilot_v002_busy_samples() -> tuple[HostSnapshot, HostSnapshot]:
    """Synthetic stand-in for handoff V002 rejection (CPUs busy, memory OK)."""

    return (
        HostSnapshot(
            selected_cpus_busy=True,
            mem_available_bytes=234_967_650_304,
            memory_psi_avg10=0.0,
            io_psi_avg10=0.0,
            disk_free_bytes=172_664_582_144,
            timestamp_utc="2026-08-01T00:00:00Z",
            notes=("synthetic_v002_like",),
        ),
        HostSnapshot(
            selected_cpus_busy=True,
            mem_available_bytes=234_967_650_304,
            memory_psi_avg10=0.0,
            io_psi_avg10=0.0,
            disk_free_bytes=172_664_582_144,
            timestamp_utc="2026-08-01T00:01:00Z",
            notes=("synthetic_v002_like",),
        ),
    )
