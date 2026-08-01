"""Resource profiles, claim evaluation, host sampling, and worker slots."""

from nhc_deprot.resources.claim import (
    ClaimGates,
    ClaimResult,
    HostSnapshot,
    evaluate_claim,
    pilot_v002_busy_samples,
)
from nhc_deprot.resources.profiles import (
    DUAL_CANDIDATE,
    OFFICIAL_DEFAULT,
    ResourceProfile,
    assert_profile_allowed_for_chemistry,
    default_collection_profile_id,
    get_profile,
    load_profile_catalog,
)
from nhc_deprot.resources.worker_pool import (
    WorkerPool,
    assert_ready_for_live_dispatch,
    build_pool,
    claim_next_root,
    complete_root,
    progress_summary,
)

# claim_runner / host_sampler imported lazily by callers to avoid heavy import graphs

__all__ = [
    "ClaimGates",
    "ClaimResult",
    "DUAL_CANDIDATE",
    "HostSnapshot",
    "OFFICIAL_DEFAULT",
    "ResourceProfile",
    "WorkerPool",
    "assert_profile_allowed_for_chemistry",
    "assert_ready_for_live_dispatch",
    "build_pool",
    "claim_next_root",
    "complete_root",
    "default_collection_profile_id",
    "evaluate_claim",
    "get_profile",
    "load_profile_catalog",
    "pilot_v002_busy_samples",
    "progress_summary",
]
