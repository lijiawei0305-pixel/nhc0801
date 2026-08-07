"""Resource profile / claim / worker pool tests (no live SSH)."""

from __future__ import annotations

import pytest

from nhc_deprot.data.paths import LEGACY_PILOT_TRAIN_ROOTS
from nhc_deprot.resources.claim import (
    HostSnapshot,
    evaluate_claim,
    pilot_v002_busy_samples,
)
from nhc_deprot.resources.profiles import (
    DUAL_CANDIDATE,
    OFFICIAL_DEFAULT,
    ResourceProfileError,
    assert_profile_allowed_for_chemistry,
    default_collection_profile_id,
    get_profile,
)
from nhc_deprot.resources.worker_pool import (
    WorkerPoolError,
    assert_ready_for_live_dispatch,
    build_pool,
    claim_next_root,
    complete_root,
    progress_summary,
)


def test_load_single_and_dual_profiles() -> None:
    single = get_profile(OFFICIAL_DEFAULT)
    assert single.worker_count == 1
    assert single.root_concurrency == 1
    assert single.smt is False
    dual = get_profile(DUAL_CANDIDATE)
    assert dual.worker_count == 2
    assert dual.requires_isolated_benchmark_receipt is True
    assert default_collection_profile_id() == OFFICIAL_DEFAULT


def test_v002_like_claim_rejected() -> None:
    result = evaluate_claim(samples=pilot_v002_busy_samples(), profile_id=OFFICIAL_DEFAULT)
    assert result.status == "LIVE_RESOURCE_CLAIM_REJECTED"
    assert result.chemistry_permitted is False
    assert any("SELECTED_CPU_BUNDLE_BUSY" in r for r in result.reasons)


def test_idle_claim_pass_single() -> None:
    samples = (
        HostSnapshot(False, 250_000_000_000, 0.0, 0.0, 200_000_000_000),
        HostSnapshot(False, 249_000_000_000, 0.0, 0.0, 199_000_000_000),
    )
    result = evaluate_claim(samples=samples, profile_id=OFFICIAL_DEFAULT)
    assert result.status == "LIVE_RESOURCE_CLAIM_PASS"
    assert result.chemistry_permitted is True


def test_dual_requires_receipt_even_if_claim_pass() -> None:
    dual = get_profile(DUAL_CANDIDATE)
    with pytest.raises(ResourceProfileError, match="selection receipt"):
        assert_profile_allowed_for_chemistry(
            dual, claim_pass=True, selection_receipt_present=False
        )
    assert_profile_allowed_for_chemistry(
        dual, claim_pass=True, selection_receipt_present=True
    )


def test_worker_pool_claim_complete_no_retry() -> None:
    single = get_profile(OFFICIAL_DEFAULT)
    pool = build_pool(single, LEGACY_PILOT_TRAIN_ROOTS, claim_pass=True)
    assert claim_next_root(pool, 0) == LEGACY_PILOT_TRAIN_ROOTS[0]
    with pytest.raises(WorkerPoolError, match="not idle"):
        claim_next_root(pool, 0)
    complete_root(pool, LEGACY_PILOT_TRAIN_ROOTS[0], success=False, reason="SCF_FAIL")
    assert pool.tasks[LEGACY_PILOT_TRAIN_ROOTS[0]].status == "failed"
    # no auto re-queue
    assert pool.tasks[LEGACY_PILOT_TRAIN_ROOTS[0]].status != "ready"
    nxt = claim_next_root(pool, 0)
    assert nxt == LEGACY_PILOT_TRAIN_ROOTS[1]
    complete_root(pool, LEGACY_PILOT_TRAIN_ROOTS[1], success=True)
    summary = progress_summary(pool)
    assert summary["failed"] == 1
    assert summary["done"] == 1
    assert summary["ready"] == 1


def test_dual_pool_two_workers() -> None:
    dual = get_profile(DUAL_CANDIDATE)
    roots = list(LEGACY_PILOT_TRAIN_ROOTS)  # 3 pilot roots
    pool = build_pool(
        dual, roots, claim_pass=True, selection_receipt_present=True
    )
    assert len(pool.slots) == 2
    a = claim_next_root(pool, 0)
    b = claim_next_root(pool, 1)
    assert a is not None and b is not None and a != b


def test_live_dispatch_refused_by_default() -> None:
    single = get_profile(OFFICIAL_DEFAULT)
    pool = build_pool(single, LEGACY_PILOT_TRAIN_ROOTS, claim_pass=True)
    with pytest.raises(WorkerPoolError, match="live_dispatch_enabled"):
        assert_ready_for_live_dispatch(pool, single)
