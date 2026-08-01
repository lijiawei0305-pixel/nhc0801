"""Weight policy unit tests (no chemistry / no HPC)."""

from __future__ import annotations

from typing import Any, cast

import pytest

from nhc_deprot.data.errors import DatasetError
from nhc_deprot.data.weight_policy import (
    assign_candidate_endpoint_weights,
    audit_split_weight_sums,
)


def test_equal_candidate_endpoint_mass_unequal_trajectories() -> None:
    records: list[dict[str, Any]] = [
        {"candidate": "A", "endpoint": "cation", "frame_index": i} for i in range(2)
    ] + [{"candidate": "A", "endpoint": "neutral", "frame_index": i} for i in range(5)]
    evidence = assign_candidate_endpoint_weights(records, candidate_count=3)
    assert evidence["observed_weight_sum"] == pytest.approx(1 / 3)
    cation = [
        float(cast(float, r["sample_weight"])) for r in records if r["endpoint"] == "cation"
    ]
    neutral = [
        float(cast(float, r["sample_weight"])) for r in records if r["endpoint"] == "neutral"
    ]
    assert sum(cation) == pytest.approx(1 / 6)
    assert sum(neutral) == pytest.approx(1 / 6)
    assert cation[0] == pytest.approx((1 / 6) / 2)
    assert neutral[0] == pytest.approx((1 / 6) / 5)


def test_weighting_rejects_missing_endpoint() -> None:
    records = [{"candidate": "A", "endpoint": "cation", "frame_index": 0}]
    with pytest.raises(DatasetError, match="both endpoints"):
        assign_candidate_endpoint_weights(records, candidate_count=1)


def test_audit_split_weight_sums_pass() -> None:
    # two candidates → each 0.5; each endpoint 0.25
    weights = {
        "A": {"cation": 0.25, "neutral": 0.25},
        "B": {"cation": 0.25, "neutral": 0.25},
    }
    out = audit_split_weight_sums(
        weights_by_candidate_endpoint=weights, candidate_count=2
    )
    assert out["status"] == "PASS"
    assert out["split_weight_sum"] == pytest.approx(1.0)


def test_audit_split_weight_sums_detects_drift() -> None:
    weights = {
        "A": {"cation": 0.3, "neutral": 0.2},
    }
    with pytest.raises(DatasetError, match="endpoint weight sum drifted"):
        audit_split_weight_sums(weights_by_candidate_endpoint=weights, candidate_count=1)
