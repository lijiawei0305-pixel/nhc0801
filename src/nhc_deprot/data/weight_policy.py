"""Sample-weight policy: equal_candidate_then_equal_endpoint_then_uniform_frames.

Ported from V004 (phase9b_aimnet2_training_dataset_v004.assign_candidate_endpoint_weights).
Counts are parameters — never hardcode pilot 235 / 3+2 as global truth.
"""

from __future__ import annotations

import math
from collections.abc import MutableMapping, Sequence
from typing import Any, Final

from nhc_deprot.data.errors import DatasetError
from nhc_deprot.training.weighted_loss import SAMPLE_WEIGHT_KEY, WEIGHTING_POLICY

ENDPOINTS: Final = ("cation", "neutral")
WEIGHT_ABS_TOL: Final = 1e-12


def assign_candidate_endpoint_weights(
    records: Sequence[MutableMapping[str, Any]],
    *,
    candidate_count: int,
) -> dict[str, object]:
    """Assign equal candidate mass, equal endpoint mass, then uniform frame mass.

    Mutates each record in-place by setting ``sample_weight``.
    ``candidate_count`` is the number of molecular roots in *this split*.
    """

    if candidate_count <= 0 or not records:
        raise DatasetError("weight assignment requires candidates and frames")
    candidates = {str(record["candidate"]) for record in records}
    if len(candidates) != 1:
        raise DatasetError("weight assignment crossed candidate ownership")
    by_endpoint: dict[str, list[MutableMapping[str, Any]]] = {
        endpoint: [] for endpoint in ENDPOINTS
    }
    for record in records:
        endpoint = str(record["endpoint"])
        if endpoint not in by_endpoint:
            raise DatasetError(f"invalid endpoint: {endpoint}")
        by_endpoint[endpoint].append(record)
    if any(not endpoint_records for endpoint_records in by_endpoint.values()):
        raise DatasetError("weight assignment requires both endpoints")

    candidate_target = 1.0 / candidate_count
    endpoint_target = candidate_target / 2.0
    endpoint_evidence: dict[str, object] = {}
    observed_candidate = 0.0
    for endpoint, endpoint_records in by_endpoint.items():
        per_frame = endpoint_target / len(endpoint_records)
        for record in endpoint_records:
            record[SAMPLE_WEIGHT_KEY] = per_frame
        observed_endpoint = per_frame * len(endpoint_records)
        observed_candidate += observed_endpoint
        endpoint_evidence[endpoint] = {
            "frame_count": len(endpoint_records),
            "per_frame_weight": per_frame,
            "target_weight_sum": endpoint_target,
            "observed_weight_sum": observed_endpoint,
        }
    if not math.isclose(observed_candidate, candidate_target, rel_tol=0.0, abs_tol=1e-15):
        raise DatasetError("candidate weight sum drifted")
    return {
        "policy": WEIGHTING_POLICY,
        "candidate": next(iter(candidates)),
        "target_weight_sum": candidate_target,
        "observed_weight_sum": observed_candidate,
        "endpoints": endpoint_evidence,
    }


def audit_split_weight_sums(
    *,
    weights_by_candidate_endpoint: dict[str, dict[str, float]],
    candidate_count: int,
    abs_tol: float = WEIGHT_ABS_TOL,
) -> dict[str, object]:
    """Audit equal-candidate / equal-endpoint mass for one split.

    ``weights_by_candidate_endpoint`` maps candidate -> endpoint -> weight sum.
    """

    if candidate_count <= 0:
        raise DatasetError("candidate_count must be positive")
    if len(weights_by_candidate_endpoint) != candidate_count:
        raise DatasetError(
            f"candidate count drifted: expected {candidate_count}, "
            f"got {len(weights_by_candidate_endpoint)}"
        )
    candidate_target = 1.0 / candidate_count
    endpoint_target = candidate_target / 2.0
    rows: list[dict[str, object]] = []
    total = 0.0
    for candidate, endpoint_map in sorted(weights_by_candidate_endpoint.items()):
        if set(endpoint_map) != set(ENDPOINTS):
            raise DatasetError(f"candidate {candidate} missing cation/neutral weight mass")
        for endpoint, value in endpoint_map.items():
            if not math.isfinite(value) or value <= 0.0:
                raise DatasetError(f"non-positive weight for {candidate}/{endpoint}")
            if not math.isclose(value, endpoint_target, rel_tol=0.0, abs_tol=abs_tol):
                raise DatasetError(
                    f"endpoint weight sum drifted for {candidate}/{endpoint}: "
                    f"{value} vs target {endpoint_target}"
                )
        candidate_sum = sum(endpoint_map.values())
        if not math.isclose(candidate_sum, candidate_target, rel_tol=0.0, abs_tol=abs_tol):
            raise DatasetError(f"candidate weight sum drifted for {candidate}")
        total += candidate_sum
        rows.append(
            {
                "candidate": candidate,
                "candidate_weight_sum": candidate_sum,
                "cation_weight_sum": endpoint_map["cation"],
                "neutral_weight_sum": endpoint_map["neutral"],
            }
        )
    if not math.isclose(total, 1.0, rel_tol=0.0, abs_tol=abs_tol):
        raise DatasetError(f"split weight sum must be 1.0, got {total}")
    return {
        "status": "PASS",
        "policy": WEIGHTING_POLICY,
        "candidate_count": candidate_count,
        "candidate_target": candidate_target,
        "endpoint_target": endpoint_target,
        "split_weight_sum": total,
        "rows": rows,
    }
