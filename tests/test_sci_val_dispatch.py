"""Unit tests for sci-val multi-GPU dispatch planning (no live chemistry)."""

from __future__ import annotations

import pytest

from nhc_deprot.data.paths import LEGACY_PILOT_VALIDATION_ROOTS
from nhc_deprot.pipeline.sci_val_dispatch import (
    SciValDispatchError,
    plan_endpoint_jobs,
    run_sci_val_campaign_4gpu,
)


def test_plan_endpoint_jobs_two_roots_four_gpus() -> None:
    roots = list(LEGACY_PILOT_VALIDATION_ROOTS)
    jobs = plan_endpoint_jobs(roots, gpu_ids=[5, 6, 4, 1])
    assert len(jobs) == 4
    keys = {(s["root_id"], s["endpoint"]) for s in jobs}
    assert keys == {
        (roots[0], "cation"),
        (roots[0], "neutral"),
        (roots[1], "cation"),
        (roots[1], "neutral"),
    }
    assert [s["gpu_index"] for s in jobs] == [5, 6, 4, 1]


def test_plan_rejects_wrong_gpu_count() -> None:
    with pytest.raises(SciValDispatchError, match="4 GPU"):
        plan_endpoint_jobs(list(LEGACY_PILOT_VALIDATION_ROOTS), gpu_ids=[0, 1, 2])


def test_baseline_max_steps_mismatch() -> None:
    with pytest.raises(SciValDispatchError, match="BASELINE_CONFIG_MISMATCH"):
        run_sci_val_campaign_4gpu(
            nhc0801_root=__import__("pathlib").Path("/tmp/nope"),
            generation_id="nhc0801-g001",
            candidates=[{"seed": 1, "epoch": 10, "weight_path": "/x"}],
            max_steps=250,
            epoch0_max_steps=100,
            dry_run=True,
        )
