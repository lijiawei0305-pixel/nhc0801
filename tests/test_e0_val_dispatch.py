"""Unit tests for Val e0 4-GPU plan (no nvidia-smi / no spawn)."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.data.paths import LEGACY_PILOT_TRAIN_ROOTS, LEGACY_PILOT_VALIDATION_ROOTS
from nhc_deprot.pipeline.e0_val_dispatch import (
    E0ValDispatchError,
    plan_val_endpoint_jobs,
)
from nhc_deprot.resources.gpu_inventory import GpuSlot, pick_gpus


def test_plan_val_endpoint_jobs_maps_four() -> None:
    # Pilot-size Val (2 roots → 4 endpoints) still used for unit mapping checks.
    roots = list(LEGACY_PILOT_VALIDATION_ROOTS)
    shards = plan_val_endpoint_jobs(
        roots,
        gpu_ids=[0, 3, 5, 7],
        log_dir=Path("/tmp/e0_test_logs"),
        batch_id="g001",
    )
    assert len(shards) == 4
    assert {s.endpoint for s in shards} == {"cation", "neutral"}
    assert {s.root_id for s in shards} == set(roots)
    assert [s.gpu_index for s in shards] == [0, 3, 5, 7]
    assert shards[0].endpoint == "cation"
    assert shards[1].endpoint == "neutral"


def test_plan_refuses_empty_roots() -> None:
    with pytest.raises(E0ValDispatchError, match=">= 1 root"):
        plan_val_endpoint_jobs(
            [],
            gpu_ids=[0, 1, 2, 3],
            log_dir=Path("/tmp/x"),
            batch_id="g001",
        )


def test_plan_refuses_train_roots() -> None:
    with pytest.raises(E0ValDispatchError, match="train roots"):
        plan_val_endpoint_jobs(
            [LEGACY_PILOT_TRAIN_ROOTS[0], list(LEGACY_PILOT_VALIDATION_ROOTS)[0]],
            gpu_ids=[0, 1, 2, 3],
            log_dir=Path("/tmp/x"),
            batch_id="g001",
        )


def test_pick_gpus_prefers_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    slots = [
        GpuSlot(0, 2000, 50, False, 2, ("python", "python")),
        GpuSlot(1, 100, 5, False, 0, ()),
        GpuSlot(2, 5000, 90, True, 1, ("vasp_std",)),
        GpuSlot(3, 200, 10, False, 0, ()),
        GpuSlot(4, 300, 10, False, 0, ()),
        GpuSlot(5, 400, 10, False, 0, ()),
        GpuSlot(6, 1000, 20, False, 1, ("python",)),
        GpuSlot(7, 150, 5, False, 0, ()),
    ]

    def fake_inv(max_gpu: int = 8) -> list[GpuSlot]:
        return slots

    monkeypatch.setattr(
        "nhc_deprot.resources.gpu_inventory.inventory_gpus", fake_inv
    )
    # free sorted by used_mib then index: 1(100), 7(150), 3(200), 4(300), 5(400)
    picked = pick_gpus(4, require_free=True)
    assert picked == [1, 7, 3, 4]
    # VASP GPU 2 never selected
    assert 2 not in pick_gpus(4, allow_shared=True)
