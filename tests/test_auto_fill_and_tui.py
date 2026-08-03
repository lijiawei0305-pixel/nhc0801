"""V002 auto-fill, pipeline status, TUI render tests (no live host required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.dashboard.tui import render_status_text
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.pipeline_status import (
    scan_generation_status,
    write_pipeline_status,
    write_step_status,
)
from nhc_deprot.resources.auto_fill import (
    AutoFillError,
    build_auto_fill_plan,
    claim_next_endpoint,
    complete_endpoint,
    compute_capacity,
    expand_pool_cpu_ids,
    plan_from_idle_mask,
    progress_endpoints,
)
from nhc_deprot.resources.claim import HostSnapshot, evaluate_claim
from nhc_deprot.resources.profiles import (
    OFFICIAL_DEFAULT,
    OFFICIAL_DEFAULT_V002,
    ResourceProfileError,
    assert_profile_allowed_for_chemistry,
    get_profile,
    load_v002_catalog,
    worker_env_for_profile,
)


def test_v002_profile_loads() -> None:
    cat = load_v002_catalog()
    assert cat["schema"].endswith("v002")
    assert cat.get("revision") == "2026-08-02c"
    prof = get_profile(OFFICIAL_DEFAULT_V002)
    assert prof.is_auto_fill
    assert prof.threads_per_worker == 10
    assert prof.cpu_pool == "0-99"
    assert prof.cpu_reserve_list == "100-111"
    assert prof.host_memory_reserve_mb == 40960
    assert prof.memory_per_endpoint_mb == 8192
    env = worker_env_for_profile(prof)
    assert env["OMP_NUM_THREADS"] == "10"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    pool = expand_pool_cpu_ids(prof)
    assert len(pool) == 100
    assert 100 not in pool and 111 not in pool
    assert 0 in pool and 99 in pool


def test_v001_still_loads() -> None:
    single = get_profile(OFFICIAL_DEFAULT)
    assert single.worker_count == 1
    assert not single.is_auto_fill


def test_capacity_formula_t10() -> None:
    # pool idle 100 cpus, 200 GiB → N_cpu=10, N_mem=floor((200-40)/8)=20 → N=10
    cap = compute_capacity(
        idle_logical_cpus=100,
        mem_available_bytes=200 * (1024**3),
        profile_id=OFFICIAL_DEFAULT_V002,
    )
    assert cap.threads_per_endpoint == 10
    assert cap.n_cpu == 10
    assert cap.n_mem == 20
    assert cap.n == 10


def test_capacity_ten_teacher_endpoints_one_wave() -> None:
    cap = compute_capacity(
        idle_logical_cpus=100,
        mem_available_bytes=236 * (1024**3),
        profile_id=OFFICIAL_DEFAULT_V002,
    )
    assert cap.n_cpu == 10
    assert cap.n_mem >= 10
    assert cap.n == 10
    # 10 endpoints × 10 threads = 100 cores; 12 reserved outside pool
    assert cap.n * cap.threads_per_endpoint == 100


def test_auto_fill_plan_claim_complete() -> None:
    idle = list(range(0, 50))  # 50 cpus → 5 slots at t=10
    queue = [
        ("ROOTA", "cation"),
        ("ROOTA", "neutral"),
        ("ROOTB", "cation"),
        ("ROOTB", "neutral"),
    ]
    plan = build_auto_fill_plan(
        idle_cpu_ids=idle,
        mem_available_bytes=250 * (1024**3),
        endpoint_queue=queue,
        claim_pass=True,
    )
    assert plan.capacity.n >= 4
    assert len(plan.slots) >= 4
    assert len(plan.tasks) == 4
    assert "ROOTA:cation" in plan.tasks and "ROOTA:neutral" in plan.tasks

    t0 = claim_next_endpoint(plan, 0)
    assert t0 is not None
    assert t0.status == "claimed"
    env = plan.slots[0].env(get_profile(OFFICIAL_DEFAULT_V002))
    assert env["OMP_NUM_THREADS"] == "10"
    assert env["CUDA_VISIBLE_DEVICES"] == ""
    assert len(plan.slots[0].cpu_ids) == 10
    complete_endpoint(plan, t0.root_id, t0.endpoint, success=False, reason="SCF_FAIL")
    assert plan.tasks[t0.key].status == "failed"
    assert progress_endpoints(plan)["ready"] == 3


def test_plan_from_idle_mask() -> None:
    plan = plan_from_idle_mask(
        pool_cpu_ids=list(range(30)),
        busy_cpu_ids=[0, 1, 2, 3],
        mem_available_bytes=100 * (1024**3),
        endpoint_queue=[("R", "cation")],
    )
    # 26 idle → N_cpu=2; mem ok
    assert plan.capacity.n >= 1
    assert len(plan.slots) >= 1
    assert 0 not in plan.slots[0].cpu_ids


def test_auto_fill_claim_pass_chemistry_flag() -> None:
    samples = (
        HostSnapshot(False, 250_000_000_000, 0.0, 0.0, 200_000_000_000),
        HostSnapshot(False, 249_000_000_000, 0.0, 0.0, 199_000_000_000),
    )
    result = evaluate_claim(samples=samples, profile_id=OFFICIAL_DEFAULT_V002)
    assert result.status == "LIVE_RESOURCE_CLAIM_PASS"
    assert result.chemistry_permitted is True
    prof = get_profile(OFFICIAL_DEFAULT_V002)
    assert_profile_allowed_for_chemistry(
        prof, claim_pass=True, selection_receipt_present=False
    )


def test_auto_fill_rejects_unset_claim() -> None:
    prof = get_profile(OFFICIAL_DEFAULT_V002)
    with pytest.raises(ResourceProfileError, match="claim"):
        assert_profile_allowed_for_chemistry(
            prof, claim_pass=False, selection_receipt_present=False
        )


def test_pipeline_status_and_tui(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    sd = layout.train_dir / "seed_20260730"
    sd.mkdir(parents=True)
    (sd / "seed_receipt.json").write_text(
        json.dumps(
            {
                "seed": 20260730,
                "status": "PASS",
                "epochs_run": 2,
                "shortlist_epochs": [1, 2],
                "epoch_logs": [
                    {
                        "epoch": 1,
                        "train": {"train_weighted_loss": 0.5},
                        "quick_validation": {"validation_weighted_loss": 1.2},
                    },
                    {
                        "epoch": 2,
                        "train": {"train_weighted_loss": 0.4},
                        "quick_validation": {"validation_weighted_loss": 1.0},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    (layout.train_dir / "campaign_receipt_live.json").write_text(
        json.dumps({"status": "LIVE_TRAIN_PASS", "failed_seed_count": 0}),
        encoding="utf-8",
    )
    write_step_status(layout, step=4, name="train", status="LIVE_TRAIN_PASS")
    snap = write_pipeline_status(layout, orchestrator_running=False)
    assert snap["schema"].startswith("nhc0801-pipeline-status")
    assert (layout.generation_root / "pipeline" / "pipeline_status.json").is_file()

    scanned = scan_generation_status(layout)
    train_step = next(s for s in scanned["steps"] if s["step"] == 4)
    assert train_step["status"] == "PASS"
    assert scanned["train_metrics"]

    text = render_status_text(
        nhc0801_root=tmp_path / "NHC0801",
        generation_id=layout.generation_id,
        color=False,
    )
    assert "NHC0801" in text
    assert "LIVE_TRAIN_PASS" in text or "PASS" in text


def test_duplicate_endpoint_rejected() -> None:
    with pytest.raises(AutoFillError, match="duplicate"):
        build_auto_fill_plan(
            idle_cpu_ids=list(range(30)),
            mem_available_bytes=100 * (1024**3),
            endpoint_queue=[("R", "cation"), ("R", "cation")],
        )
