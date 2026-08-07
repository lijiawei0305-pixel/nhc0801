"""Tests for Phase-A eligible full-pair inventory."""

from __future__ import annotations

import json
from pathlib import Path

from nhc_deprot.pipeline.eligible_full_pairs import (
    build_inventory,
    classify_root,
    write_inventory,
)


def _fake_endpoint(dir_: Path, *, done: bool, frames: int = 3) -> None:
    dir_.mkdir(parents=True, exist_ok=True)
    for i in range(frames):
        (dir_ / f"frame_{i:04d}.json").write_text(
            json.dumps(
                {
                    "schema": "nhc0801-parent-level-training-frame-v1",
                    "frame_index": i,
                    "functional": "wb97m-d3bj",
                    "basis": "def2-TZVPP",
                    "is_terminal": i == frames - 1,
                    "gradient_hartree_per_bohr": [[1e-5, 0.0, 0.0]],
                }
            ),
            encoding="utf-8",
        )
    if done:
        (dir_ / "manifest.json").write_text(
            json.dumps(
                {
                    "live_chemistry": True,
                    "dry_run": False,
                    "complete_geometry_optimization": True,
                    "frame_count": frames,
                    "final_grad_max_eh_bohr": 1e-5,
                    "final_grad_rms_eh_bohr": 1e-5,
                }
            ),
            encoding="utf-8",
        )


def test_classify_and_inventory_counts(tmp_path: Path) -> None:
    gen = tmp_path / "runs" / "nhc0801-g001"
    t = gen / "teacher_gpu_g010"
    train_r = "TRAINROOTAAAAAAAAA-UHFFFAOYSA-N"
    val_r = "VALROOTBBBBBBBBBBB-UHFFFAOYSA-N"
    pool_r = "POOLROOTCCCCCCCCC-UHFFFAOYSA-N"
    half_r = "HALFROOTDDDDDDDDDD-UHFFFAOYSA-N"

    for rid, _both in ((train_r, True), (val_r, True), (pool_r, True)):
        _fake_endpoint(t / rid / "cation", done=True)
        _fake_endpoint(t / rid / "neutral", done=True)
    _fake_endpoint(t / half_r / "cation", done=True)
    _fake_endpoint(t / half_r / "neutral", done=False, frames=2)

    inv = build_inventory(
        gen,
        train_roots=[train_r],
        val_roots=[val_r],
        target_train_roots=2,
        target_val_roots=1,
    )
    assert inv["counts"]["n_full_pairs"] == 3
    # pool + train eligible; val excluded from expanded train
    assert inv["counts"]["n_eligible_for_expanded_train"] == 2
    assert inv["counts"]["n_incomplete"] == 1
    assert inv["counts"]["train_lock_ready"] is True
    assert inv["counts"]["gap_to_train_lock_150"] == 0

    elig_ids = {r["root_id"] for r in inv["eligible_full_pairs"]}
    assert pool_r in elig_ids
    assert train_r in elig_ids
    assert val_r not in elig_ids

    out = write_inventory(inv, gen / "logs" / "eligible_full_pairs" / "eligible_full_pairs.json")
    assert out.is_file()
    assert out.with_name("eligible_full_pairs_status.json").is_file()


def test_val_not_eligible_for_train() -> None:
    slot = {
        "cation": {"done_ok": True, "frame_count": 5, "product_dirs": []},
        "neutral": {"done_ok": True, "frame_count": 5, "product_dirs": []},
    }
    row = classify_root(
        "VAL-X",
        slot,
        train_roots=[],
        val_roots=["VAL-X"],
    )
    assert row["in_val"] is True
    assert row["full_pair_pass"] is True
    assert row["eligible_for_expanded_train"] is False
    assert row["excluded_test"] is False
