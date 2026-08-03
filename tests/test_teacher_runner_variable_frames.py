"""M3: teacher_runner variable-length frames (no live PySCF)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.teacher_runner import (
    DryRunTeacherEngine,
    TeacherRunnerError,
    frames_from_endpoint_manifest,
    run_root_teacher,
    run_teacher_campaign,
    select_trajectory_frame_indices,
)
from nhc_deprot.resources.profiles import get_profile


def test_select_trajectory_frame_indices_stride_one() -> None:
    assert select_trajectory_frame_indices(12, 1) == list(range(12))
    assert select_trajectory_frame_indices(1, 1) == [0]
    assert select_trajectory_frame_indices(2, 1) == [0, 1]


def test_select_trajectory_frame_indices_keeps_start_and_end() -> None:
    # 12 frames, stride 3 → 0,3,6,9 + always last 11
    assert select_trajectory_frame_indices(12, 3) == [0, 3, 6, 9, 11]
    # last already on grid
    assert select_trajectory_frame_indices(10, 3) == [0, 3, 6, 9]
    # stride >= n still keeps start and end
    assert select_trajectory_frame_indices(5, 10) == [0, 4]
    assert select_trajectory_frame_indices(1, 5) == [0]


def test_select_trajectory_frame_indices_rejects_bad_args() -> None:
    with pytest.raises(TeacherRunnerError, match="positive"):
        select_trajectory_frame_indices(0, 1)
    with pytest.raises(TeacherRunnerError, match="stride"):
        select_trajectory_frame_indices(5, 0)
    with pytest.raises(TeacherRunnerError, match="stride"):
        select_trajectory_frame_indices(5, -1)


def test_frames_from_endpoint_manifest_prefers_frames_list() -> None:
    manifest = {
        "frame_count": 99,  # must not hard-assume this over frames
        "frames": [
            {"frame_index": 0, "path": "frame_0000.json"},
            {"frame_index": 1, "path": "frame_0001.json"},
            {"frame_index": 2, "path": "frame_0002.json"},
        ],
    }
    frames = frames_from_endpoint_manifest(manifest)
    assert len(frames) == 3
    assert [f["frame_index"] for f in frames] == [0, 1, 2]
    assert frames[2]["path"] == "frame_0002.json"


def test_frames_from_endpoint_manifest_fallback_frame_count() -> None:
    manifest = {"frame_count": 4}
    frames = frames_from_endpoint_manifest(manifest)
    assert len(frames) == 4
    assert frames[0]["path"] == "frame_0000.json"
    assert frames[3]["path"] == "frame_0003.json"


def test_dry_run_twelve_frame_fixture(tmp_path: Path) -> None:
    """12-frame endpoint fixture: no implicit 2-frame assumption."""
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    root_id = TRAIN_ROOTS[0]
    engine = DryRunTeacherEngine(frames_per_endpoint=12, trajectory_stride=1)
    receipt = run_root_teacher(
        layout=layout,
        root_id=root_id,
        engine=engine,
        dry_run=True,
    )
    assert receipt.status == "PASS"
    for endpoint in ("cation", "neutral"):
        ep = receipt.endpoints[endpoint]
        assert ep["frame_count"] == 12
        assert len(ep["frame_paths"]) == 12

        ep_dir = layout.teacher_endpoint_dir(root_id, endpoint)
        manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
        frames_meta = frames_from_endpoint_manifest(manifest)
        assert len(frames_meta) == 12
        assert manifest["frame_count"] == 12
        assert len(manifest["frames"]) == 12

        terminal_count = 0
        for entry in frames_meta:
            path = ep_dir / entry["path"]
            assert path.is_file()
            frame = json.loads(path.read_text(encoding="utf-8"))
            assert frame["frame_index"] == entry["frame_index"]
            if frame["is_terminal"]:
                terminal_count += 1
        assert terminal_count == 1
        last = json.loads((ep_dir / "frame_0011.json").read_text(encoding="utf-8"))
        assert last["is_terminal"] is True
        assert last["frame_index"] == 11


def test_dry_run_trajectory_stride_subsamples_keep_ends(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    root_id = TRAIN_ROOTS[0]
    # full 12 evaluations, stride 3 → written frames: 5 (indices 0,3,6,9,11)
    engine = DryRunTeacherEngine(frames_per_endpoint=12, trajectory_stride=3)
    receipt = run_root_teacher(
        layout=layout,
        root_id=root_id,
        engine=engine,
        dry_run=True,
    )
    assert receipt.status == "PASS"
    ep = receipt.endpoints["cation"]
    assert ep["frame_count"] == 5
    assert len(ep["frame_paths"]) == 5

    ep_dir = layout.teacher_endpoint_dir(root_id, "cation")
    manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
    frames_meta = frames_from_endpoint_manifest(manifest)
    assert len(frames_meta) == 5
    assert manifest["frame_count"] == 5
    assert manifest.get("trajectory_stride") == 3
    assert manifest.get("evaluation_count") == 12

    # written files renumbered 0..4; optimizer_step keeps original cycle indices
    expected_steps = [0, 3, 6, 9, 11]
    for i, step in enumerate(expected_steps):
        frame = json.loads((ep_dir / f"frame_{i:04d}.json").read_text(encoding="utf-8"))
        assert frame["frame_index"] == i
        assert frame["optimizer_step"] == step
        assert frame["is_terminal"] is (i == len(expected_steps) - 1)
    # only renumbered frames on disk (no leftover dense indices beyond n_written)
    assert not (ep_dir / "frame_0005.json").is_file()
    assert not (ep_dir / "frame_0011.json").is_file()


def test_dry_run_stride_default_matches_legacy_two_frame(tmp_path: Path) -> None:
    """Default DryRunTeacherEngine still writes 2 frames (historical dry-run shape)."""
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    campaign = run_teacher_campaign(
        layout=layout,
        root_ids=TRAIN_ROOTS[:1],
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(),  # frames_per_endpoint=2, stride=1
        dry_run=True,
    )
    assert campaign.status == "DRY_RUN_COMPLETE"
    ep_dir = layout.teacher_endpoint_dir(TRAIN_ROOTS[0], "cation")
    manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["frame_count"] == 2
    assert len(frames_from_endpoint_manifest(manifest)) == 2
    assert (ep_dir / "frame_0000.json").is_file()
    assert (ep_dir / "frame_0001.json").is_file()
    assert not (ep_dir / "frame_0002.json").is_file()


def test_campaign_twelve_frames_per_endpoint(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(TRAIN_ROOTS[:2])
    campaign = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=12),
        dry_run=True,
    )
    assert campaign.status == "DRY_RUN_COMPLETE"
    assert campaign.pool_progress["done"] == 2
    for root_id in roots:
        for endpoint in ("cation", "neutral"):
            ep_dir = layout.teacher_endpoint_dir(root_id, endpoint)
            manifest = json.loads((ep_dir / "manifest.json").read_text(encoding="utf-8"))
            assert manifest["frame_count"] == 12
            assert len(manifest["frames"]) == 12
            for entry in frames_from_endpoint_manifest(manifest):
                assert (ep_dir / entry["path"]).is_file()
