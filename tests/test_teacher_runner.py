"""Teacher runner dry-run tests (mindmap step 2; no live PySCF)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.paths import (
    LEGACY_PILOT_TRAIN_ROOTS,
    LEGACY_PILOT_VALIDATION_ROOTS,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.teacher_runner import (
    DryRunTeacherEngine,
    TeacherRunnerError,
    default_pilot_root_queue,
    plan_teacher_paths,
    run_teacher_campaign,
)
from nhc_deprot.resources.profiles import get_profile


def test_default_queue_is_train_plus_val_no_test() -> None:
    q = default_pilot_root_queue()
    # Active TVT may be resplit (150+16); queue = train ∪ val, no FT.
    assert list(q) == list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    assert len(q) == len(TRAIN_ROOTS) + len(VALIDATION_ROOTS)
    assert set(q).isdisjoint({"final_test", "test"})


def test_plan_teacher_paths(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    plan = plan_teacher_paths(layout, LEGACY_PILOT_TRAIN_ROOTS[:1])
    assert plan["mindmap_step"] == 2
    assert plan["roots"][0]["cation_dir"].endswith("cation")


def test_dry_run_campaign_writes_g001_tree(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(LEGACY_PILOT_TRAIN_ROOTS)  # 3 pilot roots for fast dry-run
    campaign = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=2),
        dry_run=True,
    )
    assert campaign.status == "DRY_RUN_COMPLETE"
    assert campaign.dry_run is True
    assert campaign.live_chemistry is False
    assert campaign.pool_progress["done"] == 3
    assert campaign.pool_progress["failed"] == 0

    for root_id in roots:
        for endpoint in ("cation", "neutral"):
            ep_dir = layout.teacher_endpoint_dir(root_id, endpoint)
            assert (ep_dir / "manifest.json").is_file()
            assert (ep_dir / "frame_0000.json").is_file()
            assert (ep_dir / "frame_0001.json").is_file()
            frame = json.loads((ep_dir / "frame_0000.json").read_text(encoding="utf-8"))
            assert frame["schema"] == "nhc0801-parent-level-training-frame-v1"
            assert frame["dry_run"] is True
            assert frame["parent_protocol_sha256"] == PROTOCOL_SHA256
            assert frame["lineage"]["single_point_only"] is False
        receipt = json.loads(
            (layout.teacher_root_dir(root_id) / "root_receipt.json").read_text(
                encoding="utf-8"
            )
        )
        assert receipt["status"] == "PASS"
        assert set(receipt["endpoints"]) == {"cation", "neutral"}

    camp_path = layout.teacher_dir / "campaign_receipt.json"
    assert camp_path.is_file()


def test_dry_run_dual_profile_two_slots(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(LEGACY_PILOT_TRAIN_ROOTS) + list(LEGACY_PILOT_VALIDATION_ROOTS)  # 5
    campaign = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("dual_14_13_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=1),
        dry_run=True,
    )
    assert campaign.status == "DRY_RUN_COMPLETE"
    assert campaign.pool_progress["done"] == 5


def test_live_without_auth_fails(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")

    class FakeLive:
        def run_endpoint(self, **kwargs):  # noqa: ANN003
            raise AssertionError("must not be called")

    with pytest.raises(TeacherRunnerError, match="teacher_pyscf_authorized"):
        run_teacher_campaign(
            layout=layout,
            root_ids=LEGACY_PILOT_TRAIN_ROOTS[:1],
            engine=FakeLive(),
            dry_run=False,
            teacher_pyscf_authorized=False,
            claim_pass=True,
            live_dispatch_enabled=True,
        )


def test_live_dry_engine_rejected(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    with pytest.raises(TeacherRunnerError, match="non-dry TeacherEngine"):
        run_teacher_campaign(
            layout=layout,
            root_ids=LEGACY_PILOT_TRAIN_ROOTS[:1],
            engine=DryRunTeacherEngine(),
            dry_run=False,
            teacher_pyscf_authorized=True,
            claim_pass=True,
            live_dispatch_enabled=True,
        )
