"""D3 projection + weighted dataset dry-run tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.data.weighted_dataset import audit_weighted_dataset
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.d3_projection import run_d3_campaign
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import (
    OUTPUT_MANIFEST_SCHEMA,
    assemble_weighted_dataset,
)
from nhc_deprot.resources.profiles import get_profile


def _prepare_teacher(tmp_path: Path, roots: list[str], n_frames: int = 2):
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    camp = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=get_profile("single_27_physical_v1"),
        engine=DryRunTeacherEngine(frames_per_endpoint=n_frames),
        dry_run=True,
    )
    assert camp.status == "DRY_RUN_COMPLETE"
    return layout


def test_d3_then_weighted_full_pilot_queue(tmp_path: Path) -> None:
    roots = list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    layout = _prepare_teacher(tmp_path, roots, n_frames=2)

    d3 = run_d3_campaign(layout=layout, root_ids=roots, dry_run=True)
    assert d3["status"] == "DRY_RUN_D3_PASS"
    assert d3["frame_count"] == len(roots) * 2 * 2  # roots * endpoints * frames
    assert d3["d3_recomputation_performed"] is False
    assert (layout.d3_dir / "campaign_receipt.json").is_file()

    # one receipt path
    sample = (
        layout.d3_dir
        / TRAIN_ROOTS[0]
        / "cation"
        / "frame_0000.json"
    )
    assert sample.is_file()

    weighted = assemble_weighted_dataset(
        layout=layout,
        train_roots=list(TRAIN_ROOTS),
        validation_roots=list(VALIDATION_ROOTS),
        dry_run=True,
        overwrite=True,
        run_audit=True,
    )
    assert weighted["status"] == "DRY_RUN_WEIGHTED_DATASET_PASS"
    assert weighted["frame_count"] == d3["frame_count"]
    assert weighted["audit"]["status"] == "PASS"
    assert weighted["audit"]["split_weight_sums"]["train"] == pytest.approx(1.0)
    assert weighted["audit"]["split_weight_sums"]["validation"] == pytest.approx(1.0)

    # re-audit via public API
    audit = audit_weighted_dataset(
        layout.datasets_dir,
        expected_schema=OUTPUT_MANIFEST_SCHEMA,
    )
    assert audit.status == "PASS"
    assert audit.frame_count_by_split["train"] == 3 * 2 * 2  # 3 roots * 2 ep * 2 frames
    assert audit.frame_count_by_split["validation"] == 2 * 2 * 2


def test_weighted_requires_matching_d3(tmp_path: Path) -> None:
    layout = _prepare_teacher(tmp_path, list(TRAIN_ROOTS[:1]), n_frames=1)
    # no D3 yet
    with pytest.raises(Exception, match="missing D3|no teacher"):
        assemble_weighted_dataset(
            layout=layout,
            train_roots=list(TRAIN_ROOTS[:1]),
            validation_roots=list(VALIDATION_ROOTS[:1]),
            dry_run=True,
        )


def test_d3_live_refused(tmp_path: Path) -> None:
    layout = _prepare_teacher(tmp_path, list(TRAIN_ROOTS[:1]), n_frames=1)
    with pytest.raises(Exception, match="live D3|not authorized"):
        run_d3_campaign(layout=layout, root_ids=list(TRAIN_ROOTS[:1]), dry_run=False)
