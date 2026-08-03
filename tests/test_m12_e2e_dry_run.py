"""M12 end-to-end dry-run integration (no HPC / no live DFT / no real train).

Chains: teacher → D3 (injected projector) → weighted audit → train → shortlist
→ pre_screen. Synthetic fixtures only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from nhc_deprot.data.paths import TRAIN_ROOTS, VALIDATION_ROOTS
from nhc_deprot.data.weighted_dataset import audit_weighted_dataset
from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.checkpoint_shortlist import run_shortlist_campaign
from nhc_deprot.pipeline.d3_projection import DryRunD3Projector, run_d3_campaign
from nhc_deprot.pipeline.pre_screen import (
    CAMPAIGN_SCHEMA,
    SELECTION_AUTHORITY,
    CheckpointCandidate,
    SimulatedPreScreenEngine,
    load_teacher_references_for_batch,
    run_pre_screen_campaign,
)
from nhc_deprot.pipeline.teacher_runner import DryRunTeacherEngine, run_teacher_campaign
from nhc_deprot.pipeline.weighted_dataset_writer import (
    OUTPUT_MANIFEST_SCHEMA,
    assemble_weighted_dataset,
)
from nhc_deprot.resources.profiles import get_profile
from nhc_deprot.training.config import TrainingConfig
from nhc_deprot.training.multi_seed_trainer import run_multi_seed_training

BATCH_ID = "g001"
RUN_ID = "e1f1_mlp"
# Variable-length trajectory (not the historical hard-coded 2)
FRAMES_PER_ENDPOINT = 4
DRY_RUN_EPOCHS = 4
CHECKPOINT_INTERVAL = 2


def test_m12_e2e_teacher_d3_weighted_train_shortlist_pre_screen(tmp_path: Path) -> None:
    """Full dry-run chain under a local NHC0801 sandbox."""

    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    roots = list(TRAIN_ROOTS) + list(VALIDATION_ROOTS)
    profile = get_profile("single_27_physical_v1")

    # --- 1) teacher dry-run (variable-length frames) ---
    teacher = run_teacher_campaign(
        layout=layout,
        root_ids=roots,
        profile=profile,
        engine=DryRunTeacherEngine(frames_per_endpoint=FRAMES_PER_ENDPOINT),
        dry_run=True,
    )
    assert teacher.status == "DRY_RUN_COMPLETE"
    assert teacher.dry_run is True
    assert teacher.live_chemistry is False
    assert teacher.pool_progress["failed"] == 0
    assert (layout.teacher_dir / "campaign_receipt.json").is_file()

    sample_ep = layout.teacher_endpoint_dir(TRAIN_ROOTS[0], "cation")
    man = json.loads((sample_ep / "manifest.json").read_text(encoding="utf-8"))
    assert man["frame_count"] == FRAMES_PER_ENDPOINT
    assert man["evaluation_count"] == FRAMES_PER_ENDPOINT
    assert (sample_ep / f"frame_{FRAMES_PER_ENDPOINT - 1:04d}.json").is_file()
    term = json.loads(
        (sample_ep / f"frame_{FRAMES_PER_ENDPOINT - 1:04d}.json").read_text(
            encoding="utf-8"
        )
    )
    assert term["is_terminal"] is True

    # --- 2) D3 with explicitly injected projector (no real dftd3) ---
    d3 = run_d3_campaign(
        layout=layout,
        root_ids=roots,
        projector=DryRunD3Projector(),
        dry_run=True,
        overwrite=True,
    )
    assert d3["status"] == "DRY_RUN_D3_PASS"
    assert d3["d3_recomputation_performed"] is False
    assert d3["frame_count"] == len(roots) * 2 * FRAMES_PER_ENDPOINT
    assert (layout.d3_dir / "campaign_receipt.json").is_file()

    # --- 3) weighted dataset + public audit API ---
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

    audit = audit_weighted_dataset(
        layout.datasets_dir,
        expected_schema=OUTPUT_MANIFEST_SCHEMA,
    )
    assert audit.status == "PASS"
    assert audit.frame_count == d3["frame_count"]
    assert not audit.training_started

    # --- 4) multi-seed train dry-run under train_g001/runs/<run_id>/ ---
    cfg = TrainingConfig(
        seeds=(20260730, 20260731),
        epochs=120,
        checkpoint_interval_epochs=CHECKPOINT_INTERVAL,
        run_id=RUN_ID,
        batch_size=8,
    )
    train = run_multi_seed_training(
        layout=layout,
        config=cfg,
        dry_run=True,
        dry_run_epochs=DRY_RUN_EPOCHS,
        train_batch_id=BATCH_ID,
    )
    assert train["status"] == "DRY_RUN_TRAIN_PASS"
    assert train["final_model_selected"] is False
    assert train["quick_validation_may_select_final_model"] is False
    assert train["run_id"] == RUN_ID
    assert train["failed_seed_count"] == 0

    run_dir = layout.train_batch_run_dir(BATCH_ID, RUN_ID)
    assert run_dir.is_dir()
    assert (run_dir / "train_info.json").is_file()
    assert (run_dir / "train_result.json").is_file()
    info = json.loads((run_dir / "train_info.json").read_text(encoding="utf-8"))
    assert info.get("run_id") == RUN_ID
    assert info.get("train_config_digest")
    # no bare seed_* under train_g001/
    assert not layout.train_seed_dir(BATCH_ID, 20260730).exists()

    seed_dir = layout.train_run_seed_dir(BATCH_ID, RUN_ID, 20260730)
    assert (seed_dir / "seed_result.json").is_file()
    seed_receipt = json.loads((seed_dir / "seed_result.json").read_text(encoding="utf-8"))
    assert seed_receipt["status"] == "PASS"
    assert seed_receipt.get("shortlist_epochs")

    # --- 5) quick-val shortlist (train_dir = run dir; new layout has runs/<run_id>/) ---
    # checkpoint_shortlist resolves legacy train_g00N/seed_* by default; M8 products
    # live under train_g00N/runs/<run_id>/seed_* so pass train_dir explicitly.
    short = run_shortlist_campaign(
        layout=layout,
        train_dir=run_dir,
        train_batch_id=BATCH_ID,
        recompute=True,
    )
    assert short["status"] == "SHORTLIST_PASS"
    assert short["final_model_selected"] is False
    assert short["quick_validation_may_select_final_model"] is False
    assert short["scientific_validation_required_before_final_selection"] is True
    assert short["candidate_count"] >= 2
    assert (layout.sci_val_dir / "shortlist_campaign.json").is_file()

    # --- 6) zero-DFT pre_screen (injected fake GAU_LOOSE engine) ---
    refs = load_teacher_references_for_batch(layout, BATCH_ID, list(TRAIN_ROOTS[:1]))
    assert len(refs) == 2  # cation + neutral
    assert all(r.reference_frame_index == FRAMES_PER_ENDPOINT - 1 for r in refs)

    candidates: list[CheckpointCandidate] = []
    for seed_res in train["seed_results"]:
        seed = int(seed_res["seed"])
        for ckpt in seed_res["checkpoints"]:
            epoch = int(ckpt["epoch"])
            candidates.append(
                CheckpointCandidate(
                    checkpoint_id=f"{RUN_ID}_seed_{seed}_epoch_{epoch:04d}",
                    run_id=RUN_ID,
                    seed=seed,
                    epoch=epoch,
                    weight_path=ckpt.get("path"),
                )
            )
    assert candidates, "train dry-run must emit interval checkpoints"

    # Deterministic outcomes: earlier seed/epoch slightly better RMSD so order is stable
    outcomes: dict[str, Any] = {}
    for i, c in enumerate(candidates):
        outcomes[c.checkpoint_id] = {
            "atom0_dx": 0.01 * (i + 1),
            "steps": 5 + i,
            "converged": True,
        }
    # Force one hard-fail to prove it is excluded from shortlist
    fail_id = candidates[-1].checkpoint_id
    outcomes[fail_id] = {"converged": False, "atom0_dx": 0.0, "steps": 1}

    pre = run_pre_screen_campaign(
        layout=layout,
        batch_id=BATCH_ID,
        screen_id=RUN_ID,
        candidates=candidates,
        references=refs,
        engine=SimulatedPreScreenEngine(outcomes=outcomes),
        shortlist_count=2,
        write=True,
    )
    assert pre["schema"] == CAMPAIGN_SCHEMA
    assert pre["final_model_selected"] is False
    assert pre["selection_authority"] == SELECTION_AUTHORITY
    assert pre["energy_loss_used_for_ranking"] is False
    assert pre["status"] in {"PRE_SCREEN_PASS", "PRE_SCREEN_EMPTY_SHORTLIST"}
    assert fail_id not in pre["shortlist_checkpoint_ids"]
    assert len(pre["shortlist_checkpoint_ids"]) <= 2

    receipt_path = Path(pre["receipt_path"])
    assert receipt_path.is_file()
    assert f"pre_screen_{BATCH_ID}" in str(receipt_path)
    assert RUN_ID in receipt_path.parts
    loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
    assert loaded["final_model_selected"] is False
    assert loaded["selection_authority"] == SELECTION_AUTHORITY
    assert loaded["schema"] == CAMPAIGN_SCHEMA
