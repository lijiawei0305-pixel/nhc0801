"""Map mindmap.md steps to implemented modules and gaps.

This is the single routing table for agents working in NHC0801.
"""

from __future__ import annotations

from typing import Final

# step -> (implemented modules, status, notes)
MINDMAP_IMPLEMENTATION: Final = {
    0: {
        "title": "Freeze molecular roots",
        "modules": [
            "nhc_deprot.generation.layout",
            "nhc_deprot.data.paths",
            "nhc_deprot.data.development_split",
            "docs/evidence/pilot_day1/DEVELOPMENT_SPLIT.json",
        ],
        "status": "partial",
        "notes": "scope C: pilot 5 roots; generation tree nhc0801-g001",
    },
    1: {
        "title": "Train/Val/Test by molecular_root",
        "modules": ["nhc_deprot.contracts.tvt_gates", "nhc_deprot.data.development_split"],
        "status": "gates_ready",
        "notes": "Sealed Final Test commitment only; do not open identities",
    },
    2: {
        "title": "Pure-PySCF teacher frames + D3 residual + weighted NPZ",
        "modules": [
            "nhc_deprot.pipeline.teacher_runner",
            "nhc_deprot.pipeline.d3_projection",
            "nhc_deprot.pipeline.weighted_dataset_writer",
            "nhc_deprot.generation.layout",
            "nhc_deprot.resources.worker_pool",
            "nhc_deprot.data.weight_policy",
            "scripts/nhc0801_teacher_runner.py",
            "scripts/nhc0801_d3_weighted_dry_run.py",
        ],
        "status": "dry_run_teacher_d3_weighted_ready",
        "notes": (
            "dry-run: teacher→frozen D3 receipts→weighted NPZ under g001; "
            "no silent D3 recompute; live PySCF/D3 not wired"
        ),
    },
    3: {
        "title": "Epoch-0 baseline full route",
        "modules": [
            "nhc_deprot.pipeline.epoch0_runner",
            "nhc_deprot.pipeline.live_epoch0",
            "nhc_deprot.pipeline.epoch0_receipt_audit",
            "nhc_deprot.pipeline.scientific_validation",
            "nhc_deprot.pipeline.parent_handoff",
            "nhc_deprot.resources.profiles",
            "docs/contracts/GAU_LOOSE_V002.yaml",
            "scripts/nhc0801_epoch0_dry_run.py",
            "scripts/nhc0801_check_epoch0_receipts.py",
            "scripts/nhc0801_pyscf_parent_worker.py",
        ],
        "status": "live_running_or_dry_ready",
        "notes": (
            "dry-run ready; live parent uses wb97m-d3bj worker; "
            "after live finish MUST audit campaign_receipt + root receipts"
        ),
    },
    4: {
        "title": "Train AIMNet2 on Train frames",
        "modules": [
            "nhc_deprot.training.multi_seed_trainer",
            "nhc_deprot.training.live_aimnet2",
            "nhc_deprot.training.config",
            "nhc_deprot.training.weighted_loss",
            "nhc_deprot.training.trainer_adapter",
            "nhc_deprot.data.weighted_dataset",
            "scripts/nhc0801_train_dry_run.py",
            "scripts/nhc0801_live_orchestrate.py",
        ],
        "status": "live_train_pass_pilot",
        "notes": (
            "g001 live 3×200 PASS on pilot weighted NPZ; "
            "quick-val never final-selects"
        ),
    },
    5: {
        "title": "Multi-epoch checkpoints",
        "modules": [
            "nhc_deprot.training.multi_seed_trainer",
            "nhc_deprot.training.live_aimnet2.export_checkpoint",
            "nhc_deprot.contracts.tvt_gates.quick_checkpoint_shortlist",
        ],
        "status": "live_checkpoints_written",
        "notes": "all outcomes retained; live .pt currently last-epoch export per seed",
    },
    6: {
        "title": "Quick validation on stored frames",
        "modules": [
            "nhc_deprot.training.multi_seed_trainer",
            "nhc_deprot.training.weighted_loss.WeightedEvaluationAccumulator",
        ],
        "status": "wired_live_and_dry",
        "notes": "Must not select final model",
    },
    7: {
        "title": "Shortlist checkpoints",
        "modules": [
            "nhc_deprot.pipeline.checkpoint_shortlist",
            "nhc_deprot.contracts.tvt_gates.quick_checkpoint_shortlist",
            "nhc_deprot.training.multi_seed_trainer",
            "scripts/nhc0801_shortlist_from_train.py",
        ],
        "status": "campaign_aggregator_ready",
        "notes": "per-seed shortlist → sci_val/shortlist_campaign.json; not final select",
    },
    8: {
        "title": "Full scientific validation route",
        "modules": [
            "nhc_deprot.pipeline.sci_val_campaign",
            "nhc_deprot.pipeline.scientific_validation",
            "nhc_deprot.pipeline.parent_handoff",
            "nhc_deprot.pipeline.mindmap_orchestrator",
            "scripts/nhc0801_sci_val_dry_run.py",
        ],
        "status": "dry_run_campaign_ready_live_gated",
        "notes": (
            "Writer + campaign dry-run; live engines require scientific_validation_live"
        ),
    },
    9: {
        "title": "Validation selects final checkpoint",
        "modules": [
            "nhc_deprot.pipeline.sci_val_campaign",
            "nhc_deprot.pipeline.scientific_validation.select_after_scientific_validation",
            "nhc_deprot.contracts.tvt_gates.select_scientific_checkpoint",
            "docs/contracts/NUMERIC_CALIBRATION_V001.yaml",
        ],
        "status": "selection_wired_in_campaign",
        "notes": "Selection consumes CheckpointScientificValidation.selection_payload()",
    },
    10: {
        "title": "Freeze all identities",
        "modules": [
            "nhc_deprot.pipeline.freeze_package",
            "nhc_deprot.contracts.tvt_gates.final_test_readiness",
            "scripts/nhc0801_freeze_package.py",
        ],
        "status": "provisional_freeze_ready",
        "notes": "freeze_manifest PROVISIONAL until VALIDATION_SELECTED; FT still sealed",
    },
    11: {
        "title": "Final Test once",
        "modules": ["nhc_deprot.contracts.tvt_gates.final_test_readiness"],
        "status": "sealed",
        "notes": "Commitment 834f9739…; identities not loaded; not authorized",
    },
    12: {
        "title": "No post-Test selection",
        "modules": ["nhc_deprot.contracts.tvt_gates"],
        "status": "policy",
        "notes": "Fail closed if violated",
    },
}
