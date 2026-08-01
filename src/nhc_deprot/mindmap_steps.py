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
            "nhc_deprot.pipeline.scientific_validation",
            "nhc_deprot.pipeline.parent_handoff",
            "nhc_deprot.resources.profiles",
            "docs/contracts/GAU_LOOSE_V001.yaml",
            "scripts/nhc0801_epoch0_dry_run.py",
        ],
        "status": "dry_run_runner_ready",
        "notes": (
            "dry-run: pure ref + official _0 route on Validation roots under g001/epoch0; "
            "live requires epoch0_execution + real engines + claim"
        ),
    },
    4: {
        "title": "Train AIMNet2 on Train frames",
        "modules": [
            "nhc_deprot.training.weighted_loss",
            "nhc_deprot.training.trainer_adapter",
            "nhc_deprot.data.weighted_dataset",
        ],
        "status": "adapter_only",
        "notes": "No live training; residual D3 targets required; dataset reader ready",
    },
    5: {
        "title": "Multi-epoch checkpoints",
        "modules": [],
        "status": "missing",
        "notes": "Need new trainer loop; do not reuse historical finetune best-by-val-loss",
    },
    6: {
        "title": "Quick validation on stored frames",
        "modules": ["nhc_deprot.training.weighted_loss.WeightedEvaluationAccumulator"],
        "status": "loss_ready",
        "notes": "Must not select final model",
    },
    7: {
        "title": "Shortlist checkpoints",
        "modules": ["nhc_deprot.contracts.tvt_gates.quick_checkpoint_shortlist"],
        "status": "gates_ready",
        "notes": "",
    },
    8: {
        "title": "Full scientific validation route",
        "modules": [
            "nhc_deprot.pipeline.scientific_validation",
            "nhc_deprot.pipeline.parent_handoff",
            "nhc_deprot.pipeline.mindmap_orchestrator",
        ],
        "status": "writer_ready_live_gated",
        "notes": (
            "Writer implements GAU_LOOSE→handoff→parent GAU→label; "
            "live engines require scientific_validation_live; sim backends for tests"
        ),
    },
    9: {
        "title": "Validation selects final checkpoint",
        "modules": [
            "nhc_deprot.pipeline.scientific_validation.select_after_scientific_validation",
            "nhc_deprot.contracts.tvt_gates.select_scientific_checkpoint",
            "docs/contracts/NUMERIC_CALIBRATION_V001.yaml",
        ],
        "status": "gates_and_writer_ready",
        "notes": "Selection consumes CheckpointScientificValidation.selection_payload()",
    },
    10: {
        "title": "Freeze all identities",
        "modules": ["nhc_deprot.contracts.tvt_gates.final_test_readiness"],
        "status": "gates_ready",
        "notes": "Source commit not frozen in V004",
    },
    11: {
        "title": "Final Test once",
        "modules": ["nhc_deprot.contracts.tvt_gates.final_test_readiness"],
        "status": "sealed",
        "notes": "Commitment 834f9739…; identities not loaded",
    },
    12: {
        "title": "No post-Test selection",
        "modules": ["nhc_deprot.contracts.tvt_gates"],
        "status": "policy",
        "notes": "Fail closed if violated",
    },
}
