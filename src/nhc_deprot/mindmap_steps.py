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
            "nhc_deprot.data.paths",
            "nhc_deprot.data.development_split",
            "docs/extracted/v004/*_DEVELOPMENT_SPLIT*",
        ],
        "status": "partial",
        "notes": "5 development roots frozen in V004; bulk roots not frozen",
    },
    1: {
        "title": "Train/Val/Test by molecular_root",
        "modules": ["nhc_deprot.contracts.tvt_gates", "nhc_deprot.data.development_split"],
        "status": "gates_ready",
        "notes": "Sealed Final Test commitment only; do not open identities",
    },
    2: {
        "title": "Pure-PySCF teacher frames",
        "modules": [
            "nhc_deprot.data.teacher_frames",
            "nhc_deprot.data.weighted_dataset",
            "nhc_deprot.data.weight_policy",
            "nhc_deprot.contracts.parent_protocol",
            "server autofill_* training_data",
        ],
        "status": "reader_ready_pilot_data_on_server",
        "notes": (
            "Parameterized split/NPZ reader + weight audit ready; "
            "V004 pilot 235 frames on server (counts from evidence, not hardcode); "
            "scale teacher generator not ported"
        ),
    },
    3: {
        "title": "Epoch-0 baseline full route",
        "modules": ["nhc_deprot.pipeline.parent_handoff", "docs/contracts/GAU_LOOSE_V001.yaml"],
        "status": "writer_static_only",
        "notes": "V004 epoch0 writer static audit PASS; execution NOT_RUN",
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
