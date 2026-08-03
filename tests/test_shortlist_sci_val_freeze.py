"""Tests for mindmap steps 7–10 scaffolding (shortlist / sci-val dry / freeze)."""

from __future__ import annotations

import json
from pathlib import Path

from nhc_deprot.generation.layout import init_generation
from nhc_deprot.pipeline.checkpoint_shortlist import run_shortlist_campaign
from nhc_deprot.pipeline.epoch0_receipt_audit import audit_epoch0_receipts
from nhc_deprot.pipeline.freeze_package import build_freeze_package
from nhc_deprot.pipeline.sci_val_campaign import run_sci_val_campaign
from nhc_deprot.pipeline.epoch0_runner import run_epoch0_campaign


def _write_fake_seed(train_dir: Path, seed: int, losses: list[tuple[int, float]]) -> None:
    sd = train_dir / f"seed_{seed}"
    sd.mkdir(parents=True, exist_ok=True)
    checkpoints = [
        {
            "epoch": ep,
            "validation_weighted_loss": loss,
            "path": str(sd / f"epoch_{ep:04d}.meta.json"),
            "checkpoint_selection_permitted": False,
        }
        for ep, loss in losses
    ]
    # shortlist would be computed; omit to force recompute path optional
    receipt = {
        "schema": "nhc0801-train-seed-receipt-v1",
        "seed": seed,
        "status": "PASS",
        "checkpoints": checkpoints,
        "shortlist_epochs": [losses[0][0], losses[-1][0]],
        "final_model_selected": False,
    }
    (sd / "seed_receipt.json").write_text(json.dumps(receipt), encoding="utf-8")


def test_shortlist_sci_val_freeze_chain(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    _write_fake_seed(
        layout.train_dir,
        20260730,
        [(10, 5.0), (50, 4.0), (100, 4.5), (200, 4.2)],
    )
    _write_fake_seed(
        layout.train_dir,
        20260731,
        [(10, 5.1), (60, 3.9), (120, 4.4), (200, 4.1)],
    )

    short = run_shortlist_campaign(layout=layout, recompute=True)
    assert short["status"] == "SHORTLIST_PASS"
    assert short["candidate_count"] >= 4
    assert short["final_model_selected"] is False
    assert (layout.sci_val_dir / "shortlist_campaign.json").is_file()

    sci = run_sci_val_campaign(layout=layout, dry_run=True, max_candidates=4)
    assert sci["status"] == "DRY_RUN_SCI_VAL_PASS"
    assert sci["final_test_authorized"] is False
    assert sci["selection"]["outcome"] in {"VALIDATION_SELECTED", "VALIDATION_REJECTED"}
    assert (layout.sci_val_dir / "campaign_receipt.json").is_file()

    freeze = build_freeze_package(layout=layout, repo_root=tmp_path)
    assert freeze["status"] in {"PROVISIONAL", "FROZEN"}
    assert freeze["splits"]["final_test_identities_exposed"] is False
    assert freeze["final_test_ready"] is False
    assert (layout.freeze_dir / "freeze_manifest.json").is_file()


def test_epoch0_receipt_audit_after_dry_run(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    run_epoch0_campaign(layout=layout, dry_run=True)
    report = audit_epoch0_receipts(layout=layout)
    assert report["audit_pass"] is True
    assert report["status"] == "EPOCH0_RECEIPT_AUDIT_PASS"
    assert not report["missing_roots"]
