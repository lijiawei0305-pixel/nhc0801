"""Mindmap step 7 — aggregate per-seed quick shortlists into a campaign receipt.

Quick-val frame loss only screens candidates. Never final-selects a model.
Final selection remains mindmap steps 8–9 (scientific Validation).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.tvt_gates import quick_checkpoint_shortlist
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.training.config import TrainingConfig

SHORTLIST_CAMPAIGN_SCHEMA: Final = "nhc0801-checkpoint-shortlist-campaign-v1"
MINDMAP_STEP: Final = 7


class ShortlistError(RuntimeError):
    """Shortlist aggregation failed closed."""


def _loss_from_checkpoint(ckpt: dict[str, Any]) -> float:
    if "validation_weighted_loss" in ckpt and ckpt["validation_weighted_loss"] is not None:
        return float(ckpt["validation_weighted_loss"])
    qv = ckpt.get("quick_validation") or {}
    if isinstance(qv, dict) and qv.get("validation_weighted_loss") is not None:
        return float(qv["validation_weighted_loss"])
    raise ShortlistError(f"checkpoint missing validation_weighted_loss: epoch={ckpt.get('epoch')}")


def load_seed_receipt(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ShortlistError(f"missing seed receipt: {path}")
    payload, _raw = load_json_object(path)
    return payload


def shortlist_for_seed(
    seed_receipt: dict[str, Any],
    *,
    maximum_count: int,
    recompute: bool = False,
) -> dict[str, Any]:
    """Return shortlist epochs for one seed (recompute or trust receipt)."""

    seed = seed_receipt.get("seed")
    if type(seed) is not int:
        raise ShortlistError("seed receipt missing int seed")
    checkpoints = seed_receipt.get("checkpoints") or []
    if not isinstance(checkpoints, list) or not checkpoints:
        # fall back to epoch_logs if checkpoints empty
        logs = seed_receipt.get("epoch_logs") or []
        checkpoints = []
        for row in logs:
            if not isinstance(row, dict):
                continue
            epoch = row.get("epoch")
            qv = row.get("quick_validation") or {}
            if type(epoch) is not int:
                continue
            if not isinstance(qv, dict) or qv.get("validation_weighted_loss") is None:
                continue
            # keep only interval-like entries if dense logs exist
            checkpoints.append(
                {
                    "epoch": epoch,
                    "validation_weighted_loss": qv["validation_weighted_loss"],
                }
            )
        # downsample dense logs to interval anchors if too many
        if len(checkpoints) > 40:
            # keep every 10th + last
            kept = [c for c in checkpoints if int(c["epoch"]) % 10 == 0]
            if checkpoints and checkpoints[-1] not in kept:
                kept.append(checkpoints[-1])
            checkpoints = kept

    if not checkpoints:
        raise ShortlistError(f"seed {seed}: no checkpoints or epoch_logs for shortlist")

    existing = seed_receipt.get("shortlist_epochs")
    if (
        not recompute
        and isinstance(existing, list)
        and existing
        and all(type(x) is int for x in existing)
    ):
        epochs = tuple(int(x) for x in existing)
        source = "seed_receipt.shortlist_epochs"
    else:
        shortlist_input = [
            {
                "epoch": int(c["epoch"]),
                "validation_weighted_loss": _loss_from_checkpoint(c),
            }
            for c in checkpoints
            if type(c.get("epoch")) is int
        ]
        # dedupe by epoch (keep last)
        by_epoch: dict[int, dict[str, Any]] = {}
        for row in shortlist_input:
            by_epoch[int(row["epoch"])] = row
        epochs = quick_checkpoint_shortlist(
            list(by_epoch.values()), maximum_count=maximum_count
        )
        source = "recomputed_quick_checkpoint_shortlist"

    # attach loss + meta path when available
    by_epoch_full: dict[int, dict[str, Any]] = {}
    for c in checkpoints:
        if type(c.get("epoch")) is int:
            by_epoch_full[int(c["epoch"])] = c

    candidates: list[dict[str, Any]] = []
    for epoch in epochs:
        ck = by_epoch_full.get(epoch, {})
        loss = None
        try:
            loss = _loss_from_checkpoint(ck) if ck else None
        except ShortlistError:
            loss = None
        candidates.append(
            {
                "seed": seed,
                "epoch": epoch,
                "validation_weighted_loss": loss,
                "meta_path": ck.get("path"),
                "checkpoint_selection_permitted": False,
                "quick_validation_may_select_final_model": False,
            }
        )

    return {
        "seed": seed,
        "status": seed_receipt.get("status"),
        "shortlist_epochs": list(epochs),
        "shortlist_source": source,
        "candidates": candidates,
        "final_model_selected": False,
        "selection_authority": "quick_validation_shortlist_only_not_final",
    }


def run_shortlist_campaign(
    *,
    layout: GenerationLayout,
    maximum_count_per_seed: int | None = None,
    recompute: bool = False,
    train_dir: Path | None = None,
) -> dict[str, Any]:
    """Aggregate all seed shortlists under g001/train → g001/sci_val/shortlist_campaign.json."""

    cfg = TrainingConfig()
    max_n = maximum_count_per_seed or cfg.quick_checkpoint_maximum_count_per_seed
    tdir = train_dir or layout.train_dir
    if not tdir.is_dir():
        raise ShortlistError(f"train dir missing: {tdir}")

    seed_dirs = sorted(p for p in tdir.glob("seed_*") if p.is_dir())
    if not seed_dirs:
        raise ShortlistError(f"no seed_* directories under {tdir}")

    per_seed: list[dict[str, Any]] = []
    all_candidates: list[dict[str, Any]] = []
    for sd in seed_dirs:
        receipt_path = sd / "seed_receipt.json"
        rec = load_seed_receipt(receipt_path)
        one = shortlist_for_seed(rec, maximum_count=max_n, recompute=recompute)
        # attach weight path if present for shortlist epochs
        for cand in one["candidates"]:
            ep = int(cand["epoch"])
            pt = sd / f"epoch_{ep:04d}.pt"
            cand["weight_path"] = str(pt) if pt.is_file() else None
            cand["weight_present"] = pt.is_file()
            all_candidates.append(cand)
        per_seed.append(one)

    campaign = {
        "schema": SHORTLIST_CAMPAIGN_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "status": "SHORTLIST_PASS",
        "maximum_count_per_seed": max_n,
        "seed_count": len(per_seed),
        "per_seed": per_seed,
        "candidates": all_candidates,
        "candidate_count": len(all_candidates),
        "weights_present_count": sum(1 for c in all_candidates if c.get("weight_present")),
        "final_model_selected": False,
        "quick_validation_may_select_final_model": False,
        "scientific_validation_required_before_final_selection": True,
        "final_test_payload_read": False,
        "notes": [
            "quick-val frame loss shortlist only",
            "weight_present false means meta-only; re-export .pt before live sci-val",
            "never opens Final Test",
        ],
    }

    layout.sci_val_dir.mkdir(parents=True, exist_ok=True)
    out = layout.sci_val_dir / "shortlist_campaign.json"
    write_json(out, campaign, overwrite=True)
    write_json(layout.logs_dir / "shortlist_campaign.json", campaign, overwrite=True)
    campaign["receipt_path"] = str(out)
    return campaign
