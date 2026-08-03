"""Mindmap step 10 — freeze package writer (identities, protocols, selection).

Writes under generation ``freeze/``. Never opens Final Test identities.
A package may be PROVISIONAL when selection is not yet final (e.g. waiting
for live epoch-0 / live sci-val).
"""

from __future__ import annotations

import hashlib
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Mapping

from nhc_deprot.contracts.parent_protocol import BASIS, FUNCTIONAL, PROTOCOL_ID, PROTOCOL_SHA256
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.data.paths import (
    OFFICIAL_AIMNET2_WEIGHT_SHA256,
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    SEALED_FINAL_TEST_ROOT_COUNT,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)
from nhc_deprot.generation.layout import GenerationLayout
from nhc_deprot.pipeline.parent_handoff import load_gau_loose_profile

FREEZE_SCHEMA: Final = "nhc0801-freeze-package-v1"
MINDMAP_STEP: Final = 10


class FreezeError(RuntimeError):
    """Freeze package construction failed."""


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head(repo: Path) -> dict[str, Any]:
    try:
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
        )
        dirty = subprocess.run(
            ["git", "-C", str(repo), "status", "--porcelain"],
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            return {"available": False, "reason": "no_git_head"}
        return {
            "available": True,
            "commit": head.stdout.strip(),
            "dirty": bool(dirty.stdout.strip()),
            "status_porcelain_nonempty": bool(dirty.stdout.strip()),
        }
    except OSError as exc:
        return {"available": False, "reason": str(exc)}


def build_freeze_package(
    *,
    layout: GenerationLayout,
    repo_root: Path | None = None,
    selection_receipt: Mapping[str, Any] | None = None,
    require_selection: bool = False,
    extra: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Assemble freeze manifest for mindmap step 10."""

    gau = load_gau_loose_profile()
    from nhc_deprot.pipeline.training_blockers import load_numeric_calibration

    numeric = load_numeric_calibration()
    validate_ok = numeric.get("status") == "FROZEN"

    sel = dict(selection_receipt) if selection_receipt else None
    sel_path = layout.sci_val_dir / "selection_receipt.json"
    if sel is None and sel_path.is_file():
        sel, _raw = load_json_object(sel_path)

    if require_selection and (
        not sel or sel.get("outcome") != "VALIDATION_SELECTED"
    ):
        raise FreezeError("require_selection but no VALIDATION_SELECTED receipt")

    selected = bool(sel and sel.get("outcome") == "VALIDATION_SELECTED")
    # Dry-run sci-val selection must not hard-freeze the model for production
    sci_camp_path = layout.sci_val_dir / "campaign_receipt.json"
    sci_was_dry = False
    if sci_camp_path.is_file():
        sci_camp, _ = load_json_object(sci_camp_path)
        sci_was_dry = bool(sci_camp.get("dry_run"))
    status = (
        "FROZEN"
        if selected and validate_ok and not sci_was_dry
        else "PROVISIONAL"
    )

    # optional artifacts
    artifacts: dict[str, Any] = {}
    for label, path in (
        ("shortlist_campaign", layout.sci_val_dir / "shortlist_campaign.json"),
        ("sci_val_campaign", layout.sci_val_dir / "campaign_receipt.json"),
        ("epoch0_campaign", layout.epoch0_dir / "campaign_receipt.json"),
        ("train_campaign_live", layout.train_dir / "campaign_receipt_live.json"),
        ("generation_meta", layout.generation_meta_path()),
    ):
        if path.is_file():
            artifacts[label] = {
                "path": str(path),
                "sha256": _sha256_file(path),
                "bytes": path.stat().st_size,
            }

    repo = repo_root or Path.cwd()
    git_info = _git_head(repo)

    package = {
        "schema": FREEZE_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "generation_id": layout.generation_id,
        "status": status,
        "created_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "splits": {
            "train_roots": list(TRAIN_ROOTS),
            "validation_roots": list(VALIDATION_ROOTS),
            "sealed_final_test_commitment_sha256": SEALED_FINAL_TEST_COMMITMENT_SHA256,
            "sealed_final_test_root_count": SEALED_FINAL_TEST_ROOT_COUNT,
            "final_test_identities_exposed": False,
            "final_test_payload_read": False,
        },
        "parent_protocol": {
            "id": PROTOCOL_ID,
            "sha256": PROTOCOL_SHA256,
            "functional": FUNCTIONAL,
            "basis": BASIS,
        },
        "gau_loose": {
            "profile": "GAU_LOOSE",
            "ase_fmax_ev_angstrom": gau.ase_fmax_ev_angstrom,
            "maximum_steps": gau.maximum_steps,
            "energy_change_eh": gau.energy_change_eh,
            "gradient_rms_eh_bohr": gau.gradient_rms_eh_bohr,
            "gradient_max_eh_bohr": gau.gradient_max_eh_bohr,
        },
        "epoch0_weight": {
            "official_sha256": OFFICIAL_AIMNET2_WEIGHT_SHA256,
            "member": "aimnet2_wb97m_d3_0",
        },
        "numeric_calibration": {
            "version": numeric.get("version"),
            "status": numeric.get("status"),
            "label_error_tolerance_kcal_mol": numeric.get("label_error_tolerance_kcal_mol"),
            "chosen_before_final_test": numeric.get("chosen_before_final_test"),
        },
        "selection": sel
        or {
            "outcome": "NOT_SELECTED_YET",
            "selected_epoch": None,
            "test_authorized": False,
        },
        "model_state": (
            "MODEL_FROZEN"
            if selected and not sci_was_dry
            else ("MODEL_PROVISIONAL_DRY_SELECTION" if selected and sci_was_dry else "MODEL_NOT_FROZEN")
        ),
        "numeric_addendum_state": "FROZEN" if validate_ok else "NOT_FROZEN",
        "sci_val_selection_was_dry_run": sci_was_dry,
        "source_code": git_info,
        "artifacts": artifacts,
        "final_test_ready": False,
        "final_test_gate_notes": [
            "Final Test requires independent unopened audit + full freeze",
            "PROVISIONAL freeze must not open Final Test",
            "post-Test reselection forbidden (mindmap 12)",
        ],
        "extra": dict(extra or {}),
    }

    layout.freeze_dir.mkdir(parents=True, exist_ok=True)
    out = layout.freeze_dir / "freeze_manifest.json"
    write_json(out, package, overwrite=True)
    write_json(layout.logs_dir / "freeze_manifest.json", package, overwrite=True)
    package["receipt_path"] = str(out)
    return package
