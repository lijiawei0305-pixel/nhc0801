"""Ported for NHC0801 / nhc-deprot.

Source: nhc-deprot-ranker-science-pilot (agent/phase9b-science-pilot, dirty V004 worktree).
Authority for science: /Users/cc/nhc-deprot/mindmap.md first; V004 contracts second.
Do not import production two_endpoint B3LYP/def2-SVP or fmax=0.05 preopt as parent protocol.

Train/Validation/Final-Test policy gates (chemistry-free)."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, cast

SPLITS: Final = ("train", "validation", "final_test")
ROOT_IDENTITY_FIELDS: Final = (
    "root_id",
    "canonical_identity",
    "scaffold_id",
    "lineage_root_id",
    "near_duplicate_cluster_id",
)
STRUCTURE_FIELDS: Final = ("cation_sha256", "neutral_sha256")
FINAL_TEST_GATE_FIELDS: Final = (
    "model_checkpoint_sha256",
    "train_split_sha256",
    "validation_split_sha256",
    "final_test_commitment_sha256",
    "numeric_addendum_sha256",
    "source_commit",
    "runtime_sha256",
)


class TVTContractError(RuntimeError):
    """A TVT identity or gate is malformed or unavailable."""


def canonical_json(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _sha256(value: object, *, label: str) -> str:
    if not isinstance(value, str) or len(value) != 64:
        raise TVTContractError(f"{label} is not a SHA256")
    try:
        int(value, 16)
    except ValueError as exc:
        raise TVTContractError(f"{label} is not lowercase hexadecimal") from exc
    if value != value.lower():
        raise TVTContractError(f"{label} is not lowercase hexadecimal")
    return value


def _finite(value: object, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TVTContractError(f"{label} is not numeric")
    result = float(value)
    if not math.isfinite(result):
        raise TVTContractError(f"{label} is not finite")
    return result


def audit_split_registry(payload: Mapping[str, object]) -> dict[str, object]:
    """Audit molecular-root separation across every registered identity dimension."""

    missing: list[str] = []
    owners: dict[str, dict[str, str]] = {
        field: {} for field in (*ROOT_IDENTITY_FIELDS, *STRUCTURE_FIELDS)
    }
    counts: dict[str, int] = {}
    overlaps: list[dict[str, str]] = []
    for split in SPLITS:
        profiles = payload.get(split)
        if not isinstance(profiles, list) or not profiles:
            missing.append(f"{split}:non_empty_profiles")
            counts[split] = 0
            continue
        counts[split] = len(profiles)
        for index, value in enumerate(profiles):
            if not isinstance(value, dict):
                missing.append(f"{split}[{index}]:profile_object")
                continue
            profile = cast(dict[str, object], value)
            candidate = profile.get("candidate")
            if not isinstance(candidate, str) or not candidate:
                missing.append(f"{split}[{index}]:candidate")
                continue
            for field in ROOT_IDENTITY_FIELDS:
                identity = profile.get(field)
                if not isinstance(identity, str) or not identity:
                    missing.append(f"{candidate}:{field}")
                    continue
                owner = owners[field].setdefault(identity, split)
                if owner != split:
                    overlaps.append(
                        {"dimension": field, "identity": identity, "left": owner, "right": split}
                    )
            for field in STRUCTURE_FIELDS:
                try:
                    identity = _sha256(profile.get(field), label=f"{candidate}:{field}")
                except TVTContractError:
                    missing.append(f"{candidate}:{field}")
                    continue
                owner = owners[field].setdefault(identity, split)
                if owner != split:
                    overlaps.append(
                        {"dimension": field, "identity": identity, "left": owner, "right": split}
                    )
            if profile.get("root_id") not in {None, candidate}:
                missing.append(f"{candidate}:root_id_must_equal_candidate")
    status = "PASS" if not missing and not overlaps else "BLOCKED"
    return {
        "schema": "phase9b-aimnet2-root-split-audit-v1",
        "status": status,
        "split_unit": "molecular_root",
        "counts": counts,
        "missing": sorted(set(missing)),
        "overlaps": overlaps,
        "frame_level_random_split": False,
    }


def audit_development_projection(payload: Mapping[str, object]) -> dict[str, object]:
    """Require a trainer-visible registry containing no final-test identity."""

    forbidden = {
        "final_test",
        "final_test_candidates",
        "final_test_paths",
        "final_test_receipts",
        "final_test_structure_sha256",
    }
    exposed = sorted(forbidden.intersection(payload))
    commitment = payload.get("sealed_final_test_commitment")
    commitment_valid = isinstance(commitment, dict) and set(commitment) == {
        "sha256",
        "root_count",
    }
    if commitment_valid:
        committed = cast(dict[str, object], commitment)
        try:
            _sha256(committed.get("sha256"), label="final-test commitment")
        except TVTContractError:
            commitment_valid = False
        root_count = committed.get("root_count")
        commitment_valid = commitment_valid and type(root_count) is int and root_count > 0
    return {
        "schema": "phase9b-aimnet2-development-projection-audit-v1",
        "status": "PASS" if not exposed and commitment_valid else "BLOCKED",
        "exposed_final_test_fields": exposed,
        "sealed_commitment_valid": commitment_valid,
    }


def validate_numeric_addendum(payload: Mapping[str, object]) -> dict[str, object]:
    required = {
        "label_error_tolerance_kcal_mol",
        "signed_bias_tolerance_kcal_mol",
        "catastrophic_failure_definition",
        "allowed_validation_failure_count",
        "epoch_zero_non_regression_rule",
        "minimum_pyscf_burden_reduction_fraction",
        "quick_checkpoint_shortlist_rule",
        "scientific_checkpoint_selection_rule",
        "ratification",
    }
    if payload.get("schema") != "phase9b-aimnet2-numeric-addendum-v1":
        raise TVTContractError("numeric addendum schema mismatch")
    if payload.get("status") != "FROZEN" or payload.get("chosen_before_final_test") is not True:
        raise TVTContractError("numeric addendum is not frozen before final-test")
    missing = sorted(required.difference(payload))
    if missing:
        raise TVTContractError(f"numeric addendum fields are missing: {missing}")
    label_tolerance = _finite(
        payload["label_error_tolerance_kcal_mol"], label="label error tolerance"
    )
    bias_tolerance = _finite(
        payload["signed_bias_tolerance_kcal_mol"], label="signed bias tolerance"
    )
    burden = _finite(
        payload["minimum_pyscf_burden_reduction_fraction"],
        label="minimum PySCF burden reduction",
    )
    failures = payload["allowed_validation_failure_count"]
    if label_tolerance <= 0 or bias_tolerance < 0 or not 0 <= burden <= 1:
        raise TVTContractError("numeric addendum bounds are invalid")
    if type(failures) is not int or failures < 0:
        raise TVTContractError("allowed validation failure count is invalid")
    if not isinstance(payload["quick_checkpoint_shortlist_rule"], dict):
        raise TVTContractError("quick checkpoint shortlist rule is invalid")
    selection = payload["scientific_checkpoint_selection_rule"]
    if not isinstance(selection, list) or not selection:
        raise TVTContractError("scientific checkpoint selection rule is invalid")
    if not isinstance(payload["ratification"], dict) or not payload["ratification"]:
        raise TVTContractError("numeric addendum ratification is invalid")
    return dict(payload)


def quick_checkpoint_shortlist(
    checkpoints: Sequence[Mapping[str, object]], *, maximum_count: int
) -> tuple[int, ...]:
    """Deterministically shortlist by frame loss without making final selection."""

    if maximum_count <= 0:
        raise TVTContractError("quick shortlist maximum count must be positive")
    parsed: list[tuple[int, float]] = []
    for checkpoint in checkpoints:
        epoch = checkpoint.get("epoch")
        if type(epoch) is not int or epoch <= 0:
            raise TVTContractError("checkpoint epoch is invalid")
        loss = _finite(checkpoint.get("validation_weighted_loss"), label="validation loss")
        parsed.append((epoch, loss))
    if not parsed:
        raise TVTContractError("no checkpoints are available")
    unique = {epoch for epoch, _ in parsed}
    if len(unique) != len(parsed):
        raise TVTContractError("checkpoint epochs are duplicated")
    ordered_by_loss = sorted(parsed, key=lambda item: (item[1], item[0]))
    selected: list[int] = [ordered_by_loss[0][0]]
    chronological = sorted(parsed)
    for epoch in (
        chronological[0][0],
        chronological[len(chronological) // 2][0],
        chronological[-1][0],
    ):
        if epoch not in selected and len(selected) < maximum_count:
            selected.append(epoch)
    for epoch, _ in ordered_by_loss:
        if epoch not in selected and len(selected) < maximum_count:
            selected.append(epoch)
    return tuple(selected)


def select_scientific_checkpoint(
    candidates: Sequence[Mapping[str, object]], *, numeric_addendum: Mapping[str, object]
) -> dict[str, object]:
    """Select only after complete scientific validation; frame loss is not a selector."""

    frozen = validate_numeric_addendum(numeric_addendum)
    label_limit = _finite(frozen["label_error_tolerance_kcal_mol"], label="label error tolerance")
    burden_limit = _finite(
        frozen["minimum_pyscf_burden_reduction_fraction"],
        label="minimum PySCF burden reduction",
    )
    allowed_failures = frozen["allowed_validation_failure_count"]
    if type(allowed_failures) is not int:
        raise TVTContractError("allowed validation failure count is invalid")
    eligible: list[tuple[tuple[float, ...], Mapping[str, object]]] = []
    rejected: list[dict[str, object]] = []
    for candidate in candidates:
        epoch = candidate.get("epoch")
        reasons: list[str] = []
        if type(epoch) is not int or epoch <= 0:
            raise TVTContractError("scientific-validation epoch is invalid")
        if candidate.get("all_identity_and_structure_hard_gates") is not True:
            reasons.append("STRUCTURE_OR_IDENTITY_GATE_FAILED")
        failures = candidate.get("catastrophic_failure_count")
        if type(failures) is not int or failures > allowed_failures:
            reasons.append("CATASTROPHIC_FAILURE_BUDGET_EXCEEDED")
        label_error = abs(
            _finite(candidate.get("maximum_absolute_label_error_kcal_mol"), label="label error")
        )
        if label_error > label_limit:
            reasons.append("SIGNED_LABEL_ERROR_FAILED")
        if candidate.get("critical_endpoint_non_regression_vs_epoch_zero") is not True:
            reasons.append("EPOCH_ZERO_REGRESSION")
        parent_gradient = _finite(
            candidate.get("parent_gradient_reduction_fraction"), label="parent gradient reduction"
        )
        burden = _finite(
            candidate.get("pyscf_geometry_work_reduction_fraction"),
            label="PySCF geometry work reduction",
        )
        cycles = _finite(
            candidate.get("cumulative_scf_cycle_reduction_fraction"),
            label="SCF cycle reduction",
        )
        wall = _finite(candidate.get("end_to_end_wall_reduction_fraction"), label="wall reduction")
        if burden < burden_limit:
            reasons.append("PYSCF_BURDEN_REDUCTION_FAILED")
        if reasons:
            rejected.append({"epoch": epoch, "reason_codes": reasons})
            continue
        key = (label_error, -parent_gradient, -burden, -cycles, -wall, float(epoch))
        eligible.append((key, candidate))
    if not eligible:
        return {
            "outcome": "VALIDATION_REJECTED",
            "selected_epoch": None,
            "rejected": rejected,
            "test_authorized": False,
        }
    _, selected = min(eligible, key=lambda item: item[0])
    return {
        "outcome": "VALIDATION_SELECTED",
        "selected_epoch": selected["epoch"],
        "selected_checkpoint_sha256": _sha256(
            selected.get("checkpoint_sha256"), label="selected checkpoint"
        ),
        "rejected": rejected,
        "test_authorized": False,
    }


def final_test_readiness(payload: Mapping[str, object]) -> dict[str, object]:
    """Authorize one payload read only after freeze and independent unopened proof."""

    missing: list[str] = []
    for field in FINAL_TEST_GATE_FIELDS:
        try:
            _sha256(payload.get(field), label=field)
        except TVTContractError:
            missing.append(field)
    if payload.get("model_state") != "MODEL_FROZEN":
        missing.append("model_state=MODEL_FROZEN")
    if payload.get("numeric_addendum_state") != "FROZEN":
        missing.append("numeric_addendum_state=FROZEN")
    if payload.get("unopened_audit_outcome") != "PASS":
        missing.append("unopened_audit_outcome=PASS")
    try:
        _sha256(payload.get("unopened_audit_receipt_sha256"), label="unopened audit receipt")
    except TVTContractError:
        missing.append("unopened_audit_receipt_sha256")
    if payload.get("previous_consumption_claim_exists") is not False:
        missing.append("previous_consumption_claim_exists=false")
    return {
        "outcome": "FINAL_TEST_READY_ONCE" if not missing else "FINAL_TEST_BLOCKED",
        "missing": sorted(set(missing)),
        "one_time": True,
        "retry": False,
        "checkpoint_selection_allowed": False,
        "threshold_change_allowed": False,
        "single_point_only_eligible": False,
    }


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TVTContractError("JSON root is not an object")
    return cast(dict[str, object], value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    args = parser.parse_args(argv)
    print(json.dumps(audit_split_registry(_read_json(Path(args.split))), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
