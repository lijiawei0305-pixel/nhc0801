"""Diagnose BLOCKED_BEFORE_TRAINING reasons and map each to a resolution path.

Does not authorize or start live training. Chemistry/training gates stay closed
until the user explicitly opens them.
"""

from __future__ import annotations

import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final, Mapping

import yaml

from nhc_deprot.contracts.tvt_gates import TVTContractError, validate_numeric_addendum
from nhc_deprot.data.development_split import load_packaged_v004_day1_split
from nhc_deprot.data.io_util import load_json_object
from nhc_deprot.data.weighted_dataset import audit_public_weighted_result

# Original V004 reason codes (kept for continuity)
REASON_SOURCE: Final = "SOURCE_COMMIT_NOT_FROZEN"
REASON_EPOCH0: Final = "EPOCH_ZERO_FULL_ROUTE_BASELINE_NOT_AVAILABLE"
REASON_NUMERIC: Final = "NUMERIC_CALIBRATION_RULE_NOT_PREREGISTERED"
REASON_SCI_VAL: Final = "FULL_SCIENTIFIC_VALIDATION_WRITER_NOT_IMPLEMENTED"
REASON_RESOURCE: Final = "LIVE_RESOURCE_CLAIM_REJECTED_OR_UNAVAILABLE"
REASON_AUTH: Final = "LIVE_TRAINING_NOT_AUTHORIZED"
REASON_DATASET: Final = "WEIGHTED_DEVELOPMENT_DATASET_NOT_READY"

ALL_KNOWN: Final = (
    REASON_SOURCE,
    REASON_EPOCH0,
    REASON_NUMERIC,
    REASON_SCI_VAL,
    REASON_RESOURCE,
    REASON_AUTH,
    REASON_DATASET,
)


@dataclass
class Blocker:
    code: str
    status: str  # OPEN | RESOLVED | DEFERRED
    severity: str  # hard | soft
    summary: str
    resolution: str


@dataclass
class TrainingReadiness:
    state: str  # BLOCKED_BEFORE_TRAINING | READY_FOR_AUTHORIZED_TRAINING
    blockers: list[Blocker] = field(default_factory=list)
    open_hard: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "state": self.state,
            "open_hard_blockers": list(self.open_hard),
            "blockers": [
                {
                    "code": b.code,
                    "status": b.status,
                    "severity": b.severity,
                    "summary": b.summary,
                    "resolution": b.resolution,
                }
                for b in self.blockers
            ],
            "notes": list(self.notes),
        }


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _git_source_frozen(repo: Path) -> tuple[bool, str]:
    """True when HEAD exists and worktree is clean enough to freeze a commit id."""

    git_dir = repo / ".git"
    if not git_dir.exists():
        return False, "no .git — init + commit NHC0801 tree to freeze SOURCE_COMMIT"
    try:
        head = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if head.returncode != 0:
            return False, "git HEAD missing — make an initial commit"
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=repo,
            capture_output=True,
            text=True,
            check=False,
        )
        if dirty.returncode != 0:
            return False, "git status failed"
        if dirty.stdout.strip():
            return (
                False,
                f"worktree dirty at {head.stdout.strip()[:12]}; commit or stash before freeze",
            )
        return True, head.stdout.strip()
    except OSError as exc:
        return False, f"git unavailable: {exc}"


def load_numeric_calibration(path: Path | None = None) -> dict[str, object]:
    root = _repo_root()
    cal_path = path or (root / "docs" / "contracts" / "NUMERIC_CALIBRATION_V001.yaml")
    if not cal_path.is_file():
        raise FileNotFoundError(cal_path)
    raw = yaml.safe_load(cal_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise TVTContractError("numeric calibration root must be a mapping")
    return validate_numeric_addendum(raw)


def assess_training_readiness(
    *,
    repo_root: Path | None = None,
    live_train_authorized: bool = False,
    epoch0_baseline_available: bool = False,
    scientific_validation_writer_ready: bool = False,
    resource_claim_ok: bool = False,
    weighted_dataset_public_result: Path | None = None,
) -> TrainingReadiness:
    """Return open/resolved blockers. Never starts training."""

    root = repo_root or _repo_root()
    blockers: list[Blocker] = []
    notes: list[str] = []

    # 1) Source commit
    frozen, detail = _git_source_frozen(root)
    blockers.append(
        Blocker(
            code=REASON_SOURCE,
            status="RESOLVED" if frozen else "OPEN",
            severity="hard",
            summary=detail if frozen else f"source not frozen: {detail}",
            resolution=(
                "git commit -am 'freeze NHC0801 for training' && record SHA in "
                "generation config / PHASE_STATUS.md"
                if not frozen
                else f"SOURCE_COMMIT={detail}"
            ),
        )
    )

    # 2) Numeric calibration (cleared by NUMERIC_CALIBRATION_V001.yaml)
    try:
        load_numeric_calibration(root / "docs" / "contracts" / "NUMERIC_CALIBRATION_V001.yaml")
        blockers.append(
            Blocker(
                code=REASON_NUMERIC,
                status="RESOLVED",
                severity="hard",
                summary="NUMERIC_CALIBRATION_V001.yaml frozen and schema-valid",
                resolution="no action; do not revise after Final Test open",
            )
        )
    except (OSError, TVTContractError, yaml.YAMLError) as exc:
        blockers.append(
            Blocker(
                code=REASON_NUMERIC,
                status="OPEN",
                severity="hard",
                summary=f"numeric calibration missing/invalid: {exc}",
                resolution="fix or re-freeze docs/contracts/NUMERIC_CALIBRATION_V001.yaml",
            )
        )

    # 3) Epoch-0 full route baseline
    blockers.append(
        Blocker(
            code=REASON_EPOCH0,
            status="RESOLVED" if epoch0_baseline_available else "OPEN",
            severity="hard",
            summary=(
                "epoch-0 full-route baseline available"
                if epoch0_baseline_available
                else "epoch-0 full GAU_LOOSE→parent GAU Validation baseline NOT_RUN"
            ),
            resolution=(
                "no action"
                if epoch0_baseline_available
                else (
                    "1) free CPUs / pass live resource claim; "
                    "2) user sets epoch0_execution=true; "
                    "3) run official aimnet2_wb97m_d3_0.pt only; "
                    "4) write baseline receipt under $WJW/NHC0801/runs/"
                )
            ),
        )
    )

    # 4) Full scientific Validation writer (structural implementation)
    try:
        from nhc_deprot.pipeline.scientific_validation import writer_is_implemented

        writer_ready = bool(scientific_validation_writer_ready or writer_is_implemented())
    except Exception:  # noqa: BLE001
        writer_ready = bool(scientific_validation_writer_ready)
    blockers.append(
        Blocker(
            code=REASON_SCI_VAL,
            status="RESOLVED" if writer_ready else "OPEN",
            severity="hard",
            summary=(
                "scientific Validation writer implemented "
                "(live chemistry still gated; use simulated or authorized engines)"
                if writer_ready
                else "full scientific Validation writer not implemented (mindmap 8–9)"
            ),
            resolution=(
                "wire live AIMNet2/Parent engines under scientific_validation_live=true; "
                "epoch-0 baseline still required before training"
                if writer_ready
                else (
                    "implement GAU_LOOSE → exact-byte handoff → full parent P01 GAU → "
                    "label compare writer under src/nhc_deprot/pipeline/ "
                    "(no two_endpoint; no quick-val final select)"
                )
            ),
        )
    )

    # 5) Resource claim (soft until live train)
    blockers.append(
        Blocker(
            code=REASON_RESOURCE,
            status="RESOLVED" if resource_claim_ok else "OPEN",
            severity="hard" if live_train_authorized else "soft",
            summary=(
                "resource claim OK"
                if resource_claim_ok
                else "dual-worker claim V002 REJECTED / busy CPUs — wait or re-claim"
            ),
            resolution=(
                "re-run live resource claim when free -h / nproc show idle cores; "
                "prefer single_27_physical_v1 until dual-worker calibration PASSes; "
                "never fight VASP/large PySCF jobs"
            ),
        )
    )

    # 6) Explicit live-train authorization (NHC0801 gate)
    blockers.append(
        Blocker(
            code=REASON_AUTH,
            status="RESOLVED" if live_train_authorized else "OPEN",
            severity="hard",
            summary=(
                "live AIMNet2 training authorized"
                if live_train_authorized
                else "aimnet2_train_authorized=false (default)"
            ),
            resolution=(
                "user must explicitly set aimnet2_train_authorized=true after "
                "source freeze + epoch-0 + numeric cal + sci-val writer + resource claim"
            ),
        )
    )

    # 7) Development dataset / split sanity (local evidence)
    try:
        split = load_packaged_v004_day1_split(repo_root=root, require_v004_pilot_roots=True)
        notes.append(
            f"dev split OK train={len(split.train_roots)} val={len(split.validation_roots)} "
            f"FT sealed={split.sealed_final_test.sha256[:12]}…"
        )
        result_path = weighted_dataset_public_result or (
            root
            / "docs"
            / "evidence"
            / "pilot_day1"
            / "WEIGHTED_DATASET_RESULT.json"
        )
        if not result_path.is_file():
            # fallback to legacy extract path during transition
            alt = (
                root
                / "docs"
                / "extracted"
                / "v004"
                / "PHASE9B_AIMNET2_V004_WEIGHTED_DATASET_RESULT.json"
            )
            result_path = alt if alt.is_file() else result_path
        audit_public_weighted_result(result_path)
        blockers.append(
            Blocker(
                code=REASON_DATASET,
                status="RESOLVED",
                severity="hard",
                summary="development split + weighted-dataset public result PASS",
                resolution="server NPZ remains at $WJW/.../phase9b_aimnet2_v004_weighted_dataset_v001",
            )
        )
    except Exception as exc:  # noqa: BLE001 — readiness surface
        blockers.append(
            Blocker(
                code=REASON_DATASET,
                status="OPEN",
                severity="hard",
                summary=f"dataset/split not ready: {exc}",
                resolution="restore pilot evidence JSON / fix development split loader",
            )
        )

    open_hard = [b.code for b in blockers if b.status == "OPEN" and b.severity == "hard"]
    state = (
        "READY_FOR_AUTHORIZED_TRAINING"
        if not open_hard
        else "BLOCKED_BEFORE_TRAINING"
    )
    return TrainingReadiness(state=state, blockers=blockers, open_hard=open_hard, notes=notes)


def format_readiness_report(readiness: TrainingReadiness) -> str:
    lines = [
        f"state: {readiness.state}",
        f"open_hard: {', '.join(readiness.open_hard) or '(none)'}",
        "",
    ]
    for b in readiness.blockers:
        lines.append(f"[{b.status}/{b.severity}] {b.code}")
        lines.append(f"  {b.summary}")
        lines.append(f"  → {b.resolution}")
        lines.append("")
    for note in readiness.notes:
        lines.append(f"note: {note}")
    return "\n".join(lines)
