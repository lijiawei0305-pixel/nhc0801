"""Mindmap step orchestrator — preflight + planning only by default.

Implements the automation *control plane* for mindmap.md steps 0–12:
  - initial checks (split / TVT / forbidden stacks / training blockers)
  - ordered plan of steps with gate requirements
  - refuses live chemistry/training unless explicitly authorized

This is intentionally not a silent end-to-end trainer. Live steps stay stubs
until the corresponding gate is opened.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.forbidden_stacks import (
    ForbiddenStackError,
    assert_quick_val_not_final_selector,
)
from nhc_deprot.data.development_split import DevelopmentSplit, load_packaged_v004_day1_split
from nhc_deprot.mindmap_steps import MINDMAP_IMPLEMENTATION
from nhc_deprot.pipeline.training_blockers import (
    TrainingReadiness,
    assess_training_readiness,
    format_readiness_report,
)

# Gates that block live execution (defaults false)
DEFAULT_GATES: Final = {
    "teacher_pyscf_authorized": False,
    "aimnet2_train_authorized": False,
    "epoch0_execution": False,
    "final_test_open": False,
    "scientific_validation_live": False,
    "modify_wjw_outside_NHC0801": False,
    "scheduler_submission": False,
}

# Which mindmap steps require which gate for *live* run
STEP_LIVE_GATES: Final = {
    2: ("teacher_pyscf_authorized",),
    3: ("epoch0_execution",),
    4: ("aimnet2_train_authorized",),
    5: ("aimnet2_train_authorized",),
    8: ("scientific_validation_live",),
    9: ("scientific_validation_live",),
    11: ("final_test_open",),
}


@dataclass
class StepPlan:
    step: int
    title: str
    status: str
    modules: list[str]
    live_gates_required: tuple[str, ...]
    live_allowed: bool
    action: str  # dry_run_ok | blocked | ready_if_authorized


@dataclass
class OrchestratorReport:
    preflight_ok: bool
    readiness: TrainingReadiness
    split_summary: dict[str, Any]
    steps: list[StepPlan] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "preflight_ok": self.preflight_ok,
            "readiness": self.readiness.as_dict(),
            "split_summary": self.split_summary,
            "steps": [
                {
                    "step": s.step,
                    "title": s.title,
                    "status": s.status,
                    "modules": s.modules,
                    "live_gates_required": list(s.live_gates_required),
                    "live_allowed": s.live_allowed,
                    "action": s.action,
                }
                for s in self.steps
            ],
            "errors": list(self.errors),
            "warnings": list(self.warnings),
        }


def _action_for_step(
    step: int,
    *,
    gates: dict[str, bool],
    info: dict[str, Any],
) -> tuple[bool, str]:
    required = STEP_LIVE_GATES.get(step, ())
    if not required:
        # Planning / gates / freeze steps — always dry-run safe
        return True, "dry_run_ok"
    if all(gates.get(g, False) for g in required):
        return True, "ready_if_authorized"
    return False, "blocked"


def preflight(
    *,
    repo_root: Path | None = None,
    gates: dict[str, bool] | None = None,
    load_split: Callable[..., DevelopmentSplit] | None = None,
) -> OrchestratorReport:
    """Run full initial checks and emit a mindmap execution plan (no chemistry)."""

    root = repo_root or Path(__file__).resolve().parents[3]
    active_gates = {**DEFAULT_GATES, **(gates or {})}
    errors: list[str] = []
    warnings: list[str] = []

    # Forbidden policy
    try:
        assert_quick_val_not_final_selector(
            {"quick_validation_may_select_final_model": False}
        )
    except ForbiddenStackError as exc:
        errors.append(str(exc))

    if active_gates.get("modify_wjw_outside_NHC0801"):
        errors.append("modify_wjw_outside_NHC0801 must stay false")

    # Split integrity (Train ∩ Val = ∅; Final Test sealed only)
    split_summary: dict[str, Any] = {}
    try:
        loader = load_split or load_packaged_v004_day1_split
        split = loader(repo_root=root, require_v004_pilot_roots=True)
        train = set(split.train_roots)
        val = set(split.validation_roots)
        if train & val:
            errors.append(f"train/validation overlap: {sorted(train & val)}")
        if not train or not val:
            errors.append("empty train or validation split")
        split_summary = {
            "train_roots": list(split.train_roots),
            "validation_roots": list(split.validation_roots),
            "train_count": len(train),
            "validation_count": len(val),
            "sealed_final_test_sha256": split.sealed_final_test.sha256,
            "sealed_final_test_root_count": split.sealed_final_test.root_count,
            "final_test_identities_loaded": False,
            "not_admitted": list(split.not_admitted),
        }
        if active_gates.get("final_test_open"):
            warnings.append(
                "final_test_open=true is dangerous; identities must still not leak into training"
            )
    except Exception as exc:  # noqa: BLE001
        errors.append(f"split preflight failed: {exc}")

    readiness = assess_training_readiness(
        repo_root=root,
        live_train_authorized=bool(active_gates.get("aimnet2_train_authorized")),
        epoch0_baseline_available=False,
        scientific_validation_writer_ready=False,
        resource_claim_ok=False,
    )

    steps: list[StepPlan] = []
    for step in range(0, 13):
        info = MINDMAP_IMPLEMENTATION[step]
        live_allowed, action = _action_for_step(step, gates=active_gates, info=info)
        required = STEP_LIVE_GATES.get(step, ())
        steps.append(
            StepPlan(
                step=step,
                title=str(info["title"]),
                status=str(info["status"]),
                modules=list(info.get("modules") or []),
                live_gates_required=required,
                live_allowed=live_allowed,
                action=action,
            )
        )

    # Automation rule: never plan live train if hard blockers open
    if readiness.open_hard and active_gates.get("aimnet2_train_authorized"):
        warnings.append(
            "aimnet2_train_authorized=true but hard blockers still open — "
            "orchestrator will refuse live train"
        )
        for s in steps:
            if s.step in (4, 5):
                s.live_allowed = False
                s.action = "blocked"

    preflight_ok = not errors
    return OrchestratorReport(
        preflight_ok=preflight_ok,
        readiness=readiness,
        split_summary=split_summary,
        steps=steps,
        errors=errors,
        warnings=warnings,
    )


def run_dry(repo_root: Path | None = None, gates: dict[str, bool] | None = None) -> str:
    """CLI-friendly text report for mindmap preflight + readiness."""

    report = preflight(repo_root=repo_root, gates=gates)
    lines = [
        "=== NHC0801 mindmap orchestrator (dry-run) ===",
        f"preflight_ok: {report.preflight_ok}",
        "",
        format_readiness_report(report.readiness),
        "",
        "--- split ---",
        str(report.split_summary),
        "",
        "--- steps 0–12 ---",
    ]
    for s in report.steps:
        gate = ",".join(s.live_gates_required) if s.live_gates_required else "-"
        lines.append(
            f"  [{s.step:02d}] {s.action:22s} gates={gate:28s} {s.title} ({s.status})"
        )
    if report.errors:
        lines.append("")
        lines.append("ERRORS:")
        lines.extend(f"  - {e}" for e in report.errors)
    if report.warnings:
        lines.append("")
        lines.append("WARNINGS:")
        lines.extend(f"  - {w}" for w in report.warnings)
    lines.append("")
    lines.append(
        "Live train/epoch-0/PySCF refuse by default. Open one gate at a time after preflight_ok."
    )
    return "\n".join(lines)


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser(description="NHC0801 mindmap preflight orchestrator")
    parser.add_argument("--repo", type=Path, default=None)
    args = parser.parse_args()
    print(run_dry(repo_root=args.repo))
    report = preflight(repo_root=args.repo)
    return 0 if report.preflight_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
