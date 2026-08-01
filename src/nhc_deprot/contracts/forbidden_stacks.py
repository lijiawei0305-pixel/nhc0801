"""Stacks that must never be used as parent DFT or mindmap training semantics.

Production two_endpoint B3LYP/def2-SVP + fmax=0.05 is intentionally forbidden.
Historical finetune that selects the final model by quick-val loss is forbidden.
"""

from __future__ import annotations

from typing import Final, Mapping

from nhc_deprot.contracts.parent_protocol import BASIS, FUNCTIONAL, PROTOCOL_SHA256

# Human-readable ban list (attention / docs / code review)
FORBIDDEN_PARENT_STACKS: Final = (
    "production_two_endpoint_b3lyp_def2_svp",
    "b3lyp_d3bj_def2_svp_as_parent",
    "aimnet2_preopt_fmax_0p05_as_parent_stop",
    "historical_finetune_quick_val_selects_final_model",
)

FORBIDDEN_TOKENS: Final = (
    "two_endpoint",
    "LOCKED_PROTOCOL",
    "B3LYP",
    "def2-SVP",
    "def2_SVP",
    "fmax=0.05",
    "fmax = 0.05",
)

REQUIRED_PARENT: Final = {
    "functional": FUNCTIONAL,
    "basis": BASIS,
    "protocol_sha256": PROTOCOL_SHA256,
}


class ForbiddenStackError(RuntimeError):
    """A banned science stack was requested as parent or final-selection policy."""


def assert_parent_protocol_allowed(meta: Mapping[str, object]) -> None:
    """Fail closed if protocol metadata matches the production B3LYP/SVP stack."""

    functional = str(meta.get("functional", meta.get("method", ""))).lower()
    basis = str(meta.get("basis", "")).lower()
    if "b3lyp" in functional and "svp" in basis.replace("_", "-"):
        raise ForbiddenStackError(
            "B3LYP/def2-SVP two_endpoint stack is forbidden as parent; "
            f"required parent is {FUNCTIONAL}/{BASIS} (P01 {PROTOCOL_SHA256[:12]}…)"
        )
    fmax = meta.get("fmax") or meta.get("aimnet2_fmax") or meta.get("preopt_fmax")
    if fmax is not None:
        try:
            if abs(float(fmax) - 0.05) < 1e-12 and meta.get("role") in {
                "parent_stop",
                "parent",
                "final_convergence",
            }:
                raise ForbiddenStackError(
                    "AIMNet2 fmax=0.05 must not be treated as parent convergence"
                )
        except (TypeError, ValueError):
            pass


def assert_quick_val_not_final_selector(config: Mapping[str, object]) -> None:
    if config.get("quick_validation_may_select_final_model") is True:
        raise ForbiddenStackError(
            "quick validation must not select the final model (mindmap steps 6–9)"
        )
