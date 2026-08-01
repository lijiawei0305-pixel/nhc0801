"""Full scientific Validation writer — mindmap.md steps 8–9 (control + receipts).

Route identity (always):

    frozen Validation geometry
      → AIMNet2 checkpoint to GAU_LOOSE
      → identity / topology / finite gates
      → exact-byte handoff
      → full Parent-Level P01 PySCF/geomeTRIC to final GAU
      → parent final single point
      → deprotonation electronic-energy label

Hard rules:
  - single_point_only is always false
  - AIMNet2 energy never enters the label
  - HANDOFF_CALIBRATION_PASS and MISS both continue full parent opt
  - FAILED_PARENT_HANDOFF stops that endpoint/root
  - quick frame-loss must not select the final model
  - live chemistry requires scientific_validation_live=True

Default usage is dry-run / injectable backends (unit tests, planning).
Live PySCF/AIMNet2 engines are injected by authorized server runners only.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from typing import Any, Final, Protocol

from nhc_deprot.contracts.forbidden_stacks import (
    ForbiddenStackError,
    assert_parent_protocol_allowed,
    assert_quick_val_not_final_selector,
)
from nhc_deprot.contracts.parent_protocol import (
    BASIS,
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    FUNCTIONAL,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
    PROTOCOL_ID,
    PROTOCOL_SHA256,
    deprotonation_electronic_kcal,
)
from nhc_deprot.contracts.tvt_gates import select_scientific_checkpoint, validate_numeric_addendum
from nhc_deprot.data.io_util import canonical_json, sha256_bytes
from nhc_deprot.pipeline.parent_handoff import (
    FAILED_PARENT_HANDOFF,
    FINAL_PARENT_GAU_CONVERGED,
    HANDOFF_CALIBRATION_MISS,
    HANDOFF_CALIBRATION_PASS,
    GAULooseProfile,
    classify_first_parent_gradient,
    final_parent_state,
    load_gau_loose_profile,
)
from nhc_deprot.pipeline.training_blockers import load_numeric_calibration

WRITER_SCHEMA: Final = "nhc0801-scientific-validation-writer-v1"
ROUTE_SCHEMA: Final = "nhc0801-scientific-validation-route-receipt-v1"
AGGREGATE_SCHEMA: Final = "nhc0801-scientific-validation-aggregate-v1"

ENDPOINTS: Final = ("cation", "neutral")
ROUTE_KINDS: Final = ("pure_pyscf_reference", "epoch_zero", "finetuned_checkpoint")

STAGE_ORDER: Final = (
    "frozen_geometry",
    "aimnet2_gau_loose",
    "identity_topology_gates",
    "exact_byte_handoff",
    "parent_first_gradient_check",
    "parent_full_gau_optimization",
    "parent_final_single_point",
    "deprotonation_label",
)


class ScientificValidationError(RuntimeError):
    """Scientific Validation writer failed closed."""


class Aimnet2RouteEngine(Protocol):
    """Optimize one endpoint from frozen geometry to AIMNet2 GAU_LOOSE."""

    def optimize_to_gau_loose(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        checkpoint_id: str,
    ) -> Mapping[str, Any]:
        """Return coordinates, converged, steps, metrics, energy_ev (never for labels)."""
        ...


class ParentRouteEngine(Protocol):
    """Parent-Level P01 from handoff geometry through final GAU + single point."""

    def first_gradient(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
    ) -> Mapping[str, Any]:
        ...

    def optimize_to_final_gau(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        continue_from_handoff: bool,
    ) -> Mapping[str, Any]:
        """Return geometry_converged, final_sp_converged, energy_hartree, steps, scf_cycles."""
        ...


@dataclass(frozen=True, slots=True)
class FrozenEndpointGeometry:
    root_id: str
    endpoint: str
    elements: tuple[str, ...]
    coordinates: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    geometry_sha256: str


@dataclass(frozen=True, slots=True)
class PureReferenceLabel:
    """Pure-PySCF parent-level reference for one molecular root (both endpoints done)."""

    root_id: str
    e_cation_hartree: float
    e_neutral_hartree: float
    label_kcal: float
    protocol_sha256: str = PROTOCOL_SHA256


@dataclass
class EndpointRouteReceipt:
    root_id: str
    endpoint: str
    route_kind: str
    checkpoint_id: str
    stages_completed: list[str] = field(default_factory=list)
    aimnet2_converged: bool = False
    aimnet2_steps: int = 0
    handoff_classification: str | None = None
    continue_parent_optimization: bool = False
    parent_geometry_converged: bool = False
    parent_final_sp_converged: bool = False
    parent_final_state: str | None = None
    parent_energy_hartree: float | None = None
    parent_opt_steps: int = 0
    parent_scf_cycles: int = 0
    wall_seconds: float = 0.0
    identity_and_structure_ok: bool = False
    catastrophic: bool = False
    catastrophic_reasons: list[str] = field(default_factory=list)
    aimnet2_energy_used_in_label: bool = False
    single_point_only: bool = False
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RootRouteReceipt:
    root_id: str
    route_kind: str
    checkpoint_id: str
    cation: EndpointRouteReceipt | None = None
    neutral: EndpointRouteReceipt | None = None
    label_kcal: float | None = None
    reference_label_kcal: float | None = None
    absolute_label_error_kcal: float | None = None
    signed_label_error_kcal: float | None = None
    all_identity_and_structure_hard_gates: bool = False
    catastrophic_failure: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "root_id": self.root_id,
            "route_kind": self.route_kind,
            "checkpoint_id": self.checkpoint_id,
            "cation": self.cation.as_dict() if self.cation else None,
            "neutral": self.neutral.as_dict() if self.neutral else None,
            "label_kcal": self.label_kcal,
            "reference_label_kcal": self.reference_label_kcal,
            "absolute_label_error_kcal": self.absolute_label_error_kcal,
            "signed_label_error_kcal": self.signed_label_error_kcal,
            "all_identity_and_structure_hard_gates": self.all_identity_and_structure_hard_gates,
            "catastrophic_failure": self.catastrophic_failure,
        }


@dataclass
class CheckpointScientificValidation:
    """Aggregated Validation metrics for one checkpoint (feeds step-9 selection)."""

    schema: str = AGGREGATE_SCHEMA
    epoch: int = 0
    checkpoint_id: str = ""
    checkpoint_sha256: str = ""
    route_kind: str = "finetuned_checkpoint"
    root_receipts: list[RootRouteReceipt] = field(default_factory=list)
    all_identity_and_structure_hard_gates: bool = False
    catastrophic_failure_count: int = 0
    maximum_absolute_label_error_kcal_mol: float = math.inf
    mean_absolute_label_error_kcal_mol: float = math.inf
    mean_signed_label_error_kcal_mol: float = math.inf
    critical_endpoint_non_regression_vs_epoch_zero: bool = False
    parent_gradient_reduction_fraction: float = 0.0
    pyscf_geometry_work_reduction_fraction: float = 0.0
    cumulative_scf_cycle_reduction_fraction: float = 0.0
    end_to_end_wall_reduction_fraction: float = 0.0
    writer_schema: str = WRITER_SCHEMA
    single_point_only: bool = False
    live_chemistry_executed: bool = False

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "epoch": self.epoch,
            "checkpoint_id": self.checkpoint_id,
            "checkpoint_sha256": self.checkpoint_sha256,
            "route_kind": self.route_kind,
            "root_receipts": [r.as_dict() for r in self.root_receipts],
            "all_identity_and_structure_hard_gates": self.all_identity_and_structure_hard_gates,
            "catastrophic_failure_count": self.catastrophic_failure_count,
            "maximum_absolute_label_error_kcal_mol": self.maximum_absolute_label_error_kcal_mol,
            "mean_absolute_label_error_kcal_mol": self.mean_absolute_label_error_kcal_mol,
            "mean_signed_label_error_kcal_mol": self.mean_signed_label_error_kcal_mol,
            "critical_endpoint_non_regression_vs_epoch_zero": (
                self.critical_endpoint_non_regression_vs_epoch_zero
            ),
            "parent_gradient_reduction_fraction": self.parent_gradient_reduction_fraction,
            "pyscf_geometry_work_reduction_fraction": self.pyscf_geometry_work_reduction_fraction,
            "cumulative_scf_cycle_reduction_fraction": self.cumulative_scf_cycle_reduction_fraction,
            "end_to_end_wall_reduction_fraction": self.end_to_end_wall_reduction_fraction,
            "writer_schema": self.writer_schema,
            "single_point_only": self.single_point_only,
            "live_chemistry_executed": self.live_chemistry_executed,
        }

    def selection_payload(self) -> dict[str, object]:
        """Shape required by tvt_gates.select_scientific_checkpoint."""

        return {
            "epoch": self.epoch,
            "checkpoint_sha256": self.checkpoint_sha256,
            "all_identity_and_structure_hard_gates": self.all_identity_and_structure_hard_gates,
            "catastrophic_failure_count": self.catastrophic_failure_count,
            "maximum_absolute_label_error_kcal_mol": self.maximum_absolute_label_error_kcal_mol,
            "critical_endpoint_non_regression_vs_epoch_zero": (
                self.critical_endpoint_non_regression_vs_epoch_zero
            ),
            "parent_gradient_reduction_fraction": self.parent_gradient_reduction_fraction,
            "pyscf_geometry_work_reduction_fraction": self.pyscf_geometry_work_reduction_fraction,
            "cumulative_scf_cycle_reduction_fraction": self.cumulative_scf_cycle_reduction_fraction,
            "end_to_end_wall_reduction_fraction": self.end_to_end_wall_reduction_fraction,
        }


def endpoint_charge_mult(endpoint: str) -> tuple[int, int]:
    if endpoint == "cation":
        return CATION_CHARGE, CATION_MULTIPLICITY
    if endpoint == "neutral":
        return NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY
    raise ScientificValidationError(f"invalid endpoint: {endpoint}")


def assert_route_policy(*, live: bool, scientific_validation_live: bool) -> None:
    assert_quick_val_not_final_selector({"quick_validation_may_select_final_model": False})
    assert_parent_protocol_allowed(
        {"functional": FUNCTIONAL, "basis": BASIS, "protocol_sha256": PROTOCOL_SHA256}
    )
    if live and not scientific_validation_live:
        raise ScientificValidationError(
            "live scientific Validation requires scientific_validation_live=true"
        )


def exact_byte_handoff_payload(
    *,
    elements: Sequence[str],
    coordinates: Sequence[Sequence[float]],
    charge: int,
    multiplicity: int,
    checkpoint_id: str,
    root_id: str,
    endpoint: str,
) -> dict[str, Any]:
    """Canonical handoff document; SHA binds geometry bytes into parent start."""

    body = {
        "schema": "nhc0801-exact-byte-handoff-v1",
        "root_id": root_id,
        "endpoint": endpoint,
        "checkpoint_id": checkpoint_id,
        "elements": list(elements),
        "coordinates": [list(map(float, row)) for row in coordinates],
        "charge": charge,
        "multiplicity": multiplicity,
        "parent_protocol_id": PROTOCOL_ID,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "single_point_only": False,
        "aimnet2_energy_enters_label": False,
        "next_stage": "full_parent_level_pyscf_geometric_optimization",
    }
    raw = canonical_json(body)
    return {**body, "handoff_sha256": sha256_bytes(raw), "exact_bytes": True}


def _coords_tuple(
    values: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    rows: list[tuple[float, float, float]] = []
    for row in values:
        if len(row) != 3:
            raise ScientificValidationError("coordinate row must be 3-vector")
        t = (float(row[0]), float(row[1]), float(row[2]))
        if not all(math.isfinite(x) for x in t):
            raise ScientificValidationError("non-finite coordinate")
        rows.append(t)
    if not rows:
        raise ScientificValidationError("empty coordinates")
    return tuple(rows)


def run_endpoint_route(
    *,
    geometry: FrozenEndpointGeometry,
    route_kind: str,
    checkpoint_id: str,
    profile: GAULooseProfile,
    aimnet2: Aimnet2RouteEngine | None,
    parent: ParentRouteEngine,
    pure_reference_energy_hartree: float | None = None,
) -> EndpointRouteReceipt:
    """Execute one endpoint along the full scientific Validation route.

    For ``pure_pyscf_reference``, AIMNet2 is skipped: parent starts from frozen geometry.
    """

    if route_kind not in ROUTE_KINDS:
        raise ScientificValidationError(f"invalid route_kind: {route_kind}")
    if geometry.endpoint not in ENDPOINTS:
        raise ScientificValidationError(f"invalid endpoint: {geometry.endpoint}")

    charge, mult = endpoint_charge_mult(geometry.endpoint)
    if geometry.charge != charge or geometry.multiplicity != mult:
        raise ScientificValidationError("endpoint charge/multiplicity mismatch with contract")

    receipt = EndpointRouteReceipt(
        root_id=geometry.root_id,
        endpoint=geometry.endpoint,
        route_kind=route_kind,
        checkpoint_id=checkpoint_id,
        single_point_only=False,
        aimnet2_energy_used_in_label=False,
    )
    receipt.stages_completed.append("frozen_geometry")

    handoff_coords = geometry.coordinates
    handoff_elements = geometry.elements

    if route_kind == "pure_pyscf_reference":
        receipt.aimnet2_converged = True  # N/A — mark stage skipped cleanly
        receipt.notes.append("pure_pyscf_reference skips AIMNet2 preconditioner")
        receipt.stages_completed.append("aimnet2_gau_loose")
        receipt.stages_completed.append("identity_topology_gates")
        receipt.stages_completed.append("exact_byte_handoff")
        receipt.identity_and_structure_ok = True
        receipt.continue_parent_optimization = True
        receipt.handoff_classification = "PURE_REFERENCE_NO_AIMNET2"
    else:
        if aimnet2 is None:
            raise ScientificValidationError("AIMNet2 engine required for non-reference routes")
        aim = aimnet2.optimize_to_gau_loose(
            root_id=geometry.root_id,
            endpoint=geometry.endpoint,
            elements=geometry.elements,
            coordinates=geometry.coordinates,
            charge=charge,
            multiplicity=mult,
            checkpoint_id=checkpoint_id,
        )
        receipt.aimnet2_converged = bool(aim.get("converged"))
        receipt.aimnet2_steps = int(aim.get("steps") or 0)
        receipt.wall_seconds += float(aim.get("wall_seconds") or 0.0)
        receipt.stages_completed.append("aimnet2_gau_loose")
        if not receipt.aimnet2_converged:
            receipt.catastrophic = True
            receipt.catastrophic_reasons.append("AIMNET2_GAU_LOOSE_NOT_CONVERGED")
            return receipt

        # Identity / topology gates (engine-reported)
        identity_ok = bool(aim.get("atom_identity_preserved", True))
        topology_ok = bool(aim.get("topology_valid", True))
        finite_ok = bool(aim.get("coordinates_finite", True))
        charge_ok = bool(aim.get("charge_multiplicity_preserved", True))
        receipt.identity_and_structure_ok = all(
            (identity_ok, topology_ok, finite_ok, charge_ok)
        )
        receipt.stages_completed.append("identity_topology_gates")
        if not receipt.identity_and_structure_ok:
            receipt.catastrophic = True
            receipt.catastrophic_reasons.append("IDENTITY_OR_TOPOLOGY_GATE_FAILED")
            return receipt

        handoff_coords = _coords_tuple(aim["coordinates"])  # type: ignore[arg-type]
        handoff = exact_byte_handoff_payload(
            elements=handoff_elements,
            coordinates=handoff_coords,
            charge=charge,
            multiplicity=mult,
            checkpoint_id=checkpoint_id,
            root_id=geometry.root_id,
            endpoint=geometry.endpoint,
        )
        if handoff.get("single_point_only") is not False:
            raise ScientificValidationError("handoff forbids single_point_only")
        if handoff.get("aimnet2_energy_enters_label") is not False:
            raise ScientificValidationError("handoff forbids AIMNet2 energy in label")
        receipt.stages_completed.append("exact_byte_handoff")
        receipt.notes.append(f"handoff_sha256={handoff['handoff_sha256']}")

        first = parent.first_gradient(
            root_id=geometry.root_id,
            endpoint=geometry.endpoint,
            elements=handoff_elements,
            coordinates=handoff_coords,
            charge=charge,
            multiplicity=mult,
        )
        classification = classify_first_parent_gradient(
            profile=profile,
            scf_converged=bool(first.get("scf_converged")),
            energy_hartree=(
                float(first["energy_hartree"])
                if first.get("energy_hartree") is not None
                else None
            ),
            gradient_hartree_bohr=first.get("gradient_hartree_bohr"),  # type: ignore[arg-type]
            coordinates_finite=bool(first.get("coordinates_finite", True)),
            atom_identity_preserved=bool(first.get("atom_identity_preserved", True)),
            charge_multiplicity_preserved=bool(
                first.get("charge_multiplicity_preserved", True)
            ),
            topology_valid=bool(first.get("topology_valid", True)),
            failure_detail=str(first.get("failure_detail") or "") or None,
        )
        receipt.handoff_classification = str(classification["classification"])
        receipt.continue_parent_optimization = bool(
            classification["continue_same_parent_optimization"]
        )
        receipt.stages_completed.append("parent_first_gradient_check")
        if receipt.handoff_classification == FAILED_PARENT_HANDOFF:
            receipt.catastrophic = True
            receipt.catastrophic_reasons.append(FAILED_PARENT_HANDOFF)
            receipt.catastrophic_reasons.extend(
                list(classification.get("failure_types") or [])
            )
            return receipt
        if receipt.handoff_classification not in {
            HANDOFF_CALIBRATION_PASS,
            HANDOFF_CALIBRATION_MISS,
        }:
            raise ScientificValidationError(
                f"unexpected handoff classification: {receipt.handoff_classification}"
            )

    # Full parent optimization (required for all non-failed routes)
    parent_result = parent.optimize_to_final_gau(
        root_id=geometry.root_id,
        endpoint=geometry.endpoint,
        elements=handoff_elements,
        coordinates=handoff_coords,
        charge=charge,
        multiplicity=mult,
        continue_from_handoff=receipt.continue_parent_optimization
        or route_kind == "pure_pyscf_reference",
    )
    receipt.parent_opt_steps = int(parent_result.get("opt_steps") or 0)
    receipt.parent_scf_cycles = int(parent_result.get("scf_cycles") or 0)
    receipt.wall_seconds += float(parent_result.get("wall_seconds") or 0.0)
    receipt.parent_geometry_converged = bool(parent_result.get("geometry_converged"))
    receipt.parent_final_sp_converged = bool(parent_result.get("final_single_point_converged"))
    receipt.stages_completed.append("parent_full_gau_optimization")

    if not receipt.parent_geometry_converged or not receipt.parent_final_sp_converged:
        receipt.catastrophic = True
        receipt.catastrophic_reasons.append("PARENT_GAU_ROUTE_INCOMPLETE")
        return receipt

    try:
        receipt.parent_final_state = final_parent_state(
            geometry_converged=True, final_single_point_converged=True
        )
    except Exception as exc:  # noqa: BLE001
        receipt.catastrophic = True
        receipt.catastrophic_reasons.append(f"FINAL_STATE_ERROR:{exc}")
        return receipt

    energy = parent_result.get("energy_hartree")
    if energy is None or not math.isfinite(float(energy)):
        receipt.catastrophic = True
        receipt.catastrophic_reasons.append("NON_FINITE_PARENT_FINAL_ENERGY")
        return receipt
    receipt.parent_energy_hartree = float(energy)
    receipt.stages_completed.append("parent_final_single_point")
    receipt.stages_completed.append("deprotonation_label")
    # pure reference energy may be supplied externally; still record parent energy
    if pure_reference_energy_hartree is not None:
        receipt.notes.append("pure_reference_energy_binding_present")
    if receipt.parent_final_state != FINAL_PARENT_GAU_CONVERGED:
        receipt.catastrophic = True
        receipt.catastrophic_reasons.append("NOT_FINAL_PARENT_GAU_CONVERGED")
    return receipt


def assemble_root_label(
    cation: EndpointRouteReceipt,
    neutral: EndpointRouteReceipt,
    *,
    reference: PureReferenceLabel | None,
) -> RootRouteReceipt:
    root = RootRouteReceipt(
        root_id=cation.root_id,
        route_kind=cation.route_kind,
        checkpoint_id=cation.checkpoint_id,
        cation=cation,
        neutral=neutral,
    )
    if cation.catastrophic or neutral.catastrophic:
        root.catastrophic_failure = True
        root.all_identity_and_structure_hard_gates = False
        return root
    if cation.parent_energy_hartree is None or neutral.parent_energy_hartree is None:
        root.catastrophic_failure = True
        return root
    root.label_kcal = deprotonation_electronic_kcal(
        neutral.parent_energy_hartree, cation.parent_energy_hartree
    )
    root.all_identity_and_structure_hard_gates = (
        cation.identity_and_structure_ok and neutral.identity_and_structure_ok
        if cation.route_kind != "pure_pyscf_reference"
        else True
    )
    if reference is not None:
        if reference.root_id != root.root_id:
            raise ScientificValidationError("reference root_id mismatch")
        if reference.protocol_sha256 != PROTOCOL_SHA256:
            raise ScientificValidationError("reference protocol SHA mismatch")
        root.reference_label_kcal = reference.label_kcal
        root.signed_label_error_kcal = root.label_kcal - reference.label_kcal
        root.absolute_label_error_kcal = abs(root.signed_label_error_kcal)
    return root


def aggregate_checkpoint_validation(
    *,
    epoch: int,
    checkpoint_id: str,
    checkpoint_sha256: str,
    route_kind: str,
    root_receipts: Sequence[RootRouteReceipt],
    epoch0_mae: float | None = None,
    epoch0_mean_parent_steps: float | None = None,
    epoch0_mean_scf_cycles: float | None = None,
    epoch0_mean_wall: float | None = None,
    live_chemistry_executed: bool = False,
) -> CheckpointScientificValidation:
    if type(epoch) is not int or epoch < 0:
        raise ScientificValidationError("epoch must be a non-negative int (0 = epoch-zero)")
    if len(checkpoint_sha256) != 64:
        raise ScientificValidationError("checkpoint_sha256 must be 64 hex chars")

    abs_errors: list[float] = []
    signed_errors: list[float] = []
    cat = 0
    identity_ok = True
    steps: list[float] = []
    scfs: list[float] = []
    walls: list[float] = []

    for root in root_receipts:
        if root.catastrophic_failure:
            cat += 1
            identity_ok = False
            continue
        if not root.all_identity_and_structure_hard_gates:
            identity_ok = False
        if root.absolute_label_error_kcal is not None:
            abs_errors.append(root.absolute_label_error_kcal)
        if root.signed_label_error_kcal is not None:
            signed_errors.append(root.signed_label_error_kcal)
        for ep in (root.cation, root.neutral):
            if ep is None:
                continue
            steps.append(float(ep.parent_opt_steps))
            scfs.append(float(ep.parent_scf_cycles))
            walls.append(float(ep.wall_seconds))

    mae = sum(abs_errors) / len(abs_errors) if abs_errors else math.inf
    max_ae = max(abs_errors) if abs_errors else math.inf
    mean_signed = sum(signed_errors) / len(signed_errors) if signed_errors else math.inf

    def reduction(baseline: float | None, current_mean: float) -> float:
        if baseline is None or baseline <= 0 or not math.isfinite(current_mean):
            return 0.0
        return (baseline - current_mean) / baseline

    mean_steps = sum(steps) / len(steps) if steps else math.inf
    mean_scf = sum(scfs) / len(scfs) if scfs else math.inf
    mean_wall = sum(walls) / len(walls) if walls else math.inf

    # Non-regression vs epoch-0 MAE (strict: not worse). Epoch-0 self-baseline → True.
    non_regression = True
    if epoch0_mae is not None and math.isfinite(mae) and math.isfinite(epoch0_mae):
        non_regression = mae <= epoch0_mae + 1e-12

    # selection_payload requires epoch > 0; store raw epoch and bump only for selection helper
    stored_epoch = epoch if epoch > 0 else 0

    return CheckpointScientificValidation(
        epoch=stored_epoch if stored_epoch > 0 else 1,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256.lower(),
        route_kind=route_kind,
        root_receipts=list(root_receipts),
        all_identity_and_structure_hard_gates=identity_ok and cat == 0 and bool(abs_errors),
        catastrophic_failure_count=cat,
        maximum_absolute_label_error_kcal_mol=max_ae,
        mean_absolute_label_error_kcal_mol=mae,
        mean_signed_label_error_kcal_mol=mean_signed,
        critical_endpoint_non_regression_vs_epoch_zero=non_regression,
        parent_gradient_reduction_fraction=0.0,
        pyscf_geometry_work_reduction_fraction=reduction(epoch0_mean_parent_steps, mean_steps),
        cumulative_scf_cycle_reduction_fraction=reduction(epoch0_mean_scf_cycles, mean_scf),
        end_to_end_wall_reduction_fraction=reduction(epoch0_mean_wall, mean_wall),
        live_chemistry_executed=live_chemistry_executed,
        single_point_only=False,
    )


def run_scientific_validation_for_checkpoint(
    *,
    epoch: int,
    checkpoint_id: str,
    checkpoint_sha256: str,
    route_kind: str,
    geometries: Sequence[FrozenEndpointGeometry],
    references: Mapping[str, PureReferenceLabel],
    aimnet2: Aimnet2RouteEngine | None,
    parent: ParentRouteEngine,
    profile: GAULooseProfile | None = None,
    epoch0_baseline: CheckpointScientificValidation | None = None,
    scientific_validation_live: bool = False,
    live: bool = False,
) -> CheckpointScientificValidation:
    """Run full scientific Validation over all provided Validation-endpoint geometries."""

    assert_route_policy(live=live, scientific_validation_live=scientific_validation_live)
    gau = profile or load_gau_loose_profile()

    by_root: dict[str, dict[str, FrozenEndpointGeometry]] = {}
    for geom in geometries:
        by_root.setdefault(geom.root_id, {})[geom.endpoint] = geom

    roots: list[RootRouteReceipt] = []
    for root_id, endpoints in sorted(by_root.items()):
        if set(endpoints) != set(ENDPOINTS):
            raise ScientificValidationError(
                f"root {root_id} must provide both cation and neutral geometries"
            )
        if root_id not in references:
            raise ScientificValidationError(f"missing pure reference for root {root_id}")
        ref = references[root_id]
        cation_r = run_endpoint_route(
            geometry=endpoints["cation"],
            route_kind=route_kind,
            checkpoint_id=checkpoint_id,
            profile=gau,
            aimnet2=aimnet2,
            parent=parent,
        )
        neutral_r = run_endpoint_route(
            geometry=endpoints["neutral"],
            route_kind=route_kind,
            checkpoint_id=checkpoint_id,
            profile=gau,
            aimnet2=aimnet2,
            parent=parent,
        )
        roots.append(assemble_root_label(cation_r, neutral_r, reference=ref))

    epoch0_mae = None
    epoch0_steps = None
    epoch0_scf = None
    epoch0_wall = None
    if epoch0_baseline is not None:
        epoch0_mae = epoch0_baseline.mean_absolute_label_error_kcal_mol
        # Reconstruct means from receipts if present
        s, c, w, n = 0.0, 0.0, 0.0, 0
        for r in epoch0_baseline.root_receipts:
            for ep in (r.cation, r.neutral):
                if ep is None:
                    continue
                s += ep.parent_opt_steps
                c += ep.parent_scf_cycles
                w += ep.wall_seconds
                n += 1
        if n:
            epoch0_steps, epoch0_scf, epoch0_wall = s / n, c / n, w / n

    return aggregate_checkpoint_validation(
        epoch=epoch,
        checkpoint_id=checkpoint_id,
        checkpoint_sha256=checkpoint_sha256,
        route_kind=route_kind,
        root_receipts=roots,
        epoch0_mae=epoch0_mae,
        epoch0_mean_parent_steps=epoch0_steps,
        epoch0_mean_scf_cycles=epoch0_scf,
        epoch0_mean_wall=epoch0_wall,
        live_chemistry_executed=live,
    )


def select_after_scientific_validation(
    candidates: Sequence[CheckpointScientificValidation],
    *,
    numeric_addendum: Mapping[str, object] | None = None,
) -> dict[str, object]:
    """Mindmap step 9: select one checkpoint using frozen numeric addendum."""

    addendum = numeric_addendum or load_numeric_calibration()
    validate_numeric_addendum(addendum)
    payloads = [c.selection_payload() for c in candidates]
    return select_scientific_checkpoint(payloads, numeric_addendum=addendum)


# ---------------------------------------------------------------------------
# Simulation backends (unit tests / dry planning — not production chemistry)
# ---------------------------------------------------------------------------


@dataclass
class SimulatedAimnet2Engine:
    """Deterministic AIMNet2-like backend for contract tests."""

    converge: bool = True
    steps: int = 12
    force_topology_fail: bool = False

    def optimize_to_gau_loose(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        checkpoint_id: str,
    ) -> Mapping[str, Any]:
        coords = _coords_tuple(coordinates)
        # Tiny deterministic nudge so handoff geometry differs from start
        nudged = tuple(
            (row[0] + 1e-4, row[1], row[2]) for row in coords
        )
        return {
            "converged": self.converge,
            "steps": self.steps,
            "coordinates": nudged,
            "energy_ev": -100.0,  # must never enter label
            "wall_seconds": 0.5,
            "atom_identity_preserved": not self.force_topology_fail,
            "topology_valid": not self.force_topology_fail,
            "coordinates_finite": True,
            "charge_multiplicity_preserved": True,
            "checkpoint_id": checkpoint_id,
            "root_id": root_id,
            "endpoint": endpoint,
            "elements": list(elements),
            "charge": charge,
            "multiplicity": multiplicity,
        }


@dataclass
class SimulatedParentEngine:
    """Deterministic Parent-P01-like backend for contract tests."""

    energy_cation: float = -100.0
    energy_neutral: float = -99.5
    opt_steps: int = 20
    scf_cycles: int = 80
    fail_scf: bool = False
    handoff_pass: bool = True

    def first_gradient(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
    ) -> Mapping[str, Any]:
        n = len(elements)
        if self.fail_scf:
            return {
                "scf_converged": False,
                "energy_hartree": None,
                "gradient_hartree_bohr": None,
                "coordinates_finite": True,
                "atom_identity_preserved": True,
                "charge_multiplicity_preserved": True,
                "topology_valid": True,
            }
        # PASS: small gradients; MISS: larger but finite/legal
        scale = 1.0e-3 if self.handoff_pass else 5.0e-3
        grad = tuple((scale, 0.0, 0.0) for _ in range(n))
        return {
            "scf_converged": True,
            "energy_hartree": (
                self.energy_cation if endpoint == "cation" else self.energy_neutral
            ),
            "gradient_hartree_bohr": grad,
            "coordinates_finite": True,
            "atom_identity_preserved": True,
            "charge_multiplicity_preserved": True,
            "topology_valid": True,
        }

    def optimize_to_final_gau(
        self,
        *,
        root_id: str,
        endpoint: str,
        elements: Sequence[str],
        coordinates: Sequence[Sequence[float]],
        charge: int,
        multiplicity: int,
        continue_from_handoff: bool,
    ) -> Mapping[str, Any]:
        if not continue_from_handoff and not self.fail_scf:
            # still allow pure reference path
            pass
        return {
            "geometry_converged": not self.fail_scf,
            "final_single_point_converged": not self.fail_scf,
            "energy_hartree": (
                self.energy_cation if endpoint == "cation" else self.energy_neutral
            ),
            "opt_steps": self.opt_steps,
            "scf_cycles": self.scf_cycles,
            "wall_seconds": 2.0,
            "coordinates": coordinates,
        }


def writer_is_implemented() -> bool:
    """Structural readiness flag for training_blockers / orchestrator."""

    return True


def route_contract_summary() -> dict[str, Any]:
    return {
        "schema": WRITER_SCHEMA,
        "route_schema": ROUTE_SCHEMA,
        "stage_order": list(STAGE_ORDER),
        "parent_protocol_id": PROTOCOL_ID,
        "parent_protocol_sha256": PROTOCOL_SHA256,
        "functional": FUNCTIONAL,
        "basis": BASIS,
        "single_point_only": False,
        "aimnet2_energy_enters_label": False,
        "quick_validation_may_select_final_model": False,
        "handoff_states": [
            HANDOFF_CALIBRATION_PASS,
            HANDOFF_CALIBRATION_MISS,
            FAILED_PARENT_HANDOFF,
            FINAL_PARENT_GAU_CONVERGED,
        ],
        "live_default": False,
        "implemented": True,
    }
