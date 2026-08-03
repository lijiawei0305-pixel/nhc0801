"""Zero-DFT pre-screen — mindmap step 7 strengthened (not final selection).

Only AIMNet2 GAU_LOOSE relaxation via the same ``optimize_to_gau_loose``
Protocol used by :mod:`nhc_deprot.pipeline.live_epoch0`. **No PySCF.**

Hard rules (AGENTS T1/T2):
  - Frame-level energy loss never ranks or selects.
  - Receipt must set ``final_model_selected: false`` and
    ``selection_authority: "pre_screen_shortlist_only_not_final"``.
  - Does not require ``scientific_validation_live``.

Reference geometries come from teacher products:
  start = frame_0000, terminal = is_terminal: true frame.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final, Protocol

import numpy as np

from nhc_deprot.contracts.parent_protocol import (
    CATION_CHARGE,
    CATION_MULTIPLICITY,
    NEUTRAL_CHARGE,
    NEUTRAL_MULTIPLICITY,
)
from nhc_deprot.data.io_util import load_json_object, write_json
from nhc_deprot.generation.layout import GenerationLayout

CAMPAIGN_SCHEMA: Final = "nhc0801-pre-screen-campaign-v1"
SELECTION_AUTHORITY: Final = "pre_screen_shortlist_only_not_final"
MINDMAP_STEP: Final = 7
DEFAULT_SHORTLIST_COUNT: Final = 3
ENDPOINTS: Final = ("cation", "neutral")

# Hartree/Bohr → eV/Å (same constants as weighted_dataset_writer / d3_projection)
HARTREE_TO_EV: Final = 27.211386245988
BOHR_TO_ANGSTROM: Final = 0.529177210903
FORCE_H_PER_B_TO_EV_PER_A: Final = HARTREE_TO_EV / BOHR_TO_ANGSTROM


class PreScreenError(RuntimeError):
    """Pre-screen failed closed."""


class GauLooseEngine(Protocol):
    """AIMNet2 GAU_LOOSE relaxer — same signature as live_epoch0 / sci-val."""

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
        """Return converged, steps, coordinates, identity/topology flags.

        Optional keys for force RMSE at the teacher terminal geometry:
          - ``forces_at_reference_ev_per_a``: (n_atoms, 3) model forces in eV/Å
            evaluated at the **reference** coordinates (not the relaxed ones).
        """
        ...


@dataclass(frozen=True, slots=True)
class TeacherEndpointReference:
    """Teacher start + terminal geometry for one (root, endpoint)."""

    root_id: str
    endpoint: str
    elements: tuple[str, ...]
    start_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    reference_coordinates_angstrom: tuple[tuple[float, float, float], ...]
    reference_forces_ev_per_a: tuple[tuple[float, float, float], ...]
    charge: int
    multiplicity: int
    start_frame_index: int = 0
    reference_frame_index: int = 1


@dataclass(frozen=True, slots=True)
class CheckpointCandidate:
    """One trainable checkpoint to pre-screen."""

    checkpoint_id: str
    run_id: str
    seed: int
    epoch: int
    weight_path: str | None = None


@dataclass(frozen=True, slots=True)
class EndpointScreenMetrics:
    root_id: str
    endpoint: str
    checkpoint_id: str
    identity_ok: bool
    topology_preserved: bool
    gau_loose_converged: bool
    rmsd_to_reference_angstrom: float
    aimnet2_steps_to_gau_loose: int
    force_rmse_at_reference_ev_per_a: float
    hard_gates_passed: bool
    notes: tuple[str, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateScreenResult:
    candidate: CheckpointCandidate
    per_endpoint: list[EndpointScreenMetrics] = field(default_factory=list)
    identity_ok: bool = False
    topology_preserved: bool = False
    gau_loose_converged: bool = False
    hard_gates_passed: bool = False
    mean_rmsd_to_reference_angstrom: float = math.inf
    mean_aimnet2_steps_to_gau_loose: float = math.inf
    mean_force_rmse_at_reference_ev_per_a: float = math.inf
    rank_key: tuple[Any, ...] = ()

    def as_dict(self) -> dict[str, Any]:
        return {
            "checkpoint_id": self.candidate.checkpoint_id,
            "run_id": self.candidate.run_id,
            "seed": self.candidate.seed,
            "epoch": self.candidate.epoch,
            "weight_path": self.candidate.weight_path,
            "identity_ok": self.identity_ok,
            "topology_preserved": self.topology_preserved,
            "gau_loose_converged": self.gau_loose_converged,
            "hard_gates_passed": self.hard_gates_passed,
            "mean_rmsd_to_reference_angstrom": self.mean_rmsd_to_reference_angstrom,
            "mean_aimnet2_steps_to_gau_loose": self.mean_aimnet2_steps_to_gau_loose,
            "mean_force_rmse_at_reference_ev_per_a": (
                self.mean_force_rmse_at_reference_ev_per_a
            ),
            "per_endpoint": [m.as_dict() for m in self.per_endpoint],
            # Explicitly never rank by energy
            "energy_loss_used_for_ranking": False,
            "final_model_selected": False,
        }


def forces_hartree_bohr_to_ev_angstrom(
    forces_h_per_b: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    """Convert teacher forces (Hartree/Bohr) → eV/Å."""

    out: list[tuple[float, float, float]] = []
    for row in forces_h_per_b:
        if len(row) != 3:
            raise PreScreenError("force row must have 3 components")
        out.append(
            (
                float(row[0]) * FORCE_H_PER_B_TO_EV_PER_A,
                float(row[1]) * FORCE_H_PER_B_TO_EV_PER_A,
                float(row[2]) * FORCE_H_PER_B_TO_EV_PER_A,
            )
        )
    return tuple(out)


# Nested list rows or (N, 3) ndarray — both accepted by coordinate helpers.
CoordLike = Sequence[Sequence[float]] | np.ndarray


def _as_coords(rows: CoordLike, *, label: str) -> np.ndarray:
    arr = np.asarray(rows, dtype=np.float64)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise PreScreenError(f"{label}: expected (N, 3) coordinates, got {arr.shape}")
    if not np.isfinite(arr).all():
        raise PreScreenError(f"{label}: non-finite coordinates")
    return arr


def kabsch_rmsd(coords_a: CoordLike, coords_b: CoordLike) -> float:
    """Kabsch RMSD (Å) after optimal rotation; translation removed via COM.

    Maps A onto B. Reflection-corrected SVD. Empty set → 0.0.
    Accepts nested sequences or ``(N, 3)`` ndarrays.
    """

    a = _as_coords(coords_a, label="kabsch A")
    b = _as_coords(coords_b, label="kabsch B")
    if a.shape[0] != b.shape[0]:
        raise PreScreenError(
            f"kabsch size mismatch: {a.shape[0]} vs {b.shape[0]}"
        )
    n = int(a.shape[0])
    if n == 0:
        return 0.0
    if n == 1:
        # Pure translation — RMSD after COM removal is 0
        return 0.0

    a_c = a - a.mean(axis=0)
    b_c = b - b.mean(axis=0)
    # Kabsch: R maps column vectors of A onto B  (R @ a_col ≈ b_col)
    h = a_c.T @ b_c
    u, _s, vt = np.linalg.svd(h)
    v = vt.T
    d = float(np.linalg.det(v @ u.T))
    sign = 1.0 if d >= 0.0 else -1.0
    corr = np.diag([1.0, 1.0, sign])
    r = v @ corr @ u.T
    # Row-vector form: a_aligned = a_c @ R.T
    a_aligned = a_c @ r.T
    diff = a_aligned - b_c
    return float(np.sqrt(float(np.sum(diff * diff)) / float(n)))


def heavy_atom_indices(elements: Sequence[str]) -> list[int]:
    """Indices of non-hydrogen atoms; fall back to all atoms if none."""

    idx = [i for i, el in enumerate(elements) if str(el).upper() != "H"]
    if not idx:
        return list(range(len(elements)))
    return idx


def heavy_atom_kabsch_rmsd(
    coords_a: CoordLike,
    coords_b: CoordLike,
    elements: Sequence[str],
) -> float:
    """Heavy-atom Kabsch RMSD (Å) between two geometries."""

    a = _as_coords(coords_a, label="heavy A")
    b = _as_coords(coords_b, label="heavy B")
    if a.shape[0] != len(elements) or b.shape[0] != len(elements):
        raise PreScreenError(
            f"coord/element length mismatch: {a.shape[0]}, {b.shape[0]}, "
            f"{len(elements)}"
        )
    idx = heavy_atom_indices(elements)
    return kabsch_rmsd(a[idx], b[idx])


def force_rmse(
    pred_ev_per_a: CoordLike,
    ref_ev_per_a: CoordLike,
) -> float:
    """RMSE of Cartesian forces (eV/Å) over all atoms and components."""

    p = np.asarray(pred_ev_per_a, dtype=np.float64)
    r = np.asarray(ref_ev_per_a, dtype=np.float64)
    if p.shape != r.shape or p.ndim != 2 or p.shape[1] != 3:
        raise PreScreenError(
            f"force shape mismatch: pred={p.shape} ref={r.shape}"
        )
    if p.size == 0:
        return 0.0
    if not np.isfinite(p).all() or not np.isfinite(r).all():
        raise PreScreenError("non-finite forces in RMSE")
    diff = p - r
    return float(np.sqrt(float(np.mean(diff * diff))))


def _charge_mult(endpoint: str, frame: Mapping[str, Any]) -> tuple[int, int]:
    if "charge" in frame and "multiplicity" in frame:
        return int(frame["charge"]), int(frame["multiplicity"])
    if endpoint == "cation":
        return CATION_CHARGE, CATION_MULTIPLICITY
    if endpoint == "neutral":
        return NEUTRAL_CHARGE, NEUTRAL_MULTIPLICITY
    raise PreScreenError(f"invalid endpoint: {endpoint}")


def _coords_tuple(
    rows: Sequence[Sequence[float]],
) -> tuple[tuple[float, float, float], ...]:
    out: list[tuple[float, float, float]] = []
    for row in rows:
        if len(row) != 3:
            raise PreScreenError("coordinate row must have 3 components")
        out.append((float(row[0]), float(row[1]), float(row[2])))
    return tuple(out)


def load_teacher_endpoint_reference(endpoint_dir: Path) -> TeacherEndpointReference:
    """Load start (frame_0000) + terminal (is_terminal) from a teacher endpoint dir."""

    if not endpoint_dir.is_dir():
        raise PreScreenError(f"teacher endpoint dir missing: {endpoint_dir}")

    start_path = endpoint_dir / "frame_0000.json"
    if not start_path.is_file():
        raise PreScreenError(f"missing start frame: {start_path}")
    start, _ = load_json_object(start_path)

    terminal_payload: dict[str, Any] | None = None
    terminal_index: int | None = None
    for path in sorted(endpoint_dir.glob("frame_*.json")):
        payload, _ = load_json_object(path)
        if payload.get("is_terminal") is True:
            terminal_payload = payload
            stem = path.stem  # frame_NNNN
            try:
                terminal_index = int(stem.split("_", 1)[1])
            except (IndexError, ValueError) as exc:
                raise PreScreenError(f"bad terminal frame name: {path.name}") from exc
            break
    if terminal_payload is None or terminal_index is None:
        raise PreScreenError(f"no is_terminal frame under {endpoint_dir}")

    root_id = str(start.get("root_id") or terminal_payload.get("root_id") or "")
    endpoint = str(start.get("endpoint") or terminal_payload.get("endpoint") or "")
    if not root_id or endpoint not in ENDPOINTS:
        # Infer from path: .../<root_id>/<endpoint>/
        endpoint = endpoint_dir.name
        root_id = endpoint_dir.parent.name
        if endpoint not in ENDPOINTS:
            raise PreScreenError(
                f"cannot resolve root/endpoint for {endpoint_dir}"
            )

    elements_raw = start.get("elements") or terminal_payload.get("elements")
    if not isinstance(elements_raw, list) or not elements_raw:
        raise PreScreenError(f"missing elements in teacher frames: {endpoint_dir}")
    elements = tuple(str(e) for e in elements_raw)

    start_coords = start.get("coordinates_angstrom")
    ref_coords = terminal_payload.get("coordinates_angstrom")
    if not isinstance(start_coords, list) or not isinstance(ref_coords, list):
        raise PreScreenError(f"missing coordinates_angstrom under {endpoint_dir}")

    forces_h = terminal_payload.get("forces_hartree_per_bohr")
    if not isinstance(forces_h, list):
        # Fallback: negate gradient if forces missing
        grad = terminal_payload.get("gradient_hartree_per_bohr")
        if not isinstance(grad, list):
            raise PreScreenError(
                f"terminal frame missing forces/gradient: {endpoint_dir}"
            )
        forces_h = [[-float(c) for c in row] for row in grad]

    charge, mult = _charge_mult(endpoint, start)
    return TeacherEndpointReference(
        root_id=root_id,
        endpoint=endpoint,
        elements=elements,
        start_coordinates_angstrom=_coords_tuple(start_coords),
        reference_coordinates_angstrom=_coords_tuple(ref_coords),
        reference_forces_ev_per_a=forces_hartree_bohr_to_ev_angstrom(forces_h),
        charge=charge,
        multiplicity=mult,
        start_frame_index=0,
        reference_frame_index=terminal_index,
    )


def load_teacher_references_for_batch(
    layout: GenerationLayout,
    batch_id: str,
    root_ids: Sequence[str],
) -> list[TeacherEndpointReference]:
    """Load cation+neutral references for each root under teacher_gpu_g00N/."""

    teacher_root = layout.teacher_batch_dir(batch_id)
    refs: list[TeacherEndpointReference] = []
    for root_id in root_ids:
        for endpoint in ENDPOINTS:
            ep_dir = teacher_root / root_id / endpoint
            refs.append(load_teacher_endpoint_reference(ep_dir))
    return refs


def _extract_model_forces_at_reference(
    engine: GauLooseEngine,
    aim: Mapping[str, Any],
    *,
    reference: TeacherEndpointReference,
    checkpoint_id: str,
) -> Sequence[Sequence[float]]:
    """Model forces (eV/Å) at teacher terminal geometry."""

    direct = aim.get("forces_at_reference_ev_per_a")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)):
        return direct

    # Optional duck-typed force evaluator on the same engine object
    forces_fn = getattr(engine, "forces_at_geometry", None)
    if callable(forces_fn):
        out = forces_fn(
            root_id=reference.root_id,
            endpoint=reference.endpoint,
            elements=reference.elements,
            coordinates=reference.reference_coordinates_angstrom,
            charge=reference.charge,
            multiplicity=reference.multiplicity,
            checkpoint_id=checkpoint_id,
        )
        if not isinstance(out, Mapping):
            raise PreScreenError("forces_at_geometry must return a mapping")
        forces = out.get("forces_ev_angstrom") or out.get("forces_at_reference_ev_per_a")
        if not isinstance(forces, Sequence) or isinstance(forces, (str, bytes)):
            raise PreScreenError(
                "forces_at_geometry missing forces_ev_angstrom / "
                "forces_at_reference_ev_per_a"
            )
        return forces

    raise PreScreenError(
        "engine must provide forces_at_reference_ev_per_a on optimize result "
        "or implement forces_at_geometry(...)"
    )


def evaluate_endpoint(
    engine: GauLooseEngine,
    *,
    reference: TeacherEndpointReference,
    checkpoint_id: str,
) -> EndpointScreenMetrics:
    """Run AIMNet2 GAU_LOOSE from teacher start; score vs teacher terminal."""

    aim = engine.optimize_to_gau_loose(
        root_id=reference.root_id,
        endpoint=reference.endpoint,
        elements=reference.elements,
        coordinates=reference.start_coordinates_angstrom,
        charge=reference.charge,
        multiplicity=reference.multiplicity,
        checkpoint_id=checkpoint_id,
    )

    identity_ok = all(
        (
            bool(aim.get("atom_identity_preserved", True)),
            bool(aim.get("charge_multiplicity_preserved", True)),
            bool(aim.get("coordinates_finite", True)),
        )
    )
    # Same criterion as scientific_validation: engine-reported topology_valid
    topology_preserved = bool(aim.get("topology_valid", True))
    gau_loose_converged = bool(aim.get("converged", False))
    steps = int(aim.get("steps") or 0)

    notes: list[str] = []
    coords = aim.get("coordinates")
    if coords is None:
        notes.append("missing_optimized_coordinates")
        rmsd = math.inf
        identity_ok = False
    else:
        try:
            rmsd = heavy_atom_kabsch_rmsd(
                coords,
                reference.reference_coordinates_angstrom,
                reference.elements,
            )
        except PreScreenError as exc:
            notes.append(f"rmsd_error:{exc}")
            rmsd = math.inf
            identity_ok = False

    try:
        pred_f = _extract_model_forces_at_reference(
            engine,
            aim,
            reference=reference,
            checkpoint_id=checkpoint_id,
        )
        f_rmse = force_rmse(pred_f, reference.reference_forces_ev_per_a)
    except PreScreenError as exc:
        notes.append(f"force_rmse_error:{exc}")
        f_rmse = math.inf

    hard = bool(identity_ok and topology_preserved and gau_loose_converged)
    return EndpointScreenMetrics(
        root_id=reference.root_id,
        endpoint=reference.endpoint,
        checkpoint_id=checkpoint_id,
        identity_ok=identity_ok,
        topology_preserved=topology_preserved,
        gau_loose_converged=gau_loose_converged,
        rmsd_to_reference_angstrom=float(rmsd),
        aimnet2_steps_to_gau_loose=steps,
        force_rmse_at_reference_ev_per_a=float(f_rmse),
        hard_gates_passed=hard,
        notes=tuple(notes),
    )


def _mean_or_inf(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return math.inf
    return float(sum(finite) / len(finite))


def aggregate_candidate(
    candidate: CheckpointCandidate,
    per_endpoint: Sequence[EndpointScreenMetrics],
) -> CandidateScreenResult:
    """Aggregate per-endpoint metrics; hard gates require all endpoints pass."""

    if not per_endpoint:
        raise PreScreenError(
            f"no endpoint metrics for checkpoint {candidate.checkpoint_id}"
        )
    identity_ok = all(m.identity_ok for m in per_endpoint)
    topology_ok = all(m.topology_preserved for m in per_endpoint)
    converged_ok = all(m.gau_loose_converged for m in per_endpoint)
    hard = identity_ok and topology_ok and converged_ok
    mean_rmsd = _mean_or_inf([m.rmsd_to_reference_angstrom for m in per_endpoint])
    mean_steps = _mean_or_inf(
        [float(m.aimnet2_steps_to_gau_loose) for m in per_endpoint]
    )
    mean_f = _mean_or_inf(
        [m.force_rmse_at_reference_ev_per_a for m in per_endpoint]
    )
    # Sort key: hard pass first (0), then RMSD ↑, steps ↑, force RMSE ↑
    rank_key = (
        0 if hard else 1,
        mean_rmsd,
        mean_steps,
        mean_f,
        candidate.run_id,
        candidate.seed,
        candidate.epoch,
        candidate.checkpoint_id,
    )
    return CandidateScreenResult(
        candidate=candidate,
        per_endpoint=list(per_endpoint),
        identity_ok=identity_ok,
        topology_preserved=topology_ok,
        gau_loose_converged=converged_ok,
        hard_gates_passed=hard,
        mean_rmsd_to_reference_angstrom=mean_rmsd,
        mean_aimnet2_steps_to_gau_loose=mean_steps,
        mean_force_rmse_at_reference_ev_per_a=mean_f,
        rank_key=rank_key,
    )


def rank_candidates(
    results: Sequence[CandidateScreenResult],
) -> list[CandidateScreenResult]:
    """Hard gates → RMSD ↑ → steps ↑ → force RMSE ↑. Never energy loss."""

    return sorted(results, key=lambda r: r.rank_key)


def screen_checkpoint(
    engine: GauLooseEngine,
    candidate: CheckpointCandidate,
    references: Sequence[TeacherEndpointReference],
) -> CandidateScreenResult:
    """Evaluate one checkpoint on all reference endpoints."""

    if not references:
        raise PreScreenError("references must be non-empty")
    per = [
        evaluate_endpoint(
            engine,
            reference=ref,
            checkpoint_id=candidate.checkpoint_id,
        )
        for ref in references
    ]
    return aggregate_candidate(candidate, per)


def _normalize_candidate(raw: Mapping[str, Any] | CheckpointCandidate) -> CheckpointCandidate:
    if isinstance(raw, CheckpointCandidate):
        return raw
    seed = raw.get("seed")
    epoch = raw.get("epoch")
    if type(seed) is not int or type(epoch) is not int:
        raise PreScreenError(f"candidate needs int seed/epoch: {raw}")
    run_id = str(raw.get("run_id") or "unknown_run")
    ckpt = raw.get("checkpoint_id")
    if not ckpt:
        ckpt = f"{run_id}_seed_{seed}_epoch_{epoch:04d}"
    weight = raw.get("weight_path")
    return CheckpointCandidate(
        checkpoint_id=str(ckpt),
        run_id=run_id,
        seed=int(seed),
        epoch=int(epoch),
        weight_path=str(weight) if weight is not None else None,
    )


def run_pre_screen_campaign(
    *,
    candidates: Sequence[Mapping[str, Any] | CheckpointCandidate],
    references: Sequence[TeacherEndpointReference],
    engine: GauLooseEngine | None = None,
    engine_factory: Callable[[CheckpointCandidate], GauLooseEngine] | None = None,
    layout: GenerationLayout | None = None,
    batch_id: str = "g001",
    screen_id: str | None = None,
    shortlist_count: int = DEFAULT_SHORTLIST_COUNT,
    write: bool = True,
    output_dir: Path | None = None,
) -> dict[str, Any]:
    """Screen candidates with zero DFT; write campaign receipt.

    Ranking is fixed: hard gates → RMSD ↑ → steps ↑ → force RMSE ↑.
    Energy loss is never used.
    """

    if not candidates:
        raise PreScreenError("candidates must be non-empty")
    if not references:
        raise PreScreenError("references must be non-empty")
    if engine is None and engine_factory is None:
        raise PreScreenError("engine or engine_factory required")

    cand_list = [_normalize_candidate(c) for c in candidates]
    results: list[CandidateScreenResult] = []
    for cand in cand_list:
        eng = engine_factory(cand) if engine_factory is not None else engine
        assert eng is not None
        results.append(screen_checkpoint(eng, cand, references))

    ranked = rank_candidates(results)
    shortlist_n = max(0, int(shortlist_count))
    shortlist = [r for r in ranked if r.hard_gates_passed][:shortlist_n]
    # If fewer hard-pass than requested, do not fill with hard-fail candidates
    # (sci-val must only see gate-passing pre-screen shortlist).

    sid = screen_id
    if sid is None:
        run_ids = {c.run_id for c in cand_list}
        sid = next(iter(run_ids)) if len(run_ids) == 1 else "campaign"

    out_dir: Path | None = output_dir
    if out_dir is None and layout is not None:
        out_dir = layout.pre_screen_batch_dir(batch_id) / sid

    campaign: dict[str, Any] = {
        "schema": CAMPAIGN_SCHEMA,
        "mindmap_step": MINDMAP_STEP,
        "batch_id": batch_id,
        "screen_id": sid,
        "status": "PRE_SCREEN_PASS" if shortlist else "PRE_SCREEN_EMPTY_SHORTLIST",
        "final_model_selected": False,
        "selection_authority": SELECTION_AUTHORITY,
        "scientific_validation_required_before_final_selection": True,
        "energy_loss_used_for_ranking": False,
        "ranking_rule": (
            "hard_gates_all_pass -> mean_rmsd_asc -> mean_steps_asc "
            "-> mean_force_rmse_asc"
        ),
        "shortlist_count_requested": shortlist_n,
        "candidate_count": len(ranked),
        "hard_gates_passed_count": sum(1 for r in ranked if r.hard_gates_passed),
        "reference_endpoint_count": len(references),
        "references": [
            {
                "root_id": r.root_id,
                "endpoint": r.endpoint,
                "n_atoms": len(r.elements),
                "reference_frame_index": r.reference_frame_index,
            }
            for r in references
        ],
        "ranked": [r.as_dict() for r in ranked],
        "shortlist": [r.as_dict() for r in shortlist],
        "shortlist_checkpoint_ids": [r.candidate.checkpoint_id for r in shortlist],
        "notes": [
            "zero-DFT AIMNet2 GAU_LOOSE pre-screen only",
            "not final model selection (mindmap steps 8–9 still required)",
            "frame-level energy loss is forbidden for ranking",
        ],
    }

    if write and out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)
        receipt_path = out_dir / "screen_campaign.json"
        write_json(receipt_path, campaign, overwrite=True)
        campaign["receipt_path"] = str(receipt_path)
    elif write and out_dir is None:
        campaign["receipt_path"] = None
        campaign["notes"] = list(campaign["notes"]) + [
            "write=True but no layout/output_dir; receipt not written"
        ]

    return campaign


# ---------------------------------------------------------------------------
# Deterministic fake engine for unit tests / dry-run wiring
# ---------------------------------------------------------------------------


@dataclass
class SimulatedPreScreenEngine:
    """Injectable GAU_LOOSE engine with fixed per-checkpoint outcomes.

    ``outcomes`` keys are ``checkpoint_id`` → endpoint metrics override dict, or
    a nested map ``(root_id, endpoint)`` → override. Override fields:
      coordinates, steps, converged, topology_valid, atom_identity_preserved,
      forces_at_reference_ev_per_a, charge_multiplicity_preserved,
      coordinates_finite.
    """

    outcomes: Mapping[str, Any] = field(default_factory=dict)
    default_steps: int = 10
    default_converged: bool = True
    # Non-rigid deformation of atom 0 along x (Å). Pure COM translation would
    # vanish under Kabsch, so this stretches relative geometry on purpose.
    default_atom0_dx: float = 0.0

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
    ) -> dict[str, Any]:
        override = self._lookup(checkpoint_id, root_id, endpoint)
        coords = override.get("coordinates")
        if coords is None:
            base = np.asarray(coordinates, dtype=np.float64).copy()
            dx = float(override.get("atom0_dx", self.default_atom0_dx))
            if base.shape[0] > 0 and dx != 0.0:
                base[0, 0] += dx
            coords = base.tolist()
        forces = override.get("forces_at_reference_ev_per_a")
        if forces is None:
            # Zero forces at reference by default (perfect match if teacher ~0)
            forces = [[0.0, 0.0, 0.0] for _ in elements]
        return {
            "converged": bool(override.get("converged", self.default_converged)),
            "steps": int(override.get("steps", self.default_steps)),
            "coordinates": coords,
            "energy_ev": float(override.get("energy_ev", -100.0)),  # never ranked
            "wall_seconds": 0.0,
            "atom_identity_preserved": bool(
                override.get("atom_identity_preserved", True)
            ),
            "topology_valid": bool(override.get("topology_valid", True)),
            "coordinates_finite": bool(override.get("coordinates_finite", True)),
            "charge_multiplicity_preserved": bool(
                override.get("charge_multiplicity_preserved", True)
            ),
            "forces_at_reference_ev_per_a": forces,
            "checkpoint_id": checkpoint_id,
            "root_id": root_id,
            "endpoint": endpoint,
            "elements": list(elements),
            "charge": charge,
            "multiplicity": multiplicity,
        }

    def _lookup(
        self, checkpoint_id: str, root_id: str, endpoint: str
    ) -> dict[str, Any]:
        raw = self.outcomes.get(checkpoint_id)
        if raw is None:
            return {}
        if isinstance(raw, Mapping) and (
            (root_id, endpoint) in raw or f"{root_id}/{endpoint}" in raw
        ):
            hit = raw.get((root_id, endpoint))
            if hit is None:
                hit = raw.get(f"{root_id}/{endpoint}")
            return dict(hit) if isinstance(hit, Mapping) else {}
        if isinstance(raw, Mapping):
            # Treat as flat override for all endpoints of this checkpoint
            # unless it looks nested with endpoint keys only
            if any(k in ENDPOINTS for k in raw):
                sub = raw.get(endpoint)
                return dict(sub) if isinstance(sub, Mapping) else {}
            return dict(raw)
        return {}
