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
# Multi-start replica measurement (T9 §3.1 basin evidence, 2026-08-06).
# replicas=1 keeps the historical single-shot path bit-identical.
DEFAULT_REPLICAS: Final = 1
DEFAULT_REPLICA_EPSILON_ANGSTROM: Final = 1e-4
DEFAULT_BASIN_GAP_ANGSTROM: Final = 0.01
FORCE_REPLICA_SPREAD_TOL: Final = 1e-12

# What a screened weight *is*. The epoch-zero official base weight is the
# yardstick required by NUMERIC_CALIBRATION_V001.epoch_zero_non_regression_rule,
# not a competitor: it stays in ``ranked`` but never consumes a shortlist slot
# (20260804 sci-val plan P0-2).
ROUTE_KIND_FINETUNED: Final = "finetuned_checkpoint"
ROUTE_KIND_EPOCH_ZERO: Final = "epoch_zero"
ROUTE_KINDS: Final = (ROUTE_KIND_FINETUNED, ROUTE_KIND_EPOCH_ZERO)

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
    """One weight to pre-screen.

    ``route_kind`` defaults to a fine-tuned checkpoint; pass
    :data:`ROUTE_KIND_EPOCH_ZERO` for the official base weight so it is scored
    and ranked but kept out of the sci-val shortlist.
    """

    checkpoint_id: str
    run_id: str
    seed: int
    epoch: int
    weight_path: str | None = None
    route_kind: str = ROUTE_KIND_FINETUNED

    def validate(self) -> None:
        if self.route_kind not in ROUTE_KINDS:
            raise PreScreenError(
                f"unknown route_kind {self.route_kind!r}; expected one of {ROUTE_KINDS}"
            )

    @property
    def is_epoch_zero(self) -> bool:
        return self.route_kind == ROUTE_KIND_EPOCH_ZERO


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
    # Multi-replica fields (only meaningful when replicas > 1; omitted from
    # as_dict when replicas == 1 so single-shot receipts stay byte-stable).
    replicas: int = 1
    replica_epsilon_angstrom: float | None = None
    basin_gap_angstrom: float | None = None
    modal_basin_fraction: float | None = None
    basin_count: int | None = None
    deterministic: bool | None = None
    rmsd_p10: float | None = None
    rmsd_p90: float | None = None
    steps_min: float | None = None
    steps_max: float | None = None
    basin_clusters: list[dict[str, Any]] | None = None
    replica_mean_rmsds: list[float] | None = None
    replica_mean_steps: list[float] | None = None

    def as_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "checkpoint_id": self.candidate.checkpoint_id,
            "run_id": self.candidate.run_id,
            "seed": self.candidate.seed,
            "epoch": self.candidate.epoch,
            "weight_path": self.candidate.weight_path,
            "route_kind": self.candidate.route_kind,
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
        if int(self.replicas) > 1:
            out.update(
                {
                    "replicas": int(self.replicas),
                    "replica_epsilon_angstrom": self.replica_epsilon_angstrom,
                    "basin_gap_angstrom": self.basin_gap_angstrom,
                    "modal_basin_fraction": self.modal_basin_fraction,
                    "basin_count": self.basin_count,
                    "deterministic": self.deterministic,
                    "rmsd_p10": self.rmsd_p10,
                    "rmsd_p90": self.rmsd_p90,
                    "steps_min": self.steps_min,
                    "steps_max": self.steps_max,
                    "basin_clusters": self.basin_clusters,
                    "replica_mean_rmsds": self.replica_mean_rmsds,
                    "replica_mean_steps": self.replica_mean_steps,
                }
            )
        return out


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
    teacher_batch_dir: Path | None = None,
) -> list[TeacherEndpointReference]:
    """Load cation+neutral references for each root under teacher_gpu_g00N/.

    ``teacher_batch_dir`` pins the reference set to a specific directory
    (read-only). Needed to reproduce an earlier screen while the canonical
    ``teacher_gpu_g00N/`` is being recomputed: RMSD is measured *against* these
    geometries, so mixing reference sets makes screens incomparable. Historical
    ``frame_count == 2`` products stay read-only (AGENTS T5).
    """

    teacher_root = (
        Path(teacher_batch_dir)
        if teacher_batch_dir is not None
        else layout.teacher_batch_dir(batch_id)
    )
    if not teacher_root.is_dir():
        raise PreScreenError(f"teacher reference dir missing: {teacher_root}")
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
    """Aggregate per-endpoint metrics; hard gates require all endpoints pass.

    Ranking key (ascending; hard-fail last)::

        hard_gates → mean_force_rmse → mean_steps → mean_rmsd
        → run_id / seed / epoch / checkpoint_id

    Force RMSE is the primary soft key (T1 wording + cross-device stability on
    g001 pre-screen evidence). See
    ``docs/science/T9_OPERATIONAL_20260805_no_gain_vs_epoch0.md`` §3.
    Energy loss never enters the key.
    """

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
    # Sort key: hard pass first (0), then force RMSE ↑, steps ↑, RMSD ↑
    # (T9_OPERATIONAL §3: force more stable across devices than RMSD/steps)
    rank_key = (
        0 if hard else 1,
        mean_f,
        mean_steps,
        mean_rmsd,
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
    """Hard gates → force RMSE ↑ → steps ↑ → RMSD ↑. Never energy loss."""

    return sorted(results, key=lambda r: r.rank_key)


def _percentile_sorted(sorted_vals: Sequence[float], q: float) -> float:
    """Linear interpolation percentile; ``sorted_vals`` non-empty ascending."""

    if not sorted_vals:
        return math.inf
    if len(sorted_vals) == 1:
        return float(sorted_vals[0])
    q = min(1.0, max(0.0, float(q)))
    pos = q * (len(sorted_vals) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return float(sorted_vals[lo])
    w = pos - lo
    return float(sorted_vals[lo] * (1.0 - w) + sorted_vals[hi] * w)


def _median(values: Sequence[float]) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    if not finite:
        return math.inf
    s = sorted(finite)
    n = len(s)
    mid = n // 2
    if n % 2 == 1:
        return float(s[mid])
    return float(0.5 * (s[mid - 1] + s[mid]))


def cluster_scalar_values(
    values: Sequence[float],
    *,
    gap: float = DEFAULT_BASIN_GAP_ANGSTROM,
) -> list[list[float]]:
    """Chain-cluster sorted values; new cluster when adjacent gap > ``gap``."""

    xs = sorted(float(v) for v in values if math.isfinite(float(v)))
    if not xs:
        return []
    clusters: list[list[float]] = [[xs[0]]]
    for x in xs[1:]:
        if x - clusters[-1][-1] > float(gap):
            clusters.append([x])
        else:
            clusters[-1].append(x)
    return clusters


def basin_statistics(
    rmsds: Sequence[float],
    *,
    gap: float = DEFAULT_BASIN_GAP_ANGSTROM,
) -> dict[str, Any]:
    """Basin labels from replica mean-RMSD samples (gap clustering)."""

    clusters = cluster_scalar_values(rmsds, gap=gap)
    n = sum(len(c) for c in clusters)
    if n == 0:
        return {
            "basin_count": 0,
            "modal_basin_fraction": 0.0,
            "deterministic": False,
            "basin_clusters": [],
        }
    sizes = [len(c) for c in clusters]
    modal_n = max(sizes)
    basin_clusters = [
        {
            "n": len(c),
            "fraction": len(c) / n,
            "center": float(sum(c) / len(c)),
            "rmsd_min": float(min(c)),
            "rmsd_max": float(max(c)),
        }
        for c in clusters
    ]
    # stable order: largest basin first, then by center
    basin_clusters.sort(key=lambda d: (-int(d["n"]), float(d["center"])))
    frac = modal_n / n
    return {
        "basin_count": len(clusters),
        "modal_basin_fraction": float(frac),
        "deterministic": frac == 1.0,
        "basin_clusters": basin_clusters,
    }


def replica_rng_seed(
    candidate: CheckpointCandidate,
    *,
    replica_index: int,
    base_seed: int = 0,
) -> int:
    """Deterministic per-(candidate, replica) RNG seed."""

    # Keep non-negative 31-bit for numpy Generator.
    raw = (
        int(base_seed)
        + int(candidate.seed) * 1_000_003
        + int(candidate.epoch) * 97
        + int(replica_index) * 1_000_033
        + sum(ord(ch) for ch in candidate.checkpoint_id) * 13
    )
    return int(raw % 2_147_483_647)


def aggregate_replica_results(
    candidate: CheckpointCandidate,
    replica_results: Sequence[CandidateScreenResult],
    *,
    replica_epsilon_angstrom: float,
    basin_gap_angstrom: float = DEFAULT_BASIN_GAP_ANGSTROM,
    force_spread_tol: float = FORCE_REPLICA_SPREAD_TOL,
) -> CandidateScreenResult:
    """Collapse N single-shot results into one ranked measurement.

    - force RMSE: must be identical across replicas (fail closed otherwise);
      uses the common value (still single-shot physics at the reference geom).
    - steps / RMSD ranking values: **median** across replicas.
    - basin stats from the N mean-RMSD samples (report only; not in rank_key).
    - hard_gates: all replicas must hard-pass (conservative).
    """

    if not replica_results:
        raise PreScreenError(
            f"no replica results for checkpoint {candidate.checkpoint_id}"
        )
    n = len(replica_results)
    forces = [float(r.mean_force_rmse_at_reference_ev_per_a) for r in replica_results]
    finite_f = [f for f in forces if math.isfinite(f)]
    if len(finite_f) != n:
        raise PreScreenError(
            f"non-finite force RMSE among replicas for {candidate.checkpoint_id!r}"
        )
    spread = max(finite_f) - min(finite_f)
    if spread > float(force_spread_tol):
        raise PreScreenError(
            f"force RMSE replica spread {spread:.3e} > {force_spread_tol} "
            f"for checkpoint {candidate.checkpoint_id!r} "
            f"(forces={finite_f!r}); model forward may be non-deterministic"
        )
    mean_f = float(finite_f[0])

    rmsds = [float(r.mean_rmsd_to_reference_angstrom) for r in replica_results]
    steps = [float(r.mean_aimnet2_steps_to_gau_loose) for r in replica_results]
    med_rmsd = _median(rmsds)
    med_steps = _median(steps)
    s_rmsd = sorted(v for v in rmsds if math.isfinite(v))
    s_steps = sorted(v for v in steps if math.isfinite(v))
    basin = basin_statistics(rmsds, gap=basin_gap_angstrom)

    hard = all(r.hard_gates_passed for r in replica_results)
    identity_ok = all(r.identity_ok for r in replica_results)
    topology_ok = all(r.topology_preserved for r in replica_results)
    converged_ok = all(r.gau_loose_converged for r in replica_results)

    # Representative per_endpoint: first replica (diagnostic only).
    per_endpoint = list(replica_results[0].per_endpoint)

    rank_key = (
        0 if hard else 1,
        mean_f,
        med_steps,
        med_rmsd,
        candidate.run_id,
        candidate.seed,
        candidate.epoch,
        candidate.checkpoint_id,
    )
    return CandidateScreenResult(
        candidate=candidate,
        per_endpoint=per_endpoint,
        identity_ok=identity_ok,
        topology_preserved=topology_ok,
        gau_loose_converged=converged_ok,
        hard_gates_passed=hard,
        mean_rmsd_to_reference_angstrom=med_rmsd,
        mean_aimnet2_steps_to_gau_loose=med_steps,
        mean_force_rmse_at_reference_ev_per_a=mean_f,
        rank_key=rank_key,
        replicas=n,
        replica_epsilon_angstrom=float(replica_epsilon_angstrom),
        basin_gap_angstrom=float(basin_gap_angstrom),
        modal_basin_fraction=float(basin["modal_basin_fraction"]),
        basin_count=int(basin["basin_count"]),
        deterministic=bool(basin["deterministic"]),
        rmsd_p10=_percentile_sorted(s_rmsd, 0.10) if s_rmsd else math.inf,
        rmsd_p90=_percentile_sorted(s_rmsd, 0.90) if s_rmsd else math.inf,
        steps_min=float(s_steps[0]) if s_steps else math.inf,
        steps_max=float(s_steps[-1]) if s_steps else math.inf,
        basin_clusters=list(basin["basin_clusters"]),
        replica_mean_rmsds=list(rmsds),
        replica_mean_steps=list(steps),
    )


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


def screen_checkpoint_replicas(
    engine: GauLooseEngine,
    candidate: CheckpointCandidate,
    references: Sequence[TeacherEndpointReference],
    *,
    replicas: int = DEFAULT_REPLICAS,
    replica_epsilon_angstrom: float = DEFAULT_REPLICA_EPSILON_ANGSTROM,
    basin_gap_angstrom: float = DEFAULT_BASIN_GAP_ANGSTROM,
    base_rng_seed: int = 0,
) -> CandidateScreenResult:
    """Single-shot (replicas=1) or multi-start statistical pre-screen.

    When ``replicas == 1`` this is exactly :func:`screen_checkpoint` (no
    perturbation). When ``replicas > 1``, start geometries are perturbed with
    :func:`~nhc_deprot.pipeline.basin_perturbation.perturb_start_geometry` and
    metrics are aggregated by :func:`aggregate_replica_results`.
    """

    n = int(replicas)
    if n < 1:
        raise PreScreenError(f"replicas must be >= 1, got {n}")
    if n == 1:
        return screen_checkpoint(engine, candidate, references)

    from nhc_deprot.pipeline.basin_perturbation import perturb_start_geometry

    eps = float(replica_epsilon_angstrom)
    if eps < 0.0:
        raise PreScreenError(f"replica_epsilon_angstrom must be >= 0, got {eps}")

    replica_results: list[CandidateScreenResult] = []
    for i in range(n):
        seed_i = replica_rng_seed(
            candidate, replica_index=i, base_seed=base_rng_seed
        )
        pert_refs: list[TeacherEndpointReference] = []
        for j, ref in enumerate(references):
            pr = perturb_start_geometry(
                ref,
                epsilon_angstrom=eps,
                rng_seed=seed_i + (j + 1) * 10_007,
            )
            pert_refs.append(pr.reference)
        replica_results.append(screen_checkpoint(engine, candidate, pert_refs))

    return aggregate_replica_results(
        candidate,
        replica_results,
        replica_epsilon_angstrom=eps,
        basin_gap_angstrom=float(basin_gap_angstrom),
    )


def _normalize_candidate(raw: Mapping[str, Any] | CheckpointCandidate) -> CheckpointCandidate:
    if isinstance(raw, CheckpointCandidate):
        raw.validate()
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
    candidate = CheckpointCandidate(
        checkpoint_id=str(ckpt),
        run_id=run_id,
        seed=int(seed),
        epoch=int(epoch),
        weight_path=str(weight) if weight is not None else None,
        route_kind=str(raw.get("route_kind") or ROUTE_KIND_FINETUNED),
    )
    candidate.validate()
    return candidate


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
    replicas: int = DEFAULT_REPLICAS,
    replica_epsilon_angstrom: float = DEFAULT_REPLICA_EPSILON_ANGSTROM,
    basin_gap_angstrom: float = DEFAULT_BASIN_GAP_ANGSTROM,
    replica_base_rng_seed: int = 0,
) -> dict[str, Any]:
    """Screen candidates with zero DFT; write campaign receipt.

    Ranking is fixed: hard gates → force RMSE ↑ → steps ↑ → RMSD ↑.
    Energy loss is never used. See T9_OPERATIONAL §3 for why force leads.

    ``replicas`` (default 1): when >1, each candidate is screened from N
    perturbed starts; steps/RMSD ranking values are **medians**; force is
    single-shot at the reference geometry (asserted identical across
    replicas). Basin statistics are reported but do not enter ``rank_key``.
    """

    if not candidates:
        raise PreScreenError("candidates must be non-empty")
    if not references:
        raise PreScreenError("references must be non-empty")
    if engine is None and engine_factory is None:
        raise PreScreenError("engine or engine_factory required")
    n_rep = int(replicas)
    if n_rep < 1:
        raise PreScreenError(f"replicas must be >= 1, got {n_rep}")

    cand_list = [_normalize_candidate(c) for c in candidates]
    results: list[CandidateScreenResult] = []
    for cand in cand_list:
        eng = engine_factory(cand) if engine_factory is not None else engine
        assert eng is not None
        results.append(
            screen_checkpoint_replicas(
                eng,
                cand,
                references,
                replicas=n_rep,
                replica_epsilon_angstrom=float(replica_epsilon_angstrom),
                basin_gap_angstrom=float(basin_gap_angstrom),
                base_rng_seed=int(replica_base_rng_seed),
            )
        )

    ranked = rank_candidates(results)
    shortlist_n = max(0, int(shortlist_count))
    # epoch-zero is the baseline the contract compares against, not a candidate:
    # it stays in `ranked` but must not displace a fine-tuned checkpoint from the
    # shortlist that feeds sci-val (P0-2). Hard-fail candidates never fill a slot
    # either — sci-val must only see gate-passing entries.
    shortlist = [
        r
        for r in ranked
        if r.hard_gates_passed and not r.candidate.is_epoch_zero
    ][:shortlist_n]
    epoch_zero_row: dict[str, Any] | None = None
    for position, r in enumerate(ranked, start=1):
        if r.candidate.is_epoch_zero:
            epoch_zero_row = {**r.as_dict(), "rank": position}
            break

    sid = screen_id
    if sid is None:
        run_ids = {c.run_id for c in cand_list}
        sid = next(iter(run_ids)) if len(run_ids) == 1 else "campaign"

    out_dir: Path | None = output_dir
    if out_dir is None and layout is not None:
        out_dir = layout.pre_screen_batch_dir(batch_id) / sid

    ranking_rule = (
        "hard_gates_all_pass -> mean_force_rmse_asc -> mean_steps_asc "
        "-> mean_rmsd_asc"
    )
    if n_rep > 1:
        ranking_rule = (
            "hard_gates_all_pass -> mean_force_rmse_asc "
            "-> median_replica_steps_asc -> median_replica_rmsd_asc"
        )

    notes = [
        "zero-DFT AIMNet2 GAU_LOOSE pre-screen only",
        "not final model selection (mindmap steps 8–9 still required)",
        "frame-level energy loss is forbidden for ranking",
    ]
    if n_rep > 1:
        notes.append(
            f"replica measurement: N={n_rep} "
            f"epsilon={float(replica_epsilon_angstrom):g} Å; "
            "rank uses median steps/RMSD; modal_basin_fraction reported only"
        )

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
        "ranking_rule": ranking_rule,
        "replicas": n_rep,
        "replica_epsilon_angstrom": (
            float(replica_epsilon_angstrom) if n_rep > 1 else None
        ),
        "basin_gap_angstrom": float(basin_gap_angstrom) if n_rep > 1 else None,
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
        "epoch_zero_baseline": epoch_zero_row,
        "epoch_zero_excluded_from_shortlist": epoch_zero_row is not None,
        "ranked": [r.as_dict() for r in ranked],
        "shortlist": [r.as_dict() for r in shortlist],
        "shortlist_checkpoint_ids": [r.candidate.checkpoint_id for r in shortlist],
        "notes": notes,
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
