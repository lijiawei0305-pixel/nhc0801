"""Server/local path conventions for NHC0801 teacher frames and datasets.

Authority: mindmap.md + V004 day1 layout. Frames live under $WJW (read-only
except products written inside $WJW/NHC0801).
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

DEFAULT_WJW: Final = Path("/home/plab/test/WJW")
DEFAULT_NHC0801: Final = DEFAULT_WJW / "NHC0801"

# Clean generation (scope C / parallel S) — write only under NHC0801
DEFAULT_GENERATION_ID: Final = "nhc0801-g001"
GENERATION_RUNS_RELATIVE: Final = Path("runs") / DEFAULT_GENERATION_ID

# ---------------------------------------------------------------------------
# LEGACY READ-ONLY — V004 / phase9b shared pilot frames under $WJW/data/runs
# Do NOT write new NHC0801 products here. New teacher frames go to
# runs/nhc0801-g001/teacher_gpu_g00N/ (see docs/agent/naming.md).
# The name "autofill_*" is historical server layout, not a molecular group name.
# ---------------------------------------------------------------------------
LEGACY_V004_TEACHER_ROOT_TEMPLATE: Final = "autofill_{candidate_lower}_v001"
# Back-compat alias (imports / tests)
AUTOFILL_ROOT_TEMPLATE: Final = LEGACY_V004_TEACHER_ROOT_TEMPLATE
FRAME_RELATIVE: Final = "training_data/{endpoint}/frame_{index:04d}.json"

# V004 assembled products (read-only legacy; new products go to generation tree)
V004_D3_PROJECTION: Final = Path("data/runs/phase9b_aimnet2_v004_d3_projection_v001")
V004_WEIGHTED_DATASET: Final = Path("data/runs/phase9b_aimnet2_v004_weighted_dataset_v001")

# Parent protocol identity (Parent-Level P01)
PARENT_PROTOCOL_SHA256: Final = (
    "227c22a527e567bc4de873ab743fe9f493779eccbb1a698d2913c87695ebf87a"
)

# Official epoch-0 AIMNet2 weight
OFFICIAL_AIMNET2_WEIGHT_SHA256: Final = (
    "f0f7c054539ad3261bd36f9b11c56d12f87cb723e25bea7521755bbd3ec24e28"
)

# ---------------------------------------------------------------------------
# Legacy pilot 3+2 (pre-resplit) — packaged day1 evidence only
# ---------------------------------------------------------------------------
LEGACY_PILOT_TRAIN_ROOTS: Final = (
    "ACGCNTKELWXJPN-UHFFFAOYSA-N",
    "PDIYCCLDBKWBTK-UHFFFAOYSA-N",
    "VNYHGZAUUQMMDL-UHFFFAOYSA-N",
)
LEGACY_PILOT_VALIDATION_ROOTS: Final = (
    "KZYKDQNIIMATMJ-UHFFFAOYSA-N",
    "RMEQTBVGGNKAEQ-UHFFFAOYSA-N",
)
LEGACY_PILOT_SEALED_FINAL_TEST_COMMITMENT_SHA256: Final = (
    "834f973954064565aa857e8d8c563d110d0f6256c99e54fc3283dc428efa6975"
)
LEGACY_PILOT_SEALED_FINAL_TEST_ROOT_COUNT: Final = 2

# Active development roots: TVT resplit v001 (150/16/3) if present, else pilot.
# paths.py lives at src/nhc_deprot/data/paths.py → repo root is parents[3]
_RESPLIT_DEV_PATH: Final = (
    Path(__file__).resolve().parents[3]
    / "data"
    / "splits"
    / "nhc0801_g001_tvt_resplit_v001_development.json"
)


def _active_tvt_roots() -> tuple[tuple[str, ...], tuple[str, ...], str, int]:
    import json

    if _RESPLIT_DEV_PATH.is_file():
        payload = json.loads(_RESPLIT_DEV_PATH.read_text(encoding="utf-8"))
        train = tuple(str(p["candidate"]) for p in payload.get("train") or [])
        val = tuple(str(p["candidate"]) for p in payload.get("validation") or [])
        ft = payload.get("sealed_final_test_commitment") or {}
        if train and val:
            return (
                train,
                val,
                str(ft.get("sha256") or LEGACY_PILOT_SEALED_FINAL_TEST_COMMITMENT_SHA256),
                int(ft.get("root_count") or LEGACY_PILOT_SEALED_FINAL_TEST_ROOT_COUNT),
            )
    return (
        LEGACY_PILOT_TRAIN_ROOTS,
        LEGACY_PILOT_VALIDATION_ROOTS,
        LEGACY_PILOT_SEALED_FINAL_TEST_COMMITMENT_SHA256,
        LEGACY_PILOT_SEALED_FINAL_TEST_ROOT_COUNT,
    )


TRAIN_ROOTS, VALIDATION_ROOTS, SEALED_FINAL_TEST_COMMITMENT_SHA256, SEALED_FINAL_TEST_ROOT_COUNT = (
    _active_tvt_roots()
)

# Full-library xTB product (ranker local; usable as ranking pool)
RANKER_CANDIDATES_PARQUET: Final = Path(
    "/Users/cc/nhc-deprot-ranker/data/processed/v001/candidates.parquet"
)
RANKER_FULL_RANKED_PARQUET: Final = Path(
    "/Users/cc/nhc-deprot-ranker/results/scoring_v001/full_ranked_candidates.parquet"
)
SERVER_XTB_CRUDE_CSV: Final = Path(
    "results/calculations/20260628/imid_v4_crude/imid_full_v4menu_crude_0618_method.csv"
)

# Gold XYZ for pilot roots
MOL_GOLD_XYZ: Final = Path("data/runs/mol_gold/xyz")


def legacy_v004_teacher_run_dir(runs_root: Path, candidate: str) -> Path:
    """Path to historical V004 pilot teacher run (READ-ONLY on server)."""
    return runs_root / LEGACY_V004_TEACHER_ROOT_TEMPLATE.format(
        candidate_lower=candidate.lower()
    )


def autofill_run_dir(runs_root: Path, candidate: str) -> Path:
    """Deprecated alias of :func:`legacy_v004_teacher_run_dir` (name is historical)."""
    return legacy_v004_teacher_run_dir(runs_root, candidate)


def frame_path(runs_root: Path, candidate: str, endpoint: str, index: int) -> Path:
    """Legacy V004 frame path under autofill_* layout (read-only binding)."""
    return legacy_v004_teacher_run_dir(runs_root, candidate) / FRAME_RELATIVE.format(
        endpoint=endpoint, index=index
    )


def mol_gold_xyz(wjw: Path, candidate: str, endpoint: str) -> Path:
    return wjw / MOL_GOLD_XYZ / f"{candidate}_{endpoint}.xyz"


def nhc0801_generation_root(
    nhc0801: Path | None = None, *, generation_id: str = DEFAULT_GENERATION_ID
) -> Path:
    """Authoritative run root for new NHC0801 products (not pilot phase9b paths)."""

    base = nhc0801 or DEFAULT_NHC0801
    return base / "runs" / generation_id
