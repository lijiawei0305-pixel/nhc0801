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

# V004 pilot teacher frames (read-only shared runs; legacy binding)
AUTOFILL_ROOT_TEMPLATE: Final = "autofill_{candidate_lower}_v001"
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

# Sealed Final Test commitment only (no identities)
SEALED_FINAL_TEST_COMMITMENT_SHA256: Final = (
    "834f973954064565aa857e8d8c563d110d0f6256c99e54fc3283dc428efa6975"
)
SEALED_FINAL_TEST_ROOT_COUNT: Final = 2

# Development roots (V004 day1) — identities are development-visible
TRAIN_ROOTS: Final = (
    "ACGCNTKELWXJPN-UHFFFAOYSA-N",
    "PDIYCCLDBKWBTK-UHFFFAOYSA-N",
    "VNYHGZAUUQMMDL-UHFFFAOYSA-N",
)
VALIDATION_ROOTS: Final = (
    "KZYKDQNIIMATMJ-UHFFFAOYSA-N",
    "RMEQTBVGGNKAEQ-UHFFFAOYSA-N",
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


def autofill_run_dir(runs_root: Path, candidate: str) -> Path:
    return runs_root / AUTOFILL_ROOT_TEMPLATE.format(candidate_lower=candidate.lower())


def frame_path(runs_root: Path, candidate: str, endpoint: str, index: int) -> Path:
    return autofill_run_dir(runs_root, candidate) / FRAME_RELATIVE.format(
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
