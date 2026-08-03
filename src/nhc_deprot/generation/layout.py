"""NHC0801 generation layout: nhc0801-g001 under $WJW/NHC0801/runs only.

Scope C: pilot-scale first. Parallel S: single profile default; dual needs claim.
No live chemistry here — directory + metadata only.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from nhc_deprot.contracts.parent_protocol import PROTOCOL_SHA256
from nhc_deprot.data.io_util import canonical_json, load_json_object, sha256_bytes
from nhc_deprot.data.paths import (
    DEFAULT_NHC0801,
    DEFAULT_WJW,
    SEALED_FINAL_TEST_COMMITMENT_SHA256,
    SEALED_FINAL_TEST_ROOT_COUNT,
    TRAIN_ROOTS,
    VALIDATION_ROOTS,
)

GENERATION_SCHEMA: Final = "nhc0801-generation-meta-v1"
DEFAULT_GENERATION_ID: Final = "nhc0801-g001"
SCOPE: Final = "C"
PARALLEL_STRATEGY: Final = "S"

# Subdirectories under runs/<generation_id>/
# Teacher groups: teacher_gpu_g00N/ (uniform; g001 pilot included)
# Epoch-0 groups: epoch0_val_batches/g00N/
# Fine-tune groups: train_g00N/ (same flat style as teacher_gpu_g00N/)
# Released model versions: models/v0.1/model.pt (short version tags, not long English stems)
GENERATION_SUBDIRS: Final = (
    "meta",
    "resources",
    "teacher_gpu_g001",
    "d3",
    "datasets/weighted",
    "epoch0_val_batches/g001/epoch0",
    "epoch0_val_batches/g001/logs",
    "train_g001",
    "train_g001/logs",
    "train",  # legacy pilot only; new writes use train_g00N/
    "models",
    "sci_val",
    "freeze",
    "logs",
)

# Standard human name: "g001 Epoch-0" → this relative tree under generation_root
G001_EPOCH0_REL: Final = "epoch0_val_batches/g001/epoch0"
# Standard human name: "g001 teacher" → teacher_gpu_g001/
G001_TEACHER_REL: Final = "teacher_gpu_g001"
# Standard human name: "g001 train" / "g001 微调" → train_g001/
G001_TRAIN_REL: Final = "train_g001"
# Standard human name: "v0.1" → models/v0.1/model.pt
MODELS_REL: Final = "models"


class GenerationError(RuntimeError):
    """Generation layout or metadata is invalid."""


@dataclass(frozen=True, slots=True)
class GenerationLayout:
    """Resolved paths for one generation (server or local sandbox)."""

    generation_id: str
    nhc0801_root: Path
    runs_root: Path
    generation_root: Path
    meta_dir: Path
    resources_dir: Path
    teacher_dir: Path
    d3_dir: Path
    datasets_dir: Path
    epoch0_dir: Path
    train_dir: Path
    models_dir: Path
    sci_val_dir: Path
    freeze_dir: Path
    logs_dir: Path

    def teacher_root_dir(self, root_id: str) -> Path:
        return self.teacher_dir / root_id

    def teacher_endpoint_dir(self, root_id: str, endpoint: str) -> Path:
        if endpoint not in {"cation", "neutral"}:
            raise GenerationError(f"invalid endpoint: {endpoint}")
        return self.teacher_root_dir(root_id) / endpoint

    def teacher_batch_dir(self, batch_id: str) -> Path:
        """Teacher product dir for group g00N: teacher_gpu_<batch_id>/.

        Uniform for all groups (g001 pilot, g002, g003, …). Do not use
        legacy names ``teacher/`` or ``teacher_gpu_side/`` as canonical paths.
        """
        bid = _normalize_batch_id(batch_id)
        return self.generation_root / f"teacher_gpu_{bid}"

    def resource_claim_path(self, claim_id: str) -> Path:
        return self.resources_dir / f"claim_{claim_id}.json"

    def selection_receipt_path(self) -> Path:
        return self.resources_dir / "profile_selection_receipt.json"

    def generation_meta_path(self) -> Path:
        return self.meta_dir / "generation.json"

    def epoch0_batch_root(self, batch_id: str) -> Path:
        """Root for one batch's Epoch-0 tree: epoch0_val_batches/<batch_id>/."""
        bid = _normalize_batch_id(batch_id)
        return self.generation_root / "epoch0_val_batches" / bid

    def epoch0_batch_dir(self, batch_id: str) -> Path:
        """Campaign receipts for g00N Epoch-0: .../epoch0_val_batches/<batch_id>/epoch0/."""
        return self.epoch0_batch_root(batch_id) / "epoch0"

    def train_batch_dir(self, batch_id: str) -> Path:
        """Fine-tune product dir for group g00N: train_g00N/.

        Same flat pattern as ``teacher_gpu_g00N/``.
        Human name: **g00N train** / **g00N 微调**.
        Do not use bare ``train/`` as the canonical write path for new runs.
        """
        bid = _normalize_batch_id(batch_id)
        return self.generation_root / f"train_{bid}"

    def train_batch_logs_dir(self, batch_id: str) -> Path:
        """Logs for one train group: train_g00N/logs/."""
        return self.train_batch_dir(batch_id) / "logs"

    def train_seed_dir(self, batch_id: str, seed: int) -> Path:
        """One random-seed run: train_g00N/seed_<seed>/."""
        return self.train_batch_dir(batch_id) / f"seed_{int(seed)}"

    def train_checkpoint_meta_path(self, batch_id: str, seed: int, epoch: int) -> Path:
        """Checkpoint meta: .../seed_<seed>/epoch_NNNN.meta.json."""
        return self.train_seed_dir(batch_id, seed) / f"epoch_{int(epoch):04d}.meta.json"

    def train_checkpoint_weight_path(self, batch_id: str, seed: int, epoch: int) -> Path:
        """Checkpoint weights: .../seed_<seed>/epoch_NNNN.pt."""
        return self.train_seed_dir(batch_id, seed) / f"epoch_{int(epoch):04d}.pt"

    def train_campaign_receipt_path(self, batch_id: str) -> Path:
        """Whole multi-seed train result: train_g00N/train_result.json."""
        return self.train_batch_dir(batch_id) / "train_result.json"

    def train_manifest_path(self, batch_id: str) -> Path:
        """What was trained (roots, teacher sources, hyperparams): train_info.json."""
        return self.train_batch_dir(batch_id) / "train_info.json"

    def train_seed_receipt_path(self, batch_id: str, seed: int) -> Path:
        """One seed's result: seed_result.json."""
        return self.train_seed_dir(batch_id, seed) / "seed_result.json"

    def model_version_dir(self, version: str) -> Path:
        """Released model version folder: models/v0.1/.

        Human name is just **v0.1** (or v0.2, …). Weight file is always ``model.pt``.
        """
        ver = normalize_model_version(version)
        return self.models_dir / ver

    def model_weight_path(self, version: str) -> Path:
        """Canonical weight file: models/v0.1/model.pt (never a long English basename)."""
        return self.model_version_dir(version) / "model.pt"

    def model_info_path(self, version: str) -> Path:
        """Provenance for a released version: models/v0.1/info.json."""
        return self.model_version_dir(version) / "info.json"

    def resolve_train_batch_dir_for_read(self, batch_id: str) -> Path:
        """Prefer train_g00N when it has products; else legacy train/ for g001.

        Also accepts obsolete ``train_batches/g00N`` if present (read-only).
        Empty scaffold dirs do not count as products.
        New writes must always use :meth:`train_batch_dir` → ``train_g00N/``.
        """
        bid = _normalize_batch_id(batch_id)
        candidates = [
            self.train_batch_dir(batch_id),  # train_g00N
            self.generation_root / "train_batches" / bid,  # obsolete nested form
        ]
        if bid == "g001":
            candidates.append(self.train_dir)  # legacy pilot train/

        def _has_products(d: Path) -> bool:
            if not d.is_dir():
                return False
            return bool(
                any(d.glob("seed_*"))
                or (d / "train_result.json").is_file()
                or (d / "campaign_receipt.json").is_file()
                or (d / "campaign_receipt_live.json").is_file()
            )

        for d in candidates:
            if _has_products(d):
                return d
        for d in candidates:
            if d.is_dir():
                return d
        return self.train_batch_dir(batch_id)


def _normalize_batch_id(batch_id: str) -> str:
    bid = str(batch_id).strip()
    if bid in {"g001_pilot", "pilot", "g001-pilot"}:
        return "g001"
    if bid == "nhc0801-g001":
        return "g001"
    if not bid or "/" in bid or ".." in bid:
        raise GenerationError(f"invalid batch_id: {batch_id!r}")
    return bid


def normalize_model_version(version: str) -> str:
    """Normalize a short model version tag to ``vMAJOR.MINOR`` (e.g. v0.1, v0.2).

    Accepts ``0.1``, ``v0.1``, ``V0.1``. Rejects long English stems and path junk.
    """
    raw = str(version).strip()
    if not raw or "/" in raw or "\\" in raw or ".." in raw:
        raise GenerationError(f"invalid model version: {version!r}")
    s = raw.lower()
    if s.startswith("v"):
        s = s[1:]
    parts = s.split(".")
    if len(parts) != 2 or not all(p.isdigit() for p in parts):
        raise GenerationError(
            f"model version must look like 0.1 or v0.1, got {version!r}"
        )
    # keep decimal form without stripping meaningful zeros in minor? int is fine for 0.1
    major, minor = int(parts[0]), int(parts[1])
    return f"v{major}.{minor}"


@dataclass
class GenerationMeta:
    schema: str = GENERATION_SCHEMA
    generation_id: str = DEFAULT_GENERATION_ID
    project: str = "NHC0801"
    scope: str = SCOPE
    parallel_strategy: str = PARALLEL_STRATEGY
    parent_protocol_sha256: str = PROTOCOL_SHA256
    default_resource_profile: str = "single_27_physical_v1"
    dual_profile_candidate: str = "dual_14_13_physical_v1"
    train_roots: list[str] = field(default_factory=lambda: list(TRAIN_ROOTS))
    validation_roots: list[str] = field(default_factory=lambda: list(VALIDATION_ROOTS))
    sealed_final_test_commitment_sha256: str = SEALED_FINAL_TEST_COMMITMENT_SHA256
    sealed_final_test_root_count: int = SEALED_FINAL_TEST_ROOT_COUNT
    final_test_identities_exposed: bool = False
    live_chemistry_authorized: bool = False
    source_commit: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def resolve_layout(
    *,
    generation_id: str = DEFAULT_GENERATION_ID,
    nhc0801_root: Path | None = None,
    wjw: Path | None = None,
) -> GenerationLayout:
    root = nhc0801_root
    if root is None:
        root = (wjw or DEFAULT_WJW) / "NHC0801" if wjw is not None else DEFAULT_NHC0801
    root = Path(root)
    if generation_id != generation_id.strip() or "/" in generation_id or ".." in generation_id:
        raise GenerationError(f"invalid generation_id: {generation_id!r}")
    gen = root / "runs" / generation_id
    return GenerationLayout(
        generation_id=generation_id,
        nhc0801_root=root,
        runs_root=root / "runs",
        generation_root=gen,
        meta_dir=gen / "meta",
        resources_dir=gen / "resources",
        # g001 teacher: same pattern as g002/g003… (teacher_gpu_g001/)
        teacher_dir=gen / "teacher_gpu_g001",
        d3_dir=gen / "d3",
        datasets_dir=gen / "datasets" / "weighted",
        # g001 Epoch-0: same pattern as g002/g003… (epoch0_val_batches/g001/)
        epoch0_dir=gen / "epoch0_val_batches" / "g001" / "epoch0",
        train_dir=gen / "train",
        models_dir=gen / "models",
        sci_val_dir=gen / "sci_val",
        freeze_dir=gen / "freeze",
        logs_dir=gen / "logs",
    )


def ensure_generation_tree(
    layout: GenerationLayout,
    *,
    exist_ok: bool = True,
) -> GenerationLayout:
    """Create empty generation directory tree (no chemistry)."""

    layout.generation_root.mkdir(parents=True, exist_ok=exist_ok)
    for rel in GENERATION_SUBDIRS:
        (layout.generation_root / rel).mkdir(parents=True, exist_ok=True)
    return layout


def build_generation_meta(
    *,
    generation_id: str = DEFAULT_GENERATION_ID,
    source_commit: str | None = None,
    notes: list[str] | None = None,
) -> GenerationMeta:
    return GenerationMeta(
        generation_id=generation_id,
        source_commit=source_commit,
        notes=list(notes or [])
        + [
            "scope=C: pilot train/val roots first",
            "parallel=S: single_27 default; dual only after claim+calibration receipt",
            "live_chemistry_authorized=false until explicit user gate",
        ],
    )


def write_generation_meta(layout: GenerationLayout, meta: GenerationMeta) -> dict[str, object]:
    path = layout.generation_meta_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    if meta.generation_id != layout.generation_id:
        raise GenerationError("meta.generation_id must match layout")
    if meta.final_test_identities_exposed:
        raise GenerationError("Final Test identities must not be exposed")
    if meta.live_chemistry_authorized:
        raise GenerationError(
            "refusing to write live_chemistry_authorized=true via layout helper; "
            "use explicit gate workflow"
        )
    raw = canonical_json(meta.as_dict())
    if path.exists():
        existing = path.read_bytes()
        if existing != raw:
            raise GenerationError(
                f"generation.json already exists and differs (no overwrite): {path}"
            )
        return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw), "wrote": False}
    path.write_bytes(raw)
    return {"path": str(path), "bytes": len(raw), "sha256": sha256_bytes(raw), "wrote": True}


def load_generation_meta(path: Path) -> GenerationMeta:
    payload, _ = load_json_object(path)
    if payload.get("schema") != GENERATION_SCHEMA:
        raise GenerationError(f"unexpected generation schema: {payload.get('schema')!r}")
    if payload.get("final_test_identities_exposed") is True:
        raise GenerationError("generation meta exposes Final Test identities")
    return GenerationMeta(
        schema=str(payload["schema"]),
        generation_id=str(payload["generation_id"]),
        project=str(payload.get("project", "NHC0801")),
        scope=str(payload.get("scope", SCOPE)),
        parallel_strategy=str(payload.get("parallel_strategy", PARALLEL_STRATEGY)),
        parent_protocol_sha256=str(payload.get("parent_protocol_sha256", PROTOCOL_SHA256)),
        default_resource_profile=str(
            payload.get("default_resource_profile", "single_27_physical_v1")
        ),
        dual_profile_candidate=str(
            payload.get("dual_profile_candidate", "dual_14_13_physical_v1")
        ),
        train_roots=list(payload.get("train_roots") or TRAIN_ROOTS),
        validation_roots=list(payload.get("validation_roots") or VALIDATION_ROOTS),
        sealed_final_test_commitment_sha256=str(
            payload.get(
                "sealed_final_test_commitment_sha256",
                SEALED_FINAL_TEST_COMMITMENT_SHA256,
            )
        ),
        sealed_final_test_root_count=int(
            payload.get("sealed_final_test_root_count", SEALED_FINAL_TEST_ROOT_COUNT)
        ),
        final_test_identities_exposed=bool(
            payload.get("final_test_identities_exposed", False)
        ),
        live_chemistry_authorized=bool(payload.get("live_chemistry_authorized", False)),
        source_commit=payload.get("source_commit"),  # type: ignore[arg-type]
        notes=list(payload.get("notes") or []),
    )


def init_generation(
    *,
    generation_id: str = DEFAULT_GENERATION_ID,
    nhc0801_root: Path | None = None,
    source_commit: str | None = None,
    exist_ok: bool = True,
) -> tuple[GenerationLayout, GenerationMeta, dict[str, object]]:
    """Create tree + write generation.json (local sandbox or server root)."""

    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    ensure_generation_tree(layout, exist_ok=exist_ok)
    meta = build_generation_meta(generation_id=generation_id, source_commit=source_commit)
    receipt = write_generation_meta(layout, meta)
    return layout, meta, receipt
