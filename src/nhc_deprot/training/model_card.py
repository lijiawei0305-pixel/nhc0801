"""Per-release model feature card (JSON + SVG) for models/v0.N/.

Each published fine-tune must ship a short visual card of key traits.
Metrics combine standard MLFF practice (E/F errors) with NHC0801 scientific
route outcomes (label error, handoff, vs Epoch-0 cost) — not only frame loss.
"""

from __future__ import annotations

import html
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Final

from nhc_deprot.data.io_util import write_json
from nhc_deprot.generation.layout import GenerationLayout, normalize_model_version

CARD_JSON: Final = "card.json"
CARD_SVG: Final = "card.svg"
CARD_SCHEMA: Final = "nhc0801-model-card-v1"

# Feature groups shown on the card (research MLFF + this project)
FEATURE_GROUPS: Final = (
    ("identity", "Identity"),
    ("chemistry", "Chemistry & reference"),
    ("training", "Training"),
    ("frame_metrics", "Frame metrics (screening)"),
    ("sci_metrics", "Scientific route (selection)"),
    ("provenance", "Provenance"),
)


@dataclass
class ModelCardFeatures:
    """Fields for one release card. Missing numerics stay None → shown as —."""

    # Identity
    version: str = "v0.1"
    human_title: str = "AIMNet2 NHC"
    base_model: str = "aimnet2_wb97m_d3_0"
    weight_file: str = "models/v0.1/model.pt"

    # Chemistry & reference DFT
    reaction: str = "NHC-H+ → NHC + H+"
    reference_dft: str = "ωB97M-D3(BJ)/def2-TZVPP"
    train_target: str = "short-range residual E/F (frozen D3)"
    label_rule: str = "DFT labels only (AIMNet2 energy not in ΔE_deprot)"

    # Training scope
    train_batch_id: str = "g001"
    train_roots: int | None = None
    train_frames: int | None = None
    seed: int | None = None
    epoch: int | None = None
    selection: str = "scientific validation (not quick-val)"

    # Frame-level (quick-val / MLFF standard) — screening only
    energy_mae: float | None = None  # e.g. kcal/mol or meV — unit in energy_unit
    energy_rmse: float | None = None
    energy_unit: str = "kcal/mol"
    force_mae: float | None = None
    force_rmse: float | None = None
    force_unit: str = "eV/Å"

    # Scientific route (project core)
    deprot_label_mae: float | None = None  # |ΔE_deprot model−route − DFT ref|
    deprot_label_unit: str = "kcal/mol"
    vs_epoch0_opt_steps_ratio: float | None = None  # <1 means fewer parent steps
    vs_epoch0_wall_ratio: float | None = None
    handoff_pass_rate: float | None = None  # 0–1
    topology_pass_rate: float | None = None  # 0–1
    n_val_roots_eval: int | None = None

    # Provenance
    weight_sha256_short: str | None = None  # first 12 hex
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = CARD_SCHEMA
        d["feature_groups"] = [list(g) for g in FEATURE_GROUPS]
        return d


def _fmt(v: Any, digits: int = 3) -> str:
    if v is None:
        return "—"
    if isinstance(v, float):
        if 0 <= v <= 1 and digits >= 2:
            # rates often 0–1
            return f"{v:.2%}" if v <= 1.0 and "rate" not in str(type(v)) else f"{v:.{digits}g}"
        return f"{v:.{digits}g}"
    return str(v)


def _fmt_rate(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{100.0 * float(v):.1f}%"


def _fmt_ratio(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{float(v):.2f}× vs e0"


def build_card_rows(feat: ModelCardFeatures) -> list[tuple[str, list[tuple[str, str]]]]:
    """Grouped (section_title, [(label, value), ...]) for rendering."""
    return [
        (
            "Identity",
            [
                ("Version", feat.version),
                ("Title", feat.human_title),
                ("Base", feat.base_model),
                ("File", feat.weight_file),
            ],
        ),
        (
            "Chemistry & reference",
            [
                ("Reaction", feat.reaction),
                ("DFT", feat.reference_dft),
                ("Learn", feat.train_target),
                ("Labels", feat.label_rule),
            ],
        ),
        (
            "Training",
            [
                ("From train", feat.train_batch_id),
                ("Roots / frames", f"{_fmt(feat.train_roots)} / {_fmt(feat.train_frames)}"),
                ("Seed / epoch", f"{_fmt(feat.seed)} / {_fmt(feat.epoch)}"),
                ("Selected by", feat.selection),
            ],
        ),
        (
            "Frame metrics (screening only)",
            [
                (f"E MAE ({feat.energy_unit})", _fmt(feat.energy_mae)),
                (f"E RMSE ({feat.energy_unit})", _fmt(feat.energy_rmse)),
                (f"F MAE ({feat.force_unit})", _fmt(feat.force_mae)),
                (f"F RMSE ({feat.force_unit})", _fmt(feat.force_rmse)),
            ],
        ),
        (
            "Scientific route (selection)",
            [
                (f"ΔE_deprot MAE ({feat.deprot_label_unit})", _fmt(feat.deprot_label_mae)),
                ("Parent steps vs e0", _fmt_ratio(feat.vs_epoch0_opt_steps_ratio)),
                ("Wall time vs e0", _fmt_ratio(feat.vs_epoch0_wall_ratio)),
                ("Handoff pass", _fmt_rate(feat.handoff_pass_rate)),
                ("Topology pass", _fmt_rate(feat.topology_pass_rate)),
                ("Val roots eval", _fmt(feat.n_val_roots_eval)),
            ],
        ),
        (
            "Provenance",
            [
                ("SHA256", feat.weight_sha256_short or "—"),
                ("Notes", "; ".join(feat.notes) if feat.notes else "—"),
            ],
        ),
    ]


def render_model_card_svg(feat: ModelCardFeatures, *, width: int = 900) -> str:
    """Self-contained SVG release card (no extra Python deps)."""
    rows_grouped = build_card_rows(feat)
    pad = 24
    y = pad + 8
    line_h = 22
    section_gap = 14
    col1_x = pad + 8
    col2_x = 280
    content_lines = 0
    for _, pairs in rows_grouped:
        content_lines += 1 + len(pairs)
    height = pad * 2 + 70 + content_lines * line_h + section_gap * len(rows_grouped) + 20

    def esc(s: str) -> str:
        return html.escape(s, quote=True)

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        f'<rect width="100%" height="100%" fill="#0f172a"/>',
        f'<rect x="12" y="12" width="{width - 24}" height="{height - 24}" rx="16" '
        f'fill="#1e293b" stroke="#38bdf8" stroke-width="2"/>',
        f'<text x="{pad + 8}" y="{y + 28}" fill="#f8fafc" font-size="26" '
        f'font-family="ui-sans-serif, system-ui, sans-serif" font-weight="700">'
        f"{esc(feat.human_title)}  {esc(feat.version)}</text>",
        f'<text x="{pad + 8}" y="{y + 52}" fill="#94a3b8" font-size="13" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">'
        f"NHC deprotonation · fine-tuned AIMNet2 · release card</text>",
    ]
    y += 70

    for section, pairs in rows_grouped:
        parts.append(
            f'<text x="{col1_x}" y="{y}" fill="#38bdf8" font-size="14" '
            f'font-family="ui-sans-serif, system-ui, sans-serif" font-weight="600">'
            f"{esc(section)}</text>"
        )
        y += line_h
        for label, value in pairs:
            parts.append(
                f'<text x="{col1_x}" y="{y}" fill="#94a3b8" font-size="13" '
                f'font-family="ui-sans-serif, system-ui, sans-serif">{esc(label)}</text>'
            )
            # truncate long values for layout
            v = value if len(value) <= 72 else value[:69] + "..."
            parts.append(
                f'<text x="{col2_x}" y="{y}" fill="#e2e8f0" font-size="13" '
                f'font-family="ui-monospace, monospace">{esc(v)}</text>'
            )
            y += line_h
        y += section_gap

    parts.append(
        f'<text x="{pad + 8}" y="{height - 18}" fill="#64748b" font-size="11" '
        f'font-family="ui-sans-serif, system-ui, sans-serif">'
        f"Frame metrics do not select the model · Scientific route decides · "
        f"train_g00N → v0.N</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_model_card(
    layout: GenerationLayout,
    feat: ModelCardFeatures,
    *,
    overwrite: bool = True,
) -> dict[str, str]:
    """Write models/vX.Y/card.json + card.svg."""
    ver = normalize_model_version(feat.version)
    feat.version = ver
    feat.weight_file = f"models/{ver}/model.pt"
    vdir = layout.model_version_dir(ver)
    vdir.mkdir(parents=True, exist_ok=True)
    json_path = vdir / CARD_JSON
    svg_path = vdir / CARD_SVG
    if not overwrite and (json_path.exists() or svg_path.exists()):
        raise FileExistsError(f"card already exists under {vdir}")
    write_json(json_path, feat.as_dict(), overwrite=True)
    svg_path.write_text(render_model_card_svg(feat), encoding="utf-8")
    return {"card_json": str(json_path), "card_svg": str(svg_path), "version": ver}


def card_features_from_info(
    info: dict[str, Any],
    *,
    extras: dict[str, Any] | None = None,
) -> ModelCardFeatures:
    """Build card fields from model info.json + optional metrics dict."""
    extras = extras or {}
    sha = info.get("weight_sha256")
    short = (str(sha)[:12] + "…") if sha else extras.get("weight_sha256_short")
    return ModelCardFeatures(
        version=str(info.get("version") or extras.get("version") or "v0.1"),
        human_title=str(extras.get("human_title") or "AIMNet2 NHC"),
        train_batch_id=str(info.get("train_batch_id") or extras.get("train_batch_id") or "g001"),
        seed=info.get("seed") if info.get("seed") is not None else extras.get("seed"),
        epoch=info.get("epoch") if info.get("epoch") is not None else extras.get("epoch"),
        weight_sha256_short=short,
        train_roots=extras.get("train_roots"),
        train_frames=extras.get("train_frames"),
        energy_mae=extras.get("energy_mae"),
        energy_rmse=extras.get("energy_rmse"),
        energy_unit=str(extras.get("energy_unit") or "kcal/mol"),
        force_mae=extras.get("force_mae"),
        force_rmse=extras.get("force_rmse"),
        force_unit=str(extras.get("force_unit") or "eV/Å"),
        deprot_label_mae=extras.get("deprot_label_mae"),
        deprot_label_unit=str(extras.get("deprot_label_unit") or "kcal/mol"),
        vs_epoch0_opt_steps_ratio=extras.get("vs_epoch0_opt_steps_ratio"),
        vs_epoch0_wall_ratio=extras.get("vs_epoch0_wall_ratio"),
        handoff_pass_rate=extras.get("handoff_pass_rate"),
        topology_pass_rate=extras.get("topology_pass_rate"),
        n_val_roots_eval=extras.get("n_val_roots_eval"),
        notes=list(extras.get("notes") or info.get("notes") or []),
    )


def load_card_json(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))
