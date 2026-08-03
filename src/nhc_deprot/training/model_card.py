"""Per-release model card: card.json + chart-style card.svg under models/v0.N/."""

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


@dataclass
class ModelCardFeatures:
    """Metrics for one release chart. None → bar omitted or dashed."""

    version: str = "v0.1"
    human_title: str = "AIMNet2 NHC"
    base_model: str = "aimnet2_wb97m_d3_0"
    weight_file: str = "models/v0.1/model.pt"
    reaction: str = "NHC-H+ → NHC + H+"
    reference_dft: str = "ωB97M-D3(BJ)/def2-TZVPP"
    train_target: str = "short-range residual E/F (frozen D3)"
    label_rule: str = "DFT labels only"
    train_batch_id: str = "g001"
    train_roots: int | None = None
    train_frames: int | None = None
    seed: int | None = None
    epoch: int | None = None
    selection: str = "scientific validation"
    energy_mae: float | None = None
    energy_rmse: float | None = None
    energy_unit: str = "kcal/mol"
    force_mae: float | None = None
    force_rmse: float | None = None
    force_unit: str = "eV/Å"
    deprot_label_mae: float | None = None
    deprot_label_unit: str = "kcal/mol"
    vs_epoch0_opt_steps_ratio: float | None = None
    vs_epoch0_wall_ratio: float | None = None
    handoff_pass_rate: float | None = None
    topology_pass_rate: float | None = None
    n_val_roots_eval: int | None = None
    weight_sha256_short: str | None = None
    notes: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["schema"] = CARD_SCHEMA
        return d


def _esc(s: str) -> str:
    return html.escape(s, quote=True)


def _bar_row(
    *,
    y: float,
    label: str,
    value_text: str,
    frac: float | None,
    color: str,
    track_w: float = 420,
    x0: float = 200,
) -> list[str]:
    """One horizontal bar; frac in [0,1] fills the track. None → empty track + em dash."""
    parts = [
        f'<text x="40" y="{y + 14}" fill="#334155" font-size="13" '
        f'font-family="system-ui,sans-serif">{_esc(label)}</text>',
        f'<rect x="{x0}" y="{y}" width="{track_w}" height="18" rx="4" fill="#e2e8f0"/>',
    ]
    if frac is not None and frac >= 0:
        w = max(2.0, min(1.0, float(frac)) * track_w)
        parts.append(
            f'<rect x="{x0}" y="{y}" width="{w:.1f}" height="18" rx="4" fill="{color}"/>'
        )
    parts.append(
        f'<text x="{x0 + track_w + 12}" y="{y + 14}" fill="#0f172a" font-size="13" '
        f'font-family="ui-monospace,monospace">{_esc(value_text)}</text>'
    )
    return parts


def _norm_error(val: float | None, scale: float) -> float | None:
    """Map error to bar length: smaller error → longer green bar (quality)."""
    if val is None or scale <= 0:
        return None
    # quality = max(0, 1 - val/scale)
    return max(0.0, min(1.0, 1.0 - float(val) / scale))


def _norm_ratio(val: float | None) -> float | None:
    """vs epoch0: ratio 1.0 = same; lower is better → quality bar."""
    if val is None:
        return None
    # 0.5× cost → quality 1.0; 1.0× → 0.5; 1.5× → 0
    return max(0.0, min(1.0, 1.5 - float(val)))


def render_model_card_svg(feat: ModelCardFeatures, *, width: int = 760) -> str:
    """Clean release chart: identity strip + horizontal bars (not a wall of text)."""
    # Layout
    header_h = 88
    pad = 28
    row_h = 32
    # sections of bars
    error_items: list[tuple[str, str, float | None, str]] = [
        (
            f"Energy MAE ({feat.energy_unit})",
            "—" if feat.energy_mae is None else f"{feat.energy_mae:.3g}",
            _norm_error(feat.energy_mae, scale=3.0),
            "#0ea5e9",
        ),
        (
            f"Force MAE ({feat.force_unit})",
            "—" if feat.force_mae is None else f"{feat.force_mae:.3g}",
            _norm_error(feat.force_mae, scale=0.15),
            "#0ea5e9",
        ),
        (
            f"ΔE_deprot MAE ({feat.deprot_label_unit})",
            "—" if feat.deprot_label_mae is None else f"{feat.deprot_label_mae:.3g}",
            _norm_error(feat.deprot_label_mae, scale=5.0),
            "#8b5cf6",
        ),
    ]
    rate_items: list[tuple[str, str, float | None, str]] = [
        (
            "Handoff pass",
            "—" if feat.handoff_pass_rate is None else f"{100 * feat.handoff_pass_rate:.0f}%",
            feat.handoff_pass_rate,
            "#10b981",
        ),
        (
            "Topology pass",
            "—" if feat.topology_pass_rate is None else f"{100 * feat.topology_pass_rate:.0f}%",
            feat.topology_pass_rate,
            "#10b981",
        ),
    ]
    cost_items: list[tuple[str, str, float | None, str]] = [
        (
            "Parent steps vs e0",
            "—" if feat.vs_epoch0_opt_steps_ratio is None else f"{feat.vs_epoch0_opt_steps_ratio:.2f}×",
            _norm_ratio(feat.vs_epoch0_opt_steps_ratio),
            "#f59e0b",
        ),
        (
            "Wall time vs e0",
            "—" if feat.vs_epoch0_wall_ratio is None else f"{feat.vs_epoch0_wall_ratio:.2f}×",
            _norm_ratio(feat.vs_epoch0_wall_ratio),
            "#f59e0b",
        ),
    ]

    n_rows = len(error_items) + len(rate_items) + len(cost_items) + 3  # section titles
    chart_h = n_rows * row_h + 40
    height = header_h + chart_h + 56

    meta = (
        f"{feat.train_batch_id} → {feat.version}  ·  {feat.reference_dft}  ·  "
        f"seed {feat.seed if feat.seed is not None else '—'}  "
        f"epoch {feat.epoch if feat.epoch is not None else '—'}"
    )

    parts: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        # white card
        f'<rect width="100%" height="100%" fill="#f8fafc"/>',
        f'<rect x="16" y="16" width="{width - 32}" height="{height - 32}" rx="12" '
        f'fill="#ffffff" stroke="#e2e8f0" stroke-width="1.5"/>',
        # accent bar
        f'<rect x="16" y="16" width="8" height="{height - 32}" rx="2" fill="#0ea5e9"/>',
        # title
        f'<text x="{pad + 12}" y="52" fill="#0f172a" font-size="24" font-weight="700" '
        f'font-family="system-ui,sans-serif">{_esc(feat.human_title)}  {_esc(feat.version)}</text>',
        f'<text x="{pad + 12}" y="76" fill="#64748b" font-size="13" '
        f'font-family="system-ui,sans-serif">{_esc(meta)}</text>',
        f'<text x="{pad + 12}" y="96" fill="#94a3b8" font-size="12" '
        f'font-family="system-ui,sans-serif">{_esc(feat.reaction)}  ·  {_esc(feat.weight_file)}</text>',
    ]

    y = header_h + 24

    def section(title: str) -> None:
        nonlocal y
        parts.append(
            f'<text x="{pad + 12}" y="{y}" fill="#64748b" font-size="11" '
            f'font-family="system-ui,sans-serif" font-weight="600" '
            f'letter-spacing="0.06em">{_esc(title.upper())}</text>'
        )
        y += 22

    section("Errors  (longer bar = better)")
    for label, vtxt, frac, color in error_items:
        parts.extend(_bar_row(y=y, label=label, value_text=vtxt, frac=frac, color=color))
        y += row_h

    section("Route reliability")
    for label, vtxt, frac, color in rate_items:
        parts.extend(_bar_row(y=y, label=label, value_text=vtxt, frac=frac, color=color))
        y += row_h

    section("Cost vs Epoch-0  (longer = cheaper than e0)")
    for label, vtxt, frac, color in cost_items:
        parts.extend(_bar_row(y=y, label=label, value_text=vtxt, frac=frac, color=color))
        y += row_h

    sha = feat.weight_sha256_short or "—"
    parts.append(
        f'<text x="{pad + 12}" y="{height - 28}" fill="#94a3b8" font-size="11" '
        f'font-family="ui-monospace,monospace">sha { _esc(sha) }  ·  '
        f"select: {_esc(feat.selection)}</text>"
    )
    parts.append("</svg>")
    return "\n".join(parts)


def write_model_card(
    layout: GenerationLayout,
    feat: ModelCardFeatures,
    *,
    overwrite: bool = True,
) -> dict[str, str]:
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
