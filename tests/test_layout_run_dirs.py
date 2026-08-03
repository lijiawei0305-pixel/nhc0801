"""M9: train run / pre_screen path helpers under generation layout."""

from __future__ import annotations

from pathlib import Path

from nhc_deprot.generation.layout import resolve_layout


def test_train_batch_run_dir_joins_runs_and_run_id(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    p = layout.train_batch_run_dir("g001", "e1f100_mlp_shift")
    assert p == layout.generation_root / "train_g001" / "runs" / "e1f100_mlp_shift"
    assert p == layout.train_batch_dir("g001") / "runs" / "e1f100_mlp_shift"
    assert layout.train_batch_run_dir("g002", "e1f1_mlp").name == "e1f1_mlp"
    assert layout.train_batch_run_dir("g002", "e1f1_mlp").parts[-3:] == (
        "train_g002",
        "runs",
        "e1f1_mlp",
    )


def test_train_run_seed_dir_under_run(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    seed = 20260730
    p = layout.train_run_seed_dir("g001", "e1f100_mlp_shift", seed)
    assert p == (
        layout.generation_root
        / "train_g001"
        / "runs"
        / "e1f100_mlp_shift"
        / f"seed_{seed}"
    )
    assert p.name == f"seed_{seed}"
    assert p.parent == layout.train_batch_run_dir("g001", "e1f100_mlp_shift")


def test_pre_screen_batch_dir_flat_under_generation(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    assert layout.pre_screen_batch_dir("g001") == layout.generation_root / "pre_screen_g001"
    assert layout.pre_screen_batch_dir("g003").name == "pre_screen_g003"
    # run_id lives under pre_screen_g00N/ (M10); batch helper is the group root only
    run_out = layout.pre_screen_batch_dir("g001") / "e1f100_mlp_shift"
    assert run_out.parts[-2:] == ("pre_screen_g001", "e1f100_mlp_shift")


def test_legacy_train_seed_dir_still_resolves_without_run_layer(tmp_path: Path) -> None:
    """Old train_g00N/seed_* remains as read-only fallback path (no run_id)."""
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    legacy = layout.train_seed_dir("g001", 20260730)
    assert legacy == layout.train_batch_dir("g001") / "seed_20260730"
    # no recipe run layer under train_g00N (generation-level runs/ is unrelated)
    assert legacy.parent == layout.train_batch_dir("g001")
    assert legacy.parent.name != "runs"
    # new canonical path is distinct and nested under train_g00N/runs/<run_id>/
    modern = layout.train_run_seed_dir("g001", "e1f1_mlp", 20260730)
    assert modern != legacy
    assert modern.parts[-3:] == ("runs", "e1f1_mlp", "seed_20260730")


def test_batch_id_aliases_normalize_on_new_paths(tmp_path: Path) -> None:
    layout = resolve_layout(nhc0801_root=tmp_path / "NHC0801")
    assert layout.train_batch_run_dir("g001_pilot", "e1f1_mlp") == layout.train_batch_run_dir(
        "g001", "e1f1_mlp"
    )
    assert layout.pre_screen_batch_dir("pilot") == layout.pre_screen_batch_dir("g001")
