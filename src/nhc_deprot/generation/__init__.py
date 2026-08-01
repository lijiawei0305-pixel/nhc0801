"""Generation identity and run-layout for NHC0801 clean restarts."""

from nhc_deprot.generation.layout import (
    DEFAULT_GENERATION_ID,
    GenerationLayout,
    GenerationMeta,
    build_generation_meta,
    ensure_generation_tree,
    load_generation_meta,
    write_generation_meta,
)

__all__ = [
    "DEFAULT_GENERATION_ID",
    "GenerationLayout",
    "GenerationMeta",
    "build_generation_meta",
    "ensure_generation_tree",
    "load_generation_meta",
    "write_generation_meta",
]
