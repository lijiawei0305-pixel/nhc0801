"""Dataset reader / audit errors for NHC0801."""

from __future__ import annotations


class DatasetError(RuntimeError):
    """A development split, teacher-frame path, or weighted dataset contract failed."""
