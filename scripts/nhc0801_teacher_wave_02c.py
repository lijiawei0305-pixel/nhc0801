#!/usr/bin/env python3
"""Deprecated trial name — use nhc0801_teacher_wave.py (g001 teacher wave)."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "nhc0801_teacher_wave.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
