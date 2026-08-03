#!/usr/bin/env python3
"""Deprecated name — use nhc0801_gpu_teacher_daemon.py.

Kept so existing process monitors / stewards still match ``gpu_autofill_daemon``
until redeployed. Forwards to the stable daemon.
"""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_TARGET = Path(__file__).resolve().parent / "nhc0801_gpu_teacher_daemon.py"
sys.argv[0] = str(_TARGET)
runpy.run_path(str(_TARGET), run_name="__main__")
