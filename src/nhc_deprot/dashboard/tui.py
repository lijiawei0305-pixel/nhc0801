"""SSH terminal TUI — pure read-only, default 30s refresh.

No disk writes. No ports. Ctrl+C to exit.
"""

from __future__ import annotations

import os
import shutil
import sys
import time
from pathlib import Path
from typing import Any, TextIO

from nhc_deprot.generation.layout import resolve_layout
from nhc_deprot.pipeline.pipeline_status import scan_generation_status
from nhc_deprot.resources.auto_fill import compute_capacity
from nhc_deprot.resources.profiles import OFFICIAL_DEFAULT_V002, get_profile

# ANSI (disabled if not a tty)
_RESET = "\033[0m"
_BOLD = "\033[1m"
_GREEN = "\033[32m"
_YELLOW = "\033[33m"
_RED = "\033[31m"
_CYAN = "\033[36m"
_DIM = "\033[2m"
_GRAY = "\033[90m"


def _color(enabled: bool, code: str, text: str) -> str:
    if not enabled:
        return text
    return f"{code}{text}{_RESET}"


def _status_color(enabled: bool, status: str) -> str:
    s = status.upper()
    if s in {"PASS", "SELECTED", "FROZEN", "POLICY", "SEALED"}:
        return _color(enabled, _GREEN, status)
    if s in {"RUNNING", "PROVISIONAL", "PARTIAL", "UNKNOWN"}:
        return _color(enabled, _YELLOW, status)
    if s in {"FAIL", "FAILED", "REJECTED"}:
        return _color(enabled, _RED, status)
    if s in {"NOT_STARTED"}:
        return _color(enabled, _GRAY, status)
    return status


def _try_host_resources() -> dict[str, Any]:
    """Best-effort local host snapshot (no SSH). Fail soft."""

    out: dict[str, Any] = {"available": False}
    try:
        mem_avail = 0
        with open("/proc/meminfo", encoding="utf-8") as fh:
            for line in fh:
                if line.startswith("MemAvailable:"):
                    mem_avail = int(line.split()[1]) * 1024
                    break
        nproc = os.cpu_count() or 0
        # idle estimate: load-based soft signal only
        load1 = 0.0
        try:
            load1 = os.getloadavg()[0]
        except OSError:
            pass
        # crude idle logical estimate for display only
        idle_est = max(0, int(nproc - min(nproc, load1)))
        prof = get_profile(OFFICIAL_DEFAULT_V002)
        cap = compute_capacity(
            idle_logical_cpus=idle_est,
            mem_available_bytes=mem_avail,
            profile=prof,
        )
        out = {
            "available": True,
            "nproc": nproc,
            "load1": load1,
            "mem_available_gib": round(mem_avail / (1024**3), 1),
            "idle_est": idle_est,
            "auto_fill_N": cap.n,
            "auto_fill_N_cpu": cap.n_cpu,
            "auto_fill_N_mem": cap.n_mem,
            "profile_id": cap.profile_id,
            "threads_per_endpoint": cap.threads_per_endpoint,
        }
    except Exception as exc:  # noqa: BLE001
        out = {"available": False, "error": str(exc)}
    return out


def _try_gpu_lines() -> list[str]:
    try:
        import subprocess

        proc = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,utilization.gpu,memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
        if proc.returncode != 0:
            return []
        lines = []
        for line in proc.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 4:
                lines.append(
                    f"GPU{parts[0]}: util={parts[1]}% mem={parts[2]}/{parts[3]} MiB"
                )
        return lines
    except Exception:  # noqa: BLE001
        return []


def render_status_text(
    *,
    nhc0801_root: Path,
    generation_id: str = "nhc0801-g001",
    color: bool | None = None,
) -> str:
    """One-shot text frame for TUI or logging."""

    use_color = sys.stdout.isatty() if color is None else bool(color)
    layout = resolve_layout(generation_id=generation_id, nhc0801_root=nhc0801_root)
    snap = scan_generation_status(layout)
    host = _try_host_resources()
    gpus = _try_gpu_lines()

    width = max(60, min(shutil.get_terminal_size((100, 40)).columns, 120))
    bar = "─" * (width - 2)
    lines: list[str] = []
    title = (
        f" NHC0801 | gen={snap['generation_id']} | "
        f"{snap['updated_at_utc']} | TUI read-only "
    )
    lines.append("┌" + bar + "┐")
    lines.append("│" + _color(use_color, _BOLD, title.ljust(width - 2)[: width - 2]) + "│")
    lines.append("├" + bar + "┤")

    # pipeline lights
    lights = []
    for step in snap.get("steps") or []:
        st = str(step.get("status", "?"))
        lights.append(f"[{step['step']}]{_status_color(use_color, st)}")
    pipe = " ".join(lights)
    # wrap
    lines.append("│ " + _color(use_color, _CYAN, "Pipeline:") + " " + pipe[: width - 14])
    if len(pipe) > width - 14:
        lines.append("│   " + pipe[width - 14 : width * 2])
    lines.append(
        "│ FT: "
        + _status_color(use_color, "SEALED")
        + " (never auto) | orch_running="
        + str(snap.get("orchestrator_running"))
    )

    # resources
    if host.get("available"):
        lines.append(
            "│ Resources: "
            f"load1={host.get('load1'):.2f} nproc={host.get('nproc')} "
            f"idle~{host.get('idle_est')} mem_avail={host.get('mem_available_gib')}GiB "
            f"auto_fill N={host.get('auto_fill_N')} "
            f"(cpu={host.get('auto_fill_N_cpu')} mem={host.get('auto_fill_N_mem')}) "
            f"t={host.get('threads_per_endpoint')}"
        )
        lines.append(f"│ profile={host.get('profile_id')}")
    else:
        lines.append("│ Resources: (host sample unavailable — not Linux /proc?)")
    if gpus:
        lines.append("│ " + " | ".join(gpus[:4]))
        if len(gpus) > 4:
            lines.append("│ " + " | ".join(gpus[4:8]))

    lines.append("├" + bar + "┤")
    lines.append("│ " + _color(use_color, _BOLD, "Train"))
    tmetrics = snap.get("train_metrics") or []
    if not tmetrics:
        lines.append("│   (no seed receipts)")
    for m in tmetrics:
        lines.append(
            "│   "
            f"seed={m.get('seed')} status={_status_color(use_color, str(m.get('status')))} "
            f"ep={m.get('last_epoch')}/{m.get('epochs_run')} "
            f"loss_tr={m.get('train_weighted_loss')} "
            f"loss_val={m.get('validation_weighted_loss')} "
            f"pt={'Y' if m.get('has_pt') else 'n'} "
            f"shortlist={m.get('shortlist_epochs')}"
        )

    lines.append("├" + bar + "┤")
    lines.append(
        "│ "
        + _color(use_color, _BOLD, "Scientific (label MAE primary when present)")
    )
    sci = snap.get("scientific_metrics") or {}
    if not sci:
        lines.append("│   (no epoch0/sci_val/shortlist receipts yet)")
    for key, val in sci.items():
        if not isinstance(val, dict):
            continue
        lines.append(f"│   {key}: status={val.get('status')} {val}")

    lines.append("├" + bar + "┤")
    lines.append("│ " + _color(use_color, _BOLD, "Problems (receipt status)"))
    problems = snap.get("problems") or []
    if not problems:
        lines.append("│   " + _color(use_color, _GREEN, "none"))
    for p in problems:
        lines.append(
            "│   "
            + _color(use_color, _RED, f"step{p.get('step')} {p.get('name')}")
            + f" status={p.get('status')} detail={p.get('detail')}"
        )

    lines.append("└" + bar + "┘")
    lines.append(
        _color(
            use_color,
            _DIM,
            "refresh read-only · no writes · Ctrl+C quit · RESOURCE_PROFILES_V002 auto_fill",
        )
    )
    return "\n".join(lines)


def run_tui_loop(
    *,
    nhc0801_root: Path,
    generation_id: str = "nhc0801-g001",
    interval_s: float = 30.0,
    once: bool = False,
    stream: TextIO | None = None,
) -> int:
    """Refresh loop. Returns 0 on clean exit."""

    out = stream or sys.stdout
    use_clear = out.isatty()
    while True:
        frame = render_status_text(
            nhc0801_root=nhc0801_root,
            generation_id=generation_id,
        )
        if use_clear:
            out.write("\033[2J\033[H")
        out.write(frame + "\n")
        out.flush()
        if once:
            return 0
        try:
            time.sleep(max(1.0, float(interval_s)))
        except KeyboardInterrupt:
            out.write("\nTUI exit\n")
            return 0
