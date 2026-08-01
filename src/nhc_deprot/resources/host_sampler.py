"""Read-only host resource sampling (local or SSH).

Never starts chemistry, never writes outside an explicit receipt path provided
by the caller. Parses standard Linux /proc and free/df output so unit tests
can inject canned text without a live host.
"""

from __future__ import annotations

import json
import re
import subprocess
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Final, Sequence

from nhc_deprot.resources.claim import HostSnapshot

# Remote probe: one JSON object on stdout (read-only).
_REMOTE_PROBE_PY: Final = r"""
import json, os, re, time
from datetime import datetime, timezone

def expand_cpu_list(spec):
    cpus = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))

def read_cpu_times():
    out = {}
    with open("/proc/stat") as f:
        for line in f:
            if not line.startswith("cpu"):
                continue
            parts = line.split()
            name = parts[0]
            if name == "cpu":
                continue
            if not name[3:].isdigit():
                continue
            vals = list(map(int, parts[1:]))
            out[int(name[3:])] = vals
    return out

def cpu_busy(cpu_ids, window=0.25, util_threshold=0.05):
    t0 = read_cpu_times()
    time.sleep(window)
    t1 = read_cpu_times()
    busy_ids = []
    for cid in cpu_ids:
        if cid not in t0 or cid not in t1:
            busy_ids.append(cid)
            continue
        d0, d1 = t0[cid], t1[cid]
        # user nice system idle iowait irq softirq steal
        idle0 = d0[3] + (d0[4] if len(d0) > 4 else 0)
        idle1 = d1[3] + (d1[4] if len(d1) > 4 else 0)
        total0, total1 = sum(d0), sum(d1)
        dt, di = total1 - total0, idle1 - idle0
        if dt <= 0:
            continue
        util = 1.0 - (di / dt)
        if util > util_threshold:
            busy_ids.append(cid)
    return len(busy_ids) > 0, busy_ids

def psi_avg10(path):
    try:
        with open(path) as f:
            text = f.read()
    except OSError:
        return 0.0
    # some avg10=0.00
    m = re.search(r"some avg10=([0-9.]+)", text)
    return float(m.group(1)) if m else 0.0

def mem_available():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    return 0

def disk_free(path):
    st = os.statvfs(path)
    return int(st.f_bavail * st.f_frsize)

cpu_spec = os.environ.get("NHC0801_CPU_LIST", "0")
cpu_ids = expand_cpu_list(cpu_spec)
disk_path = os.environ.get("NHC0801_DISK_PATH", "/")
busy, busy_ids = cpu_busy(cpu_ids)
payload = {
    "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    "selected_cpus_busy": bool(busy),
    "busy_cpu_ids": busy_ids,
    "cpu_list": cpu_spec,
    "mem_available_bytes": mem_available(),
    "memory_psi_avg10": psi_avg10("/proc/pressure/memory"),
    "io_psi_avg10": psi_avg10("/proc/pressure/io"),
    "disk_free_bytes": disk_free(disk_path),
    "disk_path": disk_path,
}
print(json.dumps(payload, sort_keys=True))
"""


class HostSamplerError(RuntimeError):
    """Host sampling failed closed."""


def expand_cpu_list(spec: str) -> list[int]:
    """Expand '0,2-5,7' → sorted unique ints."""

    cpus: list[int] = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            cpus.extend(range(int(a), int(b) + 1))
        else:
            cpus.append(int(part))
    return sorted(set(cpus))


def parse_probe_json(text: str) -> HostSnapshot:
    """Parse one JSON probe line into HostSnapshot."""

    try:
        payload = json.loads(text.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError) as exc:
        raise HostSamplerError(f"invalid probe JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise HostSamplerError("probe JSON root must be object")
    return HostSnapshot(
        selected_cpus_busy=bool(payload.get("selected_cpus_busy")),
        mem_available_bytes=int(payload.get("mem_available_bytes") or 0),
        memory_psi_avg10=float(payload.get("memory_psi_avg10") or 0.0),
        io_psi_avg10=float(payload.get("io_psi_avg10") or 0.0),
        disk_free_bytes=int(payload.get("disk_free_bytes") or 0),
        timestamp_utc=str(payload.get("timestamp_utc") or "") or None,
        notes=tuple(
            str(x)
            for x in (
                f"cpu_list={payload.get('cpu_list')}",
                f"busy_cpu_ids={payload.get('busy_cpu_ids')}",
                f"disk_path={payload.get('disk_path')}",
            )
        ),
    )


def parse_meminfo_available_bytes(meminfo_text: str) -> int:
    for line in meminfo_text.splitlines():
        if line.startswith("MemAvailable:"):
            return int(line.split()[1]) * 1024
    raise HostSamplerError("MemAvailable not found in meminfo")


def parse_psi_avg10(psi_text: str) -> float:
    match = re.search(r"some avg10=([0-9.]+)", psi_text)
    if not match:
        return 0.0
    return float(match.group(1))


def parse_df_available_bytes(df_text: str) -> int:
    """Parse ``df -B1 -P <path>`` second line available column."""

    lines = [ln for ln in df_text.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise HostSamplerError("df output too short")
    parts = lines[-1].split()
    if len(parts) < 4:
        raise HostSamplerError("df line malformed")
    return int(parts[3])


@dataclass(frozen=True, slots=True)
class ProbeRequest:
    cpu_list: str
    disk_path: str = "/"
    mode: str = "local"  # local | ssh
    ssh_alias: str | None = None
    ssh_extra_args: tuple[str, ...] = ()


def run_local_probe(request: ProbeRequest, *, timeout_s: float = 30.0) -> HostSnapshot:
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "NHC0801_CPU_LIST": request.cpu_list,
        "NHC0801_DISK_PATH": request.disk_path,
    }
    result = subprocess.run(
        ["python3", "-c", _REMOTE_PROBE_PY],
        capture_output=True,
        text=True,
        env=env,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise HostSamplerError(
            f"local probe failed ({result.returncode}): {result.stderr.strip()[:500]}"
        )
    return parse_probe_json(result.stdout)


def run_ssh_probe(request: ProbeRequest, *, timeout_s: float = 60.0) -> HostSnapshot:
    if not request.ssh_alias:
        raise HostSamplerError("ssh_alias required for ssh mode")
    # Feed probe script on stdin (avoids shell-quoting multi-line -c payloads).
    # Env vars set in the remote shell before python3 reads stdin.
    remote_shell = (
        f"NHC0801_CPU_LIST={request.cpu_list} "
        f"NHC0801_DISK_PATH={request.disk_path} "
        "python3 -"
    )
    cmd = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=15",
        *request.ssh_extra_args,
        request.ssh_alias,
        remote_shell,
    ]
    result = subprocess.run(
        cmd,
        input=_REMOTE_PROBE_PY,
        capture_output=True,
        text=True,
        timeout=timeout_s,
        check=False,
    )
    if result.returncode != 0:
        raise HostSamplerError(
            f"ssh probe failed ({result.returncode}): {result.stderr.strip()[:800]}"
        )
    return parse_probe_json(result.stdout)


def take_snapshot(request: ProbeRequest) -> HostSnapshot:
    if request.mode == "local":
        return run_local_probe(request)
    if request.mode == "ssh":
        return run_ssh_probe(request)
    raise HostSamplerError(f"unknown sample mode: {request.mode}")


def take_two_samples(
    request: ProbeRequest,
    *,
    interval_s: float = 5.0,
) -> tuple[HostSnapshot, HostSnapshot, dict[str, Any]]:
    """Two-sample claim protocol (read-only)."""

    s0 = take_snapshot(request)
    time.sleep(max(0.0, interval_s))
    s1 = take_snapshot(request)
    meta = {
        "interval_s": interval_s,
        "mode": request.mode,
        "cpu_list": request.cpu_list,
        "disk_path": request.disk_path,
        "ssh_alias_set": bool(request.ssh_alias),
        "sampled_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    return s0, s1, meta


def load_ssh_alias_from_config(path: Path) -> str | None:
    """Read ssh_alias from configs/server.local.yaml without importing secrets broadly."""

    if not path.is_file():
        return None
    try:
        import yaml
    except ImportError:
        # minimal parse
        for line in path.read_text(encoding="utf-8").splitlines():
            if "ssh_alias" in line and ":" in line:
                return line.split(":", 1)[1].strip().strip("'\"")
        return None
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        return None
    conn = payload.get("connection") or {}
    if isinstance(conn, dict) and conn.get("ssh_alias"):
        return str(conn["ssh_alias"])
    return None


def load_project_root_from_config(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        import yaml

        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        remote = (payload or {}).get("remote") or {}
        if isinstance(remote, dict) and remote.get("project_root"):
            return str(remote["project_root"])
    except Exception:  # noqa: BLE001
        return None
    return None


def snapshot_to_dict(snapshot: HostSnapshot) -> dict[str, Any]:
    return asdict(snapshot)
