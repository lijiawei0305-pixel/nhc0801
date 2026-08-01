"""Resource claim sampler tests (canned JSON + injected samples; no SSH required)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from nhc_deprot.generation.layout import init_generation
from nhc_deprot.resources.claim import HostSnapshot, pilot_v002_busy_samples
from nhc_deprot.resources.claim_runner import evaluate_injected_samples, run_resource_claim
from nhc_deprot.resources.host_sampler import (
    expand_cpu_list,
    parse_df_available_bytes,
    parse_meminfo_available_bytes,
    parse_probe_json,
    parse_psi_avg10,
)


def test_expand_cpu_list() -> None:
    assert expand_cpu_list("0,2-5,7") == [0, 2, 3, 4, 5, 7]


def test_parse_meminfo_and_psi_and_df() -> None:
    mem = "MemTotal: 1000 kB\nMemAvailable: 204800 kB\n"
    assert parse_meminfo_available_bytes(mem) == 204800 * 1024
    assert parse_psi_avg10("some avg10=0.12 avg60=0.00\nfull avg10=0.00\n") == pytest.approx(0.12)
    df = "Filesystem 1B-blocks Used Available Use% Mounted\n/dev/sda1 100 20 80 20% /\n"
    assert parse_df_available_bytes(df) == 80


def test_parse_probe_json() -> None:
    payload = {
        "timestamp_utc": "2026-08-02T00:00:00Z",
        "selected_cpus_busy": False,
        "busy_cpu_ids": [],
        "cpu_list": "0,2-27",
        "mem_available_bytes": 250_000_000_000,
        "memory_psi_avg10": 0.0,
        "io_psi_avg10": 0.0,
        "disk_free_bytes": 200_000_000_000,
        "disk_path": "/home/plab/test/WJW/NHC0801",
    }
    snap = parse_probe_json(json.dumps(payload))
    assert snap.selected_cpus_busy is False
    assert snap.mem_available_bytes == 250_000_000_000


def test_injected_v002_reject_writes_receipt(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    out = evaluate_injected_samples(
        layout=layout,
        samples=pilot_v002_busy_samples(),
        profile_id="single_27_physical_v1",
        claim_id="v002_like",
    )
    assert out["status"] == "LIVE_RESOURCE_CLAIM_REJECTED"
    assert any("SELECTED_CPU_BUNDLE_BUSY" in r for r in out["reasons"])
    path = layout.resource_claim_path("v002_like")
    assert path.is_file()
    body = json.loads(path.read_text(encoding="utf-8"))
    assert body["chemistry_run_allowed"] is False
    assert body["mode"] == "injected"


def test_injected_idle_pass(tmp_path: Path) -> None:
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    idle = (
        HostSnapshot(False, 250_000_000_000, 0.0, 0.0, 200_000_000_000, "t0"),
        HostSnapshot(False, 249_000_000_000, 0.0, 0.0, 199_000_000_000, "t1"),
    )
    out = evaluate_injected_samples(
        layout=layout,
        samples=idle,
        profile_id="single_27_physical_v1",
        claim_id="idle_pass",
    )
    assert out["status"] == "LIVE_RESOURCE_CLAIM_PASS"
    assert out["receipt"]["chemistry_run_allowed"] is False  # user gate still closed


def test_local_probe_if_linux(tmp_path: Path) -> None:
    import sys

    if sys.platform != "linux" and not Path("/proc/stat").exists():
        pytest.skip("local probe needs Linux /proc")
    layout, _, _ = init_generation(nhc0801_root=tmp_path / "NHC0801")
    # Use cpu 0 only for portability
    from nhc_deprot.resources.host_sampler import ProbeRequest, take_snapshot

    snap = take_snapshot(ProbeRequest(cpu_list="0", disk_path=str(tmp_path), mode="local"))
    assert snap.mem_available_bytes > 0
    assert snap.disk_free_bytes >= 0
