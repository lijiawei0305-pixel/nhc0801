#!/usr/bin/env python3
"""NHC0801 compute steward — self-heal pilot teacher + g00N teacher queue + e0.

Authority: mindmap.md + docs/contracts/COMPUTE_DISPATCH_V001.md + AGENTS naming.
Does NOT open Final Test. Does NOT retrain unless already chained.
Actions:
  - keep GPU teacher queue daemon alive (products: teacher_gpu_g00N/)
  - force-reap finished/zombie g00N teacher jobs so free GPUs are reclaimed
  - keep CPU-teacher → epoch0 primary/backup chains healthy
  - never start e0 while CPU teacher still running
  - resolve logs via canonical + legacy basenames (*_g001.out and *_02c.out)
"""

from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

NHC = Path(os.environ.get("NHC0801_ROOT", "/home/plab/test/WJW/NHC0801"))
GEN = os.environ.get("GENERATION_ID", "nhc0801-g001")
BASE = NHC / "runs" / GEN
LOGDIR = BASE / "logs"
STATE = BASE / "gpu_teacher_queue" / "state.json"
STATE_LEGACY = BASE / "gpu_autofill" / "state.json"
STEWARD_LOG = LOGDIR / "compute_steward.log"
POLL = int(os.environ.get("STEWARD_POLL_S", "60"))


def utc() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    LOGDIR.mkdir(parents=True, exist_ok=True)
    line = f"[{utc()}] {msg}"
    print(line, flush=True)
    with STEWARD_LOG.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def cmdlines() -> list[tuple[str, str]]:
    out = []
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            c = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode(
                errors="replace"
            )
        except OSError:
            continue
        if c.strip():
            out.append((name, c))
    return out


def has_proc(substr: str) -> bool:
    return any(substr in c for _, c in cmdlines())


def pid_state(pid: int) -> str:
    try:
        for line in Path(f"/proc/{pid}/status").read_text(encoding="utf-8").splitlines():
            if line.startswith("State:"):
                return line.split()[1]
    except OSError:
        return "gone"
    return "?"


def _teacher_state_path() -> Path:
    if STATE.is_file():
        return STATE
    if STATE_LEGACY.is_file():
        return STATE_LEGACY
    return STATE


def force_reap_gpu_teacher_queue() -> int:
    sp = _teacher_state_path()
    if not sp.is_file():
        return 0
    s = json.loads(sp.read_text(encoding="utf-8"))
    running = dict(s.get("running") or {})
    done = list(s.get("done") or [])
    failed = list(s.get("failed") or [])
    n = 0
    for key, info in list(running.items()):
        logp = Path(info.get("log") or "")
        text = logp.read_text(encoding="utf-8", errors="replace") if logp.is_file() else ""
        pid = int(info.get("pid") or -1)
        st = pid_state(pid) if pid > 0 else "gone"
        log_done = "JOB_EXIT" in text
        if st in {"Z", "gone"} or log_done:
            status = "FAIL"
            for line in reversed(text.splitlines()):
                if line.startswith("JOB_EXIT"):
                    status = line.split()[-1]
                    break
                if line.startswith("{"):
                    try:
                        status = json.loads(line).get("status") or status
                    except json.JSONDecodeError:
                        pass
            rec = {
                "key": key,
                "status": status,
                "gpu_index": info.get("gpu_index"),
                "batch_id": info.get("batch_id"),
                "finished_at_utc": utc(),
                "log": str(logp),
                "steward_reap": True,
                "pid_state": st,
            }
            (done if status == "PASS" else failed).append(rec)
            running.pop(key, None)
            n += 1
            if st not in {"gone"} and pid > 0:
                try:
                    os.kill(pid, signal.SIGKILL)
                except OSError:
                    pass
            log(f"reap {key} status={status} pid_state={st} gpu={info.get('gpu_index')}")
    if n:
        s["running"] = running
        s["done"] = done
        s["failed"] = failed
        s["updated_at_utc"] = utc()
        sp.write_text(json.dumps(s, indent=2) + "\n", encoding="utf-8")
    return n


def ensure_gpu_teacher_daemon() -> None:
    if has_proc("gpu_teacher_daemon.py") or has_proc("gpu_autofill_daemon.py"):
        return
    # Honor explicit pause (obsolete max_steps=100 expansion / free GPUs for e0).
    pause_note = LOGDIR / "gpu_teacher_pause.note"
    if pause_note.is_file():
        log("gpu-teacher paused (gpu_teacher_pause.note present) — not restarting")
        return
    if STATE.is_file():
        try:
            st = json.loads(STATE.read_text(encoding="utf-8"))
            if st.get("paused") is True:
                log(
                    f"gpu-teacher paused ({st.get('pause_reason') or 'state.paused'}) "
                    "— not restarting"
                )
                return
        except (OSError, json.JSONDecodeError):
            pass
    log("gpu-teacher queue daemon missing — restarting with max-steps 250")
    out = LOGDIR / "gpu_teacher_daemon.out"
    cmd = f"""
source /home/plab/test/WJW/env/envs/mlff.sh
export PYTHONPATH={NHC}/src
export PYTHONUNBUFFERED=1
cd {NHC}
python3 -u scripts/nhc0801_gpu_teacher_daemon.py \
  --nhc0801-root {NHC} \
  --generation-id {GEN} \
  --pool-csv {NHC}/docs/contracts/RIGID_SMALL_NHC_POOL_V001.csv \
  --gpu-ids 0,1,2,3,4,5,6,7 \
  --host-threads 2 --max-steps 250 --poll-seconds 15 \
  --batch-size-roots 5
"""
    with out.open("a", encoding="utf-8") as fh:
        fh.write(f"\n=== steward restart {utc()} max-steps=250 ===\n")
        subprocess.Popen(
            ["bash", "-lc", cmd],
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def ensure_cpu_e0_backup() -> None:
    if has_proc("cpu_teacher_then_e0_backup"):
        return
    # Do not spawn legacy CPU backup while teacher expansion is paused for e0 mainline.
    if (LOGDIR / "gpu_teacher_pause.note").is_file():
        return
    if has_proc("e0_val_only"):
        # Live 4-GPU e0 already owns Val baseline — do not start competing backup.
        return
    script = LOGDIR / "cpu_teacher_then_e0_backup.sh"
    if not script.is_file():
        log("cpu_e0 backup script missing — skip recreate (primary chain may still own e0)")
        return
    log("cpu→e0 backup missing — restarting")
    with (LOGDIR / "cpu_teacher_then_e0_backup_nohup.out").open("a") as fh:
        subprocess.Popen(
            ["bash", str(script)],
            stdout=fh,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )


def teacher_alive() -> bool:
    for _, c in cmdlines():
        if "teacher_wave_02c.py" in c and "gpu" not in c and "autofill" not in c:
            # exclude false positives
            if "teacher_wave_gpu" in c:
                continue
            return True
    # more precise
    for _, c in cmdlines():
        if "nhc0801_teacher_wave_02c.py" in c:
            return True
    return False


def _first_existing_log(*names: str) -> Path | None:
    for name in names:
        p = LOGDIR / name
        if p.is_file():
            return p
    return None


def teacher_exited() -> bool:
    tlog = _first_existing_log("teacher_wave_g001.out", "teacher_wave_02c.out")
    if tlog is None:
        return False
    return "TEACHER_WAVE_EXIT" in tlog.read_text(encoding="utf-8", errors="replace")


def e0_running_or_done() -> bool:
    for name in (
        "live_epoch0_g001.out",
        "live_epoch0_02c.out",
        "live_epoch0.out",
        "g001_epoch0_rerun.out",
    ):
        e0log = LOGDIR / name
        if e0log.is_file():
            txt = e0log.read_text(encoding="utf-8", errors="replace")
            if "EPOCH0_EXIT" in txt or "E0_VAL_EXIT" in txt or "START_EPOCH0" in txt:
                return True
    qlog = BASE / "epoch0_val_queue" / "job_g001.out"
    if qlog.is_file():
        txt = qlog.read_text(encoding="utf-8", errors="replace")
        if "E0_VAL_EXIT" in txt or "g001 Epoch-0" in txt:
            return True
    for _, c in cmdlines():
        # real python only — ignore bash monitors whose cmdline embeds the string
        if "python" not in c:
            continue
        if "nhc0801_live_orchestrate.py" in c and "skip-train-live" in c:
            return True
        if "e0_val_only" in c and "g001" in c:
            return True
    return False


def maybe_start_e0_if_stranded() -> None:
    """If teacher finished and neither primary chain nor e0 is active, start e0 once."""
    if teacher_alive():
        return
    if not teacher_exited() and has_proc("wave_02c_chain"):
        # chain may still be between teacher exit and e0 start
        return
    if e0_running_or_done():
        return
    if teacher_exited() or (not teacher_alive() and not has_proc("wave_02c_chain")):
        # wait for primary chain briefly
        time.sleep(30)
        if e0_running_or_done():
            return
        lock = LOGDIR / "e0_backup.lock"
        if lock.exists():
            return
        if not teacher_exited() and teacher_alive():
            return
        # only if teacher wave log shows exit OR teacher processes gone for long
        if not teacher_exited():
            # teacher process gone without EXIT line — still start e0 after marking
            log("WARN teacher processes gone without TEACHER_WAVE_EXIT — steward starts e0")
        log("STARTING g001 Epoch-0 (stranded after CPU teacher)")
        lock.write_text(str(os.getpid()), encoding="utf-8")
        e0log = LOGDIR / "live_epoch0_g001.out"
        legacy = LOGDIR / "live_epoch0_02c.out"
        try:
            if not legacy.exists():
                legacy.symlink_to("live_epoch0_g001.out")
        except OSError:
            pass
        cmd = f"""
source /home/plab/test/WJW/env/envs/mlff.sh
export PYTHONPATH={NHC}/src
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
cd {NHC}
echo START_EPOCH0_G001 $(date -u +%Y-%m-%dT%H:%M:%SZ) STEWARD parent=gpu
python3 -u -m nhc_deprot.pipeline.e0_val_only \
  --nhc0801-root {NHC} --generation-id {GEN} \
  --batch-id g001 \
  --val-roots KZYKDQNIIMATMJ-UHFFFAOYSA-N,RMEQTBVGGNKAEQ-UHFFFAOYSA-N \
  --max-steps 100 --parent-backend gpu --cuda-device 0
echo EPOCH0_EXIT=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)
"""
        with e0log.open("a", encoding="utf-8") as fh:
            subprocess.Popen(
                ["bash", "-lc", cmd],
                stdout=fh,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )


def snapshot() -> str:
    busy = set()
    for name in os.listdir("/proc"):
        if not name.isdigit():
            continue
        try:
            c = open(f"/proc/{name}/cmdline", "rb").read().replace(b"\0", b" ").decode()
        except OSError:
            continue
        if "nhc0801_pyscf_parent_worker" not in c:
            continue
        try:
            for e in open(f"/proc/{name}/environ", "rb").read().split(b"\0"):
                if e.startswith(b"CUDA_VISIBLE_DEVICES="):
                    v = e.split(b"=", 1)[1].decode().strip()
                    if v.isdigit():
                        busy.add(int(v))
        except OSError:
            pass
    free = [i for i in range(8) if i not in busy]
    af = ""
    if STATE.is_file():
        s = json.loads(STATE.read_text(encoding="utf-8"))
        af = (
            f"q={len(s.get('queue') or [])} run={len(s.get('running') or {})} "
            f"done={len(s.get('done') or [])} fail={len(s.get('failed') or [])}"
        )
    return (
        f"teacher_alive={teacher_alive()} teacher_exit={teacher_exited()} "
        f"e0={e0_running_or_done()} gpu_teacher="
        f"{has_proc('gpu_teacher_daemon') or has_proc('gpu_autofill_daemon')} "
        f"busy_gpu={sorted(busy)} free_gpu={free} AF[{af}]"
    )


def main() -> int:
    log("steward start " + snapshot())
    while True:
        try:
            n = force_reap_gpu_teacher_queue()
            if n:
                log(f"reaped {n} g00N teacher jobs")
            ensure_gpu_teacher_daemon()
            ensure_cpu_e0_backup()
            maybe_start_e0_if_stranded()
            log("ok " + snapshot())
        except Exception as exc:  # noqa: BLE001
            log(f"ERROR {type(exc).__name__}: {exc}")
        time.sleep(POLL)


if __name__ == "__main__":
    raise SystemExit(main())
