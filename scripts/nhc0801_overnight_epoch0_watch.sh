#!/usr/bin/env bash
# Overnight watcher: wait for live epoch-0, then run post-epoch0 mindmap continue.
# No systemd. Logs under runs/<gen>/logs/. Safe to re-run (marker + lock).
set -u
NHC0801_ROOT="${NHC0801_ROOT:-/home/plab/test/WJW/NHC0801}"
GEN="${GENERATION_ID:-nhc0801-g001}"
LOGDIR="$NHC0801_ROOT/runs/$GEN/logs"
# Canonical + legacy log basenames (any may receive EPOCH0_EXIT / E0_VAL_EXIT)
E0LOGS=(
  "$LOGDIR/live_epoch0_g001.out"
  "$LOGDIR/live_epoch0_02c.out"
  "$LOGDIR/live_epoch0.out"
  "$LOGDIR/g001_epoch0_rerun.out"
  "$NHC0801_ROOT/runs/$GEN/epoch0_val_queue/job_g001.out"
)
MARK="$LOGDIR/epoch0_done.marker"
LOCK="$LOGDIR/overnight_watch.lock"
CONTLOG="$LOGDIR/overnight_continue.out"
ORCH_SUBSTR="nhc0801_live_orchestrate.py"
E0_SUBSTR="e0_val_only"

mkdir -p "$LOGDIR"
exec >>"$LOGDIR/overnight_watch.log" 2>&1
echo "=== overnight watch start $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="

# single instance
if [ -f "$LOCK" ]; then
  old=$(cat "$LOCK" 2>/dev/null || true)
  if [ -n "${old:-}" ] && kill -0 "$old" 2>/dev/null; then
    echo "another overnight watch PID $old still running; exit"
    exit 0
  fi
fi
echo $$ >"$LOCK"
trap 'rm -f "$LOCK"' EXIT

e0_log_has_exit() {
  for f in "${E0LOGS[@]}"; do
    [ -f "$f" ] || continue
    if grep -qE "EPOCH0_EXIT|E0_VAL_EXIT" "$f" 2>/dev/null; then
      echo "seen exit in $f"
      return 0
    fi
  done
  return 1
}

wait_for_epoch0() {
  while true; do
    if e0_log_has_exit; then
      return 0
    fi
    # process alive?
    if ! ps aux | grep -E "$ORCH_SUBSTR|$E0_SUBSTR" | grep -v grep >/dev/null 2>&1; then
      sleep 5
      if e0_log_has_exit; then
        return 0
      fi
      echo "orchestrate process gone without EPOCH0_EXIT"
      return 1
    fi
    # light heartbeat every ~10 min
    echo "heartbeat $(date -u +%Y-%m-%dT%H:%M:%SZ) worker_cpu=$(ps -C python -o pcpu= 2>/dev/null | head -1 || echo ?)"
    sleep 120
  done
}

if [ -f "$MARK" ] && grep -qE '^DONE' "$MARK" 2>/dev/null; then
  echo "marker already DONE; skip wait"
else
  if wait_for_epoch0; then
    {
      echo DONE
      date -u +%Y-%m-%dT%H:%M:%SZ
      grep EPOCH0_EXIT "$E0LOG" | tail -3 || true
      tail -20 "$E0LOG" || true
    } >"$MARK"
    echo "wrote $MARK"
  else
    {
      echo FAILED
      date -u +%Y-%m-%dT%H:%M:%SZ
      tail -40 "$E0LOG" || true
    } >"$MARK"
    echo "epoch0 FAILED or aborted; still run audit"
  fi
fi

# env for post steps
# shellcheck disable=SC1091
source /home/plab/test/WJW/env/envs/mlff.sh
export PYTHONPATH="$NHC0801_ROOT/src"
export PYTHONUNBUFFERED=1
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-8}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-8}"
cd "$NHC0801_ROOT"

echo "=== post_epoch0_continue $(date -u +%Y-%m-%dT%H:%M:%SZ) ===" | tee -a "$CONTLOG"
# After e0: audit receipts → shortlist → sci-val dry-run → freeze → pipeline status
# (Final Test never auto; live sci-val for finetuned weights is a separate gate)
python3 -u scripts/nhc0801_post_epoch0_continue.py \
  --nhc0801-root "$NHC0801_ROOT" \
  --generation-id "$GEN" \
  >>"$CONTLOG" 2>&1
rc=$?
echo "post_epoch0_continue exit=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$CONTLOG"

python3 -u scripts/nhc0801_check_epoch0_receipts.py \
  --nhc0801-root "$NHC0801_ROOT" \
  --generation-id "$GEN" \
  >>"$CONTLOG" 2>&1 || true

python3 -u scripts/nhc0801_pipeline_status.py \
  --nhc0801-root "$NHC0801_ROOT" \
  --generation-id "$GEN" \
  --write \
  >>"$CONTLOG" 2>&1 || true

python3 -u scripts/nhc0801_tui.py \
  --nhc0801-root "$NHC0801_ROOT" \
  --generation-id "$GEN" \
  --once \
  >>"$CONTLOG" 2>&1 || true

echo "=== overnight watch finished rc=$rc $(date -u +%Y-%m-%dT%H:%M:%SZ) ==="
exit "$rc"
