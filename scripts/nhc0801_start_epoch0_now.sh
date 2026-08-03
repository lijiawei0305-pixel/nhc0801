#!/usr/bin/env bash
# Start live Epoch-0 for g001 Val roots (mindmap step 3).
# Canonical log: live_epoch0_g001.out (legacy aliases still read by stewards).
# Prefer: e0_val_only with --parent-backend gpu (see e0_val_queue_daemon).
set -u
NHC0801="${NHC0801_ROOT:-/home/plab/test/WJW/NHC0801}"
GEN="${GENERATION_ID:-nhc0801-g001}"
E0_GPU="${E0_GPU:-5}"
PARENT_BACKEND="${PARENT_BACKEND:-gpu}"
LOGDIR="$NHC0801/runs/$GEN/logs"
# Canonical write target + legacy symlink for old monitors
E0LOG="$LOGDIR/live_epoch0_g001.out"
E0LOG_LEGACY="$LOGDIR/live_epoch0_02c.out"
LOCK="$LOGDIR/e0_launch.lock"
mkdir -p "$LOGDIR"

for f in "$E0LOG" "$E0LOG_LEGACY" "$LOGDIR/live_epoch0.out" "$LOGDIR/g001_epoch0_rerun.out"; do
  if [ -f "$f" ] && grep -qE "EPOCH0_EXIT|E0_VAL_EXIT" "$f" 2>/dev/null; then
    echo "e0 already finished ($f)"
    exit 0
  fi
done
if [ -f "$LOCK" ]; then
  if pgrep -af "e0_val_only" | grep -q "batch-id g001"; then
    echo "e0 already running (e0_val_only g001)"
    exit 0
  fi
  if pgrep -af "nhc0801_live_orchestrate.py" | grep -q skip-train-live; then
    echo "e0 already running (live_orchestrate)"
    exit 0
  fi
fi

echo "e0_start $(date -u +%Y-%m-%dT%H:%M:%SZ) physical_gpu=$E0_GPU parent=$PARENT_BACKEND" >"$LOCK"
# shellcheck disable=SC1091
source /home/plab/test/WJW/env/envs/mlff.sh
export PYTHONPATH="$NHC0801/src"
export PYTHONUNBUFFERED=1
export CUDA_VISIBLE_DEVICES="$E0_GPU"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-2}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-2}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-2}"
cd "$NHC0801"

# Keep legacy path as symlink so old watches still see the same stream
ln -sfn "live_epoch0_g001.out" "$E0LOG_LEGACY" 2>/dev/null || true

{
  echo "START_EPOCH0_G001 $(date -u +%Y-%m-%dT%H:%M:%SZ) physical_gpu=$E0_GPU parent=$PARENT_BACKEND"
  # Prefer module path: products under epoch0_val_batches/g001/
  if [ "$PARENT_BACKEND" = "gpu" ]; then
    python3 -u -m nhc_deprot.pipeline.e0_val_only \
      --nhc0801-root "$NHC0801" \
      --generation-id "$GEN" \
      --batch-id g001 \
      --val-roots KZYKDQNIIMATMJ-UHFFFAOYSA-N,RMEQTBVGGNKAEQ-UHFFFAOYSA-N \
      --max-steps 100 \
      --parent-backend gpu \
      --cuda-device "$E0_GPU"
  else
    python3 -u scripts/nhc0801_live_orchestrate.py \
      --nhc0801-root "$NHC0801" \
      --generation-id "$GEN" \
      --gpu-index 0 \
      --skip-train-live \
      --allow-epoch0-without-cpu-claim \
      --epoch0-max-steps 100 \
      --claim-interval-s 2
  fi
  echo "EPOCH0_EXIT=$? $(date -u +%Y-%m-%dT%H:%M:%SZ)"
} >>"$E0LOG" 2>&1

echo DONE >"$LOGDIR/epoch0_done.marker"
date -u +%Y-%m-%dT%H:%M:%SZ >>"$LOGDIR/epoch0_done.marker"
python3 -u scripts/nhc0801_check_epoch0_receipts.py \
  --nhc0801-root "$NHC0801" --generation-id "$GEN" \
  >>"$LOGDIR/overnight_continue.out" 2>&1 || true
python3 -u scripts/nhc0801_post_epoch0_continue.py \
  --nhc0801-root "$NHC0801" --generation-id "$GEN" \
  >>"$LOGDIR/overnight_continue.out" 2>&1 || true
