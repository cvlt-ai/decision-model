#!/usr/bin/env bash
# Crash-proof supervisor for s1-v4. Watches the in-flight training run (pid in
# a file); if it dies BEFORE saving, relaunches train_v4.sh (up to N tries).
# On the first "saved" marker it runs the full eval chain and commits.
#
# Usage: supervisor.sh <train_pid> [max_extra_tries]
set -uo pipefail
cd /home/user/decision-model
LOG=results/overnight_v4.log
exec >> "$LOG" 2>&1

TRAIN_PID="${1:-}"
MAX_TRIES="${2:-3}"
CKPT=checkpoints/s1-v4/adapter_model.safetensors

saved() { grep -q '"stage": "saved"' checkpoints/s1-v4_launch.log 2>/dev/null && [ -f "$CKPT" ]; }

echo "===== $(date) supervisor start; watching pid=$TRAIN_PID, up to $MAX_TRIES relaunches ====="
try=0
while :; do
  if saved; then
    echo "===== $(date) s1-v4 saved (after $try relaunches) ====="
    break
  fi
  # is the current training process alive?
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  # training died without saving
  try=$((try+1))
  if [ "$try" -gt "$MAX_TRIES" ]; then
    echo "===== $(date) ERROR: training died and we've used all $MAX_TRIES relaunches ====="
    tail -25 checkpoints/s1-v4_launch.log
    exit 1
  fi
  echo "===== $(date) training pid=$TRAIN_PID died without saving; relaunch #$try ====="
  # reset partial logs so the 'saved' grep and jsonl stay clean
  mv -f checkpoints/s1-v4_launch.log "checkpoints/s1-v4_launch.dead$try.log" 2>/dev/null || true
  rm -f checkpoints/s1-v4_log.jsonl 2>/dev/null || true
  bash scripts/train_v4.sh > checkpoints/s1-v4_launch.log 2>&1 &
  TRAIN_PID=$!
  echo "relaunched train pid=$TRAIN_PID"
  sleep 120   # give it time to load the model / start, before the loop re-checks
done

echo "===== $(date) running v4 eval chain ====="
bash scripts/eval_v4.sh

echo "===== $(date) committing ====="
git add results/s1v4_frozen_300.json results/jevbench_s1v4_public231.json \
  results/jevbench_s1v4_public231_4096.json results/calibration_s1v4.json \
  results/compare_models_latest.txt 2>/dev/null || true
git add -A scripts/eval_v4.sh scripts/train_v4.sh scripts/overnight_v4.sh scripts/supervise_v4.sh \
  scripts/compare_models.py research/07-jevbench-headtohead.md 2>/dev/null || true
git commit -q -m "7.0: s1-v4 (4096-context retrain) — JevBench head-to-head + frozen regression" || echo "nothing to commit"

echo "===== $(date) v4 supervisor done ====="
.venv/bin/python - <<'PY' 2>/dev/null
import json, os
def L(p):
    return json.JSONDecoder().raw_decode(open(p).read())[0] if os.path.exists(p) else None
for tag, p in [("v4@1024", "results/jevbench_s1v4_public231.json"),
               ("v4@4096", "results/jevbench_s1v4_public231_4096.json")]:
    d = L(p)
    if d:
        o = d["overall"]
        print(f"MORNING: {tag} = {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("MORNING REF: v3 0.675/0.693  AlexWortega 0.814  Jev 0.866")
PY
