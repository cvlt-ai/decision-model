#!/usr/bin/env bash
# Crash-proof supervisor for s1-v5 (4096-context retrain on mixture_v5 = v3 + c2d).
# Watches the in-flight training run (pid arg 1); if it dies BEFORE saving,
# relaunches train_v5.sh (up to N tries, arg 2). On the first "saved" marker it
# runs the full eval chain (VitaminC flip probe headline + regression + JevBench +
# calibration) and commits. Same SIGBUS-death pattern as v4 attempt 1.
#
# Usage: bash scripts/supervise_v5.sh <train_pid> [max_extra_tries]
set -uo pipefail
cd /home/user/decision-model
LOG=results/overnight_v5.log
exec >> "$LOG" 2>&1

TRAIN_PID="${1:-}"
MAX_TRIES="${2:-3}"
CKPT=checkpoints/s1-v5/adapter_model.safetensors

saved() { grep -q '"stage": "saved"' checkpoints/s1-v5_launch.log 2>/dev/null && [ -f "$CKPT" ]; }

echo "===== $(date) supervisor start; watching pid=$TRAIN_PID, up to $MAX_TRIES relaunches ====="
try=0
while :; do
  if saved; then
    echo "===== $(date) s1-v5 saved (after $try relaunches) ====="
    break
  fi
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  # training died without saving
  try=$((try+1))
  if [ "$try" -gt "$MAX_TRIES" ]; then
    echo "===== $(date) ERROR: training died and we've used all $MAX_TRIES relaunches ====="
    tail -25 checkpoints/s1-v5_launch.log
    exit 1
  fi
  echo "===== $(date) training pid=$TRAIN_PID died without saving; relaunch #$try ====="
  mv -f checkpoints/s1-v5_launch.log "checkpoints/s1-v5_launch.dead$try.log" 2>/dev/null || true
  rm -f checkpoints/s1-v5_log.jsonl 2>/dev/null || true
  bash scripts/train_v5.sh > checkpoints/s1-v5_launch.log 2>&1 &
  TRAIN_PID=$!
  echo "relaunched train pid=$TRAIN_PID"
  sleep 120
done

echo "===== $(date) running v5 eval chain ====="
bash scripts/eval_v5.sh

echo "===== $(date) committing ====="
git add results/s1v5_frozen_300.json results/jevbench_s1v5_public231.json \
  results/jevbench_s1v5_public231_4096.json results/calibration_s1v5.json \
  results/vitaminc_flip_s1v5.json results/compare_models_latest.txt 2>/dev/null || true
git add -A scripts/eval_v5.sh scripts/train_v5.sh scripts/supervise_v5.sh scripts/compare_models.py 2>/dev/null || true
git commit -q -m "10.0: s1-v5 (c2d contrastive line at 4096) — VitaminC flip + regression + JevBench" || echo "nothing to commit"

echo "===== $(date) v5 supervisor done ====="
.venv/bin/python - <<'PY' 2>/dev/null
import json, os
def L(p): return json.JSONDecoder().raw_decode(open(p).read())[0] if os.path.exists(p) else None
v5=L("results/vitaminc_flip_s1v5.json"); v3=L("results/vitaminc_flip_s1v3.json")
if v5 and v3:
    print(f"MORNING: VITAMINC FLIP  v3 NEI={v3['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v3['lazy_rate']} "
          f"-> v5 NEI={v5['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={v5['lazy_rate']} "
          f"(acc {v3['accuracy']}->{v5['accuracy']})")
for tag,p in [("v5@1024","results/jevbench_s1v5_public231.json"),
              ("v5@4096","results/jevbench_s1v5_public231_4096.json")]:
    d=L(p)
    if d:
        o=d["overall"]
        print(f"MORNING: {tag} JevBench = {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("MORNING REF: v3 JevBench 0.675/0.693  AlexWortega 0.814  Jev 0.866")
PY
