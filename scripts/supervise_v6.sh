#!/usr/bin/env bash
# Crash-proof supervisor for s1-v6 (4096-context, mixture_v6 = v5 + TSI breadth).
#
# A ~29 h run (8,078 steps) has a real risk of the intermittent SIGBUS we've seen
# in v3/v4/v5. train_v6.sh saves a periodic checkpoint every 1000 steps
# (~4.3 h), so this supervisor handles a death in tiers:
#   * clean end ("saved" marker)              -> eval the full run.
#   * crash PAST ~60% with a periodic ckpt     -> eval the PARTIAL (better than
#     relaunching ~29 h from zero; a 90%-trained adapter is informative).
#   * crash EARLY (no ckpt / < 60%)            -> relaunch from zero (<= MAX_TRIES).
# On completion it runs the eval chain (mmlu_pro knowledge headline + VitaminC
# flip + regression + JevBench + calibration) and commits.
#
# Usage: bash scripts/supervise_v6.sh <train_pid> [max_extra_tries]
set -uo pipefail
cd /home/user/decision-model
LOG=results/overnight_v6.log
exec >> "$LOG" 2>&1

TRAIN_PID="${1:-}"
MAX_TRIES="${2:-3}"
CKPT=checkpoints/s1-v6/adapter_model.safetensors
EXPECTED_STEPS="${EXPECTED_STEPS:-8078}"
THRESH=$(( EXPECTED_STEPS * 60 / 100 ))   # evaluate a partial past 60%

saved() { grep -q '"stage": "saved"' checkpoints/s1-v6_launch.log 2>/dev/null && [ -f "$CKPT" ]; }
last_step() { tail -1 checkpoints/s1-v6_log.jsonl 2>/dev/null | sed -n 's/.*"step": *\([0-9]*\).*/\1/p'; }

echo "===== $(date) supervisor start; watching pid=$TRAIN_PID, up to $MAX_TRIES relaunches; partial-eval threshold=$THRESH steps ====="
try=0
while :; do
  if saved; then
    echo "===== $(date) s1-v6 saved (after $try relaunches) ====="
    break
  fi
  if [ -n "$TRAIN_PID" ] && kill -0 "$TRAIN_PID" 2>/dev/null; then
    sleep 60
    continue
  fi
  # training died without a final "saved" marker
  steps="${$(last_step):-0}"
  if [ -f "$CKPT" ] && [ "$steps" -ge "$THRESH" ]; then
    echo "===== $(date) training died at step $steps (>= $THRESH = 60%) with a periodic checkpoint; evaluating the PARTIAL instead of relaunching ====="
    break
  fi
  try=$((try+1))
  if [ "$try" -gt "$MAX_TRIES" ]; then
    echo "===== $(date) ERROR: training died at step $steps and we've used all $MAX_TRIES relaunches ====="
    tail -25 checkpoints/s1-v6_launch.log
    exit 1
  fi
  echo "===== $(date) training pid=$TRAIN_PID died at step $steps (early, < $THRESH or no ckpt); relaunch #$try ====="
  mv -f checkpoints/s1-v6_launch.log "checkpoints/s1-v6_launch.dead$try.log" 2>/dev/null || true
  rm -f checkpoints/s1-v6_log.jsonl 2>/dev/null || true
  bash scripts/train_v6.sh > checkpoints/s1-v6_launch.log 2>&1 &
  TRAIN_PID=$!
  echo "relaunched train pid=$TRAIN_PID"
  sleep 120
done

echo "===== $(date) running v6 eval chain ====="
bash scripts/eval_v6.sh

echo "===== $(date) committing ====="
git add results/s1v6_frozen_300.json results/jevbench_s1v6_public231.json \
  results/jevbench_s1v6_public231_4096.json results/calibration_s1v6.json \
  results/vitaminc_flip_s1v6.json results/vitaminc_s1v6.json results/compare_models_latest.txt 2>/dev/null || true
git add -A scripts/eval_v6.sh scripts/train_v6.sh scripts/supervise_v6.sh scripts/compare_models.py scripts/train_s1.py 2>/dev/null || true
git commit -q -m "11.1: s1-v6 (TSI breadth at 4096) — mmlu_pro knowledge + VitaminC flip + regression + JevBench" || echo "nothing to commit"

echo "===== $(date) v6 supervisor done ====="
.venv/bin/python - <<'PY' 2>/dev/null
import json, os
def L(p): return json.JSONDecoder().raw_decode(open(p).read())[0] if os.path.exists(p) else None
def J(p): return json.load(open(p)) if os.path.exists(p) else None
v6=J("results/s1v6_frozen_300.json"); v5=J("results/s1v5_frozen_300.json")
if v6 and v5:
    m6, m5 = v6["macro"], v5["macro"]
    k6, k5 = v6["families"]["mmlu_pro"]["accuracy"], v5["families"]["mmlu_pro"]["accuracy"]
    print(f"MORNING: v6 macro={m6['accuracy']:.3f} (v5 {m5['accuracy']:.3f})  "
          f"mmlu_pro={k6:.3f} (v5 {k5:.3f})  neg={m6.get('negation_mean_violation')}  "
          f"Brier={m6['brier']:.3f} (v5 {m5['brier']:.3f})")
vf6=L("results/vitaminc_flip_s1v6.json"); vf3=L("results/vitaminc_flip_s1v3.json")
if vf6 and vf3:
    print(f"MORNING: VITAMINC FLIP  v3 NEI={vf3['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={vf3['lazy_rate']} "
          f"-> v6 NEI={vf6['per_class']['NOT ENOUGH INFO']['acc']:.3f} lazy={vf6['lazy_rate']} "
          f"(acc {vf3['accuracy']}->{vf6['accuracy']})")
for tag,p in [("v6@1024","results/jevbench_s1v6_public231.json"),
              ("v6@4096","results/jevbench_s1v6_public231_4096.json")]:
    d=L(p)
    if d:
        o=d["overall"]
        print(f"MORNING: {tag} JevBench = {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("MORNING REF: v5 JevBench 0.7489 (hard 59/111)  AlexWortega 0.814  Jev 1.13 0.866")
PY
