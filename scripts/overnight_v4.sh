#!/usr/bin/env bash
# Overnight orchestrator for s1-v4: wait for training to save, then run the
# full eval chain (JevBench @4096 + @1024, frozen suite, calibration, compare)
# and commit, so it runs hands-off.
set -uo pipefail
cd /home/user/decision-model
LOG=results/overnight_v4.log
exec >> "$LOG" 2>&1

echo "===== $(date) v4 orchestrator start; waiting for checkpoint ====="
CKPT=checkpoints/s1-v4/adapter_model.safetensors
for i in $(seq 1 1440); do  # up to 24h, poll 60s
  if grep -q '"stage": "saved"' checkpoints/s1-v4_launch.log 2>/dev/null && [ -f "$CKPT" ]; then
    echo "s1-v4 training complete after ~$((i)) min"
    break
  fi
  sleep 60
done
if [ ! -f "$CKPT" ]; then
  echo "ERROR: s1-v4 adapter missing after 24h — training likely crashed:"
  tail -20 checkpoints/s1-v4_launch.log
  exit 1
fi

echo "===== $(date) launching v4 eval chain ====="
bash scripts/eval_v4.sh

echo "===== $(date) committing ====="
git add results/s1v4_frozen_300.json results/jevbench_s1v4_public231.json \
  results/jevbench_s1v4_public231_4096.json results/calibration_s1v4.json \
  results/compare_models_latest.txt 2>/dev/null || true
git add -A scripts/eval_v4.sh scripts/train_v4.sh scripts/compare_models.py \
  scripts/jevbench_run.py research/07-jevbench-headtohead.md 2>/dev/null || true
git commit -q -m "7.0: s1-v4 (4096-context retrain) — JevBench head-to-head + frozen regression" || echo "nothing to commit"

echo "===== $(date) v4 orchestrator done ====="
cat results/compare_models_latest.txt 2>/dev/null | tail -20
.venv/bin/python - <<'PY' 2>/dev/null
import json
def L(p):
    import os
    if not os.path.exists(p): return None
    return json.JSONDecoder().raw_decode(open(p).read())[0]
for tag,p in [("v4@1024","results/jevbench_s1v4_public231.json"),
              ("v4@4096","results/jevbench_s1v4_public231_4096.json")]:
    d=L(p)
    if d:
        o=d["overall"]
        print(f"MORNING: {tag} = {o['correct']}/{o['n']} = {o['accuracy']:.4f} "
              f"(hard {d['tiers']['hard']['correct']}/111)")
print("MORNING REF: v3 0.675/0.693  AlexWortega 0.814  Jev 0.866")
PY
