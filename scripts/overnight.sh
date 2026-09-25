#!/usr/bin/env bash
# Overnight orchestrator: wait for s1-v3 training to finish, then score +
# calibrate + compare + commit, so it runs hands-off while the user sleeps.
# Completion signal = training wrote "stage": "saved" AND the adapter exists.
set -uo pipefail
cd /home/user/decision-model
LOG=results/overnight_orchestrator.log
exec >> "$LOG" 2>&1

echo "===== $(date) orchestrator start; waiting for s1-v3 checkpoint ====="
CKPT=checkpoints/s1-v3/adapter_model.safetensors
for i in $(seq 1 1440); do  # up to 24h, poll every 60s
  if grep -q '"stage": "saved"' checkpoints/s1-v3_launch.log 2>/dev/null && [ -f "$CKPT" ]; then
    echo "s1-v3 training complete after ~$((i)) min polling"
    break
  fi
  sleep 60
done

# fail loudly if no checkpoint (e.g. OOM/crash) — don't score an empty dir
if [ ! -f "$CKPT" ]; then
  echo "ERROR: s1-v3 adapter missing after 24h — training likely crashed. Last log lines:"
  tail -20 checkpoints/s1-v3_launch.log
  exit 1
fi

echo "===== $(date) launching v3 eval + calibrate + compare ====="
bash scripts/eval_v3.sh

echo "===== $(date) committing results ====="
git add results/s1v3_frozen_300.json results/s1v3_perm_clean.json results/calibration_s1v3.json \
  results/s1v3_perm_robustness.json 2>/dev/null || true
git add results/*.json 2>/dev/null || true
git add -A scripts/eval_v3.sh scripts/calibrate.py scripts/compare_models.py 2>/dev/null || true
git commit -q -m "6.0: s1-v3 eval (order-aug + boolq) + offline temperature calibration + consolidated comparison" || echo "nothing to commit"

# refresh STATE.md with the headline numbers for the morning
.venv/bin/python scripts/compare_models.py > results/compare_models_latest.txt 2>&1
echo "===== $(date) orchestrator done ====="
cat results/compare_models_latest.txt
