#!/usr/bin/env bash
# Launch the permutation-robustness comparison once the main baseline frees cuda:0.
# Waits for the given PID (the baseline python) to exit, then runs the perm eval.
set -euo pipefail
cd /home/user/decision-model

BASELINE_PID="${1:-477330}"
echo "[$(date +%T)] waiting for baseline pid ${BASELINE_PID} to exit..."
while kill -0 "${BASELINE_PID}" 2>/dev/null; do
  sleep 30
done
echo "[$(date +%T)] baseline done; starting permutation-robustness run (choice families, --limit 40 --permutations 3)..."

# Choice families only (perm probe is a no-op on noul). Bounded: banking77 is
# ~15s/item and --permutations multiplies it, so 40 items x 3 passes keeps it ~35min.
.venv/bin/python -m eval.run_eval \
  --model readout \
  --ckpt checkpoints/s1-v1 --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families banking77,go_emotions,mmlu_pro,pubhealth \
  --limit 40 --permutations 3 \
  --raw results/raw_s1v1_perm.jsonl --fresh \
  --out results/s1v1_perm_robustness.json 2>&1 | grep -E "acc=|perm|rror|Traceback" | tail -20

echo "[$(date +%T)] perm run finished."
