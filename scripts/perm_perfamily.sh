#!/usr/bin/env bash
# Permutation-robustness, per-family, at a fast size. banking77 flagged
# truncation-confounded (1024-token prompt > budget). Run sequentially on cuda:0.
set -uo pipefail
cd /home/user/decision-model

echo "=== clean families limit 40 perm 3 ==="
.venv/bin/python -m eval.run_eval --model readout --ckpt checkpoints/s1-v1 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families go_emotions,mmlu_pro,pubhealth --limit 40 --permutations 3 \
  --raw results/_perm_clean.jsonl --fresh --out results/s1v1_perm_clean.json 2>&1 \
  | grep -E "acc=|perm_mean|perm_flip|rror" | tail -8

echo "=== banking77 (truncation-confounded) limit 12 perm 3 ==="
.venv/bin/python -m eval.run_eval --model readout --ckpt checkpoints/s1-v1 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families banking77 --limit 12 --permutations 3 \
  --raw results/_perm_b77.jsonl --fresh --out results/s1v1_perm_b77.json 2>&1 \
  | grep -E "acc=|perm_mean|perm_flip|rror" | tail -6

echo DONE
