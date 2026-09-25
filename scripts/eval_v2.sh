#!/usr/bin/env bash
# Score s1-v2 (order-augmented) on the frozen suite + fresh permutation run,
# so we get one comparable v2 vs v1 vs Laya table. Sequential on cuda:0.
set -uo pipefail
cd /home/user/decision-model

echo "===== [1/2] FULL FROZEN SUITE (9 families, 300/family) ====="
.venv/bin/python -m eval.run_eval \
  --model readout --ckpt checkpoints/s1-v2 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families boolq,banking77,go_emotions,mmlu_pro,pubhealth,injection,severity,baserate,negation \
  --limit 300 \
  --raw results/raw_s1v2_frozen.jsonl --fresh \
  --out results/s1v2_frozen_300.json 2>&1 | grep -E "acc=|macro|rror|Traceback" | tail -15

echo "===== [2/2] PERMUTATION RUN (choice families, clean@40 b77@12) ====="
.venv/bin/python -m eval.run_eval \
  --model readout --ckpt checkpoints/s1-v2 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families go_emotions,mmlu_pro,pubhealth \
  --limit 40 --permutations 3 \
  --raw results/raw_s1v2_perm.jsonl --fresh \
  --out results/s1v2_perm_clean.json 2>&1 | grep -E "acc=|perm_|rror|Traceback" | tail -8

echo "===== banking77 perm (truncation-flagged, @12) ====="
.venv/bin/python -m eval.run_eval \
  --model readout --ckpt checkpoints/s1-v2 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families banking77 \
  --limit 12 --permutations 3 \
  --raw results/raw_s1v2_perm_b77.jsonl --fresh \
  --out results/s1v2_perm_b77.json 2>&1 | grep -E "acc=|perm_|rror|Traceback" | tail -6

echo "===== DONE v2 scoring ====="
