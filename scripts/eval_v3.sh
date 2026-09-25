#!/usr/bin/env bash
# s1-v3 eval: frozen 9-family suite + fresh permutation, same as v2's run.
set -uo pipefail
cd /home/user/decision-model

echo "===== [1/2] FULL FROZEN SUITE (9 families, 300/family) ====="
.venv/bin/python -m eval.run_eval \
  --model readout --ckpt checkpoints/s1-v3 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families boolq,banking77,go_emotions,mmlu_pro,pubhealth,injection,severity,baserate,negation \
  --limit 300 \
  --raw results/raw_s1v3_frozen.jsonl --fresh \
  --out results/s1v3_frozen_300.json 2>&1 | grep -E "acc=|macro|rror|Traceback" | tail -15

echo "===== [2/2] PERMUTATION (clean@40, b77@12) ====="
.venv/bin/python -m eval.run_eval \
  --model readout --ckpt checkpoints/s1-v3 \
  --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --families go_emotions,mmlu_pro,pubhealth \
  --limit 40 --permutations 3 \
  --raw results/raw_s1v3_perm.jsonl --fresh \
  --out results/s1v3_perm_clean.json 2>&1 | grep -E "acc=|perm_|rror|Traceback" | tail -8

echo "===== calibration (offline, no GPU) ====="
.venv/bin/python scripts/calibrate.py --raw results/raw_s1v3_frozen.jsonl --tag s1v3 2>&1 | tail -4

echo "===== comparison table ====="
.venv/bin/python scripts/compare_models.py 2>&1 | tail -40

echo "===== DONE v3 eval+calibrate+compare ====="
