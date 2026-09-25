#!/usr/bin/env bash
# s1-v2: LoRA SFT, 1 epoch, on the ORDER-AUGMENTED mixture v2.
# Fixes both open s1-v1 items in one pass:
#   1. order invariance (shuffled-option twins) -> attacks the 0.475 mmlu perm flip
#   2. banking77 truncation (shared render, 5.4) -> states now reach the model in training
set -uo pipefail
cd /home/user/decision-model
exec .venv/bin/python scripts/train_s1.py \
  --model Qwen/Qwen3.5-4B \
  --epochs 1 \
  --mixture data/processed/mixture_v2.jsonl \
  --out checkpoints/s1-v2
