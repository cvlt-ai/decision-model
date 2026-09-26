#!/usr/bin/env bash
# s1-v5: LoRA SFT, 1 epoch, at 4096 context, mixture_v5 = mixture_v3 + Nimble c2d
# contrastive pairs (4,464 rows, 3.3% of the mix).
#
# WHY this mixture: v4 proved the hard-tier gap is a DATA problem, not context.
# v4's retrain at 4096 on mixture_v3 came back at par with v3 (the mix has almost
# no contrast structure). The c2d line adds 1,338 base/counterfactual pairs where
# one fact flips the label - the exact EVIDENCE-SENSITIVITY signal our VitaminC
# baseline exposes as weak (NEI acc 0.308, lazy_rate 0.175 on v3).
#
# c2d rows are SHORT (max 929 tok); value is the contrast, not context length.
# 4096 is kept because it's the context-capable baseline and it won't truncate
# anything in the mix; it costs nothing extra at bs=1.
#
# Recipe = v4's (bs=1/accum=32, effective batch 32, ~4255 steps). Same SIGBUS
# guards (attempt-1 pattern):
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  -> less GPU frag OOM
#   * OMP_NUM_THREADS/MKL_NUM_THREADS=8                 -> BLAS oversubscription
set -uo pipefail
cd /home/user/decision-model
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8
exec .venv/bin/python scripts/train_s1.py \
  --model Qwen/Qwen3.5-4B \
  --epochs 1 \
  --max-len 4096 \
  --bs 1 --accum 32 \
  --mixture data/processed/mixture_v5.jsonl \
  --out checkpoints/s1-v5
