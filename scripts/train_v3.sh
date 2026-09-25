#!/usr/bin/env bash
# s1-v3: LoRA SFT, 1 epoch, on the CORRECT full mixture (order-aug + boolq).
# v2's only regression was negation (0.032 -> 0.268) caused by a missing
# --include-sa flag that dropped boolq (cc-by-sa) from training; negation is
# built ON boolq. v3 = v2 + boolq. Should keep v2's gains (macro 0.710,
# banking77 0.947, go_emotions perm flip 0.0) and restore negation.
#
# bs=2/accum=16 (NOT the default 4/8): same effective batch 32 and same
# optimizer dynamics/steps, but ~half the peak GPU memory. Attempt 1 OOM-killed
# at step 3782/4021 on a long boolq-passage batch; this gives headroom.
set -uo pipefail
cd /home/user/decision-model
exec .venv/bin/python scripts/train_s1.py \
  --model Qwen/Qwen3.5-4B \
  --epochs 1 \
  --bs 2 --accum 16 \
  --mixture data/processed/mixture_v3.jsonl \
  --out checkpoints/s1-v3
