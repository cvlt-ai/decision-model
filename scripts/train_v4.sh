#!/usr/bin/env bash
# s1-v4: LoRA SFT, 1 epoch, at 4096 context (was 1024), same mixture_v3.
#
# WHY 4096: JevBench's hard tier is 2-6k-token policy documents; s1-v3 (trained
# at 1024) truncated 41/111 hard items and a 4096 *re-score* recovered only 4 of
# them — because the model was never trained on long documents, not just cut off.
# Training at 4096 makes the model CAPABLE of long context (and fixes the 40
# long training rows), but the MIXTURE has few long rows (40 >1024), so the real
# hard-tier gain still needs long-policy DATA (TSI line, next iteration). v4 is
# the context-capable baseline that iteration builds on.
#
# bs=1/accum=32 (NOT the default 4/8): same effective batch 32, same ~4021
# steps, but the per-sample peak is much lower — essential at 4096 context where
# the long rows (severity max 18k, banking77 ~792) would OOM at bs=2.
set -uo pipefail
cd /home/user/decision-model
exec .venv/bin/python scripts/train_s1.py \
  --model Qwen/Qwen3.5-4B \
  --epochs 1 \
  --max-len 4096 \
  --bs 1 --accum 32 \
  --mixture data/processed/mixture_v3.jsonl \
  --out checkpoints/s1-v4
