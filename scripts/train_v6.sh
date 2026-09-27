#!/usr/bin/env bash
# s1-v6: LoRA SFT, 1 epoch, at 4096 context, mixture_v6 = mixture_v5 + TSI breadth
# (64,000 TaskSource rows: 129 tasks, NLI/counterfactual/knowledge-heavy, tsi-perm).
#
# WHY this mixture: v5 is the best checkpoint (macro 0.750, JevBench all-public
# 0.7489) but its gap to Jev 1.13 (0.866) is mostly KNOWLEDGE (mmlu_pro 0.350 vs
# Jev 0.83). TSI is the broad general-instruction data that should close that
# gap. TSI rows are SHORT (p50 94 tok, max 625) so 4096 truncates nothing; the
# value is breadth + short-contrastive structure, not context length.
#
# Recipe = v5's (bs=1/accum=32, effective batch 32). 264,162 rows -> ~8,078 steps,
# ~13 s/step -> ~29 h. Same SIGBUS guards as v4/v5:
#   * PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True  -> less GPU frag OOM
#   * OMP_NUM_THREADS/MKL_NUM_THREADS=8                 -> BLAS oversubscription
# The SIGBUS has been INTERMITTENT (death point moved between attempts), so a
# mid-run crash is possible over 29 h. --save-every 1000 overwrites the adapter
# every ~4.3 h so a crash still leaves a usable, evaluable adapter (the supervisor
# will evaluate a late crash rather than relaunch from zero).
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
  --mixture data/processed/mixture_v6.jsonl \
  --out checkpoints/s1-v6 \
  --save-every 1000
