# Project state — 2026-09-22

Goal: local, open Jev-competitor ("System One" decision model; choice/score/noul
primitives, RLCD-trained on proper scoring rules). See
`~/.hermes/plans/2026-09-22_134149-jev-decision-model-research-and-local-training.md`
for the full plan; research/01..04 for findings.

## Done
- Phase 1: Jev mechanics (`research/01`), wire-spec `src/s1/schema.py` (+confidence
  reconstruction 1-H_norm^1.5), metrics, scoring rules w/ negation penalty. 61 CPU tests green.
- Phase 2: frozen suite **v2 = 9 families / 35,594 rows** (`data/holdout/` + MANIFEST
  sha256; adapters `eval/datasets/`; invariants `tests/test_holdout.py`).
  All three primitives covered (severity added for `score`).
- Baseline: Laya 0.3.5 (`research/04`, `results/laya_frozen_300.json`):
  macro acc 0.410 / ECE 0.303 / negation-violation 0.710 / mmlu_pro 0.137 ~chance /
  severity 0.277 ~chance. Gates for us: acc≥0.55, ECE≤0.03 post-recal, negation≤0.05,
  perm-shift≤0.02 (perm NOT yet measured on baseline).
- Phase 3 data: train splits 6/6 (66,010) + extras 7/7 (663,747) + TSI inventory
  (5,314,383 rows/510 tasks → `data/raw/tsi_task_counts.json`). License classes per
  `research/03` incl. TSI-ships-anli trap; quarantine dir enforced by `download_extra.py`.

## Not done (next units of work, in order)
1. `scripts/decontaminate.py` — MinHash (datasketch installed) train-vs-holdout; MANDATORY
   before mixture (esp. pubhealth_val, boolq/banking77 share train/test authors).
2. Mixture builder `src/s1/mixture.py` — TSI task sampling w/ license filter (exclude
   anli/*), family train splits, negation pairs, baserate items → `data/processed/mixture_v1/`.
3. M2 primary line (Open-Jev recipe: Qwen3-4B-class LoRA + logit readout + temperature
   scaling on heldout). M1 ModernBERT-large speed line. GPU gate `scripts/gpu_free.sh`;
   llama-server holds 47/47/34 GB — ask user before touching GPUs.
4. Baseline extras: `--permutations 5` on choice families; optional full 35,594-row run.
5. Calibration phase (temperature scaling + negation-consistent head), then
   Jev-compatible FastAPI server (`src/s1/schema.py` already speaks the wire format).

## Conventions
- uv venv at `.venv` (`uv pip install --python .venv/bin/python -e .`); no pip in venv.
- CPU-only runs: prefix `CUDA_VISIBLE_DEVICES=""`.
- Eval JSONs are committed (research record); raw per-answer logs regenerable.
- HF account `impossibleexchange` already authed; token at ~/.cache/huggingface/token.
- Datasets 4.x: script-based repos fail — prefer parquet mirrors (mteb/, fancyzhx/, clinc/,
  Joshua-Harris/). Verify ids via HfApi before writing them into code.
