# Project state — 2026-09-24

Goal: local, open Jev-competitor ("System One" decision model; choice/score/noul
primitives, RLCD-trained on proper scoring rules). Plan:
`~/.hermes/plans/2026-09-22_134149-jev-decision-model-research-and-local-training.md`;
research/01..05 for findings.

## ✅ v3 RESULT (done overnight 2026-09-25) — see research/06-s1v3-final.md
- **v3 = best checkpoint on all four headline axes.** macro **0.727** (Laya 0.410),
  negation violation **0.030** (Jev doc 0.19 → 6× better), post-cal **ECE 0.0265**
  (under 0.03 gate) + **Brier 0.1397** (best of all versions), MMLU-Pro 0.30 (≫ Laya
  0.137, still ≪ Jev 0.83). banking77 0.0→**0.960**. Beats Laya on all 9 families.
- v3 vs v2 is a trade, not a sweep: v3 best on macro/negation/banking77/Brier/
  injection/severity; v2 marginally better on pubhealth (0.690 vs 0.613) + baserate
  (0.926 vs 0.815, 25-row noisy family). For "beat Jev" the headline axes matter → v3.
- Calibration (paper's mandatory phase) DONE offline: `scripts/calibrate.py`.
  v3 raw ECE 0.062 (most confident model) → T=1.4 → ECE 0.0265 / Brier 0.1397.
- Order-augmentation proven on a trained family (go_emotions flip 0.113→0.05/0.0).
  mmlu/pubhealth zero-shot flip still ~0.48/0.31 (not in training mix) → future item.
- THE GAP IS KNOWLEDGE: MMLU-Pro 0.30 vs Jev 0.83 → TSI-breadth mixture is next.

## v2 result (already in — see research/05 + compare_models.py)
- macro acc **0.710** (v1 0.592, Laya 0.410); ECE 0.037; Brier 0.161.
- banking77 **0.0→0.947** (render fix now in training); baserate 0.704→0.926;
  pubhealth/injection/severity all up.
- Order-augmentation proven: go_emotions perm flip **0.113→0.000**. mmlu_pro flat
  ~0.48 (not in training mix — zero-shot; the gap to Jev's 83% is knowledge/TSI).
- ONE regression: negation-consistency 0.032→0.268 — because v2 was built WITHOUT
  boolq (missing `--include-sa`), and negation is built ON boolq. **v3 = v2 + boolq**
  is the fix; watch whether negation returns to ~0.03.

## Checkpoints
- `checkpoints/s1-v1` (75,822-row mix, 2ep, loss→0.33) — negation 0.032 (best on that axis so far)
- `checkpoints/s1-v2` (112,844-row order-aug mix, 1ep) — macro 0.710, negation regressed
- `checkpoints/s1-v3` (131,698-row order-aug + boolq, 1ep) — IN FLIGHT

## Not done (next, in order)
1. **Confirm v3**: negation recovered? banking77 ~0.95 held? macro ≥0.71?
   (compare_models_latest.txt)
2. **Calibration** already in the overnight chain (offline, no GPU) — bank ECE under 0.03.
3. **TSI-breadth mixture** → the 5.3M-row general-instruction data; this is what
   closes the mmlu_pro knowledge gap toward Jev's 83% (still our biggest deficit).
4. `scripts/decontaminate.py` MinHash fuzzy pass (exact-substring pass is done) —
   MANDATORY before any release claim.
5. Full 35,594-row frozen run for the release number (currently 300/family samples).
6. Jev-compatible FastAPI server (`src/s1/schema.py` already speaks the wire format).

## Conventions
- uv venv at `.venv`; run by `.venv/bin/python` (conda activate is broken in this shell).
- No conda env for this project. HF account `impossibleexchange`; token at
  ~/.cache/huggingface/token. llama-server holds GPU1/2 (43/47 GB) — train/eval on cuda:0.
- Datasets 4.x: script repos fail — use parquet mirrors (mteb/, Joshua-Harris/, etc.).
- License: permissive-only for release line; quarantine anli/multi_nli/snli/ai2_arc/hellaswag.
- `data/processed/` gitignored (regenerable). Eval JSONs committed as the research record.
- Launch long jobs via `terminal(timeout>600)` which auto-promotes to tracked bg + notify
  (the background=true flag flaked in this session).
