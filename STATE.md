# Project state — 2026-09-24

Goal: local, open Jev-competitor ("System One" decision model; choice/score/noul
primitives, RLCD-trained on proper scoring rules). Plan:
`~/.hermes/plans/2026-09-22_134149-jev-decision-model-research-and-local-training.md`;
research/01..05 for findings.

## ⏰ OVERNIGHT (user asleep) — do not block on user
- **s1-v3 training running** (proc tracked, completion-notify): `checkpoints/s1-v3`,
  Qwen3.5-4B LoRA, 1 epoch, mixture_v3 (order-aug + boolq), bs=2/accum=16
  (effective batch 32; lowered from 4/8 to survive a long boolq-passage GPU OOM —
  attempt 1 OOM-killed at step 3782/4021 with no adapter saved).
- **Orchestrator running** (`scripts/overnight.sh`): waits for the "saved" marker,
  then auto-runs `scripts/eval_v3.sh` = frozen 9-family suite (300/fam) + fresh
  permutation (clean@40, b77@12) + OFFLINE temperature calibration
  (`scripts/calibrate.py`) + `scripts/compare_models.py` → commits.
  Log: `results/overnight_orchestrator.log`. Outputs land in `results/`:
  s1v3_frozen_300.json, s1v3_perm_clean.json, calibration_s1v3.json,
  compare_models_latest.txt.
- On wake: read `results/compare_models_latest.txt` + STATE.md "v3 result" note.

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
