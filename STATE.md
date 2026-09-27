# Project state — 2026-09-27

Goal: local, open Jev-competitor ("System One" decision model; choice/score/noul
primitives, RLCD-trained on proper scoring rules). Plan:
`~/.hermes/plans/2026-09-22_134149-jev-decision-model-research-and-local-training.md`;
research/01..10 for findings.

## 🔵 s1-v6 IN FLIGHT (launched 2026-09-27 ~10:46 EDT) — TSI breadth line
- v6 = mixture_v6 (mixture_v5 + **TSI breadth**: 64,000 TaskSource rows, 129 tasks,
  NLI/counterfactual/knowledge-heavy, `tsi-perm` license-filtered) at 4096,
  bs=1/accum=32. 264,162 rows → **~8,078 steps**, ~13 s/step → **~29 h**.
- **Why:** v5 is best on every axis but the gap to Jev 1.13 (0.866) is mostly
  KNOWLEDGE (mmlu_pro 0.350 vs Jev 0.83). TSI is the broad general-instruction data
  meant to close that gap. TSI rows are SHORT (p50 94 tok, max 625) → 4096 truncates
  nothing; value is breadth + short-contrastive structure.
- **Live:** train PID 2595379 (bg proc_94508c8dcaf5), supervisor PID 2596269
  (proc_57e4014a5243) watching it, up to 3 relaunches, partial-eval at 4846 steps (60%).
- **Crash insurance (new):** `--save-every 1000` overwrites the adapter every ~4.3 h,
  so the intermittent SIGBUS (death point moved between smokes) costs <4.3 h.
  Supervisor evaluates a PARTIAL (≥60%) instead of relaunching from zero.
- Eval chain (auto on completion): mmlu_pro knowledge headline (the v6 target) +
  VitaminC flip + 9-family frozen regression + JevBench@1024/4096 + calibration.
  MORNING lines land in results/overnight_v6.log.
- **BUG found + fixed this session:** TSI prompts embed U+2028 (line-separator) that
  `json.dumps(ensure_ascii=False)` leaves literal; `splitlines()` then breaks mid-JSON.
  Fixed the reader (train_s1.py) + builder (mixture.py) to split on `\n`. 84 tests green.

## ✅ s1-v5 RESULT (done overnight 2026-09-27 02:16) — see research/10-s1v5-c2d.md — CURRENT BEST
- v5 = mixture_v5 (v3 + Nimble c2d contrastive line, 4,464 rows = 3.3%) at 4096,
  bs=1/accum=32, 4,158 steps, loss→0.155, 0 relaunches.
- **v5 is the new flagship.** Frozen: macro **0.750** (v3 0.727), negation **0.021**
  (v3 0.030), Brier **0.125** (v3 0.146), mmlu_pro 0.30→0.350, injection 0.849→0.984.
  Only post-cal ECE marginally worse (0.0326 vs 0.0265).
- **JevBench hard tier 46→59** (+13 over v3, +18 over v4); all-public **0.7489**
  (v3 0.693). Gap to AlexWortega 0.814 now 0.065; to Jev 0.866 is 0.117.
- c2d moved the LONG-policy hard tier (its design target), not the short VitaminC
  flip probe (NEI 0.308→0.264) — see research/10 for why that's coherent.
- Supervisor chain ran clean + committed `741e5a0`. MORNING lines in
  results/overnight_v5.log.

## ✅ v4 RESULT (2026-09-26) — see research/08-s1v4-4096.md — was best, now superseded by v5
- s1-v4 = 4096-context retrain of mixture_v3 (bs=1/accum=32, 4021 steps, loss→0.267,
  SIGBUS guards held, no crash). **Diagnosed, not a win:**
  - JevBench: v4@1024=156 (== v3@1024, no regression); v4@4096=155 (hard 41) is BELOW
    v3@4096 (hard 46). The 4096 retrain did NOT beat the free 4096 window on v3.
  - Frozen: macro 0.727→0.723 (flat); per-family wash (injection↑0.968, mmlu/pub↓).
  - Calibration got WORSE: ECE@T1 0.062→0.110; post-cal 0.0331→0.0433 (bs1/4096 dynamics).
- CONCLUSION: hard-tier gap is DATA/knowledge, not context. Mixture has only 40 rows
  >1024 tokens. Best numbers we hold: v3 frozen (macro 0.727, negation 0.030, post-cal
  ECE 0.0331) + v3@4096-window (JevBench 0.693, hard 46 — free serving change).

## v3 result — see research/06-s1v3-final.md
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
- `checkpoints/s1-v1` (75,822-row mix, 2ep, loss→0.33)
- `checkpoints/s1-v2` (112,844-row order-aug mix, 1ep) — macro 0.710, negation regressed
- `checkpoints/s1-v3` (131,698-row order-aug + boolq, 1ep) — macro 0.727, negation 0.030
- `checkpoints/s1-v4` (same mix, 4096-context, 1ep) — context-capable, no regression, but data-bound
- `checkpoints/s1-v5` (mixture_v5 = v3 + c2d, 4096, 1ep) — **CURRENT BEST: macro 0.750, negation 0.021, JevBench hard 59**
- `checkpoints/s1-v6` (mixture_v6 = v5 + TSI breadth 64k, 4096, 1ep) — **IN FLIGHT** (TSI knowledge line)

## Not done (next, in order)
1. **s1-v6 (TSI breadth) — IN FLIGHT** — the mmlu_pro axis (0.350 on v5, ≪ Jev 0.83)
   is the clearest remaining gap; TSI supplies the general-knowledge breadth. Launched
   2026-09-27 ~10:46, ~29 h. Supervisor auto-evaluates + commits on completion.
2. If v6's mmlu_pro moves but VitaminC flip (NEI/lazy) doesn't, **upsample c2d (2–4×)**
   to push the short-contrastive axis too (v5's c2d at 3.3% moved the long hard tier
   but not the short probe).
3. `scripts/decontaminate.py` MinHash fuzzy pass (exact-substring done; c2d = 0 drops).
4. Full 35,594-row frozen run for the release number (currently 300/family samples).
5. Jev-compatible FastAPI server (`src/s1/schema.py` already speaks the wire format).

## Conventions
- uv venv at `.venv`; run by `.venv/bin/python` (conda activate is broken in this shell).
- No conda env for this project. HF account `impossibleexchange`; token at
  ~/.cache/huggingface/token. llama-server holds GPU1/2 (43/47 GB) — train/eval on cuda:0.
- Datasets 4.x: script repos fail — use parquet mirrors (mteb/, Joshua-Harris/, etc.).
- License: permissive-only for release line; quarantine anli/multi_nli/snli/ai2_arc/hellaswag.
- `data/processed/` gitignored (regenerable). Eval JSONs committed as the research record.
- Launch long jobs via `terminal(timeout>600)` which auto-promotes to tracked bg + notify
  (the background=true flag flaked in this session).
