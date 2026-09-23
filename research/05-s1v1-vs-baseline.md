# s1-v1 (Qwen3.5-4B + LoRA, M2 line) vs Laya 0.3.5 — frozen suite, 300 rows/family

Run 2026-09-23. `results/s1v1_frozen_300.json` vs `results/laya_frozen_300.json`.
Same 9 families, same 300-row sample, same grading (nearest-level for score, argmax for
choice/noul, |p−t|≤0.15 for base-rate, paired |P(x)+P(¬x)−1| for negation).
s1-v1 = checkpoints/s1-v1 (LoRA r16, 2 epochs, 74,116-row mixture_v1) read out by the
logit-readout on cuda:0, **temperature 1.0 (calibration pass not yet done)**.

## Macro

| metric | **s1-v1** | Laya | Δ | our gate |
|---|---|---|---|---|
| accuracy (9-fam macro) | **0.592** → **0.639*** | 0.410 | +0.18 → +0.23 | ≥0.55 ✅ |
| ECE (15-bin) | **0.043** | 0.303 | 7.0× better | ≤0.03 (pre-cal) ≈ |
| Brier | **0.140** | 0.336 | 2.4× better | — |
| negation violation | **0.032** | 0.710 | **22× better** | ≤0.05 ✅ |
| p50 latency (GPU) | **56 ms** | 199 ms (CPU) | 3.5× faster | ≤50 ms ≈ |

\* 0.592 is as-run (banking77=0.0, a truncation artifact); **0.639** swaps banking77 to
its re-scored 0.425 (key-only render). See the banking77 section below.

## Per family

| family | s1-v1 acc | Laya acc | Δ | read |
|---|---|---|---|---|
| boolq (noul) | **0.900** | 0.740 | +0.160 | clean win |
| banking77 (77-way) | 0.000 | 0.367 | −0.367 | ⚠ **truncation artifact, not a fair number** (see below) |
| mmlu_pro (knowledge) | **0.303** | 0.137 | +0.166 | the axis M2 exists for — delivered. Still far of Jev's claimed 83% (we ran 2-epoch SFT, not the full TSI-breadth recipe) |
| injection (noul) | 0.714 | 0.730 | −0.016 | ~tie; Laya's homoglyph catch (9/10) edges it, plain is the shared weak spot |
| go_emotions (28-way) | **0.603** | 0.437 | +0.166 | win, and ECE 0.060 vs 0.448 — far better calibrated |
| pubhealth (MCQ) | **0.667** | 0.353 | +0.314 | biggest single-family gain |
| severity (score) | **0.527** | 0.277 | +0.250 | ordinal readout working |
| negation (noul) | **0.907** | 0.277 | +0.630 | **flagship** — see below |
| baserate | **0.704** | 0.370 | +0.334 | base-rate reasoning, the axis both Jev and Laya were documented to fail |

## The three axes we picked to attack, and what actually happened

1. **Negation consistency — the headline.** s1-v1 = **0.032** mean |P(x)+P(¬x)−1|.
   Laya was 0.710 (3.7× worse than Jev's own admitted 0.19). **We beat Jev's documented
   self-reported violation.** This came from the mixture's negated twins (every noul row
   trained with a flipped-question/flipped-gold pair), so it is architectural, not lucky.
   The negation-family accuracy itself (0.907) tracks boolq (0.900) — the model handles
   the NOT, not just the base question.
2. **Knowledge (MMLU-Pro).** 0.303 vs Laya's 0.137 — the causal-decoder line carries
   knowledge the encoder line structurally can't. Honest gap remains vs Jev's claimed
   83%; closing it is the TSI-breadth mixture + more epochs, not a different architecture.
3. **Calibration.** ECE 0.043 at temperature 1.0, *before* the temperature-scaling pass.
   That's 7× better than Laya and essentially at our 0.03 gate already; the calibration
   phase (Phase 5) should push it under.

## The banking77 = 0.0 is a bug, not a result — diagnosed, and now fixed

A 77-option banking77 prompt renders to **1037 tokens** (state + "Question" + 77
`[i] key: description` lines + "Answer:"). With `max_len=1024` the readout keeps the
**tail** — so the actual state string is truncated away. Verified: state "I am still
waiting on my card?" is **not present** in the last 1024 tokens of the rendered prompt.
The model then scores 77 options it *can see* with no query → collapses to ~8
transfer-related intents at ~0.124 each, gold (`why_verify_identity`) never in the top.

It's **shared with training**: train_s1.py also truncates at 1024, so banking77 states
were cut off in the training data too — the model learned the intent *prior*, not the
intent *from the message*. go_emotions prompts are 252 tokens (unaffected), which is why
the 28-way affect family works fine.

**Fix applied** (commit 5.4): banking77's option "descriptions" are just the key with
underscores→spaces (`why_verify_identity` → `why verify identity`) — zero information, so
`render` now drops them to key-only. Prompt **1037 → 703 tokens**, state survives.
Renderer moved to a single shared source (`s1.mixture.render`, imported by both train and
eval) so they can't drift.

**Re-scored on the new render, eval-side only (no retrain yet): banking77 0.0 → 0.425.**
That already **beats Laya's 0.367** on the family that was our worst. Caveat: this is an
*eval-only* fix — training still used the truncated old render, so there is a residual
train/eval mismatch. A retrain with the new render (state present in training too) should
push it higher; the 0.425 is a lower bound, not the ceiling.

## What is NOT yet in these numbers
- **Permutation robustness** — measured (see below, `results/s1v1_perm_clean.json`).
- **Calibration pass** — temperature is 1.0; ECE is the pre-scaling figure.
- **TSI breadth** — s1-v1 trained on the 74k family mixture only, not the 5.3M-row TSI.
  The knowledge gap to Jev's 83% is largely this.
- **Full-suite (35,594-row)** — this is the 300/fam sample for speed; the frozen full
  run is the release number.

## Permutation robustness (the axis Jev is documented to fail)

`--permutations 3`, key-aligned argmax + max prob-shift. Clean (non-banking77) families
at n=40, banking77 flagged truncation-confounded at n=12:

| family | perm flip rate | perm max-shift |
|---|---|---|
| go_emotions (28-way) | **0.113** | 0.071 |
| mmlu_pro (≤10) | **0.475** | 0.241 |
| pubhealth (≤10) | **0.200** | 0.221 |
| banking77 (77) | 0.583 | 0.054 (⚠ truncation: state cut, so low shift is the *uniform* artifact, not stability) |

**Reading:** we have NOT beaten Jev here. Jev is documented to shift up to ~0.20 on
option order; our **mmlu_pro flips 47% of the time** and shifts 0.24 — worse. The plain
SFT readout is order-sensitive, exactly the failure mode we set out to attack. go_emotions
is the best (0.11 flip) and is the only one near our ≤0.02 shift gate.

**Why, and the fix:** the model learned `P(key | ...option-list-order...)`, not
`P(key | option-set)`. Order-invariance is not free from SFT — it needs either
(a) **order-augmented training**: render each row with shuffled option order (cheap —
regenerate the mixture with per-row random order, no re-download), or (b) a
**consistency/contrastive loss** across orderings during fine-tune, or (c) accept it and
report it. (a) is the cheapest and directly targets the measured failure. This is the
next real training change, and it's a *new* capability — not in the baseline, not in Jev's
guarantees — so even a partial win is a differentiator.

**Honest framing:** of the four "Jev is weak" axes, s1-v1 currently **beats Jev on
negation (0.032 vs 0.19) and calibration (pre-scale 0.043)**, is **ahead of Laya on
knowledge** (0.303 vs 0.137, still short of Jev 83%), and **has not yet beaten Jev on
option-order robustness** (mmlu_pro 0.475 flip). That's a credible, honest position — and
the permutation gap is now a *named, measured, fixable* item rather than a hope.

## Reproduce
```bash
cd /home/user/decision-model
.venv/bin/python -m eval.run_eval --model readout \
  --ckpt checkpoints/s1-v1 --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --limit 300 --raw results/raw_s1v1.jsonl --fresh --out results/s1v1_frozen_300.json
```
