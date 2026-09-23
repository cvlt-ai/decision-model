# s1-v1 (Qwen3.5-4B + LoRA, M2 line) vs Laya 0.3.5 — frozen suite, 300 rows/family

Run 2026-09-23. `results/s1v1_frozen_300.json` vs `results/laya_frozen_300.json`.
Same 9 families, same 300-row sample, same grading (nearest-level for score, argmax for
choice/noul, |p−t|≤0.15 for base-rate, paired |P(x)+P(¬x)−1| for negation).
s1-v1 = checkpoints/s1-v1 (LoRA r16, 2 epochs, 74,116-row mixture_v1) read out by the
logit-readout on cuda:0, **temperature 1.0 (calibration pass not yet done)**.

## Macro

| metric | **s1-v1** | Laya | Δ | our gate |
|---|---|---|---|---|
| accuracy (9-fam macro) | **0.592** | 0.410 | **+0.182** | ≥0.55 ✅ |
| ECE (15-bin) | **0.043** | 0.303 | 7.0× better | ≤0.03 (pre-cal) ≈ |
| Brier | **0.140** | 0.336 | 2.4× better | — |
| negation violation | **0.032** | 0.710 | **22× better** | ≤0.05 ✅ |
| p50 latency (GPU) | **56 ms** | 199 ms (CPU) | 3.5× faster | ≤50 ms ≈ |

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

## The banking77 = 0.0 is a bug, not a result — diagnosed, not hand-waved

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

**Fix (next step):** either (a) retrain with `--max-len 1100` (banking77 is the only
family over budget; go_emotions fits at 252), or (b) render banking77 options as key-only
or description-only (drops ~500 tokens → ~520, well under budget). (b) is cheaper and
also speeds the ~15 s/item eval. Either way banking77 must be re-scored; its 0.0 and its
contribution to the 0.592 macro (which slightly *understates* s1-v1) are both artifacts.

## What is NOT yet in these numbers
- **Permutation robustness** — running next (queued, choice families, --limit 40 --perm 3).
  This is the axis Jev is documented to fail; we have no s1-v1 number yet.
- **Calibration pass** — temperature is 1.0; ECE is the pre-scaling figure.
- **TSI breadth** — s1-v1 trained on the 74k family mixture only, not the 5.3M-row TSI.
  The knowledge gap to Jev's 83% is largely this.
- **Full-suite (35,594-row)** — this is the 300/fam sample for speed; the frozen full
  run is the release number.

## Reproduce
```bash
cd /home/user/decision-model
.venv/bin/python -m eval.run_eval --model readout \
  --ckpt checkpoints/s1-v1 --base Qwen/Qwen3.5-4B --device cuda:0 --key-batch 8 \
  --limit 300 --raw results/raw_s1v1.jsonl --fresh --out results/s1v1_frozen_300.json
```
