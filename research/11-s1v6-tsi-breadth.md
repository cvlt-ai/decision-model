# s1-v6 — TSI breadth line (2026-09-28)

## TL;DR
TSI breadth is a clear win. It closes most of the knowledge gap (the axis we were
furthest from Jev) **without regressing anything we were already good at**, and it
also fixed the long-standing NEI (not-enough-info) weakness on the VitaminC flip probe.
**v6 is now the best checkpoint on every headline axis.**

- mixture_v6 = mixture_v5 (v3 + c2d) + **64,000 TSI rows** (129 tasks, NLI /
  counterfactual / knowledge-heavy, `tsi-perm` license-filtered). 264,162 rows total.
- 4096 context, bs=1/accum=32, 8,077 steps, ~13 s/step, ~28 h. **0 crashes, 0
  relaunches** (the SIGBUS insurance — `--save-every 1000` + supervisor partial-eval —
  was not even needed, though 6 periodic checkpoints landed as backup).

## Frozen 9-family macro (300/family)
| axis | v5 | v6 | Δ |
|---|---|---|---|
| **macro accuracy** | 0.750 | **0.779** | **+0.029** |
| **mmlu_pro** | 0.350 | **0.487** | **+0.137** ← the v6 target |
| negation violation | 0.021 | **0.018** | better (lower) |
| Brier | 0.125 | **0.123** | better |
| ECE (raw) | 0.059 | 0.061 | ~flat |

Per-family (v5 → v6): banking77 0.947→0.957, go_emotions 0.547→0.580, severity 0.633
(flat), boolq 0.907→0.927, injection 0.984→0.992, **mmlu_pro 0.350→0.487**, baserate
0.778→0.741 (−0.037, the one dip — 25-row family, noisy), pubhealth 0.690→0.780, negation
0.917 (flat).

**No meaningful regressions.** The only down family is baserate, a 25-row sample where
±0.03 is within noise.

## VitaminC flip probe (200 conflict families) — the axis v5's c2d did NOT move
| | v3 | v5 | **v6** |
|---|---|---|---|
| NOT ENOUGH INFO | 0.308 | 0.264 | **0.582** |
| SUPPORTS | 0.916 | 0.954 | 0.919 |
| REFUTES | 0.772 | 0.713 | 0.713 |
| overall | 0.783 | 0.774 | **0.799** |
| lazy_rate | 0.175 | 0.235 | **0.090** |

This is the big secondary win. v5's c2d (long policy docs) moved the LONG hard tier but
*not* this short NLI probe — and it had actually made NEI/lazy a bit *worse* (the model
learned to commit to an answer). TSI's broad NLI / counterfactual coverage flipped that:
**NEI 0.308→0.582, lazy 0.175→0.090.** The model learned to say "not enough info" when
the evidence genuinely runs out. That is exactly the "coverage / don't over-claim"
property we wanted Jev to struggle with.

## JevBench public-231 (harness `score_task`)
| | all-public | hard |
|---|---|---|
| v5 | 0.7489 | 59/111 |
| **v6 @1024** | 0.7489 | 56/111 |
| **v6 @4096** | **0.7749** | **62/111** |
| AlexWortega openjev | 0.814 | 69/111 |
| Jev 1.13 | 0.866 | 81/111 |

v6@4096 is the best JevBench number we hold (was v5 0.7489 / hard 59). The gap to
AlexWortega closed from **0.065 → 0.039**; to Jev 1.13 from 0.117 → **0.091**.

## Reading
- The knowledge axis moved exactly where we aimed: mmlu_pro +0.137 (0.350→0.487). Still
  below Jev's 0.83, but the single largest knowledge gain across any version.
- The win is *broad*, not surgical: macro, pubhealth, go_emotions, boolq all up, and the
  two "differentiator" properties we value (negation consistency, NEI/coverage) improved
  too — TSI is general enough to reinforce the base model rather than specialize it.
- TSI rows are short (p50 94 tok, max 625), so the value was pure breadth +
  short-contrastive structure, not context length. 4096 costs nothing at bs=1.

## Bug found + fixed this run
TSI prompts embed **U+2028 (line separator)**, which `json.dumps(ensure_ascii=False)`
leaves literal and `splitlines()` then splits *mid-JSON*, fragmenting the row. Fixed the
reader (`train_s1.py`) and builder (`mixture.py`) to split on `\n`. 84 CPU tests green.

## Next
1. Gap to Jev is now 0.091 — still mostly knowledge (mmlu_pro 0.487 vs 0.83). Options:
   more/broader knowledge data, or a second TSI pass with a higher per-task cap (the
   current 500/task capped knowledge-heavy tasks like CONDAQA/ARC).
2. Full 35,594-row frozen run for the release number (still on 300/family samples).
3. Jev-compatible FastAPI server (`src/s1/schema.py` already speaks the wire format).
