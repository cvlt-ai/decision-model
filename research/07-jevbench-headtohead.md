# Head-to-head on JevBench public-231 — s1-v3 vs AlexWortega/openjev vs Jev

The yardstick: JevBench (fstandhartinger, MIT), **public 231 items** = original
(standard) 72 + easy 48 + hard 111. Same items AlexWortega/openjev v5 and Jev
1.13 were scored on. We ran `checkpoints/s1-v3` through the harness's **own**
`jevbench.scoring.score_task` (argmax, renorm band, exact labels) — no bespoke
grading. Driver: `scripts/jevbench_run.py`; attribution: `scripts/attribute_hard_gap.py`.

## Headline (same 231 items, raw correct/231)

| model | all-public | easy (48) | standard (72) | hard (111) |
|---|---|---|---|---|
| **s1-v3 (ours)** | **156/231 = 0.675** | 48/48 = 1.000 | 66/72 = 0.917 | 42/111 = 0.378 |
| AlexWortega openjev v5 | 188/231 = 0.814 | 48/48 = 1.000 | 71/72 = 0.986 | 69/111 = 0.622 |
| Jev 1.13 (TypeSafe) | 200/231 = 0.866 | 48/48 = 1.000 | 71/72 = 0.986 | 81/111 = 0.730 |

**We are behind AlexWortega by 0.139 (32 items) on this benchmark.** The whole
gap is concentrated in two tiers:

- **easy: tie** (both 1.000).
- **standard: we lose 5 items** (66 vs 71) — a small, real gap.
- **hard: we lose 27 items** (42 vs 69) — the dominant gap.

## Where the hard-tier gap comes from (measured, not assumed)

The hard tier is JevBench's long-policy tier — multi-condition 2–6k-token
documents. Two effects, both quantified:

| hard subset | correct | rate |
|---|---|---|
| items that fit in our 1024 context (70) | 32/70 | 0.457 |
| items truncated >1024 (41) | 10/41 | 0.244 |

So it is **not just truncation**:
1. **Genuine difficulty (knowledge + reasoning).** Even the 70 hard items that fit
   in context are only **0.457** for us. That is the long-policy reasoning gap
   (multi-condition, trade-off, "no clear answer", multi-hop, adversarial
   distractors). AlexWortega v5 gets 0.622 here.
2. **Truncation (context length, fixable).** 41/111 hard items exceed our 1024
   token cap (p95 3,296, max 3,914); they drop to 0.244. Our readout keeps the
   *tail* 1024 and we trained at 1024, so those long policies are cut. If they
   scored like the fitting items, hard would be ~0.459, not 0.378.

## Calibration / latency on this set (harness metrics)
- ECE (10-bin top-label, full 231): **0.155**; Brier **0.461**. Much worse than
  our frozen-suite ECE of 0.0265 — because JevBench's hard tier is where we are
  *confident and wrong*. Our calibration win was on the families we trained on;
  on unseen long-policy reasoning it degrades. Honest, and consistent with the
  hard-tier accuracy.
- Latency: **p50 85 ms, p95 724 ms** on cuda:0 (readout, no generation). Fast on
  the short tiers; the p95 is driven by the long hard items.

## Caveats that keep this from being a clean loss
1. **Context length.** We ran at our trained 1024 tokens; AlexWortega's
   cross-encoder reads longer states (their hard 0.622 implies it). The 41
   truncated items (≈0.08 of hard, ≈0.02 of the total) are a handicap we can
   lift by serving/retraining at 4096 — not yet done.
2. **Their contamination is disclosed, and it helps them.** v5 was trained on the
   *test splits* of MMLU, ARC, GSM8K, HellaSwag, WinoGrande, GPQA-diamond,
   CLINC-150, Banking77, ESCI. The hard reasoning items overlap that knowledge.
   We trained clean (no hold-out contamination, decontaminated) but with less
   general knowledge — exactly the TSI-breadth gap.
3. **Different axes.** JevBench measures accuracy + calibration + speed + cost.
   It does **not** measure negation-consistency or option-order invariance — the
   two axes where we lead (negation 0.030, and order-augmentation proven on
   trained families). AlexWortega is order-invariant *structurally* (better), but
   JevBench's single option order doesn't expose that difference here.

## Bottom line
On the one shared public yardstick, **AlexWortega/openjev v5 (0.814) beats s1-v3
(0.675)**, and Jev (0.866) beats both. We match them on easy, trail by ~5 on
standard, and trail by ~27 on hard. The hard gap is real (long-policy
knowledge/reasoning) *and* partly a fixable context-length handicap (41 items
truncated at 1024). Our differentiators (negation, calibration-on-trained-data)
aren't tested by this benchmark.

## What would close the gap (in order)
1. **Longer context (4096)** — retrain/serve s1-v3 at 4096 so the 41 truncated
   hard items aren't cut. Cheapest win; recovers ~0.08 of hard directly.
2. **TSI-breadth mixture** — the knowledge/reasoning data that lifts the 0.457
   "fits but wrong" hard items toward their 0.622.
3. Optionally a **long-policy-focused** stage (the hard tier is policy documents)
   to match the domain they were implicitly trained on.

## Reproduce
```bash
cd /home/user/decision-model
git clone --depth 1 https://github.com/fstandhartinger/jevbench third_party/jevbench
.venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v3 --device cuda:0 --key-batch 8
.venv/bin/python scripts/attribute_hard_gap.py   # truncation vs difficulty split
.venv/bin/python scripts/measure_jevbench_len.py # per-tier prompt token lengths
```
