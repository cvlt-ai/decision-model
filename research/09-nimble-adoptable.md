# Nimble (bespokelabsai) — what to adopt

Cloned to `third_party/nimble` (2026-09-26). "Data, Model, Recipe for an open
Jev." Qwen3.5-**9B** + LoRA, single-token readout. On their 324 held-out: 90.1%
(base 66.4%, Jev 1.13 93.2%). The repo is the most directly *adoptable* open Jev
we've seen, because its headline is a **data method**, and v4 just proved data is
our binding constraint.

## The one thing worth stealing: contrastive (c2d) curation
- `data/train.jsonl` = 2,676 rows = **1,338 base/counterfactual pairs**,
  `method: c2d`. Every pair changes **one sentence, ≤8 words**, and **1338/1338
  flip the label** (verified: "Field E = 6 → 9" flips `apply_credit_keep_annual`
  → `execute_fallback_cancellation`).
- That is *exactly* the **evidence-sensitivity / base-rate / long-policy**
  reasoning our v4 retrain showed we're missing — the "fits-but-wrong" hard-tier
  items. Our mixture has long *documents* (boolq) but few **contrast pairs** that
  force the model to key on the one fact that decides.
- Method = MiniCheck-style: generate a context + policy, apply one small factual
  edit in Python, keep question/criteria/labels fixed, verify exactly the intended
  change with separate semantic calls + deletion tests (delete a sentence →
  proposition becomes unknown, never auto-false). Synthetic (GPT-5.6) and
  model-checked, **not** human-reviewed — so treat as training data, not truth.

## It drops into our pipeline almost as-is
- Their `input` is **the exact Jev wire schema we already speak**:
  `input.state[]` (speaker/text turns) + `input.questions.<field>.{type, criteria,
  instructions}`. `criteria` keys ARE the allowed answer codes. No conversion layer
  needed beyond a small adapter to our row schema.
- Training loss is **plain cross-entropy over the candidate logits**
  (`schema_train.py:60`) — the same **NLL proper scoring rule** we train on. No
  methodological gap; our RLCD framing is compatible. Recipe ≈ ours: LoRA r=16,
  LR 5e-5, max-len 2048, 1-epoch-ish.

## Their benchmark = a *second, complementary* yardstick (not JevBench)
- 13 **human-labeled** public subsets, 3,880 records, Jev-comparable, no Jev
  annotation needed. Macro **nimble 74.8% vs Jev 1.13 76.0%** (Jev +1.2).
- **Jev still wins calibration** on 11/13 subsets (ECE) and 10/13 (Brier) — even a
  9B model doesn't beat Jev on probabilities. Our post-cal ECE **0.0265** is a
  genuine edge *against* both. So the contrastive data lifts *accuracy*, not
  calibration — keep our proper-scoring-rule + offline-temperature line intact.
- **VitaminC-dev is the external, human-labeled analogue of contrastive curation**
  (near-identical evidence, different labels) — a ready-made eval that directly
  complements our `negation` family. Strong candidate to add to our frozen suite.
- Caveats to carry: single-token **26-code (A–Z) limit** (they use MASSIVE's 18
  scenarios, not banking77's 77 intents); synthetic labels; one seed per subset;
  contamination of pre-Qwen3.5 data is uncontrolled.

## Verdict vs the TSI plan
**Complement, not replacement.** They attack different parts of the hard gap:
- **TSI-breadth** (5.3M general rows) → *knowledge* (mmlu_pro 0.25→, breadth).
- **Nimble c2d pairs** → *evidence-sensitivity / base-rate / long-policy* (the
  "fits-but-wrong" hard items, negation-adjacent).

**Corrected after building (8.2):** the c2d rows are actually **short-to-medium**
(p50 441, p95 601, **max 929 tokens** — none over 1024). So their value is the
**contrastive structure** (near-identical state, one fact changed → label flips),
i.e. evidence-sensitivity — NOT long-context. That's a different, complementary
lever from TSI's breadth. All 2,676 pairs are in `mixture_v5` (4,464 rows with
order-aug) behind `--include-c2d`; 0 decontam drops (disjoint synthetic domains).

Cheapest high-signal first move (DONE in 8.2): c2d line built + tested (72 CPU
green). Remaining: **add VitaminC-dev to the frozen suite** (human-labeled
contrastive eval) and train `mixture_v5` at 4096 (v4's context).

## VitaminC axis built (9.1/9.2) — v3 baseline is in

Added as an **additive** contrastive eval family (NOT in the 9-family macro, so
prior numbers stay comparable). `tals/vitaminc` validation split == the published
63,054-row contrastive dev set (case_id groups Wikipedia-revision siblings; 99% of
families flip labels). 3-way choice SUPPORTS/REFUTES/NOT-ENOUGH-INFO, CC-BY-SA.

Random 300 under-measures the flip property (only 1 conflict-family in it), so the
honest instrument is `scripts/vitaminc_flip_probe.py`: scores case_ids with >=2
conflicting siblings.

**v3 BASELINE (200 conflict families, 690 rows):**
| metric | value |
|---|---|
| overall accuracy | 0.783 |
| SUPPORTS | 0.916 |
| REFUTES | 0.772 |
| **NOT ENOUGH INFO** | **0.308** ← the weakness |
| lazy_rate (same label for all siblings) | 0.175 |
| sensitivity_rate (varied + >=1 right) | 0.825 |

Reference (Nimble's own suite, their 599-row draw): Nimble-9B 76.6%, Jev 1.13 80.1%.
The two numbers the c2d data should move when v5 lands: **NEI accuracy (0.308) and
lazy_rate (0.175)**. Re-run the same probe on v5 to get the before/after.

Also fixed (9.2): `build_holdout` now **merges** the MANIFEST (a subset build no
longer clobbers the other families); all 9 original families verified byte-identical.