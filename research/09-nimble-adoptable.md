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
- **Nimble contrastive** (c2d pairs) → *evidence-sensitivity / base-rate /
  long-policy* (the "fits-but-wrong" hard items, negation-adjacent).

Cheapest high-signal first move: **add a c2d contrastive pair line to our
mixture** (reuse their 1,338 pairs directly — they're committed and format-clean —
plus generate a few thousand more over our banking77/go_emotions/severity families
where we already have the policy+state), and **add VitaminC-dev to the frozen
suite** as the human-labeled contrastive eval. Then train at 4096 (v4's context).