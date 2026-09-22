# Jev: mechanics, as far as they can be known

Written 2026-09-22. Primary sources are archived verbatim under `research/notes/raw/`
(fetched with curl from docs.typesafe.ai, so they survive vendor edits).

Every claim below is tagged:

- **[D]** documented by TypeSafe — quote or spec in `raw/`
- **[P]** probed — third-party black-box experiment (Archer Hume, ~10k API calls)
- **[C]** contested / community-reported — HN, r/LocalLLaMA, independent harnesses
- **[I]** our inference

---

## 1. What it is

Jev is TypeSafe AI's first "System One model": a hosted model that **does not generate
text**. You send it a `state` and a map of typed `questions`; it returns typed answers
with probability distributions. **[D]**

Launched 2026-09-15 in early access (blog dated 2026-09-14), waitlist removed
2026-09-20. Founded by Diogo Almeida, who is credited by TypeSafe's own primer as a
co-inventor of RLHF, plus Erik Gafni and Sasha Sheng. ~$40M seed led by DCVC, two
years in stealth. **[D]/[C]**

Positioning, from the launch post: *"Think of Jev as a frontier-intelligence function
call: unstructured state in, typed probabilistic decisions out."* **[D]**

## 2. The interface (this is the part we replicate exactly)

```
POST https://api.typesafe.ai/v1/systemone
Authorization: Bearer ***
```
**[D]** `raw/llms-full.txt` lines 117-123

Request: `state` (string | object | array), `model` (default `jev-latest` →
`jev-1.13.0`), `questions` (map of our-chosen key → Question). **[D]**

### Three primitives **[D]**

| | `criteria` shape | answer fields |
|---|---|---|
| `noul` | optional `{"true": …, "false": …}` | `type`, `noul` ∈ [0,1] |
| `choice` | map option→description, may be `null`, **max 255** | `type`, `choice`, `probabilities` (sum to 1), `confidence` |
| `score` | ordered array, **2–10 levels** | `type`, `score` (can land between levels), `legend`, `probabilities`, `confidence` |

### Details that are easy to get wrong, and we now don't **[D]**

- **Question keys are not model input.** "The key is not sent to the underlying model
  and is not used in inference." Our test `test_question_id_is_not_in_prompt`
  enforces that in our own implementation.
- **`instructions` can be structured**, not just a string: an object holding data
  fields plus a `question` field that references the others *by name in backticks*.
  Example from the API reference:
  `{"potential_duplicate": {"name": "John Smith", …}, "question": "Is the resume for the same person as \`potential_duplicate\`?"}`
- **Errors are `422`, not 400.** Plus `401`, `429`, `529 Overloaded`.
- **Noul answers carry no `confidence`** — "Noul answers don't carry one." Only
  Choice and Score do.
- **`score` is an expectation.** `score: 1.05` with `probabilities {0:0, 1:.95, 2:.05}`
  — i.e. Σ level·p. Vendor explicitly warns not to use it to reconstruct magnitudes
  by interpolation. **[D]**
- **Two nested context budgets:** 64k tokens for state + all questions combined;
  **32k for state + the single longest question**. Not one flat window. **[D]**
- Rate limits: 250k tokens/sec, 1,200 requests/min — "adjusting dynamically". **[D]**
- Text only. No image/audio/video. English-primary; CJK "accepted but currently lower
  accuracy". **[D]**
- **No per-customer fine-tuning.** "Jev is not fine-tuned or LoRA-adapted with
  customer data… the same weights serve every account." Domain adaptation happens
  entirely through the request text. **[D]**
- Retry semantics: back off on 429/529, honor `retry-after`. **[D]**

### Model versions

`jev-latest` → `jev-1.13.0`; `jev-preview` currently the same build. `GET /v1/models`
lists aliases. Vendor advises pinning a version ID if thresholds are tuned. **[D]**

## 3. Why it is fast and cheap

- **No decode loop.** "Jev outputs all probabilities in parallel instead of
  autoregressively generating by token." **[D]**
- Price $0.042/M input tokens ($42/Btok), **output free**. The zero is mechanical:
  no autoregressive decoding means no generated tokens to meter — though the
  `usage.output_tokens` field is still populated and priced at zero "for now". **[D]/[C]**
- Measured independently: 21 questions in 1.02s as direct-logit readout vs 5.33s as
  generated JSON **[C openjev]**; 777 judgments < 0.7s for ~¼ cent; 1,709 judgments
  across 11 experiments for <$0.01; median 0.35s vs 8.83s for the comparison model. **[C]**
- Published workflow eval: Jev 76.0% acc / $0.0001 / 0.4s; GPT-5.6 Luna 76.1% /
  $0.0025 / 14.5s; GPT-6 Sol 79.1% / $0.2152 / 34.3s. So Jev's win is
  **accuracy-per-dollar**, not top accuracy. **[D]/[C]**
- Headline "193.6× faster, 444.6× cheaper" carries TypeSafe's own caveat:
  *"we expect that these are on the higher end of real world gains."* **[D]**

## 4. RLCD, and what TypeSafe actually says about training

The primer frames three post-training branches from a pretrained LM: RLHF (human
preference), RLVR (verifiable rewards), and **RLCD — Reinforcement Learning for
Calibrated Decisions**. **[D]**

The entire published specification of RLCD is four bullets: **[D]**

- The model does not generate text.
- It returns decisions and probabilities.
- "Higher probability should correspond to a greater chance that the answer is correct."
- Calibration is a property of *groups* of predictions, not any single answer.

TypeSafe's own critique of RLHF, which motivates RLCD: preference optimization rewards
sycophancy and "confident-sounding hallucinations", and causes **mode dropping** —
narrowing probability mass onto a favoured style. That is an argument that a
proper-scoring-rule objective keeps the distribution honest. **[D]**

**Not published: the loss, the reward, the optimiser, the data recipe, the base model,
the architecture, or how calibration is measured.** The launch post calls the data
100% synthetic and Almeida has said the moat is data rather than architecture.
**[D]/[C]**

The closest published prior on the mechanism is *RLCR* (arXiv 2507.16806): reward =
λ·correctness − S(q, c) with S a **bounded strictly proper scoring rule** (Brier),
which is provably jointly incentive-compatible for accuracy and calibration. TypeSafe
does not cite it. We build our losses on Gneiting & Raftery (2007) proper scoring
rules, which the Hume probe lists as related work. **[I]**

## 5. Architecture — what the probes say

Hume's 10k-call probe concludes: *causal transformer, likely sparse MoE, repurposed
for decisions — shared-state encoding, isolated question branches, direct probability
readouts.* **[P]** His reasoning for causal-over-bidirectional is the strongest single
argument in the whole discourse: **84.6% on MMLU-Pro requires frontier-scale
pretraining, and every model at that scale is a causal decoder.** He also
concedes a bidirectional Jev "can't be ruled out from the outside", and that the
sparse-MoE part is "the least certain". **[P]**

Observable signatures **[P]**:

- **Output-token accounting is additive**: for yes/no, exactly 4 shared + 15 per
  answer + the token length of each question id. Since ids are documented as never
  reaching the model, this count is computed *after* inference, from the serialised
  response. So `usage.output_tokens` is a response-size artefact, not generation.
- **Server time is flat to ~100 questions**, then rises — consistent with encoding
  the state once and batching the question work.
- **Question text costs ~2× state**, token for token.
- Probing **cannot** distinguish a causal decoder from a bidirectional encoder,
  because in both the decision can read the whole input.

The mechanistic core — score a user-defined finite answer space in one pass and
softmax over *the options* rather than a ~150k vocabulary — is independently confirmed
by four open reimplementations. **[C]**

## 6. Documented weaknesses — our targets

TypeSafe publishes a jaggedness page for `jev-1.13` (reviewed 2026-09-17) admitting
nine failure modes. **[D]** This is unusually honest and it hands us a checklist:

1. **Literal reading** — answers the words written, not the intent.
2. **Math and numbers** — "not a calculator"; does not count reliably; hex/RGB/
   assembly underperform; **score levels are weak in numerical calibration**.
3. **Date/time comparison** — reads dates as text, not ordered quantities.
4. **Indirection** — double negatives and property-of-property cost accuracy.
5. **Large state full of irrelevant detail** — accuracy falls as distractors grow.
6. **Adversarial content** — "State is data, and Jev does not treat it as hostile by
   default"; injected or self-arguing text can move the answer.
7. **Contradictory instructions and criteria.**
8. **Common-sense structural invariants don't hold** — with a worked example:
   `noul = 0.22` for "is the customer asking for a refund" while the equivalent
   yes/no choice says `yes = 0.01`; and `P(refund) = 0.72` with
   `P(not refund) = 0.47`, summing to **1.19**.
9. **Generation** — obviously not; use a generative model.

Independently reported **[C]**:

- Base-rate failures: "a 6-sided die rolled a 3, is the number odd?" → 0.17;
  coin-flip heads → 0.68; with the state spelled out at maximum explicitness,
  heads → 0.1265 and choice picks tails at 0.7478.
- **Option reordering shifts probabilities by up to ~20 points.**
- Behaves like Qwen when used as an LLM (leaks base-model identity).
- Struggles with tic-tac-toe.
- No standard benchmark exists for this model class — the single most-cited
  structural criticism, and why Phase 2 mints one before any training.

Counterweight: practitioners report real production wins (10× cheaper, 2× faster
than what they had on smaller LLMs **[C]**), and TypeSafe's evals are at least
published with methodology, per-model numbers and cached sample-level results.

## 7. The clone landscape (48 hours after launch)

| Project | Approach | Notable |
|---|---|---|
| `convaiinnovations/laya` | ModernBERT-large (395M) fully fine-tuned + 2 transformer layers + option-marker scorer + act/escalate head = **421M**, ~33–38ms, Apache-2.0, weights public | Closest open artifact. **Context 1024 (8k multilingual) vs Jev 64k.** HN consensus: not frontier. |
| `deepanwadhwa/OpenDecision` | FastAPI, Jev-compatible `/v1/systemone` over `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | Adds `relation` primitive + doc retrieval. Self-declares **scores uncalibrated**. Needs Python ≥3.13. |
| `TheoLeeCJ/openjev` | Qwen3.5-4B, read option-token logits | **84.5%** agreement with Jev's published references vs **Jev 88.3%** on the same reconstructed subset; 5× speedup vs asking the same model for JSON |
| `ekzhang/openjev-sglang` | single-token readout, SGLang | MMLU-Pro (1,000q): **Jev 83%** vs Qwen3.6-35B-A3B 59%, Qwen3.8-27B 60%. BoolQ: Jev 91.6% vs 89% |
| SemIf / `AlexWortega/openjev` | 4B & 35B causal Qwen3.5 + 3-class NLI head on last token | |
| Bespoke Nimble | LoRA Qwen3.5-9B, contrastive curation | 66%→90% vs Jev 93%; ~100ms on H100 |
| Kev-0.5B | LoRA Qwen2.5-0.5B + readout head | runs on a MacBook |
| Jevlike | 40KB embedding option-attention | each candidate queries a shared context |
| DiffusionGemmaJev | diffusion LM, vLLM PR #57250 | "pretty close on benchmarks" |

The MMLU-Pro column is the decisive one and it is where the argument actually lives:
the readout *interface* is trivially cloned, but **the knowledge behind Jev's readout
is not** — 27B and 35B models doing the same trick land at 59–60% where Jev lands at
83%. Whatever Jev is, it is not merely a prompt trick on a small model. **[C]**

Two structural notes for us: three separate projects chose ModernBERT-style encoders
(the "best guess" in the HN thread), and one HN commenter asks the right design
question — whether one output line attends to other lines, since the design is
parallel and non-autoregressive. Our encoder design (§collator) makes that choice
explicit and testable.

## 8. Where NeoHorse-1-4B fits (it is not a competitor)

`TokenRhythm/NeoHorse-1-4B` is a 4B causal LM post-trained from Qwen3.5-4B for agentic
tool use and coding (Apache-2.0, 262k context, +5.93 macro over base across 10
benchmarks, arXiv 2609.08183). It generates text; Jev cannot. Different classes.

Three things worth taking from it anyway:

1. **Its routing harness is the shape of our evaluation loop.** Tasks go to a
   heterogeneous model pool, tool interactions and outcomes are recorded, capability
   demand is estimated, and capability-level feedback shapes the *next training
   mixture*. That is exactly the M1/M2 router in Phase 6 — we will route between our
   encoder and decoder variants, log which one wins per task family, and feed that
   back into data mixtures.
2. **Its data-quality pipeline is a checklist we adopt wholesale**: exact and
   near-duplicate removal, evaluation decontamination, structural validation,
   six-dimensional semantic evaluation, subscene-level Scene/Goal/Outcome labelling.
   Phase 3 implements the first three directly.
3. **The System-1/System-2 division is the actual product architecture.** A decision
   model is the cheap checker; a NeoHorse-shaped agent is the maker. Our model's
   intended customer is an agent harness, which is also what TypeSafe's own patterns
   doc says ("keep code in control, give System One narrow structured decisions").

## 9. What is NOT knowable from outside

Stated plainly so no later step mistakes an inference for a fact:

- Architecture (encoder vs decoder vs MoE) — only bounded by probe signatures.
- Base model and tokenizer — "the tokenizer doesn't reveal one" **[P]**.
- The RLCD loss, reward, and RL algorithm.
- The synthetic data recipe — described only as "100% synthetic".
- How `confidence` is computed — withheld with an explicit promise of a future
  cookbook. We reconstructed a compatible statistic (§below).
- How calibration is measured or validated.

### Our reconstruction of `confidence`

`raw/llms-full.txt` withholds the formula, but the vendor docs contain four
(distribution → reported confidence) pairs. Fitting candidates against them:

| candidate | max abs error vs published |
|---|---|
| **`1 − H_norm^1.5`** | **0.008** |
| `1 − H_norm` (nats/logK) | 0.071 |
| `1 − H` bits-normalised | 0.152 |
| `Σp²` (Gini) | 0.026 |
| `p₁ − p₂` margin | 0.060 |

We ship `1 − H_norm^1.5` (monotone, 0 on uniform, 1 on one-hot) as a
**compatibility approximation**, wired into `confidence_from_probs()` and pinned by
`test_confidence_matches_vendor_examples`. It is not their function and we will not
claim it is — the vendor says the choice of statistic is a specialised topic and hands
us the full `probabilities` precisely so we can pick our own.

## 10. Consequences for our build (what this document changes)

1. **Replicate the interface exactly**, including 422-not-400, additive output-token
   accounting, structured `instructions` with backtick references, noul-carrying-no-
   confidence, and the 255 / 2-10 / 64k-32k limits. Done in `src/s1/schema.py`.
2. **Do not chase Jev's MMLU-Pro number with a small encoder.** The clone evidence
   says a 0.4B encoder cannot get there. Hence two model lines: M1 encoder for speed,
   M2 causal decoder for knowledge.
3. **Beat them on the eight failure modes they admit**, not on their marketing axes.
   Concretely: permutation robustness (their ~20pt shift is our C5 target),
   negation consistency (P(x) + P(¬x) = 1 is a loss term we can add — they don't
   guarantee it), distractor resistance, and adversarial robustness.
4. **Calibration is the product.** Build the eval and the calibration machinery
   before training anything.
5. **Long state is their stated strength and the encoders' weakness.** Keep M2 at
   long context; measure accuracy-vs-length honestly rather than hiding it.
