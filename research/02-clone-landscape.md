# The clone landscape — and what Laya actually does when we run it

Written 2026-09-22. Landscape from sources listed at the bottom; **the measurement
section is ours** (`run_baselines.py`, raw JSON in `baseline_results.json`).

## 1. What exists, six days after launch

| Project | Approach | Size | License | Notes |
|---|---|---|---|---|
| **Laya** (`convaiinnovations/laya`) | ModernBERT-large encoder, fully fine-tuned, + 2 transformer layers, option-marker scorer, act/escalate head | 421M (322M multilingual) | Apache-2.0, weights public | Closest open artifact. Claims 33–38ms GPU. Three checkpoints (EN root / multilingual / typed-decisions). `pip install laya` works on 3.11. |
| **OpenDecision** (`deepanwadhwa/OpenDecision`) | FastAPI, Jev-compatible `/v1/systemone`, over `MoritzLaurer/ModernBERT-large-zeroshot-v2.0` | ~395M | Apache-2.0 | Zero-shot — no decision-specific training. Adds `relation` primitive + document retrieval. **Self-declares scores uncalibrated.** Needs Python ≥3.13. |
| **openjev** (`TheoLeeCJ/openjev`) | Causal Qwen3.5-4B, read option-token logits, renormalise | 4B | — | 84.5% agreement with Jev's published references vs Jev 88.3% on the same reconstructed subset. |
| **openjev-sglang** (`ekzhang/…`) | Same readout trick, SGLang serving | — | — | MMLU-Pro (1,000q): Jev **83%** vs Qwen3.6-35B-A3B **59%**, Qwen3.8-27B **60%**. BoolQ: Jev 91.6% vs 89%. |
| **SemIf** (`AlexWortega/openjev`) | Causal Qwen3.5 4B/35B + 3-class NLI head on last token | 4–35B | — | |
| **Bespoke Nimble** | LoRA Qwen3.5-9B, contrastive data curation | 9B | — | 66%→90% on curated eval vs Jev 93%; ~100ms H100. |
| **Kev-0.5B** | LoRA Qwen2.5-0.5B + readout head | 0.5B | — | Runs on a MacBook. |
| **Jevlike** | 40KB embedding option-attention | tiny | — | Each candidate queries a shared context representation. |
| **DiffusionGemmaJev** | Diffusion LM (vLLM PR #57250) | — | — | "Pretty close on benchmarks." |
| **system-one-adapter-python** (TypeSafe's own) | Wraps commercial LLMs into the Jev API shape for their evals | — | public | Asks models to output JSON+probabilities and validates; their own admission this isn't a fair speed test. |

The readout-vs-knowledge split is the whole argument: the *interface* clones in an
afternoon; on MMLU-Pro, 27–35B models doing the identical trick land at 59–60% where
Jev claims 83%. Whatever Jev is, it is not "just" a readout on a small model — that
gap has to come from training data or a larger trained base.

## 2. Our own measurement: Laya 0.3.5, CPU, this machine

12 hand-written cases (support routing, urgency, base-rate traps, negation, rubric
scores, injection detection, one factoid choice). `CUDA_VISIBLE_DEVICES=""` on
purpose — the box's GPUs are occupied by the live inference server.

**Scored: 8/9 (88.9%).** n=9, so this is a sanity signal, not a benchmark.

| case | result | observed |
|---|---|---|
| route-billing / route-technical | ok | correct option |
| urgency / not-urgency | ok | 0.738 / 0.139 |
| score-urgency (level 2 expected) | ok (tol 0.5) | 1.747 |
| score-calm (level 0 expected) | ok | 0.848 — mid-scale, weak separation |
| **injection detection** | **MISS** | P=0.403 on an explicit "IGNORE ALL PREVIOUS INSTRUCTIONS" |
| factoid choice (Au, Z=79) | ok | gold |

### Diagnostic probes (expected to fail on every known model — *not* counted)

| probe | Laya | Jev / community reference |
|---|---|---|
| die rolled 3, is it odd? | **0.155** (should be 1) | Jev reportedly 0.14–0.17 on the same question |
| unobserved fair coin, heads? | **0.158** (should be 0.50) | community saw 0.6839 on the HF Space, 0.1265 with maximal phrasing |
| P(refund) + P(not refund) | **0.127 + 0.218 = 0.345** → violation **0.655** | TypeSafe's own jaggedness page shows 0.72+0.47 = **1.19**, violation 0.19 |

Three things worth pausing on:

1. **The negation violation is 3.4× worse than Jev's documented worst case.** Laya
   answers both a question and its negation with low probability. (Both models
   appear to be treating "is the customer asking for X" as "does the ticket
   *mention* X" — the literal-reading failure TypeSafe admits, plus their admitted
   non-invariance.) Nobody guarantees `P(x)+P(¬x)=1`; **we can make it an auxiliary
   loss** (`scoring_rules.verdict_consistency_penalty`, already implemented and
   tested). This is the clearest "better, not just equal" axis we have found.
2. **The coin result differs across deployments of the same idea** (0.68 on the
   Space vs 0.158 on the local checkpoint) for a question whose honest answer is
   0.5 either way. Version drift *and* phrasing sensitivity in the same number.
   Calibration claims from single-deployment demos deserve suspicion — ours will
   ship the eval code for exactly this reason.
3. **The base-rate failures reproduce exactly**, from a different codebase and
   architecture family than Jev. This is a property of *this training recipe class*
   (synthetic decision data, small encoder), not of anyone's bug — which tells us the
   fix has to be in the data (synthetic numeric/base-rate items with verifiable
   gold) and not in the head.

### The warning Laya's own library prints at load

> `laya: this checkpoint ships temperatures outside [0.5, 5] which would distort
> confidence; clamping choice:11+=0.1006. Treat confidence from the affected buckets
> as uncalibrated.`

The leading open clone **ships a checkpoint its own SDK describes as miscalibrated
for choices with 11+ options.** If we beat Laya on ECE, we should also be able to
state *why* in one sentence: they per-bucket temperature-fit and still shipped out
of range; we hold a calibration split and refuse to publish if ECE > 0.03.

### Latency and permutation robustness (CPU)

- p50 **130.6ms** CPU (vendor's 33–38ms is GPU; both are fine numbers).
- **Permutation max shift 0.0718** across 5 option orders on route-billing — better
  than Jev's reported ~0.20, but our C5 target is ≤0.02. Position bias is present in
  every known model; the shuffled-copy augmentation (Task 3.3) is the lever.

## 3. What we deferred, and where it lands

- OpenDecision + a DIY decoder-readout baseline on the same 12 cases: **moved into
  Task 2.5**, run through `eval/run_eval.py` once the frozen suite exists, so all
  baselines share one protocol instead of ad-hoc scripts.
- Head-to-head against hosted Jev: still needs a `JEV_API_KEY` (not present); every
  comparison to Jev in this doc is to published/community numbers.

## Sources

latent.space/p/ainews-here-are-6-clones-of-jev-in · sgnt.ai/p/jev ·
firecrawl.dev/blog/what-is-jev · news.ycombinator.com/item?id=49765348 ·
docs.typesafe.ai (archived in `raw/`) · local run, 2026-09-22.
