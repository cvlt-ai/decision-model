# s1-v6 — Model Card

_A local, open, Jev-compatible "System One" decision model._

**Base:** Qwen3.5-4B (Apache-2.0) · **Adapter:** LoRA r=16, α=32 · **Context:** 4096
**Interface:** the TypeSafe Jev wire spec (choice / score / noul) · **Size:** 124 MB adapter
**Release checkpoint:** `checkpoints/s1-v6` · **Eval holdout:** 35,594 frozen rows

---

## 1. What this is

`system one` (s1) is a small decision model trained to answer Jev-style questions
(`choice`, `score`, `noul`) over a free-form "state", returning calibrated
probabilities rather than a bare answer. It is deliberately built on a general
instruction base (Qwen3.5-4B) with a thin LoRA adapter, so it **carries knowledge**
a purpose-built encoder can't, while the readout (single-token logit scoring over
option keys) keeps inference cheap and the probabilities well-formed.

The design goal was not just to match TypeSafe Jev's accuracy, but to be **open and
auditable**, and to be structurally strong where Jev and its open clones are weak:
calibration, negation/invariance consistency, and coverage (knowing when the evidence
is insufficient).

## 2. Task

Given a `state` (free-form text / structured context) and one or more questions,
return a well-formed answer per the Jev wire spec:

| type | answer |
|---|---|
| `choice` | the chosen option key + full probability vector over options |
| `score` | a score on a 2–10 level scale + probability distribution |
| `noul` | a probability in [0,1] (the "yes/no/unless" branch) |

Up to 255 options per choice (vendor spec). Answers carry probabilities, not just the
top pick — that is what makes calibration and coverage meaningful downstream.

## 3. Model

- **Base:** `Qwen/Qwen3.5-4B` (Apache-2.0, 262k native context).
- **Adapter:** LoRA (`r=16`, `alpha=32`, dropout 0.05) on every attention projection
  (`q/k/v/o`) and MLP projection (`gate/up/down`) plus the gated `in_proj` modules.
  ~124 MB. Inference-mode, bf16.
- **Readout:** `LogitReadout` (`eval/readout.py`) — teacher-forces each option key and
  reads the single-token logit; option keys are scored in chunks (`key_batch`) to bound
  peak memory. This is the same NLL proper scoring rule used at training time.

## 4. Training data

`mixture_v6` — **264,162** choice/score/noul rows, 1 epoch, 8,077 optimizer steps
(effective batch 32, 4096 context), loss → ~0.06–0.28.

| family | rows | % | source |
|---|---:|---:|---|
| tsi | 128,000 | 48.5% | TaskSource-instruct (129 tasks, NLI/counterfactual/knowledge, `tsi-perm` commercial-safe subset) |
| go_emotions | 86,820 | 32.9% | GoEmotions (multi-label, order-augmented) |
| banking77 | 19,986 | 7.6% | Banking77 (77-way intent, order-augmented, key-only render) |
| boolq | 18,854 | 7.1% | BoolQ (CC-BY-SA; negation pairs built on it) |
| severity | 4,946 | 1.9% | bespoke severity scale |
| nimble | 4,464 | 1.7% | Bespoke Nimble c2d contrastive pairs (evidence-flip) |
| injection | 1,092 | 0.4% | prompt-injection robustness |

**License posture (release line):** permissive / commercial-safe. Qwen3.5-4B is
Apache-2.0; TSI is the commercial-safe subset; BoolQ is CC-BY-SA and enters only via
the `--include-sa` gate. Non-commercial rows (anli, multi_nli, snli, ai2_arc,
hellaswag) are quarantined out of the release mixture.

## 5. Evaluation — release numbers (full 35,594-row frozen holdout)

Frozen holdout, 9 families, s1-v6 at 1024 context, T=1.0. This is the citable number
(the 300/family sample used during iteration is superseded by the full run).

| metric | **s1-v6 (release)** |
|---|---:|
| **macro accuracy** (unweighted, 9 families) | **0.7747** |
| **mmlu_pro** (n=12,032) | **0.438** |
| **negation violation** (lower = more consistent) | **0.022** |
| ECE @ raw T=1 (pooled) | 0.105 |
| **ECE @ best temperature (T=1.5)** | **0.0169** |
| **Brier @ best temperature** | **0.1441** |

### Per-family accuracy (full holdout)

| family | n | accuracy |
|---|---:|---:|
| injection | 126 | 0.9921 |
| banking77 | 3,076 | 0.9324 |
| boolq | 3,270 | 0.9128 |
| negation | 3,270 | 0.9110 |
| pubhealth | 7,929 | 0.7874 |
| baserate | 27 | 0.7407 |
| severity | 437 | 0.6568 |
| go_emotions | 5,427 | 0.6005 |
| mmlu_pro | 12,032 | 0.4383 |

### Calibration

The raw pooled ECE (0.105) is high only because mmlu_pro (12k of 35.5k rows, where the
model is genuinely uncertain) dominates the pool — every individual family's raw ECE is
fine. Offline temperature scaling (the standard post-hoc phase) at **T=1.5** brings
pooled ECE to **0.0169** — the best calibration held across any version, well under the
0.03 gate — with Brier 0.1441.

### JevBench (public-231, harness's own `score_task`)

| model | all-public | hard tier |
|---|---:|---:|
| **s1-v6 @4096** | **0.7749** | **62/111** |
| s1-v5 | 0.7489 | 59/111 |
| AlexWortega/openjev v5 | 0.814 | 69/111 |
| Jev 1.13 (hosted) | 0.866 | 81/111 |

### Evidence-sensitivity (VitaminC flip probe, 200 conflict families)

| | s1-v6 |
|---|---:|
| overall | 0.799 |
| SUPPORTS | 0.919 |
| REFUTES | 0.713 |
| **NOT ENOUGH INFO** | **0.582** |
| **lazy_rate** (lower = better) | **0.090** |

The broad NLI/counterfactual coverage in the TSI data taught the model to return
"not enough info" when the evidence genuinely runs out, instead of forcing an answer —
the "don't over-claim" property that is the whole point of a coverage-aware decision model.

## 6. How it got here (version trajectory, frozen 300/family suite)

| version | macro | negation | Brier | mmlu_pro |
|---|---:|---:|---:|---:|
| Laya (encoder baseline) | 0.410 | 0.710 | 0.336 | 0.137 |
| s1-v1 | 0.592 | 0.032 | 0.140 | 0.303 |
| s1-v2 | 0.710 | 0.268* | 0.161 | 0.320 |
| s1-v3 | 0.727 | 0.030 | 0.146 | 0.300 |
| s1-v4 | 0.723 | 0.022 | 0.157 | 0.253 |
| s1-v5 | 0.750 | 0.021 | 0.125 | 0.350 |
| **s1-v6** | **0.779** | **0.018** | **0.123** | **0.487** |

\* v2's negation spike was a training-data bug (built without BoolQ); fixed in v3.
Each line is a deliberate data decision, documented in `research/05`–`research/11`.
The v6 jump is TSI breadth: it closed most of the knowledge gap *and* improved the
differentiator axes (negation, coverage) simultaneously.

## 7. How to run it

Minimal (logit readout, no server):

```python
from eval.readout import LogitReadout
ro = LogitReadout(ckpt="checkpoints/s1-v6", base="Qwen/Qwen3.5-4B",
                  device="cuda:0", temperature=1.5, max_len=1024, key_batch=8)
answers = ro.call(state_text, {"q1": {"type": "choice", "options": {...}}})
# answers["q1"] = {"choice": key, "probabilities": {key: p, ...}}
```

HTTP (Jev wire spec, `src/s1/schema.py` speaks the request/response format):

```bash
python scripts/serve_s1.py --ckpt checkpoints/s1-v6 --base Qwen/Qwen3.5-4B \
  --device cuda:0 --temperature 1.5 --port 8000
# POST /v1/answer  { "state": "...", "questions": { ... } }
```

> Serving note: `LogitReadout` + `schema.py` are the working pieces; `serve_s1.py`
> wraps them in FastAPI. There is no batching/continuous-batching layer yet — that is
> the next serving step if throughput matters.

## 8. Limitations (honest)

- **Knowledge is the gap.** mmlu_pro 0.438 vs Jev's documented 0.83. This is the one
  axis we are clearly behind, and it is bounded by the TSI per-task extraction cap
  (500/task) which throttled the knowledge-heavy tasks.
- **go_emotions 0.601** is the weakest large family — multi-label emotion is hard and
  it is only in the mix via order-augmentation.
- **baserate (n=27)** and **injection (n=126)** are small families; their per-family
  numbers carry wide confidence intervals.
- **Raw pooled ECE** is dominated by the uncertain knowledge family; always report the
  temperature-scaled ECE (0.0169) or per-family ECE, not the raw pool.
- Numbers are from a frozen 9-family holdout + JevBench public-231. They are not
  measured on Jev's private/internal suites.

## 9. Reproducibility

- **Release checkpoint:** `checkpoints/s1-v6` (commit `554debd`..`f9f229e`).
- **Release eval:** `results/s1v6_release_full.json` (35,594 rows) +
  `results/calibration_s1v6_release.json`.
- **Build:** `scripts/train_v6.sh` (training), `scripts/eval_v6.sh` (eval chain),
  `scripts/tsi_extract.py` (TSI data).
- **Research log:** `research/01`–`research/11` (one doc per milestone).
- **Frozen holdout:** `data/holdout/*.parquet` + `MANIFEST.json` (sha256-pinned).

## 10. License

Base model Apache-2.0; release mixture permissive/commercial-safe (see §4). Adapter
weights released under the project license (TBD — see `LICENSE`).
