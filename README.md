# system one (s1)

_A local, open, Jev-compatible "System One" decision model._

`system one` answers [TypeSafe Jev](https://jev.ai)-style questions — `choice`,
`score`, `noul` — over a free-form **state**, and returns **calibrated
probabilities**, not just a bare answer. It is a thin LoRA adapter on an Apache-2.0
base (Qwen3.5-4B), so it is small (124 MB adapter), open, and auditable, while still
carrying the general knowledge a purpose-built encoder can't.

The design goal was not only to match Jev's accuracy but to be structurally strong where
Jev and its open clones are weak: **calibration**, **negation/invariance consistency**,
and **coverage** (knowing when the evidence is insufficient).

> **Status (2026-09-29):** `s1-v6` is the release checkpoint. Full 35,594-row frozen
> holdout: **macro 0.7747 · mmlu_pro 0.438 · negation 0.022 · ECE 0.0169 (T=1.5)**.
> See [`MODEL_CARD.md`](MODEL_CARD.md) for the full card.

---

## Quickstart — serve it

```bash
# uv venv (conda activate is flaky in this shell); deps in pyproject.toml
cd decision-model
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
  .venv/bin/python scripts/serve_s1.py \
  --ckpt checkpoints/s1-v6 --base Qwen/Qwen3.5-4B \
  --device cuda:0 --temperature 1.5 --max-len 1024 --key-batch 8 --port 8000
```

Then:

```bash
curl http://127.0.0.1:8000/healthz
curl http://127.0.0.1:8000/v1/answer -H 'Content-Type: application/json' -d '{
  "state": "A customer writes: I sent the transfer yesterday and it is still not in my account, my account is frozen and I cannot withdraw.",
  "questions": {
    "intent": {
      "type": "choice",
      "instructions": "Classify the customer primary intent.",
      "criteria": {
        "transaction_pending": "The money has not cleared yet.",
        "frozen_account": "The account is blocked or restricted.",
        "fee_question": "A question about charges."
      }
    },
    "urgent": { "type": "noul", "instructions": "Likely to escalate to a complaint?" }
  }
}'
```

```json
{"answers":{"intent":{"type":"choice","choice":"frozen_account",
  "probabilities":{"transaction_pending":0.178,"frozen_account":0.82,"fee_question":0.0017}},
  "urgent":{"type":"noul","noul":0.881}},"ms":498.5}
```

`s1-v6` is served at the release-calibrated **T=1.5** (see model card §5).

## The wire format

`src/s1/schema.py` is a reconstruction of the Jev wire spec (written against the vendor's
own archived docs, not guesses). A request is `{state, questions}`; each question is one
of:

| type | criteria | answer |
|---|---|---|
| `choice` | `{key: description}` (≤255 options) | chosen key + probability vector |
| `score` | ordered array (2–10 levels) | expected score + level distribution |
| `noul` | `{true, false}` or absent | probability in [0,1] |

## Repository layout

```
src/s1/            the model package
  schema.py            Jev wire-spec: request parsing, rendering, answer building
  metrics.py           ECE / Brier / coverage / negation-violation / permutation
  mixture.py           training-data builder + the single prompt-render source of truth
  scoring_rules.py     per-family grading
eval/
  readout.py           LogitReadout — single-token logit scoring over option keys
  run_eval.py          frozen-holdout builder + scored runner
  datasets/            one adapter per eval family (boolq, mmlu_pro, banking77, …)
scripts/
  train_s1.py          LoRA SFT trainer (periodic checkpoints, decontam, resume)
  serve_s1.py          FastAPI server (this README's quickstart)
  calibrate.py         offline temperature sweep (the mandatory calibration phase)
  tsi_extract.py       TSI breadth-data extractor
  jevbench_run.py      JevBench public-231 harness
  vitaminc_flip_probe.py   evidence-sensitivity (NLI) probe
  supervise_v6.sh      crash-proof supervisor (relaunch + auto-eval chain)
  eval_v6.sh           full eval chain
checkpoints/s1-v6/   the release adapter (124 MB LoRA) + tokenizer
data/holdout/        the frozen 35,594-row eval suite (sha256-pinned)
results/             the research record (every run's JSON, committed)
research/            01–11: one doc per milestone
```

## How it's scored

The headline is a **frozen holdout** — 35,594 rows across 9 families, pinned before any
training and never touched by it. The release number is the *full* holdout (not the
300/family sample used during iteration). JevBench public-231 is a second,
independent yardstick. The things that distinguish s1 (negation consistency,
option-order invariance, VitaminC flip/coverage) are measured by dedicated probes that
JevBench does not.

## Key results

| | s1-v6 (release, full holdout) |
|---|---:|
| macro accuracy (9 families) | **0.7747** |
| mmlu_pro | 0.438 |
| negation violation (lower better; Jev doc 0.19) | **0.022** |
| ECE @ best T (1.5) | **0.0169** |
| JevBench public-231 @4096 | 0.7749 (hard 62/111) |

See [`MODEL_CARD.md`](MODEL_CARD.md) §5 for the full per-family table, calibration,
JevBench, and the evidence-sensitivity probe.

## Development

- **Tests:** `.venv/bin/python -m pytest -q -m "not gpu"` (GPU tests tagged `gpu`).
- **Reproduce v6:** `scripts/train_v6.sh` (train) → `scripts/eval_v6.sh` (eval).
- **Conventions:** uv venv at `.venv` (run via `.venv/bin/python`); HF account
  `impossibleexchange`; train/eval on `cuda:0` (the llama-server holds `cuda:1`/`cuda:2`).
  License: permissive/commercial-safe for the release mixture; non-commercial rows
  quarantined. DRY, YAGNI, TDD, frequent commits.

## Where it stands vs Jev

s1-v6 is within **0.09** of Jev 1.13 on JevBench (0.7749 vs 0.866) and ahead on the
calibration and negation axes. The one axis it is clearly behind is **knowledge**
(mmlu_pro 0.438 vs Jev's documented 0.83) — that is the next lever (a broader TSI
extraction), not a spread across everything. Full version history: `research/05`–`11`.

## License

Base model Apache-2.0; release mixture permissive/commercial-safe. Adapter weights under
the project license (TBD — see `LICENSE`).
