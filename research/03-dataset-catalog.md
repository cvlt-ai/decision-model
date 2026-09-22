# Dataset catalog — verified status, licenses, and two papers that change the design

Updated 2026-09-22. Availability/license checked live against the HF API with the
configured account (impossibleexchange). Nothing marked ✗ may be used for training.

## New findings from yesterday's literature sweep (post-dates the original plan)

### Paper A — arXiv 2609.23959, "Open-Jev Judgments on CallScreenBench" (2026-09-21)
Open readout implementation ("JevLite"): **Qwen3-4B LoRA-tuned so that the
temperature-scaled softmax over two answer-label logits is P(scam)**. Results:
AUROC .974, **ECE .052**, 64.5 ms/decision on one consumer GPU, **4.9× faster than
the same backbone fine-tuned to generate** its answer.

Consequences for us:
1. **It validates the M2 recipe end-to-end** (4B LoRA + readout + temperature
   scaling) — we are not exploring blind; and its numbers give a realistic target.
2. **It challenges M1's necessity**: "a fine-tuned ModernBERT encoder is not
   significantly worse" on their task. Their honest caveats cut both ways: "the
   recipe was selected with test-set exposure" and "all callers are synthetic".
   Decision: we still build both (their task is narrow; our eval is multi-family),
   but M2 is now the *primary* model line and M1 the speed line, and we must
   pre-register our eval before recipe selection to avoid their exact mistake.
3. Their citations codify two traps we already designed around: **option order
   changes answers**, and **instruction tuning pushes answer-token scores toward
   overconfidence** (⇒ temperature scaling on a held-out split is mandatory for M2,
   not optional).

### Paper B — arXiv 2609.24052, "Calibrated Decisions at Scale" (police crash narratives, 2026-09-21)
Jev coding 195,857 Texas crash narratives with a 27-question schema; audited against
2,416 blinded human judgments. Key results:
- F1 0.908 against human labels; the probabilities **"overstate prevalence until
  recalibrated on labels"**; **"calibration varies by model rather than by paradigm,
  so each model must be audited"**; recalibration on the same labels **cuts
  calibration error 3.3×**.
- A two-decimal output grid (Jev reports 0.95, not 0.9473) **puts a floor on
  achievable calibration error for rare classes**.

Consequences: Phase 5 (calibration) is load-bearing, not polish. Our C3 gate (ECE
≤ 0.03) must be measured *after* our own recalibration, and our probe set needs
rare-class coverage because of the resolution floor.

### Also logged
- `kierandotai/jev-scout` golden-set study: **the same runs score 28% vs 88%
  depending purely on band-matching semantics for `score`** — score-grading
  semantics must be fixed *before* reading results (we adopt nearest-level with
  ±0.5 tolerance, declared here and implemented in code).
- `Aitejiu/jev-harness-lab`: 10-dataset public eval against hosted Jev; Jev fails
  model-difficulty routing (51%, no signal) and trajectory attribution (AUROC
  0.56); injection recall 85% with encoding-bypass misses (Cyrillic homoglyphs) —
  we add homoglyph/encoding variants to our injection eval.
- `CallScreenBench` (paper A): 41 scenarios / 577 per-turn decisions, synthetic.

## Eval-suite datasets (Phase 2) — verified live 2026-09-22

| HF id | license | gated | split / N (built) | primitive | role |
|---|---|---|---|---|---|
| `google/boolq` | cc-by-sa-3.0 | no | validation / **3,270** | noul | reading-comp yes/no |
| `mteb/banking77` | **mit** | no | test / **3,076** | choice (77) | intent routing |
| `TIGER-Lab/MMLU-Pro` | **mit** | no | test / **12,032** | choice (≤10) | knowledge/reasoning |
| `deepset/prompt-injections` | apache-2.0 | no | test / **126** (116 + 10 homoglyph variants) | noul | safety gate |
| `google-research-datasets/go_emotions` | apache-2.0 | no | test / **5,427** | choice (28) | fine-grained affect |
| `Joshua-Harris/PubHealthBench` | **cc-by-4.0** | no | test / **7,929** | choice (≤10) | claim support |

Build corrections found while implementing the adapters (2026-09-22):
- `PolyAI/banking77` and `bigbio/pubhealth` both **ship loading scripts, which
  datasets 4.x refuses to run**. Working mirrors: `mteb/banking77` (parquet, MIT)
  and `Joshua-Harris/PubHealthBench` (parquet, CC-BY-4.0). PubHealthBench is also
  the better pick on content — full MCQ rather than binary claim-support.
- `go_emotions` requires the full `google-research-datasets/` namespace id.
- Frozen suite total: **35,157 rows** across 8 families + `MANIFEST.json` sha256s.

Plus two
**self-authored probe families** needing no downloads: `baserate` (die/coin/card
questions with programmatically verifiable gold, incl. true-base-rate items whose
honest answer is 0.5) and `negation` (each boolq item re-asked negated → measures
the P(x)+P(¬x)=1 invariant both Jev and Laya violate).

## Training corpora (Phase 3) — verified live

| HF id | license | gated | usable? | role |
|---|---|---|---|---|
| `tasksource/tasksource-instruct-v0` | **apache-2.0** | no | ✓ | PRIMARY breadth (~485 discriminative tasks; NLI-heavy) |
| `BAAI/Infinity-Instruct` | cc-by-sa-4.0 | **gated=auto** | ⚠ needs click-through; SA license | backup breadth |
| ~~`tasksource/infinite-instruct`~~ | — | — | ✗ 404 (confirmed gone, not a probe error) | replaced by TSI |
| `nyu-mll/multi_nli` | cc-by-3.0 / cc-by-sa / mit / other | no | ✓ (use the mit-tagged subset) | NLI for noul |
| `stanfordnlp/snli` | cc-by-sa-4.0 | no | ⚠ SA license → research-only line | NLI for noul |
| `allenai/wildguardmix` | odc-by | **gated=auto** | ⚠ click-through; attribution | injection/safety train |
| `openai/gsm8k` | **mit** | no | ✓ | numeric/base-rate-adjacent reasoning |
| `allenai/ai2_arc` | cc-by-sa-4.0 | no | ⚠ SA → research line | reasoning choice |
| `rowan/hellaswag` | (none stated) | no | ⚠ flag | reasoning choice |
| `anli` | **cc-by-nc-4.0** | no | ✗ non-commercial | excluded |
| `clinc_oos` | cc-by-3.0 | no | ✓ | routing |
| `amazon_polarity` | apache-2.0 | no | ✓ | binary/score |
| `SetFit/sst5` | (none stated) | no | ⚠ flag | ordinal score |
| `social_i_qa`, `piqa`, `openbookqa` | unknown | no | ⚠ flag | commonsense |
| `facebook/xnli` | (none stated) | no | ⚠ flag | NLI multilingual (later) |

Policy: **✗ excluded; ⚠ quarantined** into a separately-named research checkpoint,
never in a permissively-licensed release. ✓ = safe for the Apache-2.0 release line.
The breadth backbone is TSI (Apache-2.0) — cleaner than the plan's original draft.

## What we do NOT have
- No hosted-Jev API key → all Jev numbers remain "vendor-published".
- No public Jev-style leaderboard exists yet (jev-scout says as much) — our suite
  being published and frozen is the contribution either way.
