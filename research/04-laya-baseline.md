# Baseline: Laya 0.3.5 on the frozen suite (300 rows/family, CPU)

Run 2026-09-22, `results/laya_frozen_300.json`, raw answers `results/raw_laya.jsonl`
(resumable; scoring is re-computable from raw without re-running inference — the
9-family re-score reproduced every 8-family number exactly from the resume log).
Suite v2 = 9 families / 35,594 rows (severity added so all three primitives —
choice, noul, score — are covered; the first baseline had no score family at all).
Machine: CPU only (`CUDA_VISIBLE_DEVICES=""`, OMP 16 threads) — GPUs held by llama-server.
Grading semantics fixed BEFORE reading results (nearest-level ±0.5 for score, argmax for
choice/noul, |p−target|≤0.15 for base-rate prob items, paired |P(x)+P(¬x)−1| for negation).

## Macro

| metric | value | our gate for M1/M2 |
|---|---|---|
| accuracy (9-family macro) | **0.410** | ≥ 0.55 first release |
| ECE (15-bin) | **0.303** | ≤ 0.03 after recalibration (paper B: recalibration cuts 3.3×) |
| Brier | 0.336 | — |
| negation violation mean | **0.710** | ≤ 0.05 (hard architectural constraint: same-head negation) |
| p50 latency (CPU) | 199 ms | ≤ 50 ms on GPU (Jev hosted: 70–500 ms) |
| permutation shift | not measured (needs `--permutations 5`) | ≤ 0.02 |

## Per family

| family | acc | ECE | read |
|---|---|---|---|
| boolq (noul) | 0.740 | 0.122 | its best showing; still below what a fine-tuned encoder should do (~0.85+) |
| banking77 (77-way) | 0.367 | 0.538 | near-duplicate intents collapse (`verify_my_identity` @ p=1.0 vs gold `why_verify_identity`) — confident-wrong is the failure mode, not confusion |
| mmlu_pro (≤10-way) | **0.137** | 0.220 | ≈ chance. Laya has essentially **no knowledge**. This is the column where Jev claims 83% — the whole reason M2 (decoder) exists |
| injection (noul) | 0.730 | 0.167 | plain 83/116 = 0.72; homoglyph variants 9/10 (small n) — encoding robustness is NOT its weakness, plain misses are |
| go_emotions (28-way) | 0.437 | 0.448 | reasonable-ish accuracy, terrible calibration (ECE ≈ accuracy/2) |
| pubhealth (MCQ) | 0.353 | 0.099 | low acc but well-calibrated low confidence — it knows it doesn't know. Contrast banking77 |
| severity (score, 4 levels) | 0.277 | 0.178 | ≈ chance (25%). First-ever score-primitive run through the runner; Laya answers a Solidity severity ladder with a flat prior (e.g. {0:.09 1:.25 2:.46 3:.19}, confidence 0.10) — honest flatness, but no signal. Vendor's own urgency example scored 1.75 on a different scale, so score transfer across domains is weak everywhere |
| negation (noul, flipped boolq) | **0.277** | 0.593 | vs 0.740 affirmative: it does not process the NOT in the question |
| baserate (27 self-authored) | 0.370 | 0.372 | parity 9/18 = coin-flip guessing; coin 1/6; card **0/3** — every unobserved-event probability confidently wrong |

## What this tells the competition plan

1. **The bar is low and the axes are known.** Laya is the only serious open artifact and it
   is near-chance on knowledge, chance-level on parity, and violates P(x)+P(¬x)=1 by 0.71.
   TypeSafe's own admitted violation was 0.19 (Jev 1.13 jaggedness page).
2. **Confident-wrong is the pattern to break.** Same accuracy with honest confidence
   (pubhealth behavior) is what ECE ≤ 0.03 means; banking77/go_emotions show the encoder
   line defaults to p≈1.0 garbage when it's outside its training distribution.
3. **Negation must be architectural, not data.** 0.74 → 0.28 under negation is a
   representation failure; we score it as a first-class gate and train the
   consistency penalty (`scoring_rules.negation_penalty`) from step 1.
4. **Knowledge is not free.** MMLU-Pro 0.137 confirms an encoder-only line cannot close
   Jev's knowledge column; M2 (Qwen3-4B-class + logit readout, Open-Jev recipe) is
   confirmed as the primary line, M1 as the speed line.
5. **Homoglyph variants were the right call but aim at the wrong foe for Laya**
   (9/10 caught). Keep them — hosted Jev reportedly misses exactly these (Aitejiu harness);
   they stay in the suite as the Jev-facing axis.
6. Latency budget confirmed: 202 ms CPU p50 on a 0.4B encoder; GPU should land ~10-20 ms,
   well inside the hosted-Jev 70-500 ms window.

## Reproduce

```bash
cd /home/user/decision-model
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=16 .venv/bin/python -m eval.run_eval \
  --model laya --limit 300 --raw results/raw_laya.jsonl --out results/laya_frozen_300.json
```
Full-suite (35,157 rows) + `--permutations 5` run is the pre-training freeze — est. ~6 h CPU;
schedule when the box is otherwise idle or after GPUs free.
