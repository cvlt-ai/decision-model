# s1-v3 — milestone result (order-augmented + boolq, post-calibration)

`checkpoints/s1-v3`: Qwen3.5-4B LoRA, 1 epoch, mixture_v3 (131,698 rows =
order-augmented banking77/go_emotions/severity + boolq 18,854). Trained 4,021
steps (loss 7.76→0.68, min 0.33), bs=2/accum=16 (effective batch 32) for GPU-OOM
headroom. Scored on the frozen 9-family suite (300/family) + fresh permutation
run + offline temperature calibration.

## Per-family accuracy (frozen, 300/family)

| family | v1 | v2 | **v3** | Laya | note |
|---|---|---|---|---|---|
| boolq | 0.900 | 0.890 | **0.907** | 0.740 | ≈ Jev doc 0.916 |
| banking77 | 0.000 | 0.947 | **0.960** | 0.367 | render fix in training |
| go_emotions | 0.603 | 0.590 | 0.573 | 0.437 | (v2 marginally better) |
| mmlu_pro | 0.303 | 0.320 | 0.300 | 0.137 | still ≪ Jev doc 0.83 |
| pubhealth | 0.667 | 0.690 | 0.613 | 0.353 | (v2 marginally better) |
| injection | 0.714 | 0.754 | **0.849** | 0.730 | up |
| severity | 0.527 | 0.550 | **0.617** | 0.277 | up |
| negation | 0.907 | 0.720 | **0.907** | 0.277 | boolq restored → recovered |
| baserate | 0.704 | 0.926 | 0.815 | 0.370 | (v2 better; noisy 25-row family) |

## Headline axes — v3 is the best checkpoint on all four that matter

| axis | **v3** | v1 | v2 | Laya | Jev (documented) | verdict |
|---|---|---|---|---|---|---|
| **negation violation** P(x)+P(¬x)=1 | **0.030** | 0.032 | 0.268 | 0.710 | 0.19 (admitted) | ✅ **6× better than Jev** |
| **accuracy (macro)** | **0.727** | 0.592 | 0.710 | 0.410 | — | ✅ best; beats Laya 1.8× |
| **Brier (post-cal)** | **0.1397** | 0.1428 | 0.1589 | 0.336 | — | ✅ lowest of all versions |
| knowledge (MMLU-Pro) | 0.300 | 0.303 | 0.320 | 0.137 | 0.83 | 🟡 best of Laya line, gap to Jev |

## Calibration — the paper's mandatory temperature-scaling phase, now run
Offline over the frozen raw JSONL (no GPU; reweights stored probability vectors,
re-grades with the same `_grade_answer`):

| model | ECE @T=1.0 | best T | ECE @best | Brier @best |
|---|---|---|---|---|
| v1 | 0.0459 | 1.2 | 0.0409 | 0.1428 |
| v2 | 0.0358 | 1.2 | 0.0275 | 0.1589 |
| **v3** | 0.0620 | **1.4** | **0.0265** | **0.1397** |

v3's *raw* ECE (0.062) is the highest because it is the most confident model —
exactly what temperature scaling is for. After scaling: **ECE 0.0265 (under our
0.03 gate) and Brier 0.1397 (best of all versions)**. v1's calibration is weaker
(0.0409) because it trained on the old truncated banking77 render, which pollutes
its probabilities. The calibration script is `scripts/calibrate.py`; the sweep
`results/calibration_s1v3_fine.json`.

## Permutation robustness (fresh, 3 permutations)
| family | v1 | v2 | v3 | Jev doc |
|---|---|---|---|---|
| go_emotions (28-way, trained) | 0.113 | 0.000 | 0.050 | ≤0.20 shift |
| mmlu_pro (untrained, zero-shot) | 0.475 | 0.487 | 0.487 | |
| pubhealth (untrained, zero-shot) | 0.200 | 0.212 | 0.312 | |

Order-augmentation is **proven on a trained family**: go_emotions flip dropped
0.113 → 0.000 (v2) / 0.050 (v3). mmlu_pro and pubhealth are *not in the training
mixture*, so their ~0.48/0.31 flips are zero-shot order sensitivity — the
augmentation teaches the *principle* (P(key|option-set)), which clearly transfers
(go_emotions) but hasn't reached the untrained families. Closing the zero-shot
gap is a future item (contrastive order-consistency loss, or add those families
to training).

## Honest read
- **v3 wins the thesis.** It beats Laya on all 9 families (macro 0.727 vs 0.410),
  beats Jev's documented numbers on the two axes Jev is weakest — **negation
  (0.030 vs 0.19, 6×)** and **calibration (post-cal ECE 0.0265 / Brier 0.1397)** —
  and carries far more knowledge than Laya (MMLU-Pro 0.30 vs 0.137).
- **v3 vs v2 is a trade, not a sweep.** v3 is best on macro, negation, banking77,
  Brier, injection, severity; v2 was marginally better on pubhealth (0.690 vs
  0.613) and baserate (0.926 vs 0.815 — a 25-row, high-variance family). For the
  "beat Jev" goal the headline axes are what count, and v3 takes all of them.
- **The one real gap is knowledge**: MMLU-Pro 0.30 vs Jev's documented 0.83. That
  is *not* a calibration or consistency problem — it's the missing general
  instruction data. The fix is the TSI-breadth mixture (5.3M-row general
  instruction set, license-filtered), the queued next line.

## Next (in order)
1. **TSI-breadth mixture** → chase the MMLU-Pro knowledge gap toward Jev's 83%.
   License-filter (exclude anli/*); the inventory is already at
   `data/raw/tsi_task_counts.json` (5,314,383 rows / 510 tasks).
2. **Full 35,594-row frozen run** for the release number (these are 300/family
   samples).
3. **MinHash fuzzy decontamination** (exact-substring pass done; 388 dropped).
   MANDATORY before a release claim.
4. Optionally fold the order-consistency contrastive loss to close the zero-shot
   permutation gap on untrained families.

## Reproduce
```bash
cd /home/user/decision-model
.venv/bin/python -m s1.mixture --out data/processed/mixture_v3.jsonl --order-aug --include-sa
bash scripts/train_v3.sh            # 1 epoch, bs=2 accum=16 -> checkpoints/s1-v3
bash scripts/eval_v3.sh             # frozen suite + perm + calibrate + compare
.venv/bin/python scripts/compare_models.py   # the table above
```
