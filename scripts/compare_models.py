#!/usr/bin/env python
"""Consolidated comparison: s1-v1 / s1-v2 / s1-v3 / Laya, plus Jev's
DOCUMENTED numbers (from research/01) as reference. Reads the result JSONs on
disk; safe to re-run anytime. No network, no GPU.

    .venv/bin/python scripts/compare_models.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
R = ROOT / "results"

MODELS = {
    "s1-v1": "s1v1_frozen_300.json",
    "s1-v2": "s1v2_frozen_300.json",
    "s1-v3": "s1v3_frozen_300.json",
    "s1-v4": "s1v4_frozen_300.json",
    "laya": "laya_frozen_300.json",
}

# Jev numbers are DOCUMENTED in research/01-jev-mechanics.md (not our run):
#   MMLU-Pro (1000q): Jev 83%        BoolQ: Jev 91.6%
#   negation violation 0.19 (self-admitted); option-order shift up to ~0.20
JEV_DOC = {
    "mmlu_pro": 0.83,
    "boolq": 0.916,
    "negation_violation": 0.19,
    "perm_shift": 0.20,
}

FAM_ORDER = ["boolq", "banking77", "go_emotions", "mmlu_pro", "pubhealth",
             "injection", "severity", "negation", "baserate"]


def load(name):
    p = R / MODELS[name]
    if not p.exists():
        return None
    return json.load(open(p))


def main():
    models = {n: load(n) for n in MODELS}
    avail = [n for n in MODELS if models[n] is not None]

    # ---- per-family accuracy table
    cols = {n: models[n]["families"] for n in avail}
    print("Per-family accuracy  (" + ", ".join(avail) + ")")
    hdr = f"{'family':14}" + "".join(f"{n:>9}" for n in avail)
    print(hdr)
    for fam in FAM_ORDER:
        row = f"{fam:14}"
        for n in avail:
            a = cols[n].get(fam, {}).get("accuracy")
            row += f"{a:9.3f}" if a is not None else f"{'–':>9}"
        print(row)

    # ---- macro / headline axes
    print("\nHeadline axes (macro):")
    print(f"{'axis':22}" + "".join(f"{n:>10}" for n in avail) + f"{'jev(doc)':>10}")

    def row(label, key, jev=None):
        out = f"{label:22}"
        for n in avail:
            v = models[n]["macro"].get(key)
            out += f"{v:.3f}".rjust(10) if v is not None else "–".rjust(10)
        if jev is not None and jev in JEV_DOC:
            out += f"{JEV_DOC[jev]:.3f}".rjust(10)
        else:
            out += "".rjust(10)
        print(out)

    row("accuracy (macro)", "accuracy")
    row("ECE", "ece")
    row("Brier", "brier")
    row("negation violation", "negation_mean_violation", jev="negation_violation")

    # ---- permutation (fresh runs, separate files)
    print("\nPermutation robustness (fresh runs; flip rate, lower=better):")
    print(f"{'family':14}" + "".join(f"{n:>10}" for n in ["s1-v1", "s1-v2", "s1-v3"]) + f"{'jev(doc)':>10}")
    for fam in ["go_emotions", "mmlu_pro", "pubhealth"]:
        out = f"{fam:14}"
        for n in ["s1-v1", "s1-v2", "s1-v3"]:
            p = R / f"{n.replace('-', '')}_perm_clean.json"
            val = "–"
            if p.exists():
                d = json.load(open(p))
                fl = d["families"].get(fam, {}).get("perm_flip_rate")
                if fl is not None:
                    val = f"{fl:.3f}"
            out += f"{val:>10}"
        out += f"{'<=0.20shift':>10}"
        print(out)

    # ---- calibration sweep (offline)
    for n in ["s1-v2", "s1-v3"]:
        c = R / f"calibration_{n.replace('-', '')}.json"
        if c.exists():
            d = json.load(open(c))
            t1 = next((r for r in d["grid"] if r["temperature"] == 1.0), {})
            print(f"\nCalibration {n}: best_T={d['best_temperature']} "
                  f"ece@best={d['best']['ece']:.4f} (T=1.0 ece={t1.get('ece', float('nan')):.4f})")

    print("\nJEV_DOC reference:", JEV_DOC)
    print("(Jev figures are from research/01 documentation, not a local run.)")


if __name__ == "__main__":
    main()
