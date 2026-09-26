"""VitaminC family-stratified flip probe.

A random 300-row sample rarely contains two conflicting siblings of the same
case_id (VitaminC families are ~3.35 rows/case), so random accuracy under-measures
the CONTRASTIVE property - the whole point of the axis. This probe selects case_ids
that carry >=2 siblings with >=2 distinct labels (true contrast families), scores all
their siblings, and reports:

  * overall + per-class accuracy on the conflict set
  * lazy rate: conflict families where the model predicted the SAME label for every
    sibling (fails to read the specific fact; a lazy prior would do this)
  * sensitivity: conflict families where the model's per-sibling predictions vary
    AND at least one sibling is graded correctly (it is tracking the label)

Usage:
  .venv/bin/python scripts/vitaminc_flip_probe.py --ckpt checkpoints/s1-v3 \
      --base Qwen/Qwen3.5-4B --device cuda:0 --families 200 --out results/vitaminc_flip_s1v3.json
"""
from __future__ import annotations
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
import datasets as _datasets_pkg  # real `datasets` lib, before eval/ shadows it
from datasets import load_dataset
import numpy as np
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "eval")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

HF = "tals/vitaminc"
SPLIT = "validation"  # == published VitaminC dev (63,054 rows)
SEED = 20260930  # same pinned seed as the eval adapter
CRIT = ["SUPPORTS", "REFUTES", "NOT ENOUGH INFO"]


def select_conflict_families(ds, n_families: int):
    """case_ids with >=2 siblings and >=2 distinct labels; deterministic sample.
    Returns {case_id: [row_indices]} for a stable, reproducible selection."""
    by_case = defaultdict(list)
    for i, r in enumerate(ds):
        by_case[str(r["case_id"])].append(i)
    conflict = {cid: idx for cid, idx in by_case.items()
                if len(idx) >= 2 and len({str(ds[j]["label"]) for j in idx}) >= 2}
    ids = sorted(conflict)
    rng = np.random.default_rng(SEED)
    sel = sorted(rng.choice(len(ids), size=min(n_families, len(ids)), replace=False))
    return {ids[k]: conflict[ids[k]] for k in sel}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--families", type=int, default=200, help="number of conflict case_ids")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    ds = load_dataset(HF, split=SPLIT)
    fams = select_conflict_families(ds, a.families)

    from eval.readout import build
    call = build(a.ckpt, a.base, a.device, a.temperature, a.max_len, key_batch=16)

    # questions template (same wording as the eval adapter)
    from eval.datasets.vitaminc import CRITERIA, INSTRUCTIONS
    questions = {"a": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": CRITERIA}}

    per = defaultdict(int); corr = defaultdict(int)
    total = 0; correct = 0
    n_lazy = 0; n_sensitive = 0; n_fams = 0
    for cid, idx in fams.items():
        preds = []
        ok = 0
        for j in idx:
            r = ds[j]
            state = {"evidence": r["evidence"], "claim": r["claim"]}
            gold = str(r["label"]).strip()
            ans = call(state, questions)["a"]
            pred = ans["choice"]
            per[gold] += 1; total += 1
            if pred == gold:
                corr[gold] += 1; correct += 1; ok += 1
            preds.append(pred)
        n_fams += 1
        if len(set(preds)) == 1:
            n_lazy += 1
        if len(set(preds)) > 1 and ok > 0:
            n_sensitive += 1

    res = {
        "n_families": n_fams,
        "n_rows": total,
        "accuracy": round(correct / total, 4) if total else None,
        "per_class": {c: {"n": per[c], "acc": round(corr[c] / per[c], 4) if per[c] else None}
                      for c in CRIT},
        "lazy_rate": round(n_lazy / n_fams, 4) if n_fams else None,
        "sensitivity_rate": round(n_sensitive / n_fams, 4) if n_fams else None,
    }
    print(json.dumps(res, indent=1))
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
