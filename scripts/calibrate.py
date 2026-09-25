"""Post-hoc temperature calibration, OFFLINE (no GPU, no model re-run).

The frozen raw JSONL stores each answer's full probability vector at T=1.0.
Temperature scaling is a deterministic re-weighting of those vectors:
    logits_i = log(p_i);  p_i(T) = p_i^(1/T) / sum_j p_j^(1/T)
so we can sweep T, re-grade every answer with the SAME `_grade_answer` the
online runner uses, and report ECE/Brier at each T. This is the standard
post-hoc calibration point (the reported key-distribution), exactly what the
paper's "mandatory temperature scaling" phase is for.

    python scripts/calibrate.py --raw results/raw_s1v3_frozen.jsonl --tag s1v3

Writes results/calibration_<tag>.json: the full sweep + the T that minimizes ECE.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s1.metrics import brier, ece  # noqa: E402
from eval.run_eval import _grade_answer, load_holdout  # noqa: E402


def _rescale(probs: list[float], T: float) -> list[float]:
    """softmax(log(p)/T) in log-space; T=1 returns p unchanged (up to fp)."""
    if T == 1.0:
        return probs
    import math

    lp = [math.log(max(p, 1e-12)) / T for p in probs]
    m = max(lp)
    ex = [math.exp(x - m) for x in lp]
    z = sum(ex)
    return [e / z for e in ex]


def _rebuild(ans: dict, T: float) -> dict:
    """Rescale an answer's distribution at temperature T, keep its type/shape."""
    t = ans["type"]
    if t == "noul":
        p = float(ans["noul"])
        np_ = _rescale([p, 1.0 - p], T)[0]
        return {"type": "noul", "noul": np_}
    if t == "choice":
        keys = list(ans["probabilities"].keys())
        np_ = _rescale([float(ans["probabilities"][k]) for k in keys], T)
        return {
            "type": "choice",
            "choice": ans["choice"],  # argmax invariant under monotone rescale
            "probabilities": {k: v for k, v in zip(keys, np_)},
        }
    if t == "score":
        keys = [int(k) for k in ans["probabilities"].keys()]
        np_ = _rescale([float(ans["probabilities"][str(k)]) for k in keys], T)
        newprobs = {str(k): v for k, v in zip(keys, np_)}
        return {
            "type": "score",
            "score": sum(k * newprobs[str(k)] for k in keys),
            "legend": ans.get("legend", {}),
            "probabilities": newprobs,
        }
    raise ValueError(f"unknown answer type {t}")


def calibrate(raw_path: Path, families: list[str], grid: list[float]) -> dict:
    data = load_holdout(families)
    uid_to_ex = {}
    for fam, rows in data.items():
        for ex in rows:
            uid_to_ex[ex.uid] = ex

    records = []  # (ex, qid, answer@T1)
    for line in raw_path.read_text().splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        ex = uid_to_ex.get(d["uid"])
        if ex is None:
            continue
        qid = next(iter(ex.questions))
        records.append((ex, qid, d["answers"][qid]))

    sweep = []
    for T in grid:
        p_top, y, ok_n = [], [], 0
        for ex, qid, ans in records:
            rescaled = _rebuild(ans, T)
            ok, pt, _ = _grade_answer(rescaled, ex.gold[qid])
            # exclude prob-gold baserate parity rows from calibration exactly as run_eval does
            if ex.gold[qid].kind == "prob" and ex.meta.get("subkind") == "parity":
                ok_n += int(ok)
                continue
            p_top.append(pt)
            y.append(1.0 if ok else 0.0)
            ok_n += int(ok)
        sweep.append({
            "temperature": T,
            "accuracy": ok_n / len(records) if records else None,
            "ece": ece(p_top, y) if p_top else None,
            "brier": brier(p_top, y) if p_top else None,
            "n": len(records),
        })

    # pick T that minimizes ECE (calibration's job; accuracy should be flat)
    best = min(sweep, key=lambda r: r["ece"])
    return {"grid": sweep, "best_temperature": best["temperature"],
            "best": best, "n_scored": len(records)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--families", default=",".join(
        ["boolq", "banking77", "mmlu_pro", "injection", "go_emotions",
         "pubhealth", "severity", "negation", "baserate"]))
    ap.add_argument("--grid", default="0.5,0.6,0.7,0.8,0.9,1.0,1.1,1.2,1.3,1.5,2.0")
    a = ap.parse_args()
    fams = [f for f in a.families.split(",") if f]
    grid = [float(x) for x in a.grid.split(",") if x]
    res = calibrate(Path(a.raw), fams, grid)
    out = Path("results") / f"calibration_{a.tag}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))
    print(f"n_scored={res['n_scored']}  best_T={res['best_temperature']} "
          f"ece@best={res['best']['ece']:.4f} brier@best={res['best']['brier']:.4f} "
          f"(T=1.0 ece={next(r['ece'] for r in res['grid'] if r['temperature'] == 1.0):.4f})")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
