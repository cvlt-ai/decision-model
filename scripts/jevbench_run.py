#!/usr/bin/env python
"""Run our s1-v3 on the JevBench public 231-item set, using the harness's own
`score_task` for grading. This is the shared yardstick AlexWortega/openjev used
for its 0.814 / Jev 0.866 numbers.

  * loads the 231 public tasks (original 72 -> 'standard' tier, easy 48, hard 111)
  * scores each with our logit-readout (native softmax, no verbalisation)
  * grades with jevbench.scoring.score_task (argmax, renorm band, exact labels)
  * reports per-tier accuracy (the 0.814-equivalent), overall, ECE (10-bin
    top-label), and Brier — plus latency.

  .venv/bin/python scripts/jevbench_run.py --ckpt checkpoints/s1-v3 --device cuda:0
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

JEB = ROOT / "third_party" / "jevbench"
for _p in (str(JEB),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from jevbench.tasks import load_jsonl  # noqa: E402
from jevbench.scoring import score_task  # noqa: E402
from jevbench.metrics import ece_top_label, brier_score  # noqa: E402
from eval.readout import build as readout_build  # noqa: E402

PUBLIC = JEB / "datasets" / "public"
TIERS = {"standard": "original.jsonl", "easy": "easy.jsonl", "hard": "hard.jsonl"}


def make_adapter(ckpt: str, base: str, device: str, temperature: float, key_batch: int):
    call = readout_build(ckpt, base, device, temperature, 1024, key_batch=key_batch)

    def run(task) -> dict:
        """-> DecisionResult-shaped dict {ok, probs, latency_s, error}"""
        q = task.question
        qdict = {"type": q["type"], "instructions": q["instructions"]}
        crit = q.get("criteria")
        if q["type"] == "noul":
            crit = crit or {}
            qdict["criteria"] = {"true": crit.get("true", ""), "false": crit.get("false", "")}
        else:
            qdict["criteria"] = crit or {}
        t0 = time.perf_counter()
        try:
            ans = call(task.state, {"a": qdict})["a"]
            lat = time.perf_counter() - t0
            if q["type"] == "noul":
                p = float(ans["noul"])
                return {"ok": True, "probs": {"yes": p, "no": 1.0 - p}, "latency_s": lat}
            # choice and score: readout returns probabilities keyed by the option
            # keys, which are exactly the task labels
            return {"ok": True, "probs": {str(k): float(v) for k, v in ans["probabilities"].items()},
                    "latency_s": lat}
        except Exception as e:
            return {"ok": False, "probs": None, "latency_s": time.perf_counter() - t0,
                    "error": f"{type(e).__name__}: {str(e)[:200]}"}

    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/s1-v3")
    ap.add_argument("--base", default="Qwen/Qwen3.5-4B")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--key-batch", type=int, default=8)
    ap.add_argument("--tiers", default=",".join(TIERS))
    ap.add_argument("--out", default="results/jevbench_s1v3_public231.json")
    a = ap.parse_args()

    tiers = [t for t in a.tiers.split(",") if t]
    run = make_adapter(a.ckpt, a.base, a.device, a.temperature, a.key_batch)

    all_records = []
    tier_acc = {}
    pair_conf, pair_ok = [], []  # (top-label confidence, correct) for ECE
    brier_sum, brier_n = 0.0, 0
    lats = []
    n_err = 0

    for tier in tiers:
        tasks = load_jsonl(str(PUBLIC / TIERS[tier]))
        correct = 0
        for t in tasks:
            r = run(t)
            if not r["ok"]:
                n_err += 1
                scored = {"valid": False, "correct": False, "predicted": None}
                probs = None
            else:
                scored = score_task(r["probs"], t)
                probs = scored.get("probs")
                if scored.get("correct"):
                    correct += 1
            lats.append(r["latency_s"])
            if probs:
                top = max(probs.values())
                pair_conf.append(top)
                pair_ok.append(1 if scored.get("correct") else 0)
                # Brier: expected label vs the model's distribution
                if t.expected is not None:
                    try:
                        brier_sum += brier_score(probs, str(t.expected), t.labels)
                        brier_n += 1
                    except Exception:
                        pass
            all_records.append({
                "tier": tier, "id": t.id, "family": t.family,
                "qtype": t.question["type"], "n_options": len(t.labels),
                "correct": scored.get("correct"), "predicted": scored.get("predicted"),
                "expected": t.expected, "valid": scored.get("valid"),
                "renormalized": scored.get("renormalized"),
                "latency_s": round(r["latency_s"], 4),
                "error": r.get("error"),
            })
        tier_acc[tier] = {"n": len(tasks), "correct": correct,
                          "accuracy": correct / len(tasks) if tasks else None}
        print(f"  [{tier:9}] {correct}/{len(tasks)} = {correct/len(tasks):.4f}", flush=True)

    total_n = sum(v["n"] for v in tier_acc.values())
    total_c = sum(v["correct"] for v in tier_acc.values())
    ece = ece_top_label(list(zip(pair_conf, pair_ok)))
    result = {
        "model": a.ckpt, "tiers": tier_acc,
        "overall": {"n": total_n, "correct": total_c, "accuracy": total_c / total_n if total_n else None},
        "ece_10bin_top_label": ece["ece"],
        "brier": brier_sum / brier_n if brier_n else None,
        "n_errors": n_err,
        "latency_p50_ms": round(statistics.median(lats) * 1000, 1) if lats else None,
        "latency_p95_ms": round(_pct(lats, 0.95) * 1000, 1) if lats else None,
        "note": "JevBench public 231 (original=standard 72, easy 48, hard 111); graded with "
                "jevbench.scoring.score_task. Compare: AlexWortega/openjev v5 = 0.814, Jev 1.13 = 0.866 on all-public.",
    }
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1) + "\n\n" +
                   json.dumps(all_records, indent=1))
    print(f"\n=== OVERALL {total_c}/{total_n} = {total_c/total_n:.4f} ===")
    print(f"ECE(10-bin top-label)={ece['ece']:.4f}  Brier={result['brier']:.4f}  "
          f"errors={n_err}  p50={result['latency_p50_ms']}ms p95={result['latency_p95_ms']}ms")
    print(f"wrote {out}")


def _pct(vals, q):
    s = sorted(vals)
    if not s:
        return 0.0
    k = (len(s) - 1) * q
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] + (s[c] - s[f]) * (k - f)


if __name__ == "__main__":
    main()
