"""Frozen holdout builder + eval runner.

    build_holdout()             -> writes data/holdout/<family>.parquet + MANIFEST.json
    python -m eval.run_eval --model laya --out results/laya.json

Scoring semantics, fixed BEFORE any result is read (jev-scout showed band semantics
alone moved their number 28% -> 88%):
  * noul / choice: exact label match (argmax)
  * prob golds (baserate coin/card): |p - target| reported; "correct" iff <= 0.15
  * score answers: nearest level (argmax) with +/-0.5 tolerance, per vendor advice
    NOT to interpolate
  * negation: paired by meta.pair_uid; violation = |p(x) + p(not x) - 1|
  * calibration: ECE (15 bins) and Brier over noul/top-label probabilities
  * permutation robustness: each choice item re-run with K shuffled option orders;
    probabilities aligned by option KEY before differencing
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

from s1.metrics import argmax_flip_rate, brier, coverage_accuracy, ece, permutation_robustness
from s1.schema import build_answer, confidence_from_probs
from s1.schema import parse_request

from .datasets import all_adapters
from .datasets.base import EvalExample, Gold, content_hash

ROOT = Path(__file__).resolve().parent.parent
HOLDOUT = ROOT / "data" / "holdout"
RESULTS = ROOT / "results"

FAMILIES = [
    "boolq",
    "banking77",
    "mmlu_pro",
    "injection",
    "go_emotions",
    "pubhealth",
    "severity",
    "negation",
    "baserate",
]


# ------------------------------------------------------------------ holdout


def build_holdout(limit: int | None = None, families: list[str] | None = None) -> dict:
    """Materialise the frozen suite. limit=None -> full configured N per family."""
    all_adapters()
    from eval import datasets as dsmod  # registry now populated

    HOLDOUT.mkdir(parents=True, exist_ok=True)
    manifest = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "families": {}}
    counts = {}
    for fam in families or FAMILIES:
        fn = dsmod.REGISTRY[fam]
        rows = fn(limit)
        seen = {r.uid for r in rows}
        if len(seen) != len(rows):
            raise ValueError(f"{fam}: duplicate uids in holdout")
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pylist([r.to_dict() for r in rows])
        pq.write_table(table, HOLDOUT / f"{fam}.parquet")
        h = content_hash(rows)
        manifest["families"][fam] = {
            "n": len(rows),
            "sha256": h,
            "adapter": f"{fn.__module__}:{fn.__qualname__}",
        }
        counts[fam] = len(rows)
    (HOLDOUT / "MANIFEST.json").write_text(json.dumps(manifest, indent=1))
    return counts


def load_holdout(families: list[str] | None = None) -> dict[str, list[EvalExample]]:
    import pyarrow.parquet as pq

    out = {}
    want = families or FAMILIES
    for fam in want:
        p = HOLDOUT / f"{fam}.parquet"
        if p.exists():
            out[fam] = [EvalExample.from_dict(d) for d in pq.read_table(p).to_pylist()]
        else:
            raise FileNotFoundError(
                f"holdout family '{fam}' missing at {p}; run: python -m eval.run_eval --build"
            )
    return out


# ------------------------------------------------------------------ backends


def laya_backend():
    """Returns fn(state, questions) -> dict of qid -> answer dict (Jev-shaped)."""
    import laya

    agent = laya.load("convaiinnovations/laya")

    def call(state, questions):
        out = agent.predict(state, questions)
        return out["answers"] if isinstance(out, dict) and "answers" in out else out

    return call


def endpoint_backend(url: str):
    import httpx

    client = httpx.Client(timeout=120)

    def call(state, questions):
        r = client.post(url, json={"state": state, "questions": questions})
        r.raise_for_status()
        return r.json()["answers"]

    return call


# ------------------------------------------------------------------ scoring


def _grade_answer(ans: dict, gold: Gold) -> tuple[bool, float, float]:
    """-> (correct, p_of_gold_or_top, p_brier_target)."""
    kind = gold.kind
    if ans["type"] == "noul":
        p = float(ans["noul"])
        if kind == "prob":
            correct = abs(p - float(gold.value)) <= 0.15
            # keep these IN calibration: Laya's 0.158 coin answer becomes a
            # 0.842-confidence wrong top-label, which ECE must punish
            top_p = max(p, 1.0 - p)
            return correct, top_p, top_p
        correct = (gold.value == "true") == (p >= 0.5)
        pred = p if p >= 0.5 else 1 - p  # top-label prob
        return correct, pred, p if gold.value == "true" else 1 - p
    if ans["type"] == "choice":
        probs = ans["probabilities"]
        correct = ans["choice"] == str(gold.value)
        # calibration column = top-label CONFIDENCE (what ECE/Brier are defined on),
        # not P(gold): P(gold) collapses discrimination into the calibration metric
        top_p = max(float(v) for v in probs.values())
        return correct, top_p, top_p
    # score: nearest-level semantics, never interpolation
    probs = {int(k): v for k, v in ans["probabilities"].items()}
    top = max(probs, key=lambda k: probs[k])
    correct = abs(top - float(gold.value)) <= 0.5
    return correct, probs[top], probs[top]


def _perm_probe(call, ex: EvalExample, k: int = 5) -> tuple[list[list[float]], list[str]]:
    """Re-run a choice item with K option orders. Returns prob matrices aligned by
    option key (rows = permutations)."""
    qid = next(iter(ex.questions))
    q = ex.questions[qid]
    if q["type"] != "choice":
        return [], []
    import random
    import zlib

    keys = list(q["criteria"].keys())
    runs = []
    # zlib.crc32, not builtin hash(): builtin hash is salted per process and would
    # silently change the permutation probe between runs.
    rng = random.Random(zlib.crc32(ex.uid.encode()))
    for i in range(k):
        order = keys[:]
        if i:
            rng.shuffle(order)
        qq = {qid: {**q, "criteria": {kk: q["criteria"][kk] for kk in order}}}
        ans = call(ex.state, qq)[qid]
        probs = [float(ans["probabilities"][kk]) for kk in keys]  # aligned by key
        runs.append(probs)
    return runs, keys


def run_eval(model: str, families: list[str], limit: int | None, permutations: int,
             endpoint: str | None = None, ckpt: str | None = None,
             raw_path: Path | None = None, fresh: bool = False) -> dict:
    if model == "laya":
        call = laya_backend()
    elif model == "endpoint":
        if not endpoint:
            raise SystemExit("--model endpoint requires --endpoint URL")
        call = endpoint_backend(endpoint)
    else:
        raise SystemExit(f"backend '{model}' not wired yet (laya|endpoint for now)")

    data = load_holdout(families)

    # incremental raw log: a crash must not burn hours of inference; on restart we
    # skip uids already answered. One JSON object per line: {uid, answers, ms}.
    if raw_path is None:
        raw_path = RESULTS / f"raw_{model}.jsonl"
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    done: dict[str, dict] = {}
    if raw_path.exists() and not fresh:
        for ln in raw_path.read_text().splitlines():
            try:
                d = json.loads(ln)
                done[d["uid"]] = d
            except json.JSONDecodeError:
                continue  # a torn last line from a crash; safe to drop
    raw_f = raw_path.open("a")

    per_family, all_p, all_y, perm_stats = {}, [], [], []
    neg_pairs: dict[str, dict] = {}
    lat = []

    for fam, rows in data.items():
        if limit:
            rows = rows[:limit]
        n_ok = 0
        fam_p, fam_y, fam_lat = [], [], []
        for ex in rows:
            if ex.uid in done:
                rec = done[ex.uid]
                answers, dt = rec["answers"], rec["ms"]
            else:
                t0 = time.perf_counter()
                answers = call(ex.state, ex.questions)
                dt = (time.perf_counter() - t0) * 1000
                raw_f.write(json.dumps({"uid": ex.uid, "answers": answers, "ms": dt}) + "\n")
                raw_f.flush()
            fam_lat.append(dt)
            qid = next(iter(ex.questions))
            ok, p_top, p_brier = _grade_answer(answers[qid], ex.gold[qid])
            n_ok += int(ok)
            # base-rate prob items are calibration probes, not scored accuracy
            if ex.gold[qid].kind == "prob" and ex.meta.get("subkind") != "parity":
                pass
            else:
                fam_p.append(p_top)
                fam_y.append(1.0 if ok else 0.0)
                all_p.append(p_top)
                all_y.append(1.0 if ok else 0.0)
            # negation pairing: the boolq side keys by its own uid, the negated
            # side keys by the uid it was derived from
            if fam == "negation":
                neg_pairs.setdefault(ex.meta["pair_uid"], {})["neg"] = answers[qid]["noul"]
            elif fam == "boolq":
                neg_pairs.setdefault(ex.uid, {})["pos"] = answers[qid]["noul"]
            if permutations > 1 and ex.questions[qid]["type"] == "choice":
                runs, _ = _perm_probe(call, ex, k=permutations)
                if len(runs) >= 2:
                    import numpy as np

                    shifts = [permutation_robustness(runs[0], r) for r in runs[1:]]
                    perm_stats.append(
                        {
                            "max_shift": float(max(shifts)),
                            "flip": float(
                                argmax_flip_rate(np.array([runs[0]]), np.array(runs[1:]))
                            )
                            if len(runs) > 1
                            else 0.0,
                        }
                    )
        lat += fam_lat
        per_family[fam] = {
            "n": len(rows),
            "accuracy": n_ok / len(rows) if rows else None,
            "ece": ece(fam_p, fam_y) if fam_p else None,
            "brier": brier(fam_p, fam_y) if fam_p else None,
            "coverage": {
                str(th): {
                    "coverage": c,
                    "accuracy": a,
                }
                for th, (c, a, _n) in [
                    (th, coverage_accuracy(fam_p, fam_y, th))
                    for th in (0.5, 0.7, 0.8, 0.9, 0.95)
                ]
            },
            "p50_ms": statistics.median(fam_lat) if fam_lat else None,
        }

    # negation violation across paired boolq items
    viol = [abs(v["neg"] + v["pos"] - 1.0) for v in neg_pairs.values() if "neg" in v and "pos" in v]

    macro = {
        "accuracy": (
            statistics.mean([f["accuracy"] for f in per_family.values() if f["accuracy"] is not None])
            if any(f["accuracy"] is not None for f in per_family.values())
            else None
        ),
        "ece": ece(all_p, all_y) if all_p else None,
        "brier": brier(all_p, all_y) if all_p else None,
        "p50_ms": statistics.median(lat) if lat else None,
        "perm_mean_max_shift": statistics.mean([p["max_shift"] for p in perm_stats]) if perm_stats else None,
        "perm_flip_rate": statistics.mean([p["flip"] for p in perm_stats]) if perm_stats else None,
        "negation_mean_violation": statistics.mean(viol) if viol else None,
    }
    raw_f.close()
    return {"model": model, "device_note": "see CLI", "families": per_family, "macro": macro}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true", help="materialise frozen holdout and exit")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--model", default="laya")
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--permutations", type=int, default=1)
    ap.add_argument("--endpoint", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--raw", default=None, help="incremental raw JSONL (resume source)")
    ap.add_argument("--fresh", action="store_true", help="ignore/delete any raw resume log")
    args = ap.parse_args()

    fams = [f for f in args.families.split(",") if f]
    if args.build:
        counts = build_holdout(args.limit, fams)
        print(json.dumps(counts, indent=1))
        return
    if args.fresh and args.raw:
        Path(args.raw).unlink(missing_ok=True)
    res = run_eval(
        args.model, fams, args.limit, args.permutations, args.endpoint, args.ckpt,
        raw_path=Path(args.raw) if args.raw else None, fresh=args.fresh,
    )
    RESULTS.mkdir(exist_ok=True)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps(res["macro"], indent=1))
    for f, v in res["families"].items():
        print(f"  {f:12} acc={v['accuracy']} ece={v['ece']} p50={v['p50_ms']}")


if __name__ == "__main__":
    main()
