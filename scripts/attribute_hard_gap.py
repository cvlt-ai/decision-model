#!/usr/bin/env python
"""Attribute the JevBench hard-tier gap: how many of the 41 truncated-at-1024
items did we get vs the 70 that fit? Isolates 'truncation' from 'genuinely hard'."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "third_party" / "jevbench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import json
from jevbench.tasks import load_jsonl
from transformers import AutoTokenizer
from s1.mixture import render as R

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
PUB = ROOT / "third_party" / "jevbench" / "datasets" / "public"
raw = (ROOT / "results" / "jevbench_s1v3_public231.json").read_text()
dec = json.JSONDecoder()
summary, idx = dec.raw_decode(raw)
records = json.loads(raw[idx:].strip())
byid = {r["id"]: r for r in records}


def prompt_len(t):
    q = t.question
    st = t.state if isinstance(t.state, str) else json.dumps(t.state)
    crit = q.get("criteria")
    if q["type"] == "noul":
        crit = crit or {}
        opts = [("true", crit.get("true", "")), ("false", crit.get("false", ""))]
    elif q["type"] == "score":
        opts = [(str(i), crit[i]) for i in range(len(crit))]
    else:
        opts = [(k, crit.get(k, "")) for k in crit]
    return len(tok(R(st, q["instructions"], opts), add_special_tokens=False)["input_ids"])


for tier, fn in [("standard", "original.jsonl"), ("easy", "easy.jsonl"), ("hard", "hard.jsonl")]:
    tasks = load_jsonl(str(PUB / fn))
    trunc = fit = ok_trunc = ok_fit = 0
    for t in tasks:
        L = prompt_len(t)
        correct = bool(byid.get(t.id, {}).get("correct"))
        if L > 1024:
            trunc += 1
            ok_trunc += int(correct)
        else:
            fit += 1
            ok_fit += int(correct)
    print(f"{tier:9} total={len(tasks)}  truncated(>1024)={trunc}  fits={fit}")
    if trunc:
        print(f"          truncated: {ok_trunc}/{trunc} = {ok_trunc/trunc:.3f}")
    print(f"          fits:      {ok_fit}/{fit} = {ok_fit/fit:.3f}" if fit else "          fits:      0/0")
    # hypothetical: truncated items answered at 'fits' accuracy
    if trunc and fit:
        hypo = (ok_fit + round(ok_fit / fit * trunc)) / len(tasks)
        print(f"          if truncated items scored like fits: ~{hypo:.3f} (vs actual {summary['tiers'][tier]['accuracy']:.3f})")
