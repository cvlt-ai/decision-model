#!/usr/bin/env python
"""Measure rendered-prompt token lengths on the JevBench public tiers, to size
how much the hard tier's 2-6k-token long policies are truncated at max_len=1024."""
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src"), str(ROOT / "third_party" / "jevbench")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import statistics
from jevbench.tasks import load_jsonl
from transformers import AutoTokenizer
from s1.mixture import render as R

tok = AutoTokenizer.from_pretrained("Qwen/Qwen3.5-4B", trust_remote_code=True)
PUB = ROOT / "third_party" / "jevbench" / "datasets" / "public"


def render_prompt(t):
    q = t.question
    st = t.state if isinstance(t.state, str) else __import__("json").dumps(t.state)
    crit = q.get("criteria")
    if q["type"] == "noul":
        crit = crit or {}
        opts = [("true", crit.get("true", "")), ("false", crit.get("false", ""))]
    elif q["type"] == "score":
        opts = [(str(i), crit[i]) for i in range(len(crit))]  # ordered levels
    else:  # choice: dict of label->description
        opts = [(k, crit.get(k, "")) for k in crit]
    return R(st, q["instructions"], opts)


for tier, fn in [("standard", "original.jsonl"), ("easy", "easy.jsonl"), ("hard", "hard.jsonl")]:
    ts = load_jsonl(str(PUB / fn))
    lens = [len(tok(render_prompt(t), add_special_tokens=False)["input_ids"]) for t in ts]
    over = sum(1 for x in lens if x > 1024)
    srt = sorted(lens)
    print(f"{tier:9} n={len(ts)} min={min(lens)} med={int(statistics.median(lens))} "
          f"p95={srt[int(0.95 * len(lens))]} max={max(lens)}  >1024: {over}/{len(ts)}")
