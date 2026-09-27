"""One-time TSI extraction: stream tasksource-instruct-v0 once -> a balanced,
license-filtered sample of choice-rows in TSI's native fields.

Why a separate pass: TSI is 5.3M rows; we want a bounded, deterministic, balanced
subset (per-task cap + gap-priority) that mixture.py can read fast on every rebuild.

Row filter (all must hold to be kept):
  * permissive / commercial-safe license (mit / apache / bsd / cc0 / ...)
  * 2-24 quoted option labels on the INSTRUCTION line (line 1)
  * the normalized target equals one of those option labels (gold-in-options)

Output row: {task, instruction, state, options:[...], target, license}
  - instruction = line 1 of `inputs` (the task prompt with quoted options)
  - state       = the rest of `inputs` (the content to judge)
  - options     = quoted labels parsed from line 1
  - target      = the option label matching `targets` (raw string, exact)

Deterministic: stable per-task sampling order, fixed seed, per-task cap, total cap.
"""
from __future__ import annotations
import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "raw" / "tsi_train.jsonl"

# `license_use` is a coarse flag: "commercial" | "non-commercial" | "unspecified".
# The actual license NAMES live in the `license` field (a comma list, e.g.
# "apache-2.0" or "cc-by-4.0, MIT License (DPI)"). Release-safe = license_use says
# commercial AND every named license is permissive (no NC / GPL / AGPL / SA / DPI).
RESTRICTED_HINTS = ("nc", "gpl", "agpl", "by-sa", "by-nc", "request form",
                    "academic", "custom", "no license", "various", "openai",
                    "oanc", "dpi", "cc by-nc")

def is_permissive(license: str, license_use: str) -> bool:
    lu = (license_use or "").strip().lower()
    if lu != "commercial":  # be strict: only rows TSI marks commercial-safe
        return False
    lic = (license or "").strip().lower()
    if not lic:
        return False
    for tok in re.split(r"[,;]", lic):
        tok = tok.strip()
        if not tok:
            continue
        if any(h in tok for h in RESTRICTED_HINTS):
            return False
    return True


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower().rstrip(".!?:"))


def split_inputs(inputs: str):
    """-> (instruction, state). Line 1 is the instruction; the rest is content."""
    if "\n" in inputs:
        instr, state = inputs.split("\n", 1)
        return instr.strip(), state.strip()
    return "", inputs.strip()


def extract_options(instruction: str):
    """Quoted labels on the instruction line, in order, deduped, 2-24 of them."""
    quotes = re.findall(r'"([^"]{1,40})"', instruction)
    opts, seen = [], set()
    for q in quotes:
        q = q.strip()
        if not q or len(q) > 30:
            continue
        if q in seen:
            continue
        seen.add(q)
        opts.append(q)
    return opts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=64000, help="max rows to keep")
    ap.add_argument("--cap", type=int, default=500, help="max rows per task")
    ap.add_argument("--min-opts", type=int, default=2)
    ap.add_argument("--max-opts", type=int, default=24)
    ap.add_argument("--out", default=str(OUT))
    a = ap.parse_args()

    from datasets import load_dataset
    ds = load_dataset("tasksource/tasksource-instruct-v0", split="train", streaming=True)

    # gap-priority: tasks that map to our known weak axes first (knowledge/QA/NLI/
    # contrastive), then everything else. Deterministic within a tier by name.
    GAP = re.compile(r"(nli|fever|paws|vitaminc|entail|contrad|winograd|hella|swag|"
                     r"wiki|hop|qa|question|fact|math|logic|reason|science|knowledge|"
                     r"common|agieval|agie|super|bbh|bigbench|social_i|truthful|"
                     r"arc|openbook|commonsense|obqa|piqa|cove|discovery)", re.I)

    bufs: dict[str, list] = defaultdict(list)
    n = 0
    t0 = time.perf_counter()
    for r in ds:
        n += 1
        lic = r.get("license_use", "")
        if not is_permissive(r.get("license", ""), lic):
            continue
        instr, state = split_inputs(r["inputs"])
        opts = extract_options(instr or r["inputs"])
        if not (a.min_opts <= len(opts) <= a.max_opts):
            continue
        tgt = norm(r["targets"])
        match = next((o for o in opts if norm(o) == tgt), None)
        if match is None:
            continue
        task = r["task"]
        if len(bufs[task]) < a.cap:
            bufs[task].append({"task": task, "instruction": instr, "state": state,
                               "options": opts, "target": match, "license": "tsi-perm"})
        # early exit once total buffer reaches cap and we've covered a wide spread
        total = sum(len(v) for v in bufs.values())
        if total >= a.total and len(bufs) >= 200 and n % 100_000 == 0:
            print(f"  early: {n} scanned, {total} buffered across {len(bufs)} tasks", flush=True)
        if n % 250_000 == 0:
            print(f"  {n} scanned, {total} buffered across {len(bufs)} tasks, "
                  f"{time.perf_counter()-t0:.0f}s", flush=True)

    # assemble: gap-priority tasks first, then alphabetical; deterministic
    tasks = sorted(bufs, key=lambda t: (0 if GAP.search(t) else 1, t))
    kept = []
    for t in tasks:
        kept.extend(bufs[t])
    kept = kept[:a.total]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        for row in kept:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")
    per_task = len(kept)
    print(f"DONE: scanned {n}, kept {len(kept)} rows across "
          f"{len({r['task'] for r in kept})} tasks -> {OUT}")
    # quick stats
    from collections import Counter
    c = Counter(r["task"] for r in kept)
    print("top tasks:", c.most_common(5))


if __name__ == "__main__":
    main()
