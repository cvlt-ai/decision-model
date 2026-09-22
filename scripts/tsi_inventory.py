"""Stream TSI once and inventory the `task` column -> data/raw/tsi_task_counts.json.

The mixture (Phase 3) needs per-task row counts to sample a balanced subset without
downloading the whole corpus twice.
"""

import json
import time
from collections import Counter
from pathlib import Path

from datasets import load_dataset

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data" / "raw" / "tsi_task_counts.json"

t0 = time.perf_counter()
counts: Counter = Counter()
n = 0
for r in load_dataset("tasksource/tasksource-instruct-v0", split="train", streaming=True):
    counts[r.get("task", "unknown")] += 1
    n += 1
    if n % 250_000 == 0:
        print(f"{n} rows, {len(counts)} tasks, {time.perf_counter()-t0:.0f}s", flush=True)
OUT.write_text(json.dumps({"n_rows": n, "counts": dict(counts.most_common())}, indent=1))
print(f"DONE {n} rows, {len(counts)} tasks in {time.perf_counter()-t0:.0f}s -> {OUT}")
