"""Download TRAIN splits for the eval families into data/raw (durable local copies).

The frozen holdout under data/holdout is eval-only by contract; training data must
never come from it. These train splits are the raw material for the Phase 3 mixture.
Decontamination against the holdout happens later (scripts/decontaminate.py), NOT here.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from datasets import load_dataset
from huggingface_hub import dataset_info

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"

# (local name, hf id, config, train split)
TRAIN_JOBS = [
    ("boolq", "google/boolq", None, "train"),
    ("banking77", "mteb/banking77", None, "train"),
    # MMLU-Pro has no train split (test/validation only, by design) -> eval-only axis;
    # knowledge training must come from TSI/other corpora, never from here.
    ("injection", "deepset/prompt-injections", None, "train"),
    ("go_emotions", "google-research-datasets/go_emotions", None, "train"),
    # validation (not train) - repo has validation/test/reviewed; decontamination
    # against the frozen test holdout is mandatory before this may train on it.
    ("pubhealth_val", "Joshua-Harris/PubHealthBench", None, "validation"),
    ("severity", "msc-smart-contract-auditing/vulnerability-severity-classification", None, "train"),
]


def main() -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    report = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": []}
    for name, hf, cfg, split in TRAIN_JOBS:
        entry: dict = {"name": name, "hf": hf, "split": split}
        t0 = time.perf_counter()
        try:
            ds = load_dataset(hf, cfg, split=split) if cfg else load_dataset(hf, split=split)
            out = RAW / f"{name}__train"
            ds.save_to_disk(str(out))
            entry.update(
                {
                    "ok": True,
                    "n": len(ds),
                    "path": str(out.relative_to(ROOT)),
                    "cols": list(ds.column_names),
                    "seconds": round(time.perf_counter() - t0, 1),
                }
            )
            try:
                card = dataset_info(hf, files_metadata=False)
                entry["license"] = str(getattr(card, "license", None) or [
                    t for t in (card.tags or []) if t.startswith("license:")
                ][:1])
            except Exception as e:  # license lookup must never fail a download
                entry["license"] = f"lookup-failed: {type(e).__name__}"
        except Exception as e:
            entry.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        report["jobs"].append(entry)
        print(json.dumps(entry))
    (RAW / "TRAIN_DOWNLOAD.json").write_text(json.dumps(report, indent=1))
    ok = [j for j in report["jobs"] if j.get("ok")]
    print(f"DONE ok={len(ok)}/{len(TRAIN_JOBS)} rows={sum(j['n'] for j in ok)}")


if __name__ == "__main__":
    main()
