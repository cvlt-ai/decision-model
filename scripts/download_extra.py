"""Phase-3 training breadth: download the verified permissive corpora.

Row caps keep the first pull bounded (disk is not the constraint; iteration speed is).
Licenses recorded per-job; anything not ✓ in research/03-dataset-catalog.md lands under
data/raw/quarantine/ so a permissive release can never pick it up by accident.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from datasets import Dataset, load_dataset, load_dataset_builder
from huggingface_hub import HfApi

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw"
QUAR = RAW / "quarantine"

# (name, hf id, config, cap, license_class)  license_class: ok | quar
JOBS = [
    # multi_nli repo carries FOUR license tags (cc-by-3.0, cc-by-sa-3.0, mit, other);
    # the original MultiNLI is known-NC-ish, so quarantine until proven, despite
    # earlier catalog note "use the mit-tagged subset" - there is no per-row provenance.
    ("multi_nli", "nyu-mll/multi_nli", "default", 200_000, "quar"),
    ("gsm8k", "openai/gsm8k", "main", 8_000, "ok"),
    ("clinc_oos", "clinc/clinc_oos", "plus", 22_500, "ok"),
    ("amazon_polarity", "fancyzhx/amazon_polarity", None, 200_000, "ok"),
    ("ai2_arc", "allenai/ai2_arc", "ARC-Challenge", 2_500, "quar"),  # cc-by-sa
    ("hellaswag", "rowan/hellaswag", None, 40_000, "quar"),          # license unstated
    ("snli", "stanfordnlp/snli", None, 200_000, "quar"),             # cc-by-sa
]


def _lic(hf: str) -> str:
    try:
        d = HfApi().dataset_info(hf, files_metadata=False)
        tags = [t for t in (d.tags or []) if t.startswith("license:")]
        return ",".join(t.split(":", 1)[1] for t in tags) or "unstated"
    except Exception as e:
        return f"lookup-failed:{type(e).__name__}"


def main() -> None:
    QUAR.mkdir(parents=True, exist_ok=True)
    report: dict = {"started": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "jobs": []}

    # record TSI config inventory without downloading it (485 tasks - pick later, deliberately)
    try:
        b = load_dataset_builder("tasksource/tasksource-instruct-v0")
        cfgs = sorted(b.builder_configs.keys())
        (RAW / "tsi_configs.json").write_text(json.dumps(cfgs, indent=1))
        report["tsi_configs"] = len(cfgs)
        print(f"TSI configs: {len(cfgs)} -> data/raw/tsi_configs.json")
    except Exception as e:
        report["tsi_configs_error"] = f"{type(e).__name__}: {str(e)[:200]}"
        print("TSI config listing failed:", report["tsi_configs_error"])

    for name, hf, cfg, cap, lic_class in JOBS:
        entry: dict = {"name": name, "hf": hf, "config": cfg, "cap": cap, "license_class": lic_class}
        t0 = time.perf_counter()
        try:
            rows = []
            src = load_dataset(hf, cfg, split="train", streaming=True) if cfg else load_dataset(hf, split="train", streaming=True)
            for i, r in enumerate(src):
                if i >= cap:
                    break
                rows.append(r)
            ds = Dataset.from_list(rows)
            out = (QUAR if lic_class == "quar" else RAW) / f"{name}__train"
            ds.save_to_disk(str(out))
            entry.update({"ok": True, "n": len(ds), "path": str(out.relative_to(ROOT)),
                          "license": _lic(hf), "seconds": round(time.perf_counter() - t0, 1)})
        except Exception as e:
            entry.update({"ok": False, "error": f"{type(e).__name__}: {str(e)[:200]}"})
        report["jobs"].append(entry)
        print(json.dumps(entry), flush=True)

    (RAW / "EXTRA_DOWNLOAD.json").write_text(json.dumps(report, indent=1))
    ok = [j for j in report["jobs"] if j.get("ok")]
    print(f"DONE ok={len(ok)}/{len(JOBS)} rows={sum(j['n'] for j in ok)}")


if __name__ == "__main__":
    main()
