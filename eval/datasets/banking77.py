"""banking77: 77-way support-intent routing.

Note: PolyAI/banking77 ships a loading script, unsupported by datasets 4.x;
mteb/banking77 is the same CC-BY-4.0 data in parquet form (test n=3,076).
"""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .sample_util import get_split, sample

HF = "mteb/banking77"

INSTRUCTIONS = "Which single customer-support intent does this message express?"


@register
def banking77(limit: int | None = 500) -> list[EvalExample]:
    ds = get_split(HF, "test")
    # full 77-label set straight from the data, not hand-typed
    all_labels = sorted({str(r["label_text"]) for r in ds})
    if len(all_labels) != 77:
        raise ValueError(f"expected 77 intents, found {len(all_labels)}")
    criteria = {lab: lab.replace("_", " ") for lab in all_labels}
    rows = sample(ds, limit, "banking77")
    out = []
    for i, r in enumerate(rows):
        gold = str(r["label_text"])
        out.append(
            EvalExample(
                family="banking77",
                uid=f"banking77/{i}",
                state=str(r["text"]),
                questions={"intent": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
                gold={"intent": Gold("label", gold)},
                meta={"n_options": len(criteria)},
            )
        )
    for e in out:
        e.validate()
    return out
