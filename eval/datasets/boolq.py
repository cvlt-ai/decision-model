"""boolq: reading-comprehension yes/no. google/boolq validation split (3,270)."""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .sample_util import get_split, sample

HF = "google/boolq"


@register
def boolq(limit: int | None = 500) -> list[EvalExample]:
    ds = get_split(HF, "validation")
    rows = sample(ds, limit, "boolq")
    out = []
    for i, r in enumerate(rows):
        out.append(
            EvalExample(
                family="boolq",
                uid=f"boolq/{r.get('idx', i)}",
                state={"passage": r["passage"], "question": r["question"]},
                questions={
                    "a": {
                        "type": "noul",
                        "instructions": (
                            "Based only on the passage in the state, is the answer to the "
                            f"question '{r['question']}' yes?"
                        ),
                        "criteria": {
                            "true": "the passage supports a yes answer",
                            "false": "the passage supports a no answer",
                        },
                    }
                },
                gold={"a": Gold("label", "true" if r["answer"] else "false")},
            )
        )
    for e in out:
        e.validate()
    return out
