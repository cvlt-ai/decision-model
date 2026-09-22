"""MMLU-Pro: the knowledge/reasoning axis. TIGER-Lab/MMLU-Pro test (12,032), MIT.

This is the family where clone readouts on 27-35B models score ~60% while Jev
claims 83% - the decisive column for whether M1 (encoder) or M2 (decoder) matters.
"""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .sample_util import get_split, sample

HF = "TIGER-Lab/MMLU-Pro"


@register
def mmlu_pro(limit: int | None = 400) -> list[EvalExample]:
    ds = get_split(HF, "test")
    rows = sample(ds, limit, "mmlu_pro")
    out = []
    for i, r in enumerate(rows):
        opts = list(r["options"])
        if not (2 <= len(opts) <= 10):
            continue  # Jev allows up to 255, but MMLU-Pro is 4-10; skip anomalies loudly
        criteria = {chr(65 + k): str(o) for k, o in enumerate(opts)}  # A..J
        gold = str(r["answer"]).strip()
        if gold not in criteria:
            continue
        out.append(
            EvalExample(
                family="mmlu_pro",
                uid=f"mmlu_pro/{r.get('question_id', i)}",
                state=str(r["question"]),
                questions={"a": {"type": "choice", "instructions": "Select the correct answer.", "criteria": criteria}},
                gold={"a": Gold("label", gold)},
                meta={"category": str(r.get("category", "")), "n_options": len(opts)},
            )
        )
    for e in out:
        e.validate()
    return out
