"""PubHealthBench: professional-guidance multiple choice (replaces pubhealth, whose
loading script is unsupported by datasets 4.x). Joshua-Harris/PubHealthBench, CC-BY-4.0.
test n=7,929. Gold is a letter A..; options are free text."""

from __future__ import annotations

import ast

from .base import EvalExample, Gold, register
from .sample_util import get_split, sample

HF = "Joshua-Harris/PubHealthBench"

INSTRUCTIONS = (
    "According to the public-health guidance implied by the question, select the correct answer."
)


def _as_list(v):
    if isinstance(v, list):
        return v
    if isinstance(v, str):
        try:
            parsed = ast.literal_eval(v)
            if isinstance(parsed, list):
                return parsed
        except (ValueError, SyntaxError):
            pass
    raise ValueError(f"unparseable options field: {str(v)[:80]}")


@register
def pubhealth(limit: int | None = 300) -> list[EvalExample]:
    ds = get_split(HF, "test")
    rows = sample(ds, limit, "pubhealth")
    out = []
    for i, r in enumerate(rows):
        try:
            opts = _as_list(r["options"])
        except ValueError:
            continue
        if not (2 <= len(opts) <= 10):
            continue
        criteria = {chr(65 + k): str(o) for k, o in enumerate(opts)}
        gold = str(r["answer"]).strip()
        if gold not in criteria:
            continue
        out.append(
            EvalExample(
                family="pubhealth",
                uid=f"pubhealth/{r.get('question_id', i)}",
                state=str(r["question"]),
                questions={"a": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
                gold={"a": Gold("label", gold)},
                meta={"category": str(r.get("category", "")), "n_options": len(opts)},
            )
        )
    for e in out:
        e.validate()
    return out
