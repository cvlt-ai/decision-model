"""Negation-consistency family: every boolq holdout item re-asked negated.

Both Jev (vendor-documented: P(refund)=0.72, P(not refund)=0.47 -> 1.19) and our
local Laya run (measured: sum 0.345) violate P(x) + P(not x) = 1. Scoring pairs
items by meta.pair_uid and reports the mean |sum - 1| violation.
"""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .boolq import boolq as _boolq


@register
def negation(limit: int | None = None) -> list[EvalExample]:
    out = []
    for ex in _boolq(limit):
        q_text = ex.state["question"]
        out.append(
            EvalExample(
                family="negation",
                uid=f"negation/{ex.uid}",
                state=ex.state,
                questions={
                    "a": {
                        "type": "noul",
                        "instructions": (
                            "Based only on the passage in the state, is it NOT the case that "
                            f"the answer to the question '{q_text}' is yes?"
                        ),
                        "criteria": {
                            "true": "the passage supports a no answer to the original question",
                            "false": "the passage supports a yes answer to the original question",
                        },
                    }
                },
                # flip the original gold
                gold={"a": Gold("label", "false" if ex.gold["a"].value == "true" else "true")},
                meta={"pair_uid": ex.uid},
            )
        )
    for e in out:
        e.validate()
    return out
