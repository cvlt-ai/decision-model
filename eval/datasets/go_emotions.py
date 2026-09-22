"""go_emotions: 28-way fine-grained affect. google-research-datasets/go_emotions test (5,427).

Multi-label in the source; `choice` is pick-one, so gold is the FIRST listed label
(the dataset's own convention for single-label eval). The multi-label rate is
recorded in meta so scoring can report how much ambiguity we collapsed.
Label names verified from the loaded dataset's ClassLabel feature, not typed by hand.
"""

from __future__ import annotations

import ast

from .base import EvalExample, Gold, register
from .sample_util import get_split, sample

HF = "google-research-datasets/go_emotions"

LABELS = [
    "admiration", "amusement", "anger", "annoyance", "approval", "caring",
    "confusion", "curiosity", "desire", "disappointment", "disapproval",
    "disgust", "embarrassment", "excitement", "fear", "gratitude", "grief",
    "joy", "love", "nervousness", "optimism", "pride", "realization",
    "relief", "remorse", "sadness", "surprise", "neutral",
]

INSTRUCTIONS = "Which single emotion best describes the dominant feeling of this Reddit comment?"


@register
def go_emotions(limit: int | None = 500) -> list[EvalExample]:
    ds = get_split(HF, "test")
    # sanity: the ClassLabel names must match our list, or the mapping is silently wrong
    names = ds.features["labels"].feature.names
    if sorted(names) != sorted(LABELS):
        raise ValueError("go_emotions label names changed upstream; update LABELS from the data")
    criteria = {lab: lab for lab in LABELS}
    rows = sample(ds, limit, "go_emotions")
    out = []
    for i, r in enumerate(rows):
        labs = r["labels"]
        if isinstance(labs, str):  # some snapshots store the list as its repr
            labs = ast.literal_eval(labs)
        labs = [int(x) for x in labs]
        if not labs:
            continue
        gold = LABELS[int(labs[0])]
        out.append(
            EvalExample(
                family="go_emotions",
                uid=f"go_emotions/{r.get('id', i)}",
                state=str(r["text"]),
                questions={"e": {"type": "choice", "instructions": INSTRUCTIONS, "criteria": criteria}},
                gold={"e": Gold("label", gold)},
                meta={"multi_label": len(labs) > 1, "n_labels": len(labs)},
            )
        )
    for e in out:
        e.validate()
    return out
