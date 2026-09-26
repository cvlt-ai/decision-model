"""vitaminc: the human-labeled contrastive axis.

VitaminC dev (tals/vitaminc `validation` split == the published 63,054-row dev set).
Contrastive by construction: one case_id groups the siblings of a single Wikipedia
revision - near-identical evidence, different labels. 99% of case_ids carry >1 distinct
label, so it directly tests EVIDENCE-SENSITIVITY (key on the specific fact, not the
topic) - the human-labeled analogue of the c2d contrastive pairs in the mixture.

3-way choice (SUPPORTS / REFUTES / NOT ENOUGH INFO), CC BY-SA 3.0. This is the
external, human-graded check for whether the c2d training data moves evidence
sensitivity. It is an ADDITIVE axis (like `negation`): NOT in the 9-family `FAMILIES`
macro, so every prior v1-v4 frozen number stays directly comparable; score it by name.
"""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .sample_util import get_split

HF = "tals/vitaminc"
# `validation` == the published VitaminC dev split (63,054 rows); the dataset card's
# "dev" is exposed by the parquet loader as `validation`. Pinned here on purpose.
SPLIT = "validation"

# criteria written once, not tuned against any split (same convention as the rest of
# the suite: the wording is a free parameter, both models see identical text).
CRITERIA = {
    "SUPPORTS": "The evidence states the claim, or the claim follows directly from the evidence.",
    "REFUTES": "The evidence states the opposite of the claim, or the claim is directly contradicted by it.",
    "NOT ENOUGH INFO": "The evidence neither establishes nor contradicts the claim; settling it would need a fact the evidence does not supply.",
}
INSTRUCTIONS = (
    "Decide how the evidence bears on the claim. Judge only from the evidence text, "
    "and not from outside knowledge about the subject."
)


@register
def vitaminc(limit: int | None = 300) -> list[EvalExample]:
    ds = get_split(HF, SPLIT)
    rows = sample_rows(ds, limit)
    out = []
    for r in rows:
        label = str(r["label"]).strip()
        if label not in CRITERIA:
            continue  # unexpected label - skip loudly rather than mislabel
        out.append(
            EvalExample(
                family="vitaminc",
                uid=f"vitaminc/{r['unique_id']}",
                state={"evidence": r["evidence"], "claim": r["claim"]},
                questions={
                    "a": {
                        "type": "choice",
                        "instructions": INSTRUCTIONS,
                        "criteria": CRITERIA,
                    }
                },
                gold={"a": Gold("label", label)},
                meta={
                    "case_id": str(r["case_id"]),
                    "revision_type": str(r["revision_type"]),
                    "page": str(r.get("page", "")),
                },
            )
        )
    for e in out:
        e.validate()
    return out


def sample_rows(ds, n: int | None):
    # Reuse the shared deterministic shuffle-then-take (pinned seed, no train split).
    from .sample_util import sample
    return sample(ds, n, "vitaminc")
