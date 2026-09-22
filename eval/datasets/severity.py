"""severity: the `score` primitive. msc-smart-contract-auditing/vulnerability-severity-classification.

MIT-licensed, human-graded Solidity vulnerability severity, 4 levels. test n=437
(used whole), train n=2,473 for the mixture. Maps onto Jev's ordered-criteria score:
1=none 2=low 3=medium 4=high. Grading is nearest-level with ±0.5 tolerance (i.e.
exact level for integer golds) - declared in run_eval, pinned in test_score_family.
"""

from __future__ import annotations

from .base import EvalExample, Gold, register
from .sample_util import get_split

HF = "msc-smart-contract-auditing/vulnerability-severity-classification"

LEVELS = ["none", "low", "medium", "high"]  # criteria index 0..3 == score level 1..4

INSTRUCTIONS = (
    "Score the severity of the security vulnerability in the Solidity code in the state: "
    "0=no vulnerability, 1=low, 2=medium, 3=high (loss of funds or logic compromise is high)."
)


@register
def severity(limit: int | None = None) -> list[EvalExample]:
    ds = get_split(HF, "test")
    out = []
    for i, r in enumerate(ds):
        sev = str(r["severity"]).strip().lower()
        if sev not in LEVELS:
            continue
        # Jev score levels are 0-based wire keys (vendor example: probs {"0":..,"1":..,"2":..}
        # with legend criteria[0]="Calm" -> level "0"), pinned in tests/test_schema.py
        level = str(LEVELS.index(sev))
        out.append(
            EvalExample(
                family="severity",
                uid=f"severity/{i}",
                state=str(r["function"]),
                questions={
                    "s": {
                        "type": "score",
                        "instructions": INSTRUCTIONS,
                        "criteria": LEVELS,
                    }
                },
                gold={"s": Gold("label", level)},
                meta={"severity": sev},
            )
        )
    for e in out:
        e.validate()
    return out[:limit] if limit else out
