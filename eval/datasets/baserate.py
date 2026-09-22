"""Self-authored base-rate / numeric probe family.

Both Jev (community reports) and our local Laya run (measured) fail simple
base-rate questions: die-roll parity answered 0.15, an unobserved coin flip
answered 0.16 when the honest answer is 0.5. TypeSafe's own jaggedness page
admits this class. We include it as a first-class eval family with
programmatically verifiable gold so the failure cannot be waved away as
"small sanity set".

Three sub-kinds, all with *stated* gold:
  parity   : "a fair die showed N; is it odd?"          gold = 1 or 0 (deterministic)
  coin     : "a fair coin was flipped, result unobserved" gold = 0.5 (base rate)
  card     : "one card drawn from a fair deck; is it a heart?" gold = 0.25
Plus distractor wording variants so we measure phrasing sensitivity separately.
"""

from __future__ import annotations

from .base import EvalExample, Gold, register

PARITY = [1, 2, 3, 4, 5, 6]


def _parity_rows() -> list[EvalExample]:
    rows = []
    for n in PARITY:
        gold = 1 if n % 2 else 0
        for phrasing, instr in [
            ("plain", f"Is the number that was rolled odd?"),
            ("restated", f"A six-sided die rolled {n}. Is {n} odd?"),
            ("negated", f"Is the number that was rolled NOT odd?"),
        ]:
            rows.append(
                EvalExample(
                    family="baserate",
                    uid=f"baserate/parity/{n}/{phrasing}",
                    state=f"A fair six-sided die was rolled and showed {n}.",
                    questions={"q": {"type": "noul", "instructions": instr}},
                    # negated phrasing inverts the gold
                    gold={"q": Gold("label", "true" if ((n % 2) == 1) != (phrasing == "negated") else "false")},
                    meta={"subkind": "parity", "deterministic": True, "phrasing": phrasing},
                )
            )
    return rows


def _coin_rows() -> list[EvalExample]:
    wordings = [
        "A fair coin was flipped once. The result was not observed. No other information exists.",
        "A coin is flipped. It is a standard fair coin. Nobody has looked at it yet.",
        "I flipped a fair coin behind my back and will not tell you the outcome.",
    ]
    instrs = [
        "Did the coin land heads up?",
        "Given only the supplied state, what is the probability that the coin landed heads up?",
    ]
    rows = []
    for i, st in enumerate(wordings):
        for j, ins in enumerate(instrs):
            rows.append(
                EvalExample(
                    family="baserate",
                    uid=f"baserate/coin/{i}/{j}",
                    state=st,
                    questions={"q": {"type": "noul", "instructions": ins}},
                    # the honest probability is exactly 0.5; scored on |p-0.5|, not argmax
                    gold={"q": Gold("prob", 0.5)},
                    meta={"subkind": "coin", "deterministic": False, "target": "0.5"},
                )
            )
    return rows


def _card_rows() -> list[EvalExample]:
    rows = []
    for i, (st, ins, p) in enumerate(
        [
            (
                "One card is drawn at random from a standard shuffled 52-card deck. The card has not been looked at.",
                "Is the card a heart?",
                0.25,
            ),
            (
                "One card is drawn at random from a standard shuffled 52-card deck. The card has not been looked at.",
                "Is the card an ace?",
                4 / 52,
            ),
            (
                "One card is drawn at random from a standard shuffled 52-card deck that has not been looked at.",
                "Is the card red?",
                0.5,
            ),
        ]
    ):
        rows.append(
            EvalExample(
                family="baserate",
                uid=f"baserate/card/{i}",
                state=st,
                questions={"q": {"type": "noul", "instructions": ins}},
                gold={"q": Gold("prob", p)},
                meta={"subkind": "card", "deterministic": False},
            )
        )
    return rows


@register
def baserate(limit: int | None = None) -> list[EvalExample]:
    # Interleave subkinds: with a small limit a plain concat returns only parity
    # rows, and the base-rate rows are the reason this family exists.
    groups = [_parity_rows(), _coin_rows(), _card_rows()]
    ordered = []
    for i in range(max(len(g) for g in groups)):
        for grp in groups:
            if i < len(grp):
                ordered.append(grp[i])
    return ordered[:limit] if limit else ordered
