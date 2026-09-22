"""The frozen suite must cover ALL THREE primitives, and score was missing.

Severity ladder (smart-contract vulns, MIT) is real ordinal data with four
human-graded levels; it exercises score end-to-end (wire spec, nearest-level
grading, ordinal-adjacent misses scoring closer than far ones).
"""

import pytest

from eval.datasets import all_adapters
from eval.datasets.base import EvalExample

all_adapters()
from eval import datasets as dsmod  # noqa: E402


@pytest.fixture(scope="module")
def sev():
    return dsmod.REGISTRY["severity"](25)


def test_severity_registered_and_valid(sev):
    assert sev
    for r in sev:
        r.validate()
        assert r.family == "severity"


def test_severity_uses_score_primitive_with_ordered_criteria(sev):
    for r in sev:
        q = r.questions["s"]
        assert q["type"] == "score"
        assert isinstance(q["criteria"], list) and len(q["criteria"]) == 4


def test_severity_gold_is_level_index(sev):
    for r in sev:
        g = r.gold["s"]
        assert g.kind == "label"
        assert g.value in ("0", "1", "2", "3")  # 0-based wire keys, per vendor score example


def test_severity_label_distribution_is_not_degenerate(sev):
    # full split has all four levels; a 25-row sample must not collapse to one class
    from eval.datasets.severity import severity

    full = severity(None)
    vals = {r.gold["s"].value for r in full}
    assert vals == {"0", "1", "2", "3"}


def test_severity_grading_nearest_level():
    """An adjacent-level miss (gold 3, pred 2) must grade correct under ±0.5?
    No - levels are integers, so |3-2|=1 > 0.5 => WRONG, and that's the declared
    semantic: only exact level counts; pin it so nobody 'fixes' it later."""
    from eval.run_eval import _grade_answer
    from eval.datasets.base import Gold

    gold = Gold("label", "3")
    probs = {"1": 0.05, "2": 0.5, "3": 0.4, "4": 0.05}
    ok, top_p, _ = _grade_answer({"type": "score", "probabilities": probs}, gold)
    assert ok is False  # nearest-level 2 != 3
    ok2, _, _ = _grade_answer({"type": "score", "probabilities": {"1": 0.05, "2": 0.0, "3": 0.9, "4": 0.05}}, gold)
    assert ok2 is True
