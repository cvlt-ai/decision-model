"""vitaminc adapter: the human-labeled contrastive axis.

Pins the contract every vitaminc number depends on: it's a legal 3-way Jev choice,
gold is always one of the three criteria keys, families (case_id) are preserved in
meta for sibling-level analysis, and sampling is deterministic under the pinned seed.
It is an ADDITIVE axis - deliberately NOT in the 9-family `FAMILIES` macro, so the
prior frozen numbers stay comparable; here we only assert registration + shape.
"""
import pytest

from eval.datasets import all_adapters
from eval.datasets.base import EvalExample

all_adapters()
from eval import datasets as dsmod  # noqa: E402


@pytest.fixture(scope="module")
def vc():
    return dsmod.REGISTRY["vitaminc"](25)


def test_registered_and_is_additive_axis(vc):
    # present in the registry...
    assert "vitaminc" in dsmod.REGISTRY
    # ...but deliberately NOT in the 9-family macro (prior numbers stay comparable)
    from eval.run_eval import FAMILIES
    assert "vitaminc" not in FAMILIES


def test_rows_valid_3way_choice(vc):
    assert vc, "vitaminc produced zero rows"
    for r in vc:
        r.validate()
        assert r.family == "vitaminc"
        q = r.questions["a"]
        assert q["type"] == "choice"
        assert set(q["criteria"]) == {"SUPPORTS", "REFUTES", "NOT ENOUGH INFO"}
        assert r.state.get("evidence") and r.state.get("claim")


def test_gold_always_among_options(vc):
    for r in vc:
        assert r.gold["a"].value in r.questions["a"]["criteria"]


def test_family_preserved_in_meta(vc):
    for r in vc:
        assert r.meta.get("case_id"), "contrastive family (case_id) must be preserved"
        assert r.meta.get("revision_type")


def test_label_not_degenerate(vc):
    vals = {r.gold["a"].value for r in vc}
    # a 25-row sample may miss one class, but must not collapse to a single label
    assert len(vals) >= 2, f"vitaminc sample collapsed to {vals}"


def test_deterministic_under_pinned_seed():
    again = dsmod.REGISTRY["vitaminc"](25)
    a = dsmod.REGISTRY["vitaminc"](25)
    assert [r.uid for r in a] == [r.uid for r in again]
    assert [r.state for r in a] == [r.state for r in again]
    assert [r.gold["a"].value for r in a] == [r.gold["a"].value for r in again]
