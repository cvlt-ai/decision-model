"""Holdout invariants. Run against ADAPTERS with small limits (no GPU, no full download
beyond the HF cache). These pin the contract every scored number depends on."""

import pytest

from eval.datasets import all_adapters
from eval.datasets.base import EvalExample

all_adapters()
from eval import datasets as dsmod  # noqa: E402

FAMS = ["boolq", "banking77", "mmlu_pro", "injection", "go_emotions", "pubhealth", "negation", "baserate"]


@pytest.fixture(scope="module")
def small():
    return {f: dsmod.REGISTRY[f](25) for f in FAMS}


def test_all_families_registered(small):
    assert set(small) == set(FAMS)


def test_rows_are_valid_jev_requests(small):
    for fam, rows in small.items():
        assert rows, f"{fam} produced zero rows"
        for r in rows:
            r.validate()  # raises ParseError if not a legal Jev request


def test_uids_unique_per_family(small):
    for fam, rows in small.items():
        uids = [r.uid for r in rows]
        assert len(uids) == len(set(uids)), f"{fam}: duplicate uids"


def test_gold_always_among_options(small):
    for fam, rows in small.items():
        for r in rows:
            for qid, gold in r.gold.items():
                q = r.questions[qid]
                if gold.kind == "prob":
                    assert 0.0 <= float(gold.value) <= 1.0
                elif q["type"] == "choice":
                    assert gold.value in q["criteria"], f"{fam}/{r.uid}"
                elif q["type"] == "noul":
                    assert gold.value in ("true", "false"), f"{fam}/{r.uid}"


def test_json_roundtrip_lossless(small):
    for fam, rows in small.items():
        for r in rows:
            back = EvalExample.from_dict(r.to_dict())
            assert back.state == r.state
            assert back.questions == r.questions
            assert set(back.gold) == set(r.gold)
            assert back.gold[next(iter(back.gold))].value == r.gold[next(iter(r.gold))].value
            assert back.meta == r.meta


def test_negation_pairs_point_at_boolq_items(small):
    boolq_uids = {r.uid for r in small["boolq"]}
    for r in small["negation"]:
        assert r.meta["pair_uid"] in boolq_uids
        assert r.family == "negation"


def test_negation_gold_is_flipped(small):
    bq = {r.uid: r for r in small["boolq"]}
    for r in small["negation"]:
        orig = bq[r.meta["pair_uid"]]
        assert r.gold["a"].value != orig.gold["a"].value


def test_baserate_interleaves_subkinds_at_small_limit():
    rows = dsmod.REGISTRY["baserate"](6)
    subs = [r.meta["subkind"] for r in rows]
    assert "coin" in subs and "parity" in subs, subs  # the whole point of interleaving
    full = dsmod.REGISTRY["baserate"](None)
    coins = [r for r in full if r.meta["subkind"] == "coin"]
    assert all(r.gold["q"].kind == "prob" and abs(r.gold["q"].value - 0.5) < 1e-9 for r in coins)


def test_baserate_negated_phasings_have_flipped_gold():
    full = dsmod.REGISTRY["baserate"](None)
    for r in full:
        if r.meta.get("phrasing") == "negated":
            n = int(r.uid.split("/")[2])
            assert r.gold["q"].value == ("false" if n % 2 else "true")


def test_adapter_is_deterministic(small):
    again = {f: dsmod.REGISTRY[f](25) for f in FAMS}
    for fam in FAMS:
        assert [r.uid for r in small[fam]] == [r.uid for r in again[fam]]
        assert [r.state for r in small[fam]] == [r.state for r in again[fam]]
