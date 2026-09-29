"""serve_s1: Jev-wire -> readout question translation + request round-trip.

The server's only non-trivial logic is _internal_question (schema.Question ->
the readout's {type, instructions, criteria} dict) and the wire round-trip.
Cover the three question types + the criteria-shape variants the readout expects.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from s1.schema import parse_request  # noqa: E402
from scripts.serve_s1 import _internal_question  # noqa: E402


def _q(type_, criteria, instructions="Q"):
    req = parse_request({"state": "s", "questions": {"q": {"type": type_,
                                                           "instructions": instructions,
                                                           "criteria": criteria}}})
    return req.questions["q"]


def test_choice_translation():
    q = _q("choice", {"a": "desc a", "b": "desc b"})
    out = _internal_question(q)
    assert out["type"] == "choice"
    assert out["criteria"] == {"a": "desc a", "b": "desc b"}
    assert out["instructions"] == "Q"


def test_choice_nested_description_dict():
    # criteria values may be {description: ...} objects; flatten to the string
    q = _q("choice", {"a": {"description": "d a"}, "b": "plain b"})
    out = _internal_question(q)
    assert out["criteria"]["a"] == "d a"
    assert out["criteria"]["b"] == "plain b"


def test_score_translation_ordered():
    q = _q("score", ["none", "some", "severe"])
    out = _internal_question(q)
    assert out["type"] == "score"
    # readout score branch indexes crit[i] with INT i, so keys are ints 0..n
    assert list(out["criteria"].keys()) == [0, 1, 2]
    assert out["criteria"][2] == "severe"


def test_noul_translation_default():
    q = _q("noul", None)
    out = _internal_question(q)
    assert out["type"] == "noul"
    assert out["criteria"] == {"true": "", "false": ""}


def test_wire_roundtrip_preserves_questions():
    body = {"state": "state text", "questions": {
        "intent": {"type": "choice", "instructions": "classify",
                   "criteria": {"x": "x d", "y": "y d"}}}}
    req = parse_request(body)
    assert set(req.questions) == {"intent"}
    assert _internal_question(req.questions["intent"])["criteria"] == {"x": "x d", "y": "y d"}
