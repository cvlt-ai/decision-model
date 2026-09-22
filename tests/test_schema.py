"""Tests for the Jev wire-spec reconstruction.

Several of these tests are asserted against *published vendor numbers* (taken from
docs.typesafe.ai and the pydantic-ai TypeSafe adapter docs, archived under
research/notes/raw/). If a number here ever disagrees with the vendor, that is a
finding, not a bug.
"""
import pytest

from s1.schema import (
    AnswerError,
    DecideRequest,
    ParseError,
    build_answer,
    confidence_from_probs,
    parse_request,
    render_instructions,
)

# ---------------------------------------------------------------- parsing


def test_noul_roundtrip():
    req = parse_request(
        {
            "state": "billed twice",
            "questions": {"esc": {"type": "noul", "instructions": "Escalate now?"}},
        }
    )
    assert req.questions["esc"].qtype == "noul"
    out = req.render()
    assert out["questions"]["esc"]["type"] == "noul"


def test_state_accepts_str_dict_and_list():
    for s in ("text", {"a": 1}, ["a", 1]):
        req = parse_request(
            {"state": s, "questions": {"q": {"type": "noul", "instructions": "yes?"}}}
        )
        assert req.state == s


def test_choice_requires_criteria():
    with pytest.raises(ParseError):
        parse_request(
            {"state": "x", "questions": {"d": {"type": "choice", "instructions": "w?"}}}
        )


def test_choice_rejects_256_options():
    crit = {f"k{i}": f"desc {i}" for i in range(256)}
    with pytest.raises(ParseError):
        parse_request(
            {
                "state": "x",
                "questions": {
                    "d": {"type": "choice", "instructions": "w?", "criteria": crit}
                },
            }
        )


def test_choice_allows_null_criteria_value():
    """Vendor: 'use null when an option needs no extra detail'."""
    req = parse_request(
        {
            "state": "x",
            "questions": {
                "d": {
                    "type": "choice",
                    "instructions": "w?",
                    "criteria": {"a": None, "b": "described"},
                }
            },
        }
    )
    assert req.questions["d"].option_keys() == ["a", "b"]


@pytest.mark.parametrize("levels", [["only one"], [str(i) for i in range(11)]])
def test_score_needs_2_to_10_levels(levels):
    with pytest.raises(ParseError):
        parse_request(
            {
                "state": "x",
                "questions": {
                    "u": {"type": "score", "instructions": "w?", "criteria": levels}
                },
            }
        )


def test_unknown_type_rejected():
    with pytest.raises(ParseError):
        parse_request(
            {"state": "x", "questions": {"q": {"type": "regress", "instructions": "?"}}}
        )


def test_empty_questions_rejected():
    with pytest.raises(ParseError):
        parse_request({"state": "x", "questions": {}})


def test_missing_state_rejected():
    with pytest.raises(ParseError):
        parse_request({"questions": {"q": {"type": "noul", "instructions": "?"}}})


# ------------------------------------------------- question ids are not model input

def test_question_id_is_not_in_prompt():
    """Vendor: the question key 'is not sent to the underlying model and is not used
    in inference'. So a distinctive id must never leak into the rendered prompt."""
    req = parse_request(
        {
            "state": "s",
            "questions": {
                "ZZQQ_canary_id": {"type": "noul", "instructions": "Is it red?"}
            },
        }
    )
    assert "ZZQQ_canary_id" not in req.render_prompt("ZZQQ_canary_id")


# ------------------------------------------------------- structured instructions

def test_structured_instructions_backtick_reference():
    """Vendor: instructions may be an object holding data fields plus a `question`
    field that refers to the other fields by name in backticks."""
    req = parse_request(
        {
            "state": {"resume": "VP Eng at Acme"},
            "questions": {
                "dup": {
                    "type": "noul",
                    "instructions": {
                        "potential_duplicate": {
                            "name": "John Smith",
                            "location": "Oakland, California",
                        },
                        "question": "Is the resume the same person as `potential_duplicate`?",
                    },
                }
            },
        }
    )
    text = render_instructions(req.questions["dup"].instructions)
    # the question text is present, and the referenced data is expanded inline
    assert "Is the resume the same person as" in text
    assert "John Smith" in text and "Oakland, California" in text


# ---------------------------------------------------------------- option order

def test_render_prompt_honours_option_order():
    req = parse_request(
        {
            "state": "s",
            "questions": {
                "d": {
                    "type": "choice",
                    "instructions": "which?",
                    "criteria": {"a": "alpha thing", "b": "beta thing"},
                }
            },
        }
    )
    p = req.render_prompt("d", option_order=["b", "a"])
    assert p.index("beta thing") < p.index("alpha thing")
    assert p.index("[0]") < p.index("[1]")


def test_options_must_be_a_permutation():
    req = parse_request(
        {
            "state": "s",
            "questions": {
                "d": {
                    "type": "choice",
                    "instructions": "which?",
                    "criteria": {"a": "x", "b": "y"},
                }
            },
        }
    )
    with pytest.raises(ValueError):
        req.render_prompt("d", option_order=["a", "a"])


# --------------------------------------------------------- confidence (reverse-compiled)

# Four (probabilities -> reported confidence) pairs lifted verbatim from vendor
# documentation. Our reconstruction must reproduce all of them within the vendor's
# own rounding.
@pytest.mark.parametrize(
    "probs,reported",
    [
        # api ref: choice {billing .88, technical .12, sales .0} -> 0.81
        ({"billing": 0.88, "technical": 0.12, "sales": 0.0}, 0.81),
        # api ref: score {0:0, 1:.95, 2:.05} -> 0.92
        ({"0": 0.0, "1": 0.95, "2": 0.05}, 0.92),
        # cloudflare ref: {billing .87, sales 0, technical .13} -> 0.8
        ({"billing": 0.87, "sales": 0.0, "technical": 0.13}, 0.80),
        # jaggedness page: yes .01 / no .99 -> 0.97
        ({"yes": 0.01, "no": 0.99}, 0.97),
    ],
)
def test_confidence_matches_vendor_examples(probs, reported):
    assert abs(confidence_from_probs(probs) - reported) <= 0.011


def test_confidence_is_zero_for_uniform_and_one_for_onehot():
    assert confidence_from_probs({"a": 1 / 3, "b": 1 / 3, "c": 1 / 3}) == pytest.approx(0.0)
    assert confidence_from_probs({"a": 1.0, "b": 0.0}) == pytest.approx(1.0)


def test_confidence_single_option_is_one():
    assert confidence_from_probs({"only": 1.0}) == 1.0


# ---------------------------------------------------------------- answers


def test_build_choice_answer_shape_and_sum():
    probs = {"billing": 0.88, "technical": 0.12, "sales": 0.0}
    a = build_answer("choice", probs, criteria={"billing": "p", "technical": "b", "sales": "s"})
    assert a["type"] == "choice"
    assert a["choice"] == "billing"
    assert sum(a["probabilities"].values()) == pytest.approx(1.0, abs=1e-6)
    assert 0.0 <= a["confidence"] <= 1.0


def test_build_score_answer_carries_legend_and_fractional_score():
    probs = {"0": 0.0, "1": 0.95, "2": 0.05}
    a = build_answer("score", probs, criteria=["Calm", "Frustrated", "Very angry"])
    assert a["score"] == pytest.approx(0.0 * 0 + 1 * 0.95 + 2 * 0.05)
    assert a["legend"] == {"0": "Calm", "1": "Frustrated", "2": "Very angry"}
    assert a["probabilities"]["1"] == 0.95


def test_build_noul_answer_has_no_confidence():
    """Vendor: 'Noul answers don't carry one [a confidence]'."""
    a = build_answer("noul", {"false": 0.05, "true": 0.95}, criteria=None)
    assert a["type"] == "noul"
    assert a["noul"] == pytest.approx(0.95)
    assert "confidence" not in a


def test_probabilities_that_do_not_sum_are_rejected():
    with pytest.raises(AnswerError):
        build_answer("choice", {"a": 0.5, "b": 0.2}, criteria={"a": "x", "b": "y"})


def test_decide_request_rejects_extra_question_fields():
    with pytest.raises(ParseError):
        parse_request(
            {
                "state": "s",
                "questions": {
                    "q": {
                        "type": "noul",
                        "instructions": "?",
                        "temperature": 0.5,
                    }
                },
            }
        )
