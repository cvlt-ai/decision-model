"""Jev wire-spec reconstruction: request parsing, prompt rendering, answer building.

Written against the vendor's own documentation (archived under
research/notes/raw/), not against guesses. Notes on the parts that are
reconstructions rather than published facts are marked RECONSTRUCTED.
"""

from __future__ import annotations

import json
import math
from typing import Any, Literal, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator

QType = Literal["choice", "score", "noul"]
Structured = Union[str, dict[str, Any], list[Any]]

MAX_OPTIONS = 255  # vendor: "a maximum of 255 options per Choice"
MIN_SCORE_LEVELS, MAX_SCORE_LEVELS = 2, 10  # vendor: "at least two; accepts up to 10"


class ParseError(ValueError):
    """The request body failed validation. Maps to HTTP 422, like the vendor."""


class AnswerError(ValueError):
    """A model's raw output could not be turned into a legal answer."""


# --------------------------------------------------------------------- questions


class Question(BaseModel):
    model_config = ConfigDict(extra="forbid", populate_by_name=True)

    qtype: QType = Field(alias="type")
    instructions: Structured
    criteria: Union[dict[str, Union[str, dict, list, None]], list[Structured], None] = None

    @field_validator("instructions")
    @classmethod
    def _instructions_not_blank(cls, v: Structured) -> Structured:
        if isinstance(v, str) and not v.strip():
            raise ValueError("instructions must not be empty")
        if isinstance(v, (dict, list)) and not v:
            raise ValueError("instructions must not be empty")
        return v

    def option_keys(self) -> list[str]:
        if self.qtype == "noul":
            return ["true", "false"]
        if self.qtype == "choice":
            assert isinstance(self.criteria, dict)
            return list(self.criteria.keys())
        assert isinstance(self.criteria, list)
        return [str(i) for i in range(len(self.criteria))]  # type: ignore[arg-type]


class DecideRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    state: Union[str, dict[str, Any], list[Any]]
    questions: dict[str, Question]
    model: str = "jev-latest"

    @field_validator("questions")
    @classmethod
    def _questions_nonempty(cls, v: dict[str, Question]) -> dict[str, Question]:
        if not v:
            raise ValueError("questions must contain at least one entry")
        for name, q in v.items():
            _validate_criteria(name, q)
        return v


def _validate_criteria(name: str, q: Question) -> None:
    if q.qtype == "choice":
        if not isinstance(q.criteria, dict) or not q.criteria:
            raise ParseError(f"question '{name}': choice requires a criteria map")
        if len(q.criteria) > MAX_OPTIONS:
            raise ParseError(
                f"question '{name}': choice accepts at most {MAX_OPTIONS} options, "
                f"got {len(q.criteria)}"
            )
    elif q.qtype == "score":
        if not isinstance(q.criteria, list):
            raise ParseError(f"question '{name}': score requires an ordered criteria array")
        if not (MIN_SCORE_LEVELS <= len(q.criteria) <= MAX_SCORE_LEVELS):
            raise ParseError(
                f"question '{name}': score requires {MIN_SCORE_LEVELS}-{MAX_SCORE_LEVELS} "
                f"levels, got {len(q.criteria)}"
            )
    else:  # noul
        if q.criteria is not None and not isinstance(q.criteria, dict):
            raise ParseError(f"question '{name}': noul criteria must be an object or absent")


def parse_request(body: Any) -> DecideRequest:
    try:
        return DecideRequest.model_validate(body)
    except ParseError:
        raise
    except Exception as exc:  # pydantic ValidationError and friends
        raise ParseError(str(exc)) from exc


# ------------------------------------------------------------------ rendering


def render_instructions(instructions: Structured) -> str:
    """Vendor: `instructions` may be an object/array holding data plus a `question`
    field that refers to the other fields by name in backticks. We expand those
    references inline so the model actually sees the data it is pointed at."""
    if isinstance(instructions, str):
        return instructions
    if isinstance(instructions, list):
        return _render_json(instructions)
    # object: pull the question out, expose the rest as named, referenceable data
    question = None
    for key in ("question", "instructions", "q"):
        if key in instructions:
            question = instructions.pop(key)
            break
    data = {k: v for k, v in instructions.items() if k != question}
    parts: list[str] = []
    if data:
        parts.append("Context:\n" + _render_json(data))
    if question is None:
        parts.append(_render_json(instructions))
    else:
        qtext = question if isinstance(question, str) else _render_json(question)
        for name, value in data.items():
            ref = f"`{name}`"
            if ref in qtext:
                qtext = qtext.replace(ref, f"{name}={_render_json(value)}")
        parts.append(qtext)
    return "\n\n".join(parts)


def _render_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=None, sort_keys=False)


class DecideRequestView:
    """Prompt rendering for backends that need a single string (encoder readout,
    causal-decoder logit readout). The API object form is canonical; this is the
    flattened projection."""

    def __init__(self, req: DecideRequest):
        self._req = req

    def state_text(self) -> str:
        s = self._req.state
        return s if isinstance(s, str) else _render_json(s)

    def prompt(self, qid: str, option_order: list[str] | None = None) -> str:
        q = self._req.questions[qid]
        keys = q.option_keys()
        order = keys if option_order is None else option_order
        if sorted(order) != sorted(keys):
            raise ValueError("option_order must be a permutation of the option keys")

        lines = ["State:", self.state_text(), "", "Question:", render_instructions(q.instructions), "", "Options:"]
        for i, key in enumerate(order):
            desc = _describe(q, key)
            lines.append(f"[{i}] {key}" + (f" — {desc}" if desc else ""))
        lines += ["", "Answer with the option number only."]
        return "\n".join(lines)


def _describe(q: Question, key: str) -> str:
    if q.qtype == "score":
        assert isinstance(q.criteria, list)
        v = q.criteria[int(key)]
        return v if isinstance(v, str) else _render_json(v)
    if isinstance(q.criteria, dict) and key in q.criteria:
        v = q.criteria[key]
        return "" if v is None else (v if isinstance(v, str) else _render_json(v))
    return ""


# ---------------------------------------------------------------- confidence

# RECONSTRUCTED. The vendor documents that `confidence` is "a statistic computed
# from the probability distribution" but deliberately withholds the formula
# ("a specialized topic that we'll keep to a separate cookbook"). We fitted
# candidate statistics against every (distribution -> reported confidence) pair
# published in their docs and picked the closest: 1 - Hn**CONF_EXPONENT, where
# Hn is entropy normalised by log(K). Residual against the four published pairs
# is <= 0.008. It is monotone, 0 for uniform, 1 for one-hot. It is a compatible
# approximation, not their function.
CONF_EXPONENT = 1.5


def confidence_from_probs(probs: dict[str, float]) -> float:
    p = _normalise(list(probs.values()))
    k = len(p)
    if k <= 1:
        return 1.0
    hn = -sum(x * math.log(x) for x in p if x > 0) / math.log(k)
    hn = min(max(hn, 0.0), 1.0)
    return float(1.0 - hn**CONF_EXPONENT)


def _normalise(values: list[float]) -> list[float]:
    total = sum(values)
    if total <= 0:
        raise AnswerError("probabilities sum to zero")
    return [v / total for v in values]


# ------------------------------------------------------------------- answers


def build_answer(
    qtype: QType,
    probs: dict[str, float],
    *,
    criteria: Union[dict[str, Any], list[Any], None],
    tol: float = 1e-3,
) -> dict[str, Any]:
    """Turn raw per-option probabilities into a Jev-shaped answer.

    Raises AnswerError rather than silently renormalising, so a broken head shows
    up as a failed request instead of a quietly wrong decision.
    """
    if not probs:
        raise AnswerError("no probabilities supplied")
    total = sum(probs.values())
    if abs(total - 1.0) > tol:
        raise AnswerError(f"probabilities must sum to 1 (got {total:.6f})")
    if any(v < -tol for v in probs.values()):
        raise AnswerError("probabilities must be non-negative")

    if qtype == "noul":
        return {"type": "noul", "noul": float(probs.get("true", 0.0))}

    if qtype == "choice":
        winner = max(probs, key=lambda k: probs[k])
        return {
            "type": "choice",
            "choice": winner,
            "probabilities": {k: float(v) for k, v in probs.items()},
            "confidence": confidence_from_probs(probs),
        }

    # score
    levels = sorted(probs, key=lambda k: int(k))
    score = sum(int(k) * v for k, v in probs.items())
    legend = {k: _level_text(criteria, int(k)) for k in levels}
    return {
        "type": "score",
        "score": float(score),
        "legend": legend,
        "probabilities": {k: float(probs[k]) for k in levels},
        "confidence": confidence_from_probs({k: probs[k] for k in levels}),
    }


def _level_text(criteria: Any, idx: int) -> str:
    if not isinstance(criteria, list) or idx >= len(criteria):
        return str(idx)
    v = criteria[idx]
    return v if isinstance(v, str) else _render_json(v)


# small helper so callers can go request -> view in one step
def view(req: DecideRequest) -> DecideRequestView:
    return DecideRequestView(req)


# bind the convenience method used by tests
def _render_prompt(self: DecideRequest, qid: str, option_order: list[str] | None = None) -> str:
    return DecideRequestView(self).prompt(qid, option_order)


def _render(self: DecideRequest) -> dict[str, Any]:
    return {
        "model": self.model,
        "state": self.state,
        "questions": {
            name: {
                "type": q.qtype,
                "instructions": q.instructions,
                **({"criteria": q.criteria} if q.criteria is not None else {}),
            }
            for name, q in self.questions.items()
        },
    }


DecideRequest.render_prompt = _render_prompt  # type: ignore[attr-defined]
DecideRequest.render = _render  # type: ignore[attr-defined]
