"""Shared eval-row plumbing."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable

from s1.schema import ParseError, parse_request

REGISTRY: dict[str, Callable[[int | None], list["EvalExample"]]] = {}


def register(fn: Callable[[int | None], list["EvalExample"]]):
    REGISTRY[fn.__name__] = fn
    return fn


@dataclass
class Gold:
    kind: str  # "label" | "prob"
    value: Any  # option key for label; float for prob

    def to_dict(self) -> dict:
        return {"kind": self.kind, "value": self.value}


@dataclass
class EvalExample:
    family: str
    uid: str
    state: Any
    questions: dict[str, dict]  # qid -> raw Jev question dict
    gold: dict[str, Gold]
    meta: dict = field(default_factory=dict)

    def validate(self) -> None:
        """Every row must be a legal Jev request before it may enter the holdout."""
        parse_request({"state": self.state, "questions": self.questions})

    def to_dict(self) -> dict:
        # state/questions/gold/meta are stored as JSON strings. Parquet struct
        # inference would (a) reject the childless struct from an empty meta and
        # (b) null-fill option keys that differ row to row (MMLU-Pro has between
        # 2 and 10 options), silently corrupting the criteria map on read-back.
        return {
            "family": self.family,
            "uid": self.uid,
            "state": json.dumps(self.state, ensure_ascii=False),
            "questions": json.dumps(self.questions, ensure_ascii=False),
            "gold": json.dumps({k: g.to_dict() for k, g in self.gold.items()}, ensure_ascii=False),
            "meta": json.dumps(self.meta, sort_keys=True, ensure_ascii=False),
        }

    @staticmethod
    def from_dict(d: dict) -> "EvalExample":
        gold_raw = json.loads(d["gold"])
        return EvalExample(
            family=d["family"],
            uid=d["uid"],
            state=json.loads(d["state"]),
            questions=json.loads(d["questions"]),
            gold={k: Gold(g["kind"], g["value"]) for k, g in gold_raw.items()},
            meta=json.loads(d["meta"]),
        )


def content_hash(rows: list[EvalExample]) -> str:
    h = hashlib.sha256()
    for r in rows:
        h.update(json.dumps(r.to_dict(), sort_keys=True, ensure_ascii=False).encode())
        h.update(b"\n")
    return h.hexdigest()


def check_gold_in_options(q: dict, gold: Gold) -> None:
    if gold.kind != "label":
        return
    qtype = q["type"]
    keys: list[str]
    if qtype == "choice":
        keys = list(q["criteria"].keys())
    elif qtype == "noul":
        keys = ["true", "false"]
    else:
        keys = [str(i) for i in range(len(q["criteria"]))]
    if str(gold.value) not in keys:
        raise ValueError(f"gold {gold.value!r} not among options {keys}")
