"""Eval-suite adapters. Each adapter yields EvalExample rows from a frozen split.

Contract (enforced in base.py):
  * test/validation splits ONLY — a train split must never reach this code path
  * fixed N with a pinned seed, so the holdout is reproducible across machines
  * every row validates against s1.schema before it is accepted
"""

from __future__ import annotations

from .base import EvalExample, REGISTRY, register


def all_adapters() -> dict:
    from . import (  # noqa: F401  (import registers each adapter)
        banking77,
        baserate,
        boolq,
        injection,
        go_emotions,
        mmlu_pro,
        negation,
        pubhealth,
        severity,
    )

    return REGISTRY


__all__ = ["EvalExample", "REGISTRY", "register", "all_adapters"]
