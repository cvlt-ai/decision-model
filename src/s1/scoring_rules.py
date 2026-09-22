"""Proper scoring rules, used as losses. Lower is better everywhere here.

This is the mathematical core of "RLCD-like" training: a *strictly proper* scoring
rule is minimised in expectation only by the true outcome distribution, so a model
that wants to minimise loss has no strategy better than reporting honest
probabilities. See Gneiting & Raftery (JASA 2007) and the bounded-reward argument in
RLCR (arXiv 2507.16806): correctness + a bounded proper rule is jointly
incentive-compatible for accuracy and calibration.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EPS = 1e-6


def _clamp(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), EPS, 1.0 - EPS)


def brier_loss(p, y) -> float:
    """Brier score: mean (p - y)^2. Strictly proper."""
    p, y = _clamp(p), np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def spherical_loss(p, y) -> float:
    """Spherical scoring rule, negated to a loss. Strictly proper."""
    p, y = _clamp(p), np.asarray(y, dtype=float)
    numer = p * y + (1.0 - p) * (1.0 - y)
    denom = np.sqrt(p**2 + (1.0 - p) ** 2)
    return float(np.mean(-numer / denom))


def log_loss(p, y) -> float:
    """Bernoulli negative log-likelihood, clamped so p=0 stays finite."""
    p, y = _clamp(p), np.asarray(y, dtype=float)
    return float(np.mean(-(y * np.log(p) + (1.0 - y) * np.log(1.0 - p))))


@dataclass
class Decomposition:
    calibration: float  # MECE: how far observed frequencies sit from predicted
    refinement: float  # sharpness/resolution: variance explained by the predictions
    uncertainty: float  # irreducible variance of the outcome itself


def gneiting_raftery(p, y, n_bins: int | None = None) -> Decomposition:
    """Murphy (1973) decomposition of the Brier score:

        Brier = calibration - refinement + uncertainty

    With n_bins=None the groups are the distinct predicted values, which makes the
    identity hold to floating point. That is the right mode for checking the math,
    but on continuous predictions each group has one member, so the calibration term
    degenerates into the raw squared error and measures nothing. For a *calibration*
    reading, pass n_bins (equal-width) so each group has many members.

    `refinement` uses the observed group means (verified numerically: the identity
    closes to 3e-17 with them and misses by ~0.18 with the forecast means).
    """
    p = _clamp(np.ravel(p))
    y = np.ravel(np.asarray(y, dtype=float))
    n = p.size
    if n == 0:
        return Decomposition(float("nan"), float("nan"), float("nan"))
    if n_bins is None:
        group_idx = np.unique(p, return_inverse=True)[1]
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)
        group_idx = np.clip(np.digitize(p, edges[1:-1]), 0, n_bins - 1)

    counts = np.bincount(group_idx).astype(float)
    safe = np.where(counts > 0, counts, 1.0)  # empty bins: compute, then mask out
    ybar = np.bincount(group_idx, weights=y) / safe
    pbar = np.bincount(group_idx, weights=p) / safe
    present = counts > 0

    calibration = float(np.sum(counts[present] * (ybar[present] - pbar[present]) ** 2) / n)
    refinement = float(np.sum(counts[present] * (ybar[present] - y.mean()) ** 2) / n)
    uncertainty = float(y.mean() * (1.0 - y.mean()))
    return Decomposition(calibration, refinement, uncertainty)


def verdict_consistency_penalty(p_yes, p_not):
    """Penalise |P(x) + P(not x) - 1|.

    TypeSafe's own jaggedness page ships this failure: P(refund) = 0.72 and
    P(not refund) = 0.47, summing to 1.19, and says plainly not to rely on the
    identity holding. It is trivially enforceable as an auxiliary loss, which is one
    concrete way our model can be *better* rather than merely equal.
    """
    a = np.asarray(p_yes, dtype=float)
    b = np.asarray(p_not, dtype=float)
    out = np.abs(a + b - 1.0)
    return float(out) if np.ndim(out) == 0 else out
