"""Evaluation metrics for decision models: accuracy is not enough, calibration is."""

from __future__ import annotations

import numpy as np


def ece(p, y, n_bins: int = 15) -> float:
    """Expected calibration error (top-label, equal-width bins)."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    if p.size == 0:
        return float("nan")
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (p >= lo) & (p <= hi) if lo == 0.0 else (p > lo) & (p <= hi)
        if not mask.any():
            continue
        total += mask.mean() * abs(p[mask].mean() - y[mask].mean())
    return float(total)


def brier(p, y) -> float:
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    return float(np.mean((p - y) ** 2))


def coverage_accuracy(p, y, thr: float) -> tuple[float, float | None, int]:
    """(coverage, accuracy-within-coverage, n_selected). acc is None when nothing
    clears the threshold."""
    p = np.asarray(p, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = p >= thr
    n = int(mask.sum())
    if n == 0:
        return 0.0, None, 0
    return float(mask.mean()), float(y[mask].mean()), n


def permutation_robustness(probs_a, probs_b) -> float:
    """Max absolute probability shift for the *same* option across two option
    orders. TypeSafe does not guarantee this; Jev reportedly shifts up to ~0.20.
    Our target (C5) is mean <= 0.02."""
    a = np.asarray(probs_a, dtype=float)
    b = np.asarray(probs_b, dtype=float)
    return float(np.max(np.abs(a - b)))


def argmax_flip_rate(base_orders, perm_orders) -> float:
    """Fraction of examples whose winning option changes when options are
    reordered. base_orders/perm_orders: (n_examples, n_options) with the SAME
    column meaning option in both (i.e. probabilities aligned by option key,
    not by position)."""
    base = np.asarray(base_orders, dtype=float)
    perm = np.asarray(perm_orders, dtype=float)
    if base.shape != perm.shape or base.shape[0] == 0:
        raise ValueError("shapes must match and be non-empty")
    return float(np.mean(np.argmax(base, axis=1) != np.argmax(perm, axis=1)))
