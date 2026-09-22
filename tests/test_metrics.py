import numpy as np

from s1.metrics import brier, coverage_accuracy, ece, permutation_robustness


def test_ece_perfectly_calibrated_is_near_zero():
    rng = np.random.default_rng(0)
    p = rng.uniform(0, 1, 10000)
    y = (rng.uniform(0, 1, 10000) < p).astype(float)
    assert ece(p, y, n_bins=15) < 0.02


def test_ece_maximally_overconfident_is_high():
    p = np.full(1000, 0.99)
    y = np.zeros(1000)
    assert ece(p, y) > 0.9


def test_ece_empty_is_nan_not_crash():
    assert np.isnan(ece(np.array([]), np.array([])))


def test_brier_matches_manual():
    p = np.array([0.9, 0.2, 0.5])
    y = np.array([1, 0, 1])
    assert abs(brier(p, y) - ((0.01 + 0.04 + 0.25) / 3)) < 1e-9


def test_coverage_accuracy_endpoints():
    p = np.array([0.99, 0.6, 0.55, 0.51])
    y = np.array([1, 1, 0, 1])
    cov, acc, n = coverage_accuracy(p, y, thr=1.0)
    assert cov == 0.0 and acc is None and n == 0
    # only 0.99 clears 0.9 -> one selection, and that one is correct
    cov, acc, n = coverage_accuracy(p, y, thr=0.9)
    assert cov == 0.25 and acc == 1.0 and n == 1
    cov, acc, n = coverage_accuracy(p, y, thr=0.5)
    assert cov == 1.0 and acc == 0.75 and n == 4


def test_coverage_curve_is_monotone_in_coverage():
    rng = np.random.default_rng(3)
    p = rng.uniform(0, 1, 500)
    y = (rng.uniform(0, 1, 500) < p).astype(float)
    covs = [coverage_accuracy(p, y, thr=t)[0] for t in np.linspace(0, 1, 21)]
    assert covs == sorted(covs, reverse=True)


def test_permutation_robustness_is_max_abs_gap():
    # |0.1-0.2|=0.10, |0.6-0.55|=0.05, |0.9-0.88|=0.02  -> max is 0.10
    assert permutation_robustness([0.1, 0.6, 0.9], [0.2, 0.55, 0.88]) == 0.1
    assert permutation_robustness([0.5, 0.5], [0.5, 0.5]) == 0.0


def test_selector_agreement_rate_counts_argmax_flips():
    from s1.metrics import argmax_flip_rate

    base = [[0.1, 0.7, 0.2], [0.8, 0.1, 0.1]]
    perm = [[0.2, 0.1, 0.7], [0.6, 0.3, 0.1]]  # first flips winner, second holds
    assert argmax_flip_rate(base, perm) == 0.5
