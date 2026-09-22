import numpy as np
import pytest

from s1.scoring_rules import (
    brier_loss,
    gneiting_raftery,
    log_loss,
    spherical_loss,
    verdict_consistency_penalty,
)


def test_brier_minimised_at_truth():
    y = np.array([1.0])
    grid = np.linspace(0, 1, 101)
    best = min(brier_loss(np.array([p]), y) for p in grid)
    assert abs(brier_loss(np.array([1.0]), y) - best) < 1e-12


def test_spherical_minimised_at_truth():
    y = np.array([1.0])
    assert spherical_loss(np.array([1.0]), y) <= spherical_loss(np.array([0.5]), y)


def test_log_loss_finite_after_clamp():
    assert np.isfinite(log_loss(np.array([0.0]), np.array([1.0])))


@pytest.mark.parametrize("fn", [brier_loss, spherical_loss, log_loss])
def test_all_are_losses_lower_is_better(fn):
    y = np.array([1.0, 0.0, 1.0])
    good = fn(np.array([0.9, 0.1, 0.8]), y)
    bad = fn(np.array([0.5, 0.5, 0.5]), y)
    awful = fn(np.array([0.1, 0.9, 0.2]), y)
    assert good < bad < awful


def test_decomposition_components_are_nonnegative():
    rng = np.random.default_rng(1)
    p = rng.uniform(0, 1, 2000)
    y = (rng.uniform(0, 1, 2000) < p).astype(float)
    d = gneiting_raftery(p, y, n_bins=15)
    assert d.calibration >= -1e-9 and d.refinement >= -1e-9 and d.uncertainty >= -1e-9
    # perfectly calibrated synthetic data: calibration term ~0
    assert d.calibration < 0.01


def test_decomposition_sums_to_brier_exactly():
    # Murphy (1973): Brier = calibration - refinement + uncertainty
    rng = np.random.default_rng(7)
    p = rng.uniform(0.05, 0.95, 3000)
    y = (rng.uniform(0, 1, 3000) < p).astype(float)
    d = gneiting_raftery(p, y)  # distinct-value groups -> exact identity
    assert abs((d.calibration - d.refinement + d.uncertainty) - brier_loss(p, y)) < 1e-12


def test_decomposition_binned_residual_matches_derivation():
    """With binned groups, writing p_i = pbar_g + delta_i:

        Brier = calibration - refinement + uncertainty
                + E[delta^2] - 2*E[y * delta]

    (the cross term survives because sum_i delta_i = 0 within a bin kills only the
    pbar part). Assert this identity exactly rather than loosening a tolerance —
    the loose version passed by coincidence before the derivation was right.
    """
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    d = gneiting_raftery(p, y, n_bins=15)
    resid = (d.calibration - d.refinement + d.uncertainty) - brier_loss(p, y)

    edges = np.linspace(0.0, 1.0, 16)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, 14)
    cnt = np.bincount(idx).astype(float)
    pbar = np.bincount(idx, weights=p) / cnt
    delta = p - pbar[idx]
    expected = 2.0 * np.mean(y * delta) - np.mean(delta**2)
    assert resid == pytest.approx(expected, abs=1e-12)
    # and the within-bin spread really is what makes the two differ
    assert np.mean(delta**2) < 0.005


def test_overconfidence_raises_the_binned_calibration_term():
    rng = np.random.default_rng(11)
    p = rng.uniform(0.05, 0.95, 4000)
    y = (rng.uniform(0, 1, 4000) < p).astype(float)
    over = np.clip(p * 1.6, 1e-6, 1 - 1e-6)
    assert gneiting_raftery(over, y, n_bins=15).calibration > (
        gneiting_raftery(p, y, n_bins=15).calibration
    )


def test_consistency_penalty_zero_for_complementary_probabilities():
    # P(x) + P(not x) must equal 1; Jev's own docs show 0.72 + 0.47 = 1.19
    assert verdict_consistency_penalty([0.3], [0.7]) == 0.0
    assert np.allclose(
        verdict_consistency_penalty([0.3, 0.9], [0.7, 0.1]), [0.0, 0.0]
    )


def test_consistency_penalty_matches_jev_documented_case():
    pen = verdict_consistency_penalty([0.72], [0.47])
    assert pen == pytest.approx(0.19)  # |0.72 + 0.47 - 1|


def test_consistency_penalty_is_batched():
    a = np.array([0.5, 0.5, 0.5])
    b = np.array([0.5, 0.4, 0.6])
    out = verdict_consistency_penalty(a, b)
    assert np.allclose(out, [0.0, 0.1, 0.1])
