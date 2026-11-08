import numpy as np
import pytest

from gridload import conformal, scoring


def test_mape_rmse_bias_by_hand():
    y, f, w = np.array([100.0, 200.0]), np.array([110.0, 180.0]), np.ones(2)
    assert scoring.mape(y, f, w) == pytest.approx(0.1)
    assert scoring.rmse(y, f, w) == pytest.approx(np.sqrt((100 + 400) / 2))
    assert scoring.bias(y, f, w) == pytest.approx(-10 / 300)


def test_weights_count_real_hours():
    y, f = np.array([100.0, 100.0]), np.array([110.0, 100.0])
    assert scoring.mape(y, f, np.array([2.0, 1.0])) == pytest.approx(20 / 3 / 100)
    assert scoring.mape(y, f, np.array([0.0, 1.0])) == 0  # the spring hour that does not exist
    assert scoring.mape(np.array([np.nan, 100.0]), f, np.ones(2)) == 0  # unmetered cells drop out


def test_pinball_by_hand():
    y = np.array([10.0])
    assert scoring.pinball(y, np.array([8.0]), 0.9, np.ones(1)) == pytest.approx(1.8)
    assert scoring.pinball(y, np.array([12.0]), 0.9, np.ones(1)) == pytest.approx(0.2)
    assert scoring.pinball(y, np.array([12.0]), 0.5, np.ones(1)) == pytest.approx(1.0)
    qf = {0.1: np.array([8.0]), 0.5: np.array([10.0]), 0.9: np.array([12.0])}
    assert scoring.mean_pinball(y, qf, np.ones(1)) == pytest.approx((0.2 + 0 + 0.2) / 3)


def test_coverage_by_hand():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    assert scoring.coverage(y, y - 0.5, y + 0.5, np.ones(4)) == 1
    assert scoring.coverage(y, np.full(4, 1.5), np.full(4, 3.5), np.ones(4)) == 0.5


def test_peak_error_by_hand():
    y = np.tile(np.arange(24.0) + 100, (2, 1))
    f = y.copy()
    f[0, 23] = 123 * 1.1
    assert scoring.peak_error(y, f, np.ones_like(y)) == pytest.approx(0.05)


def test_conformal_quantile_finite_sample():
    s = np.arange(1, 10.0)  # n = 9, ceil(10 * 0.8) = 8th smallest
    assert conformal.conformal_quantile(s, 0.2) == 8
    assert conformal.conformal_quantile(np.array([]), 0.2) == 0


def test_conformal_reaches_target_coverage():
    rng = np.random.default_rng(0)
    n_iss = 200
    y = 100 + rng.normal(0, 10, (n_iss, 2, 24, 2))
    lo, hi, p50 = np.full_like(y, 95.0), np.full_like(y, 105.0), np.full_like(y, 100.0)  # too narrow
    ok = np.ones_like(y, bool)
    lo2, hi2 = conformal.calibrate(np.arange(n_iss), y, lo, hi, p50, ok, [np.array([0, 1])])
    late = slice(60, n_iss)
    cov = np.mean((y[late] >= lo2[late]) & (y[late] <= hi2[late]))
    assert 0.76 < cov < 0.84


def test_conformal_uses_only_past_scores():
    n_iss = 30
    y = np.full((n_iss, 1, 24, 1), 100.0)
    y[20:] = 1000  # huge misses in the future must not widen earlier bands
    lo, hi, p50 = y * 0 + 95, y * 0 + 105, y * 0 + 100
    lo2, hi2 = conformal.calibrate(np.arange(n_iss), y, lo, hi, p50, np.ones_like(y, bool), [np.array([0])])
    # CQR may narrow a band that has been too wide, but nothing before issue 21 can see the misses
    assert np.all(hi2[:21] <= hi[:21] + 1e-9) and np.all(lo2[:21] >= lo[:21] - 1e-9)
    assert hi2[29, 0, 0, 0] > 500
