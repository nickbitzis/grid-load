import numpy as np
import pytest

from gridload import scoring
from gridload.features import COL, build, node_data, pairs, targets, training_pairs
from gridload.models import GBM, GBMConfig, SameHourLastWeek, SimilarDays, SplineRegression

SMALL = GBMConfig(max_iter=60, quantile_iter=40)


@pytest.fixture(scope="module")
def split(tiny):
    nd = node_data(tiny)
    i, j = training_pairs(20, 120, np.random.default_rng(0))
    X, sc = build(tiny, nd, i, j)
    y, ok = targets(nd, i, j)
    pi, pj = pairs(np.arange(121, 140), [1])
    Xt, st = build(tiny, nd, pi, pj)
    yt, okt = targets(nd, pi, pj)
    return {"X": X[ok], "y": (y / sc)[ok], "Xt": Xt, "st": st, "yt": yt, "w": okt.astype(float), "nd": nd,
            "pi": pi, "pj": pj}  # fmt: skip


def _mape(s, ratio):
    return scoring.mape(s["yt"], ratio * s["st"], s["w"])


def test_same_hour_last_week_is_the_lag(split):
    p = SameHourLastWeek().fit(split["X"], split["y"]).predict(split["Xt"])
    assert np.array_equal(p, split["Xt"][:, COL["lag_week"]])
    assert 0.02 < _mape(split, p) < 0.2


def test_similar_days_on_tiny_data(tiny, split):
    mw = SimilarDays().predict_mw(tiny, split["nd"], split["pi"], split["pj"])
    assert np.isfinite(mw).all()
    assert scoring.mape(split["yt"], mw.reshape(-1), split["w"]) < 0.1


def test_spline_regression_on_tiny_data(split):
    p = SplineRegression().fit(split["X"], split["y"]).predict(split["Xt"])
    assert np.isfinite(p).all() and _mape(split, p) < 0.1


def test_spline_regression_learns_the_u_shape(ds):
    nd = node_data(ds)
    i, j = training_pairs(400, 1400, np.random.default_rng(1))
    X, sc = build(ds, nd, i, j)
    y, ok = targets(nd, i, j)
    m = SplineRegression().fit(X[ok], (y / sc)[ok])
    probe = np.repeat(X[ok][:1], 3, axis=0)
    probe[:, COL["temp"]] = [-12, 17, 34]
    probe[:, COL["node"]] = 5
    cold, mild, hot = m.predict(probe)
    assert cold > mild and hot > mild


def test_gbm_beats_last_week_on_tiny_data(split):
    p = GBM(SMALL).fit(split["X"], split["y"]).predict(split["Xt"])
    assert _mape(split, p) < _mape(split, split["Xt"][:, COL["lag_week"]])


def test_quantile_gbm_orders_its_band(split):
    lo = GBM(SMALL, 0.1).fit(split["X"], split["y"]).predict(split["Xt"])
    hi = GBM(SMALL, 0.9).fit(split["X"], split["y"]).predict(split["Xt"])
    assert np.mean(lo < hi) > 0.95
    cov = scoring.coverage(split["yt"], lo * split["st"], hi * split["st"], split["w"])
    assert 0.5 < cov < 0.98


def test_gbm_is_deterministic(split):
    a = GBM(SMALL, seed=0).fit(split["X"], split["y"]).predict(split["Xt"][:50])
    b = GBM(SMALL, seed=0).fit(split["X"], split["y"]).predict(split["Xt"][:50])
    assert np.array_equal(a, b)
