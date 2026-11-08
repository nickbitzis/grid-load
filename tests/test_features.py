import numpy as np
import pytest

from gridload.features import (
    COL,
    FEATURES,
    MIN_ISSUE,
    build,
    node_data,
    scale7,
    similar_days_mw,
    targets,
    training_pairs,
)
from gridload.generator import ISSUE_HOUR, forecast_weather


def test_weather_guard_refuses_forecasts_not_yet_issued(ds):
    assert forecast_weather(ds, 100, 101)["temp"].shape == (24, 5)
    with pytest.raises(ValueError):
        forecast_weather(ds, 100, 99)  # issued after the target
    with pytest.raises(ValueError):
        forecast_weather(ds, 100, 108)  # beyond the 7-day archive


def test_build_rejects_bad_leads_and_early_issues(ds, nd):
    with pytest.raises(ValueError):
        build(ds, nd, np.array([100]), np.array([0]))
    with pytest.raises(ValueError):
        build(ds, nd, np.array([100]), np.array([8]))
    with pytest.raises(ValueError):
        build(ds, nd, np.array([MIN_ISSUE - 1]), np.array([1]))


def test_perturbing_the_future_changes_nothing(ds):
    """The leakage guard: change metered load from 10:00 on the issue day onward, and every actual weather value
    from the issue day on. Forecast-mode features must not move."""
    import copy

    issue = 700
    nd1 = node_data(ds)
    X1, s1 = build(ds, nd1, np.array([issue] * 7), np.arange(1, 8))
    ds2 = copy.copy(ds)
    ds2.temp = ds.temp.copy()
    ds2.temp[issue:] += 25
    ds2.cloud = ds.cloud.copy()
    ds2.cloud[issue:] = 1 - ds2.cloud[issue:]
    nd2 = node_data(ds2, flags=nd1.flags[..., :-1])
    nd2.load = nd1.load.copy()
    nd2.load[issue, ISSUE_HOUR:] *= 3
    nd2.load[issue + 1 :] *= 3
    X2, s2 = build(ds2, nd2, np.array([issue] * 7), np.arange(1, 8))
    assert np.array_equal(X1, X2) and np.array_equal(s1, s2)


def test_actual_weather_mode_differs(ds, nd):
    i, j = np.array([700]), np.array([3])
    Xf, _ = build(ds, nd, i, j, "forecast")
    Xa, _ = build(ds, nd, i, j, "actual")
    assert not np.allclose(Xf[:, COL["temp"]], Xa[:, COL["temp"]])
    assert np.allclose(Xa[: 24 * 6 : 6, COL["temp"]], ds.temp[703, :, 0])


def test_lag_week_steps_back_when_not_metered(ds, nd):
    issue = 700
    X, sc = build(ds, nd, np.array([issue]), np.array([7]))
    X = X.reshape(24, 6, -1)
    sc = sc.reshape(24, 6)
    # target day 707: same hour last week is day 700, metered only before 10:00
    assert np.isclose(X[5, 0, COL["lag_week"]] * sc[5, 0], nd.load[700, 5, 0])
    assert np.isclose(X[15, 0, COL["lag_week"]] * sc[15, 0], nd.load[693, 15, 0])


def test_lag_day_uses_latest_metered_hour(ds, nd):
    X, sc = build(ds, nd, np.array([700]), np.array([1]))
    X, sc = X.reshape(24, 6, -1), sc.reshape(24, 6)
    assert np.isclose(X[8, 2, COL["lag_day"]] * sc[8, 2], nd.load[700, 8, 2])
    assert np.isclose(X[20, 2, COL["lag_day"]] * sc[20, 2], nd.load[699, 20, 2])


def test_scale7_is_the_last_168_metered_hours(nd):
    s = scale7(nd, np.array([700]))[0, 1]
    flat = nd.load[693:701, :, 1].reshape(-1)
    assert np.isclose(s, flat[ISSUE_HOUR : ISSUE_HOUR + 168].mean())


def test_system_node_sums_zones(nd):
    assert np.allclose(nd.load[:100, :, 5], nd.load[:100, :, :5].sum(-1))


def test_training_pairs_stay_in_range():
    i, j = training_pairs(100, 400, np.random.default_rng(0))
    assert (i + j).max() <= 400 and (i + j).min() >= 100 and i.min() >= MIN_ISSUE
    assert 0.25 < np.mean(j == 1) < 0.45


def test_targets_exclude_flagged_and_unmetered(ds, nd):
    k = ds.day_index(__import__("datetime").date(2026, 1, 21))
    y, ok = targets(nd, np.array([k - 1]), np.array([1]))
    assert (~ok).sum() >= 20  # outage hours in north and the system
    y, ok = targets(nd, np.array([ds.today - 1]), np.array([1]))
    assert ok.reshape(24, 6)[ISSUE_HOUR:].sum() == 0  # today after 10:00 is not metered yet


def test_similar_days_only_look_back(ds, nd):
    a = similar_days_mw(ds, nd, np.array([700]), np.array([2]))
    nd2 = node_data(ds, flags=nd.flags[..., :-1])
    nd2.load = nd.load.copy()
    nd2.load[700:] = 0
    b = similar_days_mw(ds, nd2, np.array([700]), np.array([2]))
    assert np.array_equal(a, b) and a.shape == (1, 24, 6)


def test_feature_list():
    assert len(FEATURES) == len(set(FEATURES)) and "analog" in FEATURES and "temp" in FEATURES
