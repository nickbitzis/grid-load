from datetime import date

import numpy as np

from gridload import calendar as cal
from gridload.generator import ISSUE_HOUR, GenConfig, generate, to_hourly


def test_shapes_and_dates(ds):
    n = len(ds.dates)
    assert ds.n_hist == 1461 and n == 1461 + 8
    assert ds.load.shape == (n, 24, 5) and ds.temp_fc.shape == (n, 8, 24, 5)
    assert ds.dates[ds.today] == date(2026, 11, 8)


def test_same_seed_same_data():
    a, b = generate(GenConfig(days=60, seed=1)), generate(GenConfig(days=60, seed=1))
    c = generate(GenConfig(days=60, seed=2))
    assert np.array_equal(a.load, b.load) and not np.array_equal(a.load, c.load)


def test_temperature_u_shape(ds):
    sys_load = ds.load[: ds.n_hist].sum(2).mean(1)
    t = ds.temp[: ds.n_hist].mean(axis=(1, 2))
    cold, mild, hot = sys_load[t < -5].mean(), sys_load[(t > 14) & (t < 19)].mean(), sys_load[t > 27].mean()
    assert cold > mild * 1.2 and hot > mild * 1.2


def test_holidays_lower_than_ordinary_weekdays(ds):
    sys_load = ds.load[: ds.n_hist].sum(2).mean(1)
    weekday = (ds.dow[: ds.n_hist] < 5) & ~ds.holiday[: ds.n_hist] & ~ds.bridge[: ds.n_hist]
    ratios = []
    for k in np.where(ds.holiday[: ds.n_hist] & (ds.dow[: ds.n_hist] < 5))[0]:
        near = [d for d in range(k - 14, k + 15) if 0 <= d < ds.n_hist and weekday[d]]
        ratios.append(sys_load[k] / sys_load[near].mean())
    assert len(ratios) >= 20 and np.median(ratios) < 0.94


def test_bridge_days_between_weekday_and_holiday(ds):
    thanks = date(2025, 11, 27)
    fri = ds.day_index(date(2025, 11, 28))
    assert ds.holiday[ds.day_index(thanks)] and ds.bridge[fri]
    sys_load = ds.load.sum(2).mean(1)
    ordinary_fridays = [fri - 7, fri - 14, fri + 7]
    assert sys_load[fri] < np.mean(sys_load[ordinary_fridays])


def test_holiday_calendar():
    h = cal.holidays(2026)
    assert h[date(2026, 11, 26)] == "Thanksgiving" and h[date(2026, 7, 3)] == "Independence Day"  # observed Friday
    assert date(2026, 12, 24) in cal.bridge_days(2026)


def test_dst_hour_weights(ds):
    spring, fall = ds.day_index(date(2026, 3, 8)), ds.day_index(date(2025, 11, 2))
    assert ds.hour_weight[spring].sum() == 23 and ds.hour_weight[spring, 2] == 0
    assert ds.hour_weight[fall].sum() == 25 and ds.hour_weight[fall, 1] == 2
    others = np.delete(ds.hour_weight.sum(1), [spring, fall])
    assert set(np.unique(others)) <= {23.0, 24.0, 25.0}
    assert np.sum(ds.hour_weight.sum(1) != 24) == 8  # four springs, four falls


def test_dst_days_expand_to_real_hours(ds):
    spring, fall = ds.day_index(date(2026, 3, 8)), ds.day_index(date(2025, 11, 2))
    s = to_hourly(ds, ds.load[spring : spring + 1, :, 0], spring, spring + 1)
    f = to_hourly(ds, ds.load[fall : fall + 1, :, 0], fall, fall + 1)
    assert len(s) == 23 and not any("T02:" in ts for ts, _ in s)
    assert len(f) == 25 and len({ts for ts, _ in f}) == 25  # 01:00 appears at -05:00 and -06:00


def test_solar_follows_the_sun_not_the_clock(ds):
    jan, jul = ds.day_index(date(2026, 1, 15)), ds.day_index(date(2026, 7, 15))
    assert not ds.dst[jan] and ds.dst[jul]
    assert np.argmax(ds.clear_sky[jul]) == np.argmax(ds.clear_sky[jan]) + 1


def test_rooftop_solar_grows_and_cuts_midday_net_load(ds):
    assert ds.solar_cap[ds.today].sum() > 3 * ds.solar_cap[0].sum()
    late = slice(ds.n_hist - 365, ds.n_hist)
    assert ds.solar[late, 12].sum(1).mean() > 300
    assert ds.solar[late, 12].sum(1).mean() > 2.5 * ds.solar[:365, 12].sum(1).mean()


def test_data_center_grows(ds):
    assert ds.data_center[:30].mean() < 150 and ds.data_center[ds.n_hist - 30 : ds.n_hist].mean() > 500


def test_outage_ground_truth(ds):
    assert ds.outage.sum() == 34 and ds.outage[..., 0].sum() == 34  # north zone only
    k = ds.day_index(date(2026, 1, 21))
    assert ds.load[k, 6, 0] < 0.5 * ds.load[k - 7, 6, 0]


def test_events_shift_temperature(ds):
    heat = ds.day_index(date(2026, 7, 16))
    storm = ds.day_index(date(2026, 1, 21))
    assert ds.temp[heat].mean() > ds.temp[heat - 365].mean() + 3
    assert ds.temp[storm].mean() < ds.temp[storm - 365].mean() - 3


def test_forecast_error_grows_with_lead(ds):
    n = ds.n_hist - 8
    sd = [np.std(ds.temp_fc[:n, j] - ds.temp[j : n + j]) for j in range(8)]
    assert all(b > a for a, b in zip(sd[1:], sd[2:], strict=False))
    assert 0.8 < sd[1] < 1.6 and sd[7] > 2.5


def test_issue_day_hours_before_10_are_observed(ds):
    i = 500
    assert np.allclose(ds.temp_fc[i, 0, :ISSUE_HOUR], ds.temp[i, :ISSUE_HOUR], atol=1e-4)
    assert not np.allclose(ds.temp_fc[i, 0, ISSUE_HOUR:], ds.temp[i, ISSUE_HOUR:], atol=1e-2)


def test_school_calendar(ds):
    assert ds.school[ds.day_index(date(2026, 2, 10))]
    assert not ds.school[ds.day_index(date(2026, 7, 14))]
    assert not ds.school[ds.day_index(date(2025, 12, 29))]
