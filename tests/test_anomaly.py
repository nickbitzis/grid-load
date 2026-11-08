import numpy as np

from gridload import anomaly


def test_flag_runs_min_length_and_padding():
    low = np.array([0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], bool)
    out = anomaly.flag_runs(low, min_run=4, pad=2)
    assert out.tolist() == [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0]


def test_daylight_only_dips_are_solar_not_outages():
    low = np.zeros(48, bool)
    low[11:17] = True
    dark = np.ones(48, bool)
    dark[7:19] = False
    assert not anomaly.flag_runs(low, 4, 0, dark).any()
    low[20:28] = True
    assert anomaly.flag_runs(low, 4, 0, dark)[20:28].all()


def test_trailing_median_never_looks_ahead():
    load = np.ones((30, 24, 1))
    load[20:] = 50
    med = anomaly.trailing_median(load)
    assert np.all(med[1:21] == 1) and med[21, 0, 0] == 1


def test_detector_finds_exactly_the_outage(ds):
    raw = ds.load[: ds.n_hist]
    flags, _ = anomaly.detect(raw, dark=ds.clear_sky[: ds.n_hist] == 0)
    truth = ds.outage[: ds.n_hist]
    assert (flags & ~truth).sum() == 0
    assert (flags & truth).sum() >= 0.9 * truth.sum()


def test_clean_imputes_flagged_cells(ds, nd):
    k = ds.day_index(__import__("datetime").date(2026, 1, 21))
    assert nd.flags[k, 6, 0]
    assert nd.load[k, 6, 0] > 2 * nd.actual[k, 6, 0]
    assert nd.load[k, 6, 1] == nd.actual[k, 6, 1]  # other zones untouched


def test_runs_summary(ds, nd):
    runs = anomaly.runs(nd.flags[: ds.n_hist, :, :-1], ds.dates, ds.zone_ids)
    assert runs == [{"zone": "north", "start": "2026-01-21 02:00", "hours": 32}]
