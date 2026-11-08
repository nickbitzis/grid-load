"""Results of the quick pipeline (820 days, 300 scored daily issues)."""

from gridload.backtest import COVERAGE_BAND, gate_check


def _row(art, model, level="system", horizon="day"):
    return next(r for r in art["backtest"]["main"] if (r["model"], r["level"], r["horizon"]) == (model, level, horizon))


def test_gate_passes(art):
    g = art["backtest"]["gate"]
    assert g["passed"] and g["checks"] == {"beats_baseline": True, "coverage_in_band": True}


def test_gbm_beats_every_baseline_day_and_week(art):
    for h in ("day", "week"):
        for lvl in ("zones", "system"):
            g = _row(art, "gbm", lvl, h)["mape"]
            for m in ("same_hour_last_week", "similar_days", "spline_regression"):
                assert g < _row(art, m, lvl, h)["mape"], (h, lvl, m)


def test_interval_coverage_near_80(art):
    for h in ("day", "week"):
        for lvl in ("zones", "system"):
            assert COVERAGE_BAND[0] <= _row(art, "gbm", lvl, h)["coverage"] <= COVERAGE_BAND[1]


def test_gate_fails_when_the_model_loses():
    main = [
        {"model": "gbm", "level": "system", "horizon": "day", "mape": 0.08, "coverage": 0.8},
        {"model": "gbm", "level": "zones", "horizon": "day", "mape": 0.08, "coverage": 0.8},
        {"model": "same_hour_last_week", "level": "system", "horizon": "day", "mape": 0.07, "coverage": None},
    ]
    assert not gate_check(main)["passed"]
    main[0]["mape"] = main[1]["mape"] = 0.05
    main[1]["coverage"] = 0.6
    g = gate_check(main)
    assert not g["passed"] and g["checks"]["beats_baseline"]


def test_week_ahead_is_harder_than_day_ahead(art):
    lead = [r["mape"] for r in art["backtest"]["by_lead"] if r["model"] == "gbm"]
    assert lead[0] < lead[-1]


def test_excluding_the_outage_helps_after_it(art):
    rows = {r["variant"]: r for r in art["backtest"]["ablations"]["anomaly"]["rows"]}
    assert rows["gbm_base"]["after_zone_mape"] < rows["gbm_no_anomaly_handling"]["after_zone_mape"]
    assert art["backtest"]["ablations"]["anomaly"]["zone"] == "north"


def test_perfect_weather_flatters_the_backtest(art):
    wx = {r["variant"]: r for r in art["backtest"]["ablations"]["weather"]}
    assert wx["gbm_oracle"]["week_mape"] < wx["gbm_base"]["week_mape"]


def test_reconciled_forecasts_add_up(art):
    rec = {r["method"]: r for r in art["backtest"]["reconciliation"]}
    assert rec["gbm"]["max_gap_mw"] < 1e-6 and rec["gbm_base"]["max_gap_mw"] > 10


def test_better_forecasts_cost_less(art):
    cost = {r["model"]: r["total_usd"] for r in art["backtest"]["economics"]}
    assert cost["gbm"] < cost["similar_days"] < cost["same_hour_last_week"]


def test_peak_alerts_beat_simple_policies(art):
    pa = art["backtest"]["peak_alerts"]
    assert pa["exceedances"] > 5
    assert pa["best"]["cost"] < min(pa["never_call"]["cost"], pa["always_call"]["cost"])
    assert pa["best"]["recall"] > 0.6


def test_season_and_extreme_breakdowns(art):
    bt = art["backtest"]
    assert {r["season"] for r in bt["by_season"]} == {"winter", "spring", "summer", "fall"}
    assert bt["extreme_days"] >= 30 and len(bt["extreme"]) == 8
