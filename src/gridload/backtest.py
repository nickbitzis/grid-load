"""Rolling-origin backtest: a forecast issued at 10:00 every day for the next 7 days, using only what was known
at that moment (load through 09:59, the weather forecast issued that morning).

Models are refit every `refit_every` days on the previous `train_days` of metered days. A warm-up period before
the scored window collects out-of-sample errors for conformal calibration, reconciliation weights, and the peak
alert's error distribution, so the first scored day already has honest inputs.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from . import alerts, conformal, economics, reconcile, scoring
from .features import NodeData, build, node_data, pairs, targets, training_pairs
from .generator import MAX_LEAD, Dataset
from .models import GBM, GBMConfig, SameHourLastWeek, SimilarDays, SplineRegression

MODELS = ("same_hour_last_week", "similar_days", "spline_regression", "gbm")
MODEL_NAMES = {
    "same_hour_last_week": "Same hour last week",
    "similar_days": "Similar days",
    "spline_regression": "Spline regression",
    "gbm": "Gradient boosting, reconciled",
}
SEASONS = {12: "winter", 1: "winter", 2: "winter", 3: "spring", 4: "spring", 5: "spring",
           6: "summer", 7: "summer", 8: "summer", 9: "fall", 10: "fall", 11: "fall"}  # fmt: skip
COVERAGE_BAND = (0.75, 0.85)


@dataclass
class BacktestConfig:
    test_days: int = 365
    warmup_days: int = 56
    refit_every: int = 28
    train_days: int = 730
    gbm: GBMConfig = field(default_factory=GBMConfig)
    ablations: bool = True
    seed: int = 0
    costs: alerts.AlertCosts = field(default_factory=alerts.AlertCosts)
    prices: economics.Prices = field(default_factory=economics.Prices)


@dataclass
class Fitted:
    point: GBM
    lo: GBM
    hi: GBM


def fit_gbms(X, y, cfg: GBMConfig, seed: int = 0) -> Fitted:
    return Fitted(GBM(cfg, seed=seed).fit(X, y), GBM(cfg, 0.1, seed).fit(X, y), GBM(cfg, 0.9, seed).fit(X, y))


def training_set(ds: Dataset, nd: NodeData, first_t: int, last_t: int, rng, weather: str = "forecast"):
    ti, tj = training_pairs(first_t, last_t, rng)
    X, sc = build(ds, nd, ti, tj, weather)
    y, ok = targets(nd, ti, tj)
    return X[ok], (y / sc)[ok]


def _grid(v: np.ndarray, n_issue: int, m: int) -> np.ndarray:
    return v.reshape(n_issue, MAX_LEAD, 24, m)


def run(ds: Dataset, cfg: BacktestConfig | None = None, log=print) -> dict:
    cfg = cfg or BacktestConfig()
    t0 = time.perf_counter()
    nd = node_data(ds, handle_anomalies=True)
    nd_raw = node_data(ds, handle_anomalies=False, flags=nd.flags[..., :-1])
    M = nd.m
    last_issue = ds.n_hist - 2
    first_test = last_issue - cfg.test_days + 1
    first_issue = first_test - cfg.warmup_days
    issues = np.arange(first_issue, last_issue + 1)
    n_iss = issues.size
    leads = np.arange(1, MAX_LEAD + 1)
    shape = (n_iss, MAX_LEAD, 24, M)
    pred = {k: np.full(shape, np.nan) for k in ("same_hour_last_week", "similar_days", "spline_regression")}
    for k in ("gbm_base", "gbm_lo", "gbm_hi", "gbm_oracle", "gbm_train_actual", "gbm_no_anomaly_handling"):
        pred[k] = np.full(shape, np.nan)
    rng = np.random.default_rng(cfg.seed)
    sim = SimilarDays()
    refits = 0
    for b0 in range(0, n_iss, cfg.refit_every):
        blk = issues[b0 : b0 + cfg.refit_every]
        cut = blk[0]  # train on target days that ended before this issue day
        first_t = max(cut - cfg.train_days, 16)
        X_tr, y_tr = training_set(ds, nd, first_t, cut - 1, rng)
        f = fit_gbms(X_tr, y_tr, cfg.gbm, cfg.seed)
        ridge = SplineRegression().fit(X_tr, y_tr)
        pi, pj = pairs(blk, leads)
        Xp, sc = build(ds, nd, pi, pj)
        sl = slice(b0, b0 + blk.size)
        n = blk.size
        pred["same_hour_last_week"][sl] = _grid(SameHourLastWeek().predict(Xp) * sc, n, M)
        pred["spline_regression"][sl] = _grid(ridge.predict(Xp) * sc, n, M)
        pred["similar_days"][sl] = sim.predict_mw(ds, nd, pi, pj).reshape(n, MAX_LEAD, 24, M)
        pred["gbm_base"][sl] = _grid(f.point.predict(Xp) * sc, n, M)
        pred["gbm_lo"][sl] = _grid(f.lo.predict(Xp) * sc, n, M)
        pred["gbm_hi"][sl] = _grid(f.hi.predict(Xp) * sc, n, M)
        if cfg.ablations:
            Xa, ya = training_set(ds, nd, first_t, cut - 1, np.random.default_rng(cfg.seed + refits), "actual")
            ga = GBM(cfg.gbm, seed=cfg.seed).fit(Xa, ya)
            Xpa, _ = build(ds, nd, pi, pj, "actual")
            pred["gbm_oracle"][sl] = _grid(ga.predict(Xpa) * sc, n, M)
            pred["gbm_train_actual"][sl] = _grid(ga.predict(Xp) * sc, n, M)
            Xr, yr = training_set(ds, nd_raw, first_t, cut - 1, np.random.default_rng(cfg.seed + refits))
            gr = GBM(cfg.gbm, seed=cfg.seed).fit(Xr, yr)
            Xpr, scr = build(ds, nd_raw, pi, pj)
            pred["gbm_no_anomaly_handling"][sl] = _grid(gr.predict(Xpr) * scr, n, M)
        refits += 1
        log(f"refit {refits} at {ds.dates[cut]} on {len(y_tr):,} rows ({time.perf_counter() - t0:.0f}s)")

    # Actuals and scoring weights on the same grid.
    t_idx = issues[:, None] + leads[None, :]
    actual = nd.actual[t_idx]
    ok = np.isfinite(actual) & ~nd.flags[t_idx]
    weight = ds.hour_weight[t_idx][..., None] * ok

    # Reconciliation weights: day-ahead error variance per node over the warm-up window.
    warm = slice(0, cfg.warmup_days)
    err = (pred["gbm_base"][warm, 0] - actual[warm, 0]) ** 2
    variances = np.array([np.nansum(np.where(ok[warm, 0, :, m], err[..., m], 0)) for m in range(M)])
    variances = variances / np.maximum(ok[warm, 0].sum(axis=(0, 1)), 1)
    rec = reconcile.reconcile(pred["gbm_base"], "wls", variances)
    pred["gbm"] = rec
    pred["gbm_bottom_up"] = reconcile.reconcile(pred["gbm_base"], "bottom_up")
    shift = rec - pred["gbm_base"]
    lo = np.minimum(pred["gbm_lo"] + shift, rec)
    hi = np.maximum(pred["gbm_hi"] + shift, rec)
    groups = [np.arange(M - 1), np.array([M - 1])]
    lo_c, hi_c = conformal.calibrate(issues, actual, lo, hi, rec, ok, groups)
    pred["gbm_p10"], pred["gbm_p90"] = lo_c, hi_c
    pred["gbm_p10_raw"], pred["gbm_p90_raw"] = lo, hi

    test = issues >= first_test
    res = evaluate(ds, nd, cfg, issues, test, pred, actual, weight)
    threshold = peak_threshold(nd, int(issues[test][0]))
    res["peak_alerts"] = peak_alerts(issues, test, pred["gbm"], actual, ok, threshold, cfg.costs)
    res["config"] = {
        "test_days": cfg.test_days,
        "warmup_days": cfg.warmup_days,
        "refit_every": cfg.refit_every,
        "train_days": cfg.train_days,
        "refits": refits,
        "first_issue": ds.dates[first_test].isoformat(),
        "last_issue": ds.dates[last_issue].isoformat(),
        "seconds": round(time.perf_counter() - t0, 1),
    }
    res["reconciliation_variances"] = variances.tolist()
    res["_arrays"] = {"issues": issues, "pred": pred, "actual": actual, "ok": ok, "weight": weight}
    return res


def _sel(arr, test, lead_idx, nodes):
    return arr[test][:, lead_idx][..., nodes]


def metric_row(y, f, w, lo=None, hi=None) -> dict:
    """y, f, w: (..., 24, nodes) for one horizon slice."""
    row = {
        "mape": scoring.mape(y, f, w),
        "rmse": scoring.rmse(y, f, w),
        "bias": scoring.bias(y, f, w),
        "peak_error": scoring.peak_error(np.moveaxis(y, -1, -2), np.moveaxis(f, -1, -2), np.moveaxis(w, -1, -2)),
        "pinball": None,
        "coverage": None,
    }
    if lo is not None:
        row["pinball"] = scoring.mean_pinball(y, {0.1: lo, 0.5: f, 0.9: hi}, w) / float(np.nanmean(y))
        row["coverage"] = scoring.coverage(y, lo, hi, w)
    return row


HORIZONS = {"day": [0], "week": list(range(MAX_LEAD))}


def evaluate(ds, nd, cfg, issues, test, pred, actual, weight) -> dict:
    M = nd.m
    zones, system = list(range(M - 1)), [M - 1]
    levels = {"zones": zones, "system": system}
    main = []
    for h, li in HORIZONS.items():
        for lvl, nodes in levels.items():
            y, w = _sel(actual, test, li, nodes), _sel(weight, test, li, nodes)
            for m in MODELS:
                f = _sel(pred[m], test, li, nodes)
                q = (
                    (_sel(pred["gbm_p10"], test, li, nodes), _sel(pred["gbm_p90"], test, li, nodes))
                    if m == "gbm"
                    else ()
                )
                main.append({"horizon": h, "level": lvl, "model": m, **metric_row(y, f, w, *q)})

    # Day-ahead MAPE by season, for every node and model (the UI's grouped bars).
    t_days = issues[test] + 1
    season = np.array([SEASONS[ds.month[t]] for t in t_days])
    by_season = []
    for m_i, node in enumerate(nd.ids):
        for m in MODELS:
            for s in ("winter", "spring", "summer", "fall"):
                sel = season == s
                if not sel.any():
                    continue
                y = actual[test][sel][:, 0, :, m_i]
                f = pred[m][test][sel][:, 0, :, m_i]
                w = weight[test][sel][:, 0, :, m_i]
                by_season.append({"node": node, "model": m, "season": s, "mape": scoring.mape(y, f, w)})

    # Extreme days: top 10% of system peaks in the window, plus the named weather events.
    sys_actual = actual[test][:, 0, :, -1]
    peaks = np.nanmax(np.where(weight[test][:, 0, :, -1] > 0, sys_actual, np.nan), axis=1)
    event_days = set()
    for e in ds.events:
        e0 = ds.day_index(e.start)
        event_days |= set(range(e0, e0 + e.days))
    extreme = (peaks >= np.nanpercentile(peaks, 90)) | np.isin(t_days, list(event_days))
    ext = []
    for lvl, nodes in levels.items():
        for m in MODELS:
            y = actual[test][extreme][:, [0]][..., nodes]
            w = weight[test][extreme][:, [0]][..., nodes]
            f = pred[m][test][extreme][:, [0]][..., nodes]
            q = ()
            if m == "gbm":
                q = (
                    pred["gbm_p10"][test][extreme][:, [0]][..., nodes],
                    pred["gbm_p90"][test][extreme][:, [0]][..., nodes],
                )
            ext.append({"level": lvl, "model": m, **metric_row(y, f, w, *q)})

    # MAPE by lead day.
    by_lead = []
    for j in range(MAX_LEAD):
        for m in MODELS:
            by_lead.append(
                {
                    "lead": j + 1,
                    "model": m,
                    "mape": scoring.mape(_sel(actual, test, [j], zones), _sel(pred[m], test, [j], zones),
                                         _sel(weight, test, [j], zones)),
                }
            )  # fmt: skip

    # Weather inputs and anomaly handling (gradient boosting before reconciliation, zone level).
    ablations = {}
    if cfg.ablations:
        wx = []
        for key, label in (
            ("gbm_oracle", "Trained and tested on actual weather (what a leaky backtest reports)"),
            ("gbm_train_actual", "Trained on actual weather, tested on forecast weather"),
            ("gbm_base", "Trained and tested on forecast weather (honest)"),
        ):
            row = {"variant": key, "label": label}
            for h, li in HORIZONS.items():
                row[f"{h}_mape"] = scoring.mape(_sel(actual, test, li, zones), _sel(pred[key], test, li, zones),
                                                _sel(weight, test, li, zones))  # fmt: skip
            wx.append(row)
        ablations["weather"] = wx
        ablations["anomaly"] = anomaly_effect(ds, nd, issues, test, pred, actual, weight)
    # Reconciliation at system and zone level.
    recon = []
    for key, label in (("gbm_base", "Base forecasts (not coherent)"), ("gbm_bottom_up", "Bottom-up"), ("gbm", "WLS")):
        recon.append(
            {
                "method": key,
                "label": label,
                "zones_mape": scoring.mape(_sel(actual, test, [0], zones), _sel(pred[key], test, [0], zones),
                                           _sel(weight, test, [0], zones)),
                "system_mape": scoring.mape(_sel(actual, test, [0], system), _sel(pred[key], test, [0], system),
                                            _sel(weight, test, [0], system)),
                "max_gap_mw": float(np.nanmax(np.abs(pred[key][test][..., :-1].sum(-1) - pred[key][test][..., -1]))),
            }
        )  # fmt: skip

    # Imbalance cost of each model's day-ahead system forecast.
    econ = []
    n_test = int(test.sum())
    y = actual[test][:, 0, :, -1]
    w = ds.hour_weight[issues[test] + 1]
    for m in MODELS:
        c = economics.imbalance_cost(pred[m][test][:, 0, :, -1], y, w, cfg.prices)
        econ.append({"model": m, **{k: v * 365 / n_test for k, v in c.items()}})

    gate = gate_check(main)
    return {
        "models": {m: MODEL_NAMES[m] for m in MODELS},
        "main": main,
        "by_season": by_season,
        "extreme": ext,
        "extreme_days": int(extreme.sum()),
        "by_lead": by_lead,
        "ablations": ablations,
        "reconciliation": recon,
        "economics": econ,
        "gate": gate,
    }


def anomaly_effect(ds, nd, issues, test, pred, actual, weight) -> dict:
    """Error with and without outage handling, over the whole window and the two weeks after the outage."""
    flagged_days = np.where(nd.flags[..., :-1].any(axis=(1, 2)))[0]
    out = {"flagged_cells": int(nd.flags[..., :-1].sum()), "rows": []}
    windows = {"window": np.ones(issues.size, bool) & test}
    if flagged_days.size:
        end = flagged_days.max()
        windows["after"] = test & (issues >= end) & (issues <= end + 14)
        out["after_start"] = ds.dates[end].isoformat()
    zone = int(np.argmax(nd.flags[..., :-1].sum(axis=(0, 1))))  # the zone that lost load
    for key, label in (("gbm_no_anomaly_handling", "Outage left in"), ("gbm_base", "Outage detected and excluded")):
        row = {"variant": key, "label": label}
        for wname, sel in windows.items():
            if not sel.any():
                continue
            for lvl, nodes in (("zone", [zone]), ("system", [nd.m - 1])):
                row[f"{wname}_{lvl}_mape"] = scoring.mape(
                    actual[sel][..., nodes], pred[key][sel][..., nodes], weight[sel][..., nodes]
                )
        out["rows"].append(row)
    out["zone"] = nd.ids[zone]
    return out


def gate_check(main: list[dict]) -> dict:
    def get(model, level="system", horizon="day"):
        return next(r for r in main if r["model"] == model and r["level"] == level and r["horizon"] == horizon)

    gbm_sys, naive_sys = get("gbm"), get("same_hour_last_week")
    cov = get("gbm", "zones")["coverage"]
    checks = {
        "beats_baseline": gbm_sys["mape"] < naive_sys["mape"],
        "coverage_in_band": COVERAGE_BAND[0] <= cov <= COVERAGE_BAND[1],
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "gbm_day_mape": gbm_sys["mape"],
        "baseline_day_mape": naive_sys["mape"],
        "coverage": cov,
        "band": list(COVERAGE_BAND),
    }


def peak_threshold(nd: NodeData, before: int) -> float:
    """DR trigger: the 92nd percentile of daily system peaks over the year before `before`, rounded to 50 MW."""
    sys_load = nd.actual[max(0, before - 365) : before, :, -1]
    return float(np.round(np.nanpercentile(np.nanmax(sys_load, axis=1), 92) / 50) * 50)


def daily_peaks(rec_sys: np.ndarray, actual_sys: np.ndarray, ok_sys: np.ndarray):
    pred_peak = np.nanmax(rec_sys, axis=-1)
    act_peak = np.nanmax(np.where(np.isfinite(actual_sys), actual_sys, -np.inf), axis=-1)
    full = ok_sys.sum(-1) >= 22
    return pred_peak, np.where(full, act_peak, np.nan)


def peak_alerts(issues, test, rec, actual, ok, threshold: float, costs: alerts.AlertCosts, window: int = 120) -> dict:
    """Probability that tomorrow's system peak exceeds the DR threshold, for every issue in the backtest, then the
    precision, recall, and realized cost of calling an event whenever the probability clears a cut."""
    pred_peak, act_peak = daily_peaks(rec[:, 0, :, -1], actual[:, 0, :, -1], ok[:, 0, :, -1])
    log_r = np.log(act_peak / pred_peak)
    probs = np.full(issues.size, np.nan)
    for a in range(issues.size):
        past = log_r[max(0, a - window) : max(a - 1, 0)]  # day-ahead targets that ended before issue a
        probs[a] = alerts.exceed_probability(pred_peak[a], past, threshold)
    sel = test & np.isfinite(act_peak)
    exceeded = act_peak[sel] > threshold
    curve = alerts.cost_curve(probs[sel], exceeded, costs)
    be = alerts.break_even(costs)
    return {
        "threshold_mw": threshold,
        "days": int(sel.sum()),
        "exceedances": int(exceeded.sum()),
        "costs": {"call_usd": costs.call_usd, "miss_usd": costs.miss_usd},
        "break_even": be,
        "best": alerts.best_cut(curve, costs),
        "at_break_even": alerts.realized_cost(probs[sel], exceeded, be, costs),
        "never_call": alerts.realized_cost(probs[sel], exceeded, 1.01, costs),
        "always_call": alerts.realized_cost(probs[sel], exceeded, 0.0, costs),
        "curve": [
            {"cut": r["cut"], "cost": r["cost"], "precision": r["precision"], "recall": r["recall"]} for r in curve
        ],
        "brier": float(np.mean((probs[sel] - exceeded) ** 2)),
        "log_ratios": log_r[np.isfinite(log_r)][-window:].tolist(),
    }
