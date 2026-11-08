"""End to end: generate, backtest, retrain on everything metered, issue today's 10:00 forecast, and write
artifacts.json for the service."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import timedelta
from pathlib import Path

import numpy as np

from . import alerts, anomaly, conformal, reconcile
from .backtest import BacktestConfig, fit_gbms, run, training_set
from .features import build, node_data, pairs
from .generator import ISSUE_HOUR, MAX_LEAD, Dataset, GenConfig, generate, to_hourly
from .models import GBMConfig

UTILITY = "Prairie Line Power"


@dataclass
class PipelineConfig:
    gen: GenConfig = field(default_factory=GenConfig)
    backtest: BacktestConfig = field(default_factory=BacktestConfig)
    history_days: int = 7

    @classmethod
    def quick(cls) -> PipelineConfig:
        """About 820 days of history and a 300-day backtest: 15 to 20 seconds. Used by tests, CI, and Docker."""
        return cls(
            gen=GenConfig(days=820),
            backtest=BacktestConfig(
                test_days=300,
                warmup_days=42,
                refit_every=114,
                train_days=450,
                gbm=GBMConfig(max_iter=80, quantile_iter=60),
            ),
        )


def series(ds: Dataset, day0: int, arrays: dict[str, np.ndarray]) -> list[dict]:
    """Zip grid arrays of shape (days, 24) into real clock hours: [{"ts": ..., name: value}, ...]."""
    days = next(iter(arrays.values())).shape[0]
    cols = {k: to_hourly(ds, v, day0, day0 + days) for k, v in arrays.items()}
    out = []
    for k in range(len(next(iter(cols.values())))):
        row = {"ts": next(iter(cols.values()))[k][0]}
        for name, vals in cols.items():
            v = vals[k][1]
            row[name] = None if not np.isfinite(v) else round(v, 1)
        out.append(row)
    return out


def build_artifacts(cfg: PipelineConfig | None = None, log=print) -> dict:
    cfg = cfg or PipelineConfig()
    t0 = time.perf_counter()
    ds = generate(cfg.gen)
    log(f"generated {ds.n_hist} days x 24 hours x {len(ds.zones)} zones ({time.perf_counter() - t0:.1f}s)")
    bt = run(ds, cfg.backtest, log=log)
    arr = bt.pop("_arrays")
    nd = node_data(ds)
    M = nd.m
    today = ds.today
    issues, pred, actual, ok = arr["issues"], arr["pred"], arr["actual"], arr["ok"]

    # Live forecast issued today at 10:00, trained on every metered day.
    bc = cfg.backtest
    X_tr, y_tr = training_set(ds, nd, max(today - bc.train_days, 16), today - 1, np.random.default_rng(bc.seed + 99))
    f = fit_gbms(X_tr, y_tr, bc.gbm, bc.seed)
    leads = np.arange(1, MAX_LEAD + 1)
    pi, pj = pairs([today], leads)
    Xp, sc = build(ds, nd, pi, pj)
    shape = (MAX_LEAD, 24, M)
    base = (f.point.predict(Xp) * sc).reshape(shape)
    p50 = reconcile.reconcile(base, "wls", np.array(bt["reconciliation_variances"]))
    shift = p50 - base
    lo = np.minimum((f.lo.predict(Xp) * sc).reshape(shape) + shift, p50)
    hi = np.maximum((f.hi.predict(Xp) * sc).reshape(shape) + shift, p50)
    # Conformal widening from the last 90 backtest issues, by lead bucket and level, as in the backtest.
    recent = slice(max(0, issues.size - 90), issues.size)
    s = conformal.scores(actual[recent], pred["gbm_p10_raw"][recent], pred["gbm_p90_raw"][recent], pred["gbm"][recent])
    s = np.where(ok[recent], s, np.nan)
    p10, p90 = lo.copy(), hi.copy()
    for g in (np.arange(M - 1), np.array([M - 1])):
        for b in (np.array([0]), np.arange(1, MAX_LEAD)):
            q = conformal.conformal_quantile(s[:, b][..., g].reshape(-1))
            for j in b:
                p10[j][:, g] = lo[j][:, g] - q * p50[j][:, g]
                p90[j][:, g] = hi[j][:, g] + q * p50[j][:, g]

    # Peak alert for tomorrow.
    pa = bt["peak_alerts"]
    w_tom = ds.hour_weight[today + 1]
    sys_tom = np.where(w_tom > 0, p50[0, :, -1], np.nan)
    pred_peak = float(np.nanmax(sys_tom))
    peak_hour = int(np.nanargmax(sys_tom))
    prob = alerts.exceed_probability(pred_peak, np.array(pa["log_ratios"]), pa["threshold_mw"])
    cut = pa["best"]["cut"]
    call = prob >= cut
    costs = bc.costs
    pa["tomorrow"] = {
        "date": ds.dates[today + 1].isoformat(),
        "probability": prob,
        "predicted_peak_mw": round(pred_peak, 1),
        "p90_peak_mw": round(float(np.nanmax(np.where(w_tom > 0, p90[0, :, -1], np.nan))), 1),
        "peak_hour": peak_hour,
        "cut": cut,
        "call": bool(call),
        "expected_cost_call": costs.call_usd,
        "expected_cost_no_call": prob * costs.miss_usd,
        "action": (
            f"Call a demand response event for {peak_hour - 2:02d}:00 to {peak_hour + 2:02d}:00."
            if call
            else "No demand response call. Keep monitoring the 16:00 update."
        ),
    }
    pa.pop("log_ratios")

    # History, live forecast, and backtest replays per node.
    h0 = today - cfg.history_days + 1
    hist = nd.actual[h0 : today + 1].copy()
    hist[-1, ISSUE_HOUR:] = np.nan  # metered through 09:59 today
    nodes = {}
    for m, nid in enumerate(nd.ids):
        nodes[nid] = {
            "history": series(ds, h0, {"actual": hist[..., m]}),
            "forecast": series(ds, today + 1, {"p10": p10[..., m], "p50": p50[..., m], "p90": p90[..., m]}),
        }
    replays = []
    for e in ds.events:
        t0e, t1e = ds.day_index(e.start) - 2, ds.day_index(e.start) + e.days + 2
        a0, a1 = t0e - 1 - issues[0], t1e - 1 - issues[0]
        if a0 < 0 or a1 > issues.size:
            continue
        rep = {"kind": e.kind, "name": e.name, "start": e.start.isoformat(), "days": e.days, "nodes": {}}
        for m, nid in enumerate(nd.ids):
            flags = nd.flags[t0e:t1e, :, m]
            act = actual[a0:a1, 0, :, m]
            rep["nodes"][nid] = series(
                ds,
                t0e,
                {
                    "actual": act,
                    "p10": pred["gbm_p10"][a0:a1, 0, :, m],
                    "p50": pred["gbm"][a0:a1, 0, :, m],
                    "p90": pred["gbm_p90"][a0:a1, 0, :, m],
                    "flagged": np.where(flags, act, np.nan),
                },
            )
        replays.append(rep)

    zones = [{"id": z.id, "name": z.name, "base_mw": z.base_mw, "data_center": z.id == "east"} for z in ds.zones] + [
        {"id": "system", "name": "System total", "base_mw": None, "data_center": False}
    ]
    out = {
        "utility": UTILITY,
        "generated_seconds": round(time.perf_counter() - t0, 1),
        "today": ds.dates[today].isoformat(),
        "issued": f"{ds.dates[today].isoformat()}T{ISSUE_HOUR:02d}:00",
        "tomorrow": ds.dates[today + 1].isoformat(),
        "week_end": (ds.dates[today] + timedelta(days=MAX_LEAD)).isoformat(),
        "zones": zones,
        "nodes": nodes,
        "replays": replays,
        "anomalies": anomaly.runs(nd.flags[: ds.n_hist, :, :-1], ds.dates, ds.zone_ids),
        "events": [{"kind": e.kind, "name": e.name, "start": e.start.isoformat(), "days": e.days} for e in ds.events],
        "backtest": bt,
        "data": {
            "days": ds.n_hist,
            "first_day": ds.dates[0].isoformat(),
            "system_peak_mw": round(float(np.nanmax(nd.actual[: ds.n_hist, :, -1])), 1),
            "data_center_mw_now": round(float(ds.data_center[today].mean()), 1),
            "solar_capacity_mw_now": round(float(ds.solar_cap[today].sum()), 1),
            "solar_capacity_mw_start": round(float(ds.solar_cap[0].sum()), 1),
        },
    }
    log(f"done in {time.perf_counter() - t0:.0f}s, gate {'passed' if bt['gate']['passed'] else 'FAILED'}")
    return out


def _default(o):
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    raise TypeError(type(o))


def roundtrip(art: dict) -> dict:
    return json.loads(json.dumps(art, default=_default))


def save(art: dict, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(art, default=_default, separators=(",", ":")))


def load(path: Path) -> dict:
    return json.loads(path.read_text())
