"""Leakage-safe features for a forecast issued at 10:00 on day i for day t = i + j (j = 1..7).

Nodes are the five zones plus the system total (index 5). Rules, enforced by index arithmetic and checked by a
perturbation test:
- Load is available for every hour before 10:00 on day i, and nothing after.
- Weather comes from the forecast issued on day i (`weather="forecast"`), never from actuals, unless the caller
  asks for the oracle (`weather="actual"`) to measure what perfect weather would be worth.
- Calendar facts (holidays, school terms, DST) and installed solar capacity are known in advance.

The target is load divided by the node's mean over the last 168 available hours, so one pooled model serves
zones from 640 MW to the 6 GW system, and growth (the data center, rooftop solar) moves the scale, not the trees.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import anomaly
from .generator import ISSUE_HOUR, MAX_LEAD, Dataset

SYSTEM = "system"
FEATURES = [
    "node", "hour", "dow", "month", "doy", "holiday", "bridge", "school", "dst", "lead",
    "temp", "hum", "tmax", "tmin", "tmean", "tmean_prev", "cloud", "solar_clear",
    "lag_week", "lag_day", "recent24", "prev_day", "daytype", "analog",
]  # fmt: skip
COL = {name: k for k, name in enumerate(FEATURES)}
MIN_ISSUE = 15  # first issue day with two weeks of history behind it


@dataclass
class NodeData:
    """Per-node arrays (N, 24, M) with M = zones + system."""

    ids: tuple[str, ...]
    load: np.ndarray  # lag source: cleaned (or raw when anomaly handling is off)
    actual: np.ndarray  # metered load, for scoring
    flags: np.ndarray  # anomalous cells (excluded from training targets and scoring)
    temp: np.ndarray
    hum: np.ndarray
    temp_fc: np.ndarray  # (N, L, 24, M)
    hum_fc: np.ndarray
    solar_cap: np.ndarray  # (N, M)

    @property
    def m(self) -> int:
        return len(self.ids)


def zone_weights(ds: Dataset) -> np.ndarray:
    w = np.array([z.base_mw for z in ds.zones])
    return w / w.sum()


def node_data(ds: Dataset, handle_anomalies: bool = True, flags: np.ndarray | None = None) -> NodeData:
    raw = ds.load.copy()
    raw[ds.n_hist :] = np.nan  # the future is never metered
    raw[ds.today, ISSUE_HOUR:] = np.nan  # today is metered through 09:59
    if flags is None:
        flags, expected = anomaly.detect(raw, dark=ds.clear_sky == 0)
    else:
        expected = anomaly.trailing_median(raw)
    src = anomaly.clean(raw, flags, expected) if handle_anomalies else raw
    w = zone_weights(ds)

    def add_sys(a: np.ndarray, how: str) -> np.ndarray:
        s = a.sum(axis=-1, keepdims=True) if how == "sum" else (a * w).sum(axis=-1, keepdims=True)
        return np.concatenate([a, s], axis=-1)

    sys_flags = flags.any(axis=-1, keepdims=True)
    return NodeData(
        ids=(*ds.zone_ids, SYSTEM),
        load=add_sys(src, "sum"),
        actual=add_sys(raw, "sum"),
        flags=np.concatenate([flags, sys_flags], axis=-1)
        if handle_anomalies
        else np.zeros_like(add_sys(raw, "sum"), bool),
        temp=add_sys(ds.temp, "mean"),
        hum=add_sys(ds.hum, "mean"),
        temp_fc=add_sys(ds.temp_fc.astype(float), "mean"),
        hum_fc=add_sys(ds.hum_fc.astype(float), "mean"),
        solar_cap=add_sys(ds.solar_cap, "sum"),
    )


def daytype(ds: Dataset, t: np.ndarray) -> np.ndarray:
    """0 weekday, 1 Saturday or bridge day, 2 Sunday or holiday."""
    dt = np.zeros(t.shape, dtype=int)
    dt[(ds.dow[t] == 5) | ds.bridge[t]] = 1
    dt[(ds.dow[t] == 6) | ds.holiday[t]] = 2
    return dt


def scale7(nd: NodeData, issues: np.ndarray) -> np.ndarray:
    """Mean load over the 168 hours before 10:00 on each issue day. Returns (P, M)."""
    i = issues
    total = nd.load[i - 7, ISSUE_HOUR:].sum(1) + nd.load[i, :ISSUE_HOUR].sum(1)
    for k in range(1, 7):
        total = total + nd.load[i - k].sum(1)
    return total / 168.0


def pairs(issues, leads) -> tuple[np.ndarray, np.ndarray]:
    ii, jj = np.meshgrid(np.asarray(issues), np.asarray(leads), indexing="ij")
    return ii.reshape(-1), jj.reshape(-1)


def build(
    ds: Dataset, nd: NodeData, issues: np.ndarray, leads: np.ndarray, weather: str = "forecast"
) -> tuple[np.ndarray, np.ndarray]:
    """Feature rows for each (issue, lead) pair x 24 hours x M nodes, in that order.

    Returns X (P*24*M, F) and the scale (P*24*M,) to multiply predictions by."""
    i = np.asarray(issues, dtype=int)
    j = np.asarray(leads, dtype=int)
    if np.any(j < 1) or np.any(j > MAX_LEAD):
        raise ValueError("leads must be 1..7 days")
    if np.any(i < MIN_ISSUE):
        raise ValueError(f"issue days must be >= {MIN_ISSUE}")
    t = i + j
    P, M = i.size, nd.m
    h = np.arange(24)
    sc = scale7(nd, i)  # (P, M)

    # Same hour last week, stepping back another week if that hour is not metered yet (j = 7, h >= 10).
    lw_day = t[:, None] - 7 + np.zeros((1, 24), int)
    late = (lw_day > i[:, None]) | ((lw_day == i[:, None]) & (h[None, :] >= ISSUE_HOUR))
    lw_day = np.where(late, lw_day - 7, lw_day)
    lag_week = nd.load[lw_day, h[None, :]]  # (P, 24, M)
    ld_day = np.where(h[None, :] < ISSUE_HOUR, i[:, None], i[:, None] - 1)
    lag_day = nd.load[ld_day, h[None, :]]
    recent24 = (nd.load[i - 1, ISSUE_HOUR:].sum(1) + nd.load[i, :ISSUE_HOUR].sum(1)) / 24.0  # (P, M)
    prev_day = nd.load[i - 1].mean(1)

    if weather == "forecast":
        temp = nd.temp_fc[i, j]  # (P, 24, M)
        hum = nd.hum_fc[i, j]
        prev = nd.temp_fc[i, j - 1].mean(1)
        cloud = ds.cloud_fc[i, j].astype(float)
    elif weather == "actual":
        temp = nd.temp[t]
        hum = nd.hum[t]
        prev = nd.temp[t - 1].mean(1)
        cloud = ds.cloud[t]
    else:
        raise ValueError(weather)

    analog = similar_days_mw(ds, nd, i, j)
    solar = nd.solar_cap[i][:, None, :] * ds.clear_sky[t][:, :, None] * (1 - 0.72 * cloud)[:, None, None]
    s3 = sc[:, None, :]
    cols = {
        "node": np.broadcast_to(np.arange(M)[None, None, :], (P, 24, M)),
        "hour": np.broadcast_to(h[None, :, None], (P, 24, M)),
        "dow": ds.dow[t],
        "month": ds.month[t],
        "doy": ds.doy[t],
        "holiday": ds.holiday[t],
        "bridge": ds.bridge[t],
        "school": ds.school[t],
        "dst": ds.dst[t],
        "lead": j,
        "temp": temp,
        "hum": hum,
        "tmax": temp.max(1, keepdims=True),
        "tmin": temp.min(1, keepdims=True),
        "tmean": temp.mean(1, keepdims=True),
        "tmean_prev": prev[:, None, :],
        "cloud": cloud,
        "solar_clear": solar / s3,
        "lag_week": lag_week / s3,
        "lag_day": lag_day / s3,
        "recent24": (recent24 / sc)[:, None, :],
        "prev_day": (prev_day / sc)[:, None, :],
        "daytype": daytype(ds, t),
        "analog": analog / s3,
    }
    X = np.empty((P, 24, M, len(FEATURES)), dtype=np.float64)
    for k, name in enumerate(FEATURES):
        v = np.asarray(cols[name], dtype=float)
        while v.ndim < 3:
            v = v[..., None]
        X[..., k] = np.broadcast_to(v, (P, 24, M))
    scale = np.broadcast_to(s3, (P, 24, M)).reshape(-1)
    return X.reshape(-1, len(FEATURES)), scale.copy()


def similar_days_mw(ds: Dataset, nd: NodeData, issues, leads, k: int = 3, lookback: int = 56) -> np.ndarray:
    """Analog forecast: average profile of the k complete days in the last `lookback` days with the target's day
    type and the closest daily mean temperature to the target's forecast. Returns (P, 24, M) MW."""
    i, j = np.asarray(issues), np.asarray(leads)
    t = i + j
    dtype_t = daytype(ds, t)
    dtype_all = daytype(ds, np.arange(len(ds.dates)))
    tmean_actual = nd.temp.mean(1)  # (N, M), only read for days before the issue
    offs = np.arange(1, lookback + 1)
    cand = i[:, None] - offs[None, :]  # (P, C) days i-1 .. i-lookback, all complete at issue time
    cand_ok = (cand >= 1) & (dtype_all[np.clip(cand, 0, None)] == dtype_t[:, None])
    cand = np.clip(cand, 0, None)
    target_t = nd.temp_fc[i, j].mean(1)  # (P, M)
    dist = np.abs(tmean_actual[cand] - target_t[:, None, :])  # (P, C, M)
    dist = np.where(cand_ok[..., None], dist, np.inf)
    # fall back to the last 7 days when a day type has too few candidates (a holiday, say)
    few = cand_ok.sum(1) < k
    dist[few] = np.where(offs[None, :, None] <= 7, np.abs(tmean_actual[cand[few]] - target_t[few][:, None, :]), np.inf)
    pick = np.argsort(dist, axis=1)[:, :k]  # (P, k, M)
    days = np.take_along_axis(cand[:, :, None].repeat(nd.m, 2), pick, axis=1)  # (P, k, M)
    prof = nd.load[days[:, :, None, :], np.arange(24)[:, None], np.arange(nd.m)]  # (P, k, 24, M)
    return prof.mean(1)


def targets(nd: NodeData, issues: np.ndarray, leads: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Metered load at each row and whether it may be used (metered, not flagged). Same row order as build()."""
    t = np.asarray(issues) + np.asarray(leads)
    y = nd.actual[t]
    ok = np.isfinite(y) & ~nd.flags[t]
    return y.reshape(-1), ok.reshape(-1)


def training_pairs(first_target: int, last_target: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """One (issue, lead) pair per target day in [first_target, last_target], lead drawn at random.

    Day-ahead is the product that matters most, so lead 1 gets a third of the rows."""
    t = np.arange(first_target, last_target + 1)
    p = np.array([0.34] + [0.11] * 6)
    j = rng.choice(np.arange(1, MAX_LEAD + 1), size=t.size, p=p / p.sum())
    i = t - j
    keep = i >= MIN_ISSUE
    return i[keep], j[keep]
