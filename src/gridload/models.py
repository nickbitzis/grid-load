"""Forecast models. All of them predict load / scale7 for rows built by features.build().

- SameHourLastWeek: the planner's default. The same local hour seven days earlier (fourteen if not metered yet).
- SimilarDays: average profile of the 3 recent days of the same day type whose temperature was closest to the
  target day's forecast.
- SplineRegression: ridge on a temperature spline (the U-shape), hour x day-type, month, holiday, school, solar.
- GBM: one pooled HistGradientBoostingRegressor on weather, calendar, lags, and lead time.
- QuantileGBM: the same features with pinball loss for P10 and P90.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import SplineTransformer

from .features import COL, NodeData, similar_days_mw
from .generator import Dataset


@dataclass(frozen=True)
class GBMConfig:
    max_iter: int = 200
    learning_rate: float = 0.08
    max_leaf_nodes: int = 31
    min_samples_leaf: int = 40
    quantile_iter: int = 120
    quantile_leaves: int = 15


class SameHourLastWeek:
    name = "same_hour_last_week"

    def fit(self, X, y):
        return self

    def predict(self, X) -> np.ndarray:
        return X[:, COL["lag_week"]].copy()


class SimilarDays:
    """Needs the load grid, so it predicts directly in MW for (issue, lead) pairs."""

    name = "similar_days"

    def __init__(self, k: int = 3, lookback: int = 56):
        self.k, self.lookback = k, lookback

    def predict_mw(self, ds: Dataset, nd: NodeData, issues: np.ndarray, leads: np.ndarray) -> np.ndarray:
        return similar_days_mw(ds, nd, issues, leads, self.k, self.lookback)


class SplineRegression:
    """One ridge regression per node. Temperature enters through a cubic spline, so the model can bend into the
    U-shape: load rises on both sides of a comfortable 16 to 20 C."""

    name = "spline_regression"

    def __init__(self, alpha: float = 1.0, n_knots: int = 8):
        self.alpha, self.n_knots = alpha, n_knots
        self.models: dict[int, tuple[SplineTransformer, Ridge]] = {}

    def _design(self, X, spline) -> np.ndarray:
        temp = X[:, [COL["temp"]]]
        hour = X[:, COL["hour"]].astype(int)
        dt = X[:, COL["daytype"]].astype(int)
        month = X[:, COL["month"]].astype(int)
        hd = np.zeros((X.shape[0], 72))
        hd[np.arange(X.shape[0]), dt * 24 + hour] = 1
        mo = np.zeros((X.shape[0], 12))
        mo[np.arange(X.shape[0]), month - 1] = 1
        sp = spline.transform(temp)
        day = ((hour >= 9) & (hour <= 20)).astype(float)[:, None]
        cdd = np.clip(X[:, [COL["temp"]]] - 20, 0, None)
        extra = np.column_stack(
            [
                X[:, COL["school"]],
                X[:, COL["solar_clear"]],
                cdd * (X[:, [COL["hum"]]] - 55) / 10,
                np.clip(16 - X[:, COL["tmean_prev"]], 0, None),
            ]
        )
        return np.hstack([hd, mo, sp, sp * day, extra])

    def fit(self, X, y):
        nodes = X[:, COL["node"]].astype(int)
        for m in np.unique(nodes):
            sel = nodes == m
            spline = SplineTransformer(n_knots=self.n_knots, degree=3, extrapolation="linear").fit(
                X[sel][:, [COL["temp"]]]
            )
            r = Ridge(alpha=self.alpha).fit(self._design(X[sel], spline), y[sel])
            self.models[int(m)] = (spline, r)
        return self

    def predict(self, X) -> np.ndarray:
        nodes = X[:, COL["node"]].astype(int)
        out = np.empty(X.shape[0])
        for m in np.unique(nodes):
            sel = nodes == m
            spline, r = self.models[int(m)]
            out[sel] = r.predict(self._design(X[sel], spline))
        return out


CATEGORICAL = [COL["node"], COL["daytype"]]


class GBM:
    name = "gbm"

    def __init__(self, cfg: GBMConfig | None = None, quantile: float | None = None, seed: int = 0):
        cfg = cfg or GBMConfig()
        self.cfg, self.quantile = cfg, quantile
        kw = {"loss": "quantile", "quantile": quantile} if quantile is not None else {}
        self.model = HistGradientBoostingRegressor(
            max_iter=cfg.quantile_iter if quantile is not None else cfg.max_iter,
            learning_rate=cfg.learning_rate,
            max_leaf_nodes=cfg.quantile_leaves if quantile is not None else cfg.max_leaf_nodes,
            min_samples_leaf=cfg.min_samples_leaf,
            categorical_features=CATEGORICAL,
            early_stopping=False,
            random_state=seed,
            **kw,
        )

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X) -> np.ndarray:
        return self.model.predict(X)
