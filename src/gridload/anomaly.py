"""Detect load that is not demand (outages, meter failures) so it neither trains the models nor feeds the lags.

The detector is trailing only: a cell is compared with the median of the same hour over the previous 14 days,
so running it over the whole history never looks ahead. A storm raises heating load, so a sudden drop to under
55% of the recent normal for 4+ hours in a row is not weather. Rooftop solar can do that to net load on a sunny
holiday afternoon, but not at night, so a run must include at least 2 dark hours to count. The 4 hours after a
run are flagged too, since restoration is not demand. Flagged cells are imputed with the trailing median for use
as lag inputs, dropped as training targets, and left out of scoring.
"""

from __future__ import annotations

import numpy as np

RATIO = 0.55
MIN_RUN = 4
MIN_DARK = 2
PAD = 4  # restoration hours after a run, when load is climbing back but still not demand
WINDOW = 14


def trailing_median(load: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Median of the same hour slot over the previous `window` days. load: (N, 24, Z)."""
    n = load.shape[0]
    out = np.full_like(load, np.nan, dtype=float)
    for d in range(1, n):
        lo = max(0, d - window)
        out[d] = np.nanmedian(load[lo:d], axis=0)
    return out


def flag_runs(low: np.ndarray, min_run: int = MIN_RUN, pad: int = PAD, dark: np.ndarray | None = None) -> np.ndarray:
    """1-D boolean series: keep runs of at least `min_run` True values (with 2+ dark hours when `dark` is given),
    extended by `pad` after each run."""
    out = np.zeros_like(low)
    start = None
    for k, v in enumerate(np.append(low, False)):
        if v and start is None:
            start = k
        elif not v and start is not None:
            if k - start >= min_run and (dark is None or dark[start:k].sum() >= MIN_DARK):
                out[start : min(k + pad, low.size)] = True
            start = None
    return out


def detect(
    load: np.ndarray, dark: np.ndarray | None = None, ratio: float = RATIO, min_run: int = MIN_RUN
) -> tuple[np.ndarray, np.ndarray]:
    """Return (flags, expected): boolean (N, 24, Z) anomalous cells and the trailing-median expectation.

    dark: optional (N, 24) mask of hours with no sun."""
    expected = trailing_median(load)
    low = np.nan_to_num(load / expected, nan=1.0) < ratio
    flags = np.zeros_like(low)
    n, _, z = load.shape
    for zi in range(z):
        d = None if dark is None else dark.reshape(-1)
        flags[:, :, zi] = flag_runs(low[:, :, zi].reshape(-1), min_run, dark=d).reshape(n, 24)
    flags &= np.isfinite(load)
    return flags, expected


def clean(load: np.ndarray, flags: np.ndarray, expected: np.ndarray) -> np.ndarray:
    out = load.copy()
    out[flags] = expected[flags]
    return out


def runs(flags: np.ndarray, dates, zone_ids) -> list[dict]:
    """Summaries of flagged runs, for the API and README."""
    out = []
    n, _, z = flags.shape
    for zi in range(z):
        series = flags[:, :, zi].reshape(-1)
        k = 0
        while k < series.size:
            if series[k]:
                s = k
                while k < series.size and series[k]:
                    k += 1
                out.append(
                    {
                        "zone": zone_ids[zi],
                        "start": f"{dates[s // 24].isoformat()} {s % 24:02d}:00",
                        "hours": k - s,
                    }
                )
            k += 1
    return out
