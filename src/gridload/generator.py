"""Seeded synthetic history for Prairie Line Power, a made-up utility with five load zones.

Everything lives on a local-clock grid of shape (days, 24 hour slots, zones). Behavior follows the clock (people
wake at 06:30 local whether or not DST just started), while the sun and the diurnal temperature cycle follow
standard time, so solar output shifts by an hour in local time between winter and summer. `hour_weight` says how
many real hours each slot holds (0 for the missing spring hour, 2 for the repeated fall hour).

The weather forecast archive holds, for each issue day i (issued at 10:00 local) and lead j = 0..7 days, the
forecast for every hour of day i + j. Lead 0 is the issue day itself: hours before 10:00 are observed, the rest are
a short-range forecast. Forecast error grows with lead time and is correlated across hours and zones.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

import numpy as np

from . import calendar as cal

ISSUE_HOUR = 10  # forecasts are issued at 10:00 local; load is metered through 09:59
MAX_LEAD = 7


@dataclass(frozen=True)
class Zone:
    id: str
    name: str
    base_mw: float  # average load before weather, at the start of the history
    heat: float  # heating response: share of base at 10 heating degrees
    cool: float  # cooling response: share of base at 8 cooling degrees
    temp_offset: float  # zone climate vs the regional mean, degrees C
    solar_start: float  # rooftop solar capacity as a share of base at the start
    solar_end: float  # ... and at the end of four years
    industrial: float  # share of load with a flat daily profile
    school: float  # load added by schools in session, share of base during school hours


ZONES = (
    Zone("north", "North Plains", 640, 0.27, 0.12, -2.6, 0.04, 0.16, 0.20, 0.025),
    Zone("metro", "Metro Core", 1720, 0.16, 0.30, 0.9, 0.05, 0.22, 0.25, 0.035),
    Zone("east", "East Ridge", 760, 0.18, 0.20, 0.0, 0.05, 0.20, 0.25, 0.030),
    Zone("south", "South Valley", 1140, 0.10, 0.38, 2.4, 0.08, 0.34, 0.15, 0.035),
    Zone("west", "West Industrial", 960, 0.12, 0.12, -0.6, 0.03, 0.12, 0.60, 0.010),
)
ZONE_IDS = tuple(z.id for z in ZONES)
DATA_CENTER_ZONE = "east"

WEEKDAY = np.array(
    [0.74, 0.71, 0.70, 0.70, 0.72, 0.78, 0.88, 0.97, 1.00, 1.01, 1.02, 1.03,
     1.03, 1.03, 1.03, 1.03, 1.04, 1.07, 1.09, 1.08, 1.04, 0.97, 0.88, 0.80]
)  # fmt: skip
WEEKEND = np.array(
    [0.78, 0.74, 0.72, 0.71, 0.71, 0.73, 0.77, 0.82, 0.88, 0.93, 0.96, 0.97,
     0.97, 0.96, 0.96, 0.96, 0.97, 1.00, 1.03, 1.03, 1.00, 0.95, 0.88, 0.82]
)  # fmt: skip


@dataclass
class Event:
    kind: str  # heat_wave | winter_storm
    name: str
    start: date
    days: int
    temp_delta: float


@dataclass
class GenConfig:
    seed: int = 42
    end: date = date(2026, 11, 8)  # "today": the last day with (partial) metered load
    days: int = 1461  # four years of history
    events: tuple[Event, ...] = (
        Event("winter_storm", "January winter storm", date(2026, 1, 19), 5, -14.0),
        Event("heat_wave", "July heat wave", date(2026, 7, 13), 7, 8.0),
    )
    outage_zone: str = "north"
    outage_start: datetime = datetime(2026, 1, 21, 2)  # local time
    outage_hours: int = 34
    data_center_start_mw: float = 120.0
    data_center_end_mw: float = 600.0


@dataclass
class Dataset:
    dates: list[date]  # history plus MAX_LEAD + 1 future days whose weather exists but load is never shown
    n_hist: int
    zones: tuple[Zone, ...]
    load: np.ndarray  # (N, 24, Z) net load MW
    temp: np.ndarray  # (N, 24, Z) actual temperature C
    hum: np.ndarray  # (N, 24, Z) relative humidity %
    cloud: np.ndarray  # (N,) regional cloud cover 0..1
    temp_fc: np.ndarray  # (N, MAX_LEAD + 1, 24, Z) forecast issued day i for day i + j
    hum_fc: np.ndarray
    cloud_fc: np.ndarray  # (N, MAX_LEAD + 1)
    solar: np.ndarray  # (N, 24, Z) rooftop solar output MW (behind the meter)
    solar_cap: np.ndarray  # (N, Z) installed rooftop capacity MW, known to the utility
    clear_sky: np.ndarray  # (N, 24) clear-sky output per MW of capacity, local clock
    data_center: np.ndarray  # (N, 24) MW
    dow: np.ndarray
    month: np.ndarray
    doy: np.ndarray
    holiday: np.ndarray
    bridge: np.ndarray
    school: np.ndarray
    dst: np.ndarray
    hour_weight: np.ndarray  # (N, 24) real hours per slot
    outage: np.ndarray  # (N, 24, Z) ground truth outage cells
    events: list[Event] = field(default_factory=list)
    holiday_names: dict[date, str] = field(default_factory=dict)

    @property
    def zone_ids(self) -> tuple[str, ...]:
        return tuple(z.id for z in self.zones)

    def day_index(self, d: date) -> int:
        return (d - self.dates[0]).days

    @property
    def today(self) -> int:
        return self.n_hist - 1


def _ar1(rng, n: int, rho: float, sd: float, shape=()) -> np.ndarray:
    eps = rng.normal(0, sd * np.sqrt(1 - rho**2), size=(n, *shape))
    out = np.empty_like(eps)
    out[0] = rng.normal(0, sd, size=shape)
    for t in range(1, n):
        out[t] = rho * out[t - 1] + eps[t]
    return out


def clear_sky_profile(doy: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Clear-sky solar output per MW of capacity, by local clock hour slot (N, 24)."""
    hours = np.arange(24) + 0.5
    std_hour = hours[None, :] - dst[:, None].astype(float)  # local clock to standard time
    season = np.sin(2 * np.pi * (doy - 80) / 365.25)
    half_day = 6.0 + 1.6 * season  # hours from solar noon to sunset
    noon = 12.4
    x = (std_hour - noon) / half_day[:, None]
    shape = np.clip(np.cos(np.pi / 2 * x), 0, None) ** 1.5
    shape[np.abs(x) >= 1] = 0
    peak = 0.62 + 0.18 * season  # sun angle
    return shape * peak[:, None]


def generate(cfg: GenConfig | None = None) -> Dataset:
    cfg = cfg or GenConfig()
    rng = np.random.default_rng(cfg.seed)
    zones = ZONES
    Z = len(zones)
    n_total = cfg.days + MAX_LEAD + 1
    start = cfg.end - timedelta(days=cfg.days - 1)
    dates = [start + timedelta(days=k) for k in range(n_total)]
    years = sorted({d.year for d in dates})
    hol_names: dict[date, str] = {}
    bridges: set[date] = set()
    for y in years:
        hol_names.update(cal.holidays(y))
        bridges |= cal.bridge_days(y)
    hol = set(hol_names)

    dow = np.array([d.weekday() for d in dates])
    month = np.array([d.month for d in dates])
    doy = np.array([d.timetuple().tm_yday for d in dates], dtype=float)
    holiday = np.array([d in hol for d in dates])
    bridge = np.array([d in bridges for d in dates])
    school = np.array([cal.school_in_session(d, hol, bridges) for d in dates])
    dst = np.array([cal.is_dst(d) for d in dates])
    hour_weight = np.array([cal.hour_weights(d) for d in dates], dtype=float)
    t_years = np.arange(n_total) / 365.25

    # Weather: regional daily anomaly, zone-specific wiggle, diurnal cycle in standard time.
    season_t = 11.5 - 14.0 * np.cos(2 * np.pi * (doy - 18) / 365.25)
    regional = _ar1(rng, n_total, 0.78, 3.4)
    zonal = _ar1(rng, n_total, 0.6, 1.0, (Z,))
    events = [e for e in cfg.events if dates[0] <= e.start <= dates[-1]]
    event_delta = np.zeros(n_total)
    for e in events:
        i0 = (e.start - start).days
        k = np.arange(-1, e.days + 1)
        bump = np.sin(np.pi * (k + 1) / (e.days + 1)) ** 0.7
        idx = i0 + k
        ok = (idx >= 0) & (idx < n_total)
        event_delta[idx[ok]] += e.temp_delta * bump[ok]
    offsets = np.array([z.temp_offset for z in zones])
    tmean = season_t[:, None] + regional[:, None] + zonal + offsets[None, :] + event_delta[:, None]
    std_hour = np.arange(24)[None, :] - dst[:, None].astype(float)
    amp = 5.0 + 1.8 * np.sin(2 * np.pi * (doy - 100) / 365.25)
    diurnal = amp[:, None] * np.cos(2 * np.pi * (std_hour - 15.5) / 24)
    temp = tmean[:, None, :] + diurnal[:, :, None] + rng.normal(0, 0.5, size=(n_total, 24, Z))

    summer = 0.5 + 0.5 * np.sin(2 * np.pi * (doy - 110) / 365.25)
    hum_daily = 58 + 14 * summer + _ar1(rng, n_total, 0.6, 8.0)
    hum_diurnal = 14 * np.cos(2 * np.pi * (std_hour - 5) / 24)
    hum = np.clip(hum_daily[:, None, None] + hum_diurnal[:, :, None] + rng.normal(0, 3, (n_total, 24, Z)), 12, 100)
    for e in events:
        if e.kind == "heat_wave":
            i0 = (e.start - start).days
            hum[max(i0, 0) : i0 + e.days] = np.clip(hum[max(i0, 0) : i0 + e.days] + 12, 12, 100)
    cloud = np.clip(rng.beta(1.3, 2.2, n_total) * 0.6 + 0.4 * np.roll(rng.beta(1.3, 2.2, n_total), 1), 0, 1)

    # Load components.
    clear = clear_sky_profile(doy, dst)
    load = np.zeros((n_total, 24, Z))
    solar = np.zeros((n_total, 24, Z))
    solar_cap = np.zeros((n_total, Z))
    hours = np.arange(24)
    cool_shape = 0.55 + 0.45 * np.exp(-0.5 * ((hours - 16.5) / 3.5) ** 2)
    heat_shape = 1.0 + 0.12 * np.exp(-0.5 * ((hours - 7.5) / 1.8) ** 2) + 0.10 * np.exp(-0.5 * ((hours - 19) / 2) ** 2)
    sat = 0.97 * WEEKEND
    sun = 0.93 * WEEKEND
    profile = np.empty((n_total, 24))
    for k in range(n_total):
        if holiday[k] or dow[k] == 6:
            p = sun * (0.95 if holiday[k] else 1.0)
        elif dow[k] == 5:
            p = sat
        elif bridge[k]:
            p = 0.96 * (0.5 * WEEKDAY + 0.5 * sat)
        else:
            p = WEEKDAY
        profile[k] = p
    daily_noise = _ar1(rng, n_total, 0.7, 0.015, (Z,))
    hourly_noise = _ar1(rng, n_total * 24, 0.85, 0.01, (Z,)).reshape(n_total, 24, Z)
    years_total = cfg.days / 365.25
    for zi, z in enumerate(zones):
        growth = 1 + 0.012 * t_years
        shape = z.industrial + (1 - z.industrial) * profile
        base = z.base_mw * growth[:, None] * shape
        t_eff = 0.6 * temp[:, :, zi] + 0.4 * tmean[:, zi][:, None]
        hdd = np.clip(16.0 - t_eff, 0, None)
        cdd = np.clip(t_eff - 20.0, 0, None)
        hum_mult = np.clip(1 + 0.012 * (hum[:, :, zi] - 55), 0.6, 1.6)
        heating = z.base_mw * growth[:, None] * z.heat * (hdd / 10) ** 1.3 * heat_shape[None, :]
        cooling = z.base_mw * growth[:, None] * z.cool * (cdd / 8) ** 1.5 * cool_shape[None, :] * hum_mult
        school_mw = np.where(school[:, None] & (hours[None, :] >= 7) & (hours[None, :] <= 15), z.school * z.base_mw, 0)
        frac = np.clip(t_years / years_total, 0, None)
        cap = z.base_mw * (z.solar_start + (z.solar_end - z.solar_start) * frac**1.3)
        solar_cap[:, zi] = cap
        solar[:, :, zi] = cap[:, None] * clear * (1 - 0.72 * cloud[:, None])
        gross = (base + heating + cooling + school_mw) * (1 + daily_noise[:, zi][:, None] + hourly_noise[:, :, zi])
        load[:, :, zi] = gross - solar[:, :, zi]

    # Data center in East Ridge: new halls every six months, nearly flat, a little extra cooling in heat.
    steps = np.floor(t_years * 2) / max(np.floor(years_total * 2), 1)
    dc_mw = cfg.data_center_start_mw + (cfg.data_center_end_mw - cfg.data_center_start_mw) * np.clip(steps, 0, 1)
    dc_zone = list(ZONE_IDS).index(DATA_CENTER_ZONE)
    dc_cool = 1 + 0.004 * np.clip(temp[:, :, dc_zone] - 20, 0, None)
    data_center = dc_mw[:, None] * dc_cool * (1 + rng.normal(0, 0.01, (n_total, 24)))
    load[:, :, dc_zone] += data_center

    # Storm outage: a zone loses most of its load, then is restored over the last 8 hours.
    outage = np.zeros((n_total, 24, Z), dtype=bool)
    oz = list(ZONE_IDS).index(cfg.outage_zone)
    o_day = (cfg.outage_start.date() - start).days
    if 0 <= o_day < cfg.days:
        for k in range(cfg.outage_hours):
            ts = cfg.outage_start + timedelta(hours=k)
            di, h = (ts.date() - start).days, ts.hour
            if di >= n_total:
                break
            left = cfg.outage_hours - k
            served = 0.3 if left > 8 else 0.3 + 0.7 * (8 - left) / 8
            load[di, h, oz] *= served
            outage[di, h, oz] = True

    # Forecast archive. Error sd grows with lead hours; correlated across hours, leads, and zones.
    L = MAX_LEAD + 1
    lead_hours = np.arange(L)[:, None] * 24 + hours[None, :] - ISSUE_HOUR  # (L, 24)
    sd_t = 0.5 + 2.7 * (np.clip(lead_hours, 0, None) / 168) ** 0.75
    sd_h = 2.0 + 9.0 * (np.clip(lead_hours, 0, None) / 168) ** 0.75
    lead_walk = (
        np.cumsum(rng.normal(0, 1, (n_total, L, 1, 1)), axis=1) / np.sqrt(np.arange(1, L + 1))[None, :, None, None]
    )
    region_part = 0.75 * lead_walk + 0.45 * _smooth_hours(rng, (n_total, L, 24, 1))
    zone_part = 0.4 * _smooth_hours(rng, (n_total, L, 24, Z))
    err = (region_part + zone_part) * sd_t[None, :, :, None]
    herr = (0.7 * lead_walk + 0.6 * _smooth_hours(rng, (n_total, L, 24, Z))) * sd_h[None, :, :, None]
    observed = lead_hours < 0  # issue-day hours before 10:00 are measured, not forecast
    err[:, observed] = 0
    herr[:, observed] = 0
    temp_fc = np.full((n_total, L, 24, Z), np.nan, dtype=np.float32)
    hum_fc = np.full((n_total, L, 24, Z), np.nan, dtype=np.float32)
    cloud_fc = np.full((n_total, L), np.nan, dtype=np.float32)
    for j in range(L):
        n = n_total - j
        temp_fc[:n, j] = temp[j:] + err[:n, j]
        hum_fc[:n, j] = np.clip(hum[j:] + herr[:n, j], 5, 100)
        cloud_fc[:n, j] = np.clip(cloud[j:] + rng.normal(0, 0.12 + 0.05 * j, n), 0, 1)

    return Dataset(
        dates=dates,
        n_hist=cfg.days,
        zones=zones,
        load=load,
        temp=temp,
        hum=hum,
        cloud=cloud,
        temp_fc=temp_fc,
        hum_fc=hum_fc,
        cloud_fc=cloud_fc,
        solar=solar,
        solar_cap=solar_cap,
        clear_sky=clear,
        data_center=data_center,
        dow=dow,
        month=month,
        doy=doy,
        holiday=holiday,
        bridge=bridge,
        school=school,
        dst=dst,
        hour_weight=hour_weight,
        outage=outage,
        events=events,
        holiday_names={d: n for d, n in hol_names.items() if dates[0] <= d <= dates[-1]},
    )


def _smooth_hours(rng, shape) -> np.ndarray:
    """Unit-variance noise smoothed along the hour axis (axis 2)."""
    x = rng.normal(0, 1, shape)
    k = np.array([0.25, 0.5, 0.25])
    for _ in range(3):
        x = k[0] * np.roll(x, 1, axis=2) + k[1] * x + k[2] * np.roll(x, -1, axis=2)
    return x / x.std()


def forecast_weather(ds: Dataset, issue: int, target: int) -> dict[str, np.ndarray]:
    """The weather forecast for day `target` as it was known at 10:00 on day `issue`. The leakage guard: there is
    no way to ask for a forecast issued after the fact, or for a lead the archive does not hold."""
    lead = target - issue
    if not 0 <= lead <= MAX_LEAD:
        raise ValueError(f"no forecast issued on day {issue} covers day {target} (lead {lead} days)")
    return {"temp": ds.temp_fc[issue, lead], "hum": ds.hum_fc[issue, lead], "cloud": ds.cloud_fc[issue, lead]}


def to_hourly(ds: Dataset, values: np.ndarray, day0: int, day1: int) -> list[tuple[str, float]]:
    """Expand grid values (days, 24) for days [day0, day1) into real clock hours with ISO timestamps.

    The spring-forward slot is dropped and the fall-back slot appears twice (fold 0 and 1)."""
    out = []
    for di in range(day0, day1):
        d = ds.dates[di]
        for h in range(24):
            w = int(ds.hour_weight[di, h])
            for fold in range(w):
                ts = datetime(d.year, d.month, d.day, h, tzinfo=cal.TZ, fold=fold)
                out.append((ts.isoformat(), float(values[di - day0, h])))
    return out
