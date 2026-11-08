"""Calendar facts the load depends on and every forecaster knows in advance: holidays, bridge days, school
terms, and daylight saving transitions in the utility's time zone (US Central)."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo

TZ = ZoneInfo("America/Chicago")


def _nth_weekday(year: int, month: int, weekday: int, n: int) -> date:
    d = date(year, month, 1)
    d += timedelta(days=(weekday - d.weekday()) % 7)
    return d + timedelta(weeks=n - 1)


def _last_weekday(year: int, month: int, weekday: int) -> date:
    d = date(year, month + 1, 1) - timedelta(days=1) if month < 12 else date(year, 12, 31)
    return d - timedelta(days=(d.weekday() - weekday) % 7)


def _observed(d: date) -> date:
    if d.weekday() == 5:
        return d - timedelta(days=1)
    if d.weekday() == 6:
        return d + timedelta(days=1)
    return d


def holidays(year: int) -> dict[date, str]:
    """Utility holidays (offices and most businesses closed), on their observed weekday."""
    out = {
        _observed(date(year, 1, 1)): "New Year's Day",
        _last_weekday(year, 5, 0): "Memorial Day",
        _observed(date(year, 7, 4)): "Independence Day",
        _nth_weekday(year, 9, 0, 1): "Labor Day",
        _nth_weekday(year, 11, 3, 4): "Thanksgiving",
        _observed(date(year, 12, 25)): "Christmas Day",
    }
    return out


def bridge_days(year: int) -> set[date]:
    """Weekdays squeezed between a holiday and a weekend, plus Christmas Eve and New Year's Eve.

    Many people take them off, so load sits between a weekday and a Saturday."""
    hol = holidays(year)
    out = set()
    for d in hol:
        if d.weekday() == 1:  # Tuesday holiday: Monday bridge
            out.add(d - timedelta(days=1))
        if d.weekday() == 3:  # Thursday holiday: Friday bridge (the Friday after Thanksgiving)
            out.add(d + timedelta(days=1))
    for d in (date(year, 12, 24), date(year, 12, 31)):
        if d.weekday() < 5 and d not in hol:
            out.add(d)
    return {d for d in out if d not in hol}


def school_in_session(d: date, hol: set[date], bridge: set[date]) -> bool:
    """A simple public school calendar: late August to the end of May, weekdays, minus breaks."""
    if d.weekday() >= 5 or d in hol or d in bridge:
        return False
    m, day = d.month, d.day
    if (m == 8 and day < 21) or m in (6, 7):
        return False
    if (m == 12 and day >= 21) or (m == 1 and day <= 3):
        return False  # winter break
    thanks = _nth_weekday(d.year, 11, 3, 4)
    if thanks - timedelta(days=3) <= d <= thanks + timedelta(days=1):
        return False  # Thanksgiving week
    spring = _nth_weekday(d.year, 3, 0, 2)
    return not (spring <= d < spring + timedelta(days=5))


def utc_offset_hours(d: date, hour: int) -> int:
    return int(datetime(d.year, d.month, d.day, hour, tzinfo=TZ).utcoffset().total_seconds() // 3600)


def hour_weights(d: date) -> list[int]:
    """How many real clock hours each of the 24 local hour slots holds on date d.

    The spring-forward day has no 02:00 (weight 0, 23 hours). The fall-back day has 01:00 twice (weight 2, 25 hours).
    """
    w = [1] * 24
    before, after = utc_offset_hours(d, 0), utc_offset_hours(d, 4)
    if after > before:
        w[2] = 0
    elif after < before:
        w[1] = 2
    return w


def is_dst(d: date) -> bool:
    return utc_offset_hours(d, 12) == -5
