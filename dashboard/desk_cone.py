"""
desk_cone.py — the volatility cone (Burghardt & Lane 1990) of SPX realised vol
with the surface's ATM term structure on it (B8; pure functions, no Flask).

Realised vol over a window of h sessions is the site's RV convention (RV21 on
the Overview and Research, desk_market.rv_trailing) at horizon h:

    RV_h(t) = 100·√(252/h · Σ_{i=0..h−1} r²_{t−i}),   r_t = ln(C_t / C_{t−1}),

close-to-close, zero mean, annualised at 252 sessions. The cone at horizon h
is the percentiles of RV_h over every window (overlapping, one per session)
whose returns all lie inside the sample, the sample ending at the surface's
session so nothing after it enters. An expiry of T calendar days is placed at
T·252/365 sessions, and its ATMF is ranked against RV at that many sessions
(rounded), over the same sample.
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

HORIZONS: tuple[int, ...] = (10, 21, 42, 63, 126, 252)
PCTS: tuple[int, ...] = (5, 25, 50, 75, 95)
SESSIONS_PER_YEAR = 252
DAYS_PER_YEAR = 365
SAMPLE_START = "2010-01-04"
SAMPLES: tuple[tuple[str, str, str], ...] = (("since-2010", "Since 2010", "Since 2010"),
                                             ("last-3y", "Last 3 years", "Last 3 y"))


def session_horizon(T_days: float) -> float:
    """Calendar days to expiry as sessions: T·252/365."""
    return float(T_days) * SESSIONS_PER_YEAR / DAYS_PER_YEAR


def sample_returns(close: pd.Series, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.Series:
    """ln(C_t / C_{t−1}) of the consecutive stored closes from ``start`` to
    ``end`` (both included): the first return is the session after ``start``."""
    c = close.astype(float).sort_index()
    c = c[(c.index >= pd.Timestamp(start)) & (c.index <= pd.Timestamp(end))]
    return pd.Series(np.log(c.to_numpy(float)), index=c.index).diff().dropna()


def rv_windows(r: pd.Series, h: int) -> pd.Series:
    """RV_h of every window of h consecutive returns, indexed by its last session (%)."""
    return (np.sqrt((r.astype(float) ** 2).rolling(h).sum() * SESSIONS_PER_YEAR / h) * 100.0).dropna()


def sample_start(name: str, close: pd.Series, end: str | pd.Timestamp) -> pd.Timestamp:
    """The first close of a named sample: 2010-01-04 or the first session on or
    after the same date three years before ``end``."""
    end = pd.Timestamp(end)
    want = pd.Timestamp(SAMPLE_START) if name == "since-2010" else end - pd.DateOffset(years=3)
    idx = close.sort_index().index
    after = idx[idx >= want]
    if not len(after):
        raise ValueError(f"no close on or after {want.date()}")
    return pd.Timestamp(after[0])


def cone(r: pd.Series, horizons: Sequence[int] = HORIZONS, pcts: Sequence[int] = PCTS) -> list[dict]:
    """Per horizon: the number of windows, the percentiles of RV_h (numpy's
    linear interpolation) and the newest window's RV_h (the sample's last session)."""
    out = []
    for h in horizons:
        rv = rv_windows(r, h)
        if rv.empty:
            out.append({"h": h, "n": 0, "pct": dict.fromkeys(pcts), "current": None})
            continue
        v = rv.to_numpy(float)
        out.append({"h": h, "n": int(v.size), "pct": {p: float(np.percentile(v, p)) for p in pcts},
                    "current": float(v[-1]), "last": str(pd.Timestamp(rv.index[-1]).date())})
    return out


def rank(r: pd.Series, h: int, value: float) -> float | None:
    """The share (%) of the sample's RV_h windows below ``value``."""
    rv = rv_windows(r, h)
    if rv.empty or value is None or not math.isfinite(value):
        return None
    return float(100.0 * (rv.to_numpy(float) < value).mean())


def implied_points(listed: Sequence[Mapping[str, Any]], r: pd.Series,
                   max_h: int = HORIZONS[-1]) -> tuple[list[dict], int]:
    """The SVI-cal ATMF of each listed expiry at T·252/365 sessions, ranked
    against RV over that many sessions (rounded, at least 2) in the sample;
    expiries beyond ``max_h`` sessions are counted, not placed."""
    pts, beyond = [], 0
    for row in listed:
        T_days, atmf = row.get("T_days"), row.get("atmf")
        if T_days is None or atmf is None:
            continue
        hs = session_horizon(T_days)
        if hs > max_h:
            beyond += 1
            continue
        h = max(2, round(hs))
        pts.append({"expiry": row.get("expiry"), "T_days": float(T_days), "sessions": hs, "h": h,
                    "atmf": float(atmf), "rank": rank(r, h, float(atmf))})
    return pts, beyond
