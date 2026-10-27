"""
option_chain.py — SPX option-chain hygiene for the joint calibrator (C22).

Everything between "Yahoo returned a chain" and "the calibrator has quotes it
can trust" lives here, so the live daily pipeline (scripts/cloud_refresh.py)
and research callers that read a stored snapshot clean data the same way.

Why this module exists (C22 audit, live outage 2026-09-08..09-23):
  * The pipeline kept Yahoo's ``impliedVolatility`` and fell back to
    ``lastPrice`` whenever a side of the book was empty. Yahoo bisects its IV
    on its own (bid+ask)/2, so on an empty after-hours book the near-ATM lines
    get IV 1e-5 (and were dropped) while the OTM wings got 2^-k bisection
    artefacts (0.0156, 0.0313, ...) that survived — the calibrator then fitted
    stale, non-monotone last trades with vegas from artefact IVs and reported
    176-273 vp "SPX RMSE".
  * Even with a live book Yahoo's IV is computed against a forward that does
    not satisfy put-call parity (1.2 vp put/call gap at the same strike).
  * Only the first 8 listed expiries were downloaded — ~1.5 weeks of daily
    SPXW expiries — so every fit saw ONE expiry.

What this module does instead:
  1. ``select_calibration_expiries``: for each target tenor (14..365d) the
     nearest listed expiry with >= 7 days, unique, within 50% of the target.
  2. ``clean_option_chain``: two-sided quotes only (bid > 0, ask >= bid,
     relative spread cap), never lastPrice; one settlement style per expiry
     (PM SPXW preferred over AM SPX; AM-only expiries get T to 09:30 ET);
     per-expiry parity forward F_T and discount D_T from a regression of
     (C - P) on K over near-ATM strikes; own Black-76 implied vols from
     (mid, F_T, D_T); Yahoo's IV kept only as a diagnostic column.
  3. A quote-state gate: if too few near-ATM lines are two-sided (empty
     after-hours book) the chain is marked unusable with an explicit reason —
     the caller must fail closed rather than fit garbage.
  4. Session helpers on database.py's NYSE calendar (date-object API): which
     session a wall-clock time belongs to, so snapshots are labelled by the
     session whose quotes they hold.
  5. The origin T is measured from (``t_origin``): the session's 16:00 ET
     close for a closing book, but the quote time for a Cboe Global Trading
     Hours book. The book slots run at 21:30-00:30 ET, when the live book is
     the GTH session trading for the NEXT session (20:15 ET the evening
     before it to 09:15 ET on it); measuring T from the 16:00 close there
     overstated T by 5.5-8.5 hours (14-day expiry: 1.6%, short-dated IVs ~1%
     low). A stored chain without a quote time keeps the session close.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import TYPE_CHECKING, Any, Callable, Iterable, Optional, Sequence, Tuple, TypeVar, Union

import numpy as np
import pandas as pd
from scipy.special import ndtr
from zoneinfo import ZoneInfo

from joint_vol_calibration.data import database as _db

if TYPE_CHECKING:                     # annotations only; no import at run time
    from numpy.typing import ArrayLike  # anything np.asarray accepts (incl. pd.Series)

logger = logging.getLogger(__name__)

NY_TZ = ZoneInfo("America/New_York")

# Inputs the session helpers accept: a date (datetime and pd.Timestamp are
# subclasses) or anything pd.Timestamp parses, e.g. an ISO string.
DateLike = Union[date, str, np.datetime64]
# A recorded quote time: datetime / pd.Timestamp / ISO string; None, '' and NaN
# (a float, from a missing DataFrame cell) mean "not recorded".
QuoteTimeLike = Union[datetime, str, float, None]
_T = TypeVar("_T")

# ── Single source of truth for the calibration expiry grid ────────────────────
# Target tenors (calendar days) of the SPX smile slices the joint fit uses.
CAL_EXPIRY_TARGETS_DAYS: Tuple[int, ...] = (14, 30, 60, 91, 180, 365)
CAL_MIN_EXPIRY_DAYS: int = 7           # never use an expiry closer than this
CAL_EXPIRY_TOLERANCE: float = 0.5      # |days - target| <= 50% of target

# ── Chain hygiene defaults ────────────────────────────────────────────────────
MAX_REL_SPREAD: float = 0.35           # (ask - bid) / mid
MIN_TWO_SIDED_SHARE: float = 0.80      # near-ATM two-sided share for a live book
ATM_GATE_BAND: float = 0.03            # |K/S - 1| < 3% defines "near ATM" for the gate
MIN_PARITY_PAIRS: int = 4              # strikes with both C and P two-sided
PARITY_RATE_TOL: float = 0.015         # max |r_T(parity) - r_ref| before D is fixed

# Session close and settlement times (America/New_York)
SESSION_CLOSE_ET = time(16, 0)
PM_SETTLE_ET = time(16, 0)             # SPXW (PM-settled) expiries
AM_SETTLE_ET = time(9, 30)             # SPX monthlies (AM, SOQ-settled)
SESSION_CLOSED_AFTER_ET = time(16, 15) # daily bar treated as final after this
GTH_OPEN_ET = time(20, 15)             # Cboe Global Trading Hours open (evening before a session)
GTH_CLOSE_ET = time(9, 15)             # ... and close (morning of that session)
_MINUTES_PER_YEAR = 365.0 * 24.0 * 60.0


# ── NYSE calendar ─────────────────────────────────────────────────────────────
# One calendar for the whole project: database.py's (holiday rules plus the
# special closures 2001-09-11..14, 2004-06-11, 2007-01-02, 2012-10-29/30,
# 2018-12-05, 2025-01-09, checked against the shipped SPX history). These
# wrappers keep this module's date-object API. A second, rule-based copy lived
# here until C22 verification and disagreed on 9 days in 1995-2045.

def _as_date(d: DateLike) -> date:
    if isinstance(d, datetime):
        return d.date()
    if isinstance(d, date):
        return d
    return pd.Timestamp(d).date()


def nyse_holidays(year: int) -> set[date]:
    """Weekday NYSE closures of one calendar year (holidays + special closures)."""
    return {date.fromisoformat(h) for h in _db.nyse_holidays(f"{year}-01-01", f"{year}-12-31")}


def is_nyse_session(d: DateLike) -> bool:
    """True if ``d`` is an NYSE trading day."""
    return bool(_db.is_nyse_session(_as_date(d)))


def previous_session(d: DateLike) -> date:
    """Most recent NYSE session strictly before ``d``."""
    return date.fromisoformat(_db.previous_session(_as_date(d)))


def session_on_or_before(d: DateLike) -> date:
    """Most recent NYSE session on or before ``d`` (the session whose closing
    quotes a weekend/holiday snapshot holds)."""
    return date.fromisoformat(_db.session_on_or_before(_as_date(d)))


def sessions_between(start: DateLike, end: DateLike) -> int:
    """Number of NYSE sessions s with start < s <= end (0 when end <= start)."""
    return int(_db.sessions_between(_as_date(start), _as_date(end)))


def _now_et(now_utc: Optional[datetime] = None) -> datetime:
    now = now_utc or datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)
    return now.astimezone(NY_TZ)


def last_closed_session(now_utc: Optional[datetime] = None) -> date:
    """The most recent NYSE session whose close is final at ``now_utc``.

    A session counts as closed from 16:15 ET (after the 16:00 cash close and
    the 16:15 settlement of the index options). A run at 14:00 ET on a trading
    day therefore returns the PREVIOUS session; a Saturday run returns Friday;
    a 23:30 UTC Friday run returns that Friday. (database.last_closed_session
    with the cutoff SESSION_CLOSED_AFTER_ET.)
    """
    return date.fromisoformat(_db.last_closed_session(_now_et(now_utc),
                                                      cutoff=SESSION_CLOSED_AFTER_ET))


def ny_date(now_utc: Optional[datetime] = None) -> date:
    """Calendar date in New York at ``now_utc``."""
    return _now_et(now_utc).date()


def next_session(d: DateLike) -> date:
    """First NYSE session strictly after ``d``."""
    d = _as_date(d)
    return _db.nyse_sessions(d + timedelta(days=1), d + timedelta(days=14))[0].date()


def session_close(session: DateLike) -> datetime:
    """16:00 ET on ``session`` (New York, tz-aware): the default T origin."""
    return datetime.combine(_as_date(session), SESSION_CLOSE_ET, tzinfo=NY_TZ)


def gth_window(session: DateLike) -> Tuple[datetime, datetime]:
    """(open, close) of the Cboe Global Trading Hours session that trades FOR
    ``session``: 20:15 ET on the calendar evening before it (Sunday evening
    for a Monday, Monday evening for a Tuesday after a Monday holiday) to
    09:15 ET on it. New York, tz-aware."""
    s = _as_date(session)
    return (datetime.combine(s - timedelta(days=1), GTH_OPEN_ET, tzinfo=NY_TZ),
            datetime.combine(s, GTH_CLOSE_ET, tzinfo=NY_TZ))


def _as_instant(when: QuoteTimeLike) -> Optional[datetime]:
    """A quote time (datetime, pandas Timestamp or ISO string) as a tz-aware
    datetime; naive values are taken as UTC (database.utc_iso writes UTC).
    None/NaN/'' -> None."""
    if when is None or (isinstance(when, str) and not when.strip()):
        return None
    try:
        if pd.isna(when):
            return None
    except (TypeError, ValueError):
        pass
    ts = pd.Timestamp(when)
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    return ts.to_pydatetime()


def t_origin(quote_time: QuoteTimeLike, session: DateLike) -> datetime:
    """The New York time T is measured from, for quotes of ``session``
    observed at ``quote_time`` (UTC instant; ISO string or datetime).

    A quote time inside the GTH window of the session AFTER ``session``
    (``gth_window(next_session(session))``) is a Global Trading Hours book —
    what the Mon-Thu book slots download at 21:30-00:30 ET — and T runs from
    the quote time. Anything else is the session's closing book (weekend and
    holiday runs; on weekday evenings the book is empty 16:15-20:15 ET and
    the quote-state gate fails the run closed) and T runs from the 16:00 ET
    close, as does a chain with no recorded quote time (every snapshot stored
    before quote times were recorded).
    """
    close = session_close(session)
    qt = _as_instant(quote_time)
    if qt is None:
        return close
    lo, hi = gth_window(next_session(session))
    qt = qt.astimezone(NY_TZ)
    return qt if lo <= qt < hi else close


def _chain_quote_time(raw: Optional[pd.DataFrame]) -> Optional[datetime]:
    """The quote time a stored chain records (options_snapshots.quote_time,
    ISO UTC): the latest non-empty value, or None (legacy snapshots)."""
    if raw is None or "quote_time" not in raw.columns:
        return None
    vals = [t for t in (_as_instant(v) for v in raw["quote_time"].unique()) if t is not None]
    return max(vals) if vals else None


def year_fraction(session: DateLike, expiry: DateLike, settlement: str = "PM",
                  origin: QuoteTimeLike = None) -> float:
    """Time from the T origin to expiry settlement, in years.

    ``origin`` None (the default) is the session's 16:00 ET close; otherwise
    a datetime (tz-aware, or naive = UTC) or ISO string, e.g. the quote time
    of a Global Trading Hours book (``t_origin``). Minutes are New York
    wall-clock minutes / 525,600 (the Cboe VIX methodology convention; a DST
    change in between is not counted, as with the whole-day count). PM
    settlement is 16:00 ET on the expiry date, so from the close T =
    days/365 exactly; AM settlement (SPX monthlies, SOQ) is 09:30 ET, 6.5
    hours earlier. An ``origin`` that holds no time ('' or NaN, as a missing
    quote_time cell reads) counts as None, the session close.
    """
    start_dt = None if origin is None else _as_instant(origin)
    if start_dt is None:
        days = (_as_date(expiry) - _as_date(session)).days
        minutes = days * 1440.0
        if str(settlement).upper() == "AM":
            minutes -= (SESSION_CLOSE_ET.hour * 60 + SESSION_CLOSE_ET.minute
                        - AM_SETTLE_ET.hour * 60 - AM_SETTLE_ET.minute)
        return minutes / _MINUTES_PER_YEAR
    settle_et = AM_SETTLE_ET if str(settlement).upper() == "AM" else PM_SETTLE_ET
    settle = datetime.combine(_as_date(expiry), settle_et)            # NY wall clock
    start = start_dt.astimezone(NY_TZ).replace(tzinfo=None)
    return (settle - start).total_seconds() / 60.0 / _MINUTES_PER_YEAR


# ── Rates ─────────────────────────────────────────────────────────────────────

def tbill_discount_to_cc(d: float, days: int = 91) -> float:
    """Convert a 13-week T-bill bank-discount yield (^IRX/100) to a
    continuously-compounded rate: r = -ln(1 - d*days/360) * 365/days.

    4.03% discount ~ 4.11% cc. Same formula as database.tbill_discount_to_cc,
    which the downloader applies before storing: tbill_rates.rate and
    db.get_tbill_rate() are ALREADY cc, so do not apply this to them.
    """
    d = float(d)
    return -math.log(1.0 - d * days / 360.0) * 365.0 / days


# ── Expiry selection ──────────────────────────────────────────────────────────

def select_calibration_expiries(
    listed: Iterable[DateLike],
    session_date: DateLike,
    targets: Sequence[int] = CAL_EXPIRY_TARGETS_DAYS,
    min_days: int = CAL_MIN_EXPIRY_DAYS,
    tolerance: float = CAL_EXPIRY_TOLERANCE,
    neighbours: int = 0,
) -> list[str]:
    """Pick one listed expiry per target tenor.

    For each target (in order) the listed expiry nearest to it in calendar
    days from ``session_date`` is chosen among those with >= ``min_days``
    days, not already used by an earlier target, and within
    ``tolerance * target`` days of it. Targets with no such expiry are
    skipped (never filled twice by the same expiry — that duplicated rows
    and inflated n_spx_options before C22).

    ``neighbours`` > 0 additionally returns that many listed expiries on each
    side of every chosen one (so a bucket survives one bad slice); the
    calibrator re-selects the bucket representative from the cleaned chain.

    Returns sorted ISO date strings.
    """
    sess = _as_date(session_date)
    exp = sorted({_as_date(e) for e in listed})
    days = {e: (e - sess).days for e in exp}
    eligible = [e for e in exp if days[e] >= min_days]
    chosen: list[date] = []
    for tgt in targets:
        cands = [e for e in eligible if e not in chosen
                 and abs(days[e] - tgt) <= tolerance * tgt]
        if not cands:
            continue
        chosen.append(min(cands, key=lambda e: (abs(days[e] - tgt), days[e])))
    out = set(chosen)
    if neighbours > 0:
        for e in chosen:
            i = eligible.index(e)
            for j in range(max(0, i - neighbours), min(len(eligible), i + neighbours + 1)):
                out.add(eligible[j])
    return [e.isoformat() for e in sorted(out)]


# ── Black-76 helpers (vectorised) ─────────────────────────────────────────────

def black76_price(F: ArrayLike, K: ArrayLike, T: ArrayLike, D: ArrayLike, vol: ArrayLike,
                  is_call: Union[bool, np.ndarray]) -> np.ndarray:
    """Discounted Black-76 price, vectorised."""
    F, K, T, D, vol = (np.asarray(x, dtype=float) for x in (F, K, T, D, vol))
    sd = np.maximum(vol * np.sqrt(np.maximum(T, 0.0)), 1e-300)
    d1 = np.log(F / K) / sd + 0.5 * sd
    d2 = d1 - sd
    call = D * (F * ndtr(d1) - K * ndtr(d2))
    put = D * (K * ndtr(-d2) - F * ndtr(-d1))
    out: np.ndarray = np.where(np.asarray(is_call, dtype=bool), call, put)
    return out


def black76_vega(F: ArrayLike, K: ArrayLike, T: ArrayLike, D: ArrayLike,
                 vol: ArrayLike) -> np.ndarray:
    """dPrice/dvol (per unit vol, i.e. per 100 vol points), vectorised."""
    F, K, T, D, vol = (np.asarray(x, dtype=float) for x in (F, K, T, D, vol))
    sd = np.maximum(vol * np.sqrt(np.maximum(T, 0.0)), 1e-300)
    d1 = np.log(F / K) / sd + 0.5 * sd
    out: np.ndarray = D * F * np.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi) * np.sqrt(T)
    return out


def black76_implied_vol(price: ArrayLike, F: ArrayLike, K: ArrayLike, T: ArrayLike,
                        D: ArrayLike, is_call: Union[bool, np.ndarray],
                        lo: float = 1e-4, hi: float = 5.0, n_iter: int = 80) -> np.ndarray:
    """Vectorised Black-76 implied vol by bisection on the OTM-equivalent price.

    The price is first mapped by parity to the out-of-the-money option at the
    same strike (pure time value), which keeps precision for small prices.
    Returns NaN where the time value is <= 1e-10 * D*F or at/above the
    no-arbitrage upper bound, or where no root lies in [lo, hi].
    """
    price, F, K, T, D = (np.asarray(x, dtype=float) for x in (price, F, K, T, D))
    is_call = np.asarray(is_call, dtype=bool)
    price, F, K, T, D, is_call = np.broadcast_arrays(price, F, K, T, D, is_call)
    otm_call = K >= F
    parity = D * (F - K)                       # C - P
    otm = np.where(otm_call, np.where(is_call, price, price + parity),
                   np.where(is_call, price - parity, price))
    upper = np.where(otm_call, D * F, D * K)
    ok = (np.isfinite(otm) & (T > 0) & (K > 0) & (F > 0)
          & (otm > 1e-10 * D * F) & (otm < upper))
    a = np.full(otm.shape, lo)
    b = np.full(otm.shape, hi)
    f_hi = black76_price(F, K, T, D, b, otm_call) - otm
    ok &= f_hi > 0
    for _ in range(n_iter):
        m = 0.5 * (a + b)
        fm = black76_price(F, K, T, D, m, otm_call) - otm
        a = np.where(fm < 0, m, a)
        b = np.where(fm < 0, b, m)
    iv = 0.5 * (a + b)
    out: np.ndarray = np.where(ok & (iv > lo * 1.0001), iv, np.nan)
    return out


# ── Root / settlement handling ────────────────────────────────────────────────

def contract_root(symbol: object) -> str:
    """OCC root of a Yahoo contract symbol, e.g. 'SPXW261016C07710000' -> 'SPXW'."""
    s = str(symbol or "")
    i = 0
    while i < len(s) and s[i].isalpha():
        i += 1
    return s[:i].upper()


def settlement_of_root(root: str) -> str:
    """'AM' for the SOQ-settled SPX monthly root, 'PM' otherwise (SPXW)."""
    return "AM" if str(root).upper() == "SPX" else "PM"


# ── Chain normalisation ───────────────────────────────────────────────────────

def _normalise(raw: pd.DataFrame) -> pd.DataFrame:
    """Map a yfinance chain (or a stored options_snapshots frame) onto one
    schema: expiry, strike, right, bid, ask, root, yahoo_iv, volume,
    open_interest. lastPrice is deliberately dropped."""
    df = raw.copy()
    out = pd.DataFrame(index=df.index)
    if "expiry" in df.columns:
        out["expiry"] = pd.to_datetime(df["expiry"]).dt.strftime("%Y-%m-%d")
    elif "expiration" in df.columns:
        out["expiry"] = pd.to_datetime(df["expiration"]).dt.strftime("%Y-%m-%d")
    else:
        raise ValueError("option chain has no expiry/expiration column")
    out["strike"] = pd.to_numeric(df["strike"], errors="coerce")
    if "right" in df.columns:
        out["right"] = df["right"].astype(str).str[0].str.upper()
    elif "option_type" in df.columns:
        out["right"] = df["option_type"].astype(str).str[0].str.upper()
    else:
        raise ValueError("option chain has no right/option_type column")
    for col in ("bid", "ask"):
        out[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    if "contractSymbol" in df.columns:
        out["root"] = df["contractSymbol"].map(contract_root)
    elif "root" in df.columns:
        out["root"] = df["root"].astype(str).str.upper()
    else:
        out["root"] = ""                 # unknown (stored snapshots predate C22)
    # Yahoo's own IV, as a diagnostic only. A stored snapshot's implied_vol
    # column is not read here: in a chain stored by observed_quotes() it holds
    # this module's Black-76 IV, not Yahoo's.
    if "impliedVolatility" in df.columns:
        out["yahoo_iv"] = pd.to_numeric(df["impliedVolatility"], errors="coerce")
    elif "yahoo_iv" in df.columns:
        out["yahoo_iv"] = pd.to_numeric(df["yahoo_iv"], errors="coerce")
    else:
        out["yahoo_iv"] = np.nan
    if "contractSymbol" in df.columns or "contract_symbol" in df.columns:
        out["contract_symbol"] = df.get("contractSymbol", df.get("contract_symbol"))
    out["volume"] = pd.to_numeric(df["volume"], errors="coerce") if "volume" in df.columns else np.nan
    oi = "openInterest" if "openInterest" in df.columns else "open_interest"
    out["open_interest"] = pd.to_numeric(df[oi], errors="coerce") if oi in df.columns else np.nan
    out = out.dropna(subset=["strike"])
    return out[out["right"].isin(["C", "P"])].reset_index(drop=True)


def _resolve_settlement(df: pd.DataFrame) -> pd.DataFrame:
    """One settlement style per expiry: keep PM (SPXW) lines when an expiry has
    any; an AM-only expiry keeps its AM lines (T measured to 09:30 ET).
    Rows with an unknown root are treated as PM (stored pre-C22 snapshots)."""
    df = df.copy()
    df["settlement"] = np.where(df["root"] == "SPX", "AM", "PM")
    has_pm = df.groupby("expiry")["settlement"].transform(lambda s: (s == "PM").any())
    keep = (df["settlement"] == "PM") | ~has_pm
    return df[keep].reset_index(drop=True)


# ── Parity forward / discount ─────────────────────────────────────────────────

def _parity_fit(pairs: pd.DataFrame, T: float, spot: float,
                r_ref: Optional[float]) -> dict[str, Any]:
    """Estimate (F, D) for one expiry from two-sided C/P pairs at common strikes.

    (C - P) = D*F - D*K is linear in K. Weighted least squares on near-ATM
    strikes, each pair weighted by 1/(spread_C + spread_P)^2 (a wide deep-ITM
    quote must not move the forward), gives D = -slope and F = intercept / D.
    Pairs whose residual exceeds 3x their half-spread sum are excluded
    iteratively (stale strikes). If the regression discount is implausible (implied r_T outside
    [-2%, 12%], or more than PARITY_RATE_TOL from the reference rate when one
    is given — short expiries span too few strikes to pin the slope)
    the discount is fixed at exp(-r_ref*T) and F is the weighted median of
    K + (C - P)/D — recorded as method 'parity_fixed_D'.
    """
    K = pairs["strike"].to_numpy(float)
    y = (pairs["mid_C"] - pairs["mid_P"]).to_numpy(float)
    unc = (pairs["spr_C"] + pairs["spr_P"]).to_numpy(float) / 2.0
    unc = np.maximum(unc, 0.05)                 # 1 tick floor
    band = float(np.clip(0.30 * math.sqrt(max(T, 1e-6)), 0.02, 0.10))
    near = np.abs(np.log(K / spot)) <= band
    if near.sum() < MIN_PARITY_PAIRS:
        near = np.zeros(len(K), dtype=bool)
        near[np.argsort(np.abs(K - spot))[:max(MIN_PARITY_PAIRS, 6)]] = True
    Kn, yn, un = K[near], y[near], unc[near]
    if len(Kn) < 2:
        return {"ok": False, "reason": "fewer than 2 parity pairs"}

    def wls(Kx: np.ndarray, yx: np.ndarray, ux: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        w = 1.0 / ux
        A = np.vstack([np.ones_like(Kx), Kx]).T
        coef, *_ = np.linalg.lstsq(A * w[:, None], yx * w, rcond=None)
        return coef, yx - A @ coef

    # Robust start: a line through the weighted median forward at a nominal
    # discount, then iterate WLS on the pairs consistent with the current
    # line. Yahoo leaves some strikes with stale quotes that violate parity
    # by 20-80 index points; one OLS pass is pulled off by them.
    D0 = math.exp(-(0.04 if r_ref is None else r_ref) * T)
    x0 = Kn + yn / D0
    order = np.argsort(x0)
    cw = np.cumsum((1.0 / un ** 2)[order])
    F0 = float(x0[order][np.searchsorted(cw, 0.5 * cw[-1])])
    res = yn - D0 * (F0 - Kn)
    keep = np.abs(res) <= 3.0 * un + 0.03 * T * np.abs(F0 - Kn) + 0.5
    for _ in range(5):
        if keep.sum() < 3:
            keep = np.ones(len(Kn), dtype=bool)
        coef, _r = wls(Kn[keep], yn[keep], un[keep])
        res = yn - (coef[0] + coef[1] * Kn)
        new_keep = np.abs(res) <= 3.0 * un
        if new_keep.sum() < 3 or (new_keep == keep).all():
            break
        keep = new_keep
    Kn, yn, un = Kn[keep], yn[keep], un[keep]
    res = res[keep]
    D = -coef[1]
    method = "parity_regression"
    r_T = -math.log(D) / T if D > 0 else float("nan")
    plausible = bool(np.isfinite(r_T)) and -0.02 <= r_T <= 0.12
    if plausible and r_ref is not None:
        plausible = abs(r_T - r_ref) <= PARITY_RATE_TOL
    if plausible:
        F = coef[0] / D
    else:
        if r_ref is None:
            return {"ok": False, "reason": f"parity discount implausible (r_T={r_T:.4f})"}
        D = math.exp(-r_ref * T)
        x = Kn + yn / D
        order = np.argsort(x)
        cw = np.cumsum((1.0 / un ** 2)[order])
        F = float(x[order][np.searchsorted(cw, 0.5 * cw[-1])])
        res = yn - D * (F - Kn)
        method = "parity_fixed_D"
        r_T = r_ref
    if not (0.8 * spot < F < 1.25 * spot):
        return {"ok": False, "reason": f"parity forward implausible (F={F:.1f})"}
    return {
        "ok": True, "forward": float(F), "discount": float(D),
        "r": float(r_T), "q": float(r_T - math.log(F / spot) / T),
        "method": method, "n_pairs": int(len(Kn)),
        # spread-weighted RMS residual (index points) of the kept pairs
        "resid_rmse": (float(np.sqrt(np.sum(res ** 2 / un ** 2) / np.sum(1.0 / un ** 2)))
                       if len(res) else float("nan")),
    }


# ── Clean chain ───────────────────────────────────────────────────────────────

@dataclass
class CleanChain:
    """Result of clean_option_chain().

    quotes      : two-sided, single-settlement quotes with own Black-76 IVs
                  (columns: expiry, strike, right, root, settlement, bid, ask,
                  mid_price, rel_spread, time_to_expiry, forward, discount,
                  r_T, q_T, implied_vol, yahoo_iv, otm, log_moneyness,
                  volume, open_interest, snapshot_date)
    carry       : one row per usable expiry (expiry, T, forward, discount,
                  r, q, method, n_pairs, resid_rmse)
    diagnostics : counts, the quote-state gate inputs and the T origin
                  (quote_time, t_origin, t_origin_basis)
    ok, reason  : False (with a reason) when the chain must not be used
    """
    quotes: pd.DataFrame
    carry: pd.DataFrame
    diagnostics: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    reason: str = "ok"


_QUOTE_COLS = ["expiry", "strike", "right", "root", "settlement", "bid", "ask",
               "mid_price", "rel_spread", "time_to_expiry", "forward", "discount",
               "r_T", "q_T", "implied_vol", "yahoo_iv", "otm", "log_moneyness",
               "volume", "open_interest", "snapshot_date"]


def clean_option_chain(
    raw: Optional[pd.DataFrame],
    session_date: DateLike,
    spot: float,
    r_ref: float | Callable[[float], float] | None = None,
    max_rel_spread: float = MAX_REL_SPREAD,
    min_two_sided_share: float = MIN_TWO_SIDED_SHARE,
    atm_band: float = ATM_GATE_BAND,
    min_days: int = CAL_MIN_EXPIRY_DAYS,
    quote_time: QuoteTimeLike = None,
) -> CleanChain:
    """Clean a raw SPX chain into quotes the calibrator can trust.

    Parameters
    ----------
    raw          : yfinance chain rows (contractSymbol, strike, bid, ask,
                   impliedVolatility, expiration, option_type, ...) or a
                   stored options_snapshots frame (expiry, right, bid, ask, ...)
    session_date : the NYSE session these quotes are labelled with (the last
                   closed session when they were downloaded)
    spot         : SPX close of that session (used for the near-ATM bands
                   and to express carry as q_T; pricing uses F_T, D_T)
    r_ref        : reference continuously-compounded rate, used only when an
                   expiry's parity discount is implausible: a number (one rate
                   for every expiry, e.g. ^IRX) or a function r(T) of the
                   expiry's T in years (a term curve, e.g.
                   rates.ZeroCurve.zero_rate), evaluated at each expiry
    quote_time   : when the quotes were observed (UTC; ISO string or
                   datetime). None: the frame's ``quote_time`` column if it
                   has one (stored snapshots), else unknown. It sets the T
                   origin (``t_origin``): the quote time for a Global Trading
                   Hours book, the session's 16:00 ET close otherwise.
    """
    sess = _as_date(session_date)
    if quote_time is None:
        quote_time = _chain_quote_time(raw)
    qt = _as_instant(quote_time)
    origin = t_origin(qt, sess)
    gth = origin != session_close(sess)
    diag = {"session": str(sess), "n_raw": 0 if raw is None else int(len(raw)),
            "r_ref": (None if r_ref is None else "r(T) per expiry" if callable(r_ref)
                      else float(r_ref)),
            "quote_time": None if qt is None else _db.utc_iso(qt),
            "t_origin": origin.isoformat(timespec="seconds"),
            "t_origin_basis": "quote_time (Cboe GTH book)" if gth else "session close 16:00 ET"}
    empty = CleanChain(pd.DataFrame(columns=_QUOTE_COLS),
                       pd.DataFrame(columns=["expiry", "T", "forward", "discount",
                                             "r", "q", "method", "n_pairs",
                                             "resid_rmse"]),
                       diag, False, "empty option chain")
    if raw is None or len(raw) == 0:
        return empty
    df = _normalise(raw)
    df = _resolve_settlement(df)
    # The default (session close) keeps year_fraction's whole-day arithmetic,
    # so a chain without a GTH quote time cleans bit for bit as before.
    t0 = origin if gth else None
    df["time_to_expiry"] = [year_fraction(sess, e, s, origin=t0)
                            for e, s in zip(df["expiry"], df["settlement"])]
    df = df[df["time_to_expiry"] >= min_days / 365.0].copy()
    diag["n_after_expiry_filter"] = int(len(df))

    bid, ask = df["bid"].to_numpy(float), df["ask"].to_numpy(float)
    with np.errstate(invalid="ignore", divide="ignore"):
        mid = 0.5 * (bid + ask)
        rel = (ask - bid) / mid
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid) & np.isfinite(rel)
    df["mid_price"] = np.where(two_sided, mid, np.nan)
    df["rel_spread"] = np.where(two_sided, rel, np.nan)

    near_atm = np.abs(df["strike"].to_numpy(float) / spot - 1.0) < atm_band
    n_near = int(near_atm.sum())
    share = float(two_sided[near_atm].mean()) if n_near else 0.0
    diag.update(n_two_sided=int(two_sided.sum()), n_near_atm=n_near,
                near_atm_two_sided_share=share)

    if n_near == 0 or share < min_two_sided_share:
        reason = (f"SPX book has no live two-sided quotes: {share:.0%} of {n_near} "
                  f"near-ATM lines (|K/S-1|<{atm_band:.0%}) are two-sided "
                  f"(< {min_two_sided_share:.0%})")
        empty.reason = reason
        return empty

    df = df[two_sided & (df["rel_spread"].to_numpy(float) <= max_rel_spread)].copy()
    diag["n_two_sided_within_spread"] = int(len(df))

    carry_rows, parts, n_stale = [], [], 0
    for exp, g in df.groupby("expiry"):
        T = float(g["time_to_expiry"].iloc[0])
        g = g.assign(spread=g["ask"] - g["bid"])
        c = (g[g["right"] == "C"].groupby("strike")[["mid_price", "spread"]].mean()
             .rename(columns={"mid_price": "mid_C", "spread": "spr_C"}))
        p = (g[g["right"] == "P"].groupby("strike")[["mid_price", "spread"]].mean()
             .rename(columns={"mid_price": "mid_P", "spread": "spr_P"}))
        pairs = pd.concat([c, p], axis=1, join="inner").reset_index()
        if len(pairs) < 2:
            continue
        r_ref_T = float(r_ref(T)) if callable(r_ref) else r_ref
        fit = _parity_fit(pairs, T, spot, r_ref_T)
        if not fit["ok"]:
            logger.info("  expiry %s dropped: %s", exp, fit["reason"])
            continue
        carry_rows.append({"expiry": exp, "T": T, **{k: fit[k] for k in
                           ("forward", "discount", "r", "q", "method",
                            "n_pairs", "resid_rmse")}})
        # Parity-consistency filter on EVERY paired strike: a pair that
        # violates C - P = D(F - K) by more than 3x its half-spread sum holds
        # a stale side (which one is unknowable), so both lines are dropped.
        resid = pairs["mid_C"] - pairs["mid_P"] - fit["discount"] * (fit["forward"] - pairs["strike"])
        unc = np.maximum((pairs["spr_C"] + pairs["spr_P"]) / 2.0, 0.05)
        stale = set(pairs.loc[np.abs(resid) > 3.0 * unc + 0.05, "strike"])
        n_stale += 2 * len(stale)
        g = g[~g["strike"].isin(stale)].drop(columns="spread")
        g["forward"], g["discount"] = fit["forward"], fit["discount"]
        g["r_T"], g["q_T"] = fit["r"], fit["q"]
        parts.append(g)

    diag["n_expiries_with_carry"] = len(carry_rows)
    diag["n_dropped_parity_inconsistent"] = int(n_stale)
    if not parts:
        empty.reason = "no expiry with enough two-sided put/call pairs for a parity forward"
        empty.diagnostics = diag
        return empty

    q = pd.concat(parts, ignore_index=True)
    is_call = (q["right"] == "C").to_numpy()
    q["implied_vol"] = black76_implied_vol(q["mid_price"], q["forward"], q["strike"],
                                           q["time_to_expiry"], q["discount"], is_call)
    q["otm"] = np.where(is_call, q["strike"] >= q["forward"], q["strike"] < q["forward"])
    q["log_moneyness"] = np.log(q["strike"] / q["forward"])
    q["snapshot_date"] = sess.isoformat()
    q = q.dropna(subset=["implied_vol"])
    diag["n_clean"] = int(len(q))
    carry = pd.DataFrame(carry_rows).sort_values("T").reset_index(drop=True)
    q = q.sort_values(["time_to_expiry", "strike", "right"]).reset_index(drop=True)
    return CleanChain(q[_QUOTE_COLS], carry, diag, True, "ok")


def observed_quotes(raw: pd.DataFrame, clean: Optional[CleanChain] = None,
                    spot: Optional[float] = None, source: str = "yahoo") -> pd.DataFrame:
    """The session's observed two-sided quotes, in the options_snapshots schema.

    This is what the live pipeline stores for an ACCEPTED session: every line
    of the downloaded chain with bid > 0 and ask >= bid, after the one-
    settlement-per-expiry choice (the AM lines a PM expiry never uses are left
    out), with mid = (bid + ask)/2. It is taken BEFORE the spread cap and the
    parity-consistency filter on purpose: clean_option_chain() of these rows
    reproduces ``clean`` exactly, so a refit from the stored snapshot
    reproduces the live fit. The cleaned subset itself is not a fixed point of
    the cleaner (the parity regression on fewer pairs lands on a slightly
    different discount; on the 2026-10-16 book the 14-day r_T moved from
    3.49% to 2.96% and 6 quotes changed), so storing it would break that.

    No lastPrice and no Yahoo IV are stored. ``implied_vol`` is this module's
    Black-76 IV for the quotes that survived cleaning (``clean``) and NaN for
    the rest; ``underlying_price`` is the session close.
    """
    df = _resolve_settlement(_normalise(raw))
    bid, ask = df["bid"].to_numpy(float), df["ask"].to_numpy(float)
    two_sided = (bid > 0) & (ask > 0) & (ask >= bid)
    df = df[two_sided].copy()
    df["mid_price"] = 0.5 * (df["bid"] + df["ask"])
    key = ["expiry", "strike", "right", "root"]
    if clean is not None and clean.ok and len(clean.quotes):
        iv = clean.quotes[key + ["implied_vol"]]
        df = df.merge(iv, on=key, how="left")
    else:
        df["implied_vol"] = np.nan
    if spot is not None:
        df["underlying_price"] = float(spot)
    df["source"] = source
    cols = [c for c in ("expiry", "strike", "right", "root", "settlement", "bid", "ask",
                        "mid_price", "implied_vol", "volume", "open_interest",
                        "contract_symbol", "underlying_price", "source") if c in df.columns]
    return df[cols].sort_values(["expiry", "right", "strike"]).reset_index(drop=True)


# ── Live download ─────────────────────────────────────────────────────────────

def _with_retries(fn: Callable[[], _T], what: str, attempts: int, pause: float,
                  sleep: Callable[[float], object]) -> _T:
    """fn() with up to ``attempts`` tries (Yahoo fails transiently: empty JSON,
    429s); the last exception propagates. ``attempts`` < 1 is an error (it
    used to return None without calling fn)."""
    if attempts < 1:
        raise ValueError(f"attempts must be >= 1, got {attempts}")
    for i in range(attempts):
        try:
            return fn()
        except Exception as exc:                     # network / parse failure
            if i == attempts - 1:
                raise
            logger.info("  %s failed (%s); retrying in %.0fs", what, exc, pause * (i + 1))
            sleep(pause * (i + 1))
    raise AssertionError("unreachable: the last attempt returns or raises")


def fetch_spx_chain(
    session_date: DateLike,
    ticker_factory: Optional[Callable[[str], Any]] = None,
    symbols: Sequence[str] = ("^SPX", "^GSPC"),
    neighbours: int = 1,
    attempts: int = 3,
    pause: float = 2.0,
    sleep: Optional[Callable[[float], object]] = None,
) -> Tuple[pd.DataFrame, dict[str, Any]]:
    """Download only the calibration expiries (plus neighbours) of the SPX chain.

    Returns (raw DataFrame, meta dict with 'listed', 'requested', 'failed',
    'symbol'). The DataFrame keeps contractSymbol (for the AM/PM root) and
    adds 'expiration' and 'option_type'. lastPrice is downloaded by Yahoo but
    never used. The listing and each expiry get up to ``attempts`` tries (a
    run makes ~18 requests; one transient failure must not cost a slice).
    """
    if ticker_factory is None:
        import yfinance as yf
        ticker_factory = yf.Ticker
    if sleep is None:
        import time as _time
        sleep = _time.sleep
    meta: dict[str, Any] = {"listed": [], "requested": [], "failed": [], "symbol": None}
    tk: Any = None                    # a yfinance Ticker (untyped)
    listed: list[str] = []
    for sym in symbols:
        tk = ticker_factory(sym)
        listed = list(_with_retries(lambda: getattr(tk, "options", None) or [],
                                    f"{sym} expiry listing", attempts, pause, sleep))
        if listed:
            meta["symbol"] = sym
            break
    meta["listed"] = listed
    if not listed:
        return pd.DataFrame(), meta
    req = select_calibration_expiries(listed, session_date, neighbours=neighbours)
    meta["requested"] = req
    frames = []
    for exp in req:
        try:
            chain = _with_retries(lambda: tk.option_chain(exp), f"expiry {exp}",
                                  attempts, pause, sleep)
        except Exception as exc:                     # still failing after the retries
            meta["failed"].append(exp)
            logger.warning("  Expiry %s failed: %s", exp, exc)
            continue
        for side, part in (("call", chain.calls), ("put", chain.puts)):
            part = part.copy()
            part["expiration"] = exp
            part["option_type"] = side
            frames.append(part)
    if not frames:
        return pd.DataFrame(), meta
    return pd.concat(frames, ignore_index=True), meta
