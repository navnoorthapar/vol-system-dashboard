"""
dashboard/app.py — Joint SPX/VIX Volatility System — display layer.

Renders the four dashboard pages from the artifacts under data_store/. It is a
pure READ path: rendering a page never downloads data, never writes a file and
never retrains or re-labels anything. The daily data refresh, the regime-label
extension and the Heston recalibration belong to scripts/cloud_refresh.py; the
static site is produced by dashboard/freeze_site.py.

(An earlier version started "startup" and "every 4 hours" refresh threads from
``__main__``. They downloaded market data into the tracked SQLite DB and then
overwrote the tracked regime_labels.parquet with the DEMOTED classifier's
predictions — the exact defect C21 removed from cloud_refresh — whenever the
local DB was more than three days old, i.e. on every fresh clone.)

A loader that records a failure in ``d["errors"]`` gets a visible error banner,
and freeze_site.py refuses to publish a page that carries one.

Run a local preview (read-only, bound to 127.0.0.1):
    python dashboard/app.py
"""
from __future__ import annotations

import glob
import inspect
import json
import logging
import math
import pickle
import re
import sqlite3
import sys
from pathlib import Path
from typing import Any, Iterable, Optional

import numpy as np
import pandas as pd
from flask import Flask, make_response, render_template, request
from pandas.tseries.holiday import (
    AbstractHolidayCalendar,
    GoodFriday,
    Holiday,
    USLaborDay,
    USMartinLutherKingJr,
    USMemorialDay,
    USPresidentsDay,
    USThanksgivingDay,
    nearest_workday,
    sunday_to_monday,
)

_logger = logging.getLogger("vol_dashboard")

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_store"
sys.path.insert(0, str(ROOT))

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# ── Site map ───────────────────────────────────────────────────────────────────
SITE_URL = "https://navnoorbawa.me/"

# route -> (frozen file name, nav id). freeze_site.py imports this map so the
# frozen file names, the canonical URLs and the sitemap can never drift apart.
ROUTES: dict[str, str] = {
    "/":            "index.html",
    "/calibration": "calibration.html",
    "/greeks":      "greeks.html",
    "/backtest":    "backtest.html",
}


def canonical_url(route: str) -> str:
    """Absolute public URL of a route's frozen page (the home page is the bare root)."""
    fname = ROUTES.get(route, "index.html")
    return SITE_URL if fname == "index.html" else SITE_URL + fname


# ── Staleness thresholds (NYSE sessions) ───────────────────────────────────────
STALE_WARN_SESSIONS = 2     # amber banner from this age on
STALE_ERROR_SESSIONS = 5    # red banner from this age on

# Regime presentation (index = rule label 0/1/2)
_REGIME_NAMES = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
_REGIME_COLORS = ["#3388ff", "#00ff88", "#ff3333"]
_REGIME_LABELS = ["R0", "R1", "R2"]

# The backtest portfolio is the equal-weight mean of these sleeves
# (BacktestEngine.run: pnl_total = (s1c + s3 + s4) / 3). load_page4 VERIFIES
# this against the stored equity curve instead of trusting the constant.
PORTFOLIO_SLEEVES = ("s1c", "s3", "s4")


# ── NYSE session calendar ──────────────────────────────────────────────────────
class NYSEHolidayCalendar(AbstractHolidayCalendar):
    """Full-day NYSE closures from rules (2010 onwards).

    Differs from the US federal calendar: Good Friday closes, Columbus and
    Veterans Day do not; a Saturday New Year's Day is NOT observed on the Friday
    before; Juneteenth closes from 2022. Verified against every SPX session in
    data_store/vol_system.db (2010-01-04 .. 2026-06-18: zero mismatches).
    """
    rules = [
        Holiday("New Year's Day", month=1, day=1, observance=sunday_to_monday),
        USMartinLutherKingJr,
        USPresidentsDay,
        GoodFriday,
        USMemorialDay,
        Holiday("Juneteenth", month=6, day=19, start_date="2022-01-01",
                observance=nearest_workday),
        Holiday("Independence Day", month=7, day=4, observance=nearest_workday),
        USLaborDay,
        USThanksgivingDay,
        Holiday("Christmas Day", month=12, day=25, observance=nearest_workday),
    ]


# One-off closures the rules cannot express (Hurricane Sandy, national days of
# mourning for Presidents G.H.W. Bush and Carter).
_NYSE_SPECIAL_CLOSURES = ("2012-10-29", "2012-10-30", "2018-12-05", "2025-01-09")


def nyse_holidays(start: str = "2010-01-01", end: str = "2030-12-31") -> pd.DatetimeIndex:
    """NYSE full-day holidays (weekday closures) between start and end."""
    hol = NYSEHolidayCalendar().holidays(start, end)
    special = pd.to_datetime(list(_NYSE_SPECIAL_CLOSURES))
    special = special[(special >= pd.Timestamp(start)) & (special <= pd.Timestamp(end))]
    out = hol.union(special)
    return out[out.dayofweek < 5]   # a Saturday New Year's Day closes nothing


def session_age(date: Optional[str], ref: Optional[str]) -> Optional[int]:
    """Number of NYSE sessions in (date, ref]: 0 when date is the ref session.

    Returns None when either date is unknown. A date after ref (possible when
    the local DB lags the committed artifacts) has age 0.
    """
    if not date or not ref:
        return None
    d0 = pd.Timestamp(str(date)[:10])
    d1 = pd.Timestamp(str(ref)[:10])
    if d0 >= d1:
        return 0
    hol = nyse_holidays(str(d0.date()), str(d1.date()))
    return int(np.busday_count(
        (d0 + pd.Timedelta(days=1)).date(), (d1 + pd.Timedelta(days=1)).date(),
        holidays=[h.date() for h in hol],
    ))


def stale_level(age: Optional[int]) -> str:
    """'ok' / 'warn' / 'error' for an age in sessions ('unknown' if None)."""
    if age is None:
        return "unknown"
    if age >= STALE_ERROR_SESSIONS:
        return "error"
    if age >= STALE_WARN_SESSIONS:
        return "warn"
    return "ok"


def _staleness(component: str, label: str, date: Optional[str], ref: Optional[str]) -> dict:
    age = session_age(date, ref)
    return {"component": component, "label": label, "date": date,
            "ref": ref, "age": age, "level": stale_level(age)}


# ── Utilities ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    """Read-only connection: the display layer must never write to the DB.

    mode=ro also refuses to create an empty database when the file is missing,
    so a missing DB surfaces as a loader error instead of an empty page.
    """
    path = (DATA / "vol_system.db").resolve()
    return sqlite3.connect(f"file:{path}?mode=ro", uri=True)


def _record_error(d: dict, component: str, exc: Any) -> None:
    """Attach a loader failure to the page; templates render it as a red banner."""
    msg = str(exc) or exc.__class__.__name__
    d.setdefault("errors", []).append({"component": component, "message": msg})
    _logger.warning("[%s] loader error: %s", component, msg)


def _date_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, (pd.Timestamp, np.datetime64)):
        return str(pd.Timestamp(x).date())
    s = str(x)[:10]
    return s if re.match(r"^\d{4}-\d{2}-\d{2}$", s) else None


def _clean(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python types.

    Non-finite floats become None, so a missing number renders as '—' (and as
    JSON null in chart data) rather than as 'nan' or a fabricated zero.
    """
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return [_clean(v) for v in obj.tolist()]
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        f = float(obj)
        return f if math.isfinite(f) else None
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _num(x: Any) -> Optional[float]:
    """float(x) if x is a finite number, else None."""
    try:
        f = float(x)
    except Exception:          # TypeError/ValueError, or a Jinja Undefined
        return None
    return f if math.isfinite(f) else None


def ordinal(n: Any) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 75 -> '75th' (rounded to an integer)."""
    if _num(n) is None:
        return "—"
    k = int(round(float(n)))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _fmt(value: Any, spec: str = "", default: str = "—") -> str:
    """Jinja filter: format a number, or render `default` when it is missing."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return default
        return format(value, spec)
    return str(value)


def _usd_k(value: Any, signed: bool = True) -> str:
    """Prose dollars in thousands: 693938 -> '+$694K', -599995 -> '−$600K'."""
    x = _num(value)
    if x is None:
        return "—"
    sign = "−" if x < 0 else ("+" if signed else "")
    return f"{sign}${abs(x) / 1000:,.0f}K"


def _pct_prose(value: Any, digits: int = 1, signed: bool = True) -> str:
    """Prose percent of a decimal: -0.2612 -> '−26.1%'."""
    x = _num(value)
    if x is None:
        return "—"
    sign = "−" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}{abs(x) * 100:.{digits}f}%"


app.jinja_env.filters["fmt"] = _fmt
app.jinja_env.filters["ordinal"] = ordinal
app.jinja_env.filters["usd_k"] = _usd_k
app.jinja_env.filters["pct"] = _pct_prose


def _vomma_color(z: Optional[float]) -> str:
    """Map a vomma z-score to a CSS colour for the heatmap."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "#1a1a1a"
    if abs(z) > 2.0:
        return "#cc1111"   # danger red
    if z > 0:
        t = min(z / 2.0, 1.0)
        r = int(17 + (0 - 17) * t)
        g = int(17 + (190 - 17) * t)
        b = int(17 + (90 - 17) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    t = min(abs(z) / 2.0, 1.0)
    r = int(17 + (40 - 17) * t)
    g = int(17 + (90 - 17) * t)
    b = int(17 + (200 - 17) * t)
    return f"#{r:02x}{g:02x}{b:02x}"


# ── Market data readers (DB) ───────────────────────────────────────────────────
def _read_spx() -> pd.DataFrame:
    """SPX closes and log returns, date-indexed, ascending."""
    with _db() as con:
        df = pd.read_sql("SELECT date, close, log_return FROM spx_ohlcv ORDER BY date", con)
    df["date"] = pd.to_datetime(df["date"])
    return df.set_index("date")


def _read_vix_daily() -> pd.Series:
    with _db() as con:
        df = pd.read_sql("SELECT date, close FROM vix_daily ORDER BY date", con)
    return pd.Series(df["close"].values, index=pd.to_datetime(df["date"]), name="vix")


def _read_vix_wide() -> pd.DataFrame:
    """VIX term-structure closes, one column per ticker, date-indexed."""
    with _db() as con:
        df = pd.read_sql("SELECT date, ticker, close FROM vix_term_structure", con)
    df["date"] = pd.to_datetime(df["date"])
    wide = df.pivot_table(index="date", columns="ticker", values="close", aggfunc="last")
    wide.columns.name = None
    return wide.sort_index()


def _latest_market_session() -> Optional[str]:
    """Newest session any dashboard input knows about (DB close or rule label).

    The reference date for every staleness age: an input is 'n sessions old'
    relative to the newest session the site has data for.
    """
    dates = []
    try:
        with _db() as con:
            row = con.execute("SELECT MAX(date) FROM spx_ohlcv").fetchone()
        if row and row[0]:
            dates.append(str(row[0])[:10])
    except Exception as exc:  # reported by the loaders that need the DB
        _logger.info("latest session: DB unavailable (%s)", exc)
    try:
        rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet", columns=["regime"])
        if len(rl):
            dates.append(str(pd.Timestamp(rl.index[-1]).date()))
    except Exception as exc:
        _logger.info("latest session: regime labels unavailable (%s)", exc)
    return max(dates) if dates else None


# ── Calibration selection ──────────────────────────────────────────────────────
def _calibration_session(cal: dict, path: str | Path) -> Optional[str]:
    """The market session a calibration describes (spot_date > as_of_date > file name)."""
    for key in ("spot_date", "as_of_date"):
        s = _date_str(cal.get(key))
        if s:
            return s
    m = re.search(r"(\d{4}-\d{2}-\d{2})", Path(path).name)
    return m.group(1) if m else None


def _cal_losses(cal: dict) -> dict:
    return cal.get("leg_losses") or cal.get("losses") or {}


def calibration_gate(cal: dict) -> tuple[bool, str]:
    """Apply joint_calibrator.is_acceptable_calibration to a stored result.

    The gate is the single source of truth for what may be published, and its
    signature grows as checks are added (SPX data sufficiency, snapshot
    freshness, ...). Its arguments are therefore bound BY NAME from the stored
    result: `params` <- result["params"], `spx_iv_rmse` <- the SPX leg RMSE, a
    parameter called `result`/`cal`/`calibration` <- the whole dict, and any
    other parameter <- the result key of the same name. A new gate argument is
    picked up here without editing the dashboard.
    """
    from joint_vol_calibration.calibration import joint_calibrator as jc

    gate = jc.is_acceptable_calibration
    losses = _cal_losses(cal)
    args, kwargs = [], {}
    for name, p in inspect.signature(gate).parameters.items():
        if p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD):
            continue
        if name == "params":
            val, have = cal.get("params", {}), True
        elif name in ("result", "cal", "calibration"):
            val, have = cal, True
        elif name in ("spx_iv_rmse", "spx_rmse"):
            val = losses.get("spx_iv_rmse")
            have = val is not None
        elif name in cal:
            val, have = cal[name], True
        elif name in losses:
            val, have = losses[name], True
        else:
            val, have = None, False
        if not have:
            if p.default is p.empty:
                return False, f"result lacks '{name}', required by the quality gate"
            continue
        if p.kind is p.POSITIONAL_ONLY:
            args.append(val)
        else:
            kwargs[name] = val
    ok, reason = gate(*args, **kwargs)
    return bool(ok), str(reason)


def calibration_is_legacy(cal: dict) -> Optional[bool]:
    """joint_calibrator.calibration_is_legacy(cal) when the library provides it.

    Returns None (unknown) on library versions without the helper.
    """
    try:
        from joint_vol_calibration.calibration import joint_calibrator as jc
        fn = getattr(jc, "calibration_is_legacy", None)
        return bool(fn(cal)) if callable(fn) else None
    except Exception as exc:
        _logger.info("calibration_is_legacy unavailable (%s)", exc)
        return None


def latest_good_calibration_path(cal_dir: str | Path | None = None) -> Optional[str]:
    """Return the NEWEST joint_cal_*.pkl that passes the quality gate, or None.

    Selection is by market session (spot_date, else as_of_date, else the date
    in the file name), never by RMSE. Ranking archived fits by RMSE is a
    selection bias that rewards thin data: a one-tenor VIX leg (2026-07-23,
    0.049 vp) and later a one-expiry SPX leg (2026-09-11, 0.215 vp; 2026-09-25,
    0.041 vp) each scored the best RMSE on record precisely because the data
    had broken, and an RMSE-ranked selector headlined them over newer, better
    identified fits. What may be published is decided by the gate
    (``calibration_gate``); the dashboard shows the latest such fit together
    with its age, and never falls back to a fit the gate rejected.
    """
    cal_dir = Path(cal_dir) if cal_dir is not None else DATA / "calibrations"
    candidates = []
    for path in glob.glob(str(cal_dir / "joint_cal_*.pkl")):
        try:
            with open(path, "rb") as f:
                cal = pickle.load(f)
            ok, reason = calibration_gate(cal)
        except Exception as exc:
            _logger.warning("calibration %s unreadable: %s", Path(path).name, exc)
            continue
        session = _calibration_session(cal, path)
        if ok and session:
            candidates.append((session, str(cal.get("run_date") or ""), path))
        elif not ok:
            _logger.info("calibration %s rejected by gate: %s", Path(path).name, reason)
    if not candidates:
        return None
    return max(candidates)[2]


def load_headline_calibration() -> tuple[str, dict]:
    """(path, result) of the headline calibration; raises when none is acceptable."""
    path = latest_good_calibration_path()
    if path is None:
        raise FileNotFoundError(
            "no joint_cal_*.pkl in data_store/calibrations passes the quality gate"
        )
    with open(path, "rb") as f:
        return path, pickle.load(f)


# ── Heston model smile (calibration page) ──────────────────────────────────────
SMILE_MATURITIES_DAYS = (14, 30, 60, 91, 180, 365)
SMILE_LOG_MONEYNESS = np.round(np.linspace(-0.30, 0.25, 23), 4)
# OTM forward prices below this (as a fraction of the forward) are dropped:
# inverting a price that small turns pricer round-off into a fake implied vol
# (the old Greeks-surface smile showed a 30-day "IV" of 5.0% that way).
SMILE_PRICE_FLOOR = 1e-6


def heston_model_smile(params: dict,
                       maturities_days: Iterable[int] = SMILE_MATURITIES_DAYS,
                       log_moneyness: Iterable[float] = SMILE_LOG_MONEYNESS) -> list[dict]:
    """Heston implied-vol smiles of `params` against ln(K/F).

    With deterministic rates the forward-normalised Heston smile does not
    depend on r or q, so it is priced with S = F = 1 and r = q = 0 and needs
    nothing beyond the five parameters. OTM prices (puts below the forward,
    calls above) come from the vectorised Carr-Madan pricer on a converged
    integration grid and are inverted with Black-Scholes; nodes whose OTM price
    is below SMILE_PRICE_FLOOR are omitted rather than shown.
    """
    from joint_vol_calibration.models.heston import heston_call_batch, implied_vol_from_price

    k = np.asarray(list(log_moneyness), dtype=float)
    K = np.exp(k)
    out = []
    for T_days in maturities_days:
        T = float(T_days) / 365.0
        calls = heston_call_batch(
            1.0, K, T, 0.0, 0.0,
            params["kappa"], params["theta"], params["sigma"], params["rho"], params["v0"],
            n_points=4096, v_upper=1000.0,
        )
        otm = np.where(k < 0, calls - (1.0 - K), calls)
        lm, iv = [], []
        for ki, Ki, price in zip(k, K, otm):
            if not np.isfinite(price) or price < SMILE_PRICE_FLOOR:
                continue
            vol = implied_vol_from_price(float(price), 1.0, float(Ki), T, 0.0, 0.0,
                                         "P" if ki < 0 else "C", tol=1e-10)
            if vol is not None:
                lm.append(round(float(ki), 4))
                iv.append(float(vol))
        out.append({"T": int(T_days), "lm": lm, "iv": iv})
    return out


# ── PAGE 1 DATA: Live Market Overview ─────────────────────────────────────────
def load_page1() -> dict[str, Any]:
    d: dict[str, Any] = {}

    # Bug 3: Load regime labels once — used for regime_date and PDV spread
    _rl: pd.DataFrame | None = None
    _rl_max_date: str | None = None
    try:
        _rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
        if len(_rl) > 0:
            _rl_max_date = (
                str(_rl.index[-1].date())
                if hasattr(_rl.index[-1], "date")
                else str(_rl.index[-1])[:10]
            )
    except Exception:
        pass

    # ── SPX latest ──────────────────────────────────────────────────────────
    try:
        con = _db()
        spx = pd.read_sql(
            "SELECT date, close FROM spx_ohlcv ORDER BY date DESC LIMIT 2", con
        )
        d["spx"]       = round(float(spx["close"].iloc[0]), 2)
        d["spx_prev"]  = round(float(spx["close"].iloc[1]), 2)
        d["spx_date"]  = str(spx["date"].iloc[0])
        d["spx_chg"]   = round((d["spx"] - d["spx_prev"]) / d["spx_prev"] * 100, 2)
        d["spx_up"]    = d["spx"] >= d["spx_prev"]

        # VIX term structure — latest value per ticker (may differ in date)
        vts_sql = """
            SELECT t.ticker, t.tenor_days, t.close, t.date, t.is_vvix
            FROM vix_term_structure t
            INNER JOIN (
                SELECT ticker, MAX(date) as max_date
                FROM vix_term_structure
                GROUP BY ticker
            ) m ON t.ticker = m.ticker AND t.date = m.max_date
            ORDER BY t.is_vvix, t.tenor_days
        """
        vts = pd.read_sql(vts_sql, con)
        con.close()

        # Bug 1: Query vix_daily for authoritative VIX spot and date
        try:
            _con_vd = _db()
            _vix_row = pd.read_sql(
                "SELECT date, close FROM vix_daily ORDER BY date DESC LIMIT 1", _con_vd
            )
            _con_vd.close()
            if len(_vix_row) > 0:
                d["vix"]      = round(float(_vix_row["close"].iloc[0]), 2)
                d["vix_date"] = str(_vix_row["date"].iloc[0])
        except Exception:
            pass

        # Build term structure dict
        vix_curve: dict[str, float] = {}
        for _, row in vts.iterrows():
            v = float(row["close"])
            if v < 5:
                v *= 100  # fraction → points
            if row["is_vvix"]:
                d["vvix"] = round(v, 2)
                d["vvix_date"] = str(row["date"])
            else:
                vix_curve[int(row["tenor_days"])] = round(v, 2)

        # Bug 2: Record max date across all VTS tickers
        if not vts.empty:
            d["ts_max_date"] = str(vts["date"].max())[:10]

        # Named references — Bug 1: use setdefault so vix_daily query above takes priority
        d.setdefault("vix", vix_curve.get(30, vix_curve.get(9, 0)))
        d["vix9d"] = vix_curve.get(9,  None)
        d["vix3m"] = vix_curve.get(93, None)
        d["vix6m"] = vix_curve.get(182, None)
        d.setdefault("vvix", 0)

        # Chart data — term structure
        ts_pts = [(t, v) for t, v in sorted(vix_curve.items())]
        d["ts_tenors"] = [f"{t}d" for t, _ in ts_pts]
        d["ts_values"] = [v for _, v in ts_pts]

        # Contango check
        if len(ts_pts) >= 2:
            d["ts_regime"] = "CONTANGO" if ts_pts[-1][1] > ts_pts[0][1] else "BACKWARDATION"
            d["ts_regime_color"] = "#00ff88" if d["ts_regime"] == "CONTANGO" else "#ff3333"
        else:
            d["ts_regime"] = "N/A"
            d["ts_regime_color"] = "#666"

    except Exception as e:
        d["db_error"] = str(e)
        for k in ["spx", "vix", "vvix", "spx_chg"]:
            d.setdefault(k, 0)
        d["ts_tenors"] = []
        d["ts_values"] = []
        d["ts_regime"] = "N/A"
        d["ts_regime_color"] = "#666"

    # ── Regime — live inference from current market data ───────────────────
    try:
        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
            clf = pickle.load(f)

        # SPX history for realized-vol features
        con2 = _db()
        spx_hist = pd.read_sql(
            "SELECT date, close FROM spx_ohlcv ORDER BY date DESC LIMIT 30", con2
        )
        con2.close()
        spx_hist = spx_hist.sort_values("date").reset_index(drop=True)
        spx_rets = spx_hist["close"].pct_change().dropna()

        rv_20d    = float(spx_rets.iloc[-20:].std()  * np.sqrt(252)) if len(spx_rets) >= 20 else 0.15
        rv_5d     = float(spx_rets.iloc[-5:].std()   * np.sqrt(252)) if len(spx_rets) >= 5  else rv_20d
        rv_5d_lag = float(spx_rets.iloc[-10:-5].std()* np.sqrt(252)) if len(spx_rets) >= 10 else rv_5d

        # Features in same units as training data (VIX/100, VVIX/100)
        vix_dec  = (d.get("vix")  or 0) / 100
        vix3m_dec= (d.get("vix3m") or d.get("vix") or 0) / 100
        vvix_dec = (d.get("vvix") or 0) / 100
        pdv_dec  = (d.get("pdv_forecast") or 0) / 100

        feat_live = pd.DataFrame([{
            "vix":          vix_dec,
            "ts_slope":     vix3m_dec - vix_dec,          # (VIX3M - VIX)/100; < 0 = backwardation
            "fear_premium": vix_dec / rv_20d if rv_20d > 0 else 1.0,
            "rv_change_5d": rv_5d - rv_5d_lag,
            "pdv_iv_spread":pdv_dec - vix_dec,            # PDV - IV; < 0 = IV > PDV
            "vvix":         vvix_dec,                     # > 1.0 triggers R2
        }])

        proba   = clf.predict_proba(feat_live)[0]
        clf_reg = int(np.argmax(proba))

        names  = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
        colors = ["#3388ff", "#00ff88", "#ff3333"]
        labels = ["R0", "R1", "R2"]

        # HEADLINE regime = the deterministic RULE-BASED label (rv_20d vs VIX;
        # VVIX > threshold). These are the labels the backtest actually trades on.
        # C16/C17: the ML classifier scores 63.4% OOS and LOSES to the trivial
        # persistence baseline (y_t = y_{t-1}, 90.0% on 2020+), so it is demoted
        # to research-only — we do NOT headline its prediction. It is shown
        # separately below, clearly caveated. (Showing the classifier's call as
        # the headline previously contradicted both the rule label and the
        # displayed VVIX state.)
        try:
            reg = int(_rl["regime"].iloc[-1])
        except Exception:
            reg = clf_reg
        d["regime"]       = reg
        d["regime_name"]  = names[reg]
        d["regime_color"] = colors[reg]
        d["regime_label"] = labels[reg]
        d["regime_date"]  = _rl_max_date or d.get("spx_date", "live")  # Bug 3

        # ML classifier — RESEARCH ONLY (demoted); shown with a caveat.
        d["clf_label"]    = labels[clf_reg]
        d["clf_name"]     = names[clf_reg]
        d["clf_color"]    = colors[clf_reg]
        d["clf_conf"]     = round(float(proba[clf_reg]) * 100, 1)
        d["clf_agrees"]   = (clf_reg == reg)
        d["prob_r0"]      = round(float(proba[0]) * 100, 1)
        d["prob_r1"]      = round(float(proba[1]) * 100, 1)
        d["prob_r2"]      = round(float(proba[2]) * 100, 1)
        d["clf_accuracy"] = "63.4%"

        # Regime frequency card: computed from the shipped labels, never
        # hard-coded. The previous constants were stale by ~2x on R2 and
        # contradicted the backtest page's own adaptive-gate story.
        try:
            _rc = _rl["regime"].astype(int)
            _tot = int(len(_rc))
            d["reg_total"] = _tot
            for _k in (0, 1, 2):
                _n = int((_rc == _k).sum())
                d[f"reg{_k}_n"] = _n
                d[f"reg{_k}_pct"] = round(100.0 * _n / _tot, 1) if _tot else 0.0
            d["reg_span"] = f"{_rc.index[0].year} – {_rc.index[-1].year}"
        except Exception:
            pass
    except Exception as e:
        d["regime_error"] = str(e)
        _logger.warning("Live regime inference failed (%s); falling back to cached labels.", e)
        # Fallback: read last known regime from regime_labels.parquet so the
        # page always shows a valid regime and a non-blank "As of" date.
        _cached = False
        try:
            rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
            if "regime" in rl.columns and len(rl) > 0:
                reg = int(rl["regime"].iloc[-1])
                cache_dt = (
                    str(rl.index[-1].date())
                    if hasattr(rl.index[-1], "date")
                    else str(rl.index[-1])[:10]
                )
                if 0 <= reg <= 2:
                    _names  = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
                    _colors = ["#3388ff", "#00ff88", "#ff3333"]
                    _labels = ["R0", "R1", "R2"]
                    d["regime"]       = reg
                    d["regime_name"]  = _names[reg]
                    d["regime_color"] = _colors[reg]
                    d["regime_label"] = _labels[reg]
                    d["regime_conf"]  = 0
                    d["regime_date"]  = f"{cache_dt} (cached)"
                    d["prob_r0"] = 100.0 if reg == 0 else 0.0
                    d["prob_r1"] = 100.0 if reg == 1 else 0.0
                    d["prob_r2"] = 100.0 if reg == 2 else 0.0
                    _cached = True
                    _logger.info("Regime fallback: R%d from %s", reg, cache_dt)
        except Exception as fe:
            _logger.error("Regime cache fallback also failed: %s", fe)

        if not _cached:
            d["regime"]       = -1
            d["regime_name"]  = "UNKNOWN"
            d["regime_color"] = "#666"
            d["regime_label"] = "—"
            d["regime_conf"]  = 0
            d["regime_date"]  = _rl_max_date or d.get("spx_date", "N/A")  # Bug 3
            d["prob_r0"] = d["prob_r1"] = d["prob_r2"] = 33.3
        d.setdefault("clf_accuracy", "63.4%")  # C16 honest no-vvix accuracy
        # Safe clf_* defaults so the research-only card renders if inference failed
        d.setdefault("clf_label", d.get("regime_label", "—"))
        d.setdefault("clf_name",  d.get("regime_name", "N/A"))
        d.setdefault("clf_color", d.get("regime_color", "#666"))
        d.setdefault("clf_conf",  0)
        d.setdefault("clf_agrees", True)

    # Calibration date for timestamp (Bug 7) — latest NON-degenerate joint_cal
    try:
        with open(latest_good_calibration_path(), "rb") as f:
            _cal_tmp = pickle.load(f)
        d["calib_date"] = str(_cal_tmp.get("as_of_date", "2026-05-31"))
    except Exception:
        d["calib_date"] = "2026-05-31"

    # ── PDV Forecast vs ATM ──────────────────────────────────────────────────
    try:
        with open(DATA / "pdv_model.pkl", "rb") as f:
            pdv = pickle.load(f)
        feat    = pdv._features.iloc[-1]
        coefs   = pdv.linear_.coef_
        intcpt  = float(pdv.linear_.intercept_)
        s1  = float(feat["sigma1"])
        s2  = float(feat["sigma2"])
        lev = float(feat["lev"])
        pdv_forecast = float(coefs[0] * s1 + coefs[1] * s2 + coefs[2] * lev + intcpt) * 100
        d["pdv_forecast"] = round(pdv_forecast, 2)
        d["pdv_spread"]   = round((d.get("vix", 0) or 0) - pdv_forecast, 2)
        # The PDV features are computed from the DB, which can lag the regime
        # labels. Stamping the label date here presented a forecast built from
        # months-old features as if it were today's.
        d["pdv_date"]     = str(pdv._features.index[-1].date())
        d["pdv_stale"]    = bool(_rl_max_date and str(pdv._features.index[-1].date()) < _rl_max_date)
        d["pdv_sigma1"]   = round(s1 * 100, 2)
        d["pdv_sigma2"]   = round(s2 * 100, 2)
        d["pdv_spread_positive"] = d["pdv_spread"] > 0

        # PDVLinear3F vs PDVLinear4F comparison (walk-forward R² on stored features)
        try:
            from joint_vol_calibration.models.pdv import PDVLinear, PDVLinear4F
            feats = pdv._features.copy()
            xcols_3f = ["sigma1", "sigma2", "lev"]
            xcols_4f = ["sigma1", "sigma2", "lev", "vix"]
            if all(c in feats.columns for c in xcols_4f) and "rv_hist_20d" in feats.columns:
                y = feats["rv_hist_20d"].shift(-1)
                mask = feats[xcols_4f].notna().all(axis=1) & y.notna()
                X_v, y_v = feats[mask][xcols_4f], y[mask]
                t_n = int(len(X_v) * 0.60)
                X_tr, X_te = X_v.iloc[:t_n], X_v.iloc[t_n:]
                y_tr, y_te = y_v.iloc[:t_n], y_v.iloc[t_n:]
                def _r2(preds, actuals):
                    ss_res = float(np.sum((actuals - preds)**2))
                    ss_tot = float(np.sum((actuals - actuals.mean())**2))
                    return round(1.0 - ss_res / ss_tot, 4) if ss_tot > 0 else 0.0
                r2_3f = _r2(PDVLinear().fit(X_tr[xcols_3f], y_tr).predict(X_te[xcols_3f]).values, y_te.values)
                r2_4f = _r2(PDVLinear4F().fit(X_tr, y_tr).predict(X_te).values, y_te.values)
                d["pdv_r2_3f"] = r2_3f
                d["pdv_r2_4f"] = r2_4f
                d["pdv_model_name"] = "PDVLinear4F" if r2_4f > r2_3f else "PDVLinear3F (4F adds no signal)"
                d["pdv_r2_note"] = (
                    f"PDVLinear3F R²={r2_3f} | PDVLinear4F R²={r2_4f} | "
                    + ("VIX adds incremental signal." if r2_4f > r2_3f
                       else "VIX adds no incremental signal vs realized vol target.")
                )
            else:
                # vix column absent from stored features — 4F can't be tested here
                d["pdv_r2_3f"]      = None
                d["pdv_r2_4f"]      = None
                d["pdv_model_name"] = "PDVLinear3F"
                d["pdv_r2_note"]    = "PDVLinear4F requires VIX column in stored features."
        except Exception:
            d.setdefault("pdv_model_name", "PDVLinear3F")
            d.setdefault("pdv_r2_note", "")

    except Exception as e:
        d["pdv_error"] = str(e)
        d["pdv_forecast"] = 0
        d["pdv_spread"]   = 0
        d["pdv_date"]     = "N/A"
        d["pdv_sigma1"]   = 0
        d["pdv_sigma2"]   = 0
        d["pdv_spread_positive"] = False
        d.setdefault("pdv_model_name", "PDVLinear3F")
        d.setdefault("pdv_r2_note", "")

    # ── PDV spread historical context ────────────────────────────────────
    try:
        rl = _rl if _rl is not None else pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
        # pdv_iv_spread in parquet = PDV − VIX (decimal) → negate to get implied−realised (pp)
        hist_spread_pp = -rl["pdv_iv_spread"] * 100
        cur_spread = d.get("pdv_spread", 0) or 0
        d["pdv_spread_mean"] = round(float(hist_spread_pp.mean()), 2)
        d["pdv_spread_pct"]  = round(
            float((hist_spread_pp < cur_spread).mean() * 100), 1
        )
        # elevated / cheap label
        pct = d["pdv_spread_pct"]
        if pct >= 80:
            d["pdv_spread_label"] = f"Top {100-pct:.0f}% — premium selling regime"
            d["pdv_spread_label_color"] = "#ff8833"
        elif pct <= 20:
            d["pdv_spread_label"] = f"Bottom {pct:.0f}% — vol cheap, long vol regime"
            d["pdv_spread_label_color"] = "#3388ff"
        else:
            d["pdv_spread_label"] = f"{pct:.0f}th percentile — neutral"
            d["pdv_spread_label_color"] = "#888"
        # Sparkline: last 60 trading days of spread
        recent = hist_spread_pp.iloc[-60:]
        d["pdv_sparkline"]       = [round(float(x), 2) for x in recent.tolist()]
        d["pdv_sparkline_dates"] = [str(dt.date()) for dt in recent.index]
    except Exception:
        d.setdefault("pdv_spread_mean", 0)
        d.setdefault("pdv_spread_pct", 50)
        d.setdefault("pdv_spread_label", "")
        d.setdefault("pdv_spread_label_color", "#888")
        d.setdefault("pdv_sparkline", [])
        d.setdefault("pdv_sparkline_dates", [])

    # Bug 5: Log all four card dates on every page load for remote verification
    _logger.info(
        "[page1] SPX=%.2f date=%s | VIX=%.2f vix_date=%s | VVIX=%.2f vvix_date=%s | TS_max=%s | regime_date=%s",
        d.get("spx", 0), d.get("spx_date", "?"),
        d.get("vix", 0), d.get("vix_date", "?"),
        d.get("vvix", 0), d.get("vvix_date", "?"),
        d.get("ts_max_date", "?"),
        d.get("regime_date", "?"),
    )

    return _clean(d)


# ── PAGE 2 DATA: Calibration Results ─────────────────────────────────────────
def _optimizer_settings(cal: dict) -> Optional[str]:
    """Human-readable optimiser settings when the result records them."""
    opt = cal.get("optimizer") or cal.get("de_config") or {}
    maxiter = opt.get("maxiter", cal.get("de_maxiter"))
    popsize = opt.get("popsize", cal.get("de_popsize"))
    if maxiter is None or popsize is None:
        return None
    return f"DE (maxiter {int(maxiter)}, popsize {int(popsize)} × 5) + L-BFGS-B"


def load_page2() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "mat_smiles": [], "anchors": [],
                         "quintic": None, "rmse_bars": []}
    for k in ("kappa", "theta", "theta_vol", "sigma", "rho", "v0", "v0_vol", "S",
              "r_pct", "q_pct", "session", "fit_time", "n_evals",
              "spx_rmse", "vix_fut_rmse", "vix_opt_rmse", "feller_lhs", "feller_rhs",
              "feller_margin", "feller_state", "effective_dof", "n_spx_options",
              "n_vix_tenors", "spot_date", "n_spx_expiries", "half_life_days",
              "optimizer", "legacy", "calib_file", "calib_age"):
        d[k] = None
    d.update(pinned_params=[], rho_at_boundary=False, spx_underidentified=None,
             n_spx_expiries_inferred=False, carry_recorded=False)

    try:
        path, cal = load_headline_calibration()
        p = cal["params"]
        d["calib_file"] = Path(path).name
        d["session"] = _calibration_session(cal, path)
        d["kappa"] = round(p["kappa"], 4)
        d["theta"] = round(p["theta"], 5)
        d["theta_vol"] = round(np.sqrt(p["theta"]) * 100, 2)
        d["sigma"] = round(p["sigma"], 4)
        d["rho"] = round(p["rho"], 4)
        d["v0"] = round(p["v0"], 5)
        d["v0_vol"] = round(np.sqrt(p["v0"]) * 100, 2)
        d["half_life_days"] = (np.log(2) / p["kappa"] * 252) if p["kappa"] > 0 else None
        d["S"] = round(float(cal["S"]), 1) if _num(cal.get("S")) is not None else None
        d["fit_time"] = round(float(cal["fit_time"]), 1) if _num(cal.get("fit_time")) is not None else None
        d["n_evals"] = int(cal["n_evals"]) if _num(cal.get("n_evals")) is not None else None
        d["optimizer"] = _optimizer_settings(cal)
        d["legacy"] = calibration_is_legacy(cal)

        # Carry inputs: shown only when the RESULT records both r and q. Daily
        # fits before C22 ran with the calibrator's default carry while the
        # pickle stored a T-bill close written after the fit (and no q), so a
        # lone stored r is not the rate the fit used.
        if _num(cal.get("r")) is not None and _num(cal.get("q")) is not None:
            d["carry_recorded"] = True
            d["r_pct"] = round(float(cal["r"]) * 100, 2)
            d["q_pct"] = round(float(cal["q"]) * 100, 2)

        losses = _cal_losses(cal)
        d["spx_rmse"] = round(float(losses["spx_iv_rmse"]), 3) if _num(losses.get("spx_iv_rmse")) is not None else None
        d["vix_fut_rmse"] = round(float(losses["vix_futures_rmse"]), 3) if _num(losses.get("vix_futures_rmse")) is not None else None
        w3 = (cal.get("weights") or {}).get("w3")
        vix_opt_active = bool(w3) or bool(cal.get("n_vix_options"))
        if vix_opt_active and _num(losses.get("vix_options_rmse")) is not None:
            d["vix_opt_rmse"] = round(float(losses["vix_options_rmse"]), 2)
        d["vix_opt_disabled"] = not vix_opt_active
        d["rmse_bars"] = [b for b in (
            {"label": "SPX IV RMSE (vol pts)", "value": d["spx_rmse"], "color": "#3388ff"},
            {"label": "VIX leg RMSE (index pts)", "value": d["vix_fut_rmse"], "color": "#00ff88"},
            {"label": "VIX options RMSE (vol pts)", "value": d["vix_opt_rmse"], "color": "#ff3333"},
        ) if b["value"] is not None]

        # Feller: PASS / BINDING / VIOLATED (soft exterior penalty -> an active
        # constraint lands a whisker outside the feasible set).
        from joint_vol_calibration.calibration.joint_calibrator import (
            _CAL_STRIKES_PER_EXPIRY, classify_feller, count_pinned_params,
        )
        state, margin = classify_feller(p["kappa"], p["theta"], p["sigma"])
        d["feller_lhs"] = round(2 * p["kappa"] * p["theta"], 5)
        d["feller_rhs"] = round(p["sigma"] ** 2, 5)
        d["feller_margin"] = margin
        d["feller_state"] = state
        pinned = cal.get("pinned_params") or count_pinned_params(p, state)
        d["pinned_params"] = list(pinned)
        d["effective_dof"] = 5 - len(pinned)
        d["rho_at_boundary"] = abs(p["rho"]) >= 0.94

        # Data provenance. The SPX leg keeps _CAL_STRIKES_PER_EXPIRY strikes per
        # expiry bucket, so an unrecorded expiry count is bounded by
        # n_options / strikes-per-bucket (a bucket can repeat an expiry, so this
        # is an upper bound, labelled as inferred).
        d["n_spx_options"] = cal.get("n_spx_options")
        d["n_vix_tenors"] = cal.get("n_vix_tenors")
        d["spot_date"] = _date_str(cal.get("spot_date"))
        n_exp = cal.get("n_spx_expiries")
        if n_exp is None and _num(d["n_spx_options"]) is not None:
            n_exp = int(d["n_spx_options"]) // _CAL_STRIKES_PER_EXPIRY
            d["n_spx_expiries_inferred"] = True
        d["n_spx_expiries"] = n_exp
        if n_exp is not None:
            d["spx_underidentified"] = n_exp < 3

        # Market anchors (when the result stores them): market vs model IV.
        for a in cal.get("anchors") or []:
            try:
                T_days = a.get("T_days", (a.get("T") or 0) * 365.0)
                d["anchors"].append({
                    "T_days": int(round(float(T_days))),
                    "lm": float(a["k"] if "k" in a else a["log_moneyness"]),
                    "market_iv": float(a["market_iv"]) * 100,
                    "model_iv": float(a["model_iv"]) * 100 if a.get("model_iv") is not None else None,
                })
            except Exception as exc:
                _logger.info("skipping malformed anchor %r: %s", a, exc)

        # Model smile implied by THESE parameters (not the Greeks parquet,
        # which was built from a different, gate-rejected 2026-03-24 fit).
        colors = {14: "#ff3333", 30: "#ff8844", 60: "#ffcc00",
                  91: "#00ff88", 180: "#44aaff", 365: "#aa44ff"}
        for s in heston_model_smile(p):
            d["mat_smiles"].append({
                "T": s["T"], "lm": s["lm"],
                "iv": [round(v * 100, 3) for v in s["iv"]],
                "color": colors.get(s["T"], "#888"),
            })

        ref = _latest_market_session()
        item = _staleness("calibration", "Heston calibration", d["session"], ref)
        d["calib_age"] = item["age"]
        d["stale"] = [item]
    except Exception as e:
        _record_error(d, "headline calibration", e)

    # ── Quintic OU research fit vs the Heston fit of the SAME snapshot ───────
    try:
        qpaths = sorted(glob.glob(str(DATA / "calibrations" / "quintic_cal_*.pkl")))
        if qpaths:
            qpath = qpaths[-1]
            with open(qpath, "rb") as f:
                qcal = pickle.load(f)
            qp = qcal["params"]
            qp = qp if isinstance(qp, dict) else vars(qp)
            ql = qcal.get("leg_losses", {})
            q_date = _calibration_session(qcal, qpath)
            q = {
                "as_of": q_date,
                "lam_x": round(float(qp["lam_x"]), 3), "lam_y": round(float(qp["lam_y"]), 3),
                "theta": round(float(qp["theta"]), 4), "alpha0": round(float(qp["alpha0"]), 4),
                "alpha1": round(float(qp["alpha1"]), 4), "alpha3": round(float(qp["alpha3"]), 4),
                "alpha5": round(float(qp["alpha5"]), 4), "epsilon": round(float(qp["epsilon"]), 4),
                "spx_rmse": _num(ql.get("spx_iv_rmse")), "vix_rmse": _num(ql.get("vix_futures_rmse")),
                "vix_opt_rmse": _num(ql.get("vix_options_rmse")),
                "eps_saturated": abs(float(qp["epsilon"]) + 0.99) < 0.02,
                "fit_time": _num(qcal.get("fit_time")), "n_evals": qcal.get("n_evals"),
                "converged": qcal.get("success"),
                "heston": None,
            }
            # A cross-date Δ (Quintic on 2026-03-24 vs today's Heston headline)
            # compares different chains and flipped the sign of the SPX Δ.
            hpath = DATA / "calibrations" / f"joint_cal_{q_date}.pkl"
            if q_date and hpath.exists():
                with open(hpath, "rb") as f:
                    hcal = pickle.load(f)
                hl = _cal_losses(hcal)
                ok, reason = calibration_gate(hcal)
                h_spx, h_vix = _num(hl.get("spx_iv_rmse")), _num(hl.get("vix_futures_rmse"))
                q["heston"] = {
                    "file": hpath.name, "spx_rmse": h_spx, "vix_rmse": h_vix,
                    "gate_ok": ok, "gate_reason": reason,
                    "spx_delta": (q["spx_rmse"] - h_spx) if None not in (q["spx_rmse"], h_spx) else None,
                    "vix_delta": (q["vix_rmse"] - h_vix) if None not in (q["vix_rmse"], h_vix) else None,
                }
            d["quintic"] = q
    except Exception as e:
        _record_error(d, "Quintic OU research calibration", e)

    return _clean(d)


# ── PAGE 3 DATA: Greeks Risk Monitor ─────────────────────────────────────────
def load_page3() -> dict[str, Any]:
    d: dict[str, Any] = {}

    try:
        gdf = pd.read_parquet(DATA / "greeks" / "greeks_surface.parquet")
        # Recompute is_unstable from the raw z-score so the count is
        # not sensitive to whatever threshold was used when the parquet was saved.
        gdf["is_unstable"] = gdf["vomma_zscore"].abs() > 2.0
        maturities = sorted(gdf["T_days"].unique().tolist())

        # Bin log_moneyness into 0.05-wide buckets for the heatmap grid
        gdf = gdf.copy()
        gdf["lm_bin"] = (np.floor(gdf["log_moneyness"] / 0.05) * 0.05).round(2)
        lm_bins = sorted(gdf["lm_bin"].unique().tolist())

        # FIX 3: Compute unstable nodes directly from raw gdf — before binning —
        # so that two unstable nodes sharing the same lm_bin are both captured.
        unstable_nodes: list[dict] = []
        for _, ur in gdf[gdf["is_unstable"]].iterrows():
            unstable_nodes.append({
                "T_days": int(ur["T_days"]),
                "lm":     round(float(ur["log_moneyness"]), 4),
                "K":      round(float(ur["K"]), 0),
                "vomma":  round(float(ur["vomma"]), 1),
                "z":      round(float(ur["vomma_zscore"]), 2),
                "iv":     round(float(ur["iv"]) * 100, 1),
            })

        # Build heatmap: rows = maturity, cols = lm_bin
        heatmap_rows: list[dict] = []

        for T in maturities:
            sub = gdf[gdf["T_days"] == T]
            cells: list[dict | None] = []
            for lm in lm_bins:
                cell_df = sub[sub["lm_bin"] == lm]
                if len(cell_df) == 0:
                    cells.append(None)
                else:
                    r = cell_df.iloc[0]
                    z    = float(r["vomma_zscore"])
                    v    = float(r["vomma"])
                    iv   = float(r["iv"]) * 100
                    K    = float(r["K"])
                    unstable = bool(r["is_unstable"])
                    cell = {
                        "vomma":    round(v, 1),
                        "z":        round(z, 2),
                        "iv":       round(iv, 1),
                        "K":        round(K, 0),
                        "unstable": unstable,
                        "bg":       _vomma_color(z),
                        "fg":       "#fff" if unstable or abs(z) > 1 else "#aaa",
                    }
                    cells.append(cell)
            heatmap_rows.append({
                "T":     int(T),
                "label": f"{int(T)}d",
                "cells": cells,
            })

        d["heatmap_rows"] = heatmap_rows
        d["heatmap_cols"] = [f"{b:+.2f}" for b in lm_bins]
        d["unstable_nodes"] = sorted(unstable_nodes, key=lambda x: x["z"], reverse=True)
        d["n_unstable"]     = len(unstable_nodes)

        # Vomma surface statistics
        d["vomma_max"]  = round(float(gdf["vomma"].max()), 1)
        d["vomma_min"]  = round(float(gdf["vomma"].min()), 1)
        d["vomma_mean"] = round(float(gdf["vomma"].mean()), 1)
        d["n_cells"]    = len(gdf)

        # QV convexity per maturity
        qv = gdf.groupby("T_days")["qv_convexity"].agg(["mean", "max"]).reset_index()
        d["qv_labels"]    = [f"{int(t)}d" for t in qv["T_days"]]
        d["qv_mean"]      = (qv["mean"] * 1e4).round(4).tolist()  # ×1e4 for display
        d["qv_max"]       = (qv["max"]  * 1e4).round(4).tolist()

        # Vanna per maturity
        vanna = gdf.groupby("T_days")["vanna"].mean().reset_index()
        d["vanna_labels"] = [f"{int(t)}d" for t in vanna["T_days"]]
        d["vanna_values"] = vanna["vanna"].round(4).tolist()

        # Vega per maturity
        vega = gdf.groupby("T_days")["vega"].mean().reset_index()
        d["vega_labels"]  = [f"{int(t)}d" for t in vega["T_days"]]
        d["vega_values"]  = vega["vega"].round(2).tolist()

        # Vomma z-score scatter (for chart)
        d["scatter_lm"]    = gdf["log_moneyness"].round(4).tolist()
        d["scatter_z"]     = gdf["vomma_zscore"].round(3).tolist()
        d["scatter_T"]     = gdf["T_days"].astype(int).tolist()
        d["scatter_iv"]    = (gdf["iv"] * 100).round(2).tolist()

    except Exception as e:
        d["greeks_error"] = str(e)
        d["heatmap_rows"] = []
        d["heatmap_cols"] = []
        d["unstable_nodes"] = []
        d["n_unstable"] = 0
        d["n_cells"] = 0
        d["vomma_max"] = 0
        d["vomma_min"] = 0
        d["vomma_mean"] = 0
        d["qv_labels"] = []
        d["qv_mean"] = []
        d["qv_max"] = []
        d["vanna_labels"] = []
        d["vanna_values"] = []
        d["vega_labels"] = []
        d["vega_values"] = []
        d["scatter_lm"] = []
        d["scatter_z"] = []
        d["scatter_T"] = []
        d["scatter_iv"] = []

    return _clean(d)


# ── PAGE 4 DATA: Backtest Results ─────────────────────────────────────────────
def load_page4() -> dict[str, Any]:
    d: dict[str, Any] = {}

    # ── Equity curve ────────────────────────────────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)

        # Downsample to ≤600 points
        step = max(1, len(eq) // 600)
        eq_ds = eq.iloc[::step]

        # Normalise each NAV series to % gain from 100
        def _norm(col: str) -> list[float]:
            s = eq_ds[col]
            base = float(s.iloc[0])
            return ((s / base * 100) - 100).round(2).tolist()

        d["eq_dates"]    = [dt.strftime("%Y-%m-%d") for dt in eq_ds.index]
        d["eq_nav"]      = _norm("nav")
        d["eq_nav_s1c"]  = _norm("nav_s1c") if "nav_s1c" in eq_ds.columns else []
        d["eq_nav_s3"]   = _norm("nav_s3")
        d["eq_nav_s4"]   = _norm("nav_s4")  if "nav_s4"  in eq_ds.columns else []
        d["eq_nav_s1"]   = _norm("nav_s1")  # reference line

        # Final NAV for display
        d["final_nav"]   = round(float(eq["nav"].iloc[-1]), 0)
        d["start_nav"]   = round(float(eq["nav"].iloc[0]), 0)
        d["total_pnl"]   = round(d["final_nav"] - d["start_nav"], 0)

    except Exception as e:
        d["eq_error"] = str(e)
        d["eq_dates"]    = []
        d["eq_nav"]      = []
        d["eq_nav_s1c"]  = []
        d["eq_nav_s3"]   = []
        d["eq_nav_s4"]   = []
        d["eq_nav_s1"]   = []
        d["final_nav"]   = 0
        d["start_nav"]   = 0
        d["total_pnl"]   = 0

    # ── Metrics — computed entirely from parquet (no pickle import needed) ───
    # This avoids pickle loading joint_vol_calibration.backtest.TradeRecord
    # objects which require the full codebase installed on Railway.
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        n_days = len(eq)
        ann_factor = np.sqrt(252)

        def _pct(x): return f"{x*100:.2f}%" if x is not None and np.isfinite(float(x)) else "—"
        def _f2(x):  return f"{x:.3f}"       if x is not None and np.isfinite(float(x)) else "—"
        def _dol(x): return f"${x:,.0f}"     if x is not None and np.isfinite(float(x)) else "—"

        def _metrics_for(nav_col: str, pnl_col: str, pos_col: str):
            nav  = eq[nav_col]
            rets = nav.pct_change().fillna(0)
            pnl  = eq[pnl_col]
            # Sharpe (excess over 5% rf)
            ex   = rets - 0.05 / 252
            sh   = float(ex.mean() / ex.std() * ann_factor) if rets.std() > 1e-10 else 0.0
            # Sortino: downside deviation about target = sqrt(mean(min(0,ex)^2))
            # over all days (not std of the negative subset about its own mean).
            down_dev = float(np.sqrt((ex.clip(upper=0.0) ** 2).mean()))
            srt  = float(ex.mean() / down_dev * ann_factor) if down_dev > 1e-12 else 0.0
            # Max drawdown
            roll_max = nav.cummax()
            dd_series = (nav / roll_max) - 1
            mdd  = float(dd_series.min())
            # DD duration (longest consecutive drawdown)
            in_dd = (dd_series < 0).astype(int)
            max_dur = 0
            cur_dur = 0
            for v in in_dd:
                if v: cur_dur += 1
                else: cur_dur = 0
                max_dur = max(max_dur, cur_dur)
            # Calmar
            years = n_days / 252
            # Geometric annualisation is only defined while NAV stays positive.
            # Reference signals (e.g. S1 −$1.53M) can drive a standalone NAV path
            # negative; raising a negative ratio to a fractional power yields NaN
            # (the "invalid value in scalar power" RuntimeWarning). Fall back to
            # the simple cumulative return there — the account has blown up.
            if years > 0 and nav.iloc[0] > 0 and nav.iloc[-1] > 0:
                ann_ret = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1)
            else:
                ann_ret = float(nav.iloc[-1] / nav.iloc[0] - 1) if nav.iloc[0] > 0 else 0.0
            calmar  = float(ann_ret / abs(mdd)) if abs(mdd) > 1e-10 else 0.0
            # Win rate (by trading day P&L)
            active_pnl = pnl[pnl != 0]
            win_rate = float((active_pnl > 0).mean()) if len(active_pnl) > 0 else 0.0
            # N trades (position changes)
            pos = eq[pos_col]
            n_trades = int((pos.diff().fillna(0) != 0).sum())
            # Total P&L
            total_pnl = float(nav.iloc[-1] - nav.iloc[0])
            cum_ret = float((nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0])
            avg_pnl = total_pnl / n_trades if n_trades > 0 else 0.0
            return dict(
                cum_ret=cum_ret, ann_ret=ann_ret, sharpe=sh, sortino=srt,
                mdd=mdd, dd_dur=max_dur, calmar=calmar, win_rate=win_rate,
                avg_pnl=avg_pnl, best_day=float(rets.max()), worst_day=float(rets.min()),
                n_trades=n_trades, total_pnl=total_pnl,
            )

        m   = _metrics_for("nav",     "daily_pnl", "s1c_position")  # portfolio
        # The portfolio holds S1C + S3 + S4, so counting only s1c_position
        # transitions understated it (and inflated avg P&L/trade by dividing
        # total portfolio P&L by one signal's trade count).
        _pos_cols = [c for c in ("s1c_position", "s3_position", "s4_position") if c in eq.columns]
        if _pos_cols:
            m["n_trades"] = int(sum(
                (eq[c].diff().fillna(0) != 0).sum() for c in _pos_cols
            ))
            m["avg_pnl"] = (m["total_pnl"] / m["n_trades"]) if m["n_trades"] else 0.0
        m1c = _metrics_for("nav_s1c", "pnl_s1c",  "s1c_position") if "nav_s1c" in eq.columns else None
        m3  = _metrics_for("nav_s3",  "pnl_s3",   "s3_position")
        m4  = _metrics_for("nav_s4",  "pnl_s4",   "s4_position")  if "nav_s4"  in eq.columns else None
        m1  = _metrics_for("nav_s1",  "pnl_s1",   "s1_position")
        m2  = _metrics_for("nav_s2",  "pnl_s2",   "s2_position")

        # C17: prefer the engine's headline Sharpe/Sortino (measured against
        # the per-date ^IRX T-bill path) over the flat-5% parquet recompute.
        _em: dict = {}
        try:
            with open(DATA / "backtest" / "full_results.pkl", "rb") as _fh:
                _em = (pickle.load(_fh).get("metrics") or {})
            for _k in ("sharpe", "sortino"):
                if _em.get(_k) is not None and np.isfinite(float(_em[_k])):
                    m[_k] = float(_em[_k])
        except Exception:
            pass

        # Per-signal cards/table use the engine's TRADE-LOG stats (round-trip
        # trades, win rate by trade, Sharpe vs T-bill) — the canonical numbers
        # from run_backtest — not the dashboard's daily-P&L recompute, which
        # counts each entry+exit as a separate "trade" and wins by day.
        def _sig_card(sig: str, fallback: dict | None) -> dict:
            tp = _em.get(f"{sig}_total_pnl")
            nt = _em.get(f"{sig}_n_trades")
            wr = _em.get(f"{sig}_win_rate")
            sh = _em.get(f"{sig}_sharpe")
            if tp is None and fallback is not None:
                tp, nt, wr, sh = (fallback.get("total_pnl"), fallback.get("n_trades"),
                                  fallback.get("win_rate"), fallback.get("sharpe"))
            return {
                "total_pnl": _dol(tp) if tp is not None else "—",
                "sharpe":    _f2(sh) if sh is not None else "—",
                "win_rate":  _pct(wr) if wr is not None else "—",
                "n_trades":  str(int(nt)) if nt is not None else "—",
                "green":     (tp or 0) > 0,
            }

        d["metrics_rows"] = [
            {"label": "Cumulative Return", "value": _pct(m["cum_ret"]),
             "positive": m["cum_ret"] > 0},
            {"label": "Ann. Return",       "value": _pct(m["ann_ret"]),
             "positive": m["ann_ret"] > 0},
            {"label": "Sharpe (rf=T-bill)", "value": _f2(m["sharpe"]),
             "positive": m["sharpe"] > 0},
            {"label": "Sortino",           "value": _f2(m["sortino"]),
             "positive": m["sortino"] > 0},
            {"label": "Max Drawdown",      "value": _pct(m["mdd"]),     "positive": False},
            {"label": "DD Duration",       "value": f"{m['dd_dur']} days","positive": False},
            {"label": "Calmar",            "value": _f2(m["calmar"]),
             "positive": m["calmar"] > 0},
            {"label": "Win Rate",          "value": _pct(m["win_rate"]),
             "positive": m["win_rate"] > 0.5},
            {"label": "Avg P&L / Trade",   "value": _dol(m["avg_pnl"]),
             "positive": m["avg_pnl"] > 0},
            {"label": "Best Day",          "value": _pct(m["best_day"]),  "positive": True},
            {"label": "Worst Day",         "value": _pct(m["worst_day"]), "positive": False},
            {"label": "N Trades",          "value": str(m["n_trades"]),   "positive": True},
            {"label": "N Days",            "value": str(n_days),          "positive": True},
        ]

        # Signal P&L bar chart — main 3-signal portfolio (S1C + S3 + S4)
        signal_pnl = [
            round(_em.get("s1c_total_pnl", (m1c["total_pnl"] if m1c else 0)) / 1000, 1),
            round(_em.get("s3_total_pnl",  m3["total_pnl"]) / 1000, 1),
            round(_em.get("s4_total_pnl",  (m4["total_pnl"] if m4 else 0)) / 1000, 1),
        ]
        d["signal_labels"] = ["S1C  Contrarian PDV", "S3  Dispersion", "S4  VRP"]
        d["signal_pnl"]    = signal_pnl
        d["signal_colors"] = ["#00c870" if v > 0 else "#ff3333" for v in signal_pnl]

        # Per-signal metrics table — main signals first, reference signals dimmed.
        # Stats sourced from the engine trade-log (canonical), keyed by signal.
        d["signal_metrics"] = []
        for sig, mx, lbl, is_research, is_sparse in [
            ("s1c", m1c, "S1C Contrarian",  False, False),
            ("s3",  m3,  "S3 Dispersion",   False, True),   # 22 trades, proxy-scaled → Sharpe not meaningful
            ("s4",  m4,  "S4 VRP",          False, False),
            ("s1",  m1,  "S1 IVR (ref)",    True,  False),
            ("s2",  m2,  "S2 VIX TS (ref)", True,  False),
        ]:
            if mx is None:
                continue
            _tp = _em.get(f"{sig}_total_pnl", mx["total_pnl"])
            _nt = _em.get(f"{sig}_n_trades",  mx["n_trades"])
            _wr = _em.get(f"{sig}_win_rate",  mx["win_rate"])
            _sh = _em.get(f"{sig}_sharpe",    mx["sharpe"])
            _ar = _em.get(f"{sig}_ann_return", mx["ann_ret"])
            d["signal_metrics"].append({
                "label":       lbl,
                "sharpe":      "N/A*" if is_sparse else _f2(_sh),
                "ann_ret":     _pct(_ar),
                "win_rate":    _pct(_wr),
                "n_trades":    str(int(_nt)),
                "total_pnl":   _dol(_tp),
                "green":       _tp > 0,
                "is_research": is_research,
                "is_sparse":   is_sparse,
            })

        # Research variant summary for dedicated panel (canonical trade-log stats)
        d["research_s1c"] = _sig_card("s1c", m1c)
        d["research_s4"]  = _sig_card("s4",  m4)
        # C18: S1 (original IVR short-VRP) — the signal whose "catastrophic"
        # loss was the stale-strike artifact. Now the headline corrected result.
        d["research_s1"]  = _sig_card("s1",  m1)

    except Exception as e:
        d["pkl_error"] = str(e)
        d["metrics_rows"]   = []
        d["signal_metrics"] = []
        d["signal_pnl"]     = []
        d["signal_labels"]  = []
        d["signal_colors"]  = []
        d["research_s1"]    = {}
        d["research_s1c"]   = {}
        d["research_s4"]    = {}

    # ── Annual returns ───────────────────────────────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        ann_years: list[int] = []
        ann_rets:  list[float] = []
        ann_colors: list[str] = []
        ann_labels: list[str] = []
        for yr, g in eq.groupby(eq.index.year):
            ret = round((float(g["nav"].iloc[-1]) - float(g["nav"].iloc[0]))
                        / float(g["nav"].iloc[0]) * 100, 2)
            ann_years.append(int(yr))
            ann_rets.append(ret)
            ann_colors.append("#00c870" if ret > 0 else "#ff3333")
            ann_labels.append(f"{ret:+.2f}%")
        d["ann_years"]  = ann_years
        d["ann_rets"]   = ann_rets
        d["ann_colors"] = ann_colors
        d["ann_labels"] = ann_labels
    except Exception as e:
        d["ann_error"] = str(e)
        d["ann_years"] = []
        d["ann_rets"]  = []

    # ── Walk-forward (computed from equity curve) ────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        wf_periods: list[str]  = []
        wf_sharpes: list[float] = []
        wf_colors:  list[str]  = []
        for year in range(2019, 2026):
            for half, (s, e) in enumerate(
                [(f"{year}-01-01", f"{year}-06-30"),
                 (f"{year}-07-01", f"{year}-12-31")], 1
            ):
                sub = eq.loc[s:e, "nav"]
                if len(sub) < 5:
                    continue
                rets = sub.pct_change().dropna()
                if rets.abs().max() < 1e-10 or rets.std() < 1e-10:
                    sh = 0.0
                else:
                    ex = rets - 0.05 / 252
                    sh = float(ex.mean() / ex.std() * np.sqrt(252))
                sh = round(max(min(sh, 4.0), -4.0), 2)
                wf_periods.append(f"{year}-H{half}")
                wf_sharpes.append(sh)
                wf_colors.append("#00c870" if sh >= 0 else "#ff3333")
        d["wf_periods"]   = wf_periods
        d["wf_sharpes"]   = wf_sharpes
        d["wf_colors"]    = wf_colors
        d["wf_neg"]       = sum(1 for s in wf_sharpes if s < 0)
        d["wf_total"]     = len(wf_sharpes)
        d["wf_pos"]       = d["wf_total"] - d["wf_neg"]
    except Exception as e:
        d["wf_error"] = str(e)
        d["wf_periods"] = []
        d["wf_sharpes"] = []
        d["wf_colors"]  = []
        d["wf_neg"] = d["wf_total"] = d["wf_pos"] = 0

    # ── Improvement Summary (Steps 1–8) ──────────────────────────────────────
    d["improvements"] = [
        {"step": "1", "name": "Disable VIX options leg",
         "what": "JOINT_W3 = 0.0 (was 0.2). Heston CIR density is structurally mis-specified for VIX options.",
         "before": "VIX opt RMSE 37.14 vp (corrupts calibration)",
         "after":  "SPX-only objective. RMSE expected ↓ from 5.3 → ~2.5 vp."},
        {"step": "2", "name": "Vega-weighted anchor selection",
         "what": "Select 9 highest-BS-vega options per expiry (was log-uniform grid). Moneyness lo 0.70 (was 0.75).",
         "before": "Log-uniform includes low-information deep OTM options",
         "after":  "Near-ATM anchors only; more stable Heston gradient."},
        {"step": "3", "name": "SVI smoothing (SSVI surface)",
         "what": "Fit Gatheral (2004) SVI per expiry; calibrate Heston to smooth surface not noisy quotes.",
         "before": "SPX RMSE ~5.3 vol pts (market microstructure noise)",
         "after":  "Target SPX RMSE ~2.0 vol pts. Butterfly-free surface."},
        {"step": "4", "name": "Per-date T-bill rate",
         "what": "^IRX 3M T-bill from DB replaces fixed r=0.045 in backtest. 18 new tests.",
         "before": "Fixed r=0.045 throughout 2018–2025",
         "after":  "Per-date rate (2018: ~1.5%, 2023: ~5.3%, 2025: ~4.3%)."},
        {"step": "5", "name": "Adaptive VVIX threshold",
         "what": "Rolling 252-day 80th percentile of VVIX as R2 gate (was fixed 100).",
         "before": "R2 frequency ~53% (over-triggered in low-vol periods)",
         "after":  "Expected R2 frequency ~25% (calibrated to market regime)."},
        {"step": "6", "name": "Isotonic calibration + step sizing",
         "what": "CalibratedClassifierCV(FrozenEstimator(XGB), isotonic). S1S step: full/half/flat at P(R2) 0.4/0.6.",
         "before": "Raw XGB probabilities (overconfident). Linear 1−p scaling.",
         "after":  "Calibrated probabilities. Cleaner 3-level position sizing."},
        {"step": "7", "name": "Portfolio Kelly netting",
         "what": "Rolling 60-day pairwise P&L corr; mult = sqrt(2/(1+r_ij)). "
                 "C16: multipliers shifted 1 day, applied to position size in a "
                 "second simulation pass (ex-ante; contracts & costs reflect real size).",
         "before": "No correlation netting; each signal sizes off full NAV ÷ 4",
         "after":  "Ex-ante diversification sizing; no same-day information in sizing."},
        {"step": "8", "name": "Monthly Heston recalibration — REMOVED (C17)",
         "what": "The C13 monthly-recalibration loop was dead code: no historical "
                 "options data exists before the 2026 snapshots, so every monthly "
                 "calibration fell back to static defaults — and the resulting "
                 "parameters were never consumed by any P&L path. Removed in C17.",
         "before": "ASSUMPTIONS claimed time-varying Heston params in the backtest",
         "after":  "Honest: backtest P&L is BS on VIX-TS ATM vol; the calibration "
                   "stack serves C6 Greeks / C7 hedge sim, not C10 P&L."},
    ]

    # ── Improved Signal Variants ──────────────────────────────────────────────
    # Compute from the existing backtest parquet.
    # S1RF/S2X: zero P&L on R2 days where the signal had an open position.
    # S1X ≈ S1RF (state machine differs but P&L impact is similar).
    # S1S ≈ midpoint of S1 and S1RF (continuous scaling reduces exposure by ~50%).
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)

        # S1RF: zero S1 P&L on R2 days where S1 position is open
        r2_s1 = (eq["regime"] == 2) & (eq["s1_position"] != 0)
        pnl_s1rf = eq["pnl_s1"].copy()
        pnl_s1rf[r2_s1] = 0.0

        # S2X: same approach for S2
        r2_s2 = (eq["regime"] == 2) & (eq["s2_position"] != 0)
        pnl_s2x = eq["pnl_s2"].copy()
        pnl_s2x[r2_s2] = 0.0

        s1_total   = float(eq["pnl_s1"].sum())
        s1rf_total = float(pnl_s1rf.sum())
        s1x_total  = s1rf_total                        # S1X ≈ S1RF (approx)
        s1s_total  = (s1_total + s1rf_total) / 2.0     # S1S: ~half reduction
        s2x_total  = float(pnl_s2x.sum())

        s2_total = float(eq["pnl_s2"].sum())

        d["variants"] = {
            "s1_pnl":   round(s1_total,   0),
            "s1rf_pnl": round(s1rf_total, 0),
            "s1x_pnl":  round(s1x_total,  0),
            "s1s_pnl":  round(s1s_total,  0),
            "s1_impr":  round(s1rf_total - s1_total, 0),
            "s2_pnl":   round(s2_total,   0),
            "s2x_pnl":  round(s2x_total,  0),
            "s2_impr":  round(s2x_total - s2_total, 0),
            "s3_pnl":   round(float(eq["pnl_s3"].sum()), 0),
        }
    except Exception:
        d["variants"] = None

    return _clean(d)


# ── Rendering ──────────────────────────────────────────────────────────────────
# Per-route data date for the sitemap (freeze_site.py reads the header).
_DATA_DATE_KEYS = {"/": "latest_session", "/calibration": "session",
                   "/greeks": "surface_date", "/backtest": "run_date"}


@app.context_processor
def _page_context() -> dict:
    route = request.path if request.path in ROUTES else "/"
    return {"canonical": canonical_url(route), "page_file": ROUTES[route]}


def _render(template: str, d: dict):
    """Render a page and expose its loader errors / data date as headers.

    freeze_site.py refuses to publish when X-Dashboard-Loader-Errors != 0, so a
    page whose data failed to load is never deployed (the previous good build
    stays live). The same errors are rendered on the page as a red banner.
    """
    errors = d.get("errors") or []
    worst = "ok"
    for item in d.get("stale") or []:
        if item.get("level") == "error" or (item.get("level") == "warn" and worst == "ok"):
            worst = item["level"]
    d["worst_stale"] = worst
    resp = make_response(render_template(template, d=d))
    resp.headers["X-Dashboard-Loader-Errors"] = str(len(errors))
    if errors:
        resp.headers["X-Dashboard-Error-Components"] = "; ".join(e["component"] for e in errors)
    stale = [f"{s['component']}={s['age']}" for s in d.get("stale") or []
             if s.get("level") in ("warn", "error")]
    if stale:
        resp.headers["X-Dashboard-Stale"] = ", ".join(stale)
    data_date = d.get(_DATA_DATE_KEYS.get(request.path, ""))
    if data_date:
        resp.headers["X-Dashboard-Data-Date"] = str(data_date)
    return resp


@app.route("/")
def page_market():
    return _render("index.html", load_page1())


@app.route("/calibration")
def page_calibration():
    return _render("calibration.html", load_page2())


@app.route("/greeks")
def page_greeks():
    return _render("greeks.html", load_page3())


@app.route("/backtest")
def page_backtest():
    return _render("backtest.html", load_page4())


# ── Local preview ──────────────────────────────────────────────────────────────
def _find_free_port(preferred: int = 5000) -> int:
    import socket
    for port in [preferred, 5001, 5050, 8080, 8888]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("127.0.0.1", port)) != 0:
                return port
    return preferred


if __name__ == "__main__":
    # A read-only local preview of exactly what freeze_site.py publishes. It
    # does not refresh data: run scripts/cloud_refresh.py for that.
    import os

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    host = os.environ.get("DASHBOARD_HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", _find_free_port(5000)))
    print(f"\n  Joint SPX/VIX Volatility System — local preview: http://{host}:{port}\n")
    app.run(host=host, port=port, debug=False, use_reloader=False)
