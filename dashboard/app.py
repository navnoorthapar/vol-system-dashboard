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

Every loader reports failures in ``d["errors"]``; the templates render them as a
visible error banner instead of zeros, and freeze_site.py refuses to publish a
page that carries one.

Run a local preview (read-only, bound to 127.0.0.1):
    python dashboard/app.py
"""
from __future__ import annotations

import glob
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
_REGIME_COLORS = ["#7fadf7", "#5cc8a0", "#f0787c"]   # --blue / --accent / --red
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
    data_store/vol_system.db (2010-01-04 .. 2026-10-16, with the special
    closures below: zero mismatches).
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


MINUS = "\u2212"   # '−': numbers are printed with a true minus sign, not a hyphen


def _fmt(value: Any, spec: str = "", default: str = "—") -> str:
    """Jinja filter: format a number, or render `default` when it is missing."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return default
        return format(value, spec).replace("-", MINUS)   # typographic minus sign
    return str(value)


def _usd_k(value: Any, signed: bool = True) -> str:
    """Prose dollars in thousands: 693938 -> '+$694K', -599995 -> '−$600K'."""
    x = _num(value)
    if x is None:
        return "—"
    sign = "−" if x < 0 else ("+" if signed else "")
    if abs(x) >= 999_500:
        return f"{sign}${abs(x) / 1e6:,.1f}M"
    return f"{sign}${abs(x) / 1000:,.0f}K"


def _usd(value: Any, signed: bool = False) -> str:
    """Whole dollars with a true minus sign: -1039.2 -> '−$1,039'."""
    x = _num(value)
    if x is None:
        return "—"
    sign = "−" if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}${abs(x):,.0f}"


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
app.jinja_env.filters["usd"] = _usd
app.jinja_env.filters["pct"] = _pct_prose


def _relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of '#rrggbb'."""
    h = hex_colour.lstrip("#")
    def lin(c: int) -> float:
        c = c / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio between two opaque '#rrggbb' colours (1..21)."""
    hi, lo = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _text_colour_on(bg: str) -> str:
    """Black or white, whichever contrasts more with `bg` (always >= 4.58:1)."""
    return "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"


# Diverging heatmap ramp: blue for z < 0, orange for z > 0 (distinguishable
# under red-green colour blindness). Red is NOT used as a fill: an unstable
# node (|z| > 2) gets the end colour of its side plus a red outline and a text
# marker in the cell, so the flag does not depend on colour alone.
_HEAT_NEUTRAL = (0x1c, 0x20, 0x26)
_HEAT_NEG = (0x33, 0x62, 0xb5)
_HEAT_POS = (0xbd, 0x6e, 0x2a)


def _vomma_color(z: Optional[float]) -> str:
    """Map a vomma z-score to a CSS background colour for the heatmap."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "#1c2026"
    end = _HEAT_POS if z > 0 else _HEAT_NEG
    t = min(abs(z) / 2.0, 1.0)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(_HEAT_NEUTRAL, end))


# ── Market data readers (DB) ───────────────────────────────────────────────────
def _read_spx() -> pd.DataFrame:
    """SPX closes and log returns, date-indexed, ascending.

    ``log_return`` is the stored column (the classifier was trained on it via
    build_features); ``close_return`` is ln(close_t / close_{t-1}) rebuilt from
    the closes, which is what the PDV model is fitted on: the stored column has
    isolated holes (e.g. 2025-05-08) that would silently drop a session from
    the PDV state variables.
    """
    with _db() as con:
        df = pd.read_sql("SELECT date, close, log_return FROM spx_ohlcv ORDER BY date", con)
    df["date"] = pd.to_datetime(df["date"])
    df = df.drop_duplicates("date", keep="last").set_index("date")
    df["close_return"] = np.log(df["close"].astype(float)).diff()
    return df


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
    """joint_calibrator.is_acceptable_calibration applied to a stored result.

    The gate is called with the FULL result dict, its single-source-of-truth
    form: it applies every check from the result's own fields (degenerate
    corner, SPX RMSE, VIX tenors, SPX expiries / options / put wing, snapshot
    session == spot session, model_version) and rejects legacy (pre-C22)
    results. Binding a few keys into the old positional form skipped the
    checks whose arguments were not passed, including the legacy check.
    """
    from joint_vol_calibration.calibration import joint_calibrator as jc

    ok, reason = jc.is_acceptable_calibration(cal)
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


def scan_calibrations(cal_dir: str | Path | None = None) -> list[dict]:
    """Every joint_cal_*.pkl with its session, legacy flag and gate verdict.

    Legacy (pre-C22, model_version < 2) results are classified, never ranked:
    they fitted the VIX spot indices as futures, with ATM-call anchors and an
    assumed r/q, so their RMSEs are not comparable with model_version 2 fits.
    """
    cal_dir = Path(cal_dir) if cal_dir is not None else DATA / "calibrations"
    rows = []
    for path in sorted(glob.glob(str(cal_dir / "joint_cal_*.pkl"))):
        try:
            with open(path, "rb") as f:
                cal = pickle.load(f)
            legacy = calibration_is_legacy(cal)
            ok, reason = calibration_gate(cal)
        except Exception as exc:
            _logger.warning("calibration %s unreadable: %s", Path(path).name, exc)
            continue
        rows.append({"path": path, "session": _calibration_session(cal, path),
                     "run_date": str(cal.get("run_date") or ""),
                     "legacy": legacy, "ok": ok, "reason": reason})
    return rows


def latest_good_calibration_path(cal_dir: str | Path | None = None) -> Optional[str]:
    """Return the NEWEST model_version 2 joint_cal_*.pkl that passes the gate, or None.

    Selection is by market session (spot_date, else as_of_date, else the date
    in the file name), never by RMSE. Ranking archived fits by RMSE is a
    selection bias that rewards thin data: a one-tenor VIX leg (2026-07-23,
    0.049 vp) and later a one-expiry SPX leg (2026-09-11, 0.215 vp; 2026-09-25,
    0.041 vp) each scored the best RMSE on record precisely because the data
    had broken, and an RMSE-ranked selector headlined them over newer, better
    identified fits. Only current-schema fits are ranked (calibration_is_legacy
    excludes the pre-C22 archive); what may be published is decided by the
    gate (``calibration_gate``, the full-result form of
    is_acceptable_calibration), and the dashboard shows the latest such fit
    together with its age, never falling back to a rejected or legacy fit.
    """
    candidates = []
    for row in scan_calibrations(cal_dir):
        name = Path(row["path"]).name
        if row["legacy"]:
            _logger.info("calibration %s is legacy (pre-C22): not ranked", name)
        elif not row["ok"]:
            _logger.info("calibration %s rejected by gate: %s", name, row["reason"])
        elif row["session"]:
            candidates.append((row["session"], row["run_date"], row["path"]))
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
    calls above) come from heston_call_batch (Lewis contour integral with a
    Black-Scholes control variate, adaptive step; no FFT) and are inverted with
    Black-Scholes; nodes whose OTM price is below SMILE_PRICE_FLOOR are omitted
    rather than shown.
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
        )
        otm = np.where(k < 0, calls - (1.0 - K), calls)
        lm, iv = [], []
        for ki, Ki, price in zip(k, K, otm):
            if not np.isfinite(price) or price < SMILE_PRICE_FLOOR:
                continue
            vol = implied_vol_from_price(float(price), 1.0, float(Ki), T, 0.0, 0.0,
                                         "P" if ki < 0 else "C", tol=1e-10)
            # implied_vol_from_price returns NaN (not None) for ill-posed inputs
            if vol is not None and np.isfinite(vol):
                lm.append(round(float(ki), 4))
                iv.append(float(vol))
        out.append({"T": int(T_days), "lm": lm, "iv": iv})
    return out


# ── PDV forecast series (index page) ───────────────────────────────────────────
def pdv_features(pdv: Any, log_returns: pd.Series) -> pd.DataFrame:
    """PDV state variables rebuilt from `log_returns` with the model's own half-lives."""
    from joint_vol_calibration.models.pdv import extract_pdv_features

    return extract_pdv_features(
        log_returns.dropna(),
        halflife_fast=getattr(pdv, "halflife_fast", 5.0),
        halflife_slow=getattr(pdv, "halflife_slow", 60.0),
        halflife_lev=getattr(pdv, "halflife_lev", 10.0),
        n_lags=getattr(pdv, "n_lags", 5),
    )


def spread_percentile(history: pd.Series, current: float) -> tuple[Optional[float], int]:
    """(percent of the history strictly below `current`, number of valid points).

    NaN history rows are dropped before ranking. Counting them in the
    denominator (NaN < x is False) pinned the published percentile near zero:
    4,125 of 4,189 rows were NaN and the card read '1.1th percentile, vol
    cheap' for a spread above its own mean.
    """
    hist = pd.to_numeric(history, errors="coerce").dropna()
    if len(hist) == 0 or _num(current) is None:
        return None, int(len(hist))
    return float((hist < current).mean() * 100.0), int(len(hist))


PDV_MIN_HISTORY = 252   # fewer valid history points -> no percentile is shown


def percentile_rank(pct: Any) -> Optional[dict]:
    """Wording for a percentile = percent of prior sessions strictly below today.

    100 (above every prior session) and 0 (none below) printed as numbers read
    "100th percentile" / "top 0%" and "0th" / "bottom 0%"; they are records.
    Anything else is shown as an integer percentile capped at 1st..99th so
    rounding cannot produce those strings, and the tail share is the
    complement of that integer ("85th" with "top 15%").
    Returns {"record": "high" | "low" | None, "big": the headline figure,
    "tail": "top 15%" / "bottom 3%" / "highest on record" / ... or None}.
    """
    x = _num(pct)
    if x is None:
        return None
    if x >= 100.0:
        return {"record": "high", "big": "Record high", "tail": "highest on record"}
    if x <= 0.0:
        return {"record": "low", "big": "Record low", "tail": "lowest on record"}
    k = min(99, max(1, math.floor(x + 0.5)))
    tail = f"top {100 - k}%" if x >= 80 else (f"bottom {k}%" if x <= 20 else None)
    return {"record": None, "big": ordinal(k), "tail": tail}


def pdv_stress_forecast(pdv: Any, log_returns: pd.Series,
                        stress_date: str = "2020-03-16") -> Optional[dict]:
    """Ex-ante PDV-linear forecast of |r| on `stress_date`, made at the prior close.

    Out of sample: the OLS is refitted on the rows whose labels were already
    known at the forecast origin (labels of rows before the origin), and it
    predicts from the origin's features, which use returns through the origin
    only. The previous card quoted the in-sample KERNEL value for the next day
    (it memorised |r| of 2020-03-17, 92.4%) against the same day's move.
    """
    from joint_vol_calibration.models.pdv import PDVLinear, build_xy

    r = log_returns.dropna()
    target = pd.Timestamp(stress_date)
    if target not in r.index:
        return None
    pos = r.index.get_loc(target)
    if pos < 1:
        return None
    origin = r.index[pos - 1]
    horizon = int(getattr(pdv, "horizon", 1))
    feats = pdv_features(pdv, r)
    X, y = build_xy(r, feats, horizon=horizon)
    # y(t) is realised over sessions t+1..t+h, so it is known at the origin's
    # close only if the session h steps after t is on or before the origin.
    label_end = pd.Series(r.index, index=r.index).shift(-horizon).reindex(X.index)
    train = (label_end <= origin).to_numpy()
    model = PDVLinear().fit(X[train], y[train])
    forecast = float(model.predict(feats.loc[[origin]]).iloc[0])
    window = r.iloc[pos:pos + horizon]          # the sessions the forecast covers
    actual = float(np.sqrt((window ** 2).mean() * 252))
    return {
        "stress_date": str(target.date()), "origin": str(origin.date()),
        "forecast_pct": forecast * 100.0, "actual_pct": actual * 100.0,
        "error_pp": (forecast - actual) * 100.0, "n_train": int(np.sum(train)),
        "horizon": horizon, "window_end": str(window.index[-1].date()),
        "n_window": int(len(window)),
    }


def pdv_walk_forward_series(pdv: Any, spx: pd.DataFrame) -> pd.Series:
    """The causal PDV forecast at the shipped model's horizon, one value per session.

    regime_classifier.walk_forward_pdv_vol (the series the signals and the
    backtest trade on): row t is the forecast made at the close of t by a
    PDVLinear refitted daily on the labels fully observed by t (purged,
    embargo = horizon), from returns rebuilt from closes. Its last value is
    the live forecast (= models.pdv.latest_vol_forecast). Ranking today's
    spread against a history built with the pickle's FULL-SAMPLE
    coefficients compared an out-of-sample value with in-sample fits.
    """
    from joint_vol_calibration.signals.regime_classifier import walk_forward_pdv_vol

    horizon = int(getattr(pdv, "horizon", 1))
    kwargs = {"target": str(getattr(pdv, "target", "rv")),
              "train_min_days": int(getattr(pdv, "train_min_days", 504)),
              "halflife_fast": getattr(pdv, "halflife_fast", 5.0),
              "halflife_slow": getattr(pdv, "halflife_slow", 60.0),
              "halflife_lev": getattr(pdv, "halflife_lev", 10.0)}
    close = pd.to_numeric(spx["close"], errors="coerce")
    # ~1 s of daily refits: memoised on the exact inputs (a preview server and
    # the test-suite render the same data many times).
    key = (horizon, tuple(sorted(kwargs.items())), len(close), str(spx["date"].iloc[0]),
           str(spx["date"].iloc[-1]), float(close.sum()), float((close * close).sum()))
    if key not in _WF_CACHE:
        _WF_CACHE[key] = walk_forward_pdv_vol(spx, horizon=horizon, **kwargs).rename("pdv_vol")
    return _WF_CACHE[key].copy()


def pdv_walk_forward_skill(pdv: Any, log_returns: pd.Series, pred: pd.Series,
                           exclude_year: int = 2020) -> Optional[dict]:
    """Walk-forward R² of `pred` (pdv_walk_forward_series) on the model's own target.

    R² is taken against the realised target over the rows whose label is
    complete, pooled and without `exclude_year`.
    """
    from joint_vol_calibration.models.pdv import forward_target

    horizon = int(getattr(pdv, "horizon", 1))
    target = str(getattr(pdv, "target", "rv"))
    r = log_returns.dropna()
    if r.empty or pred is None or pred.empty:
        return None
    y = forward_target(r, horizon=horizon, target=target)
    both = pd.concat([y.rename("y"), pred.rename("p")], axis=1, join="inner").dropna()
    if len(both) < 30:
        return None

    def r2(df: pd.DataFrame) -> float:
        return float(1.0 - ((df["y"] - df["p"]) ** 2).sum() / ((df["y"] - df["y"].mean()) ** 2).sum())

    ex = both[both.index.year != exclude_year]
    return {"horizon": horizon, "target": target, "r2": r2(both),
            "r2_ex": r2(ex) if len(ex) >= 30 else None, "exclude_year": exclude_year,
            "n": int(len(both)), "start": str(both.index[0].date()),
            "end": str(both.index[-1].date())}


_WF_CACHE: dict = {}


# ── PAGE 1 DATA: Live Market Overview ─────────────────────────────────────────
def load_page1() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": []}

    # Defaults: None renders as '—'. Nothing here may default to a plausible number.
    for k in ("spx", "spx_chg", "spx_up", "spx_date", "vix", "vix_date",
              "vvix", "vvix_date", "ts_date", "regime", "regime_date",
              "reg_total", "reg0_n", "reg1_n", "reg2_n", "reg0_pct", "reg1_pct",
              "reg2_pct", "reg_span", "calib_date", "pdv_forecast", "pdv_spread",
              "pdv_date", "pdv_spread_mean", "pdv_spread_pct", "pdv_hist_n",
              "pdv_hist_start", "pdv_hist_end", "pdv_sigma1", "pdv_sigma2",
              "pdv_age", "pdv_rank", "stress", "pdv_vix", "pdv_horizon", "pdv_coef", "pdv_intercept",
              "pdv_halflives", "pdv_garch_beta", "pdv_garch_persistence",
              "clf_feature_date", "clf_label", "clf_conf", "clf_agrees",
              "prob_r0", "prob_r1", "prob_r2", "clf_top_feature", "clf_top_importance",
              "clf_oos_acc", "clf_persistence_acc", "clf_n_test", "clf_test_window",
              "clf_feature_source", "clf_n_features", "pdv_skill", "rule", "latest_session"):
        d[k] = None
    d.update(regime_name="UNAVAILABLE", regime_color="#666", regime_label="—",
             clf_color="#666")
    d.update(ts_tenors=[], ts_values=[], ts_regime="N/A", ts_regime_color="#666",
             pdv_sparkline=[], pdv_sparkline_dates=[], pdv_stale=False,
             clf_available=False, stale=[])

    # ── Market snapshot (DB) ─────────────────────────────────────────────────
    spx_df: Optional[pd.DataFrame] = None
    vix_daily: Optional[pd.Series] = None
    vix_wide: Optional[pd.DataFrame] = None
    try:
        spx_df = _read_spx()
        if len(spx_df) < 2:
            raise ValueError("spx_ohlcv has fewer than two sessions")
        last, prev = spx_df["close"].iloc[-1], spx_df["close"].iloc[-2]
        d["spx"] = round(float(last), 2)
        d["spx_date"] = str(spx_df.index[-1].date())
        d["spx_chg"] = round((float(last) - float(prev)) / float(prev) * 100, 2)
        d["spx_up"] = bool(last >= prev)

        vix_daily = _read_vix_daily()
        if len(vix_daily):
            d["vix"] = round(float(vix_daily.iloc[-1]), 2)
            d["vix_date"] = str(vix_daily.index[-1].date())

        vix_wide = _read_vix_wide()
        if "^VVIX" in vix_wide.columns and vix_wide["^VVIX"].notna().any():
            vv = vix_wide["^VVIX"].dropna()
            d["vvix"] = round(float(vv.iloc[-1]), 2)
            d["vvix_date"] = str(vv.index[-1].date())

        # Term structure from ONE date: the latest session on which every curve
        # tenor has a close. Taking each ticker's own latest row mixed today's
        # VIX with older 9D/3M/6M closes whenever Yahoo omitted the sub-tenors.
        # Tenors are config.VIX_INDEX_TENOR_DAYS, the one map every module uses
        # (Cboe: 9/30/93/184 calendar days); a local copy labelled ^VIX6M 182d.
        from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
        tenor_days = dict(VIX_INDEX_TENOR_DAYS)
        curve_cols = [c for c in tenor_days if c in vix_wide.columns]
        recent = vix_wide.loc[vix_wide.index >= vix_wide.index.max() - pd.Timedelta(days=90)]
        curve_cols = [c for c in curve_cols if recent[c].notna().any()]
        full = vix_wide[curve_cols].dropna()
        if len(curve_cols) >= 2 and len(full):
            ts_row = full.iloc[-1]
            d["ts_date"] = str(full.index[-1].date())
            pts = sorted((tenor_days[c], float(ts_row[c])) for c in curve_cols)
            d["ts_tenors"] = [f"{t}d" for t, _ in pts]
            d["ts_values"] = [round(v, 2) for _, v in pts]
            d["ts_regime"] = "CONTANGO" if pts[-1][1] > pts[0][1] else "BACKWARDATION"
            d["ts_regime_color"] = "#00ff88" if d["ts_regime"] == "CONTANGO" else "#ff3333"
    except Exception as e:
        _record_error(d, "market data (vol_system.db)", e)
        spx_df = vix_daily = vix_wide = None

    # ── Rule regime + frequencies (regime_labels.parquet only) ───────────────
    # The headline is the deterministic RULE label. It is never replaced by the
    # research classifier's output, whatever fails.
    try:
        rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
        reg_series = rl["regime"].dropna().astype(int)
        if reg_series.empty:
            raise ValueError("regime_labels.parquet has no labels")
        reg = int(reg_series.iloc[-1])
        if reg not in (0, 1, 2):
            raise ValueError(f"unknown regime label {reg}")
        d["regime"] = reg
        d["regime_name"] = _REGIME_NAMES[reg]
        d["regime_color"] = _REGIME_COLORS[reg]
        d["regime_label"] = _REGIME_LABELS[reg]
        d["regime_date"] = str(pd.Timestamp(reg_series.index[-1]).date())
        tot = int(len(reg_series))
        d["reg_total"] = tot
        for k in (0, 1, 2):
            n = int((reg_series == k).sum())
            d[f"reg{k}_n"] = n
            d[f"reg{k}_pct"] = round(100.0 * n / tot, 1)
        d["reg_span"] = f"{reg_series.index[0].year} – {reg_series.index[-1].year}"
    except Exception as e:
        _record_error(d, "rule regime labels (regime_labels.parquet)", e)

    # ── The rule's own inputs on the newest session the DB covers ────────────
    # build_rule_label_frame pairs each session's rule label with the
    # observables that produced it (rv₂₀, VIX, VVIX and the adaptive gate),
    # so the headline label can be checked against its inputs on the page.
    try:
        if spx_df is not None and vix_wide is not None:
            from joint_vol_calibration.signals.regime_classifier import build_rule_label_frame

            frame = build_rule_label_frame(
                spx_df.reset_index()[["date", "close", "log_return"]],
                vix_wide.reset_index().rename(columns={"index": "date"}),
            )
            if len(frame):
                last = frame.iloc[-1]
                t = frame.index[-1]
                vv = vix_wide["^VVIX"].get(t) if "^VVIX" in vix_wide.columns else None
                rule = {"date": str(pd.Timestamp(t).date()), "label": int(last["regime"]),
                        "rv20": _num(last.get("rv_20d")), "vix": _num(last.get("vix")),
                        "vvix": _num(vv), "gate": _num(last.get("vvix_gate")),
                        "stored": None}
                if d["regime_date"] == rule["date"]:
                    rule["stored"] = d["regime"]
                d["rule"] = rule
    except Exception as e:
        _record_error(d, "rule inputs (build_rule_label_frame)", e)

    # ── PDV forecast vs VIX, with a like-for-like history ────────────────────
    pdv = None
    pdv_vol: Optional[pd.Series] = None
    try:
        with open(DATA / "pdv_model.pkl", "rb") as f:
            pdv = pickle.load(f)
        if spx_df is None or vix_daily is None:
            raise RuntimeError("needs the market data that failed to load")
        feats = pdv_features(pdv, spx_df["close_return"])          # state variables shown on the card
        pdv_vol = pdv_walk_forward_series(pdv, spx_df.reset_index()[["date", "close", "log_return"]])
        spread = (vix_daily - 100.0 * pdv_vol).dropna()   # VIX pts − forecast vol pts
        if spread.empty:
            raise ValueError("no date has both a VIX close and a PDV forecast")
        cur_date = spread.index[-1]
        d["pdv_date"] = str(cur_date.date())
        d["pdv_forecast"] = round(float(pdv_vol.loc[cur_date]) * 100, 2)
        d["pdv_vix"] = round(float(vix_daily.loc[cur_date]), 2)
        d["pdv_spread"] = round(float(spread.iloc[-1]), 2)
        d["pdv_horizon"] = int(getattr(pdv, "horizon", 1))
        coef = np.asarray(pdv.linear_.coef_, dtype=float)
        d["pdv_coef"] = [round(float(c), 3) for c in coef]
        d["pdv_intercept"] = round(float(pdv.linear_.intercept_), 4)
        d["pdv_halflives"] = [getattr(pdv, "halflife_fast", None),
                              getattr(pdv, "halflife_slow", None)]
        g = getattr(pdv, "garch_", None)
        if g is not None and getattr(g, "beta_", None) is not None:
            d["pdv_garch_beta"] = round(float(g.beta_), 3)
            d["pdv_garch_persistence"] = round(float(g.alpha_ + g.beta_), 3)

        # State variables on the forecast date
        d["pdv_sigma1"] = round(float(feats.loc[cur_date, "sigma1"]) * 100, 2)
        d["pdv_sigma2"] = round(float(feats.loc[cur_date, "sigma2"]) * 100, 2)

        # History = the SAME quantity (VIX − the causal walk-forward forecast) on prior sessions.
        hist = spread.iloc[:-1]
        pct, n_hist = spread_percentile(hist, float(spread.iloc[-1]))
        d["pdv_hist_n"] = n_hist
        if n_hist >= PDV_MIN_HISTORY:
            d["pdv_spread_pct"] = pct          # unrounded: formatted once on the page
            d["pdv_rank"] = percentile_rank(pct)
            d["pdv_spread_mean"] = round(float(hist.mean()), 2)
            d["pdv_hist_start"] = str(hist.index[0].date())
            d["pdv_hist_end"] = str(hist.index[-1].date())
        recent = spread.iloc[-60:]
        d["pdv_sparkline"] = [round(float(x), 2) for x in recent.tolist()]
        d["pdv_sparkline_dates"] = [str(t.date()) for t in recent.index]
    except Exception as e:
        _record_error(d, "PDV forecast (pdv_model.pkl)", e)
        pdv_vol = None

    # COVID stress test, computed ex ante from the DB returns
    try:
        if pdv is not None and spx_df is not None:
            d["stress"] = pdv_stress_forecast(pdv, spx_df["close_return"])
    except Exception as e:
        _record_error(d, "PDV COVID stress test", e)

    # Out-of-sample skill of the model family on the model's own target,
    # recomputed from the DB (the card hard-coded 0.31 / 0.23, the R² of the
    # superseded next-day |r| model and of the kernel, beside a 21-day model).
    try:
        if pdv is not None and spx_df is not None and pdv_vol is not None:
            d["pdv_skill"] = pdv_walk_forward_skill(pdv, spx_df["close_return"], pdv_vol)
    except Exception as e:
        _record_error(d, "PDV walk-forward skill", e)

    # ── Research-only classifier card ────────────────────────────────────────
    # Features come from build_features — the code path the classifier was
    # trained on — with pdv_iv_spread built from the SAME source the training
    # set used (regime_classifier.classifier_features reads it from the model's
    # metadata). Building features by hand fed it pdv_iv_spread = −VIX; feeding
    # the PDV forecast to a classifier trained on the rv₂₀ proxy was a
    # training/serving mismatch.
    try:
        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        if spx_df is None or vix_wide is None:
            raise RuntimeError("needs the market data that failed to load")
        from joint_vol_calibration.signals.regime_classifier import classifier_features

        # classifier_features rebuilds pdv_iv_spread from the source recorded
        # in clf.metadata_ (and refuses a source it cannot rebuild).
        feats = classifier_features(
            clf,
            spx_df.reset_index()[["date", "close", "log_return"]],
            vix_wide.reset_index().rename(columns={"index": "date"}),
        )
        d["clf_feature_source"] = feats.attrs.get("pdv_iv_spread_source")
        cols = list(getattr(clf, "feature_names_", None) or feats.columns)
        d["clf_n_features"] = len(cols)
        row = feats.dropna(subset=cols).iloc[[-1]]
        proba = np.asarray(clf.predict_proba(row[cols])[0], dtype=float)
        c_reg = int(np.argmax(proba))
        d["clf_available"] = True
        d["clf_feature_date"] = str(row.index[0].date())
        d["clf_label"] = _REGIME_LABELS[c_reg]
        d["clf_color"] = _REGIME_COLORS[c_reg]
        d["clf_conf"] = round(float(proba[c_reg]) * 100, 1)
        d["prob_r0"], d["prob_r1"], d["prob_r2"] = (round(float(p) * 100, 1) for p in proba[:3])
        d["clf_agrees"] = (d["regime"] == c_reg) if d.get("regime_date") == d["clf_feature_date"] else None
        # Accuracy as recorded by the classifier's own evaluation (never hand-typed).
        ev = (getattr(clf, "metadata_", None) or {}).get("evaluation") or {}
        if ev.get("oos_acc") is not None and ev.get("persistence_acc") is not None:
            d["clf_oos_acc"] = round(float(ev["oos_acc"]) * 100, 1)
            d["clf_persistence_acc"] = round(float(ev["persistence_acc"]) * 100, 1)
            d["clf_n_test"] = int(ev.get("n_test") or 0)
            tw = ev.get("test_window") or (None, None)
            d["clf_test_window"] = f"{tw[0]} → {tw[1]}" if tw[0] else None
        try:
            imp = clf.feature_importance()
            d["clf_top_feature"] = str(imp.index[0])
            d["clf_top_importance"] = round(float(imp.iloc[0]) * 100, 0)
        except Exception as exc:
            _logger.info("feature importance unavailable: %s", exc)
    except Exception as e:
        _record_error(d, "research classifier (regime_classifier.pkl)", e)

    # ── Headline calibration (date only) ─────────────────────────────────────
    try:
        path, cal = load_headline_calibration()
        d["calib_date"] = _calibration_session(cal, path)
    except Exception as e:
        _record_error(d, "headline calibration", e)

    # ── Staleness of every dated input vs the newest session ─────────────────
    known = [x for x in (d["spx_date"], d["vix_date"], d["regime_date"]) if x]
    ref = max(known) if known else None
    d["latest_session"] = ref
    d["stale"] = [
        _staleness("market", "Last close (SPX)", d["spx_date"], ref),
        _staleness("regime", "Rule regime label", d["regime_date"], ref),
        _staleness("calibration", "Heston calibration", d["calib_date"], ref),
        _staleness("pdv", "PDV forecast", d["pdv_date"], ref),
    ]
    pdv_item = d["stale"][3]
    d["pdv_age"] = pdv_item["age"]
    d["pdv_stale"] = bool(pdv_item["age"] is not None and pdv_item["age"] >= 1)
    d["holidays"] = [str(h.date()) for h in nyse_holidays(
        f"{(ref or '2026')[:4]}-01-01", f"{int((ref or '2026')[:4]) + 2}-12-31")]

    _logger.info(
        "[page1] SPX=%s (%s) | VIX=%s (%s) | VVIX=%s (%s) | TS=%s | regime=%s | PDV=%s | errors=%d",
        d["spx"], d["spx_date"], d["vix"], d["vix_date"], d["vvix"], d["vvix_date"],
        d["ts_date"], d["regime_date"], d["pdv_date"], len(d["errors"]),
    )
    return _clean(d)


# ── PAGE 2 DATA: Calibration Results ─────────────────────────────────────────
def _optimizer_settings(cal: dict) -> Optional[str]:
    """The optimiser settings the result records (methods.optimizer on
    model_version 2 results), or None when it records none."""
    opt = ((cal.get("methods") or {}).get("optimizer") or cal.get("optimizer")
           or cal.get("de_config") or {})
    maxiter = opt.get("maxiter", cal.get("de_maxiter"))
    popsize = opt.get("popsize", cal.get("de_popsize"))
    if maxiter is None or popsize is None:
        return None
    members = opt.get("members")
    text = f"DE maxiter {int(maxiter)}, popsize {int(popsize)}"
    if members is not None:
        text += f" ({int(members)} members)"
    if opt.get("seed") is not None:
        text += f", seed {opt['seed']}"
    polish = opt.get("lbfgsb_polish")
    if polish is None or polish:
        text += " + L-BFGS-B polish"
    return text


def load_page2() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "mat_smiles": [], "anchors": [],
                         "quintic": None}
    for k in ("kappa", "theta", "theta_vol", "sigma", "rho", "v0", "v0_vol", "S",
              "r_pct", "q_pct", "session", "fit_time", "n_evals",
              "spx_rmse", "vix_fut_rmse", "vix_opt_rmse", "feller_lhs", "feller_rhs",
              "feller_margin", "feller_state", "effective_dof", "n_spx_options",
              "n_vix_tenors", "spot_date", "n_spx_expiries", "half_life_days",
              "optimizer", "legacy", "calib_file", "calib_age", "vix_index_rmse",
              "r_ref_pct", "r_ref_source", "carry_mode", "feller_mode", "spx_oos_rmse",
              "spx_oos_wing_rmse", "n_vix_options", "archive"):
        d[k] = None
    d.update(pinned_params=[], active_constraints=[], feller_imposed=None,
             rho_at_boundary=False, spx_underidentified=None, rho_bound=None,
             feller_penalty_weight=None, strikes_per_expiry=None,
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
        # lone stored r is not the rate the fit used. model_version 2 results
        # price every expiry off its own put-call-parity forward and discount;
        # r and q are the medians of the parity-implied r_T, q_T over the
        # anchor expiries, and r_reference is the ^IRX bill (cc) for reference.
        if _num(cal.get("r")) is not None and _num(cal.get("q")) is not None:
            d["carry_recorded"] = True
            d["r_pct"] = round(float(cal["r"]) * 100, 2)
            d["q_pct"] = round(float(cal["q"]) * 100, 2)
        d["carry_mode"] = (cal.get("methods") or {}).get("carry")
        if _num(cal.get("r_reference")) is not None:
            d["r_ref_pct"] = round(float(cal["r_reference"]) * 100, 2)
            d["r_ref_source"] = cal.get("r_reference_source")

        losses = _cal_losses(cal)
        d["spx_rmse"] = round(float(losses["spx_iv_rmse"]), 3) if _num(losses.get("spx_iv_rmse")) is not None else None
        # VIX leg: the Cboe SPOT indices (^VIX9D/^VIX/^VIX3M/^VIX6M) against
        # heston_vix_term_index. Pre-C22 results only have the old key name.
        vix_rmse = losses.get("vix_index_rmse", losses.get("vix_futures_rmse"))
        d["vix_index_rmse"] = round(float(vix_rmse), 3) if _num(vix_rmse) is not None else None
        for key, src in (("spx_oos_rmse", "spx_oos_iv_rmse"),
                         ("spx_oos_wing_rmse", "spx_oos_put_wing_rmse")):
            if _num(losses.get(src)) is not None:
                d[key] = round(float(losses[src]), 3)
        d["n_vix_options"] = cal.get("n_vix_options")
        # The VIX-options leg is in the objective only when its weight is > 0
        # (liquid contracts on the spot session alone do not activate it).
        w3 = (cal.get("weights") or {}).get("w3")
        vix_opt_active = bool(w3)
        if vix_opt_active and _num(losses.get("vix_options_rmse")) is not None:
            d["vix_opt_rmse"] = round(float(losses["vix_options_rmse"]), 2)
        d["vix_opt_disabled"] = not vix_opt_active

        # Feller: a DIAGNOSTIC (PASS / BINDING / VIOLATED and the margin), and a
        # constraint only when the fit imposed it (feller="soft"; the C22
        # default is "off"). Active constraints come from the result itself:
        # box bounds a parameter sits on, plus "feller" when it was imposed
        # and binds.
        from joint_vol_calibration.calibration.joint_calibrator import (
            _CAL_STRIKES_PER_EXPIRY, active_constraints, classify_feller,
            count_pinned_params,
        )
        state, margin = classify_feller(p["kappa"], p["theta"], p["sigma"])
        d["feller_lhs"] = round(2 * p["kappa"] * p["theta"], 5)
        d["feller_rhs"] = round(p["sigma"] ** 2, 5)
        d["feller_margin"] = margin
        d["feller_state"] = state
        # Pre-C22 fits always carried the soft penalty; model_version 2 records it.
        d["feller_mode"] = cal.get("feller") if isinstance(cal.get("feller"), str) else (
            "soft" if d["legacy"] else None)
        imposed = d["feller_mode"] != "off"
        d["feller_imposed"] = imposed
        pinned = cal.get("pinned_params")
        if pinned is None:
            pinned = count_pinned_params(p, state, feller_imposed=imposed)
        d["pinned_params"] = list(pinned)
        ac = cal.get("active_constraints")
        if ac is None:
            ac = active_constraints(p, state, feller_imposed=imposed)
        d["active_constraints"] = list(ac)
        d["effective_dof"] = 5 - len(pinned)
        from joint_vol_calibration.calibration.joint_calibrator import _FELLER_PENALTY
        from joint_vol_calibration.config import HESTON_BOUNDS
        d["rho_bound"] = float(HESTON_BOUNDS["rho"][0])
        d["rho_at_boundary"] = p["rho"] <= d["rho_bound"] + 0.01
        d["feller_penalty_weight"] = float(_FELLER_PENALTY)
        d["strikes_per_expiry"] = int(_CAL_STRIKES_PER_EXPIRY)

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
        # Unrounded: the template formats each value once (rounding to 3 dp
        # here and to 2 dp there printed 12.4249 as 12.43).
        for s in heston_model_smile(p):
            d["mat_smiles"].append({"T": s["T"], "lm": s["lm"],
                                    "iv": [v * 100 for v in s["iv"]]})

        ref = _latest_market_session()
        item = _staleness("calibration", "Heston calibration", d["session"], ref)
        d["calib_age"] = item["age"]
        d["stale"] = [item]
    except Exception as e:
        _record_error(d, "headline calibration", e)

    # Archive census: how many stored fits are current-schema and gate-passing,
    # and how many are pre-C22 (labelled, never ranked).
    try:
        rows = scan_calibrations()
        d["archive"] = {
            "n_total": len(rows),
            "n_legacy": sum(1 for r in rows if r["legacy"]),
            "n_current": sum(1 for r in rows if r["legacy"] is False),
            "n_current_ok": sum(1 for r in rows if r["legacy"] is False and r["ok"]),
        }
    except Exception as e:
        _record_error(d, "calibration archive", e)

    # ── Quintic OU research fit vs the Heston fit of the SAME session ────────
    try:
        qpaths = sorted(glob.glob(str(DATA / "calibrations" / "quintic_cal_*.pkl")))
        if qpaths:
            qpath = qpaths[-1]
            with open(qpath, "rb") as f:
                qcal = pickle.load(f)
            qp = qcal["params"]
            qp = qp if isinstance(qp, dict) else vars(qp)
            ql = qcal.get("leg_losses", {})
            qprov = qcal.get("provenance") or {}
            q_date = _date_str(qprov.get("spot_date")) or _calibration_session(qcal, qpath)
            mc = qprov.get("mc") or {}
            quad = qprov.get("vix_quadrature") or {}
            q = {
                "as_of": q_date,
                "lam_x": round(float(qp["lam_x"]), 3), "lam_y": round(float(qp["lam_y"]), 3),
                "theta": round(float(qp["theta"]), 4), "alpha0": round(float(qp["alpha0"]), 4),
                "alpha1": round(float(qp["alpha1"]), 4), "alpha3": round(float(qp["alpha3"]), 4),
                "alpha5": round(float(qp["alpha5"]), 4), "epsilon": float(qp["epsilon"]),
                # OU half-lives in calendar days, from the unrounded speeds
                "hl_x_days": math.log(2.0) / float(qp["lam_x"]) * 365.0,
                "hl_y_days": math.log(2.0) / float(qp["lam_y"]) * 365.0,
                "spx_rmse": _num(ql.get("spx_iv_rmse")),
                "spx_rmse_fresh": _num(ql.get("spx_iv_rmse_fresh_paths")),
                # model E[VIX_T] vs the VIX-option parity forward (no VX futures)
                "vix_fwd_rmse": _num(ql.get("vix_futures_rmse")),
                "vix_opt_rmse": _num(ql.get("vix_options_rmse")),
                "xi0_check": _num(ql.get("xi0_index_reproduction_rmse")),
                "at_bound": [k for k, v in (qcal.get("at_bound") or {}).items() if v],
                "eps_saturated": abs(float(qp["epsilon"]) + 0.99) < 0.02,
                "n_spx": qprov.get("n_spx_options"), "n_vix_opts": qprov.get("n_vix_options"),
                "mc_paths": mc.get("n_paths"), "quad_points": quad.get("points_per_dim"),
                "fit_time": _num(qcal.get("fit_time")), "n_evals": qcal.get("n_evals"),
                "converged": qcal.get("success"),
                "heston": None,
            }
            # Like-for-like only: the Heston fit of the SAME session, and the
            # same Heston parameters re-priced on the Quintic's own targets
            # (baseline_<session>.json, scripts/heston_quintic_baseline.py).
            hpath = DATA / "calibrations" / f"joint_cal_{q_date}.pkl"
            bpath = DATA / "calibrations" / f"baseline_{q_date}.json"
            if q_date and hpath.exists():
                with open(hpath, "rb") as f:
                    hcal = pickle.load(f)
                hl = _cal_losses(hcal)
                ok, reason = calibration_gate(hcal)
                base = json.loads(bpath.read_text()) if bpath.exists() else {}
                on_q = base.get("on_quintic_targets") or {}
                if base and (base.get("heston") or {}).get("file") != hpath.name:
                    on_q = {}        # baseline built from another Heston fit
                h_spx_q, h_fwd_q, h_opt_q = (_num(on_q.get(k)) for k in
                                             ("spx_iv_rmse", "vix_forward_rmse", "vix_options_rmse"))

                def _delta(a, b):
                    return (a - b) if None not in (a, b) else None

                q["heston"] = {
                    "file": hpath.name, "gate_ok": ok, "gate_reason": reason,
                    "legacy": calibration_is_legacy(hcal),
                    "spx_rmse": _num(hl.get("spx_iv_rmse")),
                    "spx_oos_rmse": _num(hl.get("spx_oos_iv_rmse")),
                    "n_spx": hcal.get("n_spx_options"),
                    "vix_index_rmse": _num(hl.get("vix_index_rmse", hl.get("vix_futures_rmse"))),
                    "sigma": _num((hcal.get("params") or {}).get("sigma")),
                    "active_constraints": hcal.get("active_constraints") or [],
                    "baseline_file": bpath.name if on_q else None,
                    "spx_rmse_on_q": h_spx_q, "n_spx_on_q": on_q.get("n_spx"),
                    "vix_fwd_rmse_on_q": h_fwd_q, "vix_opt_rmse_on_q": h_opt_q,
                    "spx_delta": _delta(q["spx_rmse"], h_spx_q),
                    "vix_fwd_delta": _delta(q["vix_fwd_rmse"], h_fwd_q),
                    "vix_opt_delta": _delta(q["vix_opt_rmse"], h_opt_q),
                }
            d["quintic"] = q
    except Exception as e:
        _record_error(d, "Quintic OU research calibration", e)

    return _clean(d)


# ── PAGE 3 DATA: Greeks Risk Monitor ─────────────────────────────────────────
def _greeks_provenance(path: Path) -> dict:
    """Provenance stored with the surface, normalised to one flat dict.

    greeks.risk_monitor.save_greeks_surface writes it under the parquet
    metadata key b"greeks_provenance" (params, S, r, q and as_of_date at the
    top level; the source calibration under source.calibration_file /
    source.spot_date). The older contract, a flat dict under b"provenance",
    is still read. Returned keys: source_calibration (file name), spot_date,
    S, r, q, params, plus everything else the builder recorded.
    """
    prov: dict = {}
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(path).metadata or {}
        raw = meta.get(b"greeks_provenance") or meta.get(b"provenance")
        if raw:
            prov = json.loads(raw.decode())
    except Exception as exc:
        _logger.info("greeks provenance unreadable: %s", exc)
    if not isinstance(prov, dict) or not prov:
        return {}
    src = prov.get("source") if isinstance(prov.get("source"), dict) else {}
    if not prov.get("source_calibration") and src.get("calibration_file"):
        prov["source_calibration"] = Path(str(src["calibration_file"])).name
    if not prov.get("spot_date"):
        prov["spot_date"] = src.get("spot_date") or prov.get("as_of_date")
    return prov


def load_page3() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "heatmap_rows": [], "heatmap_cols": [],
                         "unstable_nodes": [], "qv_labels": [], "qv_values": [],
                         "vanna_labels": [], "vanna_values": [], "scatter_lm": [],
                         "scatter_z": [], "scatter_T": [], "unstable_maturities": [],
                         "unstable_by_maturity": {}}
    for k in ("n_unstable", "n_cells", "n_grid", "n_missing", "n_maturities", "as_of", "surface_date",
              "vomma_max", "vomma_max_at", "vomma_min", "vomma_min_at", "vomma_mean",
              "provenance", "source_matches_headline", "greeks_age", "unstable_rule"):
        d[k] = None

    try:
        path = DATA / "greeks" / "greeks_surface.parquet"
        gdf = pd.read_parquet(path).copy()
        prov = _greeks_provenance(path)
        # The unstable flag is the surface's own rule (risk_monitor: per-node
        # vomma-P&L ratio ½·vomma·|Δσ|/vega above a threshold, recorded in the
        # provenance). Re-deriving it from a pooled |z| > 2 overrode that rule
        # and showed 5 of 79 nodes where the surface flags 31. Only a surface
        # without the ratio column falls back to the z-score, labelled as such.
        rule = prov.get("unstable_rule") if isinstance(prov.get("unstable_rule"), dict) else {}
        if "vomma_pnl_ratio" in gdf.columns and "is_unstable" in gdf.columns:
            gdf["is_unstable"] = gdf["is_unstable"].astype(bool)
            thr = _num(rule.get("threshold"))
            if thr is not None and ((gdf["vomma_pnl_ratio"].abs() > thr) != gdf["is_unstable"]).any():
                raise ValueError("greeks_surface.parquet: is_unstable disagrees with its recorded "
                                 f"rule |vomma_pnl_ratio| > {thr}")
            d["unstable_rule"] = {"kind": "pnl_ratio", "threshold": thr,
                                  "metric": rule.get("metric"), "iv_shock": rule.get("iv_shock")}
            gdf["flag_metric"] = gdf["vomma_pnl_ratio"].abs()
        else:
            gdf["is_unstable"] = gdf["vomma_zscore"].abs() > 2.0
            d["unstable_rule"] = {"kind": "zscore", "threshold": 2.0}
            gdf["flag_metric"] = gdf["vomma_zscore"].abs()
        flag_thr = d["unstable_rule"]["threshold"]
        # Grid nodes the builder dropped, with its reason (shown on the empty cells)
        dropped = {}
        for item in (prov.get("grid") or {}).get("dropped_nodes") or prov.get("dropped_nodes") or []:
            try:
                dropped[(int(item["T_days"]), round(float(item["log_moneyness"]), 4))] = str(item.get("reason") or "")
            except Exception:
                continue
        # Columns are the surface's own strike grid in ln(K/F): every maturity
        # uses the same log-moneyness grid by construction. Binning it into
        # 0.05-wide buckets (grid spacing 0.039) and drawing only the first node
        # per bucket hid 16 of 69 nodes, one of the three unstable ones among them.
        gdf["lm_key"] = gdf["log_moneyness"].round(4)
        dup = gdf.duplicated(["T_days", "lm_key"])
        if dup.any():
            raise ValueError(f"{int(dup.sum())} Greeks nodes share a (maturity, log-moneyness) cell")
        maturities = sorted(int(t) for t in gdf["T_days"].unique())
        cols = sorted(float(x) for x in gdf["lm_key"].unique())
        grid_lm = ((prov.get("grid") or {}).get("log_moneyness")) or []
        if grid_lm:   # the builder's full grid, so a column with no surviving node still shows
            cols = sorted(set(cols) | {round(float(x), 4) for x in grid_lm})
        d["n_maturities"] = len(maturities)
        d["n_cells"] = int(len(gdf))
        d["n_grid"] = len(maturities) * len(cols)
        d["n_missing"] = d["n_grid"] - d["n_cells"]
        if "as_of_date" in gdf.columns and gdf["as_of_date"].notna().any():
            d["as_of"] = _date_str(gdf["as_of_date"].dropna().iloc[0])

        cell = {(int(r.T_days), float(r.lm_key)): r for r in gdf.itertuples()}
        for T in maturities:
            cells: list[Optional[dict]] = []
            for lm in cols:
                r = cell.get((T, lm))
                if r is None:
                    cells.append({"empty": True, "reason": dropped.get((T, round(lm, 4)))})
                    continue
                z = float(r.vomma_zscore)
                m = float(r.flag_metric)
                # colour = the flag metric relative to its threshold (full at the threshold)
                shade = 2.0 * min(m / flag_thr, 1.0) if d["unstable_rule"]["kind"] == "pnl_ratio" and flag_thr else z
                bg = _vomma_color(shade)
                cells.append({        # unrounded: each value is formatted once, in the template
                    "vomma": float(r.vomma), "z": z, "metric": m,
                    "iv": float(r.iv) * 100, "K": round(float(r.K), 0),
                    "lm": lm, "unstable": bool(r.is_unstable), "bg": bg,
                    # text colour chosen per cell for contrast (white on bright green was 2.7:1)
                    "fg": _text_colour_on(bg),
                })
            d["heatmap_rows"].append({"T": T, "label": f"{T}d", "cells": cells})
        # Column labels from the stored ln(K/F), not from the 4-dp grouping key.
        raw_lm: dict = {}
        for x in list(gdf["log_moneyness"]) + list(grid_lm):
            raw_lm.setdefault(round(float(x), 4), float(x))
        d["heatmap_cols"] = [_fmt(raw_lm.get(c, c), "+.3f") for c in cols]

        unstable = gdf[gdf["is_unstable"]]
        d["unstable_nodes"] = sorted((
            {"T_days": int(u.T_days), "lm": float(u.log_moneyness),
             "K": round(float(u.K), 0), "vomma": float(u.vomma),
             "z": float(u.vomma_zscore), "iv": float(u.iv) * 100,
             "metric": float(u.flag_metric)}
            for u in unstable.itertuples()), key=lambda x: x["metric"], reverse=True)
        d["n_unstable"] = len(d["unstable_nodes"])
        d["unstable_maturities"] = sorted({n["T_days"] for n in d["unstable_nodes"]})
        d["unstable_by_maturity"] = {int(t): int(n) for t, n in gdf.groupby("T_days")["is_unstable"].sum().items()}

        imax, imin = gdf["vomma"].idxmax(), gdf["vomma"].idxmin()
        d["vomma_max"] = float(gdf.loc[imax, "vomma"])
        d["vomma_max_at"] = {"T_days": int(gdf.loc[imax, "T_days"]),
                             "lm": round(float(gdf.loc[imax, "log_moneyness"]), 3)}
        d["vomma_min"] = float(gdf.loc[imin, "vomma"])
        d["vomma_min_at"] = {"T_days": int(gdf.loc[imin, "T_days"]),
                             "lm": round(float(gdf.loc[imin, "log_moneyness"]), 3)}
        d["vomma_mean"] = float(gdf["vomma"].mean())

        # QV convexity = Var[∫v dt] per maturity: one number per maturity (it
        # does not depend on strike), so a single series is plotted.
        qv = gdf.groupby("T_days")["qv_convexity"].agg(["mean", "min", "max"])
        if ((qv["max"] - qv["min"]).abs() > 1e-12 * qv["max"].abs().clip(lower=1)).any():
            _logger.info("qv_convexity varies within a maturity; plotting the mean")
        d["qv_labels"] = [f"{int(t)}d" for t in qv.index]
        d["qv_values"] = (qv["mean"] * 1e4).tolist()

        vanna = gdf.groupby("T_days")["vanna"].mean()
        d["vanna_labels"] = [f"{int(t)}d" for t in vanna.index]
        d["vanna_values"] = vanna.round(4).tolist()

        d["scatter_lm"] = gdf["log_moneyness"].round(4).tolist()
        d["scatter_z"] = gdf["flag_metric"].tolist()      # the flag metric (ratio, or |z| fallback)
        d["scatter_T"] = gdf["T_days"].astype(int).tolist()

        # Provenance + staleness. The shipped surface records no source fit;
        # when it does (parquet metadata), compare with the headline calibration.
        d["provenance"] = prov or None
        if prov.get("params"):
            try:
                _, hcal = load_headline_calibration()
                hp = hcal["params"]
                d["source_matches_headline"] = all(
                    abs(float(prov["params"][k]) - float(hp[k])) <= 1e-9 * max(1.0, abs(float(hp[k])))
                    for k in ("kappa", "theta", "sigma", "rho", "v0"))
            except Exception as exc:
                _logger.info("headline comparison unavailable: %s", exc)
        surface_date = _date_str(prov.get("spot_date")) or d["as_of"]
        ref = _latest_market_session()
        item = _staleness("greeks", "Greeks surface", surface_date, ref)
        d["greeks_age"] = item["age"]
        d["surface_date"] = surface_date
        d["stale"] = [item]
    except Exception as e:
        _record_error(d, "Greeks surface (greeks_surface.parquet)", e)

    return _clean(d)


# ── PAGE 4 DATA: Backtest Results ─────────────────────────────────────────────
def half_year_sharpes(nav: pd.Series) -> list[dict]:
    """Per half-year Sharpe of ONE in-sample backtest run, via the engine's metrics.

    Each window's first return is measured from the previous window's last NAV
    (slicing dropped it), and the Sharpe comes from
    backtest_engine.compute_metrics with rf = 0: the NAV is the option P&L on
    $1M of collateral that is assumed to earn ^IRX separately, so the P&L is
    already an excess return (the Sharpe convention adopted for the site in
    C22; the headline KPIs are whatever the stored engine results record). The
    old chart charged a flat 5% hurdle, clipped at ±4 and called the slices
    'walk-forward OOS'.
    A window with a flat NAV has no defined Sharpe and is returned as None.
    """
    from joint_vol_calibration.backtest.backtest_engine import compute_metrics

    nav = nav.dropna()
    out = []
    for year in sorted(set(nav.index.year)):
        for half, (s, e) in enumerate(((f"{year}-01-01", f"{year}-06-30"),
                                       (f"{year}-07-01", f"{year}-12-31")), 1):
            sub = nav.loc[s:e]
            if sub.empty:
                continue
            prior = nav.loc[:pd.Timestamp(s) - pd.Timedelta(days=1)]
            window = pd.concat([prior.iloc[-1:], sub]) if len(prior) else sub
            rets = window.pct_change().dropna()
            if len(rets) < 2:
                continue
            flat = bool(rets.abs().max() < 1e-10)
            sharpe = None if flat else _num(compute_metrics(
                pd.DataFrame({"nav": window}), rf=0.0).get("sharpe"))
            out.append({
                "period": f"{year}-H{half}", "start": str(sub.index[0].date()),
                "end": str(sub.index[-1].date()), "sharpe": sharpe, "flat": flat,
                "partial": bool(sub.index[-1] < pd.Timestamp(e) - pd.Timedelta(days=7)
                                and sub.index[-1] == nav.index[-1]),
            })
    return out


def annual_returns(nav: pd.Series) -> list[dict]:
    """Calendar-year returns based on the previous year's last NAV.

    Using each year's first NAV as the base dropped the first session of every
    year, so the years did not compound to the total.
    """
    nav = nav.dropna()
    year_end = nav.groupby(nav.index.year).last()
    out, base = [], float(nav.iloc[0])
    for yr, end in year_end.items():
        out.append({"year": int(yr), "ret_pct": (float(end) / base - 1.0) * 100.0,
                    "partial": bool(yr == nav.index[-1].year and nav.index[-1].month < 12)})
        base = float(end)
    return out


def window_returns(nav: pd.Series, periods: dict, end: Optional[str] = None) -> list[dict]:
    """Return over each named (start, end) window, based on the last NAV before it opens.

    backtest_engine.compute_crisis_performance divides the window's last NAV
    by its FIRST NAV, which drops the first session's return: the page read
    "FedHikes 2022 (2022-01-01 → 2022-12-31) +5.1%" beside a 2022 annual
    return of +4.71% for the same sessions. As in annual_returns and
    half_year_sharpes, the base is the close of the session before the window
    (a window opening on the first session of the curve is based on that
    session's NAV). Windows are clipped at `end`.
    """
    nav = nav.dropna()
    out = []
    for name, (start, stop) in periods.items():
        stop = min(stop, end) if (stop and end) else (stop or end)
        sub = nav.loc[start:stop]
        prior = nav[nav.index < pd.Timestamp(start)]
        row = {"name": name, "start": start, "end": stop, "total_return": None, "n_days": 0}
        if len(sub) and (len(prior) or len(sub) > 1):
            base = float(prior.iloc[-1]) if len(prior) else float(sub.iloc[0])
            row["total_return"] = float(sub.iloc[-1]) / base - 1.0
            row["n_days"] = int(len(sub) if len(prior) else len(sub) - 1)   # returns in the window
        out.append(row)
    return out


def _trade_summary(trades: list, sig: str) -> dict:
    """Per-signal trade-log facts quoted in the page prose."""
    tr = [t for t in trades if getattr(t, "signal", None) == sig]
    if not tr:
        return {}
    pnl = np.array([float(t.net_pnl) for t in tr])
    total = float(pnl.sum())
    worst = tr[int(np.argmin(pnl))]
    by_year: dict[int, float] = {}
    for t in tr:
        y = int(pd.Timestamp(t.exit_date).year)
        by_year[y] = by_year.get(y, 0.0) + float(t.net_pnl)
    top5 = float(np.sort(pnl)[::-1][:5].sum())
    return {
        "n": len(tr), "total": total,
        "win_rate": float((pnl > 0).mean()),
        "n_short": int(sum(1 for t in tr if getattr(t, "direction", 0) < 0)),
        "worst_pnl": float(pnl.min()),
        "worst_entry": str(pd.Timestamp(worst.entry_date).date()),
        "worst_exit": str(pd.Timestamp(worst.exit_date).date()),
        "by_year": [{"year": y, "pnl": v} for y, v in sorted(by_year.items())],
        "positive_years": [y for y, v in sorted(by_year.items()) if v > 0],
        "negative_years": [{"year": y, "pnl": v} for y, v in sorted(by_year.items()) if v < 0],
        "top5_share": (top5 / total) if total > 0 else None,
    }


def _read_csv_rows(path: Path) -> list[dict]:
    df = pd.read_csv(path)
    return [_clean(r) for r in df.to_dict("records")]


def bridge_evidence(bridge: list[dict], grid: list[dict]) -> dict:
    """Numbers the audit table quotes from the C22 bridge and the β grid.

    The bridge (backtest/c22_bridge.py) applies the C22 corrections one at a
    time from the published C21 engine; each row's d_pnl is that step's effect
    on portfolio P&L. The β grid (run_backtest --beta-sensitivity) re-runs the
    canonical backtest at several ATM-vol / VIX ratios.
    """
    rows = {r["step"]: dict(r, row=int(r["row"])) for r in bridge}
    out: dict[str, Any] = {"rows": rows}
    order = sorted(bridge, key=lambda r: int(r["row"]))
    for prev, cur in zip(order, order[1:]):
        rows[cur["step"]]["prev"] = prev
    g = sorted(grid, key=lambda r: float(r["beta"]))
    out["grid"] = g

    def flip(col: str) -> Optional[dict]:
        """Adjacent grid betas where the sleeve's P&L changes sign."""
        for lo, hi in zip(g, g[1:]):
            if (lo[col] > 0) != (hi[col] > 0):
                return {"lo": float(lo["beta"]), "hi": float(hi["beta"]),
                        "positive_above": hi[col] > 0}
        return None

    out["s1_flip"] = flip("s1_pnl")
    out["s3_flip"] = flip("s3_pnl")
    at_one = [r for r in g if abs(float(r["beta"]) - 1.0) < 1e-9]
    out["s1_at_beta1"] = at_one[0]["s1_pnl"] if at_one else None

    # Evidence sentences for the audit table, each from one bridge row.
    # The β row's "before" value is the previous bridge row (already β = 1,
    # but before the later C22 steps); the grid re-runs the FULL C22
    # configuration, so its β = 1 figure is a different backtest and is
    # labelled as such ("at β = 1 S1 would show +$2.6M" beside "+$576K → …"
    # read as a contradiction).
    ev: dict[str, str] = {}
    r = rows.get("beta")
    if r and r.get("prev"):
        later = [x for x in order if int(x["row"]) > int(r["row"])]
        txt = (f"C22 bridge row {r['row']}: S1 {_usd_k(r['prev']['s1_pnl'])} → "
               f"{_usd_k(r['s1_pnl'])} from β alone"
               + (", before the later C22 steps" if later else ""))
        if out["s1_at_beta1"] is not None:
            txt += (f"; at β = 1 under the full C22 configuration (β grid), S1 shows "
                    f"{_usd_k(out['s1_at_beta1'])}")
        ev["beta"] = txt
    for step in ("r2_gate", "pdv_h21"):
        r = rows.get(step)
        if r:
            ev[step] = f"C22 bridge row {r['row']}: portfolio {_usd_k(r['d_pnl'])}"
    cost_steps = [rows[k] for k in ("hedge_costs", "financing", "sleeve_nav", "corr_netting") if k in rows]
    if cost_steps:
        ev["costs"] = ("C22 bridge rows " + ", ".join(str(r["row"]) for r in cost_steps) + ": "
                       + ", ".join(_usd_k(r["d_pnl"]) for r in cost_steps))
    r = rows.get("sharpe")
    if r and r.get("prev"):
        moved = abs(float(r["d_pnl"])) > 0.5
        ev["sharpe"] = (f"C22 bridge row {r['row']}: Sharpe {_fmt(r['prev']['sharpe'], '.2f')} → "
                        f"{_fmt(r['sharpe'], '.2f')}"
                        + (f", P&L {_usd_k(r['d_pnl'])}" if moved else " with P&L unchanged"))
    out["evidence"] = ev
    return out


# The fifth chain behind β is a live Yahoo ^SPX capture (Cboe global trading
# hours, 2026-10-12 04:39 ET, run_backtest.fetch_live_spx_chain). The DB holds
# end-of-day chains only, so the capture is kept with the test fixtures.
BETA_LIVE_CAPTURE = ROOT / "joint_vol_calibration" / "tests" / "fixtures" / "spx_chain_live_20261012.json"
_BETA_CACHE: dict = {}


def _import_run_backtest():
    """joint_vol_calibration.backtest.run_backtest without its logging side effect.

    The module calls logging.basicConfig(level=INFO) at import. The freeze runs
    as a subprocess of cloud_refresh, which relays every stderr line as a
    warning, so that call would turn every later INFO record into a warning.
    """
    root = logging.getLogger()
    handlers, level = list(root.handlers), root.level
    try:
        from joint_vol_calibration.backtest import run_backtest
    finally:
        for h in [h for h in root.handlers if h not in handlers]:
            root.removeHandler(h)
        root.setLevel(level)
    return run_backtest


def beta_estimation(regimes: Optional[pd.Series] = None) -> dict:
    """The SPX chains behind β (backtest_engine.ATM_TO_VIX_RATIO) and the later checks.

    Per-chain ratios (30-day ATM-forward implied vol over the same-time VIX)
    are run_backtest.estimate_beta(include_live=False) on the project DB: DB
    chains in BETA_ESTIMATION_SESSIONS are "included", later ones are
    "out-of-sample" checks that the frozen β does not absorb. The live
    capture's ratio is the one recorded with it (ratio_full_chain). The page
    said "estimated from five real 2026 SPX chains (0.80–0.84)" with the
    range typed in, beside a DB that holds five chains of its own — a
    different five, one of which (2026-10-16) gives 0.70.
    """
    from joint_vol_calibration.backtest.backtest_engine import BETA_ESTIMATION_SESSIONS

    rb = _import_run_backtest()
    db = Path(rb.DB_PATH)
    st = db.stat()
    key = (str(db.resolve()), st.st_mtime_ns, st.st_size)
    if key not in _BETA_CACHE:            # ~1.5 s; the test-suite renders the page many times
        _BETA_CACHE[key] = rb.estimate_beta(include_live=False)
    table = _BETA_CACHE[key]
    live = json.loads(BETA_LIVE_CAPTURE.read_text())

    def regime_on(session: Optional[str]) -> Optional[str]:
        if regimes is None or not session:
            return None
        lab = regimes.get(pd.Timestamp(session))
        return _REGIME_LABELS[int(lab)] if _num(lab) is not None else None

    chains = []
    for r in table.to_dict("records"):
        status = str(r.get("status") or "")
        role = ("estimation" if status.startswith("included") else
                "later" if status.startswith("out-of-sample") else "excluded")
        session = _date_str(r.get("session")) or str(r.get("source") or "")
        chains.append({"kind": "DB chain", "session": session, "ratio": _num(r.get("ratio")),
                       "role": role, "status": status, "regime": regime_on(session)})
    chains.append({"kind": "live capture", "session": str(live["as_of_et"])[:16] + " ET",
                   "ratio": _num(live["ratio_full_chain"]), "role": "estimation",
                   "status": str(live.get("note") or ""), "regime": None})
    est = [c for c in chains if c["role"] == "estimation" and c["ratio"] is not None]
    later = [c for c in chains if c["role"] == "later" and c["ratio"] is not None]
    missing = sorted(set(BETA_ESTIMATION_SESSIONS)
                     - {c["session"] for c in est if c["kind"] == "DB chain"})
    if missing:
        raise ValueError("β estimation chain(s) missing or excluded: " + ", ".join(missing))
    r_est = [c["ratio"] for c in est]
    r_all = r_est + [c["ratio"] for c in later]
    return {
        "chains": chains, "later": later,
        "n_db": sum(1 for c in est if c["kind"] == "DB chain"),
        "n_live": sum(1 for c in est if c["kind"] == "live capture"),
        "mean": float(np.mean(r_est)), "lo": float(min(r_est)), "hi": float(max(r_est)),
        "mean_all": float(np.mean(r_all)), "n_all": len(r_all),
    }


def load_page4() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "metrics_rows": [], "signal_metrics": [],
                         "signal_pnl": [], "signal_labels": [],
                         "research_s1": {}, "research_s1c": {}, "research_s4": {},
                         "eq_dates": [], "eq_nav": [], "eq_nav_s1c": [], "eq_nav_s3": [],
                         "eq_nav_s4": [], "eq_nav_s1": [], "eq_summary": [], "ann": [], "ann_years": [],
                         "ann_rets": [], "wf": [], "wf_periods": [], "wf_sharpes": [],
                         "crisis": [], "sleeves": {}, "open_trades": []}
    for k in ("final_nav", "total_pnl", "start_date", "end_date",
              "wf_neg", "wf_total", "kpi", "sharpe_convention", "run_date",
              "beta", "bridge"):
        d[k] = None

    # Option chains the DB holds (the roadmap states what the backtest could and
    # could not use; it hard-coded "a single options snapshot (2026-03-24)").
    d["option_snapshots"] = None
    try:
        with _db() as con:
            rows = con.execute(
                "SELECT snapshot_date, COUNT(*) FROM options_snapshots "
                "WHERE underlying = 'SPX' GROUP BY snapshot_date ORDER BY snapshot_date").fetchall()
        d["option_snapshots"] = {"dates": [r[0] for r in rows],
                                 "n_rows": int(sum(r[1] for r in rows))}
    except Exception as e:      # the roadmap then omits the count; not a page error
        _logger.info("option snapshot count unavailable: %s", e)

    eq: Optional[pd.DataFrame] = None
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        nav = eq["nav"]
        d["start_date"] = str(eq.index[0].date())
        d["end_date"] = str(eq.index[-1].date())
        d["final_nav"] = round(float(nav.iloc[-1]), 0)
        d["total_pnl"] = round(float(nav.iloc[-1] - nav.iloc[0]), 0)

        # Full-resolution curves (1.8k points): the old [::3] downsample
        # dropped the final session and clipped extremes.
        def _norm(col: str) -> list:
            if col not in eq.columns:
                return []
            s = eq[col]
            return ((s / float(s.iloc[0]) * 100) - 100).round(2).tolist()

        d["eq_dates"] = [t.strftime("%Y-%m-%d") for t in eq.index]
        # Final / lowest / highest of each curve for the table alternative, from
        # the unrounded curves: printing the 2-dp chart series at 1 dp showed S3's
        # final +5.647% as "+5.7%".
        d["eq_summary"] = []
        for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C Contrarian"), ("nav_s3", "S3 Dispersion"),
                          ("nav_s4", "S4 VRP"), ("nav_s1", "S1 PDV (reference)")):
            if col in eq.columns and eq[col].notna().any():
                rel = eq[col].dropna() / float(eq[col].dropna().iloc[0]) - 1.0
                d["eq_summary"].append({"name": name, "final": float(rel.iloc[-1]),
                                        "low": float(rel.min()), "high": float(rel.max())})
        d["eq_nav"] = _norm("nav")
        d["eq_nav_s1c"] = _norm("nav_s1c")
        d["eq_nav_s3"] = _norm("nav_s3")
        d["eq_nav_s4"] = _norm("nav_s4")
        d["eq_nav_s1"] = _norm("nav_s1")

        d["ann"] = annual_returns(nav)
        d["ann_years"] = [a["year"] for a in d["ann"]]
        d["ann_rets"] = [round(a["ret_pct"], 2) for a in d["ann"]]

        d["wf"] = half_year_sharpes(nav)
        d["wf_periods"] = [w["period"] + (" (partial)" if w["partial"] else "") for w in d["wf"]]
        d["wf_sharpes"] = [None if w["sharpe"] is None else round(w["sharpe"], 2) for w in d["wf"]]
        defined = [w["sharpe"] for w in d["wf"] if w["sharpe"] is not None]
        d["wf_neg"] = sum(1 for s in defined if s < 0)
        d["wf_total"] = len(defined)
        d["wf_flat"] = sum(1 for w in d["wf"] if w["flat"])

        from joint_vol_calibration.backtest.backtest_engine import CRISIS_PERIODS
        # The window dates are shown with each return: "Tariffs_2025" ends at the
        # backtest's last session (2025-03-24), before the April 2025 shock.
        d["crisis"] = window_returns(nav, CRISIS_PERIODS, end=d["end_date"])
    except Exception as e:
        _record_error(d, "backtest equity curve (full_results.parquet)", e)
        eq = None

    # ── Engine metrics + trade log: the single source for every KPI ──────────
    try:
        with open(DATA / "backtest" / "full_results.pkl", "rb") as fh:
            state = pickle.load(fh)
        em = state.get("metrics") or {}
        trades = state.get("all_trades") or []
        if not em:
            raise ValueError("full_results.pkl carries no engine metrics")
        d["run_date"] = _date_str(state.get("run_date"))
        d["sharpe_convention"] = em.get("sharpe_convention") or state.get("sharpe_convention")
        d["beta"] = _num(state.get("atm_to_vix_ratio", em.get("atm_to_vix_ratio")))

        if eq is not None:
            # Both artifacts must come from the same engine run.
            cum = float(eq["nav"].iloc[-1] / eq["nav"].iloc[0] - 1.0)
            if abs(cum - float(em["cumulative_return"])) > 1e-9:
                raise ValueError(
                    f"full_results.parquet (cum {cum:.6f}) and full_results.pkl "
                    f"(cum {float(em['cumulative_return']):.6f}) are from different runs")
            # The portfolio must be the equal-weight mean of PORTFOLIO_SLEEVES.
            pcols = [f"pnl_{s}" for s in PORTFOLIO_SLEEVES]
            if all(c in eq.columns for c in pcols):
                resid = (eq["daily_pnl"] - eq[pcols].sum(axis=1) / len(pcols)).abs().max()
                if resid > 1e-6 * max(1.0, float(eq["daily_pnl"].abs().max())):
                    raise ValueError("portfolio daily_pnl is not the equal-weight mean of "
                                     + "/".join(PORTFOLIO_SLEEVES))

        # Portfolio trade statistics from the trade log of the three sleeves.
        # (The engine's top-level n_trades also counts the reference sleeves;
        # counting position changes double-counted round trips.) The log
        # includes the position still open on the last session (is_open: its
        # net P&L is the entry cost alone), which the engine's win rates count
        # as a loss; the page showed "Round trips 208, win rate 38.0%" with it.
        # Win rate and average are taken over the CLOSED trades; the average
        # is portfolio dollars per trade (the portfolio is the equal-weight
        # mean of the sleeves, so each trade contributes 1/3 of its P&L).
        n_rt = sum(int(em.get(f"{s}_n_trades") or 0) for s in PORTFOLIO_SLEEVES)
        ptr = [t for t in trades if getattr(t, "signal", None) in PORTFOLIO_SLEEVES]
        if len(ptr) != n_rt:
            raise ValueError(f"the trade log has {len(ptr)} portfolio trades, the engine metrics {n_rt}")
        port_pnl = d["total_pnl"]
        if port_pnl is not None and abs(sum(float(t.net_pnl) for t in ptr) / len(PORTFOLIO_SLEEVES)
                                        - port_pnl) > 1.0:
            raise ValueError("the portfolio trades do not add up to the portfolio P&L")
        closed = np.array([float(t.net_pnl) for t in ptr if not getattr(t, "is_open", False)])
        n_closed = int(len(closed))
        d["open_trades"] = [{"signal": t.signal, "entry": str(pd.Timestamp(t.entry_date).date()),
                             "net_pnl": float(t.net_pnl)}
                            for t in trades if getattr(t, "is_open", False)
                            and t.signal in PORTFOLIO_SLEEVES + ("s1", "s2")]
        kpi = {
            "cum_ret": _num(em.get("cumulative_return")),
            "ann_ret": _num(em.get("ann_return")),
            "sharpe": _num(em.get("sharpe")),
            "sortino": _num(em.get("sortino")),
            "mdd": _num(em.get("max_drawdown")),
            "dd_dur": em.get("dd_duration_days"),
            "calmar": _num(em.get("calmar")),
            "best_day": _num(em.get("best_day")),
            "worst_day": _num(em.get("worst_day")),
            "n_days": em.get("n_days"),
            "n_trades": n_rt, "n_closed": n_closed, "n_open": n_rt - n_closed,
            "win_rate": float((closed > 0).mean()) if n_closed else None,
            "avg_pnl": float(closed.sum()) / len(PORTFOLIO_SLEEVES) / n_closed if n_closed else None,
            "s3_null_p": _num(em.get("s3_null_p_value")),
            "cash_ret": _num(em.get("cash_cum_return")),
        }
        d["kpi"] = kpi

        def _pct(x):
            return _fmt(None if x is None else x * 100, ".2f") + ("%" if x is not None else "")

        def _f(x, n=3):
            return _fmt(x, f".{n}f")

        def _dol(x):
            return "—" if x is None else (f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")

        d["metrics_rows"] = [
            {"label": "Cumulative Return", "value": _pct(kpi["cum_ret"]), "positive": (kpi["cum_ret"] or 0) > 0},
            {"label": "Ann. Return", "value": _pct(kpi["ann_ret"]), "positive": (kpi["ann_ret"] or 0) > 0},
            {"label": "Sharpe", "value": _f(kpi["sharpe"]), "positive": (kpi["sharpe"] or 0) > 0},
            {"label": "Sortino", "value": _f(kpi["sortino"]), "positive": (kpi["sortino"] or 0) > 0},
            {"label": "Max Drawdown", "value": _pct(kpi["mdd"]), "positive": False},
            {"label": "DD Duration", "value": "—" if kpi["dd_dur"] is None else f"{int(kpi['dd_dur']):,} sessions", "positive": False},
            {"label": "Calmar", "value": _f(kpi["calmar"]), "positive": (kpi["calmar"] or 0) > 0},
            {"label": "Win Rate (closed trades)", "value": _pct(kpi["win_rate"]), "positive": (kpi["win_rate"] or 0) > 0.5},
            {"label": "Avg Portfolio P&L / Closed Trade", "value": _dol(kpi["avg_pnl"]), "positive": (kpi["avg_pnl"] or 0) > 0},
            {"label": "Best Day", "value": _pct(kpi["best_day"]), "positive": True},
            {"label": "Worst Day", "value": _pct(kpi["worst_day"]), "positive": False},
            {"label": "Trades", "value": f"{n_rt:,}" + (f" ({n_rt - n_closed} open)" if n_rt > n_closed else ""),
             "positive": True},
            {"label": "Return Days", "value": "—" if kpi["n_days"] is None else f"{int(kpi['n_days']):,}", "positive": True},
        ]

        sleeves = {}
        for sig in ("s1c", "s3", "s4", "s1", "s2"):
            tp = _num(em.get(f"{sig}_total_pnl"))
            sleeves[sig] = {
                "total_pnl": tp, "sharpe": _num(em.get(f"{sig}_sharpe")),
                "ann_ret": _num(em.get(f"{sig}_ann_return")),
                "win_rate": _num(em.get(f"{sig}_win_rate")),
                "n_trades": em.get(f"{sig}_n_trades"),
                "n_open": sum(1 for t in trades if getattr(t, "signal", None) == sig
                              and getattr(t, "is_open", False)),
                "summary": _trade_summary(trades, sig),
            }
        d["sleeves"] = sleeves

        def _n_trades(s: dict) -> str:
            if s["n_trades"] is None:
                return "—"
            return str(int(s["n_trades"])) + (f" ({s['n_open']} open)" if s["n_open"] else "")

        def _card(sig: str) -> dict:
            s = sleeves[sig]
            return {"total_pnl": _dol(s["total_pnl"]), "sharpe": _f(s["sharpe"]),
                    "win_rate": _pct(s["win_rate"]), "n_trades": _n_trades(s),
                    "green": (s["total_pnl"] or 0) > 0}

        d["research_s1"], d["research_s1c"], d["research_s4"] = _card("s1"), _card("s1c"), _card("s4")

        d["signal_labels"] = ["S1C contrarian", "S3 dispersion", "S4 VRP"]
        d["signal_pnl"] = [None if sleeves[s]["total_pnl"] is None else round(sleeves[s]["total_pnl"] / 1000, 1)
                           for s in PORTFOLIO_SLEEVES]

        for sig, lbl, is_ref in (("s1c", "S1C Contrarian", False), ("s3", "S3 Dispersion", False),
                                 ("s4", "S4 VRP", False), ("s1", "S1 PDV (ref)", True),
                                 ("s2", "S2 VIX TS (ref)", True)):
            s = sleeves[sig]
            if s["total_pnl"] is None:
                continue
            d["signal_metrics"].append({
                "label": lbl, "sharpe": _f(s["sharpe"]), "ann_ret": _pct(s["ann_ret"]),
                "win_rate": _pct(s["win_rate"]), "n_trades": _n_trades(s),
                "total_pnl": _dol(s["total_pnl"]), "green": s["total_pnl"] > 0,
                "is_research": is_ref,
            })
    except Exception as e:
        _record_error(d, "backtest engine results (full_results.pkl)", e)

    # ── C22 bridge + β grid (the audit table's evidence column) ──────────────
    try:
        bridge = _read_csv_rows(DATA / "backtest" / "c22_bridge.csv")
        grid = _read_csv_rows(DATA / "backtest" / "beta_sensitivity.csv")
        if not bridge or not grid:
            raise ValueError("empty C22 bridge or β grid")
        last = max(bridge, key=lambda r: int(r["row"]))
        if d["total_pnl"] is not None and abs(float(last["pnl"]) - d["total_pnl"]) > 1.0:
            raise ValueError(f"c22_bridge.csv ends at P&L {last['pnl']:,.0f}, the stored run at "
                             f"{d['total_pnl']:,.0f}: the bridge is from another run")
        beta = d["beta"]
        if beta is not None:
            at = [r for r in grid if abs(float(r["beta"]) - beta) < 1e-9]
            if not at or (d["total_pnl"] is not None and abs(float(at[0]["pnl"]) - d["total_pnl"]) > 1.0):
                raise ValueError(f"beta_sensitivity.csv has no row matching the stored run at β = {beta}")
        d["bridge"] = bridge_evidence(bridge, grid)
    except Exception as e:
        _record_error(d, "C22 bridge / β grid (c22_bridge.csv, beta_sensitivity.csv)", e)

    # ── β: the chains it was estimated from, and the later checks ────────────
    d["beta_est"] = None
    try:
        try:
            regimes = pd.read_parquet(DATA / "signals" / "regime_labels.parquet", columns=["regime"])["regime"]
        except Exception as exc:          # the chain table then shows no regime
            _logger.info("regime labels unavailable for the β chains: %s", exc)
            regimes = None
        be = beta_estimation(regimes)
        if d["beta"] is not None and round(be["mean"], 2) != round(d["beta"], 2):
            raise ValueError(f"the estimation chains average {be['mean']:.4f}, but the stored "
                             f"run used β = {d['beta']}")
        grid_betas = [float(r["beta"]) for r in ((d["bridge"] or {}).get("grid") or [])]
        be["mean_all_in_grid"] = any(abs(g - round(be["mean_all"], 2)) < 1e-9 for g in grid_betas)
        d["beta_est"] = be
    except Exception as e:
        _record_error(d, "β estimation chains (run_backtest.estimate_beta)", e)

    # ── Facts the change log quotes, recomputed from the DB ──────────────────
    # (the log hard-coded "2018: ~1.5%" for a year whose ^IRX averaged 1.95%,
    # and "~53%" / "~25%" R2 shares for gates that give 48% / 23% here).
    d["tbill_by_year"], d["r2_share"] = {}, None
    try:
        if d["start_date"] and d["end_date"]:
            with _db() as con:
                tb = pd.read_sql("SELECT date, rate FROM tbill_rates WHERE date BETWEEN ? AND ?",
                                 con, params=(d["start_date"], d["end_date"]), parse_dates=["date"])
            if len(tb):
                yearly = tb.groupby(tb["date"].dt.year)["rate"].mean()
                d["tbill_by_year"] = {int(y): float(v) for y, v in yearly.items()}
            spx_df, vix_wide = _read_spx(), _read_vix_wide()
            from joint_vol_calibration.signals.regime_classifier import (
                VVIX_REGIME2_THRESHOLD, build_regime_labels)
            spx_in = spx_df.reset_index()[["date", "close", "log_return"]]
            vw_in = vix_wide.reset_index().rename(columns={"index": "date"})
            share = {}
            for key, gate in (("adaptive", None), ("fixed", VVIX_REGIME2_THRESHOLD)):
                lab = build_regime_labels(spx_in, vw_in, gate).loc[d["start_date"]:d["end_date"]]
                share[key] = float((lab == 2).mean()) if len(lab) else None
            share["fixed_level"] = float(VVIX_REGIME2_THRESHOLD)
            d["r2_share"] = share
    except Exception as e:
        _record_error(d, "change-log facts (T-bill rates, R2 shares)", e)

    def _pct0(x: Optional[float]) -> str:
        return "—" if x is None else f"{x * 100:.1f}%"

    # The shipped classifier's own evaluation record (quoted beside the C17 figures)
    d["clf_eval"] = None
    try:
        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
            ev = (getattr(pickle.load(f), "metadata_", None) or {}).get("evaluation") or {}
        if ev.get("oos_acc") is not None and ev.get("persistence_acc") is not None:
            d["clf_eval"] = {"oos_acc": float(ev["oos_acc"]), "persistence_acc": float(ev["persistence_acc"]),
                             "n_test": int(ev.get("n_test") or 0)}
    except Exception as exc:     # page 1 reports a broken classifier; here the clause is omitted
        _logger.info("classifier evaluation unavailable: %s", exc)

    tby = d["tbill_by_year"]
    rate_txt = ", ".join(f"{y}: {v * 100:.2f}%" for y, v in sorted(tby.items())
                         if y in (min(tby), 2023, max(tby))) if tby else "not available"
    sh = d["r2_share"] or {}

    # ── Improvement Summary (Steps 1–8) — a historical change log ─────────────
    d["improvements"] = [
        {"step": "1", "name": "Disable VIX options leg",
         "what": "JOINT_W3 = 0.0 (was 0.2). Attributed at the time to the Heston CIR density; "
                 "C22 traced the 37.14 vp to a broken VIX put pricer, spot-VIX vegas and a "
                 "stale chain, so it was not evidence about the model.",
         "before": "VIX-options leg in the objective with a broken pricer",
         "after":  "SPX + VIX index objective; VIX-options leg off (too few sessions of quotes)."},
        {"step": "2", "name": "Vega-weighted anchor selection",
         "what": "Select 9 highest-BS-vega options per expiry (was log-uniform grid). Moneyness lo 0.70 (was 0.75).",
         "before": "Log-uniform includes low-information deep OTM options",
         "after":  "Near-ATM anchors only. C22 reverted it: the top-vega strikes were adjacent "
                   "ATM calls and the fit never saw the put wing; anchors are now a "
                   "forward-moneyness z-grid with >= 3 wing puts per expiry."},
        {"step": "3", "name": "SVI smoothing — never wired in; removed (C22)",
         "what": "SVI/SSVI helpers were written, but only tests called them: the calibrator "
                 "always fitted the quotes directly. C22 removed them.",
         "before": "Claimed: Heston calibrated to an SSVI-smoothed surface",
         "after":  "Calibrates to two-sided parity-forward Black-76 IVs; no smoothing step."},
        {"step": "4", "name": "Per-date T-bill rate",
         "what": "^IRX 3M T-bill from the DB replaces the fixed r = 0.045 in the backtest.",
         "before": "Fixed r = 0.045 throughout 2018–2025",
         "after":  f"Per-date rate (annual mean, cc: {rate_txt})."},
        {"step": "5", "name": "Adaptive VVIX threshold",
         "what": "Rolling 252-day 80th percentile of VVIX as R2 gate (was fixed 100).",
         "before": f"R2 on {_pct0(sh.get('fixed'))} of backtest sessions with the fixed 100-point gate",
         "after":  f"R2 on {_pct0(sh.get('adaptive'))} of backtest sessions."},
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
    return _clean(d)


# ── Rendering ──────────────────────────────────────────────────────────────────
# Per-route data date for the sitemap (freeze_site.py reads the header).
_DATA_DATE_KEYS = {"/": "latest_session", "/calibration": "session",
                   "/greeks": "surface_date", "/backtest": "run_date"}


# Chart.js is vendored (dashboard/static, copied into the site by freeze_site.py)
# so no third-party script sits on the critical path. The relative src works at
# every route because all routes are top-level. `sri` is the sha384 of the
# vendored file, which is byte-identical to the npm build on jsDelivr and unpkg;
# base.html falls back to `cdn` (with the same SRI) only if the local copy fails,
# and to the charts' data summaries if both fail.
CHARTJS = {
    "version": "4.4.3",
    "src": "static/chart.umd-4.4.3.js",
    "sri": "sha384-tgbB5AKnszdcfwcZtTfuhR3Ko1XZdlDfsLtkxiiAZiVkkXCkFmp+FQFh+V/UTo54",
    "cdn": "https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.js",
}


@app.context_processor
def _page_context() -> dict:
    route = request.path if request.path in ROUTES else "/"
    return {"canonical": canonical_url(route), "page_file": ROUTES[route],
            "chartjs": CHARTJS}


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
