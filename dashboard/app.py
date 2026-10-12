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


def pdv_forecast_series(pdv: Any, log_returns: pd.Series) -> pd.Series:
    """PDV linear forecast (annualised decimal) for every date of `log_returns`.

    The features are rebuilt from the supplied returns and fed to the pickled
    model's fitted OLS (its coefficients are unchanged; dates after the fit
    window are out of sample). Re-using the pickle's stored TRAINING features
    instead froze the published forecast at their last date (2025-12-30) while
    it was subtracted from the current VIX.
    """
    return pdv.linear_.predict(pdv_features(pdv, log_returns)).rename("pdv_vol")


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
        "horizon": horizon,
    }


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
              "pdv_age", "stress", "pdv_vix", "pdv_horizon", "pdv_coef", "pdv_intercept",
              "pdv_halflives", "pdv_garch_beta", "pdv_garch_persistence",
              "clf_feature_date", "clf_label", "clf_conf", "clf_agrees",
              "prob_r0", "prob_r1", "prob_r2", "clf_top_feature", "clf_top_importance",
              "latest_session"):
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
        tenor_days = {"^VIX9D": 9, "^VIX": 30, "^VIX3M": 93, "^VIX6M": 182}
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

    # ── PDV forecast vs VIX, with a like-for-like history ────────────────────
    pdv = None
    pdv_vol: Optional[pd.Series] = None
    try:
        with open(DATA / "pdv_model.pkl", "rb") as f:
            pdv = pickle.load(f)
        if spx_df is None or vix_daily is None:
            raise RuntimeError("needs the market data that failed to load")
        feats = pdv_features(pdv, spx_df["log_return"])
        pdv_vol = pdv.linear_.predict(feats).rename("pdv_vol")
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

        # History = the SAME quantity (VIX − this model's forecast) on prior dates.
        hist = spread.iloc[:-1]
        pct, n_hist = spread_percentile(hist, float(spread.iloc[-1]))
        d["pdv_hist_n"] = n_hist
        if n_hist >= PDV_MIN_HISTORY:
            d["pdv_spread_pct"] = round(pct, 1)
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
            d["stress"] = pdv_stress_forecast(pdv, spx_df["log_return"])
    except Exception as e:
        _record_error(d, "PDV COVID stress test", e)

    # ── Research-only classifier card ────────────────────────────────────────
    # Features come from build_features — the code path the classifier was
    # trained on — with the PDV forecast series computed above. (Building them
    # by hand fed it pdv_iv_spread = −VIX, because the PDV forecast was read
    # before it had been computed, and used different RV definitions.)
    try:
        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
            clf = pickle.load(f)
        if spx_df is None or vix_wide is None or pdv_vol is None:
            raise RuntimeError("needs the market data and PDV forecast that failed to load")
        from joint_vol_calibration.signals.regime_classifier import build_features

        feats = build_features(
            spx_df.reset_index()[["date", "close", "log_return"]],
            vix_wide.reset_index().rename(columns={"index": "date"}),
            pdv_vol_series=pdv_vol,
        )
        cols = list(getattr(clf, "feature_names_", None) or feats.columns)
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
def _greeks_provenance(path: Path) -> dict:
    """Provenance stored with the surface (parquet metadata 'provenance' JSON)."""
    prov: dict = {}
    try:
        import pyarrow.parquet as pq
        meta = pq.read_schema(path).metadata or {}
        raw = meta.get(b"provenance")
        if raw:
            prov = json.loads(raw.decode())
    except Exception as exc:
        _logger.info("greeks provenance unreadable: %s", exc)
    return prov


def load_page3() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "heatmap_rows": [], "heatmap_cols": [],
                         "unstable_nodes": [], "qv_labels": [], "qv_values": [],
                         "vanna_labels": [], "vanna_values": [], "scatter_lm": [],
                         "scatter_z": [], "scatter_T": [], "unstable_maturities": []}
    for k in ("n_unstable", "n_cells", "n_grid", "n_missing", "n_maturities", "as_of", "surface_date",
              "vomma_max", "vomma_max_at", "vomma_min", "vomma_min_at", "vomma_mean",
              "provenance", "source_matches_headline", "greeks_age"):
        d[k] = None

    try:
        path = DATA / "greeks" / "greeks_surface.parquet"
        gdf = pd.read_parquet(path).copy()
        # Recompute the flag from the raw z-score so the count does not depend
        # on whichever threshold was used when the parquet was written.
        gdf["is_unstable"] = gdf["vomma_zscore"].abs() > 2.0
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
                    cells.append(None)
                    continue
                z = float(r.vomma_zscore)
                cells.append({
                    "vomma": round(float(r.vomma), 1), "z": round(z, 2),
                    "iv": round(float(r.iv) * 100, 1), "K": round(float(r.K), 0),
                    "lm": lm, "unstable": bool(r.is_unstable), "bg": _vomma_color(z),
                    "fg": "#fff" if bool(r.is_unstable) or abs(z) > 1 else "#aaa",
                })
            d["heatmap_rows"].append({"T": T, "label": f"{T}d", "cells": cells})
        d["heatmap_cols"] = [f"{c:+.3f}" for c in cols]

        unstable = gdf[gdf["is_unstable"]]
        d["unstable_nodes"] = sorted((
            {"T_days": int(u.T_days), "lm": round(float(u.log_moneyness), 4),
             "K": round(float(u.K), 0), "vomma": round(float(u.vomma), 1),
             "z": round(float(u.vomma_zscore), 2), "iv": round(float(u.iv) * 100, 1)}
            for u in unstable.itertuples()), key=lambda x: x["z"], reverse=True)
        d["n_unstable"] = len(d["unstable_nodes"])
        d["unstable_maturities"] = sorted({n["T_days"] for n in d["unstable_nodes"]})

        imax, imin = gdf["vomma"].idxmax(), gdf["vomma"].idxmin()
        d["vomma_max"] = round(float(gdf.loc[imax, "vomma"]), 1)
        d["vomma_max_at"] = {"T_days": int(gdf.loc[imax, "T_days"]),
                             "lm": round(float(gdf.loc[imax, "log_moneyness"]), 3)}
        d["vomma_min"] = round(float(gdf.loc[imin, "vomma"]), 1)
        d["vomma_min_at"] = {"T_days": int(gdf.loc[imin, "T_days"]),
                             "lm": round(float(gdf.loc[imin, "log_moneyness"]), 3)}
        d["vomma_mean"] = round(float(gdf["vomma"].mean()), 1)

        # QV convexity = Var[∫v dt] per maturity: one number per maturity (it
        # does not depend on strike), so a single series is plotted.
        qv = gdf.groupby("T_days")["qv_convexity"].agg(["mean", "min", "max"])
        if ((qv["max"] - qv["min"]).abs() > 1e-12 * qv["max"].abs().clip(lower=1)).any():
            _logger.info("qv_convexity varies within a maturity; plotting the mean")
        d["qv_labels"] = [f"{int(t)}d" for t in qv.index]
        d["qv_values"] = (qv["mean"] * 1e4).round(4).tolist()

        vanna = gdf.groupby("T_days")["vanna"].mean()
        d["vanna_labels"] = [f"{int(t)}d" for t in vanna.index]
        d["vanna_values"] = vanna.round(4).tolist()

        d["scatter_lm"] = gdf["log_moneyness"].round(4).tolist()
        d["scatter_z"] = gdf["vomma_zscore"].round(3).tolist()
        d["scatter_T"] = gdf["T_days"].astype(int).tolist()

        # Provenance + staleness. The shipped surface records no source fit;
        # when it does (parquet metadata), compare with the headline calibration.
        prov = _greeks_provenance(path)
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


def load_page4() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "metrics_rows": [], "signal_metrics": [],
                         "signal_pnl": [], "signal_labels": [],
                         "research_s1": {}, "research_s1c": {}, "research_s4": {},
                         "eq_dates": [], "eq_nav": [], "eq_nav_s1c": [], "eq_nav_s3": [],
                         "eq_nav_s4": [], "eq_nav_s1": [], "ann": [], "ann_years": [],
                         "ann_rets": [], "wf": [], "wf_periods": [], "wf_sharpes": [],
                         "crisis": [], "sleeves": {}}
    for k in ("final_nav", "total_pnl", "start_date", "end_date",
              "wf_neg", "wf_total", "kpi", "sharpe_convention", "run_date"):
        d[k] = None

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

        from joint_vol_calibration.backtest.backtest_engine import compute_crisis_performance
        d["crisis"] = [{"name": k, "total_return": v.get("total_return"),
                        "n_days": v.get("n_days")}
                       for k, v in compute_crisis_performance(eq).items()]
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

        # Portfolio trade statistics from the engine's per-sleeve trade-log
        # stats. (The engine's top-level n_trades also counts the reference
        # sleeves; counting position changes double-counted round trips.)
        n_rt = sum(int(em.get(f"{s}_n_trades") or 0) for s in PORTFOLIO_SLEEVES)
        wins = sum(float(em.get(f"{s}_win_rate") or 0.0) * int(em.get(f"{s}_n_trades") or 0)
                   for s in PORTFOLIO_SLEEVES)
        port_pnl = d["total_pnl"]
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
            "n_round_trips": n_rt,
            "win_rate": (wins / n_rt) if n_rt else None,
            "avg_pnl": (port_pnl / n_rt) if (n_rt and port_pnl is not None) else None,
        }
        d["kpi"] = kpi

        def _pct(x):
            return "—" if x is None else f"{x * 100:.2f}%"

        def _f(x, n=3):
            return "—" if x is None else f"{x:.{n}f}"

        def _dol(x):
            return "—" if x is None else (f"−${abs(x):,.0f}" if x < 0 else f"${x:,.0f}")

        d["metrics_rows"] = [
            {"label": "Cumulative Return", "value": _pct(kpi["cum_ret"]), "positive": (kpi["cum_ret"] or 0) > 0},
            {"label": "Ann. Return", "value": _pct(kpi["ann_ret"]), "positive": (kpi["ann_ret"] or 0) > 0},
            {"label": "Sharpe", "value": _f(kpi["sharpe"]), "positive": (kpi["sharpe"] or 0) > 0},
            {"label": "Sortino", "value": _f(kpi["sortino"]), "positive": (kpi["sortino"] or 0) > 0},
            {"label": "Max Drawdown", "value": _pct(kpi["mdd"]), "positive": False},
            {"label": "DD Duration", "value": "—" if kpi["dd_dur"] is None else f"{int(kpi['dd_dur'])} sessions", "positive": False},
            {"label": "Calmar", "value": _f(kpi["calmar"]), "positive": (kpi["calmar"] or 0) > 0},
            {"label": "Win Rate (per round trip)", "value": _pct(kpi["win_rate"]), "positive": (kpi["win_rate"] or 0) > 0.5},
            {"label": "Avg P&L / Round Trip", "value": _dol(kpi["avg_pnl"]), "positive": (kpi["avg_pnl"] or 0) > 0},
            {"label": "Best Day", "value": _pct(kpi["best_day"]), "positive": True},
            {"label": "Worst Day", "value": _pct(kpi["worst_day"]), "positive": False},
            {"label": "Round-Trip Trades", "value": str(n_rt), "positive": True},
            {"label": "Return Days", "value": "—" if kpi["n_days"] is None else str(int(kpi["n_days"])), "positive": True},
        ]

        sleeves = {}
        for sig in ("s1c", "s3", "s4", "s1", "s2"):
            tp = _num(em.get(f"{sig}_total_pnl"))
            sleeves[sig] = {
                "total_pnl": tp, "sharpe": _num(em.get(f"{sig}_sharpe")),
                "ann_ret": _num(em.get(f"{sig}_ann_return")),
                "win_rate": _num(em.get(f"{sig}_win_rate")),
                "n_trades": em.get(f"{sig}_n_trades"),
                "summary": _trade_summary(trades, sig),
            }
        d["sleeves"] = sleeves

        def _card(sig: str) -> dict:
            s = sleeves[sig]
            return {"total_pnl": _dol(s["total_pnl"]), "sharpe": _f(s["sharpe"]),
                    "win_rate": _pct(s["win_rate"]),
                    "n_trades": "—" if s["n_trades"] is None else str(int(s["n_trades"])),
                    "green": (s["total_pnl"] or 0) > 0}

        d["research_s1"], d["research_s1c"], d["research_s4"] = _card("s1"), _card("s1c"), _card("s4")

        d["signal_labels"] = ["S1C  Contrarian PDV", "S3  Dispersion", "S4  VRP"]
        d["signal_pnl"] = [None if sleeves[s]["total_pnl"] is None else round(sleeves[s]["total_pnl"] / 1000, 1)
                           for s in PORTFOLIO_SLEEVES]

        for sig, lbl, is_ref in (("s1c", "S1C Contrarian", False), ("s3", "S3 Dispersion", False),
                                 ("s4", "S4 VRP", False), ("s1", "S1 IVR (ref)", True),
                                 ("s2", "S2 VIX TS (ref)", True)):
            s = sleeves[sig]
            if s["total_pnl"] is None:
                continue
            d["signal_metrics"].append({
                "label": lbl, "sharpe": _f(s["sharpe"]), "ann_ret": _pct(s["ann_ret"]),
                "win_rate": _pct(s["win_rate"]),
                "n_trades": "—" if s["n_trades"] is None else str(int(s["n_trades"])),
                "total_pnl": _dol(s["total_pnl"]), "green": s["total_pnl"] > 0,
                "is_research": is_ref,
            })
    except Exception as e:
        _record_error(d, "backtest engine results (full_results.pkl)", e)

    # ── Improvement Summary (Steps 1–8) — a historical change log ─────────────
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
