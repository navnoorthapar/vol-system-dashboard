"""
dashboard/app.py — Joint SPX/VIX Volatility System — display layer.

Renders the dashboard pages from the artifacts under data_store/. It is a
pure READ path: rendering a page never downloads data, never writes a file and
never retrains or re-labels anything. The daily data refresh, the regime-label
extension and the Heston recalibration belong to scripts/cloud_refresh.py; the
static site is produced by dashboard/freeze_site.py.

Pages (ROUTES): Overview, Surface, Calibration, Risk, Research, Corrections and
Model Risk. Every chart is a JSON spec rendered by static/desk.js (Apache
ECharts, vendored with SRI) with a server-rendered data table behind it; all
page JavaScript lives in static files, so the pages carry a strict
Content-Security-Policy (script-src 'self').

(An earlier version started "startup" and "every 4 hours" refresh threads from
``__main__``. They downloaded market data into the tracked SQLite DB and then
overwrote the tracked regime_labels.parquet with the DEMOTED classifier's
predictions — the exact defect C21 removed from cloud_refresh — whenever the
local DB was more than three days old, i.e. on every fresh clone.)

Every loader reports failures in ``d["errors"]``; the templates render them as a
visible error banner instead of zeros, and freeze_site.py refuses to publish a
page that carries one. Artifacts that a later chain of work produces (the
quote-grade surface, inference, scenario grids; dashboard/ARTIFACTS.md) render
an explicit "not yet generated" panel while absent, and a loader error if they
exist but break their contract.

Run a local preview (read-only, bound to 127.0.0.1):
    python dashboard/app.py
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import itertools
import json
import logging
import math
import os
import pickle
import re
import sqlite3
import subprocess
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from flask import Flask, make_response, render_template, request
from flask import g as request_globals
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
DASH = Path(__file__).resolve().parent
DATA = ROOT / "data_store"
sys.path.insert(0, str(ROOT))
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_artifacts as artifacts  # noqa: E402  (dashboard/ modules, after the path setup)
import desk_charts as charts  # noqa: E402
import desk_market  # noqa: E402
import desk_monitor  # noqa: E402
import desk_surface  # noqa: E402
from desk_format import (  # noqa: E402, F401  (re-exported: tests and templates use them)
    _HEAT_NEG,
    _HEAT_NEUTRAL,
    _HEAT_POS,
    BERLIN_11,
    CIVIDIS_11,
    MINUS,
    SPX_MULTIPLIER,
    _fmt,
    _num,
    _pct_prose,
    _relative_luminance,
    _text_colour_on,
    _usd,
    _usd_k,
    _vomma_color,
    cash_greeks,
    cell,
    contrast_ratio,
    div_colour,
    ordinal,
    plural,
    pnl,
    prose,
    round_sig,
    seq_colour,
    signed,
    sparkline,
    tone,
    usd_desk,
)

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)

# ── Site map ───────────────────────────────────────────────────────────────────
SITE_URL = "https://navnoorbawa.me/"

# route -> frozen file name. freeze_site.py imports this map so the frozen file
# names, the canonical URLs and the sitemap can never drift apart.
ROUTES: dict[str, str] = {
    "/":            "index.html",
    "/surface":     "surface.html",
    "/calibration": "calibration.html",
    "/risk":        "risk.html",
    "/research":    "research.html",
    "/corrections": "corrections.html",
    "/model-risk":  "model-risk.html",
}

# The navigation: (route, label, g-shortcut key, one-line description).
NAV: list[tuple[str, str, str, str]] = [
    ("/", "Overview", "o", "What the system is, its live state, three key results and data freshness."),
    ("/surface", "Surface", "s",
     "The SPX implied-vol surface: market quotes, model overlays and residuals in half-spreads."),
    ("/calibration", "Calibration", "c",
     "The headline joint SPX/VIX Heston fit: parameters, bounds, out-of-sample error, identifiability."),
    ("/risk", "Risk", "r",
     "Greeks of the headline surface, scenario grid and the daily P&L explain of a hedged straddle."),
    ("/research", "Research", "e",
     "The 2018–2025 delta-hedged straddle backtest (mark-to-model), its inference and β sensitivity."),
    ("/corrections", "Corrections", "x",
     "Correction rounds C16–C23 as postmortems, each tied to commits and regression tests."),
    ("/model-risk", "Model Risk", "m", "Model inventory and monitoring checks with explicit thresholds."),
]

# Old page URLs (shared on social links) keep working: freeze_site.py writes a
# small redirect page at each old file name.
LEGACY_REDIRECTS: dict[str, str] = {"greeks.html": "risk.html", "backtest.html": "research.html"}


def canonical_url(route: str) -> str:
    """Absolute public URL of a route's frozen page (the home page is the bare root)."""
    fname = ROUTES.get(route, "index.html")
    return SITE_URL if fname == "index.html" else SITE_URL + fname


# ── Staleness thresholds (NYSE sessions) ───────────────────────────────────────
STALE_WARN_SESSIONS = 2     # amber banner from this age on
STALE_ERROR_SESSIONS = 5    # red banner from this age on

# Regime presentation (index = rule label 0/1/2)
_REGIME_NAMES = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
_REGIME_COLORS = ["#56b4e9", "#009e73", "#e69f00"]   # --cat-0 / --cat-2 / --cat-1 (categorical, not status)
_REGIME_LABELS = ["R0", "R1", "R2"]

# The backtest portfolio is the equal-weight mean of these sleeves
# (BacktestEngine.run: pnl_total = (s1c + s3 + s4) / 3). load_page4 VERIFIES
# this against the stored equity curve instead of trusting the constant.
PORTFOLIO_SLEEVES = ("s1c", "s3", "s4")
# S3 is priced as the index leg of a dispersion trade: a SHORT SPX straddle
# (backtest_engine.S3_LEG_DIRECTION = -1); no single-stock option data, so the
# single-name leg is not modelled. "S3 dispersion" overclaimed what is traded.
S3_NAME = "S3 · short SPX straddle"                     # tables
S3_SHORT = "S3 index-leg short"                         # chart legends and categories
S3_FULL = "S3 · short SPX straddle (index leg of a dispersion trade; no single-name leg)"


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


def session_age(date: str | None, ref: str | None) -> int | None:
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


def stale_level(age: int | None) -> str:
    """'ok' / 'warn' / 'error' for an age in sessions ('unknown' if None)."""
    if age is None:
        return "unknown"
    if age >= STALE_ERROR_SESSIONS:
        return "error"
    if age >= STALE_WARN_SESSIONS:
        return "warn"
    return "ok"


def _staleness(component: str, label: str, date: str | None, ref: str | None) -> dict:
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


@contextlib.contextmanager
def _loader_boundary(d: dict, component: str) -> Iterator[None]:
    """Run one loader section of a page: any exception it raises is recorded on
    the page by _record_error (a red banner and X-Dashboard-Loader-Errors, which
    freeze_site.py refuses to publish) and the page goes on to its next section."""
    try:
        yield
    except Exception as e:  # noqa: BLE001 (the loader boundary: recorded, rendered, blocks the publish)
        _record_error(d, component, e)


def _load_pickle(path: str | Path) -> Any:
    """Unpickle one of the repository's own artifacts (calibration results, the
    PDV and classifier models, the backtest state), read-only."""
    with open(path, "rb") as f:
        return pickle.load(f)  # noqa: S301 (trusted input: pickles this codebase wrote and committed)


def _date_str(x: Any) -> str | None:
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


# Number formatting, contrast and colour scales live in desk_format.py (imported
# above); these filters are what the templates use.
app.jinja_env.filters["fmt"] = _fmt
app.jinja_env.filters["ordinal"] = ordinal
app.jinja_env.filters["usd_k"] = _usd_k
app.jinja_env.filters["usd"] = _usd
app.jinja_env.filters["usd_desk"] = usd_desk
app.jinja_env.filters["pct"] = _pct_prose
app.jinja_env.filters["signed"] = signed
app.jinja_env.filters["pnl"] = pnl
app.jinja_env.filters["tone"] = tone
app.jinja_env.filters["plural"] = plural
app.jinja_env.filters["prose"] = prose


def with_unit(text: Any, unit: str) -> str:
    """Append a unit to a formatted number, but never to the missing-value dash
    ("—", not "—%")."""
    text = "" if text is None else str(text)
    return text if text in ("", "—") else f"{text}{unit}"


app.jinja_env.filters["u"] = with_unit


_GREEK = re.compile(r"([\u0370-\u03ff\u1f00-\u1fff]+)")


def sym(text: Any) -> Any:
    """Wrap Greek letters in <span class="sym"> so an upper-cased heading keeps them.

    CSS text-transform: uppercase turns β into Β (read as a B) and σ into Σ;
    the .sym class switches the transform off for those characters only.
    """
    from markupsafe import Markup, escape

    if text is None:
        return ""
    escaped = str(escape(text))
    # the only markup is the fixed <span> wrapper: the text itself is escaped above
    return Markup(_GREEK.sub(lambda m: f'<span class="sym">{m.group(1)}</span>', escaped))  # noqa: S704


app.jinja_env.filters["sym"] = sym


def path_breaks(text: Any) -> Any:
    """A repository path or test ID that wraps only between its names: a <wbr>
    after each '/', '_' and '::' and between the words of a CamelCase name
    ('Test<wbr>Generate<wbr>Signal1<wbr>Contrarian'), and a name with a hyphen (a
    date) in a no-wrap span, since the browser also breaks after '-'. With
    `.tbl code.path { overflow-wrap: break-word }` a narrow column breaks
    'data_store/vol_system.db' after 'vol_', never as 'vol_syste' / 'm.db'
    (word-break: break-all did) or '2026-' / '10-16'; at 320 px a test class
    broke as 'TestGenerateSignal1Con' / 'trarian'."""
    from markupsafe import Markup, escape

    if text is None:
        return ""
    parts = [p for p in re.split(r"(?<=[/_])|(?<=::)|(?<=[a-z0-9])(?=[A-Z])", str(escape(text))) if p]
    # the only markup is <wbr> and the span: the text itself is escaped first
    return Markup("<wbr>".join(f'<span class="nowrap">{p}</span>' if "-" in p else p  # noqa: S704
                               for p in parts))


app.jinja_env.filters["path"] = path_breaks


def date_nowrap(text: Any) -> Any:
    """Prose with every ISO date in a no-wrap span: the browser breaks after the
    hyphen, so '2025-01-17' split as '2025-' / '01-17' at a line end."""
    from markupsafe import Markup, escape

    if text is None:
        return ""
    return Markup(re.sub(r"\b(\d{4}-\d{2}-\d{2})\b", r'<span class="nowrap">\1</span>',  # noqa: S704
                         str(escape(text))))


app.jinja_env.filters["dates"] = date_nowrap


def monitored_for(element_id: str) -> list[dict]:
    """The Model Risk checks whose evidence is the element ``element_id`` of the
    page being rendered (a Jinja global, so the imported panel macro sees it)."""
    return list((getattr(request_globals, "monitored", None) or {}).get(element_id) or [])


app.jinja_env.globals["monitored_for"] = monitored_for


# ── Evidence graph: m.cite() calls and the 'Cited on' footers ────────────────
# A page that restates another page's figure cites it with the _macros.html
# cite(label, href) macro. The registry is read from the templates themselves
# (every m.cite call whose href is a literal 'page.html#id'), so a footer lists
# exactly the pages whose cite() calls point at its panel, whatever data a
# build has.
_CITE_CALL = re.compile(r"m\.cite\((?:[^()]|\([^()]*\))*?,\s*'([\w.-]+\.html)#([\w-]+)'\s*\)")
_PANEL_CALL = re.compile(r"\{%\s*call m\.panel\('([\w-]+)'")


def cite_registry(template_dir: Path | None = None) -> dict[str, list[dict]]:
    """{'page.html#id': [{'page', 'panel', 'label'}, ...]}: who cites each element.
    'panel' is the citing panel's id when the cite() call sits inside one."""
    tdir = Path(template_dir) if template_dir is not None else DASH / "templates"
    labels = {ROUTES[r]: label for r, label, *_ in NAV}
    out: dict[str, list[dict]] = {}
    for f in sorted(tdir.glob("*.html")):
        if f.name not in labels:
            continue
        src = f.read_text()
        panels = [(m_.start(), m_.group(1)) for m_ in _PANEL_CALL.finditer(src)]
        for c in _CITE_CALL.finditer(src):
            page, anchor = c.group(1), c.group(2)
            if page == f.name:
                continue                  # a same-page reference is not cross-page evidence
            panel = None
            opened = [(pos, pid) for pos, pid in panels if pos < c.start()]
            if opened:
                pos, pid = opened[-1]
                seg = src[pos:c.start()]
                if seg.count("{% call ") > seg.count("{% endcall %}"):
                    panel = pid
            entry = {"page": f.name, "panel": panel, "label": labels[f.name]}
            lst = out.setdefault(f"{page}#{anchor}", [])
            if not any(e["page"] == entry["page"] and e["panel"] == entry["panel"] for e in lst):
                lst.append(entry)
    return out


_CITES = cite_registry()


def cited_on(element_id: str) -> list[dict]:
    """The other pages that cite this element of the page being rendered (its footer)."""
    page = ROUTES.get(request.path, "")
    out: list[dict] = []
    for e in _CITES.get(f"{page}#{element_id}") or []:
        if all(o["page"] != e["page"] for o in out):        # one link per citing page
            out.append(e)
    return out


app.jinja_env.globals["cited_on"] = cited_on


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


VX_TABLE = "data_store/vol_system.db"     # table vix_futures_daily


def vx_last_date() -> str | None:
    """The newest session the market DB stores VX futures for (None: no futures table)."""
    with _db() as con:
        try:
            row = con.execute("SELECT MAX(date) FROM vix_futures_daily").fetchone()
        except sqlite3.OperationalError:          # a market DB without the futures table
            return None
    return str(row[0])[:10] if row and row[0] else None


def vx_prev_date(before: str) -> str | None:
    """The VX futures table's last session before ``before`` (None if none)."""
    with _db() as con:
        try:
            row = con.execute("SELECT MAX(date) FROM vix_futures_daily WHERE date < ?", (before,)).fetchone()
        except sqlite3.OperationalError:
            return None
    return str(row[0])[:10] if row and row[0] else None


def vx_session_rows(session: str) -> list[dict]:
    """The VX futures rows of exactly ``session``: the query of
    database.get_vix_futures_curve (contracts live on the session, monthly only,
    price = COALESCE(settle, close)), the rows the calibrator scores out of sample.
    Never a nearby session's rows: an empty list when the session has none."""
    with _db() as con:
        try:
            df = pd.read_sql(
                "SELECT expiry_date, expiry_month, is_weekly, COALESCE(settle, close) AS price, "
                "days_to_expiry FROM vix_futures_daily WHERE date = ? AND expiry_date > ? AND is_weekly = 0 "
                "ORDER BY expiry_date", con, params=(session, session))
        except (sqlite3.OperationalError, pd.errors.DatabaseError):
            return []
    return cast("list[dict]", df.to_dict("records"))


def _latest_market_session() -> str | None:
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
def _calibration_session(cal: dict, path: str | Path) -> str | None:
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


def calibration_is_legacy(cal: dict) -> bool | None:
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
            cal = _load_pickle(path)
            legacy = calibration_is_legacy(cal)
            ok, reason = calibration_gate(cal)
        except Exception as exc:
            _logger.warning("calibration %s unreadable: %s", Path(path).name, exc)
            continue
        rows.append({"path": path, "session": _calibration_session(cal, path),
                     "run_date": str(cal.get("run_date") or ""),
                     "legacy": legacy, "ok": ok, "reason": reason})
    return rows


def latest_good_calibration_path(cal_dir: str | Path | None = None) -> str | None:
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


def calibration_stale_cause(session: str | None, data_dir: Path | None = None,
                            rows: list[dict] | None = None) -> str | None:
    """Why the headline fit is older than the newest session, from what the store holds.

    The stale banner said "No newer fit passed the quality gate" whether or not
    a newer fit existed. The cause is read from the store: a stored fit after
    the headline that failed the gate (with the gate's recorded reason); a
    stored SPX chain after it with no fit; or neither. The daily job stores a
    session's chain and fit only when the fit passes the gate, so with neither
    stored the store cannot say whether a later session was fitted and
    rejected or not fitted at all, and the text says so.
    """
    if not session:
        return None
    data = Path(data_dir) if data_dir is not None else DATA
    chains = sorted(m.group(1) for f in (data / "options").glob("spx_*.parquet")
                    if (m := re.fullmatch(r"spx_(\d{4}-\d{2}-\d{2})\.parquet", f.name)))
    rows = rows if rows is not None else scan_calibrations(data / "calibrations")
    newer = sorted((r for r in rows if r["session"] and r["session"] > session),
                   key=lambda r: r["session"])
    if newer:
        last = newer[-1]
        why = ("is a pre-C22 (legacy) result, never ranked" if last["legacy"]
               else f"failed the quality gate: {last['reason']}")
        more = f" ({len(newer)} stored fits after {session}, none accepted)" if len(newer) > 1 else ""
        return f"The {last['session']} fit is stored but {why}{more}; the {session} fit is shown."
    later = [c for c in chains if c > session]
    if later:
        return f"An SPX closing chain for {later[-1]} is stored but no fit of it; the {session} fit is shown."
    return (f"No SPX closing chain or fit after {session} is stored. The daily job stores both only for a "
            "fit that passes the quality gate, so whether a later session was fitted and rejected, or not "
            "fitted, is in the job's run log, not in the store.")


def load_headline_calibration() -> tuple[str, dict]:
    """(path, result) of the headline calibration; raises when none is acceptable."""
    path = latest_good_calibration_path()
    if path is None:
        raise FileNotFoundError(
            "no joint_cal_*.pkl in data_store/calibrations passes the quality gate"
        )
    return path, _load_pickle(path)


# ── Heston model smile (calibration page) ──────────────────────────────────────
SMILE_MATURITIES_DAYS = (14, 30, 60, 91, 180, 365)
SMILE_LOG_MONEYNESS = np.round(np.linspace(-0.30, 0.25, 23), 4)
# OTM forward prices below this (as a fraction of the forward) are dropped:
# inverting a price that small turns pricer round-off into a fake implied vol
# (the old Greeks-surface smile showed a 30-day "IV" of 5.0% that way).
SMILE_PRICE_FLOOR = 1e-6


def heston_model_smile(params: dict,
                       maturities_days: Iterable[float] = SMILE_MATURITIES_DAYS,
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


SKEW_FD_STEP = 0.01     # ln(K/F) step of the Heston ATM-skew differences (and half of it)


def heston_atm_skew(params: dict, maturities_days: Sequence[float],
                    h: float = SKEW_FD_STEP) -> list[float | None]:
    """∂σ/∂ln(K/F) at the forward of the Heston smile of ``params``, per maturity,
    in vol (decimal) per unit ln-moneyness.

    Central differences of heston_model_smile at steps h and h/2, combined by
    Richardson extrapolation, (4·D(h/2) − D(h))/3, which cancels the h² error
    term: a plain ±0.01 difference is 2% off at 13 days (the short-dated smile
    is strongly curved at the money). None where a node did not invert."""
    hh = h / 2.0
    ks = (-h, -hh, hh, h)
    out: list[float | None] = []
    for r in heston_model_smile(params, maturities_days=maturities_days, log_moneyness=ks):
        iv = dict(zip(r["lm"], r["iv"]))
        got = [iv.get(round(k, 4)) for k in ks]
        v = [float(x) for x in got if x is not None]
        if len(v) != len(ks):
            out.append(None)
            continue
        d_h = (v[3] - v[0]) / (2.0 * h)
        d_hh = (v[2] - v[1]) / (2.0 * hh)
        out.append((4.0 * d_hh - d_h) / 3.0)
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


def spread_percentile(history: pd.Series, current: float) -> tuple[float | None, int]:
    """(percent of the history strictly below `current`, number of valid points).

    NaN history rows are dropped before ranking. Counting them in the
    denominator (NaN < x is False) pinned the published percentile near zero:
    4,125 of 4,189 rows were NaN and the card read '1.1th percentile, vol
    cheap' for a spread above its own mean.
    """
    hist = pd.to_numeric(history, errors="coerce").dropna()
    if len(hist) == 0 or _num(current) is None:
        return None, len(hist)
    return float((hist < current).mean() * 100.0), len(hist)


PDV_MIN_HISTORY = 252   # fewer valid history points -> no percentile is shown


def percentile_rank(pct: Any) -> dict | None:
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
                        stress_date: str = "2020-03-16") -> dict | None:
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
        "n_window": len(window),
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
                           exclude_year: int = 2020) -> dict | None:
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
            "n": len(both), "start": str(both.index[0].date()),
            "end": str(both.index[-1].date())}


_WF_CACHE: dict = {}


# ── PAGE 1 DATA: Live Market Overview ─────────────────────────────────────────
def _rule_labels(d: dict) -> pd.Series | None:
    """The stored rule labels; sets the headline label keys of ``d``."""
    reg_series: pd.Series | None = None
    with _loader_boundary(d, "rule regime labels (regime_labels.parquet)"):
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
    return reg_series


def load_classifier_card() -> dict[str, Any]:
    """The research-only ML regime classifier and the rule-label distribution.

    Rendered on the Research page only: the classifier is demoted (it loses to
    "predict yesterday's regime") and feeds nothing, so it is not on the
    Overview. A failure here is the Research page's loader error.
    """
    d: dict[str, Any] = {"errors": []}
    for k in ("regime", "regime_date", "reg_total", "reg0_n", "reg1_n", "reg2_n", "reg0_pct", "reg1_pct",
              "reg2_pct", "reg_span", "clf_feature_date", "clf_label", "clf_conf", "clf_agrees",
              "prob_r0", "prob_r1", "prob_r2", "clf_top_feature", "clf_top_importance",
              "clf_oos_acc", "clf_persistence_acc", "clf_n_test", "clf_test_window",
              "clf_feature_source", "clf_n_features"):
        d[k] = None
    d.update(regime_name="UNAVAILABLE", regime_color="#9aa3b0", regime_label="—", clf_color="#9aa3b0",
             clf_available=False)
    spx_df: pd.DataFrame | None = None
    vix_wide: pd.DataFrame | None = None
    with _loader_boundary(d, "market data (vol_system.db)"):
        spx_df = _read_spx()
        vix_wide = _read_vix_wide()
    reg_series = _rule_labels(d)
    if reg_series is not None:
        tot = len(reg_series)
        d["reg_total"] = tot
        for k in (0, 1, 2):
            n = int((reg_series == k).sum())
            d[f"reg{k}_n"] = n
            d[f"reg{k}_pct"] = round(100.0 * n / tot, 1)
        d["reg_span"] = f"{reg_series.index[0].year} – {reg_series.index[-1].year}"
    # ── Research-only classifier card ────────────────────────────────────────
    # Features come from build_features — the code path the classifier was
    # trained on — with pdv_iv_spread built from the SAME source the training
    # set used (regime_classifier.classifier_features reads it from the model's
    # metadata). Building features by hand fed it pdv_iv_spread = −VIX; feeding
    # the PDV forecast to a classifier trained on the rv₂₀ proxy was a
    # training/serving mismatch.
    with _loader_boundary(d, "research classifier (regime_classifier.pkl)"):
        clf = _load_pickle(DATA / "signals" / "regime_classifier.pkl")
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

    return _clean(d)


def load_page1() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": []}

    # Defaults: None renders as '—'. Nothing here may default to a plausible number.
    for k in ("spx", "spx_chg", "spx_up", "spx_date", "vix", "vix_date",
              "vvix", "vvix_date", "ts_date", "regime", "regime_date",
              "calib_date", "pdv_forecast", "pdv_spread",
              "pdv_date", "pdv_spread_mean", "pdv_spread_pct", "pdv_hist_n",
              "pdv_hist_start", "pdv_hist_end", "pdv_sigma1", "pdv_sigma2",
              "pdv_age", "pdv_rank", "stress", "pdv_vix", "pdv_horizon", "pdv_coef", "pdv_intercept",
              "pdv_halflives", "pdv_garch_beta", "pdv_garch_persistence",
              "pdv_skill", "rule", "latest_session"):
        d[k] = None
    d.update(regime_name="UNAVAILABLE", regime_color="#9aa3b0", regime_label="—")
    d.update(ts_tenors=[], ts_values=[], ts_regime="N/A",
             pdv_sparkline=[], pdv_sparkline_dates=[], pdv_stale=False, stale=[])

    # ── Market snapshot (DB) ─────────────────────────────────────────────────
    spx_df: pd.DataFrame | None = None
    vix_daily: pd.Series | None = None
    vix_wide: pd.DataFrame | None = None
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
    except Exception as e:
        _record_error(d, "market data (vol_system.db)", e)
        spx_df = vix_daily = vix_wide = None

    # ── Rule regime (regime_labels.parquet only) ─────────────────────────────
    # The headline is the deterministic RULE label. The research classifier is
    # not on this page at all (Research page, load_classifier_card).
    reg_series = _rule_labels(d)

    # ── The rule's own inputs on the newest session the DB covers ────────────
    # build_rule_label_frame pairs each session's rule label with the
    # observables that produced it (rv₂₀, VIX, VVIX and the adaptive gate),
    # so the headline label can be checked against its inputs on the page.
    with _loader_boundary(d, "rule inputs (build_rule_label_frame)"):
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
                # The stored label OF THE INPUTS' SESSION. The headline label can be
                # newer than the market DB (the bot labels the session before the DB
                # copy is refreshed); the card then showed the newest label beside
                # an older session's inputs as if they had produced it.
                if reg_series is not None and pd.Timestamp(t) in reg_series.index:
                    rule["stored"] = int(reg_series.loc[pd.Timestamp(t)])
                rule["headline_differs"] = d["regime_date"] is not None and d["regime_date"] != rule["date"]
                d["rule"] = rule

    # ── PDV forecast vs VIX, with a like-for-like history ────────────────────
    pdv = None
    pdv_vol: pd.Series | None = None
    try:
        pdv = _load_pickle(DATA / "pdv_model.pkl")
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
    with _loader_boundary(d, "PDV COVID stress test"):
        if pdv is not None and spx_df is not None:
            d["stress"] = pdv_stress_forecast(pdv, spx_df["close_return"])

    # Out-of-sample skill of the model family on the model's own target,
    # recomputed from the DB (the card hard-coded 0.31 / 0.23, the R² of the
    # superseded next-day |r| model and of the kernel, beside a 21-day model).
    with _loader_boundary(d, "PDV walk-forward skill"):
        if pdv is not None and spx_df is not None and pdv_vol is not None:
            d["pdv_skill"] = pdv_walk_forward_skill(pdv, spx_df["close_return"], pdv_vol)

    # ── Headline calibration (date only) ─────────────────────────────────────
    with _loader_boundary(d, "headline calibration"):
        path, cal = load_headline_calibration()
        d["calib_date"] = _calibration_session(cal, path)

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
    if d["stale"][2]["level"] in ("warn", "error"):
        with _loader_boundary(d, "calibration archive"):
            d["stale"][2]["cause"] = calibration_stale_cause(d["calib_date"])
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
def _optimizer_settings(cal: dict) -> str | None:
    """The optimiser settings the result records (methods.optimizer on
    model_version 2 results), or None when it records none."""
    opt = ((cal.get("methods") or {}).get("optimizer") or cal.get("optimizer")
           or cal.get("de_config") or {})
    if isinstance(opt, dict) and str(opt.get("method", "")).startswith("trf"):
        n = opt.get("n_sobol_starts") or opt.get("n_starts")
        text = "Trust-region least squares" + (f", {int(n)} Sobol starts" if n else "")
        ms = cal.get("multistart") or {}
        if ms.get("n_starts_at_best") is not None and ms.get("n_starts"):
            text += f" ({int(ms['n_starts_at_best'])} of {int(ms['n_starts'])} in the best basin)"
        return text
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
    d: dict[str, Any] = {"errors": [], "stale": [], "mat_smiles": [], "quintic": None,
                         "param_tiles": [], "history": [], "smiles": [], "charts": {},
                         "ident": None, "oos_rows": [], "vix_rows": []}
    for k in ("kappa", "theta", "theta_vol", "sigma", "rho", "v0", "v0_vol", "S",
              "r_pct", "q_pct", "session", "fit_time", "n_evals",
              "spx_rmse", "vix_fut_rmse", "vix_opt_rmse", "feller_lhs", "feller_rhs",
              "feller_margin", "feller_state", "effective_dof", "n_spx_options",
              "n_vix_tenors", "spot_date", "n_spx_expiries", "half_life_days",
              "optimizer", "legacy", "calib_file", "calib_age", "vix_index_rmse",
              "r_ref_pct", "r_ref_source", "r_ref_label", "r_ref_detail", "carry_mode",
              "feller_mode", "spx_oos_rmse",
              "spx_oos_wing_rmse", "n_vix_options", "archive"):
        d[k] = None
    d.update(pinned_params=[], active_constraints=[], feller_imposed=None,
             rho_at_boundary=False, spx_underidentified=None, rho_bound=None,
             feller_penalty_weight=None, strikes_per_expiry=None,
             n_spx_expiries_inferred=False, carry_recorded=False, anchors_by_expiry=[],
             anchor_rule=None, expiry_rule=None)

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
        # anchor expiries, and r_reference is the reference rate (cc): the ^IRX
        # bill for every fit before C23, the Treasury zero curve's 91-day rate
        # for a fit whose rate_reference record says so.
        if _num(cal.get("r")) is not None and _num(cal.get("q")) is not None:
            d["carry_recorded"] = True
            d["r_pct"] = round(float(cal["r"]) * 100, 2)
            d["q_pct"] = round(float(cal["q"]) * 100, 2)
        d["carry_mode"] = (cal.get("methods") or {}).get("carry")
        # the anchors the fit priced and the parity carry it priced them with, per
        # expiry, as the fit recorded them (anchors_by_expiry, C22+)
        d["anchors_by_expiry"] = [
            {"expiry": _date_str(a.get("expiry")), "T_days": _num(a.get("T_days")), "n": a.get("n"),
             "n_puts": a.get("n_puts"), "n_wing_puts": a.get("n_wing_puts"),
             "kf_min": _num(a.get("kf_min")), "kf_max": _num(a.get("kf_max")),
             "r_pct": None if _num(a.get("r")) is None else float(a["r"]) * 100.0,
             "q_pct": None if _num(a.get("q")) is None else float(a["q"]) * 100.0}
            for a in cal.get("anchors_by_expiry") or []]
        d["anchor_rule"] = (cal.get("methods") or {}).get("anchors")
        d["expiry_rule"] = (cal.get("methods") or {}).get("expiries")
        if _num(cal.get("r_reference")) is not None:
            d["r_ref_pct"] = round(float(cal["r_reference"]) * 100, 2)
            d["r_ref_source"] = cal.get("r_reference_source")
            kind = (cal.get("rate_reference") or {}).get("kind")
            if kind == "treasury_curve":
                d["r_ref_label"] = "Treasury r(91d)"
                d["r_ref_detail"] = ("Treasury zero curve, 91-day zero rate, continuously "
                                     "compounded; reference")
            else:
                d["r_ref_label"] = "^IRX"
                d["r_ref_detail"] = "13-week bill, continuously compounded; reference"

        losses = _cal_losses(cal)
        d["spx_rmse"] = (round(float(losses["spx_iv_rmse"]), 3) if _num(losses.get("spx_iv_rmse")) is not None
                         else None)
        # VIX leg: the Cboe SPOT indices (^VIX9D/^VIX/^VIX3M/^VIX6M) against
        # heston_vix_term_index. Pre-C22 results only have the old key name.
        vix_rmse = _num(losses.get("vix_index_rmse", losses.get("vix_futures_rmse")))
        d["vix_index_rmse"] = round(vix_rmse, 3) if vix_rmse is not None else None
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
            _CAL_STRIKES_PER_EXPIRY,
            active_constraints,
            classify_feller,
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
        # parameters not on a bound: a count of interior parameters, not of what
        # the data identify (κ is interior but rides the σ cap: see sigma_ridge)
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
        d["model_version"] = cal.get("model_version")
        d["fit_run_utc"] = cal.get("run_utc")
    except Exception as e:
        _record_error(d, "headline calibration", e)
        cal = None

    if cal is not None:
        try:
            calibration_extras(d, cal)
        except artifacts.ArtifactError as e:
            _record_error(d, "calibration diagnostics (identifiability / surface residuals)", e)
        except Exception as e:
            _record_error(d, "calibration diagnostics", e)

    # Archive census: how many stored fits are current-schema and gate-passing,
    # and how many are pre-C22 (labelled, never ranked).
    with _loader_boundary(d, "calibration archive"):
        rows = scan_calibrations()
        for item in d["stale"]:
            if item["component"] == "calibration" and item["level"] in ("warn", "error"):
                item["cause"] = calibration_stale_cause(d["session"], rows=rows)
        d["archive"] = {
            "n_total": len(rows),
            "n_legacy": sum(1 for r in rows if r["legacy"]),
            "n_current": sum(1 for r in rows if r["legacy"] is False),
            "n_current_ok": sum(1 for r in rows if r["legacy"] is False and r["ok"]),
        }

    # ── Quintic OU research fit vs the Heston fit of the SAME session ────────
    with _loader_boundary(d, "Quintic OU research calibration"):
        qpaths = sorted(glob.glob(str(DATA / "calibrations" / "quintic_cal_*.pkl")))
        if qpaths:
            qpath = qpaths[-1]
            qcal = _load_pickle(qpath)
            qp = qcal["params"]
            qp = qp if isinstance(qp, dict) else vars(qp)
            ql = qcal.get("leg_losses", {})
            qprov = qcal.get("provenance") or {}
            q_date = _date_str(qprov.get("spot_date")) or _calibration_session(qcal, qpath)
            mc = qprov.get("mc") or {}
            quad = qprov.get("vix_quadrature") or {}
            q = {
                "as_of": q_date, "file": Path(qpath).name,
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
                hcal = _load_pickle(hpath)
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


UNSTABLE_FAR_K = 0.09     # |ln(K/F)| from which a node counts as away from the money (a K_EDGES edge)


def unstable_vega_share(gdf: pd.DataFrame) -> dict | None:
    """The convexity-dominated nodes weighted by what they hold: the vega-weighted
    share of the is_unstable nodes, and where they sit. The KPI read '41 of 83',
    a node count that weighs a deep-wing node with little vega like an ATM one."""
    if gdf is None or not len(gdf) or "vega" not in gdf.columns:
        return None
    vega = gdf["vega"].astype(float)
    unst = gdf["is_unstable"].astype(bool)
    far = gdf["log_moneyness"].astype(float).abs() >= UNSTABLE_FAR_K - 1e-12
    tot = float(vega.sum())
    if not tot > 0:
        return None
    return {"share": float(vega[unst].sum()) / tot, "n": int(unst.sum()), "n_nodes": len(gdf),
            "n_far": int((unst & far).sum()), "far_k": UNSTABLE_FAR_K,
            "vega_far_mean": float(vega[far].mean()) if far.any() else None,
            "vega_near_mean": float(vega[~far].mean()) if (~far).any() else None}


def load_page3() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "heatmap_rows": [], "heatmap_cols": [],
                         "unstable_nodes": [], "qv_labels": [], "qv_values": [],
                         "vanna_labels": [], "vanna_values": [], "scatter_lm": [],
                         "scatter_z": [], "scatter_T": [], "unstable_maturities": [],
                         "unstable_by_maturity": {}}
    for k in ("n_unstable", "n_cells", "n_grid", "n_missing", "n_maturities", "as_of", "surface_date",
              "vomma_max", "vomma_max_at", "vomma_min", "vomma_min_at", "vomma_mean",
              "provenance", "source_matches_headline", "greeks_age", "unstable_rule", "hedge",
              "carry_by_maturity"):
        d[k] = None
    d.update(charts={}, greek_grids=[], scenario=None)
    gdf: pd.DataFrame | None = None

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
                node = (int(item["T_days"]), round(float(item["log_moneyness"]), 4))
                dropped[node] = str(item.get("reason") or "")
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
        d["n_cells"] = len(gdf)
        d["n_grid"] = len(maturities) * len(cols)
        d["n_missing"] = d["n_grid"] - d["n_cells"]
        if "as_of_date" in gdf.columns and gdf["as_of_date"].notna().any():
            d["as_of"] = _date_str(gdf["as_of_date"].dropna().iloc[0])

        cell = {(int(r.T_days), float(r.lm_key)): r for r in gdf.itertuples()}
        for T in maturities:
            cells: list[dict | None] = []
            for lm in cols:
                r = cell.get((T, lm))
                if r is None:
                    cells.append({"empty": True, "reason": dropped.get((T, round(lm, 4)))})
                    continue
                z = float(r.vomma_zscore)
                m = float(r.flag_metric)
                # colour = the flag metric relative to its threshold (full at the threshold)
                shade = (2.0 * min(m / flag_thr, 1.0)
                         if d["unstable_rule"]["kind"] == "pnl_ratio" and flag_thr else z)
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
        d["unstable_by_maturity"] = {int(t): int(n)
                                     for t, n in gdf.groupby("T_days")["is_unstable"].sum().items()}
        d["unstable_vega"] = unstable_vega_share(gdf)

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
        # Per-maturity carry (C23: Treasury r(T), q(T) from the fit's parity
        # forwards); the surface then records no single r or q.
        carry_raw = prov.get("carry")
        carry: dict = carry_raw if isinstance(carry_raw, dict) else {}
        crows = [c for c in carry.get("by_maturity") or []
                 if isinstance(c, dict) and _num(c.get("r")) is not None and _num(c.get("q")) is not None]
        d["carry_by_maturity"] = None
        if crows:
            d["carry_by_maturity"] = {
                "r_lo": min(c["r"] for c in crows), "r_hi": max(c["r"] for c in crows),
                "q_lo": min(c["q"] for c in crows), "q_hi": max(c["q"] for c in crows),
                "t_lo": min(c["T_days"] for c in crows), "t_hi": max(c["T_days"] for c in crows),
                "discount": carry.get("discount"), "n": len(crows)}
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
        gdf = None

    try:
        risk_extras(d, gdf)
    except artifacts.ArtifactError as e:
        _record_error(d, "scenario grid (data_store/greeks/scenario_grid.parquet)", e)
    except Exception as e:
        _record_error(d, "risk views", e)
    return _clean_keep(d)


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
    idx = cast(pd.DatetimeIndex, nav.index)
    year_end = nav.groupby(idx.year).last()
    out, base = [], float(nav.iloc[0])
    for yr, end in year_end.items():
        days = idx[idx.year == yr]
        out.append({"year": int(yr), "ret_pct": (float(end) / base - 1.0) * 100.0,
                    "partial": bool(yr == idx[-1].year and idx[-1].month < 12),
                    "first": str(days[0].date()), "last": str(days[-1].date()), "sessions": len(days)})
        base = float(end)
    return out


def partial_year_note(ann: list[dict]) -> str | None:
    """'2025 YTD: 2025-01-02 → 2025-03-24, 57 sessions' for each partial year of
    annual_returns, so a year label marked YTD always carries the span it covers."""
    parts = [f"{a['year']} YTD: {a['first']} → {a['last']}, {a['sessions']:,} sessions"
             for a in ann if a.get("partial")]
    return "; ".join(parts) or None


def window_returns(nav: pd.Series, periods: dict, end: str | None = None) -> list[dict]:
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
    for name, (start, period_end) in periods.items():
        stop = min(period_end, end) if (period_end and end) else (period_end or end)
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


def _entries_and_rolls(trades: list, sig: str, eq: pd.DataFrame | None) -> dict:
    """How a sleeve's trade records split into entries and expiry rolls.

    A position held past its straddle's expiry is closed with exit_reason
    "expiry" and continued as a new record entered that day, so a sleeve's
    trade count is its entries (flat -> on in the stored position series)
    plus those rolls: S3's 39 trades are 26 entries and 13 rolls. Returns
    {"entries", "rolls"} only when the two add up to the trade count.
    """
    col = f"{sig}_position"
    tr = [t for t in trades if getattr(t, "signal", None) == sig]
    if eq is None or col not in eq.columns or not tr:
        return {}
    pos = pd.to_numeric(eq[col], errors="coerce").fillna(0.0)
    entries = int(((pos != 0) & (pos.shift(1, fill_value=0.0) == 0)).sum())
    expiry_exits = {pd.Timestamp(t.exit_date) for t in tr if getattr(t, "exit_reason", None) == "expiry"}
    rolls = sum(1 for t in tr if pd.Timestamp(t.entry_date) in expiry_exits)
    return {"entries": entries, "rolls": rolls} if entries + rolls == len(tr) else {}


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
    for prev, cur in itertools.pairwise(order):
        rows[cur["step"]]["prev"] = prev
    g = sorted(grid, key=lambda r: float(r["beta"]))
    out["grid"] = g

    def flip(col: str) -> dict | None:
        """Adjacent grid betas where the sleeve's P&L changes sign."""
        for lo, hi in itertools.pairwise(g):
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


_PRE_C22_CACHE: dict = {}


def pre_c22_spread_facts(spx: pd.DataFrame, vix: pd.Series, start: str, end: str,
                         history_start: str) -> dict | None:
    """How the pre-C22 PDV spread behaved in the backtest window, from the data.

    σ̂ is the pre-C22 forecast as the engine replays it for the C22 bridge
    (BacktestEngine with pdv_horizon = 1: one PDVLinear per year on the
    1-day |r|·√252 target, fitted on returns before January 1), on the
    engine's own data window (SPX from `history_start`, the stored
    log_return column). Returns the share of sessions in [start, end] on
    which VIX − σ̂ exceeded the S1 entry threshold (signal_engine
    S1_ENTRY_THRESHOLD, 2 vol pts), and mean σ̂ over the mean realised RMS
    vol of the next PDV_SPREAD_HORIZON (21) sessions. The audit row quoted
    both numbers as typed text ("0.71×", "95%"), while the code comments
    say ~96% and ≈0.66×.
    """
    from joint_vol_calibration.backtest.backtest_engine import PDV_SPREAD_HORIZON, BacktestEngine
    from joint_vol_calibration.models.pdv import forward_target
    from joint_vol_calibration.signals.signal_engine import S1_ENTRY_THRESHOLD

    frame = spx.loc[history_start:end, ["close", "log_return"]]
    v = pd.to_numeric(vix, errors="coerce").loc[history_start:end]
    key = (start, end, history_start, len(frame), float(frame["close"].sum()),
           float(frame["log_return"].fillna(0).abs().sum()), len(v), float(v.fillna(0).sum()))
    if key in _PRE_C22_CACHE:
        return _PRE_C22_CACHE[key]
    eng = BacktestEngine(start_date=start, end_date=end, pdv_horizon=1)
    sig = eng._build_walkforward_pdv_vol(frame.rename_axis("date").reset_index())
    out = None
    if sig is not None and not sig.empty:
        sig.index = pd.to_datetime(sig.index)
        w = (pd.concat([sig.rename("f"), (v / 100.0).rename("v")], axis=1, join="inner")
             .loc[start:end].dropna())
        r = np.log(frame["close"].astype(float)).diff().dropna()
        y = forward_target(r, horizon=PDV_SPREAD_HORIZON, target="rv")
        both = pd.concat([sig.rename("f"), y.rename("y")], axis=1, join="inner").loc[start:end].dropna()
        if len(w) and len(both):
            out = {"share": float(((w["v"] - w["f"]) > S1_ENTRY_THRESHOLD).mean()),
                   "n": len(w), "threshold_vp": float(S1_ENTRY_THRESHOLD) * 100.0,
                   "ratio": float(both["f"].mean() / both["y"].mean()),
                   "horizon": int(PDV_SPREAD_HORIZON), "history_start": history_start}
    _PRE_C22_CACHE[key] = out
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


def beta_estimation(regimes: pd.Series | None = None) -> dict:
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

    def regime_on(session: str | None) -> str | None:
        if regimes is None or not session:
            return None
        lab = regimes.get(pd.Timestamp(session))
        return _REGIME_LABELS[int(lab)] if lab is not None and _num(lab) is not None else None

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
                         "crisis": [], "sleeves": {}, "open_trades": [], "charts": {}}
    for k in ("final_nav", "total_pnl", "start_date", "end_date",
              "wf_neg", "wf_total", "kpi", "sharpe_convention",
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

    eq: pd.DataFrame | None = None
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
        for col, name in (("nav", "Portfolio"), ("nav_s1c", "S1C Contrarian"), ("nav_s3", S3_SHORT),
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
        d["ann_partial_note"] = partial_year_note(d["ann"])

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
    trades: list = []
    with _loader_boundary(d, "backtest engine results (full_results.pkl)"):
        state = _load_pickle(DATA / "backtest" / "full_results.pkl")
        em = state.get("metrics") or {}
        trades = state.get("all_trades") or []
        if not em:
            raise ValueError("full_results.pkl carries no engine metrics")
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
        n_closed = len(closed)
        d["open_trades"] = [{"signal": t.signal, "entry": str(pd.Timestamp(t.entry_date).date()),
                             "net_pnl": float(t.net_pnl)}
                            for t in trades if getattr(t, "is_open", False)
                            and t.signal in (*PORTFOLIO_SLEEVES, "s1", "s2")]
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
            # The null test sizes every path, the real one included, at S3's
            # BASE kelly (the engine's pass 1); the S3 P&L the page reports is
            # after the pass-2 sizing overlay (+$41,952 vs +$56,471 on
            # 2018-2025). The base-size figure is shown only when the results
            # record it (metrics["s3_null_real_pnl"]); it is never typed in.
            "s3_null_p": _num(em.get("s3_null_p_value")),
            "s3_null_real": _num(em.get("s3_null_real_pnl")),
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
            {"label": "Cumulative Return", "value": _pct(kpi["cum_ret"]),
             "positive": (kpi["cum_ret"] or 0) > 0},
            {"label": "Ann. Return", "value": _pct(kpi["ann_ret"]), "positive": (kpi["ann_ret"] or 0) > 0},
            {"label": "Sharpe", "value": _f(kpi["sharpe"]), "positive": (kpi["sharpe"] or 0) > 0},
            {"label": "Sortino", "value": _f(kpi["sortino"]), "positive": (kpi["sortino"] or 0) > 0},
            {"label": "Max Drawdown", "value": _pct(kpi["mdd"]), "positive": False},
            {"label": "DD Duration",
             "value": "—" if kpi["dd_dur"] is None else f"{int(kpi['dd_dur']):,} sessions",
             "positive": False},
            {"label": "Calmar", "value": _f(kpi["calmar"]), "positive": (kpi["calmar"] or 0) > 0},
            {"label": "Win Rate (closed trades)", "value": _pct(kpi["win_rate"]),
             "positive": (kpi["win_rate"] or 0) > 0.5},
            {"label": "Avg Portfolio P&L / Closed Trade", "value": _dol(kpi["avg_pnl"]),
             "positive": (kpi["avg_pnl"] or 0) > 0},
            {"label": "Best Day", "value": _pct(kpi["best_day"]), "positive": True},
            {"label": "Worst Day", "value": _pct(kpi["worst_day"]), "positive": False},
            {"label": "Trades",
             "value": f"{n_rt:,}" + (f" ({n_rt - n_closed} open)" if n_rt > n_closed else ""),
             "positive": True},
            {"label": "Return Days", "value": "—" if kpi["n_days"] is None else f"{int(kpi['n_days']):,}",
             "positive": True},
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
                **_entries_and_rolls(trades, sig, eq),
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

        d["signal_labels"] = ["S1C contrarian", S3_SHORT, "S4 VRP"]
        d["signal_pnl"] = [None if sleeves[s]["total_pnl"] is None
                           else round(sleeves[s]["total_pnl"] / 1000, 1)
                           for s in PORTFOLIO_SLEEVES]

        for sig, lbl, is_ref in (("s1c", "S1C Contrarian", False), ("s3", S3_NAME, False),
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

    # ── C22 bridge + β grid (the audit table's evidence column) ──────────────
    with _loader_boundary(d, "C22 bridge / β grid (c22_bridge.csv, beta_sensitivity.csv)"):
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

    # ── β: the chains it was estimated from, and the later checks ────────────
    d["beta_est"] = None
    with _loader_boundary(d, "β estimation chains (run_backtest.estimate_beta)"):
        try:
            regimes = pd.read_parquet(DATA / "signals" / "regime_labels.parquet",
                                      columns=["regime"])["regime"]
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

    # ── What each sleeve trades (Research #sleeve-spec) ──────────────────────
    d["sleeve_spec"] = None
    with _loader_boundary(d, "sleeve specification (backtest_engine, signal_engine, full_results.pkl)"):
        d["sleeve_spec"] = sleeve_spec(trades)

    # ── Implied against subsequently realised vol, and the β·VIX premium ────
    d["vrp"] = None
    with _loader_boundary(d, "implied vs realised vol (vol_system.db spx_ohlcv, vix_daily)"):
        later = (d.get("beta_est") or {}).get("later") or []
        v = vrp_view(_read_spx(), _read_vix_daily(), d.get("start_date"), d.get("end_date"), d.get("beta"),
                     later[-1] if later else None, trades)
        d["charts"].update({f"vrp_{k}": c for k, c in v.pop("charts").items()})
        d["vrp"] = v

    # ── The pre-C22 PDV spread (audit row): how often it signalled short vol ─
    d["pre_c22_spread"] = None
    with _loader_boundary(d, "pre-C22 PDV spread facts (BacktestEngine, pdv_horizon = 1)"):
        if d["start_date"] and d["end_date"]:
            rb = _import_run_backtest()
            d["pre_c22_spread"] = pre_c22_spread_facts(
                _read_spx(), _read_vix_wide()["^VIX"], d["start_date"], d["end_date"], rb.DATA_START)
            if d["pre_c22_spread"] is None:
                raise ValueError("no pre-C22 PDV forecast could be built for the backtest window")

    # ── Facts the change log quotes, recomputed from the DB ──────────────────
    # (the log hard-coded "2018: ~1.5%" for a year whose ^IRX averaged 1.95%,
    # and "~53%" / "~25%" R2 shares for gates that give 48% / 23% here).
    d["tbill_by_year"], d["r2_share"] = {}, None
    with _loader_boundary(d, "change-log facts (T-bill rates, R2 shares)"):
        if d["start_date"] and d["end_date"]:
            with _db() as con:
                tb = pd.read_sql("SELECT date, rate FROM tbill_rates WHERE date BETWEEN ? AND ?",
                                 con, params=(d["start_date"], d["end_date"]), parse_dates=["date"])
            if len(tb):
                yearly = tb.groupby(tb["date"].dt.year)["rate"].mean()
                d["tbill_by_year"] = {int(y): float(v) for y, v in yearly.items()}
            spx_df, vix_wide = _read_spx(), _read_vix_wide()
            from joint_vol_calibration.signals.regime_classifier import (
                VVIX_REGIME2_THRESHOLD,
                build_regime_labels,
            )
            spx_in = spx_df.reset_index()[["date", "close", "log_return"]]
            vw_in = vix_wide.reset_index().rename(columns={"index": "date"})
            share = {}
            for key, gate in (("adaptive", None), ("fixed", VVIX_REGIME2_THRESHOLD)):
                lab = build_regime_labels(spx_in, vw_in, gate).loc[d["start_date"]:d["end_date"]]
                share[key] = float((lab == 2).mean()) if len(lab) else None
            share["fixed_level"] = float(VVIX_REGIME2_THRESHOLD)
            d["r2_share"] = share

    def _pct0(x: float | None) -> str:
        return "—" if x is None else f"{x * 100:.1f}%"

    # The shipped classifier's own evaluation record (quoted beside the C17 figures)
    d["clf_eval"] = None
    try:
        clf = _load_pickle(DATA / "signals" / "regime_classifier.pkl")
        ev = (getattr(clf, "metadata_", None) or {}).get("evaluation") or {}
        if ev.get("oos_acc") is not None and ev.get("persistence_acc") is not None:
            d["clf_eval"] = {"oos_acc": float(ev["oos_acc"]), "persistence_acc": float(ev["persistence_acc"]),
                             "n_test": int(ev.get("n_test") or 0)}
    except Exception as exc:     # page 1 reports a broken classifier; here the clause is omitted
        _logger.info("classifier evaluation unavailable: %s", exc)

    tby = d["tbill_by_year"]
    ytd_years = {a["year"] for a in d.get("ann") or [] if a.get("partial")}
    rate_txt = ", ".join(f"{y}{' YTD' if y in ytd_years else ''}: {v * 100:.2f}%"
                         for y, v in sorted(tby.items())
                         if y in (min(tby), 2023, max(tby))) if tby else "not available"
    sh = d["r2_share"] or {}

    # ── Improvement Summary (Steps 1–8) — a historical change log ─────────────
    d["improvements"] = [
        {"step": "1", "name": "Disable VIX options leg",
         "what": "JOINT_W3 = 0.0 (was 0.2). Attributed at the time to the Heston CIR density; "
                 "C22 traced the 37.14 vol pts to a broken VIX put pricer, spot-VIX vegas and a "
                 "stale chain, so it was not evidence about the model.",
         "before": "VIX-options leg in the objective with a broken pricer",
         "after":  "SPX + VIX index objective; VIX-options leg off (too few sessions of quotes)."},
        {"step": "2", "name": "Vega-weighted anchor selection",
         "what": ("Select 9 highest-BS-vega options per expiry (was log-uniform grid). "
                  "Moneyness lo 0.70 (was 0.75)."),
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
         "what": ("CalibratedClassifierCV(FrozenEstimator(XGB), isotonic). "
                  "S1S step: full/half/flat at P(R2) 0.4/0.6."),
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
    try:
        research_extras(d, eq)
    except artifacts.ArtifactError as e:
        _record_error(d, "backtest inference artifacts (data_store/backtest/inference)", e)
    except Exception as e:
        _record_error(d, "research views", e)
    # The demoted ML classifier and the rule-label distribution live here, not
    # on the Overview: research prototypes stay off the first screen.
    card = load_classifier_card()
    for err in card.pop("errors"):
        if err not in d["errors"]:
            d["errors"].append(err)
    d["clf"] = card
    return _clean_keep(d)


# ── C23 desk views: shared helpers ────────────────────────────────────────────
PARAM_META = (
    ("kappa", "κ", "mean reversion", ".3f"),
    ("theta", "θ", "long-run variance", ".4f"),
    ("sigma", "σ", "vol of variance", ".3f"),
    ("rho", "ρ", "spot–vol correlation", ".3f"),
    ("v0", "v₀", "initial variance", ".4f"),
)
# A parameter is "at a bound" when it sits within this fraction of its box
# width from the edge (the calibrator's own pinned_params test is the primary
# source; this only labels which edge).
BOUND_REL_TOL = 1e-4

# P&L diverging scale: orange (loss) -> panel colour -> blue (gain), the same
# blue/orange convention as the signed numbers.
PNL_DIV = ("#fbb360", "#a8743f", "#5a4630", "#2a2f37", "#1c2127", "#25344a", "#2f5a96", "#4c90f0", "#8abbff")


def pnl_colour(value: Any, limit: float) -> str | None:
    from desk_format import ramp
    x = _num(value)
    if x is None or limit <= 0:
        return None
    return ramp(PNL_DIV, 0.5 + 0.5 * max(-1.0, min(1.0, x / limit)))


def _gated_history() -> list[dict]:
    """Every current-schema, gate-passing fit, oldest first (parameter history)."""
    out = []
    for row in scan_calibrations():
        if row["legacy"] is not False or not row["ok"] or not row["session"]:
            continue
        c = _load_pickle(row["path"])
        losses = _cal_losses(c)
        out.append({"session": row["session"], "params": dict(c["params"]),
                    "rmse": _num(losses.get("spx_iv_rmse")), "oos": _num(losses.get("spx_oos_iv_rmse")),
                    "vix": _num(losses.get("vix_index_rmse")),
                    "wing": _num(losses.get("spx_oos_put_wing_rmse")),
                    "vx": _num(losses.get("vx_futures_rmse_oos")),
                    "pinned": list(c.get("pinned_params") or [])})
    return sorted(out, key=lambda r: r["session"])


RANGE_NOTE = "a range across sessions, not a sampling interval"


def session_range(values: Iterable[Any], spec: str = ".2f", what: str = "sessions") -> str | None:
    """'5 sessions: 0.40–0.62' over the finite values (None with fewer than two)."""
    xs = [float(x) for x in values if _num(x) is not None]
    if len(xs) < 2:
        return None
    lo, hi = min(xs), max(xs)
    signed_ = any(x < 0 for x in xs)
    f = (lambda x: signed(x, spec)) if signed_ else (lambda x: _fmt(x, spec))
    return f"{len(xs)} {what}: {f(lo)} to {f(hi)}" if signed_ else f"{len(xs)} {what}: {f(lo)}–{f(hi)}"


def rank_text(value: Any, values: Iterable[Any], spec: str = ".2f", lower_is_better: bool = True,
              what: str = "gated sessions") -> str | None:
    """'worst of 5 gated sessions (1.28–2.96)': where ``value`` ranks among
    ``values`` (which include it), best first, and their range."""
    v = _num(value)
    xs = [float(x) for x in values if _num(x) is not None]
    if v is None or len(xs) < 2:
        return None
    better = sum(1 for x in xs if (x < v - 1e-12 if lower_is_better else x > v + 1e-12))
    n, r = len(xs), min(better + 1, len(xs))
    word = "best" if r == 1 else "worst" if r == n else ordinal(r)
    return f"{word} of {n} {what} ({_fmt(min(xs), spec)}–{_fmt(max(xs), spec)})"


def _bound_side(value: float, lo: float, hi: float) -> str | None:
    span = (hi - lo) or 1.0
    if abs(value - hi) <= BOUND_REL_TOL * span:
        return "upper"
    if abs(value - lo) <= BOUND_REL_TOL * span:
        return "lower"
    return None


def _heat_cols(edges: Sequence[float]) -> list[str]:
    """Bucket ranges for cell details: '−0.30…−0.20'."""
    return [f"{_fmt(a, '+.2f')}…{_fmt(b, '+.2f')}" for a, b in itertools.pairwise(edges)]


def _heat_heads(edges: Sequence[float]) -> list[str]:
    """Compact column heads: each bucket's midpoint, '−0.25'."""
    return [_fmt((a + b) / 2.0, "+.3f").rstrip("0").rstrip(".") if abs((a + b) / 2.0) > 1e-12 else "0"
            for a, b in itertools.pairwise(edges)]


K_EDGES = (-0.30, -0.20, -0.14, -0.09, -0.05, -0.02, 0.0, 0.02, 0.05, 0.09, 0.14, 0.20, 0.30)
# K_EDGES plus an open bucket on each wing: pd.cut over K_EDGES alone returned
# NaN for ln(K/F) ≤ −0.30, so 108 of the 2,913 quotes of 2026-10-16 (the deep put
# wing, where the surfaces fit worst) were in no cell of any grid.
K_BINS = (-np.inf, *K_EDGES, np.inf)


def k_bucket(k: Any) -> pd.Series:
    """Bucket of each ln(K/F) on K_BINS: 0 is ≤ −0.30, len(K_EDGES) is > +0.30."""
    ks = k if isinstance(k, pd.Series) else pd.Series(k, dtype=float)
    return pd.cut(ks.astype(float), K_BINS, labels=False)


def k_buckets_used(k: Any) -> list[int]:
    """The interior buckets always, a wing bucket only when a quote falls in it."""
    present = {int(b) for b in k_bucket(k).dropna()}
    n = len(K_BINS) - 1
    return [b for b in range(n) if 0 < b < n - 1 or b in present]


def k_bucket_label(b: int, compact: bool = False) -> str:
    """'≤ −0.30', '−0.30…−0.20' (or the midpoint '−0.25' when compact), '> +0.30'."""
    n = len(K_BINS) - 1
    if b == 0:
        return f"≤ {_fmt(K_EDGES[0], '+.2f')}"
    if b == n - 1:
        return f"> {_fmt(K_EDGES[-1], '+.2f')}"
    lo, hi = K_EDGES[b - 1], K_EDGES[b]
    return _heat_heads((lo, hi))[0] if compact else _heat_cols((lo, hi))[0]


# ── Calibration page extras ──────────────────────────────────────────────────
def sigma_ridge(diag: dict, params: dict) -> dict | None:
    """What the σ profile says about κ: does it move with the σ cap?

    The pages said "4 of 5 parameters are set by the data" with σ on its box
    ceiling, while the profile refits show κ rising with every cap (1.2 at
    σ ≤ 1 to 27.4 with σ ≤ 10, κ ≤ 60): where the fit sits on the κ–σ ridge is
    set by the box. ``moves`` is true when κ rises with the cap across the
    profile (rank correlation ≥ 0.8) and its top row's κ is at least 1.5× the
    fit's.
    """
    if diag.get("state") != "ok":
        return None
    rows = [r for r in diag.get("sigma_profile") or []
            if _num(r.get("sigma_cap")) is not None
            and _num((r.get("params") or {}).get("kappa")) is not None]
    rows = sorted(rows, key=lambda r: float(r["sigma_cap"]))
    if len(rows) < 3:
        return None
    caps = pd.Series([float(r["sigma_cap"]) for r in rows])
    kap = pd.Series([float(r["params"]["kappa"]) for r in rows])
    rho = float(caps.corr(kap, method="spearman"))
    top = rows[-1]
    k_fit = _num(params.get("kappa"))
    rse = ((diag.get("ident") or {}).get("relative_standard_errors") or {}).get("kappa")
    return {"moves": bool(rho >= 0.8 and k_fit and float(top["params"]["kappa"]) >= 1.5 * k_fit),
            "rank_corr": rho, "kappa_fit": k_fit,
            "cap_lo": float(rows[0]["sigma_cap"]), "kappa_lo": float(rows[0]["params"]["kappa"]),
            "cap_top": float(top["sigma_cap"]), "kappa_top": float(top["params"]["kappa"]),
            "sigma_top": _num(top["params"].get("sigma")), "kappa_max": _num(diag.get("kappa_max")),
            "kappa_rse": _num(rse)}


# A refit whose model VVIX is within this fraction of the ^VVIX close counts as
# matching it; the trade-off sentence states the threshold it uses.
VVIX_MATCH_REL = 0.10
_COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def count_word(n: int) -> str:
    """'four' for 4 (prose counts up to ten), digits above."""
    return _COUNT_WORDS[n] if 0 <= n < len(_COUNT_WORDS) else f"{n:,}"


def vvix_tradeoff(profiles: Sequence[dict]) -> list[dict]:
    """Per diagnosed session, the σ profile's production-cap refit and its widest-cap refit.

    ``profiles`` is desk_artifacts.load_sigma_profiles()["sessions"]. A session
    counts only when both refits record SPX out-of-sample RMSE and the model
    VVIX against the market. ``far``: even the widest refit prices VVIX more
    than VVIX_MATCH_REL above the ^VVIX close. ``binds``: σ sits on the
    production cap in the production-cap refit.
    """
    out = []
    for p in profiles:
        rows = [r for r in p["rows"] if None not in (r["spx_oos_iv_rmse"], r["vvix_model"], r["vvix_market"])]
        cap = p.get("production_cap")
        prod = next((r for r in rows if cap is not None and abs(r["sigma_cap"] - cap) < 1e-9), None)
        if not rows or prod is None:
            continue
        top = rows[-1]
        out.append({"session": p["session"], "production_cap": cap, "prod": prod, "top": top,
                    "binds": bool(prod["sigma_at_cap"]),
                    "far": top["vvix_model"] > (1.0 + VVIX_MATCH_REL) * top["vvix_market"]})
    return out


def vvix_tradeoff_sentence(tr: Sequence[dict]) -> str | None:
    """The joint SPX/VIX trade-off in one paragraph, from vvix_tradeoff().

    It replaced the unconditional "Heston needs a vol-of-vol that prices VVIX
    far above the market", which the σ ≤ 10 refit of 2026-10-16 contradicts
    (VVIX 112.7 against 109.9). Every number is read from the σ profiles.
    """
    if not tr:
        return None
    n = len(tr)
    far = [t for t in tr if t["far"]]
    esc = [t for t in tr if not t["far"]]
    parts = []
    if far:
        lead = (f"On {'the one' if n == 1 else 'all ' + str(n) if len(far) == n else f'{len(far)} of {n}'}"
                f" diagnosed session{'' if n == 1 else 's'}")
        caps = sorted({_fmt(t["top"]["sigma_cap"], "g") for t in far})
        pairs = ", ".join(f"{_fmt(t['top']['vvix_model'], '.0f')} vs {_fmt(t['top']['vvix_market'], '.0f')}"
                          for t in far)
        parts.append(f"{lead} even the σ ≤ {' / '.join(caps)} refit prices VVIX more than "
                     f"{VVIX_MATCH_REL * 100:.0f}% above the ^VVIX close ({pairs}), so a wider σ box "
                     f"does not reconcile the SPX smile with VVIX there.")
    clauses = []
    for t in esc:
        pr, tp = t["prod"], t["top"]
        bind = f"the σ ≤ {_fmt(t['production_cap'], 'g')} box binds, and " if t["binds"] else ""
        v_oos = "lowers" if tp["spx_oos_iv_rmse"] < pr["spx_oos_iv_rmse"] else "raises"
        v_vv = "lowers" if tp["vvix_model"] < pr["vvix_model"] else "raises"
        refit = (f"the σ ≤ {_fmt(tp['sigma_cap'], 'g')} refit (σ {_fmt(tp['sigma'], '.2f')}, "
                 f"κ {_fmt(tp['kappa'], '.1f')}, Feller margin {signed(tp['feller_margin'], '.1f')})")
        clauses.append(
            f"on {t['session']} {bind}{refit} {v_oos} OOS RMSE {_fmt(pr['spx_oos_iv_rmse'], '.2f')} → "
            f"{_fmt(tp['spx_oos_iv_rmse'], '.2f')} vol pts and {'' if v_vv == v_oos else v_vv + ' '}VVIX "
            f"{_fmt(pr['vvix_model'], '.0f')} → {_fmt(tp['vvix_model'], '.1f')} against "
            f"{_fmt(tp['vvix_market'], '.1f')}")
    if clauses:
        verb = "does" if len(far) == 1 else "do"
        tail = ("one session" if len(esc) == 1 else f"{count_word(len(esc))} sessions") + (
            f", the other {count_word(len(far))} {verb} not escape" if far else "")
        body = "; ".join(clauses) + f"; {tail}."
        parts.append(body[0].upper() + body[1:])
    return " ".join(parts)


def box_refit_note(tr: Sequence[dict], session: str | None) -> str:
    """The widest σ refit of the headline session, for the Model Risk box-bound check:
    ' σ ≤ 10 refit: OOS 1.19, VVIX 112.7 (2026-10-16 only)' when it is the one
    session whose widest refit comes within VVIX_MATCH_REL of the ^VVIX close."""
    t = next((x for x in tr if x["session"] == session), None)
    if t is None:
        return ""
    tp = t["top"]
    esc = [x["session"] for x in tr if not x["far"]]
    scope = (f" ({session} only; on the other {count_word(len(tr) - 1)} diagnosed sessions it stays more "
             f"than {VVIX_MATCH_REL * 100:.0f}% above the ^VVIX close)" if esc == [session] and len(tr) > 1
             else f" ({session})")
    return (f" σ ≤ {_fmt(tp['sigma_cap'], 'g')} refit: OOS {_fmt(tp['spx_oos_iv_rmse'], '.2f')}, "
            f"VVIX {_fmt(tp['vvix_model'], '.1f')} vs {_fmt(tp['vvix_market'], '.1f')}{scope}.")


def sigma_frontier_spec(tr_profiles: Sequence[dict], headline: str | None) -> dict | None:
    """The σ frontier: SPX out-of-sample RMSE against model VVIX − market, one line per session.

    Each point is one refit of the σ profile, labelled with its σ cap; the
    production cap's point is drawn larger, caps above it hollow. The headline
    session is drawn in the headline colour, the others muted.
    """
    sess = []
    for p in tr_profiles:
        rows = [r for r in p["rows"] if None not in (r["spx_oos_iv_rmse"], r["vvix_model"], r["vvix_market"])]
        if rows:
            sess.append((p, rows))
    if not sess:
        return None
    series, tbl = [], []
    for p, rows in sess:
        cap = p.get("production_cap")
        head = p["session"] == headline
        series.append({
            "name": p["session"], "type": "line", "color": "headline" if head else "muted",
            "width": 2.0 if head else 1.2, "symbol": "circle", "size": 6 if head else 5,
            "z": 4 if head else 2,
            "legend": True, "legendKey": p["session"] if head else "Other sessions",
            "data": [[r["spx_oos_iv_rmse"], r["vvix_model"] - r["vvix_market"]] for r in rows],
            "pointLabels": [_fmt(r["sigma_cap"], "g") for r in rows],
            "hollowPoints": [cap is not None and r["sigma_cap"] > cap + 1e-9 for r in rows],
            "bigPoints": [cap is not None and abs(r["sigma_cap"] - cap) < 1e-9 for r in rows]})
        tbl += [[p["session"], _fmt(r["sigma_cap"], "g"), _fmt(r["spx_oos_iv_rmse"], ".2f"),
                 _fmt(r["vvix_model"], ".1f"), _fmt(r["vvix_market"], ".2f"),
                 signed(r["vvix_model"] - r["vvix_market"], ".1f")] for r in rows]
    series.sort(key=lambda s: s["color"] != "headline")          # its legend button first; z puts it on top
    spec = charts.xy(
        "sigma-frontier", "σ frontier: SPX out-of-sample error against model VVIX − market, by σ cap",
        x={"type": "value", "name": "SPX OOS IV RMSE (vol pts)", "digits": 1, "tipDigits": 2, "label": "OOS",
           "unit": " vol pts"},
        y={"name": "VVIX model − market (index pts)", "digits": 0, "tipDigits": 1, "signed": True},
        series=series, tooltip="item", csv="sigma_frontier.csv",
        marks={"y": [{"value": 0, "label": "model = market", "color": "muted", "dash": True}]},
        grid={"right": 30, "top": 18},
        describe=(f"{len(sess)} diagnosed session{'' if len(sess) == 1 else 's'}, one line each; each point "
                  "is a refit under one σ cap (labelled), the production cap's point larger, caps above it "
                  "hollow. Down and to the left is better on both legs."),
        table=charts.table("σ profile by session: SPX out-of-sample RMSE and model VVIX against the ^VVIX "
                           "close",
                           ["Session", "σ cap#", "SPX OOS RMSE (vol pts)#", "VVIX model#",
                            "VVIX market#", "VVIX − market (index pts)#"], tbl))
    spec["pointCsv"] = "sigma_cap"
    return spec


SKEW_SHORT_DAYS = 30.0     # 'the short end' of the skew comparison: listed expiries under 30 days


def _session_skews(s: dict) -> dict:
    """One stored session's ATM skews: SVI-cal (analytic, per slice) and eSSVI
    (surface.json term_structure.atm_skew_essvi), with their power-law slopes."""
    sj = s["surface"]
    sl = sorted((((sj.get("surfaces") or {}).get("svi_cal") or {}).get("params") or {}).get("slices") or [],
                key=lambda x: float(x["T"]))
    ts = sorted((t for t in sj.get("term_structure") or [] if _num(t.get("atm_skew_essvi")) is not None),
                key=lambda t: float(t["T_days"]))
    t_svi = [float(x["T"]) for x in sl]
    svi = [desk_surface.svi_atm_skew(x) for x in sl]
    t_ess = [float(t["T_days"]) / desk_surface.DAYS_PER_YEAR for t in ts]
    ess = [float(t["atm_skew_essvi"]) for t in ts]
    return {"session": s["session"], "expiries": [str(x["expiry"])[:10] for x in sl], "T": t_svi, "svi": svi,
            "essvi_expiries": [str(t["expiry"])[:10] for t in ts], "T_essvi": t_ess, "essvi": ess,
            "slopes": {"svi_all": desk_surface.skew_slope(t_svi, svi),
                       "essvi_all": desk_surface.skew_slope(t_ess, ess),
                       "svi_30": desk_surface.skew_slope(t_svi, svi, SKEW_SHORT_DAYS),
                       "essvi_30": desk_surface.skew_slope(t_ess, ess, SKEW_SHORT_DAYS)}}


def skew_term_view(params: dict, fit_session: str | None, sessions: Sequence[dict],
                   refit: dict | None) -> dict | None:
    """The ATM-skew term structure (Calibration #skew-term).

    The market's skew per stored session (SVI-cal and eSSVI), its power-law
    slope in T (all expiries and T ≥ 30 d), and the Heston skew at the
    published parameters ``params`` and at the σ-profile's widest refit, both at
    the listed maturities of the headline session (the fit's own session when a
    surface report exists for it, else the newest one)."""
    if not sessions:
        return None
    per = [_session_skews(s) for s in sessions]
    per = [p for p in per if p["T"]]
    if not per:
        return None
    by = {p["session"]: p for p in per}
    head = by.get(fit_session or "") or per[-1]
    days = [t * desk_surface.DAYS_PER_YEAR for t in head["T"]]
    hes = heston_atm_skew(params, days)
    hes_refit = heston_atm_skew(refit["params"], days) if refit else None
    ess_by = dict(zip(head["essvi_expiries"], head["essvi"]))
    rows = []
    for e, t, m, h_, r_ in zip(head["expiries"], head["T"], head["svi"], hes, hes_refit or [None] * len(hes)):
        rows.append({"expiry": e, "T": t, "T_days": t * desk_surface.DAYS_PER_YEAR, "svi": m,
                     "essvi": _num(ess_by.get(e)), "heston": h_, "refit": r_,
                     "ratio": (abs(h_) / abs(m)) if (h_ is not None and m) else None})
    return {"head": head["session"], "fit_session": fit_session, "sessions": per, "rows": rows,
            "heston_params": params, "refit": refit,
            "slope_ranges": {k: ([p["slopes"][k]["slope"] for p in per if p["slopes"][k]])
                             for k in ("svi_all", "essvi_all", "svi_30", "essvi_30")}}


def skew_term_compare(v: dict) -> dict | None:
    """Published Heston against the SVI-cal market skew at the headline session's
    listed expiries: the |Heston| / |market| ratio range, the shortest expiry's
    pair, and below 30 days how many expiries have the market above Heston."""
    rows = [r for r in v["rows"] if r["ratio"] is not None]
    if not rows:
        return None
    short = [r for r in rows if r["T_days"] < SKEW_SHORT_DAYS]
    first = rows[0]
    near30 = min(rows, key=lambda r: abs(r["T_days"] - 30.0))
    return {"lo": min(r["ratio"] for r in rows), "hi": max(r["ratio"] for r in rows), "n": len(rows),
            "first": first, "near30": near30, "n_short": len(short),
            "short_market_above": sum(1 for r in short if abs(r["svi"]) > abs(r["heston"])),
            "steeper_all": all(r["ratio"] > 1 for r in rows),
            "flatter_all": all(r["ratio"] < 1 for r in rows)}


def skew_term_sentence(v: dict) -> str | None:
    """What the chart shows, computed: Heston against the SVI-cal market skew and
    SVI-cal against eSSVI near 30 days."""
    c = skew_term_compare(v)
    if not c:
        return None
    f = c["first"]
    if c["steeper_all"]:
        rel = "steeper than the SVI-cal market skew at every listed expiry"
    elif c["flatter_all"]:
        rel = "flatter than the SVI-cal market skew at every listed expiry"
    else:
        n_st = sum(1 for r in v["rows"] if r["ratio"] is not None and r["ratio"] > 1)
        rel = f"steeper than the SVI-cal market skew on {n_st} of {c['n']} listed expiries"
    out = (f"On {v['head']} the published Heston fit's ATM skew is {rel} "
           f"(×{_fmt(c['lo'], '.2f')} to ×{_fmt(c['hi'], '.2f')}; {_fmt(abs(f['heston']), '.2f')} against "
           f"{_fmt(abs(f['svi']), '.2f')} at {_fmt(f['T_days'], '.0f')} d).")
    if c["n_short"]:
        out += (f" Below {_fmt(SKEW_SHORT_DAYS, '.0f')} days the market line lies above the Heston line on "
                f"{c['short_market_above']} of {c['n_short']} expiries.")
    n = c["near30"]
    if n["essvi"] is not None and n["svi"]:
        out += (f" eSSVI's ψ(θ) ties the skew across expiries, so the two fits of the same quotes differ: at "
                f"{_fmt(n['T_days'], '.0f')} d eSSVI gives {signed(n['essvi'], '.3f')} and SVI-cal "
                f"{signed(n['svi'], '.3f')} ({_pct_prose(n['essvi'] / n['svi'] - 1, 0)}).")
    return out


def skew_term_spec(v: dict) -> dict:
    """Log–log chart of |ATM skew| against T, every stored session, and its table."""
    head = v["head"]
    hp = next(p for p in v["sessions"] if p["session"] == head)
    series: list[dict[str, Any]] = []
    for p in v["sessions"]:
        if p["session"] == head:
            continue
        series.append({"name": f"SVI-cal {p['session']}", "legendKey": "SVI-cal, other sessions",
                       "legend": True, "type": "line", "color": "muted", "width": 1, "z": 1,
                       "data": [[t, abs(s)] for t, s in zip(p["T"], p["svi"])]})
    series.append({"name": f"SVI-cal {head}", "legend": True, "type": "line",
                   "color": model_colour("svi_cal"), "width": 2, "symbol": "circle", "size": 4, "z": 3,
                   "data": [[t, abs(s)] for t, s in zip(hp["T"], hp["svi"])]})
    series.append({"name": f"eSSVI {head}", "legend": True, "type": "line", "color": model_colour("essvi"),
                   "width": 1.5, "dash": "dotted", "z": 3,
                   "data": [[t, abs(s)] for t, s in zip(hp["T_essvi"], hp["essvi"])]})
    series.append({"name": "Heston, published fit", "legend": True, "type": "line",
                   "color": model_colour("heston_v2"), "width": 1.5, "z": 3,
                   "data": [[r["T"], abs(r["heston"])] for r in v["rows"] if r["heston"] is not None]})
    if v["refit"]:
        series.append({"name": f"Heston, σ ≤ {_fmt(v['refit']['cap'], 'g')} refit", "legend": True,
                       "type": "line", "color": model_colour("heston_v2"), "width": 1.5, "dash": True, "z": 3,
                       "data": [[r["T"], abs(r["refit"])] for r in v["rows"] if r["refit"] is not None]})
    trows = [[head, r["expiry"], _fmt(r["T_days"], ".0f"), signed(r["svi"], ".3f"), signed(r["essvi"], ".3f"),
              signed(r["heston"], ".3f"), signed(r["refit"], ".3f")] for r in v["rows"]]
    for p in v["sessions"]:
        if p["session"] == head:
            continue
        eb = dict(zip(p["essvi_expiries"], p["essvi"]))
        trows += [[p["session"], e, _fmt(t * desk_surface.DAYS_PER_YEAR, ".0f"), signed(s, ".3f"),
                   signed(eb.get(e), ".3f"), "—", "—"] for e, t, s in zip(p["expiries"], p["T"], p["svi"])]
    refit_lab = f"Heston σ ≤ {_fmt(v['refit']['cap'], 'g')}#" if v["refit"] else "Heston refit#"
    # log axes bounded by the data (ECharts' default runs to whole decades, two
    # thirds of the plot empty), a 20% margin each side
    pts: list[list[float]] = [pt for s in series for pt in s["data"] if pt[0] and pt[1]]
    xb = (round_sig(min(p_[0] for p_ in pts) / 1.2, 2), round_sig(max(p_[0] for p_ in pts) * 1.2, 2))
    yb = (round_sig(min(p_[1] for p_ in pts) / 1.2, 2), round_sig(max(p_[1] for p_ in pts) * 1.2, 2))
    return charts.xy(
        "chart-skew-term", f"ATM skew against maturity on log–log axes, {head} and every stored session",
        x={"type": "value", "log": True, "name": "T (years, log scale)", "digits": 2, "tipDigits": 3,
           "min": xb[0], "max": xb[1]},
        y={"type": "value", "log": True, "name": "|skew| (vol per unit ln(K/F), log)", "digits": 1,
           "tipDigits": 3, "min": yb[0], "max": yb[1]},
        series=series, tooltip="item", csv=f"atm_skew_term_{head}.csv", grid={"left": 56},
        describe=(f"|∂σ/∂ln(K/F)| at the forward against T in years, both axes logarithmic: the SVI-cal "
                  f"surface of {head} (solid) and of the other stored sessions (grey), the eSSVI fit of "
                  f"{head} (dotted), and the Heston smile at the published parameters and at the σ-profile "
                  "refit (dashed). A straight line is a power law; skews are negative and plotted as "
                  "magnitudes."),
        table=charts.table(
            "ATM skew ∂σ/∂ln(K/F) at the forward (vol per unit ln-moneyness; negative = puts over calls); "
            f"Heston at {head}'s listed maturities only",
            ["Session", "Expiry", "T (d)#", "SVI-cal#", "eSSVI#", "Heston published#", refit_lab], trows))


def implied_beta_spec(rows: Sequence[dict], cols: Sequence[str], canonical: float | None) -> dict:
    """The β that reprices each put-write index from β·VIX, by year (the table
    beside it as a chart), against the backtest's fixed β."""
    years = list(rows[0]["years"])
    series = [{"name": r["index"], "type": "line", "color": "headline" if i == 0 else "info",
               "dash": i > 0, "symbol": "circle", "size": 5, "width": 1.5, "legend": True,
               "data": [r["years"].get(y) for y in years]} for i, r in enumerate(rows)]
    return charts.xy(
        "chart-implied-beta", "β implied by the put-write indices, by year, against the backtest's β",
        x={"type": "category", "categories": list(cols), "name": ""},
        y={"name": "Implied β", "digits": 2, "tipDigits": 3},
        series=series, xs=list(cols), csv="implied_beta_by_year.csv",
        marks=({"y": [{"value": canonical, "label": f"backtest β {_fmt(canonical, '.2f')}",
                       "color": "muted"}]} if canonical is not None else None),
        describe=("Per year, the β at which β·VIX reprices PUT and WPUT; the dashed line is the backtest's "
                  "fixed β. A YTD year is partial (see the note under the table)."),
        table=charts.table("β implied by the put-write indices by year", ["Index"] + [c + "#" for c in cols],
                           [[r["index"]] + [_fmt(r["years"].get(y), ".2f") for y in years] for r in rows]))


def gated_fits_spec(hist: Sequence[dict], session: str | None) -> dict | None:
    """In-sample and out-of-sample SPX IV RMSE of every gated fit, by session."""
    if len(hist) < 2:
        return None
    sess = [h["session"] for h in hist]
    col = model_colour("heston_v2")
    return charts.xy(
        "chart-gated-fits", "SPX IV RMSE of every gated fit, in and out of sample, by session",
        x={"type": "category", "categories": sess, "name": "", "interval": 0},
        y={"name": "RMSE (vol pts)", "digits": 1, "tipDigits": 2, "zero": True},
        series=[{"name": "Out of sample", "type": "line", "color": col, "symbol": "circle", "size": 6,
                 "width": 1.5, "legend": True, "data": [h["oos"] for h in hist]},
                {"name": "In sample", "type": "line", "color": col, "dash": True, "symbol": "circle",
                 "size": 4, "width": 1.2, "legend": True, "data": [h["rmse"] for h in hist]}],
        xs=sess, csv="gated_fits_rmse.csv",
        marks={"x": [{"value": session, "label": "headline", "color": "muted"}]} if session in sess else None,
        describe=("Each gate-passing fit's SPX implied-vol RMSE on its anchors (dashed) and on the quotes it "
                  "did not use (solid), oldest first; the headline fit is marked."),
        table=charts.table("SPX IV RMSE (vol pts) of the gated fits",
                           ["Session", "In sample#", "Out of sample#", "Put wing OOS#"],
                           [[h["session"], _fmt(h["rmse"], ".2f"), _fmt(h["oos"], ".2f"),
                             _fmt(h.get("wing"), ".2f")] for h in hist]))


def oos_rows_of(cal: dict) -> list[dict]:
    """A fit's out-of-sample SPX IV RMSE per expiry (vol pts), with days to expiry."""
    oos = (cal.get("oos") or {}).get("spx_oos_rmse_by_expiry") or {}
    t_by_exp = {str(c.get("expiry")): float(c["T"]) * 365.0 for c in cal.get("carry_by_expiry") or []
                if _num(c.get("T")) is not None}
    anchor_exp = {str(a.get("expiry")) for a in cal.get("anchors_by_expiry") or []}
    return sorted(({"expiry": e, "T_days": t_by_exp.get(e), "rmse": float(v), "anchor": e in anchor_exp}
                   for e, v in oos.items() if _num(v) is not None), key=lambda r: r["expiry"])


VX_LEG_REL_TOL = 1e-6     # the page's recomputed VX RMSE against the fit's own record


def vx_empty_sentence(session: str, last: str | None) -> str:
    """Why a gated fit has no VX leg on the page (nothing is drawn in its place)."""
    if last is None:
        return f"The market database has no VX futures table; none stored for {session}."
    if session > last:
        return f"VX futures table ends {last}; none stored for {session}."
    return f"No monthly VX settlements are stored for {session}."


def vx_leg_spec(session: str, leg: dict) -> dict:
    """One gated fit's VX leg: settlements against the model E^Q[VIX_T], gap labels."""
    rows = leg["rows"]
    return charts.xy(
        f"chart-vx-{session}", f"VX futures on {session}: settlements against the Heston fit's E[VIX_T]",
        x={"type": "value", "name": "Days to expiry (calendar)", "digits": 0, "min": 0, "zero": True},
        y={"name": "VX (index pts)", "digits": 1, "tipDigits": 2},
        series=[{"name": "Heston E[VIX_T]", "type": "line", "color": model_colour("heston_v2"), "width": 1.6,
                 "legend": True, "data": leg["curve"]},
                {"name": "VX settlement", "type": "scatter", "color": "headline", "size": 7, "legend": True,
                 "data": [[r["days"], r["settle"]] for r in rows],
                 "pointLabels": [signed(r["gap"], ".2f") for r in rows], "pointPrefix": "model − settle ",
                 "labelPos": "bottom"}],
        tooltip="item", csv=f"vx_leg_{session}.csv", grid={"left": 46, "top": 12},
        describe=(f"{leg['n']} monthly VX settlements of {session} and the fit's model price at each "
                  f"expiry; RMSE {_fmt(leg['rmse'], '.2f')} index pts, model below the settlement on "
                  f"{leg['n_below']} of {leg['n']}."),
        table=charts.table(f"VX futures on {session}: settlement and model (index pts)",
                           ["Contract", "Expiry", "Days#", "Settlement#", "Model E[VIX_T]#",
                            "Model − settlement#"],
                           [[r["month"], r["expiry"], _fmt(r["days"], ".0f"), _fmt(r["settle"], ".2f"),
                             _fmt(r["model"], ".2f"), signed(r["gap"], ".2f")] for r in rows]))


def vx_leg_view(hist: Sequence[dict], headline: str | None) -> dict:
    """The VX futures leg of every gated fit (Calibration #vixleg): each fit against
    the settlements of its own session only, the RMSE recomputed and checked
    against the fit's record, and an explicit empty state where the session has
    no settlements. The tab shown first is the newest fit with a leg."""
    last = vx_last_date()
    views = []
    for h in hist:
        rows = vx_session_rows(h["session"]) if last is not None else []
        leg = desk_market.vx_model_leg(h["params"], rows)
        rec = h.get("vx")
        views.append({
            "session": h["session"], "headline": h["session"] == headline, "leg": leg, "recorded": rec,
            "matches_record": (leg is not None and rec is not None
                               and abs(leg["rmse"] - rec) <= VX_LEG_REL_TOL * max(1.0, abs(rec))),
            "spec": vx_leg_spec(h["session"], leg) if leg else None,
            "empty": None if leg else vx_empty_sentence(h["session"], last)})
    with_leg = [v["session"] for v in views if v["leg"]]
    rmses = [v["leg"]["rmse"] for v in views if v["leg"]]
    return {"views": views, "last": last, "default": (with_leg[-1] if with_leg else headline),
            "n_with": len(with_leg), "range": (min(rmses), max(rmses)) if rmses else None,
            "headline_empty": next((v["empty"] for v in views if v["headline"] and not v["leg"]), None)}


def calibration_extras(d: dict, cal: dict) -> None:
    """Parameter tiles, history, OOS by expiry, VIX leg, smiles, identifiability."""
    from joint_vol_calibration.config import HESTON_BOUNDS

    p = cal["params"]
    session = d.get("session")
    hist = _gated_history()
    d["history"] = [{"session": h["session"], **{k: h["params"][k] for k, *_ in PARAM_META},
                     "oos": h["oos"], "rmse": h["rmse"], "vix": h["vix"], "vx": h["vx"]} for h in hist]
    d["vx_leg"] = vx_leg_view(hist, session)
    gated = gated_fits_spec(hist, session)
    if gated:
        d["charts"]["gated"] = gated
    diag = artifacts.calibration_diagnostics(cal, DATA, session)
    d["ident"] = diag
    d["ridge"] = sigma_ridge(diag, p)
    # the joint SPX/VIX trade-off across every diagnosed session's σ profile
    try:
        sp = artifacts.load_sigma_profiles(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "calibration diagnostics σ profiles (diagnostics/<session>.json)", e)
        sp = {"state": "error"}
    profiles = (sp.get("sessions") or []) if sp.get("state") == "ok" else []
    d["vvix_tradeoff"] = vvix_tradeoff(profiles)
    d["vvix_sentence"] = vvix_tradeoff_sentence(d["vvix_tradeoff"])
    frontier = sigma_frontier_spec(profiles, session)
    if frontier:
        d["charts"]["frontier"] = frontier
    # the published fit's model VVIX (its own diagnostics file, when it names this fit)
    own = next((x for x in profiles if x["session"] == session), None)
    sf = (own or {}).get("stored_fit") or {}
    d["vvix"] = ({"model": sf["vvix_model"], "market": sf["vvix_market"]}
                 if sf.get("file") == d.get("calib_file") and None not in (sf.get("vvix_model"),
                                                                           sf.get("vvix_market")) else None)
    d["sigma_profile_market"] = next((r["vvix_market"] for r in (own or {}).get("rows") or []
                                      if r["vvix_market"] is not None), None)
    # where this fit ranks among the gated fits (lower error is better), one tile each:
    # its own unrounded history row (the tiles' values are rounded for display)
    own_h = next((h for h in hist if h["session"] == session), None)
    d["fit_ranks"] = {key: (rank_text(own_h[key], [h[key] for h in hist]) if own_h else None)
                      for key in ("rmse", "oos", "wing", "vix")}
    gaps = [x["stored_fit"]["vvix_model"] - x["stored_fit"]["vvix_market"] for x in profiles
            if None not in ((x.get("stored_fit") or {}).get("vvix_model"),
                            (x.get("stored_fit") or {}).get("vvix_market"))]
    d["fit_ranks"]["vvix"] = (rank_text(d["vvix"]["model"] - d["vvix"]["market"], gaps, ".0f",
                                        what="diagnosed sessions, model − market")
                              if d.get("vvix") else None)
    # Heston against rough Heston on every diagnosed session (summary.json):
    # the same anchors, residuals, VIX leg and out-of-sample quotes per session.
    try:
        ds = artifacts.load_diagnostics_summary(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "calibration diagnostics summary (diagnostics/summary.json)", e)
        ds = {"state": "error"}
    d["rough_vs_heston"] = ds
    if ds.get("state") == "ok":
        jr = joint_fit_result(ds, session)
        d["rough_vs_heston"] = {**ds, "result": jr, "rows": [
            {**r, "rough_minus_heston": (
                r["joint_rough_oos"] - r["sigma_cap2_oos"]
                if r["joint_rough_oos"] is not None and r["sigma_cap2_oos"] is not None else None),
             "rough_is_heston": r["joint_rough_H"] is not None and r["joint_rough_H"] >= 0.5 - 1e-9}
            for r in ds["rows"]]}
    # the ATM-skew term structure: the market's power law against Heston (#skew-term)
    d["skew_term"] = None
    d["chain_badge"] = None
    try:
        desk_inputs = artifacts.load_surface_desk_inputs(DATA)
        sk = skew_term_view(p, session, desk_inputs, artifacts.sigma_profile_refit(DATA, session))
    except artifacts.ArtifactError as e:
        _record_error(d, "ATM-skew term structure (surface/<session>/surface.json, diagnostics)", e)
        desk_inputs, sk = [], None
    # the chain this fit's session was fitted on, as the surface report records it
    own_chain = next((s for s in desk_inputs if s["session"] == session), None)
    if own_chain:
        d["chain_badge"] = chain_badge(own_chain["surface"], [
            n for s in desk_inputs if s["session"] != session for n in [session_expiries(s["surface"])]
            if n is not None])
    if sk:
        rv = d.get("rough_vs_heston")
        jr = rv.get("result") if isinstance(rv, dict) else None
        sk["rough_H"] = ({"H": jr.get("rough_H"), "on_floor": jr.get("rough_H_on_bound"),
                          "session": jr["session"]} if jr else None)
        sk["sentence"] = skew_term_sentence(sk)
        sk["compare"] = skew_term_compare(sk)
        d["skew_term"] = sk
        d["charts"]["skew_term"] = skew_term_spec(sk)
    ident = diag.get("ident") if diag.get("state") == "ok" else {}
    se = (ident or {}).get("standard_errors") or {}
    pinned = set(d.get("pinned_params") or [])
    tiles = []
    for key, symbol, name, spec in PARAM_META:
        lo, hi = (float(x) for x in HESTON_BOUNDS[key])
        val = float(p[key])
        side = _bound_side(val, lo, hi)
        if key in pinned and side is None:
            side = "bound"
        series = [h["params"][key] for h in hist]
        tiles.append({
            "key": key, "sym": symbol, "name": name, "value": val, "spec": spec,
            "lo": lo, "hi": hi, "side": side, "se": _num(se.get(key)),
            "spark": sparkline(series, bound=(hi if side == "upper" else lo if side == "lower" else None)),
            "n_hist": len(series), "hist_first": hist[0]["session"] if hist else None,
            "hist_last": hist[-1]["session"] if hist else None,
            "hist_min": min(series) if series else None, "hist_max": max(series) if series else None,
        })
    d["param_tiles"] = tiles

    # OOS RMSE by expiry, with the in-sample RMSE as the reference line.
    rows = oos_rows_of(cal)
    d["oos_rows"] = rows
    d["n_oos"] = (cal.get("oos") or {}).get("n_spx_oos")
    if rows:
        cats = [f"{r['expiry'][2:]} · {_fmt(r['T_days'], '.0f')}d" if r["T_days"] is not None else r["expiry"]
                for r in rows]
        marks = {"y": [{"value": d["spx_rmse"], "label": f"in-sample {_fmt(d['spx_rmse'], '.2f')}",
                        "color": "info"}]} if d.get("spx_rmse") is not None else None
        d["charts"]["oos"] = charts.xy(
            "chart-oos", "Out-of-sample SPX IV RMSE by expiry",
            x={"type": "category", "categories": cats, "name": "Expiry · days to expiry", "interval": 0},
            y={"name": "RMSE (vol pts)", "digits": 1, "tipDigits": 2, "zero": True},
            series=[{"name": "OOS RMSE", "type": "bar", "color": model_colour("heston_v2"),
                     "data": [r["rmse"] for r in rows], "fill": 0.55}],
            xs=cats, marks=marks, csv=f"oos_rmse_by_expiry_{session}.csv",
            grid={"bottom": 64},
            describe=(f"{len(rows)} expiries; RMSE from {_fmt(min(r['rmse'] for r in rows), '.2f')} to "
                      f"{_fmt(max(r['rmse'] for r in rows), '.2f')} vol pts against "
                      f"{_fmt(d.get('spx_rmse'), '.2f')} in sample."),
            table=charts.table("Out-of-sample RMSE by expiry (vol pts)",
                               ["Expiry", "Days#", "RMSE (vol pts)#", "Anchor expiry"],
                               [[r["expiry"], _fmt(r["T_days"], ".0f"), _fmt(r["rmse"], ".2f"),
                                 "yes" if r["anchor"] else "no"] for r in rows]))

    # VIX leg: Cboe spot indices against the model's index values.
    vix = [v for v in cal.get("vix_index") or [] if _num(v.get("market")) is not None]
    d["vix_rows"] = []
    for vrow in vix:
        model = _num(vrow.get("model"))
        d["vix_rows"].append({"ticker": vrow["ticker"], "tenor": int(vrow["tenor_days"]),
                              "market": float(vrow["market"]), "model": model,
                              "diff": (model - float(vrow["market"])) if model is not None else None})
    if vix:
        d["charts"]["vix"] = charts.xy(
            "chart-vixleg", "VIX index term structure: market and the Heston fit",
            x={"type": "value", "name": "Tenor (calendar days)", "digits": 0, "min": 0, "zero": True},
            y={"name": "Index level (vol pts)", "digits": 1, "tipDigits": 2},
            series=[{"name": "Cboe index (market)", "type": "line", "color": "headline", "symbol": "circle",
                     "size": 7, "width": 1.5, "legend": True,
                     "data": [[r["tenor"], r["market"]] for r in d["vix_rows"]]},
                    {"name": "Heston model", "type": "line", "color": model_colour("heston_v2"),
                     "symbol": "diamond", "size": 7,
                     "dash": True, "legend": True,
                     "data": [[r["tenor"], r["model"]] for r in d["vix_rows"] if r["model"] is not None]}],
            tooltip="axis", csv=f"vix_leg_{session}.csv",
            describe="Market VIX-family indices and the fitted model's values at 9, 30, 93 and 184 days.",
            table=charts.table("VIX indices: market vs model (index points)",
                               ["Index", "Tenor (days)#", "Market#", "Model#", "Model − market#"],
                               [[r["ticker"], r["tenor"], _fmt(r["market"], ".2f"), _fmt(r["model"], ".2f"),
                                 signed(r["diff"], ".2f")] for r in d["vix_rows"]]))

    if d.get("mat_smiles"):
        sms = d["mat_smiles"]
        d["charts"]["model_smile"] = charts.xy(
            "chart-model-smile", f"Heston model smile of the {session} fit, one curve per maturity",
            x={"type": "value", "name": "ln(K/F)", "digits": 2, "signed": True},
            y={"name": "Implied vol (%)", "digits": 0, "tipDigits": 2},
            series=[{"name": f"{s_['T']}d", "type": "line", "color": charts.color_cycle(i), "width": 1.5,
                     "legend": True, "data": [[k, v] for k, v in zip(s_["lm"], s_["iv"])]}
                    for i, s_ in enumerate(sms)],
            tooltip="item", csv=f"model_smile_{session}.csv",
            describe=("Model implied vols at the fitted parameters for 14 to 365 days; model curves, "
                      "not market data."),
            table=charts.table("Model implied vol (%) at the lowest and highest log-moneyness plotted",
                               ["Maturity", "Lowest ln(K/F)#", "IV#", "Highest ln(K/F)#", "IV#"],
                               [[f"{s_['T']}d", _fmt(s_["lm"][0], "+.3f"), _fmt(s_["iv"][0], ".2f"),
                                 _fmt(s_["lm"][-1], "+.3f"), _fmt(s_["iv"][-1], ".2f")]
                                for s_ in sms if s_["lm"]]))

    # Smile small multiples: one panel per anchor expiry. Market quotes and the
    # model's IVs at those quotes come from the surface residual table when it
    # exists for this session (the Heston rows; its in_sample flag marks the
    # calibration anchors); otherwise the panel shows the model curve only.
    surf = artifacts.load_surface(DATA, session) if session in artifacts.surface_sessions(DATA) else \
        artifacts.absent("surface", DATA / "surface" / str(session) / "residuals.parquet", data_dir=DATA)
    d["smile_source"] = {k: v for k, v in surf.items() if k != "residuals"}
    res = None
    if surf.get("state") == "ok":
        hkey = heston_model_key(surf["models"])
        if hkey is not None:
            res = surf["residuals"][surf["residuals"]["model"] == hkey]
            d["smile_model"] = hkey
    for a in cal.get("anchors_by_expiry") or []:
        exp = str(a.get("expiry"))
        T_days = _num(a.get("T_days"))
        if T_days is None:
            continue
        series, tbl_rows = [], []
        if res is not None and (res["expiry"] == exp).any():
            q = res[res["expiry"] == exp].sort_values("k")
            ins = q["in_sample"] if "in_sample" in q.columns else pd.Series(False, index=q.index)
            if {"bid_iv", "ask_iv"} <= set(q.columns):
                series.append({"name": "Bid–ask", "type": "errorbar", "color": "muted", "legend": True,
                               "data": [[round(r.k, 3), round(r.bid_iv * 100, 2), round(r.ask_iv * 100, 2)]
                                        for r in q.itertuples()]})
            # the out-of-sample mids muted and small, the fit's anchors as large
            # diamonds drawn above them in their own colour (they looked the same)
            series.append({"name": "Mid (out of sample)", "type": "scatter", "color": "muted",
                           "hollow": True, "size": 4, "opacity": 0.6, "legend": True,
                           "data": [[round(r.k, 3), round(r.mid_iv * 100, 2)]
                                    for r, i in zip(q.itertuples(), ins) if not i]})
            if ins.any():
                series.append({"name": "Mid (fit anchor)", "type": "scatter", "color": ANCHOR_COLOUR,
                               "symbol": "diamond", "size": 9, "z": 4, "legend": True,
                               "data": [[round(r.k, 3), round(r.mid_iv * 100, 2)]
                                        for r, i in zip(q.itertuples(), ins) if i]})
            series.append({"name": "Heston", "type": "line", "color": model_colour("heston_v2"), "width": 1.6,
                           "legend": True,
                           "data": [[round(float(k), 3), round(float(v) * 100, 2)]
                                    for k, v in zip(q["k"], q["model_iv"]) if _num(v) is not None]})
            # the table behind the chart: one row per ln(K/F) bucket (the CSV has every quote)
            qb = q.assign(kb=k_bucket(q["k"]), ins=ins.values)
            for b, g in qb.groupby("kb"):
                tbl_rows.append([k_bucket_label(int(b)), len(g), int(g["ins"].sum()),
                                 _fmt(g["mid_iv"].median() * 100, ".2f"),
                                 _fmt(g["model_iv"].median() * 100, ".2f"),
                                 signed(float(g["error_hs"].median()), ".1f"),
                                 _fmt(g["inside_spread"].astype(bool).mean() * 100, ".0f")])
            source = "quotes"
            inside = float(q["inside_spread"].astype(bool).mean() * 100.0)
            note = (f"{len(q)} quotes, {int(ins.sum())} of them fit anchors; Heston inside the bid–ask on "
                    f"{_fmt(inside, '.0f')}%, median |model − mid| "
                    f"{_fmt(float(q['error_hs'].abs().median()), '.1f')} half-spreads")
        else:
            sm = heston_model_smile(p, maturities_days=(T_days,))[0]
            series.append({"name": "Heston", "type": "line", "color": model_colour("heston_v2"), "width": 1.6,
                           "data": [[k, v * 100] for k, v in zip(sm["lm"], sm["iv"])]})
            tbl_rows = [[_fmt(k, "+.3f"), _fmt(v * 100, ".2f")] for k, v in zip(sm["lm"], sm["iv"])]
            source = "model"
            note = None
        d["smiles"].append({
            "expiry": exp, "T_days": T_days, "n": a.get("n"), "source": source, "note": note,
            "spec": charts.xy(
                f"chart-smile-{exp}", f"Smile {exp} ({_fmt(T_days, '.0f')} days)",
                x={"type": "value", "name": "ln(K/F)", "digits": 2, "signed": True, "label": "k"},
                y={"name": "IV (%)", "digits": 0, "tipDigits": 2},
                series=series, tooltip="item", csv=f"smile_{exp}.csv",
                grid={"left": 40, "top": 10, "bottom": 34},
                describe=(("Market bid–ask quotes, mids and the Heston IVs at the quoted strikes"
                           if source == "quotes"
                           else "Heston model smile only (market quotes need the surface residual table)")
                          + f" for the {exp} expiry."),
                table=(charts.table(f"{exp}: quotes by ln(K/F) bucket (medians; IV in %)",
                                    ["ln(K/F)", "Quotes#", "Anchors#", "Mid IV#", "Heston IV#",
                                     "(Heston − mid) / half-spread#", "Inside the bid–ask (%)#"], tbl_rows)
                       if source == "quotes" else
                       charts.table(f"{exp}: Heston model implied vol (%) by ln(K/F)",
                                    ["ln(K/F)#", "Heston IV#"], tbl_rows))),
        })


# ── Risk page extras ─────────────────────────────────────────────────────────
GREEK_TABS = (
    # key, tab label, column, scale ('seq' | 'div'), cell format, unit note
    ("iv", "IV", "iv", "seq", ".1f", "implied vol, %"),
    ("vega", "Vega", "vega", "seq", ",.1f", "per unit index, per 1.00 vol"),
    ("vanna", "Vanna", "vanna", "div", "+.3f", "per unit index, per 1.00 vol"),
    ("delta", "OTM delta", "delta", "div", "+.2f",
     ("forward delta of the out-of-the-money option: "
      "N(d₁) for the call at K ≥ F, N(d₁) − 1 for the put at K < F")),
)


def otm_forward_delta(call_spot_delta: Any, q: Any, T: Any, log_moneyness: Any) -> float | None:
    """Forward delta of the out-of-the-money option at a Greeks-surface node.

    greeks_surface.parquet stores the call's spot delta e^{−qT}·N(d₁) at every
    node (risk_monitor: "call delta"). The desk quotes the OTM option's forward
    delta: N(d₁) for the call when K ≥ F, N(d₁) − 1 for the put when K < F
    (put–call parity), so a 30%-OTM put reads about −0.01, never +0.99.
    """
    d, q_, t_, k_ = _num(call_spot_delta), _num(q), _num(T), _num(log_moneyness)
    if d is None or k_ is None:
        return None
    nd1 = d * float(np.exp((q_ or 0.0) * (t_ or 0.0)))       # e^{−qT}N(d₁) → N(d₁)
    return nd1 - 1.0 if k_ < 0 else nd1


def risk_extras(d: dict, gdf: pd.DataFrame | None) -> None:
    """Greek grids, the scenario grid and the 2020 hedged-straddle P&L explain."""
    if gdf is not None and len(gdf):
        maturities = sorted(int(t) for t in gdf["T_days"].unique())
        cols = sorted(float(x) for x in gdf["log_moneyness"].round(4).unique())
        cellmap = {(int(r.T_days), round(float(r.log_moneyness), 4)): r for r in gdf.itertuples()}
        grids = []
        for key, label, col, scale, fmt, unit in GREEK_TABS:
            if key == "delta":
                vals = pd.Series([otm_forward_delta(r.delta, getattr(r, "q", None), getattr(r, "T", None),
                                                    r.log_moneyness) for r in gdf.itertuples()],
                                 index=gdf.index, dtype=float)
            else:
                vals = gdf[col].astype(float) * (100.0 if key == "iv" else 1.0)
            lo, hi = float(vals.min()), float(vals.max())
            lim = float(vals.abs().max()) or 1.0
            rows = []
            for T in maturities:
                cells = []
                for c in cols:
                    r = cellmap.get((T, round(c, 4)))
                    if r is None:
                        cells.append(cell("", None,
                                          detail=f"T {T}d · ln(K/F) {_fmt(c, '+.3f')}: no stored Greeks"))
                        continue
                    v = float(vals.loc[r.Index])
                    bg = seq_colour(v, lo, hi) if scale == "seq" else div_colour(v, lim)
                    det = (f"T {T}d · ln(K/F) {_fmt(c, '+.3f')} · K {_fmt(r.K, ',.0f')} · "
                           f"IV {_fmt(cast(float, r.iv) * 100, '.1f')}% · {label} "
                           f"{signed(v, fmt.lstrip('+')) if fmt.startswith('+') else _fmt(v, fmt)}"
                           + ((" (put)" if c < 0 else " (call)") if key == "delta" else ""))
                    # a signed value that rounds to zero carries no sign ('0.00', not '−0.00')
                    txt = signed(v, fmt.lstrip("+")) if fmt.startswith("+") else _fmt(v, fmt)
                    cells.append(cell(txt, bg, detail=det))
                rows.append({"label": f"{T}d", "cells": cells})
            grids.append({"key": key, "label": label, "unit": unit, "scale": scale, "rows": rows,
                          "cols": [_fmt(c, "+.2f") for c in cols], "lo": lo, "hi": hi, "lim": lim,
                          "fmt": fmt})
        d["greek_grids"] = grids

    if d.get("qv_labels"):
        d["charts"]["qv"] = charts.xy(
            "chart-qv", "Quadratic-variation convexity by maturity (×10⁴)",
            x={"type": "category", "categories": d["qv_labels"], "name": "Maturity", "interval": 0},
            y={"name": "Var[∫v dt] × 10⁴", "digits": 1, "tipDigits": 3, "zero": True},
            series=[{"name": "QV convexity × 10⁴", "type": "bar", "color": "cat0", "fill": 0.55,
                     "data": d["qv_values"]}],
            xs=d["qv_labels"], csv="qv_convexity.csv",
            describe=("One value per maturity under the surface's Heston parameters; it grows with "
                      "the horizon."),
            table=charts.table("QV convexity × 10⁴ by maturity", ["Maturity", "QV convexity × 10⁴#"],
                               [[lab, _fmt(v, ".3f")] for lab, v in zip(d["qv_labels"], d["qv_values"])]))
    if d.get("vanna_labels"):
        d["charts"]["vanna"] = charts.xy(
            "chart-vanna", "Mean vanna by maturity (∂²V/∂S∂σ)",
            x={"type": "category", "categories": d["vanna_labels"], "name": "Maturity", "interval": 0},
            y={"name": "Mean vanna", "digits": 2, "tipDigits": 4, "signed": True},
            series=[{"name": "Mean vanna", "type": "bar", "signed": True, "data": d["vanna_values"]}],
            xs=d["vanna_labels"], csv="vanna_by_maturity.csv",
            describe="Black–Scholes vanna averaged over the strikes stored at each maturity.",
            table=charts.table("Mean vanna by maturity (per unit index, per 1.00 vol)",
                               ["Maturity", "Mean vanna#"],
                               [[lab, signed(v, ".4f")]
                                for lab, v in zip(d["vanna_labels"], d["vanna_values"])]))
    if d.get("scatter_lm"):
        ur = d.get("unstable_rule") or {}
        thr = ur.get("threshold") if ur.get("threshold") is not None else 2.0
        mats = sorted(set(d["scatter_T"]))
        name = "Vomma P&L ratio" if ur.get("kind") == "pnl_ratio" else "|Vomma z-score|"
        d["charts"]["scatter"] = charts.xy(
            "chart-flag", f"{name} against ln(K/F), all maturities",
            x={"type": "value", "name": "ln(K/F)", "digits": 2, "signed": True},
            y={"name": name, "digits": 1, "tipDigits": 3, "min": 0},
            series=[{"name": f"{T}d", "type": "scatter", "color": charts.color_cycle(i), "size": 6,
                     "legend": True,
                     "data": [[x, z] for x, z, t in zip(d["scatter_lm"], d["scatter_z"], d["scatter_T"])
                              if t == T]}
                    for i, T in enumerate(mats)],
            tooltip="item", csv="vomma_flag_metric.csv",
            marks={"y": [{"value": thr, "label": f"flag {_fmt(thr, '.2f')}", "color": "breach"}]},
            describe=(f"{len(d['scatter_lm'])} nodes; {d.get('n_unstable')} above the flag threshold "
                      f"{_fmt(thr, '.2f')}."),
            table=charts.table(f"{name} by node (largest ten)", ["Maturity", "ln(K/F)#", name + "#"],
                               [[f"{t}d", _fmt(x, "+.3f"), _fmt(z, ".3f")] for x, z, t in
                                sorted(zip(d["scatter_lm"], d["scatter_z"], d["scatter_T"]),
                                       key=lambda r: -r[1])[:10]]))

    d["scenario"] = artifacts.load_scenario_grid(DATA)
    d["scenario_more"] = []
    d["cash_greeks"] = None
    if d["scenario"].get("state") == "ok":
        g_all = d["scenario"]["grid"]
        d["cash_greeks"] = reference_cash_greeks(d["scenario"].get("provenance") or {})
        lim = float(g_all["pnl"].abs().max()) or 1.0      # one colour scale for every dynamics

        def heat(g: pd.DataFrame) -> dict:
            spots = sorted(g["spot_shock_pct"].unique())
            vols = sorted(g["vol_shock_vp"].unique())
            lut = {(float(a), float(b)): float(c) for a, b, c in
                   zip(g["spot_shock_pct"].to_numpy(float), g["vol_shock_vp"].to_numpy(float),
                       g["pnl"].to_numpy(float))}
            def one(s: float, v: float) -> dict:
                pnl_ = lut.get((float(s), float(v)))
                c = cell(signed(pnl_, ",.0f"), pnl_colour(pnl_, lim),
                         detail=(f"spot {signed(s, '.0f', '%')}, vol {signed(v, '.0f', ' vol pts')}: "
                                 f"P&L {signed(pnl_, ',.0f')} idx pts"))
                # the same cell in dollars for one SPX contract (×100), shown by the units toggle
                usd = None if pnl_ is None else pnl_ * SPX_MULTIPLIER
                c["alt"] = _usd(usd, signed=True)
                c["alt_detail"] = (f"spot {signed(s, '.0f', '%')}, vol {signed(v, '.0f', ' vol pts')}: "
                                   f"P&L {_usd(usd, signed=True)} per SPX contract")
                return c
            return {
                "cols": [signed(s, ".0f", "%") for s in spots],
                "rows": [{"label": signed(v, ".0f", " vol pts"), "cells": [one(s, v) for s in spots]}
                         for v in vols]}

        g = g_all
        if "dynamics" in g_all.columns:
            dyns = sorted(g_all["dynamics"].astype(str).unique())
            dyn = "sticky_strike" if "sticky_strike" in dyns else dyns[0]
            g = g_all[g_all["dynamics"].astype(str) == dyn]
            d["scenario"]["dynamics_shown"], d["scenario"]["dynamics_all"] = dyn, dyns
            # the other dynamics (sticky-moneyness: the smile moves with the forward)
            d["scenario_more"] = [{"dynamics": x, "heat": heat(g_all[g_all["dynamics"].astype(str) == x])}
                                  for x in dyns if x != dyn]
        d["scenario_heat"] = heat(g)

    # C7: daily-hedged 1-year ATM straddle through 2020, Greek P&L explain.
    try:
        h = pd.read_parquet(DATA / "backtest" / "delta_hedge_2020.parquet")
    except Exception as e:
        _record_error(d, "C7 hedge simulation (delta_hedge_2020.parquet)", e)
        return
    comps = [("pnl_gamma", "Gamma", "cat0"), ("pnl_vega_a", "Vega", "cat1"), ("pnl_theta", "Theta", "cat2"),
             ("pnl_vanna", "Vanna", "cat4"), ("pnl_volga", "Volga", "cat3"),
             ("pnl_financing", "Financing", "muted"), ("pnl_residual_a", "Residual", "cat5")]
    need = {"pnl_total"} | {c for c, *_ in comps}
    missing = need - set(h.columns)
    if missing:
        _record_error(d, "C7 hedge simulation (delta_hedge_2020.parquet)",
                      ValueError("missing columns " + ", ".join(sorted(missing))))
        return
    h = h.copy()
    h.index = pd.to_datetime(h.index)
    days = h.dropna(subset=["pnl_total"])
    explained = days[[c for c, *_ in comps if c != "pnl_residual_a"]].sum(axis=1)
    resid = days["pnl_total"] - explained
    if (resid - days["pnl_residual_a"]).abs().max() > 1e-6 * max(1.0, days["pnl_total"].abs().max()):
        _record_error(d, "C7 hedge simulation (delta_hedge_2020.parquet)",
                      ValueError("pnl_residual_a is not total minus the Greek terms"))
        return
    tot = days["pnl_total"]
    xs = [t.strftime("%Y-%m-%d") for t in h.index]
    series = [{"name": name, "type": "line", "color": color, "width": 1.3, "legend": True,
               "data": [round(float(v), 4) for v in h[c].fillna(0.0).cumsum()]} for c, name, color in comps]
    series.insert(0, {"name": "Total", "type": "line", "color": "headline", "width": 2.2, "legend": True,
                      "data": [round(float(v), 4) for v in h["pnl_total"].fillna(0.0).cumsum()]})
    d["hedge"] = {
        "n": len(days), "start": str(h.index[0].date()), "end": str(h.index[-1].date()),
        "explained_share": float(1.0 - resid.var(ddof=1) / tot.var(ddof=1)),
        "rms_total": float(np.sqrt((tot ** 2).mean())), "rms_resid": float(np.sqrt((resid ** 2).mean())),
        # the second-order Taylor check: the residual's sd against the daily P&L's
        "resid_sd": float(resid.std(ddof=1)), "total_sd": float(tot.std(ddof=1)),
        "cum": {name: float(h[c].fillna(0.0).sum()) for c, name, _ in comps},
        "cum_total": float(tot.sum()),
        "K": _num(h["S"].iloc[0]),
        "unstable_days": int(h["is_unstable"].astype(bool).sum()) if "is_unstable" in h.columns else None,
        # where the expansion misses: the largest daily residuals and the share of
        # the residual's sum of squares inside the Feb–Apr 2020 stress window
        "top_resid": [{"date": str(cast(pd.Timestamp, t).date()), "value": float(v)} for t, v in
                      days["pnl_residual_a"].reindex(days["pnl_residual_a"].abs().sort_values(ascending=False).index[:3]).items()],
        "covid_resid_share": float((days["pnl_residual_a"].loc["2020-02-20":"2020-04-30"] ** 2).sum()
                                   / max((days["pnl_residual_a"] ** 2).sum(), 1e-300)),
    }
    if "is_unstable" in h.columns and h["is_unstable"].astype(bool).any():
        flagged = h.index[h["is_unstable"].astype(bool)]
        d["hedge"]["flag_first"] = str(flagged.min().date())
        d["hedge"]["flag_last"] = str(flagged.max().date())
    d["charts"]["hedge"] = charts.xy(
        "chart-hedge", "Cumulative P&L explain of a delta-hedged 1-year ATM straddle, 2020",
        x={"type": "time", "name": "", "csv": "date"},
        y={"name": "Cumulative P&L (index points)", "digits": 0, "tipDigits": 2, "signed": True},
        series=series, xs=xs, zoom=True, csv="hedge_attribution_2020.csv",
        marks={"y": [{"value": 0, "label": "", "color": "muted", "dash": False}]},
        describe=(f"{len(days)} daily rebalances from {d['hedge']['start']} to {d['hedge']['end']}; total "
                  f"{signed(d['hedge']['cum_total'], ',.1f')} index points, residual "
                  f"{signed(d['hedge']['cum']['Residual'], ',.2f')}."),
        table=charts.table("Cumulative P&L by Greek, 2020 (index points per straddle)",
                           ["Component", "Cumulative P&L#"],
                           [["Total", signed(d["hedge"]["cum_total"], ",.2f")]]
                           + [[n, signed(v, ",.2f")] for n, v in d["hedge"]["cum"].items()]))


def reference_cash_greeks(prov: dict) -> dict | None:
    """Cash Greeks of the scenario grid's reference book (one 30-day SPX straddle
    struck at the forward, Black-76 at the fit's ATM-forward vol), per SPX
    contract, from the grid's provenance (desk_format.cash_greeks)."""
    raw = {k: _num(prov.get(k)) for k in ("S", "forward", "strike", "T", "iv_atm_forward", "discount")}
    vals = {k: v for k, v in raw.items() if v is not None and v > 0}
    if len(vals) != len(raw):
        return None
    g = cash_greeks(vals["S"], vals["forward"], vals["strike"], vals["T"], vals["iv_atm_forward"],
                    vals["discount"])
    as_of = _date_str(prov.get("as_of_date"))
    age = session_age(as_of, _latest_market_session())
    return {**g, **vals, "tenor_days": _num(prov.get("tenor_days")), "as_of": as_of, "age": age,
            "age_level": stale_level(age), "multiplier": SPX_MULTIPLIER}


# ── Research page extras ─────────────────────────────────────────────────────
MONTHS = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")


def _runs(mask: pd.Series) -> list[tuple]:
    """(first, last) index labels of each run of True values."""
    out: list[tuple] = []
    start: Any = None
    prev: Any = None
    for t, v in mask.items():
        if v and start is None:
            start = t
        if not v and start is not None:
            out.append((start, prev))
            start = None
        prev = t
    if start is not None:
        out.append((start, prev))
    return out


def research_extras(d: dict, eq: pd.DataFrame | None) -> None:
    """Tear-sheet series: equity with regime shading, drawdown, rolling Sharpe,
    monthly returns grid, sleeve bars, annual and half-year bars, inference."""
    if eq is not None:
        xs = d["eq_dates"]
        series = [
            {"name": "Portfolio", "type": "line", "color": "headline", "width": 2.2, "legend": True,
             "data": d["eq_nav"]},
            {"name": "S1C contrarian", "type": "line", "color": "cat0", "width": 1.3, "legend": True,
             "data": d["eq_nav_s1c"]},
            {"name": S3_SHORT, "type": "line", "color": "cat1", "width": 1.3, "legend": True,
             "data": d["eq_nav_s3"]},
            {"name": "S4 VRP", "type": "line", "color": "cat4", "width": 1.3, "legend": True,
             "data": d["eq_nav_s4"]},
            {"name": "S1 (reference)", "type": "line", "color": "muted", "width": 1.2, "dash": True,
             "legend": True, "hidden": True, "data": d["eq_nav_s1"]},
        ]
        if "nav_tbill" in eq.columns and eq["nav_tbill"].notna().any():
            tb = eq["nav_tbill"]
            series.append({"name": "T-bill (collateral)", "type": "line", "color": "cat2", "width": 1.2,
                           "dash": True, "legend": True,
                           "data": ((tb / float(tb.iloc[0]) - 1.0) * 100.0).round(3).tolist()})
        for s_ in series:                 # ~1,800 daily points a line: drawn with LTTB downsampling
            s_["sampling"] = "lttb"
        r2 = (eq["regime"] == 2) if "regime" in eq.columns else pd.Series(False, index=eq.index)
        areas = [{"from": a.strftime("%Y-%m-%d"), "to": b.strftime("%Y-%m-%d"), "color": "watch",
                  "alpha": 0.07} for a, b in _runs(r2)]
        d["r2_share_bt"] = float(r2.mean())
        d["charts"]["equity"] = charts.xy(
            "chart-equity", "Cumulative return from $1M: portfolio, sleeves and T-bills",
            x={"type": "time", "csv": "date"},
            y={"name": "Cumulative return (%)", "digits": 0, "tipDigits": 2, "signed": True, "tickUnit": "%",
               "unit": "%"},
            series=series, xs=xs, zoom=True, zoom_presets=(("1Y", 1), ("3Y", 3), ("All", 0)),
            marks={"areas": areas, "y": [{"value": 0, "label": "", "color": "muted", "dash": False}]},
            csv="equity_curves.csv",
            describe=(f"Daily from {d['start_date']} to {d['end_date']}; R2 sessions (shaded) are "
                      f"{_fmt(d['r2_share_bt'] * 100, '.0f')}% of the window."),
            table=charts.table(f"Cumulative return from {d['start_date']} to {d['end_date']}",
                               ["Series", "Final#", "Lowest#", "Highest#"],
                               [[r["name"], _pct_prose(r["final"]), _pct_prose(r["low"]),
                                 _pct_prose(r["high"])] for r in d.get("eq_summary") or []]))
        nav = eq["nav"]
        dd = (nav / nav.cummax() - 1.0) * 100.0
        d["charts"]["drawdown"] = charts.xy(
            "chart-dd", "Portfolio drawdown from its running peak",
            x={"type": "time", "csv": "date"},
            y={"name": "Drawdown (%)", "digits": 0, "tipDigits": 2, "tickUnit": "%", "unit": "%", "max": 0},
            series=[{"name": "Drawdown", "type": "line", "color": "neg", "width": 1.2, "area": True,
                     "sampling": "lttb", "data": dd.round(3).tolist()}],
            xs_from="chart-equity", zoom=False, csv="drawdown.csv", grid={"top": 10, "bottom": 30},
            describe=(f"Deepest drawdown {_fmt(float(dd.min()), '.1f')}% on "
                      f"{cast(pd.Timestamp, dd.idxmin()).date()}."),
            table=charts.table("Drawdown", ["Statistic", "Value"],
                               [["Deepest (%)", _fmt(float(dd.min()), ".2f")],
                                ["Date of the trough", str(cast(pd.Timestamp, dd.idxmin()).date())]]))
        # rolling 126-session Sharpe (rf = 0, the engine's convention)
        ret = nav.pct_change()
        roll = (ret.rolling(126).mean() / ret.rolling(126).std(ddof=1)) * np.sqrt(252)
        roll = roll.where(ret.rolling(126).std(ddof=1) > 1e-12)
        d["charts"]["rolling"] = charts.xy(
            "chart-rolling", "Rolling 126-session Sharpe (rf = 0)",
            x={"type": "time", "csv": "date"},
            y={"name": "Sharpe (annualised)", "digits": 1, "tipDigits": 2, "signed": True},
            series=[{"name": "Rolling Sharpe", "type": "line", "color": "cat0", "width": 1.3,
                     "sampling": "lttb",
                     "data": [None if not np.isfinite(v) else round(float(v), 3) for v in roll]}],
            xs_from="chart-equity", csv="rolling_sharpe_126.csv", grid={"top": 10, "bottom": 30},
            marks={"y": [{"value": 0, "label": "", "color": "muted", "dash": False}]},
            describe="Annualised Sharpe of the trailing 126 daily returns; blank where the NAV was flat.",
            table=charts.table("Rolling Sharpe", ["Statistic", "Value#"],
                               [["Latest",
                                 _fmt(_num(roll.dropna().iloc[-1]) if roll.notna().any() else None, "+.2f")],
                                ["Lowest", _fmt(_num(roll.min()), "+.2f")],
                                ["Highest", _fmt(_num(roll.max()), "+.2f")]]))
        # monthly returns grid (against the prior month-end NAV)
        nav_idx = cast(pd.DatetimeIndex, nav.index)
        month_end = nav.groupby([nav_idx.year, nav_idx.month]).last()
        base = float(nav.iloc[0])
        grid: dict = {}
        prev = base
        for ym, v in month_end.items():
            yr, mo = cast(tuple[int, int], ym)
            grid.setdefault(int(yr), {})[int(mo)] = float(v) / prev - 1.0
            prev = float(v)
        lim = max((abs(v) for m in grid.values() for v in m.values()), default=0.01) or 0.01
        d["monthly"] = {"cols": [*MONTHS, "Year"], "rows": []}
        ann = {a["year"]: a["ret_pct"] / 100.0 for a in d.get("ann") or []}
        part = {a["year"]: a for a in d.get("ann") or [] if a.get("partial")}
        for yr in sorted(grid):
            cells = []
            for mo in range(1, 13):
                v = grid[yr].get(mo)
                cells.append(cell(signed(v * 100 if v is not None else None, ".1f"), pnl_colour(v, lim),
                                  detail=(f"{MONTHS[mo - 1]} {yr}: "
                                          f"{signed(v * 100 if v is not None else None, '.2f', '%')}")))
            y = ann.get(yr)
            py = part.get(yr)
            yl = f"{yr} YTD" if py else str(yr)
            span = f" ({py['first']} → {py['last']})" if py else ""
            cells.append(cell(signed(y * 100 if y is not None else None, ".1f"),
                              pnl_colour(y, max(lim * 3, 0.01)),
                              detail=f"{yl}{span}: {signed(y * 100 if y is not None else None, '.2f', '%')}"))
            d["monthly"]["rows"].append({"label": yl, "cells": cells})
        d["monthly"]["lim"] = lim
    # sleeves, annual returns, half-year Sharpe as signed bars
    if d.get("signal_labels"):
        d["charts"]["sleeves"] = charts.xy(
            "chart-sleeves", "Total P&L by portfolio sleeve ($000s)",
            x={"type": "category", "categories": d["signal_labels"], "interval": 0},
            y={"name": "P&L ($000s)", "digits": 0, "tipDigits": 1, "signed": True},
            series=[{"name": "P&L ($000s)", "type": "bar", "signed": True, "data": d["signal_pnl"]}],
            xs=d["signal_labels"], horizontal=True, tooltip="axis", csv="sleeve_pnl.csv",
            grid={"left": 110, "bottom": 30},
            describe="Mark-to-model P&L of the three portfolio sleeves over the backtest window.",
            table=charts.table("Total P&L by sleeve ($000s)", ["Sleeve", "P&L ($000s)#"],
                               [[lab, signed(v, ",.1f")]
                                for lab, v in zip(d["signal_labels"], d["signal_pnl"])]))
    if d.get("ann"):
        labels = [f"{a['year']}{' YTD' if a['partial'] else ''}" for a in d["ann"]]
        d["charts"]["annual"] = charts.xy(
            "chart-annual", "Annual return against the prior year-end NAV",
            x={"type": "category", "categories": labels, "interval": 0, "fit": True},
            y={"name": "Return (%)", "digits": 0, "tipDigits": 2, "signed": True, "tickUnit": "%",
               "unit": "%"},
            series=[{"name": "Return", "type": "bar", "signed": True, "data": d["ann_rets"]}],
            xs=labels, csv="annual_returns.csv",
            describe=("Calendar-year returns; they compound to the cumulative return."
                      + (f" {d['ann_partial_note']}." if d.get("ann_partial_note") else "")),
            table=charts.table("Annual return (%)", ["Year", "Return#"],
                               [[lab, signed(v, ".2f", "%")] for lab, v in zip(labels, d["ann_rets"])]))
    if d.get("wf"):
        d["charts"]["halfyear"] = charts.xy(
            "chart-halfyear", "Half-year Sharpe of the single in-sample run (rf = 0)",
            x={"type": "category", "categories": d["wf_periods"], "interval": 0, "fit": True},
            y={"name": "Sharpe", "digits": 0, "tipDigits": 2, "signed": True},
            series=[{"name": "Sharpe", "type": "bar", "signed": True, "data": d["wf_sharpes"]}],
            xs=d["wf_periods"], csv="half_year_sharpe.csv", grid={"bottom": 70},
            describe=f"{d.get('wf_neg')} of {d.get('wf_total')} half-years have a negative Sharpe.",
            table=charts.table("Half-year Sharpe (in-sample, rf = 0)", ["Half-year", "Sharpe#"],
                               [[w, _fmt(v, "+.2f", "flat NAV — none")]
                                for w, v in zip(d["wf_periods"], d["wf_sharpes"])]))
    inference_extras(d, artifacts.load_inference(DATA))
    d["forward"] = artifacts.load_forward_ledger(DATA)
    d["forward_protocol"] = forward_ledger_protocol()
    if d["forward"].get("state") == "ok":
        led = d["forward"].pop("latest")
        d["forward"]["inputs"] = led.get("inputs") or {}
        d["forward"]["rows"] = [{"sleeve": SLEEVE_NAMES.get(sl, sl.upper()), "key": sl,
                                 "target": (led.get("signals") or {}).get(sl, {}).get("straddle_direction"),
                                 **(led.get("summary") or {}).get(sl, {}),
                                 "position": next((ps for ps in led.get("positions") or []
                                                   if ps.get("sleeve") == sl), None)}
                                for sl in ("s1c", "s3", "s4", "s1", "s2")]


def forward_ledger_protocol() -> dict:
    """The forward test's rules as joint_vol_calibration/backtest/forward_ledger.py
    and option_chain.py set them, shown while no entry is registered, and the one
    status sentence the Research panel and the Overview's pipeline row share. The
    job records a session whenever its captured book is live, whatever the fit does
    (scripts/cloud_refresh.record_forward_ledger); entries are written only then."""
    from joint_vol_calibration.backtest import forward_ledger as fl
    from joint_vol_calibration.data import option_chain as oc

    close = oc.SESSION_CLOSE_ET.strftime("%H:%M")
    path = f"{fl.FORWARD_DIR.relative_to(ROOT).as_posix()}/ledger_<session>.json"
    rows = [
        ("Records", (f"Each session, per sleeve: its target (long, short or flat) and the listed "
                     f"PM-settled SPX straddle nearest {fl.TARGET_DAYS} calendar days out (at least "
                     f"{fl.MIN_DAYS}), at the strike nearest the parity forward, sized per "
                     f"${fl.NAV_PER_SLEEVE:,.0f} of sleeve NAV.")),
        ("Prices", (f"The session's captured book, quoted at or after the {close} ET close (an earlier "
                    f"one is refused): bought at the ask, sold at the bid; ${fl.COMMISSION_PER_LEG:.2f} "
                    "per contract per leg, booked separately.")),
        ("Hash chain", (f"One file per session, schema {fl.SCHEMA}, holding the SHA-256 of the previous "
                        "file and of the book it used; its commit time registers the targets before any "
                        "later price exists.")),
        ("Marks", ("Every later captured book at bid, ask and mid; at expiry, |S − K| on the official SPX "
                   "close. A straddle a book cannot price stays unmarked at its last mark, never at a "
                   "made-up price; the backtest's β·VIX mark is recorded beside the mid.")),
        ("Invalidated by", ("An edited or missing registered file (every later hash breaks, and this "
                            "site then refuses to publish), or a book quoted before the close.")),
    ]
    pending = (f"no entry yet: the first is written on the next session whose captured book is live "
               f"(quoted at or after the {close} ET close, at least {oc.MIN_TWO_SIDED_SHARE:.0%} of "
               "near-the-money lines two-sided), whatever the fit does")
    return {"rows": rows, "pending": pending, "path": path,
            "module": "joint_vol_calibration/backtest/forward_ledger.py"}


SPEC_SLEEVES = (("s1c", "portfolio"), ("s3", "portfolio"), ("s4", "portfolio"), ("s1", "reference"),
                ("s2", "reference"))


def _vp(x: float) -> str:
    """A decimal-vol threshold in vol points: 0.02 -> '2', 0.005 -> '0.5'."""
    return _gm(x * 100)


def _gm(x: float) -> str:
    """A constant as written ('g'), with a true minus sign: -0.5 -> '−0.5'."""
    return f"{x:g}".replace("-", MINUS)


def sleeve_spec(trades: Sequence[Any]) -> dict:
    """Research #sleeve-spec: what each sleeve trades, read from the engine and
    signal modules at render time (no constant is typed here) and the side and
    exit counts from the stored trade log."""
    from joint_vol_calibration.backtest import backtest_engine as be
    from joint_vol_calibration.signals import signal_engine as se
    by: dict[str, dict] = {}
    for t in trades:
        sig = str(getattr(t, "signal", ""))
        e = by.setdefault(sig, {"long": 0, "short": 0, "exits": {}})
        e["long" if int(getattr(t, "direction", 0)) > 0 else "short"] += 1
        why = str(getattr(t, "exit_reason", "") or "")
        e["exits"][why] = e["exits"].get(why, 0) + 1
    straddle = (f"{be.STRADDLE_TENOR_DAYS}-calendar-day ATM SPX straddle, delta-hedged daily, marked at "
                "Black–Scholes on β·VIX")
    overlay = " · overlay"
    costs = (f"{be.BID_ASK_VOL_PTS:g} vol pts + ${be.COMMISSION_PER_LEG:.2f} a contract per leg, each way; "
             f"hedge slippage {be.SLIPPAGE_PCT:.2%}")
    r2 = "a move into R2 closes it"
    rows = {
        "s1c": {"name": "S1C · contrarian PDV", "instrument": straddle,
                "entry": (f"PDV forecast > VIX + {_vp(se.S1_ENTRY_THRESHOLD)} vol pts in R0 → short; "
                          f"VIX > PDV + {_vp(se.S1_ENTRY_THRESHOLD)} vol pts in R1 → long"),
                "exit": (f"the spread back inside ±{_vp(se.S1_EXIT_BAND)} vol pts or through zero; {r2}"),
                "hold": se.S1_MAX_HOLD,
                "sizing": (f"tanh(|spread| / {_vp(se.S1_STRENGTH_SCALE)} vol pts) × {se.S1_MAX_KELLY:.0%} of "
                           f"sleeve NAV{overlay}")},
        "s3": {"name": S3_NAME, "instrument": f"SHORT {straddle}; the index leg only",
               "entry": (f"z of VIX / VVIX ({se.S3_LOOKBACK}-session) < {_gm(se.S3_ZSCORE_ENTRY)} "
                         "in R0 or R1"),
               "exit": f"z > {_gm(se.S3_ZSCORE_EXIT)}",
               "hold": se.S3_MAX_HOLD,
               "sizing": (f"min(1, |z| / {_gm(se.S3_STRENGTH_SCALE)}) × {se.S3_MAX_KELLY:.0%} of sleeve "
                          f"NAV{overlay}")},
        "s4": {"name": "S4 · VRP", "instrument": straddle,
               "entry": (f"z_vrp > {_gm(se.S4_Z_ENTRY_SHORT)} and falling, in R1 → short; "
                         f"z_vrp < {_gm(se.S4_Z_ENTRY_LONG)} and rising, in R0 → long"),
               "exit": f"|z_vrp| < {_gm(se.S4_Z_EXIT)}; {r2}",
               "hold": se.S4_MAX_HOLD,
               "sizing": (f"min(1, |z_vrp| / {_gm(se.S4_STRENGTH_SCALE)}) × {se.S4_MAX_KELLY:.0%} of "
                          f"sleeve NAV{overlay}")},
        "s1": {"name": "S1 · PDV short-VRP", "instrument": straddle,
               "entry": (f"PDV forecast > VIX + {_vp(se.S1_ENTRY_THRESHOLD)} vol pts in R0 → long; "
                         f"VIX > PDV + {_vp(se.S1_ENTRY_THRESHOLD)} vol pts in R1 → short"),
               "exit": f"the spread back inside ±{_vp(se.S1_EXIT_BAND)} vol pts or through zero",
               "hold": se.S1_MAX_HOLD,
               "sizing": (f"tanh(|spread| / {_vp(se.S1_STRENGTH_SCALE)} vol pts) × {se.S1_MAX_KELLY:.0%} of "
                          "sleeve NAV")},
        "s2": {"name": "S2 · VIX term structure", "instrument": straddle,
               "entry": (f"z of the VIX3M − VIX slope ({se.S2_LOOKBACK}-session) > "
                         f"{_gm(se.S2_ZSCORE_ENTRY)} in R0 or R1 → short; slope < 0 (backwardation) in R0 or "
                         "R1 → long"),
               "exit": (f"|z| < {_gm(se.S2_ZSCORE_EXIT)}, or VIX moves {se.S2_STOP_LOSS_PCT:.0%} of its "
                        "entry level against the straddle"),
               "hold": se.S2_MAX_HOLD,
               "sizing": (f"min(1, |z| / {_gm(se.S2_STRENGTH_SCALE)}) × {se.S2_MAX_KELLY:.0%} of "
                          "sleeve NAV")},
    }
    exit_words = {"signal": "signal", "expiry": "at expiry, rolled", "open": "open at the end"}
    out = []
    for key, role in SPEC_SLEEVES:
        c = by.get(key) or {"long": 0, "short": 0, "exits": {}}
        sides = [f"{side} {c[side]}" for side in ("short", "long") if c[side]]
        exits = [f"{n} {exit_words.get(w, w)}" for w, n in sorted(c["exits"].items(), key=lambda kv: -kv[1])]
        out.append({"key": key, "role": role, **rows[key], "costs": costs, "n": c["long"] + c["short"],
                    "sides": " / ".join(sides) or "no trades", "exits": ", ".join(exits)})
    cb = by.get("combined") or {"long": 0, "short": 0}
    return {"rows": out,
            "combined": {"long": cb["long"], "short": cb["short"], "n": cb["long"] + cb["short"]},
            "margin": (f"a short straddle is capped at NAV / ({be.MARGIN_RATIO:.0%} × K × "
                       f"{be.CONTRACT_MULT}) contracts, its initial margin"),
            "z_vrp": (f"z_vrp: the {se.S4_LOOKBACK}-session z-score of VRP = VIX − RV₂₀ (annualised), the "
                      "day's premium of implied over realised vol"),
            "constants": {
                "STRADDLE_TENOR_DAYS": be.STRADDLE_TENOR_DAYS, "S1_ENTRY_THRESHOLD": se.S1_ENTRY_THRESHOLD,
                "S1_MAX_HOLD": se.S1_MAX_HOLD, "S3_MAX_HOLD": se.S3_MAX_HOLD,
                "S4_Z_ENTRY_SHORT": se.S4_Z_ENTRY_SHORT, "S4_Z_ENTRY_LONG": se.S4_Z_ENTRY_LONG,
                "S4_Z_EXIT": se.S4_Z_EXIT, "MAX_KELLY": se.S1_MAX_KELLY, "MARGIN_RATIO": be.MARGIN_RATIO,
                "SLIPPAGE_PCT": be.SLIPPAGE_PCT, "S3_PROXY_COST": be.S3_PROXY_COST}}


# a round reference row of the β table, between the backtest's and the newest chain's
VRP_REFERENCE_BETA = 0.80


def market_vol_frame(spx: pd.DataFrame, vix: pd.Series) -> pd.DataFrame:
    """VIX, RV21 trailing and RV21 forward on every session with a VIX close (vol pts)."""
    r = (spx["close_return"] if "close_return" in spx.columns
         else pd.Series(np.log(spx["close"].to_numpy(float)), index=spx.index).diff())
    out = pd.DataFrame({"vix": vix.astype(float)})
    out["rv_trl"] = desk_market.rv_trailing(r).reindex(out.index)
    out["rv_fwd"] = desk_market.rv_forward(r).reindex(out.index)
    return out.dropna(subset=["vix"])


def vix_rv_now(spx: pd.DataFrame, vix: pd.Series) -> dict | None:
    """The newest session's VIX against its trailing RV21 (the Overview line)."""
    f = market_vol_frame(spx, vix).dropna(subset=["rv_trl"])
    if f.empty:
        return None
    last = f.iloc[-1]
    return {"date": str(pd.Timestamp(f.index[-1]).date()), "vix": float(last["vix"]),
            "rv": float(last["rv_trl"]), "spread": float(last["vix"] - last["rv_trl"])}


def _r2(x: Any) -> float | None:
    v = _num(x)
    return None if v is None else round(v, 2)


def vrp_view(spx: pd.DataFrame, vix: pd.Series, start: str | None, end: str | None, beta_bt: float | None,
             newest_chain: dict | None, trades: Sequence[Any]) -> dict:
    """Research #vrp: VIX against trailing and subsequently realised RV21, the ex-post
    premium a β·VIX mark leaves over the backtest window (a description, not P&L),
    calendar-year means at the backtest's β, and the S4 entries on the same axis."""
    f = market_vol_frame(spx, vix)
    xs = [str(pd.Timestamp(t).date()) for t in f.index]
    betas = [(1.0, "VIX itself")]
    if beta_bt is not None:
        betas.append((round(beta_bt, 2), "the backtest's mark"))
    betas.append((VRP_REFERENCE_BETA, "round reference"))
    if newest_chain:
        betas.append((round(float(newest_chain["ratio"]), 2),
                      (f"newest chain's ATM / VIX ({newest_chain['session']}: "
                       f"{_fmt(newest_chain['ratio'], '.3f')})")))
    seen: set = set()
    rows = []
    for b, why in betas:
        if b in seen:
            continue
        seen.add(b)
        st = desk_market.vrp_stats(f["vix"], f["rv_fwd"], b, start, end)
        if st:
            rows.append({**st, "why": why,
                         "backtest": beta_bt is not None and abs(b - round(beta_bt, 2)) < 1e-9})
    full = desk_market.vrp_stats(f["vix"], f["rv_fwd"], 1.0)
    b_bt = round(beta_bt, 2) if beta_bt is not None else None
    years = (desk_market.yearly_means(desk_market.vrp(f["vix"], f["rv_fwd"], b_bt).loc[start:end])
             if b_bt is not None else [])
    now = vix_rv_now(spx, vix)
    fwd_last = f["rv_fwd"].dropna()
    s4 = [t for t in trades if getattr(t, "signal", None) == "s4"]
    vix_at = dict(zip(xs, f["vix"].round(2).tolist()))
    entries: dict[int, list[list]] = {d_: [] for d_ in (-1, 1)}
    for t in s4:
        day = str(pd.Timestamp(t.entry_date).date())
        if day in vix_at and int(getattr(t, "direction", 0)) in entries:
            entries[int(t.direction)].append([day, vix_at[day]])
    view: dict[str, Any] = {"rows": rows, "full": full, "years": years, "now": now, "start": start,
                            "end": end,
            "beta_bt": b_bt, "first": xs[0] if xs else None, "last": xs[-1] if xs else None,
            "fwd_last": str(pd.Timestamp(fwd_last.index[-1]).date()) if len(fwd_last) else None,
            "n_s4": len(s4), "n_s4_short": len(entries[-1]), "n_s4_long": len(entries[1]),
            "charts": {}}
    if not xs:
        return view
    window = ([{"from": start, "to": end, "color": "info", "alpha": 0.07, "label": "backtest window"}]
              if start and end else [])
    yr_all = desk_market.yearly_means(f["vix"])
    v1 = desk_market.vrp(f["vix"], f["rv_fwd"], 1.0)
    yv = desk_market.yearly_means(v1)
    yt = {k: {y["year"]: y["mean"] for y in desk_market.yearly_means(f[k])} for k in ("rv_trl", "rv_fwd")}
    yt["vrp"] = {y["year"]: y["mean"] for y in yv}
    lev_series = [
        {"name": "VIX", "type": "line", "color": "headline", "width": 1.3, "legend": True, "sampling": "lttb",
         "data": [_r2(x) for x in f["vix"]]},
        # not blue or orange: the bars below are signed in those two
        {"name": "RV21 trailing", "type": "line", "color": "cat2", "width": 1.1, "legend": True,
         "sampling": "lttb", "data": [_r2(x) for x in f["rv_trl"]]},
        {"name": "RV21 forward", "type": "line", "color": "cat3", "width": 1.1, "legend": True,
         "sampling": "lttb", "data": [_r2(x) for x in f["rv_fwd"]]}]
    if b_bt is not None:
        lev_series.append({"name": f"{_fmt(b_bt, '.2f')}·VIX (backtest mark)", "type": "line",
                           "color": "muted", "width": 1.1, "dash": True, "legend": True, "sampling": "lttb",
                           "derive": {"a": "VIX", "k": b_bt, "digits": 4}})
    lev_series += [
        {"name": "S4 short entry", "type": "scatter", "color": "cat4", "symbol": "triangle", "rotate": 180,
         "size": 8, "z": 5, "legend": True, "data": entries[-1]},
        {"name": "S4 long entry", "type": "scatter", "color": "cat4", "symbol": "triangle", "size": 8, "z": 5,
         "legend": True, "data": entries[1]}]
    view["charts"]["levels"] = charts.xy(
        "chart-vrp-levels",
        "VIX against trailing and subsequently realised 21-session vol, with the S4 entries",
        x={"type": "time", "csv": "date"},
        y={"name": "Vol (vol pts)", "digits": 0, "tipDigits": 2},
        series=lev_series, xs=xs, zoom=True, zoom_presets=(("1Y", 1), ("3Y", 3), ("All", 0)),
        marks={"areas": window}, csv="vix_rv21.csv", link="vrp", tooltip="axis",
        describe=(f"Daily {xs[0]} to {xs[-1]}: the VIX close, RV21 over the 21 sessions to each date and "
                  f"over the 21 sessions after it (to {view['fwd_last']}); the backtest window is shaded "
                  f"and the {len(s4)} S4 entries ({len(entries[-1])} short, {len(entries[1])} long) are "
                  "marked at the VIX close of their entry session."),
        table=charts.table("Calendar-year means (vol pts; each over the sessions that have the value: the "
                           "forward one stops 21 sessions before the data); the CSV has every session",
                           ["Year", "VIX#", "RV21 trailing#", "RV21 forward#", "VIX − RV21 forward#"],
                           [[y["year"] if not y["partial"] else f"{y['year']} (to {y['last']})",
                             _fmt(y["mean"], ".2f"), _fmt(yt["rv_trl"].get(y["year"]), ".2f"),
                             _fmt(yt["rv_fwd"].get(y["year"]), ".2f"),
                             signed(yt["vrp"].get(y["year"]), ".2f")]
                            for y in yr_all]))
    v1n = v1.dropna()
    pos = {y_: float((g > 0).mean() * 100) for y_, g in v1n.groupby(pd.DatetimeIndex(v1n.index).year)}
    view["charts"]["bars"] = charts.xy(
        "chart-vrp-bars", "Ex-post premium at β = 1: VIX minus the next 21 sessions' realised vol",
        x={"type": "time", "csv": "date"},
        y={"name": "VIX − RV21 fwd (vol pts)", "digits": 0, "tipDigits": 2, "signed": True},
        # the levels chart's VIX − RV21 forward (its data; the CSV and the table carry it)
        series=[{"name": "VIX − RV21 forward", "type": "bar", "signed": True, "barMaxWidth": 3,
                 "derive": {"from": "chart-vrp-levels", "a": "VIX", "b": "RV21 forward"}}],
        xs_from="chart-vrp-levels", zoom="inside", link="vrp", tooltip="axis", grid={"top": 8, "bottom": 30},
        marks={"areas": window, "y": [{"value": 0, "label": "", "color": "muted", "dash": False}]},
        csv="vix_minus_rv21_forward.csv",
        describe=(f"{_fmt(full['n'] if full else None, ',')} sessions with a forward window; positive on "
                  f"{_fmt(full['share_pos'] if full else None, '.1f')}%, mean "
                  f"{signed(full['mean'] if full else None, '.2f')} vol pts. Zooms with the chart above."),
        table=charts.table("VIX − RV21 forward by calendar year (vol pts); the CSV has every session",
                           ["Year", "Mean#", "Above zero (%)#", "Sessions#"],
                           [[y["year"] if not y["partial"] else f"{y['year']} (to {y['last']})",
                             signed(y["mean"], ".2f"), _fmt(pos.get(y["year"]), ".1f"), y["n"]] for y in yv]))
    if years:
        labels = [f"{y['year']} YTD" if y["partial"] else str(y["year"]) for y in years]
        view["charts"]["years"] = charts.xy(
            "chart-vrp-years", f"Mean ex-post premium of the {_fmt(b_bt, '.2f')}·VIX mark by calendar year, "
            "backtest window",
            x={"type": "category", "categories": labels, "name": "", "interval": 0, "fit": True},
            y={"name": f"{_fmt(b_bt, '.2f')}·VIX − RV21 fwd (vol pts)", "digits": 0, "tipDigits": 2,
               "signed": True},
            series=[{"name": f"Mean at β = {_fmt(b_bt, '.2f')}", "type": "bar", "signed": True,
                     "data": [round(y["mean"], 3) for y in years]}],
            xs=labels, csv=f"vrp_beta_{_fmt(b_bt, '.2f')}_by_year.csv", grid={"top": 10, "bottom": 30},
            marks={"y": [{"value": 0, "label": "", "color": "muted", "dash": False}]},
            describe=(f"Calendar-year means of {_fmt(b_bt, '.2f')}·VIX − RV21 forward, {start} → {end}; "
                      + "; ".join(f"{y['year']} YTD: {y['first']} → {y['last']}, {y['n']} sessions"
                                  for y in years if y["partial"])),
            table=charts.table(f"Mean of {_fmt(b_bt, '.2f')}·VIX − RV21 forward by calendar year (vol pts)",
                               ["Year", "Mean#", "Sessions#", "From", "To"],
                               [[lab, signed(y["mean"], ".2f"), y["n"], y["first"], y["last"]]
                                for lab, y in zip(labels, years)]))
    return view


SLEEVE_NAMES = {"portfolio": "Portfolio", "s1": "S1 PDV (ref)", "s1c": "S1C contrarian",
                "s2": "S2 VIX TS (ref)", "s3": S3_NAME, "s4": "S4 VRP",
                "always_short_straddle": "Always-short straddle"}


def inference_extras(d: dict, inf: dict) -> None:
    """Research-page inference tables from data_store/backtest/inference (inference_summary/1)."""
    d["inference"] = {k: v for k, v in inf.items() if k not in ("summary", "benchmarks", "forecast")}
    d["sharpe_ci"] = artifacts.bootstrap_ci(inf, "portfolio", "sharpe")
    if inf.get("state") != "ok":
        return
    summ = inf["summary"]
    sett = summ.get("bootstrap_settings") or {}
    d["boot_settings"] = {"level": _num(sett.get("level")) or 0.95, "reps": sett.get("reps"),
                          "method": sett.get("method"), "block_rule": sett.get("block_rule")}
    order = [x for x in (sett.get("series_order") or []) if x in (summ.get("bootstrap") or {})] or \
        list((summ.get("bootstrap") or {}).keys())
    order = (["portfolio"] if "portfolio" in order else []) + [x for x in order if x != "portfolio"]
    cf = summ.get("closed_form") or {}
    dsr = ((summ.get("trials") or {}).get("dsr") or {})
    d["boot_rows"] = []
    for sname in order:
        sh = artifacts.bootstrap_ci(inf, sname, "sharpe") or {}
        c = cf.get(sname) or {}
        d["boot_rows"].append({
            "series": SLEEVE_NAMES.get(sname, sname), "key": sname, "point": sh.get("point"),
            "lo": sh.get("lo"), "hi": sh.get("hi"), "excludes_zero": sh.get("excludes_zero"),
            "psr": _num(c.get("psr_gt_0")), "skew": _num(c.get("skew")), "kurt": _num(c.get("kurtosis")),
            "min_trl": _num(c.get("min_trl_years")),
            "dsr": _num((dsr.get(sname) or {}).get("n_eff_c16_c22"))})
    tri = summ.get("trials") or {}
    nch = tri.get("n_choices") or {}
    d["dsr_note"] = {"n_eff": _num(nch.get("n_eff_c16_c22")), "n_raw": _num(nch.get("n_raw_c16_c22")),
                     "n_all": _num(nch.get("n_raw_all_rounds")),
                     "ledger": (tri.get("ledger") or {}).get("path")}
    for r in d["boot_rows"]:
        rng = (tri.get("dsr_range") or {}).get(r["key"]) or {}
        r["dsr_lo"], r["dsr_hi"] = _num(rng.get("min")), _num(rng.get("max"))
    nulls = (summ.get("nulls") or {})
    d["null_settings"] = nulls.get("settings") or {}
    d["null_rows"] = []
    for sname, t in (nulls.get("sleeves") or {}).items():
        circ = ((t or {}).get("circular") or {}).get("total_pnl") or {}
        shuf = ((t or {}).get("shuffle") or {}).get("total_pnl") or {}
        d["null_rows"].append({"series": SLEEVE_NAMES.get(sname, sname), "key": sname,
                               "real": _num(circ.get("real")),
                               "null_mean": _num(circ.get("null_mean")), "q05": _num(circ.get("null_q05")),
                               "q95": _num(circ.get("null_q95")),
                               "timing": _num(circ.get("timing_component")),
                               "p": _num(circ.get("p_value")), "p_holm": _num(circ.get("p_holm_6_sleeves")),
                               "n": circ.get("n_paths"), "p_shuffle": _num(shuf.get("p_value")),
                               "n_shuffle": shuf.get("n_paths")})
    d["null_rows"].sort(key=lambda r: list(SLEEVE_NAMES).index(r["key"]) if r["key"] in SLEEVE_NAMES else 99)
    mt = summ.get("multiple_testing") or {}
    fam = [("published", "Published sleeves vs cash"),
           ("trial_family", "Every configuration tried (trials ledger)"),
           ("beta_estimated", "Configurations at an estimated β")]
    d["mt_rows"] = []
    for key, label in fam:
        spa = mt.get(f"spa_{key}") or {}
        stepm = mt.get(f"stepm_{key}") or {}
        pbo = mt.get(f"pbo_{key}") or (mt.get("pbo_all_trials") if key == "trial_family" else None) or {}
        if not spa and not stepm and not pbo:
            continue
        d["mt_rows"].append({"label": label, "n": spa.get("n_strategies") or pbo.get("n_configurations"),
                             "spa_p": _num(spa.get("p_consistent")), "best": spa.get("best"),
                             "superior": stepm.get("superior"), "pbo": _num(pbo.get("pbo")),
                             "pbo_n": pbo.get("n_configurations")})
    if mt.get("pbo_portfolio_configs"):
        d["pbo_portfolio"] = mt["pbo_portfolio_configs"]
    d["inf_headline"] = list(summ.get("headline") or [])
    d["inf_conventions"] = summ.get("conventions") or {}
    # real-settlement benchmarks (benchmarks.json)
    bm = inf.get("benchmarks") or {}
    co = bm.get("co_movement") or {}
    idx = [x for x in (bm.get("indices") or []) if isinstance(x, str)]
    if co and idx:
        d["bench_indices"] = idx
        d["bench_rows"] = []
        for sname in ("portfolio", "s1c", "s3", "s4", "always_short_straddle"):
            if sname not in co:
                continue
            cells = {}
            for ix in idx:
                a = ((co[sname].get(ix) or {}).get("all") or {})
                cells[ix] = {"corr": _num(a.get("corr_daily")), "beta": _num(a.get("beta")),
                             "sharpe_diff": _num(a.get("sharpe_diff"))}
            d["bench_rows"].append({"series": SLEEVE_NAMES.get(sname, sname), "key": sname, "cells": cells})
        ib = bm.get("implied_beta") or {}
        d["implied_beta"] = []
        for ix in ("PUT", "WPUT"):
            b = ib.get(ix) or {}
            if not b:
                continue
            by = b.get("by_year") or {}
            py = b.get("partial_years") or {}
            d["implied_beta"].append({"index": ix, "overall": _num(b.get("overall")), "n": b.get("n_periods"),
                                      "years": {str(y): _num(v if not isinstance(v, dict) else v.get("beta"))
                                                for y, v in by.items()},
                                      "partial": {str(y): p for y, p in py.items() if isinstance(p, dict)}})
        # a year the artifact records as partial is headed 'YTD' and footnoted with
        # its window and each index's roll periods (benchmarks.json partial_years)
        if d["implied_beta"]:
            years = list(d["implied_beta"][0]["years"])
            ytd = [y for y in years if any(y in r["partial"] for r in d["implied_beta"])
                   or str(y) in {str(k) for k in (bm.get("partial_years") or {})}]
            d["implied_beta_cols"] = [f"{y} YTD" if y in ytd else y for y in years]
            notes = []
            for y in ytd:
                per = [r for r in d["implied_beta"] if y in r["partial"]]
                win = next((r["partial"][y].get("window") for r in per if r["partial"][y].get("window")),
                           (bm.get("partial_years") or {}).get(y))
                span = " → ".join(str(win).split("..")) if win else None
                legs = [f"{r['index']} {int(r['partial'][y]['n_periods']):,} roll periods (expiries "
                        f"{r['partial'][y].get('first_expiry')} → {r['partial'][y].get('last_expiry')})"
                        for r in per if r["partial"][y].get("n_periods") is not None]
                notes.append(f"{y} YTD covers " + (span or "part of the year")
                             + (f": {', '.join(legs)}" if legs else ""))
            d["implied_beta_note"] = "; ".join(notes) or None
        d["canonical_beta"] = _num(ib.get("canonical_beta"))
        d["bench_window"] = bm.get("window")
        if d["implied_beta"]:
            d["charts"]["implied_beta"] = implied_beta_spec(d["implied_beta"], d["implied_beta_cols"],
                                                            d["canonical_beta"])
    # volatility-forecast evaluation (forecast_eval.json)
    fe = inf.get("forecast") or {}
    bw = fe.get("backtest_window") or {}
    q = ((bw.get("losses") or {}).get("qlike") or {})
    if q:
        mcs = ((bw.get("mcs") or {}).get("qlike") or {}).get("90")
        mcs_set = set(mcs if isinstance(mcs, list)
                      else (mcs or {}).get("included", []) if isinstance(mcs, dict) else [])
        d["forecast_rows"] = sorted(
            [{"model": m, "desc": (fe.get("models") or {}).get(m),
              "qlike": _num(v if not isinstance(v, dict) else v.get("mean")),
              "in_mcs": (m in mcs_set) if mcs_set else None, "r2": _num((bw.get("r2_vol") or {}).get(m)),
              "rank": ((bw.get("rank") or {}).get("qlike") or {}).get(m),
              "mean_vol": _num((bw.get("mean_forecast_vol") or {}).get(m))} for m, v in q.items()],
            key=lambda r: (r["qlike"] is None, r["qlike"]))
        d["forecast_meta"] = {"start": bw.get("start"), "end": bw.get("end"), "n": bw.get("n"),
                              "horizon": bw.get("horizon"), "realised": _num(bw.get("mean_realised_vol"))}


# ── Overview ──────────────────────────────────────────────────────────────────
SURFACE_HEADLINE_MODEL = "svi_cal"   # the arbitrage-free (butterfly- and calendar-certified) surface


def _range_txt(values: Iterable[Any], spec: str) -> str | None:
    vals = [v for v in (_num(x) for x in values) if v is not None]
    if not vals:
        return None
    return f"{_fmt(min(vals), spec)}–{_fmt(max(vals), spec)}"


def surface_result(surf_oos: list[dict], session: str | None) -> dict | None:
    """Key result 1: the quote-grade surface's held-out error in half-spreads.

    The headline is the arbitrage-free SVI-cal surface's out-of-sample median
    |model − mid| / half-spread on ``session`` (the newest surface session when
    None), with Heston on the same session beside it and the range of the
    surface's figure over every stored session. A range across sessions is not
    a sampling interval and the card says so.
    """
    rows = [r for r in surf_oos if r["model"] == SURFACE_HEADLINE_MODEL]
    if not rows:
        return None
    sessions = sorted({r["session"] for r in rows})
    sess = session if session in sessions else sessions[-1]
    cur = next(r for r in rows if r["session"] == sess)
    hkey = heston_model_key(r["model"] for r in surf_oos if r["session"] == sess)
    hes = next((r for r in surf_oos if r["session"] == sess and r["model"] == hkey), None)
    return {
        "session": sess, "n_sessions": len(sessions), "model": SURFACE_HEADLINE_MODEL,
        "median_abs_hs": cur["median_abs_hs"], "inside_spread_pct": cur["inside_spread_pct"], "n": cur["n"],
        "butterfly": cur["butterfly"], "calendar": cur["calendar"],
        "range": _range_txt((r["median_abs_hs"] for r in rows), ".2f"),
        "inside_range": _range_txt((r["inside_spread_pct"] for r in rows), ".0f"),
        "heston": ({"median_abs_hs": hes["median_abs_hs"], "inside_spread_pct": hes["inside_spread_pct"],
                    "n": hes["n"]} if hes else None),
        "heston_range": _range_txt((r["median_abs_hs"] for r in surf_oos if r["model"] == hkey), ".1f"),
    }


def joint_fit_result(diag: dict, session: str | None) -> dict | None:
    """Key result 2: the joint SPX/VIX fit out of sample, Heston and rough Heston.

    From calibrations/diagnostics/summary.json: the production joint Heston
    fit's out-of-sample SPX RMSE (``sigma_cap2_oos``) and the joint rough
    Heston fit's on the same quotes, for ``session`` (the headline fit's) and
    over every diagnosed session. "Lower" needs a margin of 0.01 vol pts; a
    rough fit that returns H = 1/2 is Heston and is counted as the same fit.
    """
    if diag.get("state") != "ok":
        return None
    rows = [r for r in diag["rows"] if r["sigma_cap2_oos"] is not None and r["joint_rough_oos"] is not None]
    if not rows:
        return None
    cur = next((r for r in rows if r["session"] == session), rows[-1])
    lower = sum(1 for r in rows if r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01)
    same = sum(1 for r in rows if r["joint_rough_H"] is not None and r["joint_rough_H"] >= 0.5 - 1e-9)
    return {"session": cur["session"], "n_sessions": len(rows), "heston_oos": cur["sigma_cap2_oos"],
            "rough_oos": cur["joint_rough_oos"], "rough_H": cur["joint_rough_H"],
            "rough_H_on_bound": "H" in cur["joint_rough_on_bound"], "heston_on_bound": cur["on_bound"],
            "heston_vix": cur["joint_heston_vix_index"], "rough_vix": cur["joint_rough_vix_index"],
            "kappa_rse": _num((cur["relative_standard_errors"] or {}).get("kappa")),
            "n_lower": lower, "n_same": same,
            "heston_range": _range_txt((r["sigma_cap2_oos"] for r in rows), ".2f"),
            "rough_range": _range_txt((r["joint_rough_oos"] for r in rows), ".2f")}


def key_results(p2: dict, p4: dict | None, surf_oos: list[dict], diag: dict, inf: dict) -> list[dict]:
    """The Overview's three results, each with its interval or the reason it has none:
    the quote-grade surface in half-spreads, the joint SPX/VIX fit (Heston and
    rough Heston) out of sample, and the backtest Sharpe with its bootstrap
    interval and the timing null."""
    out = []
    # 1. the quote-grade surface, in units of the quoted half-spread
    sr = surface_result(surf_oos, p2.get("session"))
    if sr and _num(sr["median_abs_hs"]) is not None:
        h = sr["heston"]
        cert = (f"; {sr['butterfly']} butterfly and {sr['calendar']} calendar violations on its dense grid"
                if sr["butterfly"] is not None and sr["calendar"] is not None else "")
        # what is held out and what the quotes are, from the surface's own record
        fit_in = artifacts.surface_fit_inputs(DATA, sr["session"])
        held = ("alternate strikes held out" if str(fit_in.get("rule") or "").startswith("alternate strikes")
                else "held-out strikes")
        src = str(fit_in.get("source") or "").lower()
        qt = fit_in.get("quote_time")
        snap = (" Yahoo close snapshot, no quote time." if src in ("yfinance", "yahoo") and not qt
                else (f" Quotes as of {qt}." if qt else ""))
        out.append({
            "key": f"Arbitrage-free SVI surface · {held}",
            "value": _fmt(sr["median_abs_hs"], ".2f"), "unit": "half-spreads",
            "detail": (f"Median |model − mid| of the arbitrage-free, calendar-ordered SVI surface on "
                       f"{_fmt(sr['n'], ',')} held-out quotes of {sr['session']}; "
                       f"{_fmt(sr['inside_spread_pct'], '.1f')}% inside the bid–ask{cert}."
                       + (f" Heston on its own held-out quotes of that session: "
                          f"{_fmt(h['median_abs_hs'], '.1f')} half-spreads, "
                          f"{_fmt(h['inside_spread_pct'], '.1f')}% inside." if h else "") + snap),
            "ci": (f"Range over {sr['n_sessions']} stored sessions: {sr['range']} half-spreads "
                   f"({sr['inside_range']}% inside)" if sr["n_sessions"] > 1 else None),
            # shown beside the range (it was rendered only when there was no range)
            "ci_kind": "range",
            "ci_note": ("A range across sessions, not a sampling interval. Held-out strikes sit between "
                        "fitted ones (an interpolation test)."),
            "href": "surface.html#scorecard", "source": f"data_store/surface/{sr['session']}/scorecard.json"})
    else:
        out.append({"key": "Arbitrage-free SVI surface · held-out strikes", "value": "—", "unit": "",
                    "detail": "The surface report has not run for any stored session, so no model is scored "
                              "in half-spreads yet.",
                    "ci": None, "ci_note": "needs data_store/surface/<session>/scorecard.json "
                                           "(scripts/surface_report.py), not yet generated",
                    "href": "surface.html", "source": "data_store/surface/<session>/scorecard.json"})
    # 2. the joint SPX/VIX fit out of sample: Heston against rough Heston
    jr = joint_fit_result(diag, p2.get("session"))
    if jr:
        bound = " (σ on its box cap)" if "sigma" in (jr["heston_on_bound"] or []) else ""
        h_txt = (f"H {_fmt(jr['rough_H'], '.2f')}{' on its floor' if jr['rough_H_on_bound'] else ''}"
                 if jr["rough_H"] is not None else "H —")
        n_txt = f"the {_fmt(p2.get('n_oos'), ',')}" if p2.get("session") == jr["session"] else "the"
        out.append({
            "key": "Joint SPX/VIX fit · out of sample",
            "value": _fmt(jr["heston_oos"], ".2f"), "unit": "vol pts, Heston",
            "detail": (f"SPX IV RMSE on {n_txt} quotes outside the fit, {jr['session']}, with the four VIX "
                       f"indices in the objective{bound}. Rough Heston on the same quotes: "
                       f"{_fmt(jr['rough_oos'], '.2f')} "
                       f"({h_txt}); lower on {jr['n_lower']} of {jr['n_sessions']} sessions, and on "
                       f"{jr['n_same']} it returns H = ½, i.e. Heston."),
            "ci": (f"Range over {jr['n_sessions']} sessions: Heston {jr['heston_range']}, rough Heston "
                   f"{jr['rough_range']} vol pts"
                   + (f"; κ ±{_fmt(jr['kappa_rse'] * 100, '.0f')}% (Gauss–Newton curvature scale)"
                      if jr["kappa_rse"] is not None else "")),
            "ci_kind": "range", "ci_note": None, "href": "calibration.html#rough-vs-heston",
            "source": "data_store/calibrations/diagnostics/summary.json"})
    elif p2.get("spx_oos_rmse") is not None:
        out.append({"key": "Joint SPX/VIX fit · out of sample",
                    "value": _fmt(p2["spx_oos_rmse"], ".2f"), "unit": "vol pts, Heston",
                    "detail": (f"SPX IV RMSE on the {_fmt(p2.get('n_oos'), ',')} quotes outside the fit "
                               f"({p2.get('session')}), with the four VIX indices in the objective."),
                    "ci": None, "ci_note": ("rough Heston and the range over sessions need the calibration "
                                            "diagnostics (diagnostics/summary.json), not yet generated"),
                    "href": "calibration.html#oos",
                    "source": f"data_store/calibrations/{p2.get('calib_file')}"})
    # 3. backtest: the portfolio Sharpe with its bootstrap interval, data snooping and timing nulls
    if p4 and p4.get("kpi"):
        k = p4["kpi"]
        ci = artifacts.bootstrap_ci(inf, "portfolio", "sharpe") if inf.get("state") == "ok" else None
        summ = (inf.get("summary") or {}) if ci else {}
        s3 = ((summ.get("nulls") or {}).get("sleeves") or {}).get("s3")
        circ = ((s3 or {}).get("circular") or {}).get("total_pnl") or {}
        spa_pub = (summ.get("multiple_testing") or {}).get("spa_published")
        spa = _num(spa_pub.get("p_consistent")) if isinstance(spa_pub, dict) else None
        if ci:
            p_txt = (f"S3 entry timing vs {_fmt(circ.get('n_paths'), ',')} exposure-matched circular "
                     f"shifts: p "
                     f"{_fmt(_num(circ.get('p_value')), '.3f')} (Holm over 6 sleeves "
                     f"{_fmt(_num(circ.get('p_holm_6_sleeves')), '.2f')})"
                     if circ.get("p_value") is not None else "")
            spa_txt = (f"; Hansen SPA, published series vs cash: p {_fmt(spa, '.2f')}"
                       if spa is not None else "")
            out.append({"key": "Backtest Sharpe · mark-to-model",
                        "value": _fmt(ci["point"], "+.2f"), "unit": "portfolio",
                        "detail": (f"{p4.get('start_date', '')[:4]}–{p4.get('end_date', '')[:4]}, straddles "
                                   f"marked at β·VIX (no option prices in the window){spa_txt}. {p_txt}."),
                        "ci": (f"{_fmt(ci['level'] * 100, '.0f')}% CI {_fmt(ci['lo'], '+.2f')} to "
                               f"{_fmt(ci['hi'], '+.2f')} "
                               f"({ci.get('method') or 'bootstrap'}, {_fmt(ci.get('reps'), ',')} draws)"),
                        "ci_kind": "interval", "ci_note": None, "href": "research.html#inference",
                        "source": "data_store/backtest/inference/summary.json"})
        else:
            p = k.get("s3_null_p")
            out.append({"key": "Signal timing vs random entry",
                        "value": _fmt(p, ".3f"), "unit": "p (S3)",
                        "detail": (f"{'Not significant' if p is not None and p >= 0.05 else 'Significant'}"
                                   f" at 5%; "
                                   f"portfolio Sharpe {_fmt(k.get('sharpe'), '.2f')} (mark-to-model, "
                                   f"{p4.get('start_date', '')[:4]}–{p4.get('end_date', '')[:4]})"),
                        "ci": None,
                        "ci_note": "the bootstrap interval needs the inference artifact, not yet generated",
                        "href": "research.html#inference", "source": "data_store/backtest/full_results.pkl"})
    return out


FORECAST_LOSSES = (("mse", "21-day MSE"), ("qlike", "QLIKE"))


def pdv_forecast_rank(fe: dict | None, model: str = "pdv", level: str = "90") -> dict | None:
    """Where the PDV forecaster ranks among the evaluated forecasters, on both losses.

    From forecast_eval.json backtest_window: rank.{mse, qlike} (1 = best), the
    model confidence set at ``level``% per loss, and the best model per loss.
    The Overview showed the PDV forecast without saying it ranks 6th of 7 on
    QLIKE; showing only that rank would be selective the other way (2nd on MSE).
    """
    bw = (fe or {}).get("backtest_window") or {}
    ranks = bw.get("rank") or {}
    out: dict[str, Any] = {"losses": [], "start": bw.get("start"), "end": bw.get("end"),
                           "horizon": bw.get("horizon"), "level": level}
    for key, label in FORECAST_LOSSES:
        r = ranks.get(key) or {}
        if model not in r:
            continue
        mcs = ((bw.get("mcs") or {}).get(key) or {}).get(level)
        inc = (mcs if isinstance(mcs, list) else (mcs or {}).get("included")) if mcs else None
        best = sorted(m for m in r if r[m] == min(r.values()))
        out["losses"].append({"key": key, "label": label, "rank": int(r[model]), "n": len(r),
                              "in_mcs": (model in inc) if isinstance(inc, list) else None, "best": best})
    if not out["losses"]:
        return None
    bests = {tuple(x["best"]) for x in out["losses"]}
    out["best_both"] = list(next(iter(bests))) if len(bests) == 1 else None
    out["descs"] = (fe or {}).get("models") or {}
    return out


def findings_at_a_glance(p2: dict, p4: dict | None, surf_oos: list[dict], diag: dict, inf: dict,
                         pipeline: dict | None) -> list[dict]:
    """The Overview's three columns: what works, what does not, what is next.

    Each item is one computed sentence with one link to the panel that shows
    it; an item whose artifact is missing is left out, never filled in. 'Next'
    items say what is not done yet. The columns carry the same weight: the
    template renders them with the same classes.
    """
    works: list[dict] = []
    fails: list[dict] = []
    nxt: list[dict] = []

    def item(lead: str, href: str, text: str) -> dict:
        return {"lead": lead, "href": href, "text": text}
    # works: the arbitrage-free surface on held-out strikes, the VIX replication, the gate
    sr = surface_result(surf_oos, p2.get("session"))
    if sr and _num(sr["median_abs_hs"]) is not None:
        works.append(item(
            "Arbitrage-free SVI surface", "surface.html#scorecard",
            f"median {_fmt(sr['median_abs_hs'], '.2f')} half-spreads from the mid, "
            f"{_fmt(sr['inside_spread_pct'], '.1f')}% inside the bid–ask, on the held-out strikes of "
            f"{sr['session']}."))
        try:
            rep = artifacts.load_surface(DATA, sr["session"])
        except artifacts.ArtifactError:
            rep = {}
        indices = ((rep.get("vix") or {}).get("indices") or {}) if rep.get("state") == "ok" else {}
        vix = indices.get("VIX") or {}
        gap = _num(vix.get("gap_vs_official"))
        if gap is not None:
            works.append(item(
                "VIX replicated from the stored chain", "surface.html#vix-replication",
                f"{signed(gap, '.2f')} index pts from the Cboe close (replicated − official) on "
                f"{sr['session']}, by the Cboe rules."))
    arc = p2.get("archive") or {}
    if arc.get("n_legacy") is not None:
        works.append(item(
            "The gate keeps broken-data fits off the page", "calibration.html#history",
            f"{plural(arc['n_legacy'], 'pre-C22 fit')} labelled legacy, none published; "
            f"{arc.get('n_current_ok')} of {arc.get('n_current')} current-schema fits pass."))
    # does not work: Heston's short end, the backtest, the classifier
    short = [r for r in p2.get("oos_rows") or [] if r["T_days"] is not None and r["T_days"] <= 30]
    worst = max(short, key=lambda r: r["rmse"]) if short else None
    if worst:
        bits = [(f"{_fmt(worst['rmse'], '.2f')} vol pts out of sample on the {worst['expiry']} expiry "
                 f"({_fmt(worst['T_days'], '.0f')} days)")]
        if "sigma" in (p2.get("pinned_params") or []):
            bits.append("σ on its box cap")
        vv = p2.get("vvix") or {}
        if vv:
            bits.append(f"VVIX {_fmt(vv['model'], '.0f')} against {_fmt(vv['market'], '.0f')}")
        fails.append(item("One-factor Heston at the short end", "calibration.html#oos",
                          ", ".join(bits) + f" ({p2.get('session')})."))
    # Heston's ATM skew against the market's power law in T (#skew-term): 'works'
    # only when within 10% of the SVI-cal market skew at every listed expiry
    sk = p2.get("skew_term") or {}
    cmp_, rng = sk.get("compare"), (sk.get("slope_ranges") or {}).get("svi_all") or []
    if cmp_ and rng:
        f0 = cmp_["first"]
        rel = ("at every listed expiry" if (cmp_["steeper_all"] or cmp_["flatter_all"])
               else f"across the {cmp_['n']} listed expiries")
        text = (f"×{_fmt(cmp_['lo'], '.2f')} to ×{_fmt(cmp_['hi'], '.2f')} the SVI-cal market skew {rel} of "
                f"{sk['head']} ({_fmt(abs(f0['heston']), '.2f')} against {_fmt(abs(f0['svi']), '.2f')} at "
                f"{_fmt(f0['T_days'], '.0f')} d); the market's slope in T is {signed(min(rng), '.2f')} to "
                f"{signed(max(rng), '.2f')} over {plural(len(rng), 'session')} (SVI-cal, all expiries).")
        if cmp_["lo"] >= 0.9 and cmp_["hi"] <= 1.1:
            works.append(item("Heston's ATM skew matches the market's", "calibration.html#skew-term", text))
        else:
            lead = ("Heston's ATM skew is steeper than the market's" if cmp_["steeper_all"] else
                    "Heston's ATM skew is flatter than the market's" if cmp_["flatter_all"] else
                    "Heston's ATM skew against the market's")
            fails.append(item(lead, "calibration.html#skew-term", text))
    ci = artifacts.bootstrap_ci(inf, "portfolio", "sharpe") if inf.get("state") == "ok" else None
    k = (p4 or {}).get("kpi") or {}
    if k.get("cum_ret") is not None and ci:
        years = f"{str((p4 or {}).get('start_date'))[:4]}–{str((p4 or {}).get('end_date'))[:4]}"
        fails.append(item(
            "The mark-to-model backtest loses", "research.html#inference",
            f"{_pct_prose(k['cum_ret'], 1)} over {years}, Sharpe {_fmt(ci['point'], '+.2f')} "
            f"({_fmt(ci['level'] * 100, '.0f')}% CI {_fmt(ci['lo'], '+.2f')} to {_fmt(ci['hi'], '+.2f')})."))
    try:
        clf = _load_pickle(DATA / "signals" / "regime_classifier.pkl")
        ev = (getattr(clf, "metadata_", None) or {}).get("evaluation") or {}
    except (OSError, EOFError, AttributeError, pickle.UnpicklingError) as exc:
        _logger.info("classifier evaluation unavailable for the findings: %s", exc)
        ev = {}
    acc, base = _num(ev.get("oos_acc")), _num(ev.get("persistence_acc"))
    if acc is not None and base is not None:
        fails.append(item(
            "The ML regime classifier", "research.html#research-classifier",
            f"{_fmt(acc * 100, '.1f')}% out-of-sample accuracy against {_fmt(base * 100, '.1f')}% for "
            "the predict-yesterday baseline; research only, it never trades."))
    # next: what is not done yet
    fw = (pipeline or {}).get("forward") or {}
    n_fw = (fw.get("n") or 0) if fw.get("state") == "ok" else 0
    nxt.append(item(
        "Forward test at executable prices", "research.html#forward",
        f"not yet evidence: {plural(n_fw, 'registered session')} so far; the ledger fixes targets "
        "before any later price exists."))
    jr = joint_fit_result(diag, p2.get("session"))
    if jr:
        h_txt = "sits on its 0.02 floor" if jr["rough_H_on_bound"] else "is weakly identified"
        nxt.append(item(
            "Rough Heston in production", "calibration.html#rough-vs-heston",
            f"not adopted: lower out-of-sample error on {jr['n_lower']} of {jr['n_sessions']} sessions, "
            f"but H {h_txt} ({jr['session']})."))
    # the VX futures leg: scored out of sample only, and not at all where none is stored
    vl = p2.get("vx_leg") or {}
    legs = [v["leg"] for v in vl.get("views") or [] if v.get("leg")]
    if legs:
        lo, hi = vl["range"]
        n_c, n_b = sum(g["n"] for g in legs), sum(g["n_below"] for g in legs)
        rng = _fmt(lo, ".2f") if len(legs) == 1 else f"{_fmt(lo, '.2f')}–{_fmt(hi, '.2f')}"
        tail = f" {vl['headline_empty']}" if vl.get("headline_empty") else ""
        nxt.append(item(
            "VX futures in the joint objective", "calibration.html#vixleg",
            f"not done: the fits score VX settlements out of sample only, {rng} index pts RMSE on "
            f"{len(legs)} of {len(vl['views'])} gated fits, the model below the settlement on {n_b} of "
            f"{n_c} contracts.{tail}"))
    from joint_vol_calibration.config import HESTON_BOUNDS
    nxt.append(item(
        "A wider production σ box", "calibration.html#sigma-tradeoff",
        f"not done: σ ≤ {_fmt(float(HESTON_BOUNDS['sigma'][1]), 'g')} is kept because the NN pricer's "
        "domain, the vomma shock and the bound chips are tied to it (Corrections C23-M1)."))
    return [{"key": "works", "title": "Works", "glyph": "✓", "items": works},
            {"key": "fails", "title": "Does not work", "glyph": "✕", "items": fails},
            {"key": "next", "title": "Next · not done yet", "glyph": "→", "items": nxt}]


def latest_cboe_captures(data_dir: Path | None = None) -> list[dict]:
    """The newest stored Cboe book per underlying (data_store/options/cboe/*.json records)."""
    root = Path(data_dir or DATA) / "options" / "cboe"
    newest: dict[str, dict] = {}
    for f in sorted(root.glob("*.json")) if root.is_dir() else []:
        try:
            rec = json.loads(f.read_text())
        except (OSError, ValueError) as exc:
            raise ValueError(f"options/cboe/{f.name}: unreadable record ({exc})") from exc
        und = str(rec.get("underlying") or f.name.split("_")[0]).upper()
        key = (str(rec.get("session")), str(rec.get("quote_time")))
        if und not in newest or key > (str(newest[und].get("session")), str(newest[und].get("quote_time"))):
            newest[und] = {**rec, "file": f.name}
    return [newest[u] for u in sorted(newest)]


def pipeline_state(diag: dict) -> dict:
    """What the daily job has stored: books, fits, derived artifacts, the forward ledger."""
    out: dict[str, Any] = {"captures": [], "fits": None, "fits_newest": None, "surface_sessions": [],
                           "diag_sessions": [], "forward": None}
    out["captures"] = latest_cboe_captures()
    fits = [r for r in scan_calibrations() if r["legacy"] is False and r["ok"] and r["session"]]
    out["fits"] = len(fits)
    out["fits_newest"] = max((r["session"] for r in fits), default=None)
    out["surface_sessions"] = artifacts.surface_sessions(DATA)
    out["diag_sessions"] = [r["session"] for r in diag.get("rows") or []] if diag.get("state") == "ok" else []
    try:
        out["forward"] = artifacts.load_forward_ledger(DATA)
    except artifacts.ArtifactError as e:
        out["forward"] = {"state": "error", "message": str(e)}
    out["forward_pending"] = forward_ledger_protocol()["pending"]
    return out


TS_OVERLAY_SESSIONS = ((1, "1 session earlier"), (5, "5 sessions earlier"), (21, "21 sessions earlier"))


def overview_vix_complex(d: dict) -> None:
    """The Overview's VIX complex: the term structure's earlier sessions (dashed,
    dated), the VX futures curve on its own date, VIX/VIX3M since 2010 and its
    percentile. Futures stay on their own as-of date: never on the spot chart."""
    from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
    vw = _read_vix_wide()
    tenor_days = dict(VIX_INDEX_TENOR_DAYS)
    cols = [c for c in tenor_days if c in vw.columns]
    full = vw[cols].dropna() if cols else pd.DataFrame()
    d["ts_overlays"] = []
    if len(full) and d.get("ts_date") == str(full.index[-1].date()):
        for k, what in TS_OVERLAY_SESSIONS:
            if len(full) > k:
                row, day = full.iloc[-1 - k], str(full.index[-1 - k].date())
                pts = sorted((tenor_days[c], float(row[c])) for c in cols)
                d["ts_overlays"].append({"date": day, "k": k, "name": f"{day} ({what})",
                                         "days": [t for t, _ in pts],
                                         "values": [round(v, 2) for _, v in pts]})
    # VIX / VIX3M since the start of the term-structure table
    if "^VIX" in vw.columns and "^VIX3M" in vw.columns:
        ratio = (vw["^VIX"] / vw["^VIX3M"]).dropna()
        rv = desk_market.ratio_view(ratio)
        d["vix_ratio"] = rv
        if rv:
            rv["rank"] = percentile_rank(rv["pct"])
            xs = [str(pd.Timestamp(t).date()) for t in ratio.index]
            years = pd.DatetimeIndex(ratio.index).year
            by_year = [(int(y), len(g), float((g > 1).mean() * 100), float(g.mean()))
                       for y, g in ratio.groupby(years)]
            d["charts"]["vix_ratio"] = charts.xy(
                "chart-vix-ratio",
                f"VIX / VIX3M, {rv['first']} to {rv['last']}, shaded above 1 (backwardation)",
                x={"type": "time", "csv": "date"},
                y={"name": "VIX / VIX3M", "digits": 1, "tipDigits": 3},
                series=[{"name": "VIX / VIX3M", "type": "line", "color": "cat0", "width": 1.0,
                         "sampling": "lttb", "data": [round(float(x), 3) for x in ratio]}],
                xs=xs, csv="vix_over_vix3m.csv", grid={"top": 10, "bottom": 30},
                marks={"y": [{"value": 1.0, "label": "1.0", "color": "muted", "dash": False}],
                       "areas": [{"from": a, "to": b, "color": "watch", "alpha": 0.22}
                                 for a, b in rv["runs"]]},
                describe=(f"Daily VIX over VIX3M closes; above 1 (the 30-day index over the 3-month one) on "
                          f"{_fmt(rv['share_above'], '.1f')}% of {rv['n']:,} sessions; "
                          f"today {_fmt(rv['today'], '.3f')}."),
                table=charts.table("VIX / VIX3M by calendar year; the CSV has every session",
                                   ["Year", "Sessions#", "Above 1 (%)#", "Mean#"],
                                   [[y, n, _fmt(sh, ".1f"), _fmt(m_, ".3f")] for y, n, sh, m_ in by_year]))
    # the VX futures curve: its own newest date, the session before as an overlay
    d["vx_curve"] = None
    last = vx_last_date()
    if last:
        cur = desk_market.vx_curve(vx_session_rows(last))
        prev_date = vx_prev_date(last)
        prev = desk_market.vx_curve(vx_session_rows(prev_date)) if prev_date else None
        if cur:
            ref = _latest_market_session()
            age = session_age(last, ref)
            d["vx_curve"] = {**cur, "date": last, "prev_date": prev_date if prev else None, "prev": prev,
                             "age": age, "level": stale_level(age), "ref": ref}
            pv = {r["expiry"]: r["settle"] for r in (prev or {}).get("rows") or []}
            cats = [r["expiry"] for r in cur["rows"]]
            series = [{"name": f"{last} settlements", "type": "line", "color": "headline", "symbol": "circle",
                       "size": 6, "width": 1.6, "legend": True, "z": 4,
                       "data": [r["settle"] for r in cur["rows"]]}]
            if prev:
                series.append({"name": f"{prev_date} settlements", "type": "line", "color": "muted",
                               "symbol": "circle", "size": 4, "width": 1.1, "dash": True, "legend": True,
                               "data": [pv.get(c) for c in cats]})
            d["charts"]["vx_curve"] = charts.xy(
                "chart-vx-curve", f"VX monthly futures settlements on {last}" + (
                    f", with {prev_date}" if prev else ""),
                x={"type": "category", "categories": [c[2:] for c in cats],
                   "name": "Contract (final settlement)", "interval": 0, "rotate": 45},
                y={"name": "VX (index pts)", "digits": 1, "tipDigits": 2},
                series=series, xs=[c[2:] for c in cats], csv=f"vx_curve_{last}.csv",
                grid={"top": 12, "bottom": 64},
                describe=(f"{cur['n']} monthly contracts on {last}, front month first; "
                          "M1 → M2 "
                          f"{_fmt(cur['m1m2'] * 100 if cur['m1m2'] is not None else None, '+.2f')}%."),
                table=charts.table(f"VX monthly settlements, {last}" + (f" and {prev_date}" if prev else ""),
                                   ["Contract", "Final settlement", "Days#", f"{last}#",
                                    *([f"{prev_date}#", "Change#"] if prev else [])],
                                   [[r["month"], r["expiry"], _fmt(r["days"], ".0f"),
                                     _fmt(r["settle"], ".2f"),
                                     *([_fmt(pv.get(r["expiry"]), ".2f"),
                                        (signed(r["settle"] - pv[r["expiry"]], ".2f")
                                         if r["expiry"] in pv else "—")]
                                       if prev else [])] for r in cur["rows"]]))


def load_overview() -> dict[str, Any]:
    d = load_page1()
    d.setdefault("charts", {})
    p2 = load_page2()
    for e in p2.get("errors") or []:
        if e not in d["errors"]:
            d["errors"].append(e)
    d["fit"] = {k: p2.get(k) for k in ("session", "calib_file", "spx_rmse", "spx_oos_rmse",
                                       "spx_oos_wing_rmse", "vix_index_rmse", "active_constraints",
                                       "pinned_params", "effective_dof",
                                       "feller_state", "feller_margin", "n_spx_options", "n_spx_expiries",
                                       "calib_age", "model_version", "oos_rows", "n_oos", "vvix")}
    d["fit"]["tiles"] = p2.get("param_tiles") or []
    d["fit"]["ridge"] = p2.get("ridge")
    p4 = None
    with _loader_boundary(d, "backtest engine results (full_results.pkl)"):
        st = _load_pickle(DATA / "backtest" / "full_results.pkl")
        em = st.get("metrics") or {}
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        p4 = {"kpi": {"sharpe": _num(em.get("sharpe")), "s3_null_p": _num(em.get("s3_null_p_value")),
                      "cum_ret": _num(em.get("cumulative_return"))},
              "start_date": str(pd.Timestamp(eq.index[0]).date()),
              "end_date": str(pd.Timestamp(eq.index[-1]).date())}
    try:
        surf = artifacts.load_surface(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "surface artifacts", e)
        surf = {"state": "error"}
    try:
        inf = artifacts.load_inference(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "backtest inference artifacts", e)
        inf = {"state": "error"}
    surf_oos: list[dict] = []
    try:
        surf_oos = artifacts.surface_oos_by_session(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "surface scorecards", e)
    try:
        diag = artifacts.load_diagnostics_summary(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "calibration diagnostics summary", e)
        diag = {"state": "error"}
    d["key_results"] = key_results(p2, p4, surf_oos, diag, inf)
    d["pdv_forecast_rank"] = pdv_forecast_rank(inf.get("forecast")) if inf.get("state") == "ok" else None
    # What the σ ≤ 2 box costs the headline fit (its σ profile, when diagnosed)
    row = next((r for r in diag.get("rows") or [] if r["session"] == p2.get("session")), None)
    from joint_vol_calibration.config import HESTON_BOUNDS
    rg = p2.get("ridge") or {}
    d["fit"]["free_sigma"] = ({"sigma": row["sigma_free"], "kappa": row["kappa_free"],
                               "oos": row["sigma_free_oos"], "oos_cap": row["sigma_cap2_oos"],
                               "loss_rel": row["sigma_free_loss_rel"],
                               # the refit's own box (the σ profile's top cap and its κ ceiling),
                               # wider than production's: "the same fit" it was not
                               "sigma_max": rg.get("cap_top"), "kappa_max": rg.get("kappa_max"),
                               "kappa_box": float(HESTON_BOUNDS["kappa"][1])}
                              if row and row["sigma_free"] is not None else None)
    d["pipeline"] = None
    with _loader_boundary(d, "pipeline state (options/cboe, calibrations, forward)"):
        d["pipeline"] = pipeline_state(diag)
    d["findings"] = []
    with _loader_boundary(d, "findings at a glance"):
        d["findings"] = findings_at_a_glance(p2, p4, surf_oos, diag, inf, d["pipeline"])
    with _loader_boundary(d, "VIX complex (vol_system.db vix_term_structure, vix_futures_daily)"):
        overview_vix_complex(d)
    if d.get("ts_values"):
        days = [int(t.rstrip("d")) for t in d["ts_tenors"]]
        ov = d.get("ts_overlays") or []
        series = [{"name": f"{d.get('ts_date')} (latest)", "type": "line", "color": "cat0",
                   "symbol": "circle", "size": 7, "width": 2, "legend": True, "z": 4,
                   "data": [[t, v] for t, v in zip(days, d["ts_values"])]}]
        for o, col in zip(ov, ("cat2", "cat4", "cat3")):
            series.append({"name": o["name"], "type": "line", "color": col, "symbol": "circle", "size": 4,
                           "width": 1.2, "dash": True, "legend": True,
                           "data": [[t, v] for t, v in zip(o["days"], o["values"])]})
        d["charts"]["ts"] = charts.xy(
            "chart-ts", f"VIX index term structure on {d.get('ts_date')}, with earlier sessions",
            x={"type": "value", "name": "Tenor (calendar days)", "digits": 0, "min": 0, "zero": True},
            y={"name": "Index level (vol pts)", "digits": 1, "tipDigits": 2},
            series=series, tooltip="item",
            csv=f"vix_term_structure_{d.get('ts_date')}.csv", grid={"top": 12},
            describe=("^VIX9D, ^VIX, ^VIX3M and ^VIX6M on one session, and dashed on "
                      + ", ".join(o["name"] for o in ov) + "; tenor on a linear axis, straight segments."),
            table=charts.table(f"VIX-family index levels (vol pts), {d.get('ts_date')} and earlier sessions",
                               ["Tenor", f"{d.get('ts_date')}#", *(f"{o['date']}#" for o in ov)],
                               [[t, _fmt(v, ".2f"), *(_fmt(o["values"][i], ".2f") for o in ov)]
                                for i, (t, v) in enumerate(zip(d["ts_tenors"], d["ts_values"]))]))
    d["pdv_spark"] = sparkline(d.get("pdv_sparkline") or [], height=36, zero=True)
    # VIX against the realised vol of the last 21 sessions (Research #vrp has the history)
    d["vix_rv"] = None
    with _loader_boundary(d, "VIX − RV21 (vol_system.db spx_ohlcv, vix_daily)"):
        d["vix_rv"] = vix_rv_now(_read_spx(), _read_vix_daily())
    d["surface_state"] = surf.get("state")
    d["inference_state"] = inf.get("state")
    # one line of the surface in desk coordinates (the Surface page's 1M row)
    d["surface_line"] = None
    if surf.get("state") == "ok" and surf.get("surface"):
        with _loader_boundary(d, "surface in desk coordinates"):
            cm = desk_surface.constant_maturity(
                desk_surface.delta_grid(surf["surface"], surf["residuals"], surf.get("vix")))
            d["surface_line"] = surface_state_line(cm, surf["session"], _latest_market_session())
    return d


# ── Surface page ──────────────────────────────────────────────────────────────
MODEL_LABELS = {"heston": "Heston", "heston_v2": "Heston (v2 fit)", "essvi": "eSSVI", "svi": "SVI",
                "svi_cal": "SVI, calendar-ordered",
                "ah": "Andreasen–Huge", "bates": "Bates", "rough_heston": "Rough Heston",
                "quintic": "Quintic OU", "lifted_heston": "Lifted Heston"}
ARB_LABELS = {"C1_outright": "Outright bounds", "C2_vertical_bound": "Vertical bound",
              "C2_vertical_spread": "Vertical spread", "C3_vertical_butterfly": "Butterfly",
              "C4_calendar_spread": "Calendar spread", "C5_calendar_vertical_spread": "Calendar vertical",
              "C6_calendar_butterfly": "Calendar butterfly"}


MODEL_SHORT = {"heston_v2": "Heston", "svi_cal": "SVI-cal", "rough_heston": "Rough Heston"}

# One colour per model on every chart of the site (the smile slice drew Heston
# in the orange the by-expiry chart beside it used for SVI). Okabe–Ito tokens;
# cat3 (yellow) is kept for fit anchors, market quotes are 'headline' (mids)
# and 'muted' (bid–ask).
MODEL_COLOURS = {"heston_v2": "cat1", "heston": "cat1", "svi_cal": "cat0", "svi": "cat2", "essvi": "cat4",
                 "rough_heston": "cat5"}
ANCHOR_COLOUR = "cat3"


def model_colour(m: str) -> str:
    return MODEL_COLOURS.get(m, "cat5")


def model_label(m: str) -> str:
    return MODEL_LABELS.get(m, m.replace("_", " ").title())


def model_short(m: str) -> str:
    return MODEL_SHORT.get(m, model_label(m))


def heston_model_key(models: Iterable[str]) -> str | None:
    """The production Heston fit among a surface artifact's models ('heston_v2', 'heston')."""
    return next((m for m in models if str(m).lower().startswith("heston")), None)


def _model_surface_heat(params: dict) -> dict:
    """Heston IV grid of the headline fit (model, not market) as a heatmap."""
    lm = list(np.round(np.linspace(-0.30, 0.20, 11), 3))
    rows = heston_model_smile(params, maturities_days=SMILE_MATURITIES_DAYS, log_moneyness=lm)
    vals = [v * 100 for r in rows for v in r["iv"]]
    lo, hi = (min(vals), max(vals)) if vals else (0.0, 1.0)
    out_rows, pts = [], []
    for r in rows:
        by = dict(zip(r["lm"], r["iv"]))
        cells = []
        for k in lm:
            v = by.get(round(float(k), 4))
            if v is None:
                cells.append(cell("", None,
                                  detail=f"{r['T']}d · ln(K/F) {_fmt(k, '+.2f')}: price too small to invert"))
                pts.append([float(k), float(r["T"]), None])      # the 3D surface needs the full grid
                continue
            cells.append(cell(_fmt(v * 100, ".1f"), seq_colour(v * 100, lo, hi),
                              detail=(f"{r['T']}d · ln(K/F) {_fmt(k, '+.2f')} · "
                                      f"model IV {_fmt(v * 100, '.2f')}%")))
            pts.append([float(k), float(r["T"]), round(v * 100, 3)])
        out_rows.append({"label": f"{r['T']}d", "cells": cells})
    return {"cols": [_fmt(k, "+.2f") for k in lm], "rows": out_rows, "lo": lo, "hi": hi, "points": pts}


def _bucket_grid(q: pd.DataFrame, exps: pd.Series, value, colour, detail,
                 used: Sequence[int] | None = None) -> list[dict]:
    """Heatmap rows: one per expiry, one cell per ln(K/F) bucket in ``used`` (K_BINS).

    Each cell records how many quotes it holds (``n``), so the cells of a grid
    add up to the quotes it shows."""
    qb = q.assign(kb=k_bucket(q["k"]))
    used = list(used) if used is not None else k_buckets_used(q["k"])
    rows = []
    for e in exps.index:
        qe = qb[qb["expiry"] == e]
        cells = []
        for b in used:
            g = qe[qe["kb"] == b]
            if g.empty:
                empty = cell("", None, detail=f"{e} · ln(K/F) {k_bucket_label(b)}: no quotes")
                cells.append({**empty, "n": 0})
                continue
            v = value(g)
            cells.append({**cell(v["text"], colour(v["value"]), detail=detail(g, v)), "n": len(g)})
        rows.append({"label": f"{e[2:]} · {_fmt(exps[e], '.0f')}d", "cells": cells})
    return rows


# ── the surface in desk coordinates (desk_surface.py) ────────────────────────
DESK_GRID_HEADS = ("Expiry", "T (d)", "F", *(lab for _, lab, _ in desk_surface.PILLARS),
                   "RR25", "BF25", "RR10", "BF10", "ATM skew", "VS vol", "VS − ATMF",
                   "Median half-spread", "Quotes")
EXTRAP_TAG = "◇ extrapolated"


def _short_exp(e: str | None) -> str:
    return e[2:] if e else "—"


def desk_grid_view(listed: Sequence[dict], cm: Sequence[dict]) -> dict:
    """Display rows of the delta-quoted grid: listed expiries and constant maturity.

    Vol cells take a cividis tint on one scale for both tables; RR and BF the
    diverging ramp, each pair (RR25/RR10, BF25/BF10) symmetric about 0 on its
    own limit. An extrapolated pillar carries the '◇ extrapolated' tag."""
    keys = [k for k, _, _ in desk_surface.PILLARS]
    vols = [r["vol"][k] for r in [*listed, *cm] for k in keys if r["vol"][k] is not None]
    lo, hi = (min(vols), max(vols)) if vols else (0.0, 1.0)

    def lim(names: Sequence[str]) -> float:
        xs = [abs(r[n]) for r in [*listed, *cm] for n in names if r.get(n) is not None]
        return max(xs) if xs else 1.0

    rr_lim, bf_lim = lim(("rr25", "rr10")), lim(("bf25", "bf10"))

    def tint(text: str, bg: str | None, title: str = "", xp: bool = False) -> dict:
        c = {"text": text, "title": title, "xp": xp}
        if bg is not None:
            c.update(bg=bg, fg=_text_colour_on(bg))
        return c

    def row(r: dict, head: str, sub: str = "") -> dict:
        cells = [tint(_fmt(r["T_days"], ".0f"), None), tint(_fmt(r.get("forward"), ",.2f"), None)]
        for k, lab, _ in desk_surface.PILLARS:
            v, xp = r["vol"][k], r["extrap"][k]
            kk = (r.get("k") or {}).get(k)
            title = (f"{lab}: ln(K/F) {signed(kk, '.3f')}" if kk is not None else lab) + (
                f", outside the quoted ln(K/F) range {signed(r.get('k_lo'), '.3f')} to "
                f"{signed(r.get('k_hi'), '.3f')}" if xp and r.get("k_lo") is not None else
                (", beyond a bracketing expiry's quoted strikes" if xp else ""))
            cells.append(tint(_fmt(v, ".2f"), seq_colour(v, lo, hi) if v is not None else None, title, xp))
        for n, limit in (("rr25", rr_lim), ("bf25", bf_lim), ("rr10", rr_lim), ("bf10", bf_lim)):
            v = r.get(n)
            cells.append(tint(signed(v, ".2f"), div_colour(v, limit) if v is not None else None,
                              "" if v is not None else "a pillar is extrapolated or missing"))
        cells += [tint(signed(r.get("skew_vp"), ".2f"), None), tint(_fmt(r.get("vs"), ".2f"), None),
                  tint(signed(r.get("vs_minus_atmf"), ".2f"), None),
                  tint(_fmt(r.get("half_spread_vp"), ".2f"), None), tint(_fmt(r.get("n_quotes"), ","), None)]
        return {"head": head, "sub": sub, "cells": cells, "available": r.get("available", True)}

    rows_l = [row(r, r["expiry"]) for r in listed]
    rows_c = []
    for r in cm:
        if not r["available"]:
            sub = "outside the listed expiries"
        elif r["lower"] == r["upper"]:
            sub = f"= {_short_exp(r['lower'])}"
        else:
            sub = f"{_short_exp(r['lower'])} → {_short_exp(r['upper'])}"
        rows_c.append(row(r, r["label"], sub))
    n_xp = sum(1 for r in listed for k in keys if r["extrap"][k])
    return {"heads": list(DESK_GRID_HEADS), "listed": rows_l, "cm": rows_c, "vol_lo": lo, "vol_hi": hi,
            "rr_lim": rr_lim, "bf_lim": bf_lim, "n_extrap": n_xp, "extrap_tag": EXTRAP_TAG,
            # the skew column's term structure, against Heston, is on the Calibration page
            "links": {"ATM skew": "/calibration#skew-term"},
            "any_extrap": n_xp > 0 or any(r["extrap"][k] for r in cm for k in keys)}


def _state_values(cm: Sequence[dict]) -> dict[str, float | None]:
    by = {r["label"]: r for r in cm}
    m1, y1 = by.get("1M") or {}, by.get("1Y") or {}
    return {"atmf": m1.get("atmf"), "rr25": m1.get("rr25"), "bf25": m1.get("bf25"),
            "vs": m1.get("vs_minus_atmf"),
            "slope": (y1["atmf"] - m1["atmf"]) if (m1.get("atmf") is not None and y1.get("atmf") is not None)
            else None}


def surface_state_tiles(cm: Sequence[dict], listed: Sequence[dict],
                        all_cm: Sequence[Sequence[dict]] = ()) -> list[dict]:
    """The 'surface state' KPI row: 1M ATMF, RR25, BF25, VS − ATMF and the ATMF
    slope 1M → 1Y, all from the constant-maturity rows (never extrapolated),
    each with its range over every stored session's grid (``all_cm``)."""
    by = {r["label"]: r for r in cm}
    m1, y1 = by.get("1M") or {}, by.get("1Y") or {}
    every = [_state_values(c) for c in all_cm]

    def rng(key: str) -> str | None:
        return session_range(e[key] for e in every)
    span = (f"{_fmt(listed[0]['T_days'], '.0f')}–{_fmt(listed[-1]['T_days'], '.0f')} d"
            if listed else "no listed expiries")
    where = (f"between {m1['lower']} and {m1['upper']}" if m1.get("available") and m1["lower"] != m1["upper"]
             else (f"the {m1['lower']} expiry" if m1.get("available") else f"outside the listed {span}"))
    slope = (y1["atmf"] - m1["atmf"]) if (m1.get("atmf") is not None and y1.get("atmf") is not None) else None
    return [
        {"label": "1M ATMF", "value": _fmt(m1.get("atmf"), ".2f"), "unit": "%",
         "sub": f"SVI-cal at 30 days, {where}", "range": rng("atmf")},
        {"label": "1M RR25", "value": signed(m1.get("rr25"), ".2f"), "unit": "vol pts",
         "sub": "25Δ call − 25Δ put; negative: puts over calls", "range": rng("rr25")},
        {"label": "1M BF25", "value": signed(m1.get("bf25"), ".2f"), "unit": "vol pts",
         "sub": "mean of the 25Δ wings − ATMF", "range": rng("bf25")},
        {"label": "1M VS − ATMF", "value": signed(m1.get("vs_minus_atmf"), ".2f"), "unit": "vol pts",
         "sub": "variance-swap vol over ATMF: the skew premium", "range": rng("vs")},
        {"label": "ATMF slope 1M → 1Y", "value": signed(slope, ".2f"), "unit": "vol pts",
         "sub": (f"1Y {_fmt(y1.get('atmf'), '.2f')}% − 1M {_fmt(m1.get('atmf'), '.2f')}%"
                 if slope is not None else f"1Y outside the listed {span}"), "range": rng("slope")},
    ]


def surface_state_line(cm: Sequence[dict], session: str | None, ref: str | None) -> dict | None:
    """The Overview's one line of the surface state: the 1M constant-maturity row."""
    m1 = next((r for r in cm if r["label"] == "1M"), None)
    if not m1 or not m1["available"]:
        return None
    age = session_age(session, ref)
    return {"atmf": _fmt(m1["atmf"], ".2f"), "rr25": signed(m1["rr25"], ".2f"),
            "bf25": signed(m1["bf25"], ".2f"), "session": session,
            "age": ("the newest session" if age == 0 else
                    (f"{plural(age, 'session')} old" if age is not None else "age unknown"))}


# what the stored chain is, as the page badge names it (surface.json inputs.source)
CHAIN_SOURCES = {"yfinance": "Yahoo Finance close snapshot", "yahoo": "Yahoo Finance close snapshot",
                 "cboe": "Cboe"}


def chain_badge(surface_json: dict, other_expiries: Sequence[int] = ()) -> str | None:
    """'Chain: Yahoo Finance close snapshot · no quote time · 6,191 rows → 2,913
    scored · 17 expiries (other stored sessions 40–43)', from surface.json inputs
    and universe: counts only, no cause is read into a thin session."""
    inp = surface_json.get("inputs") or {}
    uni = surface_json.get("universe") or {}
    src = str(inp.get("source") or uni.get("source") or "").strip()
    if not src:
        return None
    qt = inp.get("quote_time") or uni.get("quote_time")
    parts = [f"Chain: {CHAIN_SOURCES.get(src.lower(), src)}",
             f"quotes at {str(qt).replace('T', ' ')}" if qt else "no quote time"]
    rows, scored = _num(inp.get("chain_rows")), _num(uni.get("n_universe"))
    if rows is not None and scored is not None:
        parts.append(f"{_fmt(int(rows), ',')} rows → {_fmt(int(scored), ',')} scored")
    n = _num(uni.get("n_expiries"))
    if n is not None:
        oth = sorted(int(x) for x in other_expiries)
        rng = (f"{oth[0]}" if oth[0] == oth[-1] else f"{oth[0]}–{oth[-1]}") if oth else ""
        parts.append(f"{int(n)} expiries" + (f" (other stored session{'s' if len(oth) > 1 else ''} {rng})"
                                              if oth else ""))
    return " · ".join(parts)


def session_expiries(sj: dict) -> int | None:
    """The listed expiries of a stored surface session (universe.n_expiries, else the term structure)."""
    n = _num((sj.get("universe") or {}).get("n_expiries"))
    return int(n) if n is not None else (len(sj.get("term_structure") or []) or None)


def _weight(w: float) -> str:
    return signed(w, ".0f") if abs(abs(w) - 1.0) < 1e-12 else signed(w, ".2f")


def violation_view(v: dict) -> dict:
    """One largest_mid_violations row: its constraint, its legs grouped by expiry
    ('2027-03-31 +1 P6260 −1 P6265'), a short form ('2027-03-31 P 6260/6265'),
    and the shortfall in index points and in dollars per SPX contract."""
    legs = v.get("legs") or []
    exps = list(dict.fromkeys(str(lg["expiry"])[:10] for lg in legs))
    groups = [{"expiry": e,
               "legs": [f"{_weight(float(lg['weight']))} {lg['right']}{_fmt(float(lg['strike']), '.0f')}"
                        for lg in legs if str(lg["expiry"])[:10] == e]} for e in exps]
    by = [f"{g['expiry']} " + " ".join(g["legs"]) for g in groups]
    rights = "/".join(dict.fromkeys(str(lg["right"]) for lg in legs))
    strikes = "/".join(dict.fromkeys(_fmt(float(lg["strike"]), ".0f") for lg in legs))
    short = f"{'/'.join(exps)} {rights} {strikes}"
    typ = str(v.get("type") or "")
    sf = _num(v.get("shortfall_index_points"))
    return {"type": typ, "label": typ.replace("_", " ", 1).replace("_", " "),
            "constraint": ARB_LABELS.get(typ, typ), "legs": by, "leg_groups": groups, "short": short,
            "shortfall": sf,
            "usd": sf * SPX_MULTIPLIER if sf is not None else None, "executable": bool(v.get("executable"))}


def surface_comparison(head: str, inputs: Sequence[dict]) -> dict | None:
    """The headline session's constant-maturity grid against every earlier stored
    session (Surface #surface-change): per earlier session, newest first, its
    gap in NYSE sessions (session_age) and desk_surface.cm_delta's rows; and
    every session's listed grid, for the ATM and RR25 term charts. Stored
    sessions are irregular (March to October), so a change is never called a
    day's or the previous session's."""
    grids: dict[str, dict] = {}
    for s in inputs:
        listed = desk_surface.delta_grid(s["surface"], s["quotes"], s["vix"])
        grids[s["session"]] = {"listed": listed, "cm": desk_surface.constant_maturity(listed)}
    earlier = sorted((x for x in grids if x < head), reverse=True)
    if head not in grids or not earlier:
        return None
    views = [{"session": old, "gap": session_age(old, head),
              "rows": desk_surface.cm_delta(grids[head]["cm"], grids[old]["cm"])} for old in earlier]
    return {"head": head, "default": earlier[0], "views": views, "grids": grids}


def forward_vol_series(fw: Sequence[dict]) -> list[dict]:
    """The forward ATM vol as steps between the constant-maturity points, and a
    'calendar' flag marker (never a NaN) where the forward variance is negative."""
    pts: list[list[Any]] = []
    for f in fw:
        if f["fwd_vol"] is None:
            pts.append([f["t1_days"], None])          # a gap in the steps
            continue
        pts += [[f["t1_days"], f["fwd_vol"]], [f["t2_days"], f["fwd_vol"]]]
    out = [{"name": "Forward ATM vol, SVI-cal (1M → 1Y)", "type": "line", "color": model_colour("svi_cal"),
            "width": 2, "dash": "dotted", "legend": True, "z": 3, "data": pts}]
    flags = [f for f in fw if f["flag"]]
    if flags:
        out.append({"name": "Calendar flag: negative forward variance", "type": "scatter",
                    "symbol": "triangle", "color": "breach", "size": 10, "legend": True, "z": 4,
                    "data": [[(f["t1_days"] + f["t2_days"]) / 2, f["atm1"]] for f in flags]})
    return out


# Cboe's constant maturities of the VIX family (days), when the artifact does not record one
VIX_FAMILY_DAYS = {"VIX9D": 9, "VIX": 30, "VIX3M": 93, "VIX6M": 184}


def vix_official_markers(vx: dict | None) -> list[dict]:
    """The official VIX-family closes at their constant maturities (vix_replication.json
    indices.<name>.official), for markers on the variance-swap strip."""
    out = []
    for tick, default_days in VIX_FAMILY_DAYS.items():
        row = ((vx or {}).get("indices") or {}).get(tick) or {}
        off = row.get("official")
        close = _num(off.get("close")) if isinstance(off, dict) else _num(off)
        if close is None:
            continue
        rec = row.get(row.get("headline_method") or "") or row.get("parity_forward") or {}
        days = _num(rec.get("constant_maturity_days")) if isinstance(rec, dict) else None
        out.append({"ticker": tick, "days": days if days is not None else float(default_days),
                    "close": close})
    return out


def rr_term_spec(cmp_: dict) -> dict:
    """RR25 against days to expiry (log axis) for every stored session: the headline
    solid, the others grey, the session chosen on the comparison tabs dashed."""
    head = cmp_["head"]
    series: list[dict[str, Any]] = []
    rows = []
    for s, g in sorted(cmp_["grids"].items()):
        pts = [[r["T_days"], r["rr25"]] for r in g["listed"]]
        rows += [[s, r["expiry"], _fmt(r["T_days"], ".0f"), signed(r["rr25"], ".2f")] for r in g["listed"]]
        if s == head:
            continue
        series.append({"name": f"RR25 {s} (grey)", "legendKey": "Every stored session", "legend": True,
                       "type": "line", "color": "muted", "width": 1, "z": 1, "data": pts})
        series.append({"name": f"RR25 {s}", "legend": True, "type": "line", "color": model_colour("svi_cal"),
                       "width": 1.5, "dash": True, "group": s, "z": 2, "data": pts})
    series.append({"name": f"RR25 {head}", "legend": True, "type": "line", "color": model_colour("svi_cal"),
                   "width": 2, "symbol": "circle", "size": 4, "z": 3,
                   "data": [[r["T_days"], r["rr25"]] for r in cmp_["grids"][head]["listed"]]})
    days = [r["T_days"] for g in cmp_["grids"].values() for r in g["listed"]]
    return charts.xy(
        "chart-rr-term", f"25-delta risk reversal by days to expiry, {head} and every stored session",
        x={"type": "value", "log": True, "name": "Days to expiry (log scale)", "digits": 0,
           "min": round_sig(min(days) / 1.15, 2), "max": round_sig(max(days) * 1.15, 2)},
        y={"name": "RR25 (vol pts)", "digits": 0, "tipDigits": 2, "signed": True},
        series=series, tooltip="item", csv=f"rr25_term_{head}.csv",
        describe=(f"RR25 = 25Δ call vol − 25Δ put vol of the SVI-cal surface per listed expiry: {head} "
                  "solid, the other stored sessions grey, the session chosen on the comparison tabs dashed. "
                  "A wing extrapolated beyond the quoted strikes leaves a gap."),
        table=charts.table("RR25 (vol pts) per listed expiry and stored session, SVI-cal surface",
                           ["Session", "Expiry", "T (d)#", "RR25#"], rows))


def load_surface() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "charts": {}, "surf": None, "model_heat": None,
                         "session": None, "models": [], "expiries": []}
    with _loader_boundary(d, "headline calibration"):
        path, cal = load_headline_calibration()
        d["calib_session"] = _calibration_session(cal, path)
        d["calib_file"] = Path(path).name
        d["model_heat"] = _model_surface_heat(cal["params"])
        d["model_params"] = dict(cal["params"])
    try:
        surf = artifacts.load_surface(DATA)
        surface_extras(d, surf)
    except artifacts.ArtifactError as e:
        _record_error(d, "surface artifacts (data_store/surface)", e)
    except Exception as e:
        _record_error(d, "surface views", e)
    return _clean_keep(d)


def surface_extras(d: dict, surf: dict) -> None:
    """Every Surface-page view of one session's surface artifacts."""
    d["surf"] = {k: v for k, v in surf.items() if k != "residuals"}
    if surf.get("state") != "ok":
        mh = d.get("model_heat")
        if mh and mh["points"]:
            d["surface3d"] = {"id": "surface3d", "points": mh["points"], "zmin": mh["lo"], "zmax": mh["hi"],
                              "stops": list(CIVIDIS_11), "kind": "model"}
        return
    res = surf["residuals"]
    mkt = artifacts.market_quotes(res)
    session = d["session"] = surf["session"]
    ref = _latest_market_session()
    d["stale"] = [_staleness("surface", "Surface artifacts", session, ref)]
    models = surf["models"]
    # the page badge listed none
    d["models"] = [{"key": m, "label": model_label(m), "short": model_short(m)} for m in models]
    # scores: the producer's scorecard when present, else computed on the residual table
    if surf.get("scorecard_rows"):
        d["score_rows"] = [{"key": r["model"], "label": model_label(r["model"]),
                            "short": model_short(r["model"]), "oos": r["oos"],
                            "in_sample": r.get("in_sample"), "n_params": r.get("n_params"),
                            "n_in_sample": r.get("n_in_sample"), "description": r.get("description"),
                            "dense": r.get("dense") or {}, "ex_arb": r.get("ex_arb")}
                           for r in surf["scorecard_rows"]]
        d["score_source"] = "scorecard.json"
    else:
        d["score_rows"] = [{"key": m, "label": model_label(m), "short": model_short(m),
                            "oos": artifacts.residual_stats(res, m)} for m in models]
        d["score_source"] = "residuals.parquet"
    d["score_notes"] = list((surf.get("scorecard") or {}).get("notes") or [])
    # each model's out-of-sample median |z| on every stored session (its scorecard.json)
    oos_all = artifacts.surface_oos_by_session(DATA)
    d["score_ranges"] = {m: session_range(r["median_abs_hs"] for r in oos_all if r["model"] == m)
                         for m in models}
    oos_by = {(r["session"], r["model"]): r["median_abs_hs"] for r in oos_all}
    # "same quotes" only when every model is scored on as many out-of-sample
    # quotes (the surfaces hold out alternate strikes; Heston holds out all but
    # its anchors, so their sets differ)
    n_scored = {(r.get("oos") or {}).get("n") for r in d["score_rows"]}
    d["same_quotes"] = len(n_scored) == 1 and None not in n_scored
    exps = mkt.groupby("expiry")["T_days"].median().sort_values()
    d["expiries"] = [{"expiry": e, "T_days": float(t), "n": int((mkt["expiry"] == e).sum())}
                     for e, t in exps.items()]
    d["n_quotes"] = len(mkt)
    d["median_half_spread"] = float(mkt["half_spread_vp"].median())

    # the SVI-cal surface in desk coordinates: delta pillars, RR/BF, ATM skew and
    # var-swap vol per listed expiry and at constant maturity (desk_surface.py).
    # A slice whose delta cannot be inverted raises: the page fails closed.
    sj = surf.get("surface") or {}
    if (((sj.get("surfaces") or {}).get("svi_cal") or {}).get("params") or {}).get("slices"):
        listed = desk_surface.delta_grid(sj, res, surf.get("vix"))
        cm = desk_surface.constant_maturity(listed)
        d["desk"] = desk_grid_view(listed, cm)
        # against every earlier stored session, at constant maturity
        desk_inputs = artifacts.load_surface_desk_inputs(DATA)
        d["chain_badge"] = chain_badge(sj, [n for s in desk_inputs if s["session"] != session
                                            for n in [session_expiries(s["surface"])] if n is not None])
        d["expiries_by_session"] = {s["session"]: session_expiries(s["surface"]) for s in desk_inputs}
        cmp_ = surface_comparison(session, desk_inputs)
        every_cm = [g["cm"] for g in (cmp_ or {}).get("grids", {}).values()]
        d["state_tiles"] = surface_state_tiles(cm, listed, every_cm)
        if cmp_:
            d["compare"] = {k: v for k, v in cmp_.items() if k != "grids"}
            d["charts"]["rr_term"] = rr_term_spec(cmp_)
            d["_cmp_grids"] = cmp_["grids"]

    # market IV heatmap: median mid IV of the quotes in each (expiry, ln(K/F)) bucket,
    # the open wings included; the colour ramp spans the cells shown (it spanned
    # every quote, so no cell reached the top of its legend)
    used = k_buckets_used(mkt["k"])
    qb = mkt.assign(kb=k_bucket(mkt["k"]))
    cell_iv = qb.groupby(["expiry", "kb"])["mid_iv"].median() * 100
    lo, hi = float(cell_iv.min()), float(cell_iv.max())
    d["iv_heat"] = {
        "cols": [k_bucket_label(b, compact=True) for b in used], "lo": lo, "hi": hi,
        "rows": _bucket_grid(
            mkt, exps,
            value=lambda g: {"value": float(g["mid_iv"].median() * 100),
                             "text": _fmt(float(g["mid_iv"].median() * 100), ".1f")},
            colour=lambda v: seq_colour(v, lo, hi),
            detail=lambda g, v: (f"mid IV {_fmt(v['value'], '.2f')}% (median of {len(g)}) · half-spread "
                                 f"{_fmt(float(g['half_spread_vp'].median()), '.2f')} vol pts"), used=used),
        "compact": True}
    d["iv_heat"]["n"] = sum(c["n"] for r in d["iv_heat"]["rows"] for c in r["cells"])
    pts = []
    # 3D x of a bucket: its midpoint; of an open wing, the median ln(K/F) of its quotes
    xs3 = {b: (round((K_BINS[b] + K_BINS[b + 1]) / 2, 4) if 0 < b < len(K_BINS) - 2
               else round(float(qb.loc[qb["kb"] == b, "k"].median()), 4)) for b in used}
    for e in exps.index:
        for b in used:
            g = qb[(qb["expiry"] == e) & (qb["kb"] == b)]
            pts.append([xs3[b], float(exps[e]),
                        round(float(g["mid_iv"].median() * 100), 3) if len(g) else None])
    if sum(1 for q_ in pts if q_[2] is not None) >= 9:
        d["surface3d"] = {"id": "surface3d", "points": pts, "zmin": lo, "zmax": hi,
                          "stops": list(CIVIDIS_11), "kind": "quotes"}

    # residual heatmaps, one per model, out-of-sample quotes only, in half-spreads
    d["resid_heat"] = []
    for m in models:
        q = res[res["model"] == m]
        if "in_sample" in q.columns:
            q = q[~q["in_sample"]]
        q = q[pd.to_numeric(q["model_iv"], errors="coerce").notna()]
        d["resid_heat"].append({
            "key": m, "label": model_label(m), "cols": [k_bucket_label(b, compact=True) for b in used],
            "n": len(q), "compact": True,
            "rows": _bucket_grid(
                q, exps,
                value=lambda g: {"value": float(g["error_hs"].median()),
                                 "text": signed(float(g["error_hs"].median()), ".1f")},
                colour=lambda v: div_colour(v, 5.0),
                detail=lambda g, v: (f"{signed(v['value'], '.2f')} half-spreads (median of {len(g)}) · "
                                     f"{_fmt(g['inside_spread'].astype(bool).mean() * 100, '.0f')}% "
                                     "inside"), used=used)})

    # smile slices: one chart, the expiry chosen by tabs (hash state "exp"); every
    # model at the quoted strikes, each with one legend toggle across expiries
    # and its site-wide colour; Heston and the headline surface shown at first.
    # A model's points list only IVs, in the order of the expiry's mids (every
    # model is priced on the same quotes), and take their ln(K/F) from them.
    hkey = heston_model_key(models)
    ranked = [r["model"] for r in surf.get("scorecard_rows") or []] or list(models)
    overlay = [m for m in [hkey, *ranked] if m and m in models]
    overlay = list(dict.fromkeys(overlay))
    shown = {hkey, SURFACE_HEADLINE_MODEL} & set(overlay) or set(overlay[:2])
    d["slice_models"] = [model_label(m) for m in overlay]
    d["slice_shown"] = [model_label(m) for m in overlay if m in shown]
    d["default_exp"] = (min(d["expiries"], key=lambda x: abs(x["T_days"] - 30.0))["expiry"]
                        if d["expiries"] else None)
    series, groups = [], []
    hres = res[res["model"] == hkey] if hkey else None
    n_anchor = int(hres["in_sample"].astype(bool).sum()) if (hres is not None and "in_sample" in hres) else 0
    for e in exps.index:
        q = mkt[mkt["expiry"] == e].sort_values("k")
        mid_name = f"{e} mid"
        if {"bid_iv", "ask_iv"} <= set(q.columns):
            series.append({"name": f"{e} bid–ask", "legendKey": "Bid–ask", "legend": True,
                           "type": "errorbar", "color": "muted", "group": e,
                           "data": [[round(r.k, 3), round(r.bid_iv * 100, 2), round(r.ask_iv * 100, 2)]
                                    for r in q.itertuples()]})
        series.append({"name": mid_name, "legendKey": "Mid", "legend": True, "type": "scatter",
                       "color": "headline", "size": 4, "group": e,
                       "data": [[round(r.k, 3), round(r.mid_iv * 100, 2)] for r in q.itertuples()]})
        by_model = {}
        for m in overlay:
            qm = res[(res["model"] == m) & (res["expiry"] == e)].set_index("k")["model_iv"]
            qm = qm[~qm.index.duplicated()]
            ivs = [(round(float(v) * 100, 2) if _num(v) is not None else None)
                   for v in qm.reindex(q["k"].to_numpy())]
            by_model[m] = ivs
            series.append({"name": f"{e} {model_label(m)}", "legendKey": model_label(m), "legend": True,
                           "hidden": m not in shown, "type": "line", "color": model_colour(m), "width": 1.5,
                           "group": e, "xFrom": mid_name, "data": ivs})
        if n_anchor and hres is not None:
            ha = hres[(hres["expiry"] == e) & hres["in_sample"].astype(bool)].sort_values("k")
            series.append({"name": f"{e} Heston fit anchor", "legendKey": "Heston fit anchors",
                           "legend": True, "type": "scatter", "symbol": "diamond", "size": 8, "z": 4,
                           "color": ANCHOR_COLOUR,
                           "group": e, "data": [[round(r.k, 3), round(r.mid_iv * 100, 2)]
                                                for r in ha.itertuples()]})
        # the same expiry as a table: medians by ln(K/F) bucket (the chart's data
        # equivalent; desk.js shows the selected expiry's, every one without JS)
        qb_e = q.assign(kb=k_bucket(q["k"]).to_numpy())
        rows_e = []
        for b in [b_ for b_ in used if (qb_e["kb"] == b_).any()]:
            sel = (qb_e["kb"] == b).to_numpy()
            g = qb_e[sel]
            rows_e.append([k_bucket_label(b), len(g), _fmt(g["bid_iv"].median() * 100, ".2f"),
                           _fmt(g["mid_iv"].median() * 100, ".2f"), _fmt(g["ask_iv"].median() * 100, ".2f")]
                          + [_fmt(pd.Series([v for v, s_ in zip(by_model[m], sel) if s_],
                                            dtype=float).median(), ".2f") for m in overlay])
        groups.append({"group": e, "shown": e == d["default_exp"], **charts.table(
            f"{e} ({_fmt(exps[e], '.0f')} days): median implied vol (%) by ln(K/F) bucket",
            ["ln(K/F)", "Quotes#", "Bid#", "Mid#", "Ask#"] + [model_short(m) + "#" for m in overlay],
            rows_e)})
    hs_by_exp = mkt.groupby("expiry")["half_spread_vp"].median()
    slice_tbl = charts.table("Quotes per expiry",
                             ["Expiry", "Days#", "Quotes#", "Median half-spread (vol pts)#"],
                             [[x["expiry"], _fmt(x["T_days"], ".0f"), x["n"],
                               _fmt(float(hs_by_exp[x["expiry"]]), ".2f")] for x in d["expiries"]])
    slice_tbl["groups"] = groups
    d["charts"]["slices"] = charts.xy(
        "chart-slices", f"Smile slice with bid–ask quotes and model overlays, {session}",
        x={"type": "value", "name": "ln(K/F)", "digits": 2, "signed": True},
        y={"name": "Implied vol (%)", "digits": 0, "tipDigits": 2},
        series=series, tooltip="item", csv=f"smile_slices_{session}.csv",
        marks={"x": [{"value": 0, "label": "F", "color": "muted"}]},
        describe=("One expiry at a time (choose it above the chart): bid–ask bars, mids"
                  + (" and the Heston fit's anchors (diamonds)" if n_anchor else "")
                  + f"; model IVs at the quoted strikes for {', '.join(d['slice_models'])} "
                    f"({' and '.join(d['slice_shown'])} shown, the others behind their toggles). "
                    "The dashed line is the forward, ln(K/F) = 0."),
        table=slice_tbl)

    # out-of-sample median |model − mid| by expiry, one line per model, on the
    # same quotes as the scorecard beside it (the scorecard's by_expiry rows mix
    # in the fitted quotes, so they are not used here)
    if d["score_rows"]:
        cats = [x["expiry"] for x in d["expiries"]]
        labels = [f"{e[2:]} · {_fmt(t, '.0f')}d" for e, t in zip(cats, [x["T_days"] for x in d["expiries"]])]
        ser, tbl = [], []
        for r in d["score_rows"]:
            qo = res[res["model"] == r["key"]]
            if "in_sample" in qo.columns:
                qo = qo[~qo["in_sample"].astype(bool)]
            z = pd.to_numeric(qo["error_hs"], errors="coerce").abs()
            med = z.groupby(qo["expiry"]).median()
            cnt = z.groupby(qo["expiry"]).count()
            vals = [_num(med.get(e)) if cnt.get(e, 0) else None for e in cats]
            ser.append({"name": r["label"], "type": "line", "color": model_colour(r["key"]),
                        "symbol": "circle", "size": 5, "width": 1.5, "legend": True, "data": vals})
            tbl.append([r["label"]] + [_fmt(v, ".1f") for v in vals])
        d["charts"]["by_expiry"] = charts.xy(
            "chart-by-expiry",
            "Median |model − mid| in half-spreads by expiry, out-of-sample quotes, every model",
            x={"type": "category", "categories": labels, "name": "", "interval": "auto", "rotate": 40},
            y={"name": "Median |z| (half-spreads)", "digits": 0, "tipDigits": 2, "zero": True, "log": False},
            series=ser, xs=labels, csv=f"median_abs_hs_by_expiry_{session}.csv", grid={"bottom": 72},
            marks={"y": [{"value": 1.0, "label": "inside the spread", "color": "ok"}]},
            describe=("Each model's out-of-sample quotes of each expiry, the scorecard's quote sets; "
                      "1 half-spread is the edge of the quoted bid–ask."),
            table=charts.table("Median |model − mid| (half-spreads) by expiry, out-of-sample quotes",
                               ["Model"] + [c + "#" for c in labels], tbl))

    # ATM term structure (surface.json): market ATM vol and the eSSVI ATM vol and skew
    ts = (surf.get("surface") or {}).get("term_structure") or []
    cmp_grids = d.pop("_cmp_grids", None)
    if ts:
        ts = sorted(ts, key=lambda t: float(t["T_days"]))
        atm_series: list[dict[str, Any]] = [
            {"name": "ATM mid (quotes)", "type": "scatter", "color": "headline", "size": 6, "legend": True,
             "data": [[float(t["T_days"]), _num(t.get("atm_vol_mid_quotes"))] for t in ts
                      if _num(t.get("atm_vol_mid_quotes")) is not None]},
            {"name": "eSSVI ATM", "type": "line", "color": model_colour("essvi"), "width": 1.5,
             "legend": True, "data": [[float(t["T_days"]), _num(t.get("atm_vol_essvi"))] for t in ts
                      if _num(t.get("atm_vol_essvi")) is not None]}]
        atm_groups = []
        if cmp_grids and session in cmp_grids:
            # the SVI-cal ATM of this session and of the earlier session chosen on the
            # comparison tabs (#surface-change), each named with its date
            atm_series.append({"name": f"SVI-cal ATM {session}", "type": "line",
                               "color": model_colour("svi_cal"), "width": 1.5, "legend": True,
                               "data": [[r["T_days"], r["atmf"]] for r in cmp_grids[session]["listed"]]})
            for old in (v["session"] for v in d["compare"]["views"]):
                lst = cmp_grids[old]["listed"]
                atm_series.append({"name": f"SVI-cal ATM {old}", "type": "line",
                                   "color": model_colour("svi_cal"), "width": 1.2, "dash": True,
                                   "legend": True, "group": old,
                                   "data": [[r["T_days"], r["atmf"]] for r in lst]})
                atm_groups.append({"group": old, "shown": old == d["compare"]["default"], **charts.table(
                    f"SVI-cal ATM vol of {old}, the earlier session compared",
                    ["Expiry", "Days#", "ATM (%)#"],
                    [[r["expiry"], _fmt(r["T_days"], ".0f"), _fmt(r["atmf"], ".2f")] for r in lst])})
        if cmp_grids and session in cmp_grids:
            d["forward_vols"] = desk_surface.forward_vols(cmp_grids[session]["cm"])
            atm_series += forward_vol_series(d["forward_vols"])
        atm_tbl = charts.table("ATM term structure", ["Expiry", "Days#", "ATM mid (%)#", "eSSVI ATM (%)#",
                                                      "eSSVI ATM skew#", "Parity r#", "Parity q#", "Quotes#"],
                               [[t.get("expiry"), _fmt(_num(t.get("T_days")), ".0f"),
                                 _fmt(_num(t.get("atm_vol_mid_quotes")), ".2f"),
                                 _fmt(_num(t.get("atm_vol_essvi")), ".2f"),
                                 signed(_num(t.get("atm_skew_essvi")), ".3f"),
                                 _pct_prose(_num(t.get("r")), 2, signed=False),
                                 _pct_prose(_num(t.get("q")), 2, signed=False),
                                 _fmt(t.get("n_quotes"), ",")] for t in ts])
        if atm_groups:
            atm_tbl["groups"] = atm_groups
        d["charts"]["atm_ts"] = charts.xy(
            "chart-atm-ts", "ATM implied vol by expiry: quotes, the eSSVI and SVI-cal fits"
            + (", and an earlier session" if atm_groups else ""),
            x={"type": "value", "name": "Days to expiry", "digits": 0, "min": 0},
            y={"name": "ATM vol (%)", "digits": 1, "tipDigits": 2},
            series=atm_series, tooltip="axis", csv=f"atm_term_structure_{session}.csv",
            describe=("ATM implied vol per quoted expiry, from the mids at the parity forward and from the "
                      "eSSVI and SVI-cal surfaces, and the forward ATM vol between the constant maturities "
                      "1M, 2M, 3M, 6M and 1Y as dotted steps"
                      + ("; the dashed line is the SVI-cal ATM of the earlier session chosen on the "
                         "comparison tabs, named with its date." if atm_groups else ".")),
            table=atm_tbl)

    # static-arbitrage screen and repair (surface.json)
    arb = (surf.get("surface") or {}).get("arbitrage") or {}
    if arb:
        keys = list((arb.get("n_rows") or {}).keys()) or list((arb.get("mid_violations") or {}).keys())
        d["arb_rows"] = [{"key": k, "label": ARB_LABELS.get(k, k.replace("_", " ")),
                          "rows": (arb.get("n_rows") or {}).get(k),
                          "mid": (arb.get("mid_violations") or {}).get(k),
                          "mid_tick": (arb.get("mid_violations_gt_tick") or {}).get(k),
                          "exec": (arb.get("executable_violations") or {}).get(k),
                          "max_mid": (arb.get("max_mid_violation_index_points") or {}).get(k)} for k in keys]
        d["arb"] = {k: arb.get(k) for k in ("n_quotes", "n_quotes_in_executable_violation",
                                            "n_quotes_in_mid_violation", "constraint_set")}
        d["arb_repair"] = arb.get("repair") or {}
        lv = [violation_view(v) for v in arb.get("largest_mid_violations") or [] if isinstance(v, dict)]
        d["largest_violations"] = sorted(lv, key=lambda x: -(x["shortfall"] or 0.0))
        mx = [x for x in ((arb.get("max_mid_violation_index_points") or {}).values()) if _num(x) is not None]
        d["arb_max_mid"] = max(mx) if mx else None
        d["arb_mid_rows"] = sum(int(r["mid"]) for r in d["arb_rows"] if _num(r["mid"]) is not None)
        d["arb_surfaces"] = [{"label": model_label(k), "dense": (v or {}).get("dense_grid") or {}}
                             for k, v in ((surf.get("surface") or {}).get("surfaces") or {}).items()]
        d["fit_set"] = (surf.get("surface") or {}).get("fit_set") or {}
        d["universe"] = (surf.get("surface") or {}).get("universe") or {}
        d["inputs"] = (surf.get("surface") or {}).get("inputs") or {}

    # VIX replicated from the same chain (vix_replication.json)
    vx = surf.get("vix")
    if vx:
        d["vix_rows"] = []
        for tick in ("VIX9D", "VIX", "VIX3M", "VIX6M"):
            row = (vx.get("indices") or {}).get(tick)
            if not row:
                continue
            off = row.get("official")
            off_close = _num(off.get("close")) if isinstance(off, dict) else _num(off)
            d["vix_rows"].append({"ticker": tick, "replicated": _num(row.get("replicated")),
                                  "official": off_close, "gap": _num(row.get("gap_vs_official")),
                                  "method": row.get("headline_method"),
                                  "in_range": row.get("official_inside_day_range"),
                                  "day_range": ([_num(off.get("low")), _num(off.get("high"))]
                                                if isinstance(off, dict) and off.get("low") is not None
                                                else None)})
        d["vix_beta"] = vx.get("beta") or {}
        # The replication's ATM / VIX interpolates the ATM mids of the VIX's own
        # two terms with its constant-maturity weights; the backtest's β (Research)
        # is the 30-day ATM-forward Black-76 vol over the VIX. Both are shown,
        # each under its own name, so no two pages give one label two values.
        d["beta_estimator"] = {}
        try:
            d["beta_estimator"] = {c["session"]: c["ratio"] for c in beta_estimation(None)["chains"]
                                   if c["kind"] == "DB chain" and c["ratio"] is not None}
        except (OSError, ValueError, KeyError, ImportError, sqlite3.Error) as exc:
            # supplementary: the replication's own figure stands, the comparison is left out
            _logger.warning("β estimator ratios unavailable for the surface page: %s", exc)
        d["vix_notes"] = list(vx.get("gap_notes") or [])
        d["vix_deviations"] = list(vx.get("deviations") or [])
        d["vix_markers"] = vix_official_markers(vx)
        d["vix_calc_time"] = vx.get("calculation_time_ny")
        # each index's two terms and its gap split as gap_attribution splits it (#vix-terms-table)
        d["vix_terms"] = [{**r, "surface_label": model_short(r["surface"]) if r["surface"] else None}
                          for r in desk_surface.vix_terms(vx)]
        d["strip"] = [{"expiry": r.get("expiry"), "days": _num(r.get("days")),
                       "vs": _num(r.get("var_swap_vol")), "atm": _num(r.get("atm_vol_quotes")), "diff": _num(r.get("vs_minus_atm_vp")),
                       "ratio": _num(r.get("atm_over_vs")), "settlement": r.get("settlement"),
                       "n": (r.get("n_puts") or 0) + (r.get("n_calls") or 0)}
                      for r in sorted(vx.get("strip") or [], key=lambda r: _num(r.get("days")) or 0)]
        if d["strip"]:
            d["charts"]["strip"] = charts.xy(
                "chart-strip", "Variance-swap vol and ATM vol by expiry, same chain",
                x={"type": "value", "name": "Days to expiry", "digits": 0, "min": 0},
                y={"name": "Vol (%)", "digits": 0, "tipDigits": 2},
                # market-derived lines, not models: neither takes a model's colour
                series=[{"name": "Variance-swap vol (Cboe formula)", "type": "line", "color": "info",
                         "symbol": "circle", "size": 5, "width": 1.5, "legend": True,
                         "data": [[r["days"], r["vs"]] for r in d["strip"]
                                  if r["vs"] is not None and r["days"] is not None]},
                        {"name": "ATM vol (quotes)", "type": "line", "color": "headline", "symbol": "diamond",
                         "size": 5, "width": 1.5, "dash": True, "legend": True,
                         "data": [[r["days"], r["atm"]] for r in d["strip"]
                                  if r["atm"] is not None and r["days"] is not None]}]
                + ([{"name": "Cboe close (16:15 ET): VIX9D, VIX, VIX3M, VIX6M", "type": "scatter",
                     "color": "headline", "hollow": True, "size": 10, "legend": True, "z": 4,
                     "data": [[m_["days"], m_["close"]] for m_ in d["vix_markers"]]}]
                   if d["vix_markers"] else []),
                tooltip="axis", csv=f"varswap_strip_{session}.csv",
                describe=("The single-term variance-swap vol of every stored expiry against its ATM vol; "
                          "the gap is the skew premium."
                          + (" Hollow circles: the official VIX-family closes at their constant maturities "
                             "(9, 30, 93 and 184 days), where the replication's gaps show against the strip."
                             if d["vix_markers"] else "")),
                table=charts.table("Variance-swap strip (vols in %; var-swap − ATM in vol pts)",
                                   ["Expiry", "Days#", "Var-swap vol#", "ATM vol#",
                                    "Var-swap − ATM (vol pts)#",
                                    "ATM / var-swap#", "Strikes#"],
                                   [[r["expiry"], _fmt(r["days"], ".0f"), _fmt(r["vs"], ".2f"),
                                     _fmt(r["atm"], ".2f"), signed(r["diff"], ".2f"), _fmt(r["ratio"], ".3f"),
                                     _fmt(r["n"], ",")] for r in d["strip"]]))

    # every session the surface report has covered (index.json)
    idx = surf.get("index") or []
    if idx:
        mk = list(models)
        d["history_models"] = [{"key": m, "label": model_short(m)} for m in mk]
        d["surface_history"] = [{"session": r["session"], "n": r.get("n_quotes"),
                                 "arb_out": r.get("arbitrage_quotes_outside_spread"),
                                 "vix_rep": _num(r.get("vix_replicated")),
                                 "vix_off": _num(r.get("vix_official")),
                                 "beta": _num(r.get("atm_over_vix_official")),
                                 # the out-of-sample median |z| of the scorecard, as every headline
                                 "oos": {m: oos_by.get((r["session"], m)) for m in mk}} for r in idx]
        d["vix_gap_range"] = session_range(
            (h["vix_rep"] - h["vix_off"]) for h in d["surface_history"]
            if h["vix_rep"] is not None and h["vix_off"] is not None)


def _clean_keep(d: dict) -> dict:
    """_clean for a page dict whose chart specs are already JSON-ready."""
    specs = d.pop("charts", {})
    out = _clean(d)
    out["charts"] = specs
    return out


# ── Corrections page ─────────────────────────────────────────────────────────
CORRECTIONS_PATH = DASH / "corrections.json"


def concise_label(label: str) -> str:
    """A step name without its trailing parenthetical ('R2 gate = upstream default
    (adaptive (rolling 80th pct))' -> 'R2 gate = upstream default'): the axis label
    desk.js falls back to where the full name does not fit its band. The tooltip
    and the chart's table keep the full name."""
    return re.sub(r"\s*\((?:[^()]|\([^()]*\))*\)\s*$", "", label).strip() or label


def waterfall_spec(bridge: list[dict]) -> dict:
    """C22 waterfall as stacked bars: an invisible base plus the step, in $000s.

    Every level of this bridge is a loss, so the bars hang below zero: the base
    spans from 0 to the end of the step nearer zero and the visible bar covers
    the step. A bridge that crosses zero is drawn as plain step bars instead.

    The stacked encodings are drawing geometry, not data (a step that raised P&L
    is a negative segment below zero), so the spec carries its own CSV rows
    (step, Δ P&L, P&L after) and tooltip lines, and each step bar's tip is set
    only where that bar is drawn. Bars run horizontally: seventeen step names
    collided when rotated under a vertical chart on a phone. desk.js wraps each
    name into the lines its band holds (labelShare / labelMax bound the label
    column; labelsShort are the concise names it falls back to), so no name is
    cut with an ellipsis as it was ('PDV spread: 21-day RMS-vol forecast (daily pur...').
    """
    start = float(bridge[0]["pnl"]) - float(bridge[0]["d_pnl"])
    levels = [start] + [float(r["pnl"]) for r in bridge]
    cats = ["C21 published"] + [str(r["label"]) for r in bridge] + ["C22 canonical"]
    k = 1e3
    base: list[float | None] = [None]
    raise_: list[float | None] = [None]
    lower: list[float | None] = [None]
    total: list[float | None] = [start / k]
    tip_raise: list[float | None] = [None]
    tip_lower: list[float | None] = [None]
    after: list[float | None] = [start / k]
    all_neg = all(v <= 0 for v in levels)
    for r, before in zip(bridge, levels[:-1]):
        dp = float(r["d_pnl"])
        level = before + dp
        if all_neg:
            base.append(max(before, level) / k)
            raise_.append(-abs(dp) / k if dp > 0 else None)
            lower.append(-abs(dp) / k if dp < 0 else None)
        else:
            base.append(None)
            raise_.append(dp / k if dp > 0 else None)
            lower.append(dp / k if dp < 0 else None)
        total.append(None)
        tip_raise.append(dp / k if dp > 0 else None)
        tip_lower.append(dp / k if dp < 0 else None)
        after.append(level / k)
    base.append(None)
    raise_.append(None)
    lower.append(None)
    total.append(levels[-1] / k)
    tip_raise.append(None)
    tip_lower.append(None)
    after.append(levels[-1] / k)
    spec = charts.xy(
        "chart-waterfall",
        "C22 waterfall: portfolio P&L from the published C21 run to the canonical run ($000s)",
        x={"type": "category", "categories": cats, "interval": 0},
        y={"name": "Portfolio P&L ($000s)", "digits": 0, "tipDigits": 1, "signed": True, "zero": True,
           "max": 0 if all_neg else None},
        series=[
            {"name": "base", "type": "bar", "stack": "w", "invisible": True, "data": base},
            {"name": "Step raises P&L", "type": "bar", "stack": "w", "color": "pos", "legend": True,
             "data": raise_, "tip": tip_raise},
            {"name": "Step lowers P&L", "type": "bar", "stack": "w", "color": "neg", "legend": True,
             "data": lower, "tip": tip_lower},
            {"name": "Run total", "type": "bar", "stack": "w", "color": "headline", "fill": 0.35,
             "legend": True, "data": total},
        ],
        xs=cats, tooltip="axis", horizontal=True, grid={"bottom": 30, "left": 160, "top": 8},
        csv="c22_waterfall.csv",
        describe=(f"From {_usd_k(start)} (C21 as published) to {_usd_k(levels[-1])} in {len(bridge)} "
                  "cumulative steps; each bar is one correction's effect on portfolio P&L."),
        table=charts.table("C22 bridge: each correction applied in turn ($)",
                           ["Row#", "Correction", "Δ P&L#", "P&L after#", "Sharpe after#"],
                           [["—", "C21 as published", "", _usd(start), ""]]
                           + [[r["row"], r["label"], _usd(r["d_pnl"], signed=True), _usd(r["pnl"]),
                               _fmt(r["sharpe"], ".2f")] for r in bridge]))
    spec["labelShare"], spec["labelMax"] = 0.45, 380      # desk.js wraps the step names in that column
    spec["labelFont"] = "sans"                             # names, not numbers
    short = [concise_label(c) for c in cats]
    if short != cats:
        spec["labelsShort"] = short
    spec["tipRows"] = [{"name": "P&L after", "data": round_sig(after), "digits": 1, "signed": True}]
    spec["csv_rows"] = {
        "columns": ["row", "step", "d_pnl_usd", "pnl_after_usd", "sharpe_after"],
        "rows": ([["", "C21 published", "", round(start, 2), ""]]
                 + [[int(r["row"]), str(r["label"]), round(float(r["d_pnl"]), 2), round(float(r["pnl"]), 2),
                     round(float(r["sharpe"]), 4)] for r in bridge])}
    return spec


def load_corrections() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "charts": {}, "rounds": []}
    try:
        reg = json.loads(CORRECTIONS_PATH.read_text())
        d["repo"] = reg["repo"]
    except Exception as e:
        _record_error(d, "corrections registry (dashboard/corrections.json)", e)
        return _clean_keep(d)
    # The C22 evidence (bridge rows, β chains, R2 shares, the pre-C22 spread
    # replay, the S3 null) is the backtest page's own computation, so the two
    # pages can never quote different numbers.
    p4 = load_page4()
    for err in p4.get("errors") or []:
        d["errors"].append(err)
    d["p4"] = {k: p4.get(k) for k in ("bridge", "beta", "beta_est", "r2_share", "pre_c22_spread", "kpi",
                                      "sleeves", "clf_eval", "total_pnl", "final_nav", "start_date",
                                      "end_date")}
    bridge = None
    with _loader_boundary(d, "C22 bridge (c22_bridge.csv)"):
        bridge = _read_csv_rows(DATA / "backtest" / "c22_bridge.csv")
        if not bridge:
            raise ValueError("empty C22 bridge")
        bridge = sorted(bridge, key=lambda r: int(r["row"]))
    final: dict[str, float | None] | None = None
    with _loader_boundary(d, "backtest engine results (full_results.pkl)"):
        em = (_load_pickle(DATA / "backtest" / "full_results.pkl").get("metrics") or {})
        final = {"cum_ret": _num(em.get("cumulative_return")), "sharpe": _num(em.get("sharpe"))}
        if bridge and abs(float(bridge[-1]["sharpe"]) - (final["sharpe"] or 0.0)) > 1e-6:
            raise ValueError("c22_bridge.csv does not end at the stored run's Sharpe")
    # The C23 data-correction waterfall: the C22 published run restated row by row
    # on the Cboe record (c23_bridge.py), shown in the C23 postmortem.
    d["c23_bridge"] = None
    with _loader_boundary(d, "C23 bridge (c23_bridge.csv)"):
        c23 = sorted(_read_csv_rows(DATA / "backtest" / "c23_bridge.csv"), key=lambda r: int(r["row"]))
        if not c23:
            raise ValueError("empty C23 bridge")
        final_sharpe = (final or {}).get("sharpe")
        if final_sharpe is not None and abs(float(c23[-1]["sharpe"]) - final_sharpe) > 1e-6:
            raise ValueError("c23_bridge.csv does not end at the stored run's Sharpe")
        d["c23_bridge"] = [{"row": int(r["row"]), "step": r["step"], "label": r["label"],
                            "pnl": float(r["pnl"]),
                            "d_pnl": float(r["d_pnl"]), "sharpe": float(r["sharpe"])} for r in c23]
    steps = {r["step"]: r for r in bridge or []}
    d["c21_pnl"] = (float(bridge[0]["pnl"]) - float(bridge[0]["d_pnl"])) if bridge else None
    for rnd in reg["rounds"]:
        r = dict(rnd)
        r["findings"] = []
        for finding in rnd["findings"]:
            f = dict(finding)
            if f.get("c23_bridge_steps") and d.get("c23_bridge"):
                c23s = {str(x["step"]): x for x in d["c23_bridge"]}
                rows23 = [c23s[s] for s in f["c23_bridge_steps"] if s in c23s]
                if len(rows23) != len(f["c23_bridge_steps"]):
                    _record_error(d, "corrections registry", ValueError(
                        f"{rnd['id']}: C23 bridge step(s) {f['c23_bridge_steps']} not all in c23_bridge.csv"))
                f["c23_rows"] = [x["row"] for x in rows23]
                f["c23_d_pnl"] = float(sum(x["d_pnl"] for x in rows23))
            if f.get("bridge_steps") and steps:
                rows = [steps[s] for s in f["bridge_steps"] if s in steps]
                if len(rows) != len(f["bridge_steps"]):
                    _record_error(d, "corrections registry", ValueError(
                        f"{rnd['id']}: bridge step(s) {f['bridge_steps']} not all in c22_bridge.csv"))
                f["bridge_rows"] = [int(x["row"]) for x in rows]
                f["d_pnl"] = float(sum(float(x["d_pnl"]) for x in rows))
                first = min(rows, key=lambda x: int(x["row"]))
                last = max(rows, key=lambda x: int(x["row"]))
                prev = [x for x in bridge if int(x["row"]) == int(first["row"]) - 1]
                f["sharpe_from"] = _num(prev[0]["sharpe"]) if prev else None
                f["sharpe_to"] = _num(last["sharpe"])
            r["findings"].append(f)
        r["commit_links"] = [{"sha": c, "short": c[:7], "url": f"{reg['repo']}/commit/{c}"}
                             for c in rnd["commits"]]
        r["n_tests"] = sum(len(f.get("tests") or []) for f in rnd["findings"])
        d["rounds"].append(r)
    # the page's data date (sitemap lastmod, X-Dashboard-Data-Date): its newest round
    d["latest_round_date"] = max((str(r.get("date_end") or r["date"]) for r in reg["rounds"]), default=None)
    if bridge:
        d["charts"]["waterfall"] = waterfall_spec(bridge)
    pts = reg["headline_series"]["points"]
    hs = []
    for ptn in pts:
        v = ptn.get("cum_ret_pct")
        cum = final.get("cum_ret") if final else None
        if v is None and ptn.get("from_artifact") and cum is not None:
            v = cum * 100.0
        hs.append((ptn["round"], v))
    d["charts"]["headline"] = charts.xy(
        "chart-headline", "Published portfolio cumulative return after each round",
        x={"type": "category", "categories": [r for r, _ in hs], "interval": 0},
        y={"name": "Cumulative return (%)", "digits": 0, "tipDigits": 1, "signed": True, "tickUnit": "%",
           "unit": "%"},
        series=[{"name": "Published cumulative return", "type": "line", "color": "headline",
                 "symbol": "circle", "size": 7, "data": [v for _, v in hs]}],
        xs=[r for r, _ in hs], csv="headline_by_round.csv", grid={"top": 12},
        describe="C17 to C22 as published at the time; C23 from the stored canonical run.",
        table=charts.table("Published cumulative return by round (%)", ["Round", "Cumulative return#"],
                           [[r, signed(v, ".1f", "%")] for r, v in hs]))
    d["headline_trend"] = headline_trend_sentence(hs)
    d["n_findings"] = sum(len(r["findings"]) for r in d["rounds"])
    d["register"] = findings_register(d["rounds"])
    d["n_tests"] = sum(r["n_tests"] for r in d["rounds"])
    return _clean_keep(d)


def findings_register(rounds: Sequence[dict]) -> list[dict]:
    """One row per finding of every round, for the register above the postmortems:
    its id (the registry's own, else <round>.<n>), status, the portfolio P&L
    effect where a bridge CSV measures it (c22_bridge.csv rows named by
    bridge_steps, c23_bridge.csv rows by c23_bridge_steps), the number of tests
    that pin it and the anchor of its postmortem entry."""
    out = []
    for r in rounds:
        for n, f in enumerate(r["findings"], 1):
            pnl_ = f.get("d_pnl") if f.get("bridge_rows") else f.get("c23_d_pnl")
            src = ("c22_bridge.csv row" + ("s " if len(f["bridge_rows"]) > 1 else " ")
                   + ", ".join(str(x) for x in f["bridge_rows"]) if f.get("bridge_rows")
                   else ("c23_bridge.csv rows " + ", ".join(str(x) for x in f["c23_rows"])
                         if f.get("c23_rows") else None))
            out.append({"id": f.get("id") or f"{r['id']}.{n}", "round": r["id"], "n": n,
                        "flaw": f.get("flaw"), "status": f.get("status"), "pnl": pnl_, "pnl_src": src,
                        "n_tests": len(f.get("tests") or []), "anchor": finding_anchor(r["id"], n)})
    return out


def finding_anchor(round_id: str, n: int) -> str:
    """The id of a finding's entry on the Corrections page: 'c16-f4'."""
    return f"{round_id.lower()}-f{n}"


app.jinja_env.globals["finding_anchor"] = finding_anchor


def headline_trend_sentence(series: Sequence[tuple[str, Any]], tol: float = 0.05) -> str:
    """What the corrections did to the published result, read off the series.

    The page said "every successive fix made the backtest worse" beside a chart
    on which C22 raised it from −26.1% to −21.6%: the sentence now names every
    round that raised the result, and says "every" only when none did.
    """
    pts = [(str(r), float(v)) for r, v in series if _num(v) is not None]
    if len(pts) < 2:
        return ""
    steps = list(itertools.pairwise(pts))
    ups = [(a, b) for a, b in steps if b[1] - a[1] > tol]
    downs = [(a, b) for a, b in steps if b[1] - a[1] < -tol]
    (r0, v0), (rn, vn) = pts[0], pts[-1]
    span = f"from {signed(v0, '.1f', '%')} after {r0} to {signed(vn, '.1f', '%')} after {rn}"
    loss = all(v < 0 for _, v in pts)
    tail = "; no round restored a positive return" if loss else ""
    if not ups:
        return (f"No correction raised the published result ({span}){tail}; "
                "the original edge was a stack of artifacts.")
    up_txt = ", ".join(f"{b[0]} raised it from {signed(a[1], '.1f', '%')} to {signed(b[1], '.1f', '%')}"
                       for a, b in ups)
    lead = "Most corrections lowered" if len(downs) > len(ups) else "The corrections moved"
    return f"{lead} the published result ({span}); {up_txt}{tail}."


# ── Model risk page ──────────────────────────────────────────────────────────
INVENTORY_PATH = DASH / "model_inventory.json"
# The monitors' models, by their IDs in the canonical inventory (model_inventory.json,
# the IDs docs/model_validation uses): the dashboard had its own numbering, so
# "M4" was the rule labeller here and the joint calibrator in the pack.
INV_CALIBRATOR, INV_GREEKS, INV_BETA, INV_LABELS, INV_CLASSIFIER = "M4", "M6", "M7", "M10", "M11"
INV_VX = "M5"
VX_LAG_CHECK = "VX futures table lag behind the market DB (NYSE sessions)"
CHAIN_VENDORS = {"yfinance": "V1", "yahoo": "V1", "cboe": "V2"}


def chain_vendor(source: Any) -> str:
    """The vendor-input ID of a stored chain's source ('Data' when it is not recorded)."""
    return CHAIN_VENDORS.get(str(source or "").lower(), "Data")

# Monitoring thresholds: (green if value <= g, amber if <= a, else red) unless
# stated otherwise. They are this page's working limits for a research system,
# stated so a reader can disagree with them; they are not regulatory limits.
MONITOR_LIMITS = {
    "calib_age": (1, 4),            # NYSE sessions since the headline fit
    "params_at_bound": (0, 1),      # parameters on a box bound
    "oos_rmse": (1.5, 3.0),         # vol pts, all out-of-sample quotes
    "short_oos_rmse": (2.0, 5.0),   # vol pts, worst expiry within 30 days
    "vix_rmse": (0.5, 1.0),         # index points
    "beta_drift": (0.05, 0.10),     # |latest chain ATM/VIX − frozen β|
    "label_lag": (0, 1),            # sessions the market DB lags the labels
    "median_abs_hs": (2.0, 5.0),    # half-spreads, Heston out-of-sample median |model − mid|
    "vix_rep_gap": (0.5, 1.5),      # index pts, |VIX replicated from the chain − Cboe close|
    "arb_exec_share": (2.0, 5.0),   # % of scored quotes in an executable static-arbitrage row
}


def _rag(value: float | None, limits: tuple) -> str:
    if value is None:
        return "na"
    g, a = limits
    return "ok" if value <= g else ("watch" if value <= a else "breach")


def vx_futures_oos_text(rows: list[dict] | None = None) -> str:
    """M5's limitation, from the gated fits: the range of their out-of-sample VX
    futures RMSE (losses.vx_futures_rmse_oos, index pts) and the sessions that
    store none. It was typed into model_inventory.json as '1.85–3.18'."""
    rows = rows if rows is not None else scan_calibrations()
    have: list[tuple[str, float]] = []
    none: list[tuple[str, None]] = []
    for r in rows:
        if r["legacy"] is not False or not r["ok"] or not r["session"]:
            continue
        v = _num(_cal_losses(_load_pickle(r["path"])).get("vx_futures_rmse_oos"))
        if v is not None:
            have.append((r["session"], v))
        else:
            none.append((r["session"], None))
    if not have:
        return "an unmeasured amount: no gate-passing fit records VX futures out of sample"
    vals = [v for _, v in have]
    rng = _fmt(vals[0], ".2f") if len(vals) == 1 else f"{_fmt(min(vals), '.2f')}–{_fmt(max(vals), '.2f')}"
    tail = f"; none stored for {', '.join(s for s, _ in sorted(none))}" if none else ""
    last = vx_last_date()
    src = ("; settlements from vix_futures_daily in data_store/vol_system.db"
           + (f", through {last}" if last else ""))
    return f"{rng} index pts RMSE out of sample ({plural(len(have), 'session')}{tail}{src})"


# Where each check's evidence is shown (Model Risk links each value there, and
# that panel links back with a 'Monitored' chip): page file and element id.
CHECK_EVIDENCE = {
    "calib_age": ("calibration.html", "history"),
    "params_at_bound": ("calibration.html", "identifiability"),
    "oos_rmse": ("calibration.html", "oos"),
    "short_oos_rmse": ("calibration.html", "oos"),
    "vix_rmse": ("calibration.html", "vixleg"),
    "feller": ("calibration.html", "feller"),
    "median_abs_hs": ("surface.html", "scorecard"),
    "greeks_source": ("risk.html", "greeks-grid"),
    "beta_drift": ("research.html", "beta-chains"),
    "label_lag": ("index.html", "state-regime"),
    "vx_lag": ("calibration.html", "vixleg"),
    "classifier": ("research.html", "research-classifier"),
    "vix_rep_gap": ("surface.html", "vix-replication"),
    "arb_exec_share": ("surface.html", "arbitrage"),
}


def worst_short_expiry(cal: dict, max_days: float = 30.0) -> dict | None:
    """The fit's worst out-of-sample expiry within ``max_days`` (oos_rows_of)."""
    short = [r for r in oos_rows_of(cal) if r["T_days"] is not None and r["T_days"] <= max_days]
    return max(short, key=lambda r: r["rmse"]) if short else None


def fit_check_values(cal: dict) -> dict:
    """The gated-fit checks of one fit: today's row and every stored session's dot
    are this function of their own pickle."""
    losses = _cal_losses(cal)
    pinned = cal.get("pinned_params")
    w = worst_short_expiry(cal)
    return {"params_at_bound": float(len(pinned)) if pinned is not None else None,
            "oos_rmse": _num(losses.get("spx_oos_iv_rmse")),
            "short_oos_rmse": w["rmse"] if w else None, "short_expiry": w["expiry"] if w else None,
            "vix_rmse": _num(losses.get("vix_index_rmse", losses.get("vix_futures_rmse")))}


def surface_check_values(surf: dict) -> dict:
    """The surface-report checks of one stored session (scorecard, VIX replication,
    the static-arbitrage screen): today's row and each session's dot."""
    hkey = heston_model_key(surf["models"])
    row = next((r for r in surf.get("scorecard_rows") or [] if r["model"] == hkey), None)
    st_ = row["oos"] if row else (artifacts.residual_stats(surf["residuals"], hkey) if hkey else {})
    rep = ((surf.get("vix") or {}).get("indices") or {}).get("VIX") or {}
    gap = _num(rep.get("gap_vs_official"))
    arb = (surf.get("surface") or {}).get("arbitrage") or {}
    n_ex, n_q = _num(arb.get("n_quotes_in_executable_violation")), _num(arb.get("n_quotes"))
    return {"median_abs_hs": _num(st_.get("median_abs_hs")), "inside_pct": _num(st_.get("inside_spread_pct")),
            "n_hs": _num(st_.get("n")), "scorecard": bool(row), "rep_gap": gap,
            "vix_rep_gap": abs(gap) if gap is not None else None,
            "arb_exec_share": n_ex / n_q * 100.0 if (n_ex is not None and n_q) else None,
            "n_ex": n_ex, "n_q": n_q}


def load_model_risk() -> dict[str, Any]:
    d: dict[str, Any] = {"errors": [], "stale": [], "charts": {}, "inventory": [], "monitors": []}
    with _loader_boundary(d, "model inventory (dashboard/model_inventory.json)"):
        inv = json.loads(INVENTORY_PATH.read_text())
        # numbers in the inventory's prose are computed from the artifacts, never typed
        vx_txt = None
        for m in inv["models"]:
            if "{vx_futures_oos_rmse}" in m.get("limitations", ""):
                vx_txt = vx_txt or vx_futures_oos_text()
                m["limitations"] = m["limitations"].replace("{vx_futures_oos_rmse}", vx_txt)
        d["inventory"] = inv["models"]
        d["vendor_inputs"] = inv.get("vendor_inputs") or []
        d["inventory_about"] = inv.get("about")
    p2 = load_page2()
    for e in p2.get("errors") or []:
        d["errors"].append(e)
    mon = []

    def add(mid, check, value, text, limits_key, artifact, note, status=None, limits_text=None, key=None,
            history=None, digits=".2f"):
        lim = MONITOR_LIMITS.get(limits_key)
        st = status if status is not None else _rag(value, lim) if lim else "na"
        page, anchor = CHECK_EVIDENCE.get(key or limits_key, (None, None))
        mon.append({"model": mid, "check": check, "value": text, "status": st, "num": value,
                    "limits": limits_text or (f"OK ≤ {lim[0]:g} · watch ≤ {lim[1]:g} · breach above" if lim
                                              else "informational"),
                    "artifact": artifact, "note": note, "key": key or limits_key,
                    "limits_key": limits_key or None,
                    "id": "check-" + desk_monitor.slug(key or limits_key or check),
                    "href": f"{page}#{anchor}" if page else None, "page": page, "anchor": anchor,
                    "bullet": desk_monitor.bullet(value, lim, digits) if lim else None,
                    "history": (desk_monitor.history_view(history, lim, digits)
                                if (lim and history) else None)})
    # the gated fits: today's values and each session's, from one function of the pickle
    gated: list[tuple[str, dict]] = []
    head: dict = {}
    with _loader_boundary(d, "gate-passing fits (data_store/calibrations)"):
        gated.extend((r["session"], fit_check_values(_load_pickle(r["path"]))) for r in scan_calibrations()
                     if r["legacy"] is False and r["ok"] and r["session"])
        gated.sort(key=lambda x: x[0])
        head = next((v for s_, v in gated if s_ == p2.get("session")), {})

    def fit_hist(k):
        return [(s_, v[k]) for s_, v in gated]
    fit_art = f"data_store/calibrations/{p2.get('calib_file') or 'joint_cal_<session>.pkl'}"
    add(INV_CALIBRATOR, "Headline fit age (NYSE sessions)", _num(p2.get("calib_age")),
        _fmt(p2.get("calib_age")),
        "calib_age", fit_art, "Newest gate-passing fit against the newest session with data.", digits=".0f")
    pab = head.get("params_at_bound")
    add(INV_CALIBRATOR, "Parameters on a box bound",
        pab if pab is not None else float(len(p2.get("pinned_params") or [])),
        ", ".join(p2.get("active_constraints") or []) or "none", "params_at_bound", fit_art,
        "A parameter on its bound is set by the box, not the market." + box_refit_note(
            p2.get("vvix_tradeoff") or [], p2.get("session")), history=fit_hist("params_at_bound"),
        digits=".0f")
    add(INV_CALIBRATOR, "Out-of-sample SPX IV RMSE (vol pts)", head.get("oos_rmse"),
        _fmt(head.get("oos_rmse"), ".2f"),
        "oos_rmse", fit_art, f"{_fmt(p2.get('n_oos'), ',')} quotes not used in the fit.",
        history=fit_hist("oos_rmse"))
    add(INV_CALIBRATOR, "Worst short-dated expiry OOS RMSE (≤ 30 days)", head.get("short_oos_rmse"),
        (f"{_fmt(head['short_oos_rmse'], '.2f')} ({head['short_expiry']})"
         if head.get("short_oos_rmse") is not None else "—"), "short_oos_rmse", fit_art,
        "One-factor Heston misses the short end; see the calibration page.",
        history=fit_hist("short_oos_rmse"))
    add(INV_CALIBRATOR, "VIX index RMSE (index pts)", head.get("vix_rmse"),
        _fmt(head.get("vix_rmse"), ".2f"),
        "vix_rmse", fit_art, "^VIX9D, ^VIX, ^VIX3M, ^VIX6M against the model.", history=fit_hist("vix_rmse"))
    add(INV_CALIBRATOR, "Feller condition (diagnostic, not imposed)", None,
        f"{p2.get('feller_state') or '—'}, margin {_fmt(p2.get('feller_margin'), '+.2f')}", "", fit_art,
        "Not a no-arbitrage condition; reported, not limited.", status="info", key="feller")
    # M3: β drift — the newest stored chain against the frozen backtest β
    with _loader_boundary(d, "β estimation chains"):
        st = _load_pickle(DATA / "backtest" / "full_results.pkl")
        beta = _num(st.get("atm_to_vix_ratio", (st.get("metrics") or {}).get("atm_to_vix_ratio")))
        be = beta_estimation(None)
        later = be.get("later") or []
        newest = later[-1] if later else None
        drift = abs(newest["ratio"] - beta) if (newest and beta is not None) else None
        chains = sorted((c for c in be.get("chains") or [] if c["role"] in ("estimation", "later")
                         and c["ratio"] is not None), key=lambda c: str(c["session"]))
        add(INV_BETA, "β drift: newest chain's ATM vol / VIX against the frozen β", drift,
            (f"{_fmt(newest['ratio'], '.3f')} on {newest['session']} vs β = {_fmt(beta, '.2f')}" if newest
             else "no later chain"),
            "beta_drift", ["data_store/backtest/full_results.pkl", "data_store/vol_system.db"],
            "The backtest marks every straddle at β·VIX: β as the run recorded it (atm_to_vix_ratio) "
            "against the market DB's newest chain.", digits=".3f",
            history=([(str(c["session"]), abs(c["ratio"] - beta)) for c in chains]
                     if (newest and beta is not None and chains and chains[-1] is newest) else None))
    # M4: market DB lag behind the labels (the regime card's inputs)
    with _loader_boundary(d, "rule labels / market DB"):
        labels = pd.read_parquet(DATA / "signals" / "regime_labels.parquet", columns=["regime"])
        lab_date = str(pd.Timestamp(labels.index[-1]).date())
        with _db() as con:
            db_date = str(con.execute("SELECT MAX(date) FROM spx_ohlcv").fetchone()[0])[:10]
        lag = session_age(db_date, lab_date)
        add(INV_LABELS, "Market database lag behind the rule labels (sessions)", _num(lag), _fmt(lag),
            "label_lag",
            ["data_store/signals/regime_labels.parquet", "data_store/vol_system.db"],
            f"Labels {lab_date}; market DB {db_date}.", digits=".0f")
    # M5: the VX futures table behind the market DB (the fits' out-of-sample VX leg)
    with _loader_boundary(d, "VX futures table / market DB"):
        vx_date = vx_last_date()
        with _db() as con:
            db_date = str(con.execute("SELECT MAX(date) FROM spx_ohlcv").fetchone()[0])[:10]
        vx_lag = session_age(vx_date, db_date) if vx_date else None
        add(INV_VX, VX_LAG_CHECK, _num(vx_lag),
            _fmt(vx_lag) if vx_date else "no futures table", "label_lag", VX_TABLE,
            f"vix_futures_daily through {vx_date or '—'}; market DB {db_date}. Only the fits' out-of-sample "
            f"VX score reads them, so a fit after {vx_date or 'the table'} has none.", key="vx_lag",
            digits=".0f")
    # M2: Greeks surface built from the headline fit
    g = load_page3()
    for e in g.get("errors") or []:
        d["errors"].append(e)
    add(INV_GREEKS, "Greeks surface built from the headline fit", None,
        "yes" if g.get("source_matches_headline")
        else ("no" if g.get("source_matches_headline") is False else "unknown"),
        "", "data_store/greeks/greeks_surface.parquet", f"Surface {g.get('surface_date')}.",
        status=("ok" if g.get("source_matches_headline")
                else ("breach" if g.get("source_matches_headline") is False else "watch")),
        limits_text="OK if yes · watch if unknown · breach if no", key="greeks_source")
    # M8: classifier vs persistence (research only)
    with _loader_boundary(d, "research classifier (regime_classifier.pkl)"):
        clf = _load_pickle(DATA / "signals" / "regime_classifier.pkl")
        ev = (getattr(clf, "metadata_", None) or {}).get("evaluation") or {}
        if ev.get("oos_acc") is not None:
            gap = (float(ev["oos_acc"]) - float(ev["persistence_acc"])) * 100
            add(INV_CLASSIFIER, "Classifier accuracy minus the persistence baseline (pts)", gap,
                signed(gap, ".1f"),
                "", "data_store/signals/regime_classifier.pkl",
                "Research only since C17: below the no-skill baseline, so it never trades.",
                status="breach" if gap < 0 else "ok", limits_text="OK ≥ 0 · breach below", key="classifier")
    # surface artifacts: spread-unit fit quality and data checks on the stored chain
    try:
        surf = artifacts.load_surface(DATA)
    except artifacts.ArtifactError as e:
        _record_error(d, "surface artifacts (data_store/surface)", e)
        surf = {"state": "error"}
    if surf.get("state") == "ok":
        sart = f"data_store/surface/{surf['session']}/"
        today = surface_check_values(surf)
        shist: list[tuple[str, dict]] = []
        with _loader_boundary(d, "surface artifacts of the stored sessions (data_store/surface/<session>)"):
            for s_ in artifacts.surface_sessions(DATA):
                one = surf if s_ == surf["session"] else artifacts.load_surface(DATA, s_)
                if one.get("state") == "ok":
                    shist.append((s_, surface_check_values(one)))

        def surf_hist(k):
            return [(s_, v[k]) for s_, v in shist] if shist and shist[-1][0] == surf["session"] else None
        add(INV_CALIBRATOR, "Heston median |model − mid|, out of sample (half-spreads)",
            today["median_abs_hs"],
            f"{_fmt(today['median_abs_hs'], '.1f')} ({_fmt(today['inside_pct'], '.1f')}% inside)",
            "median_abs_hs", sart + ("scorecard.json" if today["scorecard"] else "residuals.parquet"),
            f"{_fmt(today['n_hs'], ',.0f')} quotes of {surf['session']}; Yahoo quotes carry no quote time.",
            history=surf_hist("median_abs_hs"), digits=".1f")
        vendor = chain_vendor(((surf.get("surface") or {}).get("inputs") or {}).get("source"))
        add(vendor, "VIX replicated from the stored chain − Cboe close (index pts)",
            today["vix_rep_gap"], signed(today["rep_gap"], ".2f") if today["rep_gap"] is not None else "—",
            "vix_rep_gap", sart + "vix_replication.json",
            "Cboe VIX rules on the stored two-sided chain; a gap points at a thin or stale strip.",
            history=surf_hist("vix_rep_gap"))
        share = today["arb_exec_share"]
        add(vendor, "Quotes in an executable static-arbitrage row (% of scored quotes)", share,
            f"{_fmt(share, '.1f')}% ({_fmt(today['n_ex'], ',.0f')} of {_fmt(today['n_q'], ',.0f')})"
            if share is not None else "—",
            "arb_exec_share", sart + "surface.json",
            "Bid/ask-level butterfly, vertical and calendar screen of the stored chain.",
            history=surf_hist("arb_exec_share"), digits=".1f")
    else:
        for mid_, chk, key_ in ((INV_CALIBRATOR, "Heston median |model − mid|, out of sample (half-spreads)",
                                 "median_abs_hs"),
                                ("Data", "VIX replicated from the stored chain − Cboe close (index pts)",
                                 "vix_rep_gap"),
                                ("Data", "Quotes in an executable static-arbitrage row (% of scored quotes)",
                                 "arb_exec_share")):
            add(mid_, chk, None, "artifact not yet generated", "", "data_store/surface/<session>/",
                "Shown once the surface report has run.", status="na", limits_text="—", key=key_)
    # worst first: BREACH, WATCH, OK, INFO, then N/A; within a status, in inventory
    # order (the vendor-data checks last), the order they were added within a model
    order = [m["id"] for m in d["inventory"]] + [v.get("id") for v in d.get("vendor_inputs") or []] + ["Data"]
    mon = desk_monitor.order_checks(mon, order)
    names = {m["id"]: m["name"] for m in d["inventory"]}
    names.update({v.get("id"): v.get("source") for v in d.get("vendor_inputs") or []})
    for m in mon:
        m["model_name"] = names.get(m["model"])
    d["monitors"] = mon
    d["rag_counts"] = {k: sum(1 for m in mon if m["status"] == k)
                       for k in ("ok", "watch", "breach", "info", "na")}
    d["limits"] = MONITOR_LIMITS
    return _clean_keep(d)


_BACKLINK_CACHE: dict = {}


def monitor_backlinks() -> dict[str, dict[str, list[dict]]]:
    """{page file: {element id: [check, ...]}}: the Model Risk checks whose evidence
    each panel shows, for its 'Monitored' chips (the Model Risk loader, cached on
    the inputs' modification times)."""
    key = (*_stat_key(DATA / "vol_system.db", DATA / "calibrations", DATA / "surface",
                      DATA / "signals" / "regime_labels.parquet", DATA / "greeks" / "greeks_surface.parquet",
                      DATA / "backtest" / "full_results.pkl", INVENTORY_PATH), str(DATA))
    if key in _BACKLINK_CACHE:
        return _BACKLINK_CACHE[key]
    out: dict[str, dict[str, list[dict]]] = {}
    try:
        d = load_model_risk()
    except Exception as exc:  # noqa: BLE001 (the chips are a convenience; Model Risk reports its own errors)
        _logger.info("monitor backlinks unavailable: %s", exc)
        return out
    for m in d.get("monitors") or []:
        if m.get("page") and m.get("anchor"):
            out.setdefault(m["page"], {}).setdefault(m["anchor"], []).append(
                {"model": m["model"], "status": m["status"], "id": m["id"], "check": m["check"]})
    _BACKLINK_CACHE.clear()
    _BACKLINK_CACHE[key] = out
    return out


# ── Page chrome: status bar, ticker strip, freshness table ───────────────────
_CHROME_CACHE: dict = {}


def _stat_key(*paths: Path) -> tuple:
    out: list[tuple[str, int | None, int | None]] = []
    for p in paths:
        try:
            st = Path(p).stat()
            out.append((str(p), st.st_mtime_ns, st.st_size))
        except OSError:
            out.append((str(p), None, None))
    return tuple(out)


def build_info() -> dict:
    """The build's commit: BUILD_SHA / GITHUB_SHA, else `git rev-parse` (read-only)."""
    sha = os.environ.get("BUILD_SHA") or os.environ.get("GITHUB_SHA")
    if not sha:
        try:
            sha = subprocess.run(  # noqa: S603 (a fixed argv, no external input)
                ["git", "-C", str(ROOT), "rev-parse", "HEAD"],  # noqa: S607 (git from PATH, as in CI)
                capture_output=True, text=True, timeout=5, check=False).stdout.strip() or None
        except Exception as exc:          # git missing: the status bar says "unknown"
            _logger.info("git rev-parse unavailable: %s", exc)
            sha = None
    return {"sha": sha, "short": sha[:7] if sha else None}


def load_chrome() -> dict:
    """Ticker strip, per-source as-of dates and staleness for every page.

    Failures here are shown (a red "n/a" in the status bar), not raised: each
    page's own loader owns its errors, and any page error blocks the publish.
    """
    cal_dir = DATA / "calibrations"
    key = (*_stat_key(DATA / "vol_system.db", DATA / "signals" / "regime_labels.parquet", cal_dir,
                      DATA / "greeks" / "greeks_surface.parquet", DATA / "backtest" / "full_results.parquet",
                      DATA / "surface", DATA / "backtest" / "inference"), str(DATA))
    if key in _CHROME_CACHE:
        return _CHROME_CACHE[key]
    c: dict[str, Any] = {"ticker": [], "sources": [], "worst": "ok", "build": build_info(),
                         "model_version": None}
    spx_date = vix_date = labels_date = None
    try:
        spx = _read_spx()
        last, prev = float(spx["close"].iloc[-1]), float(spx["close"].iloc[-2])
        spx_date = str(spx.index[-1].date())
        c["ticker"].append({"label": "SPX", "value": _fmt(last, ",.2f"),
                            "chg": pnl((last / prev - 1) * 100, ".2f", "%", glyph=True), "note": None})
        vd = _read_vix_daily()
        vix_date = str(vd.index[-1].date())
        c["ticker"].append({"label": "VIX", "value": _fmt(float(vd.iloc[-1]), ".2f"),
                            "chg": pnl(float(vd.iloc[-1]) - float(vd.iloc[-2]), ".2f", glyph=True),
                            "note": None if vix_date == spx_date else vix_date})
        vw = _read_vix_wide()
        if "^VVIX" in vw.columns:
            vv = vw["^VVIX"].dropna()
            c["ticker"].append({"label": "VVIX", "value": _fmt(float(vv.iloc[-1]), ".2f"),
                                "chg": pnl(float(vv.iloc[-1]) - float(vv.iloc[-2]), ".2f", glyph=True),
                                "note": (None if str(vv.index[-1].date()) == spx_date
                                         else str(vv.index[-1].date()))})
        cols = [t for t in ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M") if t in vw.columns]
        full = vw[cols].dropna() if len(cols) == 4 else pd.DataFrame()
        if len(full):
            row = full.iloc[-1]
            ts_date = str(full.index[-1].date())
            slope = float(row["^VIX6M"] - row["^VIX9D"])
            c["ticker"].append({"label": "9D→6M", "value": signed(slope, ".2f"),
                                "chg": "contango" if slope > 0 else "backwardation",
                                "note": None if ts_date == spx_date else ts_date})
            c["ticker"].append({"label": "VIX/VIX3M",
                                "value": _fmt(float(row["^VIX"] / row["^VIX3M"]), ".3f"),
                                "chg": None, "note": None if ts_date == spx_date else ts_date})
    except Exception as exc:
        _logger.info("chrome: market data unavailable (%s)", exc)
    try:
        rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet",
                             columns=["regime"])["regime"].dropna()
        labels_date = str(pd.Timestamp(rl.index[-1]).date())
        lab = int(rl.iloc[-1])
        c["ticker"].append({"label": "RULE", "value": _REGIME_LABELS[lab], "chg": _REGIME_NAMES[lab].lower(),
                            "note": labels_date})
    except Exception as exc:
        _logger.info("chrome: labels unavailable (%s)", exc)
    calib_date = greeks_date = None
    try:
        path = latest_good_calibration_path()
        if path:
            cal = _load_pickle(path)
            calib_date = _calibration_session(cal, path)
            c["model_version"] = cal.get("model_version")
    except Exception as exc:
        _logger.info("chrome: calibration unavailable (%s)", exc)
    try:
        prov = _greeks_provenance(DATA / "greeks" / "greeks_surface.parquet")
        greeks_date = _date_str(prov.get("spot_date") or prov.get("as_of_date"))
    except Exception as exc:
        _logger.info("chrome: greeks unavailable (%s)", exc)
    bt_end = None
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet", columns=["nav"])
        bt_end = str(pd.Timestamp(eq.index[-1]).date())
    except Exception as exc:
        _logger.info("chrome: backtest unavailable (%s)", exc)
    surf_sessions = artifacts.surface_sessions(DATA)
    vx_date = None
    try:
        vx_date = vx_last_date()
    except (sqlite3.Error, OSError) as exc:
        _logger.info("chrome: VX futures unavailable (%s)", exc)
    ref = max([x for x in (spx_date, labels_date) if x], default=None)
    c["latest"] = ref
    rows = [
        ("SPX", "SPX close (market DB)", spx_date, "data_store/vol_system.db", True),
        ("VIX", "VIX close (market DB)", vix_date, "data_store/vol_system.db", True),
        ("LABELS", "Rule regime labels", labels_date, "data_store/signals/regime_labels.parquet", True),
        ("FIT", "Headline Heston fit", calib_date, "data_store/calibrations/", True),
        ("GREEKS", "Greeks surface", greeks_date, "data_store/greeks/greeks_surface.parquet", True),
        ("SURFACE", "Quote-grade surface", surf_sessions[-1] if surf_sessions else None,
         "data_store/surface/<session>/", False),
        ("BACKTEST", "Backtest window end (fixed)", bt_end, "data_store/backtest/full_results.parquet",
         False),
        # aged and styled like the others, but no headline number reads it (the fits
        # score VX futures out of sample only), so it is not in the status bar and
        # does not set the page state; Model Risk carries its lag as a check
        ("VX", "VX futures settlements", vx_date, VX_TABLE, True),
    ]
    reference_only = {"VX"}
    for short, label, date, art, aged in rows:
        age = session_age(date, ref) if (aged and date) else None
        level = (stale_level(age) if aged else "ok") if date else ("missing" if aged else "unknown")
        c["sources"].append({"short": short, "label": label, "date": date, "age": age, "level": level,
                             "artifact": art, "aged": aged, "bar": short not in reference_only})
        if short in reference_only:
            continue
        if level in ("error", "missing"):
            c["worst"] = "error"
        elif level == "warn" and c["worst"] == "ok":
            c["worst"] = "warn"
    c["holidays"] = [str(h.date()) for h in nyse_holidays(
        f"{(ref or '2026')[:4]}-01-01", f"{int((ref or '2026')[:4]) + 2}-12-31")]
    _CHROME_CACHE.clear()
    _CHROME_CACHE[key] = c
    return c


# ── Rendering ──────────────────────────────────────────────────────────────────
# Per-route data date for the sitemap (freeze_site.py reads the header): the
# date of the data each page shows.
_DATA_DATE_KEYS = {"/": "latest_session", "/calibration": "session", "/risk": "surface_date",
                   "/research": "end_date", "/surface": "session", "/corrections": "latest_round_date",
                   "/model-risk": None}

# Third-party code served from dashboard/static/vendor (copied into the site by
# freeze_site.py): no script comes from another origin. `sri` is the sha384 of
# the vendored file; freeze_site.py refuses to publish if a file no longer
# matches, and the pages carry it as the integrity attribute.
#   echarts: a tree-shaken build of Apache ECharts 6.1.0 (Apache-2.0) made by
#            dashboard/echarts_build (pinned lockfile; `npm ci && node build.mjs`
#            reproduces the file byte for byte).
#   echarts-gl: the npm dist file of echarts-gl 2.1.0 (BSD-3-Clause) with one
#            patch (dashboard/echarts_build/patch_gl.mjs): its compositor's
#            `new Function` is replaced by a small arithmetic parser, so the
#            pages need no 'unsafe-eval'. Loaded only when a reader asks for the
#            3D surface view.
VENDORED: dict[str, dict] = {
    "echarts": {
        "version": "6.1.0", "license": "Apache-2.0",
        "src": "static/vendor/echarts-6.1.0-desk.min.js",
        "sri": "sha384-91oWllvVjknURH3iVY4aXknr/M7IadTMvoXGVuobX0GYrzD8gMLUwsVnC+tRn1k4",
        "upstream": "npm echarts@6.1.0, tree-shaken by dashboard/echarts_build",
        "license_file": "echarts-LICENSE.txt",
    },
    "echarts-gl": {
        "version": "2.1.0", "license": "BSD-3-Clause",
        "src": "static/vendor/echarts-gl-2.1.0-csp.min.js",
        "sri": "sha384-YXckRCK1pIX+JWGuGeaHlMZOpZViZwkgsac4HKofKED09rnmDjqdcStCMfmOOp0T",
        "upstream": "npm echarts-gl@2.1.0 dist, patched by dashboard/echarts_build/patch_gl.mjs",
        "license_file": "echarts-gl-LICENSE.txt",
    },
}

# The site's own stylesheet and script (same origin; versioned by content hash
# so a cached old copy is never paired with a new page).
OWN_ASSETS = ("static/desk.css", "static/desk.js")
FONT_PRELOAD = "static/fonts/InterDesk-4.1-wght400-700.woff2"
# Both faces the first screen's text and chart axes use: ECharts draws canvas
# text once, so a chart initialised before Desk Mono arrived kept the fallback
# font (desk.js also redraws such charts when document.fonts is ready).
FONT_PRELOADS = (FONT_PRELOAD, "static/fonts/DeskMono-400.woff2")

# Every page is static; all JavaScript is in static files, chart data is in
# <script type="application/json"> blocks (never executed), so scripts are
# limited to this origin. style-src allows inline style attributes (heatmap
# cell colours). frame-ancestors / report-uri are not honoured in a <meta> CSP.
CSP = ("default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; "
       "img-src 'self' data:; font-src 'self'; connect-src 'self'; object-src 'none'; "
       "base-uri 'self'; form-action 'none'")


def own_asset(rel: str) -> dict:
    """src with a content-hash query (cache busting) for one of OWN_ASSETS."""
    data = (DASH / rel).read_bytes()
    return {"src": f"{rel}?v={hashlib.sha256(data).hexdigest()[:12]}", "bytes": len(data)}


@app.context_processor
def _page_context() -> dict:
    route = request.path if request.path in ROUTES else "/"
    return {"canonical": canonical_url(route), "page_file": ROUTES[route], "route": route,
            "nav": NAV, "routes": ROUTES, "vendored": VENDORED, "csp": CSP,
            "cividis": CIVIDIS_11, "berlin": BERLIN_11, "s3_full": S3_FULL, "spx_multiplier": SPX_MULTIPLIER,
            "css": own_asset("static/desk.css"), "js": own_asset("static/desk.js"),
            "font_preloads": FONT_PRELOADS, "chrome": load_chrome(), "page_title": {
                r: label for r, label, *_ in NAV}.get(route, "Overview")}


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
    # the Model Risk checks whose evidence this page shows (its panels' 'Monitored' chips)
    request_globals.monitored = (monitor_backlinks().get(ROUTES.get(request.path, ""), {})
                                 if request.path != "/model-risk" else {})
    resp = make_response(render_template(template, d=d))
    resp.headers["X-Dashboard-Loader-Errors"] = str(len(errors))
    if errors:
        resp.headers["X-Dashboard-Error-Components"] = "; ".join(e["component"] for e in errors)
    stale = [f"{s['component']}={s['age']}" for s in d.get("stale") or []
             if s.get("level") in ("warn", "error")]
    if stale:
        resp.headers["X-Dashboard-Stale"] = ", ".join(stale)
    key = _DATA_DATE_KEYS.get(request.path)
    data_date = d.get(key) if key else None
    if data_date:
        resp.headers["X-Dashboard-Data-Date"] = str(data_date)
    return resp


@app.route("/")
def page_overview():
    return _render("index.html", load_overview())


@app.route("/surface")
def page_surface():
    return _render("surface.html", load_surface())


@app.route("/calibration")
def page_calibration():
    return _render("calibration.html", load_page2())


@app.route("/risk")
def page_risk():
    return _render("risk.html", load_page3())


@app.route("/research")
def page_research():
    return _render("research.html", load_page4())


@app.route("/corrections")
def page_corrections():
    return _render("corrections.html", load_corrections())


@app.route("/model-risk")
def page_model_risk():
    return _render("model-risk.html", load_model_risk())


@app.route("/greeks")
@app.route("/backtest")
def page_legacy():
    """Old routes of the local preview (the frozen site writes redirect pages)."""
    from flask import redirect
    return redirect({"/greeks": "/risk", "/backtest": "/research"}[request.path], code=301)


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
