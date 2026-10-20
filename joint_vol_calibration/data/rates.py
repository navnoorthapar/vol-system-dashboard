"""
rates.py — a daily US Treasury zero curve with look-ahead gating (C23).

Replaces "one 13-week bill rate for every maturity" (^IRX via Yahoo, see
database.get_tbill_rate) with a term structure built from free public data:

  US Treasury, home.treasury.gov (US Government work, one CSV per year):
    daily_treasury_bill_rates   4/6/8/13/17/26/52-week bills, bank-discount
                                AND coupon-equivalent; FRBNY quotations at
                                approximately 3:30 PM ET
    daily_treasury_yield_curve  the par yield curve 1M..30Y (CMT): a monotone
                                convex spline through indicative bid-side
                                prices at or near 3:30 PM, floored at zero
  NY Fed, markets.newyorkfed.org/api (no key; the Terms of Use permit
  automated access and redistribution provided the notice and disclaimer in
  SOFR_NOTICE accompany the data):
    SOFR            overnight, for the box-spread comparison only
  FRED, fred.stlouisfed.org/graph/fredgraph.csv (keyless) — implemented
  (download_fred) but NOT used by refresh_rates by default: FRED's Terms of
  Use say "Don't do any data mining, scraping or extraction of FRED data"
  and bar using FRED content to develop or train machine learning
  (DATA_SOURCES.md). Its DGS* series equal the Treasury par yields exactly
  (46,200 of 46,200 values 2010-2026, checked 2026-10-20).

Quotes are stored as published in database.rates_daily (value = quote/100)
and persisted as yearly partitions (data_store/market/rates_daily/); this
module converts them to zero-coupon prices.

Conventions (match the pricers: exp(-r T), T in ACT/365 years):
  bill, bank discount d, n days:      P = 1 - d n / 360
  bill / CMT coupon-equivalent y:     n <= 182:  P = 1 / (1 + y n / 365)
                                      n  > 182:  P = 1 / ((1 + y/2)(1 + y (n/365 - 1/2)))
                                      (Treasury's coupon-equivalent formula)
  par yield y_T beyond 1 year:        semiannual par bond priced at 1,
                                      bootstrapped on a 6-month grid with
                                      par yields interpolated linearly in T
                                      (an approximation: CMT par yields are
                                      a fitted curve, not traded bonds)
  zero rate                           r(T) = -ln P(T) / T, continuously compounded
  interpolation                       linear in ln P(T) between nodes, i.e.
                                      piecewise-constant forward rates; flat
                                      zero rate before the first node (T <
                                      28 days uses the 4-week rate) and flat
                                      forward after the last
  bill maturities                     the nominal 7 x weeks days (the quote is
                                      for the most recently auctioned bill,
                                      whose actual days to maturity differ by
                                      a few days; not modelled)

Curve choice for one date: the bills when that date has at least two bill
quotes (every Treasury business day of the 2010-2026 record), else the par
curve's 1M-1Y points; then the par curve beyond 1 year. The two Treasury products are not
the same numbers: on 2026-10-19 the 13-week bill's coupon equivalent was
4.20% against a 3-month par yield of 4.27% (4-week 4.01% vs 1-month 4.11%).
The bills are actual zero-coupon prices, so they set the short end.

Measured on the committed record (2026-10-20):
  r(91 d) against Yahoo's ^IRX (a different source and quote time; the rate
  used before C23), 2,210 sessions 2018-01-02..2026-10-16: median +1.2 bp,
  95th pct 5.8 bp, worst 26 bp (2022-09-19, a bad ^IRX print);
  r(91 d) with the 13-week node removed (interpolated from its neighbours,
  every 5th date 2018-2026): median 1.65 bp, 95th pct 13.9 bp;
  a bill's bank-discount and coupon-equivalent quotes imply the same rate to
  under 2 bp (rounding to 0.01%).

Look-ahead: a curve as of session D uses the newest Treasury date <= D (the
same-day ~3:30 PM ET quotes, treated as known at D's close like the 16:00
index closes; same_day=False requires a date < D). SOFR and H.15 values are
published the NEXT business day (SOFR at ~8:00 ET), so a value dated d is
used only for as-of dates after d. A curve older than ``max_age_days``
calendar days raises database.StaleDataError (the bond market is closed on
Columbus and Veterans Day, when the NYSE trades: the previous day's curve
is used then).

API: discount_factor(as_of, T), zero_rate(as_of, T), get_zero_curve(as_of)
-> ZeroCurve, sofr(as_of), parity_rate_comparison(...), refresh_rates(...).
CLI: python -m joint_vol_calibration.data.rates refresh|parity ...
"""

from __future__ import annotations

import argparse
import io
import json
import logging
import math
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd
import requests

from joint_vol_calibration.config import HIST_START
from joint_vol_calibration.data import database as db

logger = logging.getLogger(__name__)

# ── Series registry ───────────────────────────────────────────────────────────

BILL_WEEKS = (4, 6, 8, 13, 17, 26, 52)

# Treasury par-curve column -> (series, tenor in months)
_PAR_COLUMNS = {
    "1 Mo": ("UST_PAR_1M", 1.0), "1.5 Month": ("UST_PAR_1.5M", 1.5), "2 Mo": ("UST_PAR_2M", 2.0),
    "3 Mo": ("UST_PAR_3M", 3.0), "4 Mo": ("UST_PAR_4M", 4.0), "6 Mo": ("UST_PAR_6M", 6.0),
    "1 Yr": ("UST_PAR_1Y", 12.0), "2 Yr": ("UST_PAR_2Y", 24.0), "3 Yr": ("UST_PAR_3Y", 36.0),
    "5 Yr": ("UST_PAR_5Y", 60.0), "7 Yr": ("UST_PAR_7Y", 84.0), "10 Yr": ("UST_PAR_10Y", 120.0),
    "20 Yr": ("UST_PAR_20Y", 240.0), "30 Yr": ("UST_PAR_30Y", 360.0),
}
PAR_MONTHS = {s: m for s, m in _PAR_COLUMNS.values()}

# FRED id -> series. DGS* are the same constant-maturity par yields as the
# Treasury file (identical on every overlapping date checked, see tests).
FRED_SERIES = {
    "DGS1MO": "UST_PAR_1M", "DGS3MO": "UST_PAR_3M", "DGS6MO": "UST_PAR_6M", "DGS1": "UST_PAR_1Y",
    "DGS2": "UST_PAR_2Y", "DGS3": "UST_PAR_3Y", "DGS5": "UST_PAR_5Y", "DGS7": "UST_PAR_7Y",
    "DGS10": "UST_PAR_10Y", "DGS20": "UST_PAR_20Y", "DGS30": "UST_PAR_30Y",
    "SOFR": "SOFR", "DTB3": "H15_BILL_3M_BD",
}
# Series dated d that are published on the next business day.
NEXT_DAY_SERIES = ("SOFR", "H15_BILL_3M_BD")
# The NY Fed Terms of Use ("Use Restrictions", "Reference Rates", last updated
# 6/9/2023) require this notice AND disclaimer with any use or distribution of
# reference-rate data; the bracketed fields of their template are filled in.
SOFR_NOTICE = ("The SOFR Data is subject to the Terms of Use posted at newyorkfed.org. "
               "The New York Fed is not responsible for publication of the SOFR by "
               "joint-vol-calibration, does not sanction or endorse any particular "
               "republication, and has no liability for your use.")


def bill_series(weeks: int, basis: str = "BD") -> str:
    """'UST_BILL_13W_BD' (bank discount) or 'UST_BILL_13W_CE' (coupon equivalent)."""
    return f"UST_BILL_{int(weeks)}W_{basis.upper()}"


# ── Downloads (no key needed) ─────────────────────────────────────────────────

_TREASURY_CSV = ("https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
                 "daily-treasury-rates.csv/{year}/all?type={kind}&field_tdr_date_value={year}"
                 "&page&_format=csv")
TREASURY_KINDS = ("daily_treasury_bill_rates", "daily_treasury_yield_curve")
_FRED_CSV = "https://fred.stlouisfed.org/graph/fredgraph.csv?id={ids}"
_NYFED_SOFR = ("https://markets.newyorkfed.org/api/rates/secured/sofr/search.json"
               "?startDate={start}&endDate={end}")


# An honest User-Agent: FRED drops requests that send the browser string the
# Cboe session uses (connection closed without a response, every attempt
# on 2026-10-20) and answers this one; Treasury accepts both.
_USER_AGENT = "joint-vol-calibration/0.22 (+https://github.com/navnoorthapar/vol-system-dashboard)"
_SESSION = requests.Session()
_SESSION.headers.update({"User-Agent": _USER_AGENT, "Accept": "text/csv,text/plain,*/*;q=0.8"})


def _get_text(url: str, timeout: float = 60.0, attempts: int = 3, pause: float = 2.0) -> str:
    """GET ``url`` as text, retrying a connection error or a 5xx up to ``attempts`` times."""
    import time
    for k in range(attempts):
        try:
            resp = _SESSION.get(url, timeout=timeout)
            if resp.status_code < 500 or k == attempts - 1:
                resp.raise_for_status()
                return resp.text
        except (requests.ConnectionError, requests.Timeout):
            if k == attempts - 1:
                raise
        time.sleep(pause * (k + 1))
    raise RuntimeError("unreachable")


def parse_treasury_csv(text: str, kind: str) -> pd.DataFrame:
    """One yearly Treasury CSV -> long rows [date, series, value, source='treasury'].

    kind 'daily_treasury_bill_rates': '<N> WEEKS BANK DISCOUNT' / '<N> WEEKS
    COUPON EQUIVALENT' columns -> UST_BILL_<N>W_BD / _CE. kind
    'daily_treasury_yield_curve': '1 Mo' .. '30 Yr' -> UST_PAR_<tenor>.
    Columns a year does not carry are simply absent (first dates seen
    2026-10-20: 8-week bill and 2-month par 2018-10-16, 17-week bill and
    4-month par 2022-10-19, 6-week bill and 1.5-month par 2025-02-18);
    unknown columns are ignored with a warning. Values are percent in the
    file, decimal here.
    """
    df = pd.read_csv(io.StringIO(text))
    if "Date" not in df.columns:
        raise ValueError(f"Treasury {kind} CSV has no Date column: {df.columns.tolist()}")
    dates = pd.to_datetime(df["Date"], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    out = []
    for col in df.columns:
        if col == "Date":
            continue
        name = col.strip()
        series = None
        if kind == "daily_treasury_bill_rates":
            parts = name.upper().split()
            if len(parts) >= 3 and parts[1] == "WEEKS" and parts[0].isdigit():
                basis = {"BANK": "BD", "COUPON": "CE"}.get(parts[2])
                series = bill_series(int(parts[0]), basis) if basis else None
        elif kind == "daily_treasury_yield_curve":
            series = _PAR_COLUMNS.get(name, (None,))[0]
        if series is None:
            logger.warning("Treasury %s: ignoring unknown column %r", kind, col)
            continue
        vals = pd.to_numeric(df[col], errors="coerce") / 100.0
        out.append(pd.DataFrame({"date": dates, "series": series, "value": vals}))
    if not out:
        return pd.DataFrame(columns=["date", "series", "value", "source"])
    res = pd.concat(out, ignore_index=True).dropna(subset=["date", "value"])
    res["source"] = "treasury"
    return res.sort_values(["date", "series"]).reset_index(drop=True)


def download_treasury_year(year: int, kind: str) -> pd.DataFrame:
    """One year of one Treasury file (parse_treasury_csv)."""
    if kind not in TREASURY_KINDS:
        raise ValueError(f"kind must be one of {TREASURY_KINDS}")
    return parse_treasury_csv(_get_text(_TREASURY_CSV.format(year=int(year), kind=kind)), kind)


def parse_fred_csv(text: str) -> pd.DataFrame:
    """fredgraph.csv (observation_date, <id>...) -> long rows [date, series, value, source='fred'].

    Ids are mapped with FRED_SERIES (unknown ids keep their FRED name);
    FRED's '.' / empty for no observation is dropped; percent -> decimal.
    """
    df = pd.read_csv(io.StringIO(text))
    dcol = "observation_date" if "observation_date" in df.columns else df.columns[0]
    dates = pd.to_datetime(df[dcol], errors="coerce").dt.strftime("%Y-%m-%d")
    out = []
    for col in df.columns:
        if col == dcol:
            continue
        vals = pd.to_numeric(df[col], errors="coerce") / 100.0
        out.append(pd.DataFrame({"date": dates, "series": FRED_SERIES.get(col, col), "value": vals}))
    res = pd.concat(out, ignore_index=True).dropna(subset=["date", "value"])
    res["source"] = "fred"
    return res.sort_values(["date", "series"]).reset_index(drop=True)


# The graph CSV answers at most this many ids per request and silently drops
# the rest (2026-10-20: 13 ids requested, the 13th, DTB3, absent; 12 all served).
FRED_MAX_IDS_PER_REQUEST = 12


def download_fred(ids: Iterable[str] = tuple(FRED_SERIES), start: Optional[str] = None) -> pd.DataFrame:
    """Keyless FRED download of ``ids`` (parse_fred_csv), rows from ``start`` on.

    One request per FRED_MAX_IDS_PER_REQUEST ids. The multi-series graph CSV
    returns each series' whole history whatever the start (checked
    2026-10-20), so the window is applied after parsing. An id the response
    does not carry raises ValueError (FRED drops extras without an error).
    """
    ids = list(dict.fromkeys(str(i) for i in ids))
    frames = []
    for k in range(0, len(ids), FRED_MAX_IDS_PER_REQUEST):
        chunk = ids[k:k + FRED_MAX_IDS_PER_REQUEST]
        text = _get_text(_FRED_CSV.format(ids=",".join(chunk)))
        header = [c.strip() for c in text.splitlines()[0].split(",")] if text else []
        missing = [i for i in chunk if i not in header]
        if missing:
            raise ValueError(f"FRED response lacks requested series {missing} (columns {header})")
        frames.append(parse_fred_csv(text))
    if not frames:
        return pd.DataFrame(columns=["date", "series", "value", "source"])
    df = pd.concat(frames, ignore_index=True).sort_values(["date", "series"]).reset_index(drop=True)
    return df[df["date"] >= start].reset_index(drop=True) if start else df


def parse_nyfed_sofr(payload: dict) -> pd.DataFrame:
    """NY Fed reference-rate JSON ({'refRates': [{effectiveDate, percentRate, ...}]})
    -> rows [date, series='SOFR', value, source='nyfed'] (percent -> decimal)."""
    rows = [(r.get("effectiveDate"), r.get("percentRate")) for r in payload.get("refRates") or []
            if str(r.get("type", "SOFR")).upper() == "SOFR"]
    df = pd.DataFrame(rows, columns=["date", "value"])
    df["value"] = pd.to_numeric(df["value"], errors="coerce") / 100.0
    df = df.dropna().assign(series="SOFR", source="nyfed")
    return df[["date", "series", "value", "source"]].sort_values("date").reset_index(drop=True)


def download_nyfed_sofr(start: str = "2018-04-02", end: Optional[str] = None) -> pd.DataFrame:
    """SOFR history from the NY Fed's markets API (one request; parse_nyfed_sofr)."""
    end = end or date.today().isoformat()
    return parse_nyfed_sofr(json.loads(_get_text(_NYFED_SOFR.format(start=start, end=end))))


def refresh_rates(start_year: Optional[int] = None, end_year: Optional[int] = None, *,
                  sofr: bool = True, fred_ids: Sequence[str] = ()) -> dict:
    """Download the Treasury bill and par files for [start_year, end_year] (2 requests a
    year) and, with ``sofr``, the NY Fed SOFR history into database.rates_daily.
    ``fred_ids`` (default none; see the FRED terms in the module docstring) adds FRED
    series, e.g. the DGS* par yields during a Treasury outage. Returns rows per source
    and errors."""
    start_year = start_year or int(HIST_START[:4])
    end_year = end_year or date.fromisoformat(db.last_closed_session()).year
    out = {"treasury": 0, "nyfed": 0, "fred": 0, "errors": []}
    for year in range(int(start_year), int(end_year) + 1):
        for kind in TREASURY_KINDS:
            try:
                df = download_treasury_year(year, kind)
                out["treasury"] += db.insert_rates_daily(df) if not df.empty else 0
            except Exception as e:
                out["errors"].append(f"treasury {kind} {year}: {e}")
                logger.warning("Treasury %s %d failed: %s", kind, year, e)
    if sofr:
        try:
            df = download_nyfed_sofr(start=max(f"{start_year}-01-01", "2018-04-02"),
                                     end=f"{end_year}-12-31")
            out["nyfed"] = db.insert_rates_daily(df) if not df.empty else 0
        except Exception as e:
            out["errors"].append(f"nyfed sofr: {e}")
            logger.warning("NY Fed SOFR download failed: %s", e)
    if fred_ids:
        try:
            df = download_fred(fred_ids, start=f"{start_year}-01-01")
            out["fred"] = db.insert_rates_daily(df) if not df.empty else 0
        except Exception as e:
            out["errors"].append(f"fred: {e}")
            logger.warning("FRED download failed: %s", e)
    return out


# ── Quote store: database rows plus the committed partitions ──────────────────

_QUOTES_CACHE: dict = {}


def _store_key(root: Path) -> tuple:
    files = sorted((root / "rates_daily").glob("rates_daily_*.parquet"))
    dbp = Path(db.DB_PATH)
    return (str(dbp), dbp.stat().st_mtime_ns if dbp.exists() else None, str(root),
            tuple((f.name, f.stat().st_mtime_ns) for f in files))


def load_rate_quotes(root=None) -> pd.DataFrame:
    """Every stored rate quote: database.rates_daily over the persisted partitions
    (the database wins on a (date, series) clash). Cached until either changes.
    Columns [date (str), series, value, source]."""
    root = Path(root) if root is not None else db.MARKET_DIR
    key = _store_key(root)
    hit = _QUOTES_CACHE.get(key)
    if hit is not None:
        return hit
    part = db.read_market_partitions("rates_daily", root)
    try:
        rows = db.get_rates_daily("9999-12-31")
    except Exception as e:          # a read-only or absent database: the partitions alone
        logger.info("rates: database not readable (%s); using the partitions only", e)
        rows = pd.DataFrame(columns=["date", "series", "value", "source"])
    df = pd.concat([part, rows], ignore_index=True).drop_duplicates(["date", "series"], keep="last")
    df = df.sort_values(["date", "series"]).reset_index(drop=True)
    _QUOTES_CACHE.clear()
    _QUOTES_CACHE[key] = df
    _curve_cache.clear()
    _index_cache.clear()
    return df


# ── Instrument conversions ────────────────────────────────────────────────────

def price_from_discount(d, days):
    """Bill price per 1 of face from a bank-discount yield: 1 - d * days / 360."""
    return 1.0 - np.asarray(d, float) * np.asarray(days, float) / 360.0


def price_from_coupon_equivalent(y, days):
    """Zero price from a coupon-equivalent (bond-equivalent) yield, Treasury's formula:
    days <= 182: 1 / (1 + y days/365); else 1 / ((1 + y/2)(1 + y (days/365 - 1/2)))."""
    y, n = np.asarray(y, float), np.asarray(days, float)
    short = 1.0 / (1.0 + y * n / 365.0)
    long_ = 1.0 / ((1.0 + y / 2.0) * (1.0 + y * (n / 365.0 - 0.5)))
    return np.where(n <= 182.0, short, long_)


def cc_rate(price, t):
    """Continuously compounded rate of a zero price over t years: -ln(P)/t."""
    return -np.log(np.asarray(price, float)) / np.asarray(t, float)


def sofr_to_cc(s: float) -> float:
    """SOFR (simple, ACT/360, one day) as a continuously compounded ACT/365 rate."""
    return math.log1p(s / 360.0) * 365.0


# ── The curve ─────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class ZeroCurve:
    """Zero-coupon curve of one quote date.

    t, log_df : node times (ACT/365 years, strictly increasing, > 0) and ln P(t)
    curve_date: the quote date used; as_of: the date it was requested for
    short_end : 'bills' | 'par' (which instruments set t <= 1 year)
    nodes     : DataFrame [instrument, t, quote, basis, zero_cc], one row per input
    """
    as_of: str
    curve_date: str
    t: np.ndarray
    log_df: np.ndarray
    short_end: str
    nodes: pd.DataFrame = field(repr=False)

    def _log_df(self, T) -> np.ndarray:
        T = np.asarray(T, float)
        tt = np.concatenate([[0.0], self.t])
        ll = np.concatenate([[0.0], self.log_df])
        out = np.interp(T, tt, ll)
        f_last = (ll[-2] - ll[-1]) / (tt[-1] - tt[-2])           # flat forward beyond the last node
        out = np.where(T > tt[-1], ll[-1] - f_last * (T - tt[-1]), out)
        return np.where(T <= 0.0, 0.0, out)

    def discount_factor(self, T):
        """P(0, T) for T in years (scalar or array); 1 for T <= 0."""
        out = np.exp(self._log_df(T))
        return float(out) if np.ndim(out) == 0 else out

    def zero_rate(self, T):
        """Continuously compounded zero rate r(T) = -ln P(T) / T; the first segment's rate as T -> 0."""
        T = np.asarray(T, float)
        r0 = -self.log_df[0] / self.t[0]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = np.where(T > 0.0, -self._log_df(T) / np.where(T > 0.0, T, 1.0), r0)
        return float(out) if np.ndim(out) == 0 else out

    def forward_rate(self, t1, t2):
        """Continuously compounded forward rate between t1 < t2."""
        t1, t2 = np.asarray(t1, float), np.asarray(t2, float)
        out = (self._log_df(t1) - self._log_df(t2)) / (t2 - t1)
        return float(out) if np.ndim(out) == 0 else out


def _bill_nodes(q: pd.Series) -> list:
    nodes = []
    for w in BILL_WEEKS:
        n = 7 * w
        bd, ce = q.get(bill_series(w, "BD")), q.get(bill_series(w, "CE"))
        if bd is not None and np.isfinite(bd):
            p, quote, basis = float(price_from_discount(bd, n)), bd, "bank_discount"
        elif ce is not None and np.isfinite(ce):
            p, quote, basis = float(price_from_coupon_equivalent(ce, n)), ce, "coupon_equivalent"
        else:
            continue
        nodes.append((bill_series(w, "BD")[:-3], n / 365.0, quote, basis, p))
    return nodes


def _par_short_nodes(q: pd.Series) -> list:
    nodes = []
    for s, m in PAR_MONTHS.items():
        if m > 12 or s not in q or not np.isfinite(q[s]):
            continue
        t = m / 12.0
        nodes.append((s, t, q[s], "coupon_equivalent", float(price_from_coupon_equivalent(q[s], t * 365.0))))
    return nodes


def _bootstrap_par(q: pd.Series, short_t: np.ndarray, short_ldf: np.ndarray) -> list:
    """Semiannual par bonds beyond 1 year -> zero nodes on the 6-month grid."""
    long_pts = sorted((PAR_MONTHS[s] / 12.0, q[s]) for s in PAR_MONTHS
                      if PAR_MONTHS[s] >= 12 and s in q and np.isfinite(q[s]))
    if not any(t > 1.0 for t, _ in long_pts):
        return []
    tp, yp = np.array([p[0] for p in long_pts]), np.array([p[1] for p in long_pts])
    short = ZeroCurve("", "", short_t, short_ldf, "", pd.DataFrame())
    grid = np.arange(0.5, tp.max() + 1e-9, 0.5)
    dfs: list = []
    nodes = []
    for t in grid:
        if t <= 1.0 + 1e-12:
            dfs.append(short.discount_factor(t))
            continue
        y = float(np.interp(t, tp, yp))
        p = (1.0 - y / 2.0 * sum(dfs)) / (1.0 + y / 2.0)
        if not (0.0 < p < 1.5):
            raise ValueError(f"par bootstrap failed at {t:.1f}y (P={p})")
        dfs.append(p)
        tag = next((s for s, m in PAR_MONTHS.items() if abs(m / 12.0 - t) < 1e-9), f"UST_PAR_interp_{t:g}Y")
        nodes.append((tag, t, y, "par_semiannual", p))
    return nodes


@dataclass(frozen=True)
class QuoteIndex:
    """Treasury quotes grouped by date, for building many curves fast."""
    dates: np.ndarray                  # sorted unique quote dates (YYYY-MM-DD strings)
    by_date: dict                      # date -> Series(series -> value)


def index_quotes(quotes: pd.DataFrame) -> QuoteIndex:
    """QuoteIndex of the UST_* rows of long rate quotes [date, series, value, ...]."""
    q = quotes[quotes["series"].astype(str).str.startswith("UST_")]
    by_date = {str(d): g.drop_duplicates("series", keep="last").set_index("series")["value"]
               for d, g in q.groupby("date", sort=True)}
    return QuoteIndex(np.array(sorted(by_date), dtype=object), by_date)


def build_zero_curve(quotes, as_of: str, *, max_age_days: int = 7, same_day: bool = True) -> ZeroCurve:
    """The zero curve as of ``as_of`` from long rate quotes [date, series, value, ...]
    (or a QuoteIndex of them).

    Uses ONE quote date: the newest Treasury date <= as_of (< as_of with
    same_day=False) that has a bill or par quote. Raises StaleDataError when
    that date is more than ``max_age_days`` calendar days before as_of, and
    LookupError when there is none.
    """
    db._validate_date(as_of)
    idx = quotes if isinstance(quotes, QuoteIndex) else index_quotes(quotes)
    k = int(np.searchsorted(idx.dates, as_of, side="right" if same_day else "left"))
    if k == 0:
        raise LookupError(f"no Treasury quote {'on or ' if same_day else ''}before {as_of}")
    curve_date = str(idx.dates[k - 1])
    age = (pd.Timestamp(as_of) - pd.Timestamp(curve_date)).days
    if age > max_age_days:
        raise db.StaleDataError(f"newest Treasury curve before {as_of} is {curve_date} ({age} days old; "
                                f"allowed {max_age_days})")
    day = idx.by_date[curve_date]
    short = _bill_nodes(day)
    short_end = "bills"
    if len(short) < 2:
        short, short_end = _par_short_nodes(day), "par"
    if not short:
        raise LookupError(f"no usable short-end quote on {curve_date}")
    short.sort(key=lambda n: n[1])
    st = np.array([n[1] for n in short])
    sl = np.log(np.array([n[4] for n in short]))
    long_nodes = _bootstrap_par(day, st, sl)
    nodes = short + [n for n in long_nodes if n[1] > st[-1] + 1e-9]
    t = np.array([n[1] for n in nodes])
    ldf = np.log(np.array([n[4] for n in nodes]))
    if np.any(np.diff(t) <= 0):
        raise ValueError(f"non-increasing curve nodes on {curve_date}")
    frame = pd.DataFrame({"instrument": [n[0] for n in nodes], "t": t, "quote": [n[2] for n in nodes],
                          "basis": [n[3] for n in nodes], "zero_cc": -ldf / t})
    return ZeroCurve(as_of=as_of, curve_date=curve_date, t=t, log_df=ldf, short_end=short_end, nodes=frame)


_curve_cache: dict = {}


_index_cache: dict = {}


def get_zero_curve(as_of: str, *, max_age_days: int = 7, same_day: bool = True, root=None) -> ZeroCurve:
    """build_zero_curve over the stored quotes (load_rate_quotes), cached per argument set."""
    quotes = load_rate_quotes(root)
    idx = _index_cache.get(id(quotes))
    if idx is None:
        _index_cache.clear()
        idx = _index_cache[id(quotes)] = index_quotes(quotes)
    key = (as_of, max_age_days, same_day, id(quotes))
    if key not in _curve_cache:
        _curve_cache[key] = build_zero_curve(idx, as_of, max_age_days=max_age_days, same_day=same_day)
    return _curve_cache[key]


def discount_factor(as_of: str, T, **kw):
    """P(0, T) on the Treasury zero curve as of ``as_of`` (T in ACT/365 years)."""
    return get_zero_curve(as_of, **kw).discount_factor(T)


def zero_rate(as_of: str, T, **kw):
    """Continuously compounded Treasury zero rate r(T) as of ``as_of`` (T in ACT/365 years)."""
    return get_zero_curve(as_of, **kw).zero_rate(T)


def sofr(as_of: str, *, max_age_days: int = 7, root=None) -> dict:
    """The newest SOFR published by ``as_of``: dated strictly before it (SOFR for
    business day d is published at ~8:00 ET on the next business day).
    Returns {date, sofr (ACT/360 simple), cc (ACT/365 continuous), notice}."""
    q = load_rate_quotes(root)
    q = q[(q["series"] == "SOFR") & (q["date"] < as_of)]
    if q.empty:
        raise LookupError(f"no SOFR published before {as_of}")
    row = q.iloc[-1]
    age = (pd.Timestamp(as_of) - pd.Timestamp(row["date"])).days
    if age > max_age_days:
        raise db.StaleDataError(f"newest SOFR before {as_of} is {row['date']} ({age} days old)")
    return {"date": str(row["date"]), "sofr": float(row["value"]), "cc": sofr_to_cc(float(row["value"])),
            "notice": SOFR_NOTICE}


# ── Parity-implied (box) rates against the curve ──────────────────────────────

def parity_rate_comparison(carry: pd.DataFrame, as_of: str, *, curve: Optional[ZeroCurve] = None,
                           par_curve: Optional[ZeroCurve] = None,
                           sofr_cc: Optional[float] = None) -> pd.DataFrame:
    """Per expiry: the SPX put-call-parity rate r_T against the Treasury curve.

    ``carry`` is option_chain.CleanChain.carry (expiry, T, r, method, ...);
    only rows fitted by regression (method 'parity_regression', i.e. not the
    fixed-discount fallback, whose r is the reference rate) are compared.
    Columns: expiry, days, T (ACT/365 years from the chain's T origin),
    r_parity (continuously compounded, -ln D_T / T), r_bills (the curve's
    zero rate at the same T), r_par (a curve built from the par yields only,
    if given), r_sofr (overnight SOFR, flat, if given), the spreads in basis
    points, and the regression's n_pairs / resid_rmse when ``carry`` has
    them. A positive spread is the extra yield the options' implied box
    earns over Treasuries.
    """
    curve = curve or get_zero_curve(as_of)
    c = carry[carry["method"] == "parity_regression"].copy()
    out = pd.DataFrame({"expiry": c["expiry"].astype(str).str[:10].to_numpy(), "T": c["T"].to_numpy(float),
                        "r_parity": c["r"].to_numpy(float)})
    out["days"] = (out["T"] * 365.0).round(2)
    out["r_bills"] = curve.zero_rate(out["T"].to_numpy())
    out["spread_bills_bp"] = (out["r_parity"] - out["r_bills"]) * 1e4
    if par_curve is not None:
        out["r_par"] = par_curve.zero_rate(out["T"].to_numpy())
        out["spread_par_bp"] = (out["r_parity"] - out["r_par"]) * 1e4
    if sofr_cc is not None:
        out["r_sofr"] = sofr_cc
        out["spread_sofr_bp"] = (out["r_parity"] - sofr_cc) * 1e4
    for col in ("n_pairs", "resid_rmse"):
        if col in c.columns:
            out[col] = c[col].to_numpy()
    out.insert(0, "as_of", as_of)
    out.insert(1, "curve_date", curve.curve_date)
    return out[["as_of", "curve_date", "expiry", "days", "T", "r_parity", "r_bills", "spread_bills_bp"]
               + [c for c in ("r_par", "spread_par_bp", "r_sofr", "spread_sofr_bp", "n_pairs", "resid_rmse")
                  if c in out.columns]]


def par_only_curve(as_of: str, root=None, **kw) -> ZeroCurve:
    """The curve with the par yields setting the short end too (for comparison)."""
    q = load_rate_quotes(root)
    q = q[~q["series"].astype(str).str.startswith("UST_BILL_")]
    return build_zero_curve(q, as_of, **kw)


def committed_chain_comparison(snapshot_dates: Optional[Sequence[str]] = None, *, root=None) -> pd.DataFrame:
    """parity_rate_comparison for every committed SPX chain (data_store/options/spx_*.parquet).

    Each chain is cleaned with the calibrator's cleaner
    (option_chain.clean_option_chain, the session's SPX close as spot, T from
    the chain's own T origin) but WITHOUT a reference rate (r_ref=None), so
    every expiry keeps its regression r_T as long as it lies in [-2%, 12%].
    The calibrator itself passes r_ref = ^IRX and replaces an r_T more than
    option_chain.PARITY_RATE_TOL from it by ^IRX; the added columns r_irx
    (that reference, database.get_tbill_rate) and calibrator_uses_parity
    (|r_parity - r_irx| <= PARITY_RATE_TOL) say which expiries the
    calibrator discounts with the parity D_T. Read-only.
    """
    from joint_vol_calibration.data import option_chain as oc
    opt_dir = Path(db.OPTIONS_DIR)
    paths = sorted(opt_dir.glob("spx_*.parquet"))
    if snapshot_dates:
        paths = [p for p in paths if p.stem[4:] in set(snapshot_dates)]
    frames = []
    for p in paths:
        session = p.stem[4:]
        raw = pd.read_parquet(p)
        spx = db.get_spx_ohlcv(session, start_date=session)
        if spx.empty:
            logger.warning("no SPX close for %s; skipped", session)
            continue
        clean = oc.clean_option_chain(raw, session, float(spx["close"].iloc[-1]), r_ref=None)
        if not len(clean.carry):
            logger.warning("chain %s: no carry rows (%s)", session, clean.reason)
            continue
        try:
            s = sofr(session, root=root)["cc"]
        except LookupError:
            s = None
        cmp_ = parity_rate_comparison(clean.carry, session, curve=get_zero_curve(session, root=root),
                                      par_curve=par_only_curve(session, root=root), sofr_cc=s)
        cmp_.insert(2, "chain_source", ",".join(sorted(set(db.canonical_source(raw["source"]).dropna())))
                    if "source" in raw.columns else "")
        cmp_["r_irx"] = db.get_tbill_rate(session)
        cmp_["calibrator_uses_parity"] = (cmp_["r_parity"] - cmp_["r_irx"]).abs() <= oc.PARITY_RATE_TOL
        frames.append(cmp_)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def summarize_parity_comparison(cmp_: pd.DataFrame,
                                buckets: Sequence[float] = (0, 30, 91, 365, 10_000)) -> pd.DataFrame:
    """Median and interquartile range of the parity-minus-curve spreads (bp) by
    days-to-expiry bucket [lo, hi), over every row of committed_chain_comparison."""
    df = cmp_.copy()
    labels = [f"{int(lo)}-{int(hi)}d" if hi < 10_000 else f">={int(lo)}d" for lo, hi in zip(buckets[:-1], buckets[1:])]
    df["bucket"] = pd.cut(df["days"], list(buckets), right=False, labels=labels)
    rows = []
    for b, g in df.groupby("bucket", observed=True):
        row = {"bucket": str(b), "n": int(len(g)), "sessions": int(g["as_of"].nunique())}
        for col in ("spread_bills_bp", "spread_par_bp", "spread_sofr_bp"):
            if col in g.columns:
                q1, med, q3 = np.nanpercentile(g[col].to_numpy(float), [25, 50, 75])
                row[f"{col}_median"], row[f"{col}_iqr"] = float(med), float(q3 - q1)
        rows.append(row)
    return pd.DataFrame(rows)


# ── Command line ──────────────────────────────────────────────────────────────

def main(argv=None) -> int:
    """python -m joint_vol_calibration.data.rates refresh [--start-year Y] [--persist]
    python -m joint_vol_calibration.data.rates parity [--out CSV] [--summary CSV]"""
    ap = argparse.ArgumentParser(prog="python -m joint_vol_calibration.data.rates")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("refresh")
    r.add_argument("--start-year", type=int, default=None)
    r.add_argument("--end-year", type=int, default=None)
    r.add_argument("--persist", action="store_true", help="also write data_store/market/rates_daily/")
    p = sub.add_parser("parity")
    p.add_argument("--out", default=None)
    p.add_argument("--summary", default=None, help="also write summarize_parity_comparison() as CSV")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    if args.cmd == "refresh":
        res = refresh_rates(args.start_year, args.end_year)
        if args.persist:
            res["persisted"] = [str(x) for x in db.export_market_partitions(tables=["rates_daily"])]
        print(json.dumps(res, indent=2))
        return 1 if res["errors"] else 0
    df = committed_chain_comparison()
    if args.out:
        df.to_csv(args.out, index=False, float_format="%.6f")
    summ = summarize_parity_comparison(df)
    if args.summary:
        summ.to_csv(args.summary, index=False, float_format="%.2f")
    print(df.to_string(index=False))
    print(summ.to_string(index=False))
    return 0


if __name__ == "__main__":     # pragma: no cover
    raise SystemExit(main())
