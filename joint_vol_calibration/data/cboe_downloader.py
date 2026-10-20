"""
cboe_downloader.py — Downloads free data directly from Cboe's public endpoints.

Endpoints (verified live 2026-10-12; offline captures under tests/fixtures/):
  1. Index history CSVs     cdn.cboe.com/api/global/us_indices/daily_prices/<INDEX>_History.csv
                            VIX, VIX9D, VIX3M, VIX6M (DATE,OPEN,HIGH,LOW,CLOSE),
                            VVIX and VRO (DATE,<INDEX>). The official series.
  2. VX final settlements   VRO_History.csv — the special opening quotation of
                            VIX on VX/VIX-options expiry mornings. Used on the
                            standard monthly expiry dates only: in 2015-2020
                            the file also holds forward-filled rows for other
                            weekdays (see download_vix_futures_settlements).
  3. VX per-contract CSV    cdn.cboe.com/data/us/futures/market_statistics/
                            historical_data/VX/VX_<YYYY-MM-DD>.csv — daily OHLC,
                            Settle, volume, OI for one contract (file named by
                            its expiry date).
  4. VX daily settlements   www.cboe.com/us/futures/market_statistics/settlement/
                            csv?dt=<YYYY-MM-DD> — every CFE contract's settlement
                            for one trade date (header only on closed days).
  5. Delayed option quotes  cdn.cboe.com/api/global/delayed_quotes/options/_SPX.json
                            and _VIX.json (index symbols take a leading
                            underscore; 'SPX.json' is a 403). Full chain, all
                            expiries, bid/ask/IV/greeks/OI, in one request.
                            (cdn.cboe.com 307-redirects these to cdn-api.cboe.com;
                            requests follows it.)

The old VX_History.csv and resources/futures/vx-historical-settlement-data.csv
URLs return 403 AccessDenied: they are S3 keys that no longer exist.

Financial context:
  VIX = the 30-day risk-neutral expected volatility of SPX (a spot-starting
  variance-swap rate). A VX future expiring at T settles on VIX AT T — the 30-day
  volatility over [T, T+30d] — so the futures curve is a FORWARD volatility
  curve; it is not the same object as the spot-starting VIX9D/VIX3M/VIX6M
  indices. These are the prices a VIX-futures model leg should be fitted to.
"""

import hashlib
import io
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Optional

import numpy as np
import pandas as pd
import requests

from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS, VVIX_TENOR_DAYS
from joint_vol_calibration.data import database as db

logger = logging.getLogger(__name__)

# ── Cboe public data URLs ─────────────────────────────────────────────────────

_CDN = "https://cdn.cboe.com"
_INDEX_HISTORY_URL = _CDN + "/api/global/us_indices/daily_prices/{index}_History.csv"
_VX_CONTRACT_URL = _CDN + "/data/us/futures/market_statistics/historical_data/VX/VX_{expiry}.csv"
_VX_SETTLEMENT_URL = "https://www.cboe.com/us/futures/market_statistics/settlement/csv?dt={date}"
_CBOE_QUOTE_API = _CDN + "/api/global/delayed_quotes/options/{symbol}.json"

# Kept for callers that imported it: the VIX index history file.
_VIX_HISTORY_URL = _INDEX_HISTORY_URL.format(index="VIX")

# Cash-index underlyings whose quote symbol carries a leading underscore.
_INDEX_UNDERLYINGS = {"SPX", "VIX", "XSP", "NDX", "RUT", "DJX"}

# Yahoo-style ticker -> Cboe index file name.
_TERM_INDEX_FILES = {"^VIX9D": "VIX9D", "^VIX": "VIX", "^VIX3M": "VIX3M", "^VIX6M": "VIX6M",
                     "^VVIX": "VVIX"}
_TERM_TENORS = {**VIX_INDEX_TENOR_DAYS, "^VVIX": VVIX_TENOR_DAYS}

# Futures month codes, as in Cboe symbols ('VX/V6') and contract names ('V (Oct 2026)').
_MONTH_CODES = {"F": 1, "G": 2, "H": 3, "J": 4, "K": 5, "M": 6,
                "N": 7, "Q": 8, "U": 9, "V": 10, "X": 11, "Z": 12}

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json,text/csv,text/plain,*/*;q=0.8",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


def _get(url: str, timeout: float = 30.0) -> requests.Response:
    """GET ``url`` (following redirects) and raise on an HTTP error status."""
    resp = _SESSION.get(url, timeout=timeout)
    resp.raise_for_status()
    return resp


# ── Index history (VIX family, VVIX, VRO) ─────────────────────────────────────

def download_index_history(index: str) -> pd.DataFrame:
    """
    Download one Cboe index's full daily history.

    Files with OHLC columns (VIX, VIX9D, VIX3M, VIX6M) keep them; single-value
    files (VVIX, VRO) put the value in ``close`` and leave open/high/low NaN.
    Dates are MM/DD/YYYY in the file and YYYY-MM-DD here. Cboe also prints
    values on some days the NYSE is closed (e.g. 2026-01-19, MLK day); those
    rows are returned as published — database.validate_sessions flags them.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, source='cboe'].
    """
    url = _INDEX_HISTORY_URL.format(index=index.upper())
    logger.info("Downloading %s history from Cboe ...", index.upper())
    df = pd.read_csv(io.StringIO(_get(url).text))
    df.columns = [c.strip().lower() for c in df.columns]
    if "date" not in df.columns:
        raise ValueError(f"No DATE column in {url}: {df.columns.tolist()}")
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce").dt.strftime("%Y-%m-%d")
    value_cols = [c for c in df.columns if c != "date"]
    if "close" not in df.columns:
        if len(value_cols) != 1:
            raise ValueError(f"Cannot identify the value column in {url}: {df.columns.tolist()}")
        df = df.rename(columns={value_cols[0]: "close"})
    for col in ("open", "high", "low", "close"):
        df[col] = pd.to_numeric(df[col], errors="coerce") if col in df.columns else np.nan
    df = df.dropna(subset=["date", "close"])
    df = df.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    df["source"] = "cboe"
    logger.info("Downloaded %d %s rows (%s to %s)", len(df), index.upper(),
                df["date"].iloc[0] if len(df) else None, df["date"].iloc[-1] if len(df) else None)
    return df[["date", "open", "high", "low", "close", "source"]]


def download_vix_history() -> pd.DataFrame:
    """
    Download VIX index daily OHLC from Cboe's public CDN (the official series).

    Financial interpretation:
      VIX close = market's 30-day forward-looking annualised vol expectation.
      High VIX = fear / uncertainty. Low VIX = complacency.
      We use this as input to regime classification and to compare against
      realised vol (20-day rolling SPX realised vol).

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, source='cboe'].
    """
    return download_index_history("VIX")


def download_vix_term_structure_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    tickers: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    The VIX term-structure indices (and VVIX) from Cboe's official files.

    Same long format as yf_downloader.download_vix_term_structure, so the two
    can be cross-checked (database.cross_check_vix) or stored interchangeably.

    Returns
    -------
    DataFrame with columns [date, ticker, tenor_days, close, is_vvix, source='cboe'].
    """
    frames = []
    for ticker in (tickers or _TERM_INDEX_FILES):
        df = download_index_history(_TERM_INDEX_FILES[ticker])
        df["ticker"] = ticker
        frames.append(df)
    out = pd.concat(frames, ignore_index=True)
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]
    out["tenor_days"] = out["ticker"].map(_TERM_TENORS).astype(int)
    out["is_vvix"] = out["ticker"].eq("^VVIX")
    out = out.sort_values(["date", "tenor_days"]).reset_index(drop=True)
    return out[["date", "ticker", "tenor_days", "close", "is_vvix", "source"]]


# ── Other Cboe indices (C23) ──────────────────────────────────────────────────

# Same URL template as the VIX family. Name -> (what it is, first date in the
# file as fetched 2026-10-20). Files with OHLC columns keep them; the others
# (SKEW, DSPX, VIXEQ and the strategy benchmarks) carry one value per day.
CBOE_INDEX_CATALOG = {
    "VIX1D": ("1-day VIX: expected 1-day SPX volatility (variance-swap method on 0-1 DTE options)",
              "2022-05-13"),
    "VIX1Y": ("1-year VIX: the VIX method at a 1-year horizon", "2007-01-03"),
    "SKEW": ("Cboe SKEW Index: 100 - 10 x (S&P 500 30-day risk-neutral skewness)", "1990-01-02"),
    "COR1M": ("Cboe 1-month implied correlation index (S&P 500, 50 largest names)", "2006-01-03"),
    "COR3M": ("Cboe 3-month implied correlation index", "2006-01-03"),
    "DSPX": ("Cboe S&P 500 Dispersion Index: 30-day expected dispersion of constituent returns",
             "2014-06-19"),
    "VIXEQ": ("Cboe S&P 500 constituent volatility index (the single-name leg of DSPX)", "2014-06-19"),
    # Strategy benchmarks (total-return indices of systematic SPX option strategies)
    "PUT": ("Cboe S&P 500 PutWrite Index (monthly cash-secured ATM put write)", "1991-03-04"),
    "WPUT": ("Cboe S&P 500 One-Week PutWrite Index", "2006-01-31"),
    "BXM": ("Cboe S&P 500 BuyWrite Index (monthly ATM covered call)", "2002-03-22"),
    "BFLY": ("Cboe S&P 500 Iron Butterfly Index", "1986-06-20"),
    "CNDR": ("Cboe S&P 500 Iron Condor Index", "1986-06-20"),
}
# What pipeline.refresh_cboe_indices downloads unless told otherwise.
DEFAULT_EXTRA_INDICES = ("VIX1D", "VIX1Y", "SKEW", "COR1M", "COR3M", "DSPX")


def download_cboe_indices(symbols: Iterable[str] = DEFAULT_EXTRA_INDICES,
                          start: Optional[str] = None,
                          end: Optional[str] = None) -> pd.DataFrame:
    """
    Daily history of Cboe indices outside the VIX family, one file each.

    ``symbols`` are Cboe file names (CBOE_INDEX_CATALOG lists the known
    ones; any other name is tried with the same URL template). Rows are
    returned as published — including the days the NYSE was closed;
    database.insert_cboe_index drops those. [start, end] trims inclusively.

    Returns
    -------
    DataFrame with columns [date, symbol, open, high, low, close, source='cboe'],
    sorted by (date, symbol). A symbol whose download fails raises.
    """
    frames = []
    for name in symbols:
        sym = str(name).upper().lstrip("^_")
        df = download_index_history(sym)
        df["symbol"] = sym
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["date", "symbol", "open", "high", "low", "close", "source"])
    out = pd.concat(frames, ignore_index=True)
    if start:
        out = out[out["date"] >= start]
    if end:
        out = out[out["date"] <= end]
    out = out.sort_values(["date", "symbol"]).reset_index(drop=True)
    return out[["date", "symbol", "open", "high", "low", "close", "source"]]


# ── VX Futures Settlements ────────────────────────────────────────────────────

def download_vix_futures_settlements(
    start: Optional[str] = None,
    end: Optional[str] = None,
    daily_history: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Final settlement values of the standard monthly VX contracts expiring in [start, end].

    Source: Cboe's VRO index (the special opening quotation of VIX on the
    expiry morning), taken ONLY on the standard monthly expiry dates
    (database.vx_monthly_expiry). VRO_History.csv is not a clean list of
    settlements: through 2015-2020 it also carries a row for most other
    weekdays, forward-filled from an earlier value (e.g. 20.09 on every day
    from 2019-01-11 to 01-22, although the January 2019 contract settled at
    18.87 on 01-16), so its other rows are not used and weekly expiries are
    not stored here.

    If ``daily_history`` (download_vix_futures_daily_history output) is
    given, each contract's own expiry-day Settle — Cboe's final settlement in
    the per-contract file — takes precedence; a disagreement with VRO above
    half a cent is logged. (Checked 2026-10-12 over the 160 monthly
    contracts 2013-06 .. 2026-09 whose file publishes an expiry-day settle:
    159 agree exactly; the exception is the 2019-01-16 row above.)

    These are the ground-truth points for model validation: if a model
    implies settlement S' and the market settled at S, |S' - S| is direct
    evidence of model error.

    Returns
    -------
    DataFrame with columns [settlement_date, expiry_month, settlement_price,
    is_weekly (always 0), source ('cboe_vro' | 'cboe_contract')].
    """
    vro = download_index_history("VRO").set_index("date")["close"]
    lo = start or vro.index.min()
    hi = end or db.last_closed_session()
    months = pd.period_range(pd.Timestamp(lo).to_period("M"), pd.Timestamp(hi).to_period("M"), freq="M")
    expiries = [e for e in (db.vx_monthly_expiry(p.year, p.month) for p in months) if lo <= e <= hi]
    rows = {e: (float(vro[e]), "cboe_vro") for e in expiries if e in vro.index}

    if daily_history is not None and not daily_history.empty:
        final = daily_history[(daily_history["date"] == daily_history["expiry_date"])
                              & daily_history["settle"].notna()
                              & daily_history["expiry_date"].isin(expiries)]
        for e, price in zip(final["expiry_date"], final["settle"]):
            if e in rows and abs(rows[e][0] - price) > 0.005:
                logger.warning("VX %s final settlement: contract file %.4f vs VRO %.4f — using the "
                               "contract file", e, price, rows[e][0])
            rows[e] = (float(price), "cboe_contract")

    out = pd.DataFrame(
        [(e, e[:7], p, 0, src) for e, (p, src) in sorted(rows.items())],
        columns=["settlement_date", "expiry_month", "settlement_price", "is_weekly", "source"],
    )
    logger.info("VX final settlements: %d monthly contracts (%s .. %s)", len(out), lo, hi)
    return out


# ── VX Futures Daily ──────────────────────────────────────────────────────────

def _finish_vx_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Shared tail of the VX parsers: expiry month, weekly flag, days to expiry."""
    df["expiry_month"] = df["expiry_date"].str[:7]
    monthly = {m: db.vx_expiry_for_month(m) for m in df["expiry_month"].unique()}
    df["is_weekly"] = (df["expiry_date"] != df["expiry_month"].map(monthly)).astype(int)
    df["days_to_expiry"] = (pd.to_datetime(df["expiry_date"]) - pd.to_datetime(df["date"])).dt.days
    return df


def parse_vx_contract_csv(text: str, expiry_date: str) -> pd.DataFrame:
    """
    Parse one per-contract VX file (Trade Date, Futures, Open, High, Low, Close,
    Settle, Change, Total Volume, EFP, Open Interest).

    Settle is the official daily mark (on the expiry row it is the final
    settlement value); Cboe writes 0 for "not published" (files before mid-2013)
    and for "no trade" in the price columns — those become NaN. The file's
    name, not its 'Futures' text ('V (Oct 2026)' for weekly and monthly alike),
    identifies the contract.

    Returns
    -------
    DataFrame with columns [date, expiry_date, expiry_month, symbol, is_weekly,
    open, high, low, close, settle, volume, open_interest, days_to_expiry, source].
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if "trade_date" not in df.columns or "settle" not in df.columns:
        raise ValueError(f"Unexpected VX contract file columns: {df.columns.tolist()}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "expiry_date": expiry_date,
        "symbol": None,
        "open": pd.to_numeric(df.get("open"), errors="coerce"),
        "high": pd.to_numeric(df.get("high"), errors="coerce"),
        "low": pd.to_numeric(df.get("low"), errors="coerce"),
        "close": pd.to_numeric(df.get("close"), errors="coerce"),
        "settle": pd.to_numeric(df["settle"], errors="coerce"),
        "volume": pd.to_numeric(df.get("total_volume"), errors="coerce").astype("Int64"),
        "open_interest": pd.to_numeric(df.get("open_interest"), errors="coerce").astype("Int64"),
        "source": "cboe",
    })
    for col in ("open", "high", "low", "close", "settle"):
        out[col] = out[col].where(out[col] > 0)
    out = out.dropna(subset=["date"])
    out = out[out["settle"].notna() | out["close"].notna()]
    out = out[out["date"] <= expiry_date]
    return _finish_vx_frame(out.reset_index(drop=True))


def download_vx_contract_history(expiry_date: str) -> pd.DataFrame:
    """Daily history of the VX contract expiring on ``expiry_date`` (see parse_vx_contract_csv)."""
    db._validate_date(expiry_date)
    return parse_vx_contract_csv(_get(_VX_CONTRACT_URL.format(expiry=expiry_date)).text, expiry_date)


def parse_vx_settlement_csv(text: str, trade_date: str) -> pd.DataFrame:
    """
    Parse the daily CFE settlement file (Product, Symbol, Expiration Date, Price)
    for ``trade_date``, keeping VX (not the VXM mini) rows.

    Symbols: 'VX/V6' is the standard monthly, 'VX41/V6' a weekly (week 41).
    Note: on days a weekly does not trade, Cboe prints a MONTHLY contract's
    settlement for it, not always the one of its own contract month: on
    2026-10-16 the weeklies expiring 10-28 to 11-11 (VX43/V6 .. VX45/X6) all
    showed the October monthly's 19.9075 and those expiring 11-24 and 12-01 the
    November monthly's 19.2382. See database.get_vix_futures_curve, which
    excludes weeklies by default.

    Returns
    -------
    DataFrame with columns [date, expiry_date, expiry_month, symbol, is_weekly,
    settle, days_to_expiry, source]; empty when Cboe published no settlements
    (weekend, holiday, or not yet posted).
    """
    df = pd.read_csv(io.StringIO(text))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]
    if df.empty:
        return pd.DataFrame(columns=["date", "expiry_date", "expiry_month", "symbol", "is_weekly",
                                     "settle", "days_to_expiry", "source"])
    df = df[df["product"].astype(str).str.strip() == "VX"]
    out = pd.DataFrame({
        "date": trade_date,
        "expiry_date": pd.to_datetime(df["expiration_date"], errors="coerce").dt.strftime("%Y-%m-%d"),
        "symbol": df["symbol"].astype(str).str.strip(),
        "settle": pd.to_numeric(df["price"], errors="coerce"),
        "source": "cboe_settlement",
    }).dropna(subset=["expiry_date", "settle"])
    out = out[out["expiry_date"] >= trade_date]
    out = _finish_vx_frame(out.reset_index(drop=True))
    # The symbol is authoritative for weekly vs monthly ('VX/V6' vs 'VX41/V6').
    out["is_weekly"] = (~out["symbol"].str.match(r"^VX/[A-Z]\d+$")).astype(int)
    return out


def download_vx_settlements(trade_date: str) -> pd.DataFrame:
    """Every VX contract's daily settlement on ``trade_date`` (see parse_vx_settlement_csv)."""
    db._validate_date(trade_date)
    return parse_vx_settlement_csv(_get(_VX_SETTLEMENT_URL.format(date=trade_date)).text, trade_date)


def download_vix_futures_daily_history(
    start: Optional[str] = None,
    end: Optional[str] = None,
    expiries: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """
    Download daily history for the standard monthly VX contracts.

    By default: every monthly contract expiring from ``start`` (default one
    year before ``end``) to nine months after ``end`` (default the last closed
    session), one per-contract file each; or exactly the contracts in
    ``expiries``. Rows are trimmed to trade dates in [start, end]. A contract
    whose file is missing is logged and skipped.

    Financial interpretation:
      The futures curve (settle by expiry) is the market's forward-vol curve:
        - Contango (far > near): the usual calm-market shape
        - Backwardation (near > far): elevated spot VIX, fear premium in front month

    Returns
    -------
    DataFrame as parse_vx_contract_csv, all contracts concatenated, sorted by
    (date, expiry_date).
    """
    end = end or db.last_closed_session()
    start = start or (pd.Timestamp(end) - pd.DateOffset(years=1)).strftime("%Y-%m-%d")
    if expiries is None:
        months = pd.period_range(pd.Timestamp(start).to_period("M"),
                                 (pd.Timestamp(end) + pd.DateOffset(months=9)).to_period("M"), freq="M")
        expiries = [db.vx_monthly_expiry(p.year, p.month) for p in months]
        expiries = [e for e in expiries if e >= start]
    frames = []
    for exp in expiries:
        try:
            frames.append(download_vx_contract_history(exp))
        except Exception as e:
            logger.warning("VX contract %s unavailable: %s", exp, e)
    if not frames:
        return pd.DataFrame()
    df = pd.concat(frames, ignore_index=True)
    df = df[(df["date"] >= start) & (df["date"] <= end)]
    df = df.sort_values(["date", "expiry_date"]).reset_index(drop=True)
    logger.info("Downloaded %d VX daily rows over %d contract(s)", len(df), df["expiry_date"].nunique())
    return df


# ── Cboe delayed option quotes ────────────────────────────────────────────────

def _quote_symbol(underlying: str) -> str:
    und = underlying.upper().lstrip("_^")
    return f"_{und}" if und in _INDEX_UNDERLYINGS else und


# The delayed-quotes feed lags the market by this much (Cboe states 15
# minutes; measured 15m01s-15m02s on 2026-10-20, see parse_cboe_options_payload).
STATED_DELAY_MINUTES = 15
QUOTE_DELAY = timedelta(minutes=STATED_DELAY_MINUTES)


def delayed_quote_time(payload_timestamp, now: Optional[datetime] = None) -> str:
    """ISO UTC instant a delayed-quotes payload shows: its ``timestamp`` (UTC,
    'YYYY-MM-DD HH:MM:SS') minus QUOTE_DELAY, else the fetch time minus it."""
    stamp = pd.to_datetime(payload_timestamp, errors="coerce", utc=True)
    if pd.isna(stamp):
        return db.utc_iso((now or db._utcnow()) - QUOTE_DELAY)
    return (stamp - QUOTE_DELAY).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_cboe_options_payload(payload: dict, underlying: str,
                               now: Optional[datetime] = None) -> pd.DataFrame:
    """
    Turn a delayed-quotes JSON payload into options_snapshots rows.

    Each option carries only its OCC symbol ('option', e.g. SPXW261016C07700000)
    plus quote fields; expiry, right, strike and root are parsed from it. SPX
    and SPXW share expiry dates and strikes but settle differently (AM vs PM),
    so the root is kept.

    Dating: the snapshot is the NY session of the UNDERLYING's last trade
    (data.last_trade_time, New York local time). It is cross-checked against
    database.quote_session(now); a disagreement is logged and the payload wins.
    time_to_expiry runs from that session's 16:00 close to each contract's
    settlement (database.time_to_settlement). underlying_price is data.close,
    falling back to data.current_price (during the session both hold the
    delayed last index value; measured 2026-10-20 12:04 ET).

    quote_time is the instant the quotes show: the payload ``timestamp``
    (when Cboe generated the file) minus QUOTE_DELAY. The feed is delayed:
    on 2026-10-20 a payload stamped 16:04:29Z carried a latest SPX option
    trade and an underlying last trade at 11:49:27 ET (15:49:27Z), 15m02s
    earlier; the VIX payload likewise (15m01s). Without a timestamp the
    fetch time minus QUOTE_DELAY is used.

    mid_price is set only for two-sided markets (bid > 0 and ask >= bid);
    Cboe's iv of 0 (not computed) becomes NaN. Greeks are stored as Cboe
    delivers them.
    """
    data = payload.get("data") or {}
    options = data.get("options") or []
    if not options:
        logger.warning("No options in Cboe payload for %s", underlying)
        return pd.DataFrame()

    ltt = pd.to_datetime(data.get("last_trade_time"), errors="coerce")
    clock_session = db.quote_session(now)
    if pd.isna(ltt):
        session = clock_session
        logger.warning("Cboe %s payload has no last_trade_time; dating by the clock (%s)",
                       underlying, session)
    else:
        session = db.session_on_or_before(ltt.date())
        if now is not None and session != clock_session:
            logger.warning("Cboe %s quotes are from session %s but the clock says %s; using %s",
                           underlying, session, clock_session, session)

    quote_time = delayed_quote_time(payload.get("timestamp"), now)
    spot = data.get("close") or data.get("current_price")
    spot = float(spot) if spot else np.nan

    raw = pd.DataFrame(options)
    parsed = [db.parse_occ_symbol(s) for s in raw.get("option", pd.Series(dtype=object))]
    ok = np.array([p is not None for p in parsed], dtype=bool)
    if not ok.all():
        logger.warning("Cboe %s: skipped %d row(s) without an OCC symbol", underlying, int((~ok).sum()))
    raw = raw[ok].reset_index(drop=True)
    parsed = [p for p in parsed if p is not None]
    if raw.empty:
        logger.warning("Parsed 0 rows from Cboe %s options payload", underlying)
        return pd.DataFrame()

    def num(col):
        return pd.to_numeric(raw[col], errors="coerce") if col in raw.columns else pd.Series(np.nan, index=raw.index)

    bid, ask, iv = num("bid"), num("ask"), num("iv")
    root = [p["root"] for p in parsed]
    settlement = [db.settlement_for_root(r) for r in root]
    expiry = [p["expiry"] for p in parsed]
    df = pd.DataFrame({
        "snapshot_date": session,
        "expiry": expiry,
        "strike": [p["strike"] for p in parsed],
        "right": [p["right"] for p in parsed],
        "bid": bid,
        "ask": ask,
        "mid_price": ((bid + ask) / 2.0).where((bid > 0) & (ask >= bid)),
        "implied_vol": iv.where(iv > 0),
        "open_interest": num("open_interest").round().astype("Int64"),
        "volume": num("volume").round().astype("Int64"),
        "delta": num("delta"),
        "gamma": num("gamma"),
        "vega": num("vega"),
        "theta": num("theta"),
        "time_to_expiry": [db.time_to_settlement(session, e, s) for e, s in zip(expiry, settlement)],
        "root": root,
        "contract_symbol": raw["option"].astype(str).to_numpy(),
        "settlement": settlement,
        "underlying_price": spot,
        "quote_time": quote_time,
        "source": "cboe",
    })
    df = df[(df["strike"] > 0) & (df["time_to_expiry"] > 0)]
    df = df.sort_values(["expiry", "root", "right", "strike"]).reset_index(drop=True)
    return df


def download_cboe_options_snapshot(underlying: str, now: Optional[datetime] = None) -> pd.DataFrame:
    """
    Download the current (15-min delayed) options chain from Cboe's public API.

    One request returns every listed expiry with bid/ask, IV, greeks and open
    interest — the preferred source for the daily snapshot (Yahoo needs one
    request per expiry and has no greeks).

    Parameters
    ----------
    underlying : str — 'SPX' or 'VIX'
    now        : fetch time for the dating cross-check (default: the clock)

    Returns
    -------
    DataFrame with the options_snapshots columns (see parse_cboe_options_payload):
      [snapshot_date, expiry, strike, right, bid, ask, mid_price, implied_vol,
       open_interest, volume, delta, gamma, vega, theta, time_to_expiry, root,
       contract_symbol, settlement, underlying_price, quote_time, source]
    """
    symbol = _quote_symbol(underlying)
    url = _CBOE_QUOTE_API.format(symbol=symbol)
    logger.info("Fetching %s options snapshot from Cboe (%s) ...", underlying.upper(), url)
    now = now or db._utcnow()
    df = parse_cboe_options_payload(_get(url, timeout=60).json(), underlying.upper(), now=now)
    if not df.empty:
        if db.is_intraday(now):
            logger.warning("Cboe %s chain fetched during the %s session: intraday quotes",
                           underlying.upper(), df["snapshot_date"].iloc[0])
        logger.info("Fetched %d %s option rows (%d expiries) for session %s", len(df),
                    underlying.upper(), df["expiry"].nunique(), df["snapshot_date"].iloc[0])
    return df


# ── Timestamped chain capture (C23; not wired into the daily bot) ─────────────

# Per-option fields the delayed-quotes payload carries beyond the
# options_snapshots schema; kept only with include_extras=True.
_CAPTURE_EXTRAS = {"bid_size": "bid_size", "ask_size": "ask_size", "theo": "theo", "rho": "rho",
                   "last_trade_price": "last_trade_price", "last_trade_time": "option_last_trade_time"}

# The payload's `timestamp` is when Cboe generated the file (on 2026-10-20 a
# fetch at 12:29:03Z returned timestamp 12:28:16Z, Last-Modified 12:28:19Z);
# the quotes in it are QUOTE_DELAY older (delayed_quote_time).


@dataclass
class ChainCapture:
    """One delayed-quotes download of a full option chain, with its timestamps.

    frame       options_snapshots rows (parse_cboe_options_payload): snapshot_date
                is the NY session of the underlying's last trade, quote_time the
                instant the delayed quotes show (payload timestamp minus
                QUOTE_DELAY, ISO UTC), source 'cboe'; plus the _CAPTURE_EXTRAS
                columns when include_extras=True
    session     that NY session
    quote_time  the frame's quote_time
    book_state  where quote_time falls relative to that session (book_state_at):
                'intraday' | 'closing' | 'gth' | 'pre_open'. Classified at the
                quote instant, not the fetch time, so it agrees with
                option_chain.t_origin(quote_time, session)
    fetched_at  client clock at the request (ISO UTC)
    payload_timestamp (file generation, ISO UTC), underlying_last_trade_time
    (NY local, as published), underlying_price, url, sha256 (of the raw
    response bytes), n_bytes
    """
    underlying: str
    frame: pd.DataFrame
    session: Optional[str]
    quote_time: str
    book_state: str
    fetched_at: str
    payload_timestamp: Optional[str]
    underlying_last_trade_time: Optional[str]
    underlying_price: float
    url: str
    sha256: str
    n_bytes: int

    def summary(self) -> dict:
        """JSON-ready metadata (no quotes): counts, timestamps, hash."""
        f = self.frame
        two = int(((f["bid"] > 0) & (f["ask"] >= f["bid"])).sum()) if len(f) else 0
        return {
            "underlying": self.underlying, "session": self.session, "quote_time": self.quote_time,
            "book_state": self.book_state, "fetched_at": self.fetched_at,
            "payload_timestamp": self.payload_timestamp, "stated_delay_minutes": STATED_DELAY_MINUTES,
            "underlying_last_trade_time": self.underlying_last_trade_time,
            "underlying_price": self.underlying_price, "url": self.url,
            "sha256": self.sha256, "n_bytes": self.n_bytes, "rows": int(len(f)),
            "two_sided_rows": two, "expiries": int(f["expiry"].nunique()) if len(f) else 0,
            "roots": sorted(f["root"].dropna().unique().tolist()) if len(f) else [],
        }


def book_state_at(at: datetime, session: Optional[str]) -> str:
    """Which book a quote feed shows at instant ``at`` for quotes of ``session``.

    Pass the instant the quotes show (a delayed feed's quote_time, see
    delayed_quote_time), not the fetch time.

    'intraday'  09:30 ET to the 16:15 data cutoff of ``session`` itself
    'gth'       inside the Global Trading Hours window (20:15 ET the evening
                before, to 09:15 ET) of the session AFTER ``session``: these
                quotes trade for the next session (option_chain.t_origin
                measures T from the quote time for them)
    'pre_open'  09:15-09:30 ET of the next session (GTH closed, RTH not open)
    'closing'   anything else: the evening after the close, weekends and
                holidays — the session's closing book
    """
    from joint_vol_calibration.data import option_chain as oc
    if session is None:
        return "closing"
    ny = db._now_ny(at)
    if db.is_intraday(at) and ny.date().isoformat() == session:
        return "intraday"
    nxt = oc.next_session(session)
    lo, hi = oc.gth_window(nxt)
    if lo <= ny < hi:
        return "gth"
    if ny.date() == nxt and hi <= ny < datetime.combine(nxt, db.SESSION_OPEN, db.NY_TZ):
        return "pre_open"
    return "closing"


def capture_option_chain(underlying: str, now: Optional[datetime] = None, *,
                         include_extras: bool = False, timeout: float = 60.0) -> ChainCapture:
    """
    Download one full delayed-quotes chain (``_SPX.json`` / ``_VIX.json``) and
    return it normalised and timestamped (ChainCapture). Stores nothing.

    The frame is what download_cboe_options_snapshot returns (the
    options_snapshots schema, quote_time = the delayed instant the quotes
    show and source='cboe' on every row),
    computed from the same bytes the sha256 is taken of, so a later
    re-parse of a privately archived payload can be checked against the
    hash. ``now`` is the fetch time used for dating (default: the clock).

    Licensing: Cboe's website terms allow viewing and downloading for
    personal, non-commercial use and bar storing or redistributing the
    Materials (DATA_SOURCES.md). Nothing here writes the quotes to the
    repository; capture_option_chains writes only where its caller says.
    """
    symbol = _quote_symbol(underlying)
    url = _CBOE_QUOTE_API.format(symbol=symbol)
    fetched = now or db._utcnow()
    resp = _get(url, timeout=timeout)
    raw_bytes = resp.content
    payload = json.loads(raw_bytes.decode("utf-8"))
    return _capture_from_payload(underlying, payload, raw_bytes, url, fetched, include_extras)


def _capture_from_payload(underlying: str, payload: dict, raw_bytes: bytes, url: str,
                          fetched: datetime, include_extras: bool = False) -> ChainCapture:
    und = underlying.upper().lstrip("_^")
    frame = parse_cboe_options_payload(payload, und, now=fetched)
    data = payload.get("data") or {}
    if include_extras and not frame.empty:
        raw = pd.DataFrame(data.get("options") or [])
        raw = raw[raw.get("option", pd.Series(dtype=object)).astype(str).isin(frame["contract_symbol"])]
        keep = {k: v for k, v in _CAPTURE_EXTRAS.items() if k in raw.columns}
        extra = raw[["option", *keep]].rename(columns={"option": "contract_symbol", **keep})
        for col in extra.columns:
            if col not in ("contract_symbol", "option_last_trade_time"):
                extra[col] = pd.to_numeric(extra[col], errors="coerce")
        frame = frame.merge(extra.drop_duplicates("contract_symbol"), on="contract_symbol", how="left")
    session = str(frame["snapshot_date"].iloc[0]) if len(frame) else None
    stamp = pd.to_datetime(payload.get("timestamp"), errors="coerce")
    spot = data.get("close") or data.get("current_price")
    quote_time = str(frame["quote_time"].iloc[0]) if len(frame) else delayed_quote_time(
        payload.get("timestamp"), fetched)
    quoted_at = datetime.fromisoformat(quote_time.replace("Z", "+00:00"))
    return ChainCapture(
        underlying=und, frame=frame, session=session, quote_time=quote_time,
        book_state=book_state_at(quoted_at, session),
        fetched_at=db.utc_iso(fetched),
        payload_timestamp=None if pd.isna(stamp) else stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        underlying_last_trade_time=data.get("last_trade_time"),
        underlying_price=float(spot) if spot else float("nan"),
        url=url, sha256=hashlib.sha256(raw_bytes).hexdigest(), n_bytes=len(raw_bytes),
    )


def capture_option_chains(underlyings: Iterable[str] = ("SPX", "VIX"), now: Optional[datetime] = None, *,
                          out_dir=None, include_extras: bool = False) -> dict:
    """
    capture_option_chain for each underlying, back to back (the SPX and VIX
    books a joint fit needs, fetched about a second apart). Returns
    {underlying: ChainCapture}; a failed download is logged and left out.

    With ``out_dir`` (required to write anything; there is no default under
    data_store/) each frame goes to <out_dir>/<und>_<session>_<HHMMSSZ>.parquet
    and every summary to <out_dir>/<und>_<session>_<HHMMSSZ>.json. Each
    capture's own fetch time is recorded; ``now`` overrides the clock for all.
    """
    out = {}
    for und in underlyings:
        try:
            cap = capture_option_chain(und, now=now, include_extras=include_extras)
        except Exception as e:
            logger.warning("Cboe %s chain capture failed: %s", und, e)
            continue
        out[cap.underlying] = cap
        if out_dir is not None and cap.session is not None:
            from pathlib import Path
            d = Path(out_dir)
            d.mkdir(parents=True, exist_ok=True)
            stem = f"{cap.underlying.lower()}_{cap.session}_{cap.fetched_at[11:19].replace(':', '')}Z"
            cap.frame.to_parquet(d / f"{stem}.parquet", index=False, compression="zstd")
            (d / f"{stem}.json").write_text(json.dumps(cap.summary(), indent=2) + "\n")
        logger.info("Captured %s chain: %d rows, session %s, book %s, payload %s", cap.underlying,
                    len(cap.frame), cap.session, cap.book_state, cap.payload_timestamp)
    return out


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalise_expiry_month(series: pd.Series) -> pd.Series:
    """
    Convert VX contract names to 'YYYY-MM'.

    Handles Cboe's per-contract 'Futures' text ('V (Oct 2026)', 'F (Jan 2024)'),
    'Jan-24', 'JAN/2024', 'Jan 2024', 'January 2024', and 'YYYY-MM'. Anything
    else is returned unchanged (with a warning) rather than guessed.
    """
    fmts = ("%b %Y", "%B %Y", "%b-%y", "%b-%Y", "%B-%Y", "%b/%Y", "%B/%Y", "%Y-%m")

    def _parse_one(val) -> str:
        if pd.isna(val):
            return ""
        s = str(val).strip()
        m = re.match(r"^([FGHJKMNQUVXZ])\s*\(\s*([A-Za-z]{3,9})\s+(\d{4})\s*\)$", s)
        if m:
            s = f"{m.group(2)} {m.group(3)}"
        for fmt in fmts:
            try:
                return datetime.strptime(s.title() if fmt != "%Y-%m" else s, fmt).strftime("%Y-%m")
            except ValueError:
                continue
        logger.warning("Unrecognised VX contract name %r", val)
        return str(val)

    return series.apply(_parse_one)

