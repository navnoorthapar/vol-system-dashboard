"""
cboe_downloader.py — Downloads free data directly from CBOE's public endpoints.

Free data available from CBOE:
  1. VIX index history          — CBOE CDN CSV (daily OHLC since 1990)
  2. VIX futures settlement     — CBOE settlement price archive CSVs
  3. VIX futures daily history  — CBOE historical futures data by contract
  4. SPX/VIX options (delayed)  — CBOE public quote API (15-min delay)

Financial context:
  VIX = 30-day realised vol expectation under risk-neutral measure.
  VIX futures are forward contracts on the VIX index at expiry.
  The term structure of VIX futures encodes the market's forward vol expectations.
  We need this data to compute the Heston/PDV model implied VIX futures prices
  and compare them to market prices — the core "gap" we are trying to close.
"""

import io
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import requests

logger = logging.getLogger(__name__)

# ── CBOE public data URLs ─────────────────────────────────────────────────────

_VIX_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VIX_History.csv"
)

# VIX futures settlement archive — one CSV per year, rows = contracts settled that year
# Format: https://cdn.cboe.com/resources/futures/vx-historical-settlement-data.csv
_VIX_SETTLEMENT_URL = (
    "https://cdn.cboe.com/resources/futures/vx-historical-settlement-data.csv"
)

# CBOE delayed quote API — free, 15-min delay
_CBOE_QUOTE_API = "https://cdn.cboe.com/api/global/delayed_quotes/options/{underlying}.json"

# CBOE VIX futures daily price history (all contracts, ~2004-present)
# This is the main historical futures file published by CBOE
_VIX_FUTURES_HISTORY_URL = (
    "https://cdn.cboe.com/api/global/us_indices/daily_prices/VX_History.csv"
)

_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
}

_SESSION = requests.Session()
_SESSION.headers.update(_HEADERS)


# ── VIX Index History ─────────────────────────────────────────────────────────

def download_vix_history() -> pd.DataFrame:
    """
    Download VIX index daily OHLC from CBOE's public CDN.

    The CSV has headers: Date,OPEN,HIGH,LOW,CLOSE
    Date format in file: MM/DD/YYYY — we normalise to YYYY-MM-DD.

    Financial interpretation:
      VIX close = market's 30-day forward annualised vol expectation.
      High VIX = fear / uncertainty. Low VIX = complacency.
      We use this as input to regime classification and to compare against
      realised vol (20-day rolling SPX realised vol).

    Returns
    -------
    DataFrame with columns [date, open, high, low, close].
    """
    logger.info("Downloading VIX history from CBOE CDN ...")
    resp = _SESSION.get(_VIX_HISTORY_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(
        io.StringIO(resp.text),
        skiprows=0,
    )

    # Normalise column names
    df.columns = [c.strip().lower() for c in df.columns]

    # CBOE uses "date" or "DATE" — handle both
    date_col = [c for c in df.columns if "date" in c.lower()]
    if not date_col:
        raise ValueError(f"No date column found. Columns: {df.columns.tolist()}")
    df.rename(columns={date_col[0]: "date"}, inplace=True)

    # Parse date — CBOE uses MM/DD/YYYY
    df["date"] = pd.to_datetime(df["date"], format="mixed", dayfirst=False)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d")

    # Rename OHLC if needed
    rename_map = {}
    for col in df.columns:
        stripped = col.strip().lower()
        if stripped in ("open", "high", "low", "close"):
            rename_map[col] = stripped
    df.rename(columns=rename_map, inplace=True)

    numeric_cols = ["open", "high", "low", "close"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df.dropna(subset=["date", "close"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d VIX daily rows (%s to %s)",
                len(df), df["date"].iloc[0], df["date"].iloc[-1])
    return df[["date", "open", "high", "low", "close"]]


# ── VIX Futures Settlements ───────────────────────────────────────────────────

def download_vix_futures_settlements() -> pd.DataFrame:
    """
    Download VIX futures final settlement prices from CBOE's archive.

    CBOE publishes a CSV with every VIX futures final settlement
    going back to 2004. Each row contains:
      - settlement_date: the Wednesday on which the contract expired
      - expiry_month: the contract identifier (e.g. 'Jan-24')
      - settlement_price: the official VIX settlement value

    Financial interpretation:
      Settlement price = special opening quotation (SOQ) of the VIX
      on the morning of the third Wednesday of the expiry month.
      This is the price at which all open VIX futures positions close.
      These are ground truth data points for model validation —
      if Heston implies VIX settlement S' but market settled at S,
      the absolute error |S' - S| is direct evidence of model failure.

    Returns
    -------
    DataFrame with columns [settlement_date, expiry_month, settlement_price].
    expiry_month is normalised to 'YYYY-MM' format.
    """
    logger.info("Downloading VIX futures settlements from CBOE ...")
    resp = _SESSION.get(_VIX_SETTLEMENT_URL, timeout=30)
    resp.raise_for_status()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = [c.strip() for c in df.columns]

    # CBOE format has columns like: Date, Futures, Open, High, Low, Close, ...
    # or "Trade Date", "Expiration Date", "Final Settlement Price"
    # Normalise to our schema
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    # Map common CBOE column names
    col_map = {
        "trade_date": "settlement_date",
        "expiration_date": "expiry_month",
        "final_settlement_price": "settlement_price",
        "settle": "settlement_price",
        "settlement": "settlement_price",
        "futures": "expiry_month",
        "date": "settlement_date",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    # If we have settlement_date and expiry_month
    if "settlement_date" not in df.columns:
        logger.warning("Unexpected CBOE settlement file format. Columns: %s", df.columns.tolist())
        return pd.DataFrame(columns=["settlement_date", "expiry_month", "settlement_price"])

    df["settlement_date"] = pd.to_datetime(df["settlement_date"], format="mixed", errors="coerce")
    df["settlement_date"] = df["settlement_date"].dt.strftime("%Y-%m-%d")
    df["settlement_price"] = pd.to_numeric(df["settlement_price"], errors="coerce")
    df.dropna(subset=["settlement_date", "settlement_price"], inplace=True)

    # Normalise expiry_month to YYYY-MM
    if "expiry_month" in df.columns:
        df["expiry_month"] = _normalise_expiry_month(df["expiry_month"])

    df.sort_values("settlement_date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d VIX futures settlement rows", len(df))
    return df[["settlement_date", "expiry_month", "settlement_price"]]


# ── VIX Futures Daily History ──────────────────────────────────────────────────

def download_vix_futures_daily_history() -> pd.DataFrame:
    """
    Download the full VIX futures price history (all contracts, daily).

    CBOE publishes a unified CSV with daily OHLCV for all historical
    VIX futures contracts from 2004 to present.

    Each row represents one contract on one trading day.
    We compute days_to_expiry = expiry_date - trade_date.

    Financial interpretation:
      The term structure (futures prices by expiry) tells us:
        - Contango (far > near): market expects vol to rise
        - Backwardation (near > far): elevated spot VIX, fear premium in front month
      The slope of the term structure is one of our regime classifier features.

    Returns
    -------
    DataFrame with columns:
      [date, expiry_month, open, high, low, close, volume, open_interest,
       days_to_expiry]
    """
    logger.info("Downloading VIX futures daily history from CBOE ...")
    resp = _SESSION.get(_VIX_FUTURES_HISTORY_URL, timeout=60)
    resp.raise_for_status()

    raw = resp.text
    # CBOE VX history is a long CSV, may have comment rows at top
    # Skip lines that don't look like data
    lines = [l for l in raw.split("\n") if l.strip() and not l.startswith("#")]
    clean = "\n".join(lines)

    df = pd.read_csv(io.StringIO(clean))
    df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

    # Standard CBOE VX history columns:
    # trade_date, futures, open, high, low, close, settle, change,
    # total_volume, efp, open_interest
    col_map = {
        "trade_date": "date",
        "futures":    "expiry_month",
        "settle":     "close",
        "total_volume": "volume",
    }
    df.rename(columns={k: v for k, v in col_map.items() if k in df.columns}, inplace=True)

    if "date" not in df.columns:
        raise ValueError(f"Cannot find date column. Got: {df.columns.tolist()}")

    df["date"] = pd.to_datetime(df["date"], format="mixed", errors="coerce").dt.strftime("%Y-%m-%d")
    df["expiry_month"] = _normalise_expiry_month(df["expiry_month"])

    for col in ["open", "high", "low", "close"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    for col in ["volume", "open_interest"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    # Compute days_to_expiry: expiry_month 'YYYY-MM' → last business day of that month
    df["expiry_date"] = pd.to_datetime(df["expiry_month"] + "-01") + pd.offsets.BMonthEnd(0)
    df["days_to_expiry"] = (df["expiry_date"] - pd.to_datetime(df["date"])).dt.days
    df.drop(columns=["expiry_date"], inplace=True)

    # Drop rows with non-positive days_to_expiry (expired contracts)
    df = df[df["days_to_expiry"] > 0]

    df.dropna(subset=["date", "close", "expiry_month"], inplace=True)
    df.sort_values(["date", "days_to_expiry"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    final_cols = ["date", "expiry_month", "open", "high", "low", "close",
                  "volume", "open_interest", "days_to_expiry"]
    final_cols = [c for c in final_cols if c in df.columns]

    logger.info("Downloaded %d VIX futures daily rows (%s to %s)",
                len(df), df["date"].iloc[0], df["date"].iloc[-1])
    return df[final_cols]


# ── CBOE Delayed Options Quote ─────────────────────────────────────────────────

def download_cboe_options_snapshot(underlying: str) -> pd.DataFrame:
    """
    Download the current (15-min delayed) options chain from CBOE's public API.

    CBOE publishes delayed quotes for SPX and VIX options at no cost
    via their JSON endpoint. This is primarily used for:
      1. Daily snapshotting to build historical options database over time
      2. Live dashboard updates (with 15-min delay caveat)

    Parameters
    ----------
    underlying : str — 'SPX' or 'VIX'

    Returns
    -------
    DataFrame with columns:
      [snapshot_date, expiry, strike, right, bid, ask, mid_price,
       implied_vol, open_interest, volume, delta, gamma, vega, theta,
       time_to_expiry]

    Note:
      - implied_vol from CBOE is already annualised.
      - time_to_expiry is in years.
    """
    ticker = underlying.upper()
    url = _CBOE_QUOTE_API.format(underlying=ticker)

    logger.info("Fetching %s options snapshot from CBOE ...", ticker)
    resp = _SESSION.get(url, timeout=30)
    resp.raise_for_status()

    payload = resp.json()
    today = datetime.today().strftime("%Y-%m-%d")

    rows = []
    data = payload.get("data", {})

    # CBOE JSON structure: data.options is a list of option objects
    options = data.get("options", [])
    if not options:
        # Try alternate structure
        options = data.get("optionChain", [])

    if not options:
        logger.warning("No options data returned for %s", ticker)
        return pd.DataFrame()

    for opt in options:
        try:
            expiry_raw = opt.get("expiration_date") or opt.get("expiration") or ""
            if not expiry_raw:
                continue

            expiry = pd.to_datetime(expiry_raw, format="mixed").strftime("%Y-%m-%d")
            tte = _years_to_expiry(today, expiry)
            if tte <= 0:
                continue

            row = {
                "snapshot_date":  today,
                "expiry":         expiry,
                "strike":         float(opt.get("strike", 0)),
                "right":          "C" if str(opt.get("option_type", "")).upper().startswith("C") else "P",
                "bid":            _safe_float(opt.get("bid")),
                "ask":            _safe_float(opt.get("ask")),
                "mid_price":      _safe_float(opt.get("mid")),
                "implied_vol":    _safe_float(opt.get("iv")),
                "open_interest":  _safe_int(opt.get("open_interest")),
                "volume":         _safe_int(opt.get("volume")),
                "delta":          _safe_float(opt.get("delta")),
                "gamma":          _safe_float(opt.get("gamma")),
                "vega":           _safe_float(opt.get("vega")),
                "theta":          _safe_float(opt.get("theta")),
                "time_to_expiry": tte,
            }
            # Compute mid if not provided
            if row["mid_price"] is None and row["bid"] is not None and row["ask"] is not None:
                row["mid_price"] = (row["bid"] + row["ask"]) / 2.0

            rows.append(row)
        except (KeyError, ValueError, TypeError) as e:
            logger.debug("Skipping option row due to parse error: %s", e)
            continue

    if not rows:
        logger.warning("Parsed 0 rows from CBOE %s options response", ticker)
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Drop zero/invalid strikes
    df = df[df["strike"] > 0]
    df.sort_values(["expiry", "strike", "right"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Fetched %d %s option rows for snapshot date %s", len(df), ticker, today)
    return df


# ── Utilities ─────────────────────────────────────────────────────────────────

def _normalise_expiry_month(series: pd.Series) -> pd.Series:
    """
    Convert various VIX futures contract name formats to 'YYYY-MM'.

    CBOE uses formats like: 'Jan-24', 'JAN/2024', 'F4', '(F)Jan 2024', '2024-01'
    We normalise all to 'YYYY-MM'.
    """
    def _parse_one(val: str) -> str:
        if pd.isna(val):
            return ""
        val = str(val).strip()
        # Already YYYY-MM
        if len(val) == 7 and val[4] == "-":
            return val
        # Try pandas to_datetime
        try:
            dt = pd.to_datetime(val, format="mixed", errors="coerce")
            if pd.notna(dt):
                return dt.strftime("%Y-%m")
        except Exception:
            pass
        # Try month-abbreviation formats like 'Jan-24' or 'Jan 2024'
        for fmt in ("%b-%y", "%b %Y", "%b-%Y", "%B %Y", "%B-%Y"):
            try:
                dt = datetime.strptime(val.strip("()"), fmt)
                return dt.strftime("%Y-%m")
            except ValueError:
                continue
        return val  # return as-is if cannot parse

    return series.apply(_parse_one)


def _years_to_expiry(today: str, expiry: str) -> float:
    """Return time to expiry in years (act/365)."""
    t = datetime.strptime(today, "%Y-%m-%d")
    e = datetime.strptime(expiry, "%Y-%m-%d")
    return max((e - t).days / 365.0, 0.0)


def _safe_float(val) -> Optional[float]:
    try:
        return float(val) if val is not None and val != "" else None
    except (ValueError, TypeError):
        return None


def _safe_int(val) -> Optional[int]:
    try:
        return int(val) if val is not None and val != "" else None
    except (ValueError, TypeError):
        return None
