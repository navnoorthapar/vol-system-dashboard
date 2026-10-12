"""
yf_downloader.py — Yahoo Finance data downloads for SPX OHLCV and options.

yfinance is the source for:
  1. SPX daily OHLCV (^GSPC) — going back to 1927
  2. VIX index daily levels (^VIX) — Cboe's own CSV is the reference (see
     cboe_downloader and database.cross_check_vix)
  3. The Cboe VIX term-structure indices ^VIX9D/^VIX/^VIX3M/^VIX6M and ^VVIX —
     Yahoo serves their full daily history
  4. The 13-week T-bill yield (^IRX)
  5. SPX/VIX option chains — the CURRENT chain only

Limitations to be aware of:
  - yfinance option chains are snapshots of the current quotes; historical
    chains are NOT available. Option history exists only as the chains this
    project stores (database.insert_options_snapshot) and persists
    (database.export_options_snapshot).
  - Yahoo history is revised after the fact: bars are dropped and restored,
    and the occasional bad print appears (2026-02-06: ^VIX served with the
    ^VIX3M value). The database upserts never delete a stored session, and
    its validation hooks log missing sessions and cross-source mismatches.
  - yfinance treats ``end`` as EXCLUSIVE. Every function here takes an
    INCLUSIVE end and shifts it internally via ``_yf_end`` — do not pre-shift.
    The default end is the last NYSE session whose close has printed
    (database.last_closed_session), not the machine's "yesterday".

Financial context:
  SPX OHLCV is the primary time series for:
    - Computing realised volatility (20d, 30d, 60d rolling windows)
    - Fitting the PDV model (sigma(t) = f(past log returns))
    - Delta-hedging backtest (we need daily SPX closes)
    - Regime classification features
"""

import logging
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf

from joint_vol_calibration.config import (
    SPX_OPTIONS_TICKER,
    SPX_TICKER,
    TBILL_TICKER,
    VIX_INDEX_TENOR_DAYS,
    VIX_OPTIONS_TICKER,
    VIX_TICKER,
    VVIX_TENOR_DAYS,
)
from joint_vol_calibration.data import database as db

logger = logging.getLogger(__name__)


def _yf_end(end: str) -> str:
    """Convert an INCLUSIVE end date to the EXCLUSIVE form yfinance expects.

    Every download function in this module documents ``end`` as inclusive, but
    ``yfinance`` (both ``download`` and ``Ticker.history``) stops one day
    *before* ``end``. Passing the run date straight through therefore dropped
    the session that had just closed on every run that fired before midnight
    UTC — the site was one session stale, and the calibrator (which pins the
    VIX leg to the SPX spot session) found no term-structure row for that
    session. (Yahoo serves the full daily history of ^VIX9D/^VIX3M/^VIX6M;
    the missing rows were this off-by-one, not a current-day-only feed.)
    Shift by one calendar day so the last requested session is actually
    returned.
    """
    return (datetime.strptime(end, "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")


def _default_end(end: Optional[str]) -> str:
    """``end`` itself, or the last closed NYSE session when None."""
    return end if end is not None else db.last_closed_session()


# ── SPX OHLCV ─────────────────────────────────────────────────────────────────

def download_spx_ohlcv(
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download SPX (S&P 500) daily OHLCV from Yahoo Finance.

    Uses the ^GSPC ticker which goes back to 1927.
    We normalise column names and add a log_return column.

    Parameters
    ----------
    start : str 'YYYY-MM-DD' — start date (inclusive).
    end   : str 'YYYY-MM-DD' — end date (inclusive). Defaults to the last
            closed NYSE session.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, volume].
    Note: log_return is computed in database.py to avoid duplication.
    """
    end = _default_end(end)

    logger.info("Downloading SPX OHLCV from %s to %s ...", start, end)

    ticker = yf.Ticker(SPX_TICKER)
    df = ticker.history(
        start=start,
        end=_yf_end(end),
        interval="1d",
        auto_adjust=True,   # adjusts for splits and dividends
        back_adjust=False,
    )

    if df.empty:
        logger.error("yfinance returned empty DataFrame for SPX")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.reset_index(inplace=True)

    # Normalise column names
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    rename_map = {"datetime": "date", "index": "date", "stock_splits": "splits"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    # Keep only OHLCV
    keep = ["date", "open", "high", "low", "close", "volume"]
    df = df[[c for c in keep if c in df.columns]].copy()

    # Validate
    n_before = len(df)
    df.dropna(subset=["close"], inplace=True)
    df = df[df["close"] > 0]
    if len(df) < n_before:
        logger.warning("Dropped %d SPX rows with invalid close prices", n_before - len(df))

    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["volume"] = df["volume"].fillna(0).astype(int)

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d SPX OHLCV rows (%s to %s)",
                len(df), df["date"].iloc[0], df["date"].iloc[-1])
    return df


# ── VIX Index ─────────────────────────────────────────────────────────────────

def download_vix_index(
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download VIX index daily OHLC from Yahoo Finance.

    Yahoo Finance carries ^VIX back to 1990. Cboe's VIX_History.csv is the
    official series (cboe_downloader.download_vix_history); Yahoo is the
    same-evening source, since the Cboe file can lag by a session.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, source='yfinance'].
    """
    end = _default_end(end)

    logger.info("Downloading VIX index from Yahoo Finance %s to %s ...", start, end)

    df = yf.download(VIX_TICKER, start=start, end=_yf_end(end), interval="1d",
                     auto_adjust=True, progress=False)

    if df.empty:
        logger.error("yfinance returned empty DataFrame for VIX")
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.reset_index(inplace=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    rename_map = {"datetime": "date", "date": "date", "price": "close"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    keep = ["date", "open", "high", "low", "close"]
    df = df[[c for c in keep if c in df.columns]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df.dropna(subset=["close"], inplace=True)
    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)
    df["source"] = "yfinance"

    logger.info("Downloaded %d VIX rows from Yahoo Finance", len(df))
    return df


# ── 13-Week T-Bill Rate (^IRX) ────────────────────────────────────────────────

def download_tbill_rate(
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download the 13-week US T-bill yield from Yahoo Finance (^IRX).

    ^IRX quotes the 13-week bill's BANK-DISCOUNT yield in percent (4.5 means
    4.5%). It is not a continuously compounded rate, which is what the pricers
    and the backtest's exp(-r T) discounting assume, so each print d is
    converted with database.tbill_discount_to_cc:

        rate = -ln(1 - d * 91/360) * 365/91       (d as a decimal)

    which adds ~3-10 bp at 2-5% yields. Zero and negative prints are real
    (2020-03-19..27 printed between -0.105% and -0.028%) and are kept.

    Zero look-ahead: callers must gate reads via get_tbill_rate(as_of_date).

    Returns
    -------
    DataFrame with columns [date, rate, discount_yield]: rate is the
    continuously compounded decimal, discount_yield the raw quote / 100.
    """
    end = _default_end(end)

    logger.info("Downloading 13-week T-bill yield (%s) from %s to %s ...", TBILL_TICKER, start, end)

    df = yf.download(TBILL_TICKER, start=start, end=_yf_end(end), interval="1d",
                     auto_adjust=False, progress=False)

    if df.empty:
        logger.error("yfinance returned empty DataFrame for %s", TBILL_TICKER)
        return pd.DataFrame()

    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.reset_index(inplace=True)
    df.columns = [c[0].lower() if isinstance(c, tuple) else c.lower() for c in df.columns]

    rename_map = {"datetime": "date", "date": "date"}
    df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns}, inplace=True)

    if "close" not in df.columns:
        logger.error("%s DataFrame missing 'close' column: %s", TBILL_TICKER, df.columns.tolist())
        return pd.DataFrame()

    df = df[["date", "close"]].copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")
    df["close"] = pd.to_numeric(df["close"], errors="coerce")
    df.dropna(subset=["close"], inplace=True)

    df["discount_yield"] = df["close"] / 100.0                  # percent -> decimal
    df["rate"] = db.tbill_discount_to_cc(df["discount_yield"])  # -> continuously compounded
    df = df[["date", "rate", "discount_yield"]]

    df.sort_values("date", inplace=True)
    df.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d T-bill rate rows (%d at or below zero)",
                len(df), int((df["discount_yield"] <= 0).sum()))
    return df


# ── SPX / VIX Options Chains ──────────────────────────────────────────────────

def snapshot_spx_options() -> pd.DataFrame:
    """
    Download the current SPX options chain from Yahoo Finance.

    Yahoo's ^SPX chain lists every expiry it carries (dozens) and, under one
    expiry date, mixes AM-settled SPX and PM-settled SPXW contracts; each row's
    root and settlement style are recovered from its OCC contract symbol.

    Financial context:
      The SPX implied vol surface is the primary calibration target.
      We fit models to minimise distance from this surface.
      The key features per option are:
        - Implied vol (IV): Yahoo's own IV — opaque, and not re-derived from
          our mid; calibrators should invert the mid themselves
        - Bid/ask spread: liquidity proxy (wide spread → don't calibrate here)

    Returns
    -------
    DataFrame with the options_snapshots columns (see _fetch_yf_options).
    """
    logger.info("Fetching SPX options chain from Yahoo Finance ...")
    return _fetch_yf_options(SPX_OPTIONS_TICKER, "SPX")


def snapshot_vix_options() -> pd.DataFrame:
    """
    Download the current VIX options chain from Yahoo Finance.

    VIX options are the third leg of the joint calibration.
    They are European-style and settle to the SOQ (Special Opening Quotation)
    of VIX on expiration Wednesday.

    Key difference from SPX options:
      - Underlying is VIX itself (a computed index), not a tradeable asset
      - Put-call parity does NOT hold the same way as for equity options
      - Forward VIX ≠ spot VIX × e^(rT) because VIX is non-investable
      - The correct pricing requires integrating over the vol-of-vol process

    Returns
    -------
    Same schema as snapshot_spx_options().
    """
    logger.info("Fetching VIX options chain from Yahoo Finance ...")
    return _fetch_yf_options(VIX_OPTIONS_TICKER, "VIX")


def _underlying_close(ticker, session: str) -> float:
    """Close of ``ticker`` on ``session`` (the latest bar if that session has none yet)."""
    try:
        hist = ticker.history(period="10d", interval="1d")
    except Exception as e:   # pragma: no cover - network
        logger.warning("Could not fetch the underlying level: %s", e)
        return float("nan")
    if hist is None or hist.empty:
        return float("nan")
    idx = pd.to_datetime(hist.index)
    if idx.tz is not None:
        idx = idx.tz_convert(db.NY_TZ).tz_localize(None)
    closes = pd.Series(hist["Close"].to_numpy(), index=idx.strftime("%Y-%m-%d"))
    return float(closes.get(session, closes.iloc[-1]))


def _fetch_yf_options(ticker_symbol: str, underlying_name: str,
                      now: Optional[datetime] = None) -> pd.DataFrame:
    """
    Internal helper: fetch all available option expirations for a ticker.

    The snapshot is dated by the NY session the quotes belong to
    (database.quote_session): a chain fetched on a weekend, a holiday or
    before the open carries the previous session's quotes and gets that date.
    time_to_expiry is measured from that session's close to each contract's
    settlement (database.time_to_settlement; AM vs PM from the OCC root).

    Parameters
    ----------
    ticker_symbol   : yfinance ticker string (e.g. '^SPX', '^VIX')
    underlying_name : label to use in the output DataFrame
    now             : fetch time (default: the clock); used for dating

    Returns
    -------
    Combined DataFrame across all available expirations.
    """
    fetched_at = now or db._utcnow()
    session = db.quote_session(fetched_at)
    if db.is_intraday(fetched_at):
        logger.warning("%s chain fetched during the %s session: these are intraday quotes, "
                       "not closing quotes", underlying_name, session)
    ticker = yf.Ticker(ticker_symbol)

    try:
        expirations = ticker.options
    except Exception as e:
        logger.error("Failed to fetch option expirations for %s: %s", ticker_symbol, e)
        return pd.DataFrame()

    if not expirations:
        logger.warning("No option expirations returned for %s", ticker_symbol)
        return pd.DataFrame()

    all_rows = []
    for expiry in expirations:
        try:
            chain = ticker.option_chain(expiry)
        except Exception as e:
            logger.warning("Failed to fetch chain for %s expiry %s: %s",
                           ticker_symbol, expiry, e)
            continue

        for right, df_side in [("C", chain.calls), ("P", chain.puts)]:
            if df_side is None or df_side.empty:
                continue
            df_side = df_side.copy()
            df_side["right"] = right
            df_side["expiry"] = expiry
            all_rows.append(df_side)

    if not all_rows:
        logger.warning("No option data for %s", ticker_symbol)
        return pd.DataFrame()

    raw = pd.concat(all_rows, ignore_index=True)

    # yfinance column names vary slightly by version — normalise
    col_map = {
        "contractSymbol":    "contract_symbol",
        "lastTradeDate":     "last_trade_date",
        "strike":            "strike",
        "lastPrice":         "last_price",
        "bid":               "bid",
        "ask":               "ask",
        "change":            "change",
        "percentChange":     "pct_change",
        "volume":            "volume",
        "openInterest":      "open_interest",
        "impliedVolatility": "implied_vol",
        "inTheMoney":        "in_the_money",
        "contractSize":      "contract_size",
        "currency":          "currency",
    }
    raw.rename(columns={k: v for k, v in col_map.items() if k in raw.columns}, inplace=True)
    raw.columns = [c.lower() for c in raw.columns]

    # Build clean output — construct via dict so every column broadcasts correctly.
    # DO NOT assign column-by-column to an empty DataFrame: in pandas 2.x the scalar
    # snapshot_date assignment produces a 0-row Series that never propagates.
    expiry_str = pd.to_datetime(raw["expiry"]).dt.strftime("%Y-%m-%d")
    symbols = raw.get("contract_symbol", pd.Series(None, index=raw.index))
    roots = [(db.parse_occ_symbol(s) or {}).get("root", "") for s in symbols]
    settlement = [db.settlement_for_root(r) for r in roots]
    bid  = pd.to_numeric(raw.get("bid",  pd.Series(np.nan, index=raw.index)), errors="coerce")
    ask  = pd.to_numeric(raw.get("ask",  pd.Series(np.nan, index=raw.index)), errors="coerce")
    iv   = pd.to_numeric(raw.get("implied_vol", pd.Series(np.nan, index=raw.index)), errors="coerce")
    oi   = pd.to_numeric(raw.get("open_interest", pd.Series(np.nan, index=raw.index)), errors="coerce")
    vol  = pd.to_numeric(raw.get("volume",        pd.Series(np.nan, index=raw.index)), errors="coerce")
    tte  = [db.time_to_settlement(session, e, s) for e, s in zip(expiry_str, settlement)]

    out = pd.DataFrame({
        "snapshot_date":    session,                      # scalar → broadcasts to all n rows
        "expiry":           expiry_str.values,
        "strike":           pd.to_numeric(raw["strike"], errors="coerce").values,
        "right":            raw["right"].values,
        "bid":              bid.values,
        "ask":              ask.values,
        "mid_price":        ((bid + ask) / 2.0).values,
        "implied_vol":      iv.values,
        "open_interest":    oi.astype("Int64").values,
        "volume":           vol.astype("Int64").values,
        "time_to_expiry":   tte,
        "root":             roots,
        "contract_symbol":  symbols.values,
        "settlement":       settlement,
        "underlying_price": _underlying_close(ticker, session),
        "quote_time":       db.utc_iso(fetched_at),
        "source":           "yfinance",
    })

    # Drop expired and zero-strike rows
    out = out[(out["strike"] > 0) & (out["time_to_expiry"] > 0)]

    # Drop options with zero/NaN implied vol (not tradeable / illiquid)
    out = out[out["implied_vol"] > 0]

    # Filter out very wide bid/ask (> 20% of mid) — these are illiquid strikes
    spread = out["ask"] - out["bid"]
    out = out[
        out["mid_price"].isna() |
        out["mid_price"].le(0) |
        (spread / out["mid_price"].replace(0, np.nan) < 0.20)
    ]

    out.sort_values(["expiry", "right", "strike"], inplace=True)
    out.reset_index(drop=True, inplace=True)

    logger.info("Fetched %d %s option rows across %d expirations (quote session %s)",
                len(out), underlying_name, out["expiry"].nunique(), session)
    return out


# ── VIX Term Structure ────────────────────────────────────────────────────────

# Cboe publishes constant-maturity volatility indices at several horizons. Each
# is a SPOT-starting variance-swap rate: the expected volatility from TODAY to
# its horizon T, VIX_T^2 = (1/T) * integral_0^T E[v_s] ds. They are NOT VIX
# futures. A future expiring at T settles on the 30-day volatility over
# [T, T+30d], so in contango the 30-day future sits well above ^VIX (on
# 2026-09-22 the 29-day VX settled at 17.36 against ^VIX 14.21). Model them
# with the term-index formula, not a futures curve; real VX futures come from
# cboe_downloader.
#
# We store all of them daily and use them for:
#   1. Term structure slope (regime classifier feature)
#   2. Joint calibration target (the VIX leg)
#   3. VIX-of-VIX (^VVIX) for volga/vomma regime detection
#
# Horizons come from config.VIX_INDEX_TENOR_DAYS — the one tenor map.

_VIX_TERM_TICKERS = {**VIX_INDEX_TENOR_DAYS, "^VVIX": VVIX_TENOR_DAYS}


def download_vix_term_structure(
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download the Cboe VIX term-structure indices from Yahoo Finance.

    Downloads ^VIX9D, ^VIX, ^VIX3M, ^VIX6M, and ^VVIX in one call.
    Returns a tidy long-format DataFrame with one row per (date, ticker).

    Financial interpretation:
      slope = VIX3M - VIX9D  (term structure slope, key regime feature)
      slope > 0 : upward-sloping (contango) — the usual calm-market shape
      slope < 0 : inverted — front VIX elevated, spike regime

    Parameters
    ----------
    start, end : str 'YYYY-MM-DD' (inclusive; end defaults to the last closed session)

    Returns
    -------
    DataFrame with columns [date, ticker, tenor_days, close, is_vvix, source].
    """
    end = _default_end(end)

    logger.info("Downloading VIX term structure indices from %s to %s ...", start, end)

    tickers_str = " ".join(_VIX_TERM_TICKERS.keys())
    raw = yf.download(tickers_str, start=start, end=_yf_end(end), interval="1d",
                      auto_adjust=True, progress=False)

    if raw.empty:
        logger.error("VIX term structure download returned empty")
        return pd.DataFrame()

    # yfinance returns MultiIndex columns when multiple tickers
    # Shape: (dates, (field, ticker))
    if isinstance(raw.columns, pd.MultiIndex):
        close_df = raw["Close"].copy()
    else:
        close_df = raw[["Close"]].copy()

    close_df.index = pd.to_datetime(close_df.index).tz_localize(None)
    close_df.reset_index(inplace=True)
    close_df.rename(columns={"Date": "date", "index": "date", "Datetime": "date"}, inplace=True)

    # Melt to long format: (date, ticker, close)
    id_vars = [c for c in close_df.columns if c in ("date", "Date")]
    if not id_vars:
        close_df.rename(columns={close_df.columns[0]: "date"}, inplace=True)
        id_vars = ["date"]

    melted = close_df.melt(id_vars=id_vars[0], var_name="ticker", value_name="close")
    melted.rename(columns={id_vars[0]: "date"}, inplace=True)

    # Map ticker → tenor_days
    melted["tenor_days"] = melted["ticker"].map(_VIX_TERM_TICKERS)
    melted["is_vvix"]    = melted["ticker"] == "^VVIX"
    melted["date"] = pd.to_datetime(melted["date"]).dt.strftime("%Y-%m-%d")
    melted.dropna(subset=["close"], inplace=True)
    melted = melted[melted["close"] > 0].copy()
    melted["source"] = "yfinance"
    melted.sort_values(["date", "tenor_days"], inplace=True)
    melted.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d VIX term structure rows (dates: %s to %s)",
                len(melted), melted["date"].iloc[0], melted["date"].iloc[-1])
    return melted[["date", "ticker", "tenor_days", "close", "is_vvix", "source"]]


# ── Realised Volatility Helper ────────────────────────────────────────────────

def compute_realised_vol(
    df_spx: pd.DataFrame,
    window: int = 20,
    annualisation: int = 252,
) -> pd.DataFrame:
    """
    Compute rolling realised volatility from SPX log returns.

    Realised vol (RV) is the central empirical quantity in vol trading:
      RV_t = sqrt(252 / N * sum_{i=1}^{N} r_{t-i}^2)
    where r_t = ln(S_t / S_{t-1}).

    This is the 'realised' side of the Implied vs Realised spread:
      Edge = IV (VIX) - RV (SPX)
    When Edge > 0: vol sellers are being paid a premium (short gamma profitable)
    When Edge < 0: realised vol surprises to the upside (long gamma profitable)

    Parameters
    ----------
    df_spx  : DataFrame with [date, close] or [date, log_return].
    window  : int — rolling window in trading days (default 20 ≈ 1 month).
    annualisation : int — trading days per year.

    Returns
    -------
    DataFrame with columns [date, rv_{window}d] (vol in decimal, annualised).
    """
    df = df_spx.copy()
    df["date"] = pd.to_datetime(df["date"])
    df.sort_values("date", inplace=True)

    if "log_return" not in df.columns:
        df["log_return"] = np.log(df["close"] / df["close"].shift(1))

    col = f"rv_{window}d"
    df[col] = (
        df["log_return"]
        .rolling(window)
        .apply(lambda x: np.sqrt(annualisation * np.sum(x**2) / window), raw=True)
    )

    df["date"] = df["date"].dt.strftime("%Y-%m-%d")
    return df[["date", col]].dropna().reset_index(drop=True)
