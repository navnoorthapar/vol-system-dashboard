"""
yf_downloader.py — Yahoo Finance data downloads for SPX OHLCV and options.

yfinance is the primary source for:
  1. SPX daily OHLCV (^GSPC) — going back to 1927
  2. VIX index daily levels (^VIX) — as a cross-check against CBOE CSV
  3. SPX/VIX current options chains — near-term expirations only

Limitations to be aware of:
  - yfinance options chains: only provides current and near-term expirations
    (typically < 3 months out). Historical chains are NOT available.
  - For historical options chains we must rely on snapshots accumulated over time
    (see pipeline.py which calls this daily) or CBOE bulk data files.
  - yfinance data has occasional gaps/errors — we validate and flag them.

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

logger = logging.getLogger(__name__)

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
    end   : str 'YYYY-MM-DD' — end date (inclusive). Defaults to yesterday.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close, volume].
    Note: log_return is computed in database.py to avoid duplication.
    """
    if end is None:
        end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading SPX OHLCV from %s to %s ...", start, end)

    ticker = yf.Ticker("^GSPC")
    df = ticker.history(
        start=start,
        end=end,
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
    Download VIX index daily OHLC from Yahoo Finance (cross-check vs CBOE CSV).

    Yahoo Finance carries ^VIX back to 1990.
    We use CBOE as primary source; this is a secondary validation source.

    Returns
    -------
    DataFrame with columns [date, open, high, low, close].
    """
    if end is None:
        end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading VIX index from Yahoo Finance %s to %s ...", start, end)

    df = yf.download("^VIX", start=start, end=end, interval="1d",
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

    logger.info("Downloaded %d VIX rows from Yahoo Finance", len(df))
    return df


# ── SPX / VIX Options Chains ──────────────────────────────────────────────────

def snapshot_spx_options() -> pd.DataFrame:
    """
    Download the current SPX options chain from Yahoo Finance.

    yfinance provides options for near-term expirations (usually 8-12 dates).
    We snapshot all available expirations.

    Financial context:
      The SPX implied vol surface is the primary calibration target.
      We fit models to minimise distance from this surface.
      The key features per option are:
        - Implied vol (IV): the market's vol quote — what we calibrate to
        - Delta: moneyness proxy
        - Bid/ask spread: liquidity proxy (wide spread → don't calibrate here)

    Returns
    -------
    DataFrame with columns:
      [snapshot_date, expiry, strike, right, bid, ask, mid_price,
       implied_vol, open_interest, volume, delta, time_to_expiry]
    """
    logger.info("Fetching SPX options chain from Yahoo Finance ...")
    return _fetch_yf_options("^SPX", "SPX")


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
    return _fetch_yf_options("^VIX", "VIX")


def _fetch_yf_options(ticker_symbol: str, underlying_name: str) -> pd.DataFrame:
    """
    Internal helper: fetch all available option expirations for a ticker.

    Parameters
    ----------
    ticker_symbol   : yfinance ticker string (e.g. '^SPX', '^VIX')
    underlying_name : label to use in the output DataFrame

    Returns
    -------
    Combined DataFrame across all available expirations.
    """
    today = datetime.today().strftime("%Y-%m-%d")
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
    n = len(raw)
    expiry_str = pd.to_datetime(raw["expiry"]).dt.strftime("%Y-%m-%d")
    bid  = pd.to_numeric(raw.get("bid",  pd.Series(np.nan, index=raw.index)), errors="coerce")
    ask  = pd.to_numeric(raw.get("ask",  pd.Series(np.nan, index=raw.index)), errors="coerce")
    iv   = pd.to_numeric(raw.get("implied_vol", pd.Series(np.nan, index=raw.index)), errors="coerce")
    oi   = pd.to_numeric(raw.get("open_interest", pd.Series(np.nan, index=raw.index)), errors="coerce")
    vol  = pd.to_numeric(raw.get("volume",        pd.Series(np.nan, index=raw.index)), errors="coerce")
    tte  = (pd.to_datetime(expiry_str) - pd.to_datetime(today)).dt.days / 365.0

    out = pd.DataFrame({
        "snapshot_date":  today,                        # scalar → broadcasts to all n rows
        "expiry":         expiry_str.values,
        "strike":         pd.to_numeric(raw["strike"], errors="coerce").values,
        "right":          raw["right"].values,
        "bid":            bid.values,
        "ask":            ask.values,
        "mid_price":      ((bid + ask) / 2.0).values,
        "implied_vol":    iv.values,
        "open_interest":  oi.astype("Int64").values,
        "volume":         vol.astype("Int64").values,
        "time_to_expiry": tte.values,
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

    logger.info("Fetched %d %s option rows across %d expirations",
                len(out), underlying_name, out["expiry"].nunique())
    return out


# ── VIX Term Structure ────────────────────────────────────────────────────────

# CBOE publishes the full VIX term structure as separate index products.
# These are NOT futures prices — they are the implied variance at each horizon
# ON EACH DAY. However, they encode the same information as the VIX futures
# term structure and serve as our primary free proxy.
#
# Relationship to VIX futures:
#   VIX_futures(T) = E^Q[VIX(T)] ≈ sqrt(E^Q[VIX^2(T)]) ≈ VIX_{Td}(today)
#   where VIX_{Td} is the CBOE index measuring Td-day implied vol.
#   The approximation error is the Jensen inequality gap + risk premium,
#   typically 0.5–2 VIX points.
#
# We store all four indices daily and use them for:
#   1. Term structure slope (regime classifier feature)
#   2. Joint calibration target (with appropriate risk-premium correction)
#   3. VIX-of-VIX (^VVIX) for volga/vomma regime detection

_VIX_TERM_TICKERS = {
    "^VIX9D": 9,    # 9-day
    "^VIX":   30,   # 30-day (standard VIX)
    "^VIX3M": 93,   # 3-month
    "^VIX6M": 182,  # 6-month
    "^VVIX":  30,   # VIX of VIX — tracks vol of vol; used for C6 vomma regime
}


def download_vix_term_structure(
    start: str = "2010-01-01",
    end: Optional[str] = None,
) -> pd.DataFrame:
    """
    Download the CBOE VIX term structure indices from Yahoo Finance.

    Downloads ^VIX9D, ^VIX, ^VIX3M, ^VIX6M, and ^VVIX in one call.
    Returns a tidy long-format DataFrame with one row per (date, tenor).

    Financial interpretation:
      slope = VIX3M - VIX9D  (term structure slope, key regime feature)
      slope > 0 : contango — market expects vol to rise (backwardation in VIX)
      slope < 0 : inverted — front VIX elevated, spike regime

    Parameters
    ----------
    start, end : str 'YYYY-MM-DD'

    Returns
    -------
    DataFrame with columns [date, tenor_days, close].
    Indexed by (date, tenor_days).
    """
    if end is None:
        end = (datetime.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    logger.info("Downloading VIX term structure indices from %s to %s ...", start, end)

    tickers_str = " ".join(_VIX_TERM_TICKERS.keys())
    raw = yf.download(tickers_str, start=start, end=end, interval="1d",
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
    melted = melted[melted["close"] > 0]
    melted.sort_values(["date", "tenor_days"], inplace=True)
    melted.reset_index(drop=True, inplace=True)

    logger.info("Downloaded %d VIX term structure rows (dates: %s to %s)",
                len(melted), melted["date"].iloc[0], melted["date"].iloc[-1])
    return melted[["date", "ticker", "tenor_days", "close", "is_vvix"]]


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
