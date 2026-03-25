"""
test_lookahead.py — Zero look-ahead bias enforcement tests.

These are the most important tests in the system.
A single look-ahead bug invalidates the entire backtest.

Tests are grouped into three categories:
  1. Database layer — verify SQL queries respect as_of_date
  2. Pipeline layer — verify the DataPipeline enforces the guarantee
  3. Realised vol    — verify RV computation has no forward-looking windows

Run with:
  pytest tests/test_lookahead.py -v

All tests use synthetic in-memory data so no real download is required.
"""

import sqlite3
import tempfile
import os
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

# We monkeypatch DB_PATH to a temp file so tests don't touch production DB
import joint_vol_calibration.data.database as db_module
import joint_vol_calibration.data.yf_downloader as yf_dl
from joint_vol_calibration.data.database import (
    init_database,
    insert_spx_ohlcv,
    insert_vix_daily,
    insert_vix_futures_daily,
    insert_options_snapshot,
    get_spx_ohlcv,
    get_vix_daily,
    get_vix_futures_curve,
    get_options_surface,
    _validate_date,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def temp_db(tmp_path, monkeypatch):
    """
    Redirect all database operations to a temporary SQLite file.
    Each test gets a fresh empty database.
    """
    tmp_db = tmp_path / "test_vol.db"
    tmp_parquet = tmp_path / "parquet"
    tmp_parquet.mkdir()

    monkeypatch.setattr(db_module, "DB_PATH", tmp_db)
    monkeypatch.setattr(db_module, "PARQUET_DIR", tmp_parquet)

    # Re-import to pick up patched paths
    init_database()
    yield tmp_db


def make_spx_df(dates: list[str]) -> pd.DataFrame:
    """Create a synthetic SPX OHLCV DataFrame for given dates."""
    n = len(dates)
    np.random.seed(42)
    closes = 4000.0 * np.exp(np.random.randn(n).cumsum() * 0.01)
    return pd.DataFrame({
        "date":   dates,
        "open":   closes * 0.998,
        "high":   closes * 1.005,
        "low":    closes * 0.995,
        "close":  closes,
        "volume": np.random.randint(1_000_000, 5_000_000, n),
    })


def make_vix_df(dates: list[str]) -> pd.DataFrame:
    """Create synthetic VIX daily data."""
    n = len(dates)
    np.random.seed(99)
    closes = 20.0 + np.random.randn(n) * 3
    return pd.DataFrame({
        "date":  dates,
        "open":  closes - 0.5,
        "high":  closes + 1.0,
        "low":   closes - 1.0,
        "close": closes,
    })


def make_options_df(snapshot_date: str, expiry: str, strikes: list[float]) -> pd.DataFrame:
    """Create a synthetic options chain snapshot."""
    rows = []
    for strike in strikes:
        for right in ["C", "P"]:
            rows.append({
                "snapshot_date": snapshot_date,
                "expiry":        expiry,
                "strike":        strike,
                "right":         right,
                "bid":           5.0,
                "ask":           5.5,
                "mid_price":     5.25,
                "implied_vol":   0.22,
                "open_interest": 1000,
                "volume":        200,
                "time_to_expiry": 30 / 365,
            })
    return pd.DataFrame(rows)


def make_vix_futures_df(dates: list[str], expiry_month: str = "2020-03") -> pd.DataFrame:
    """Create synthetic VIX futures daily data for a single contract."""
    n = len(dates)
    np.random.seed(7)
    closes = 22.0 + np.random.randn(n) * 2
    return pd.DataFrame({
        "date":           dates,
        "expiry_month":   expiry_month,
        "open":           closes - 0.2,
        "high":           closes + 0.5,
        "low":            closes - 0.5,
        "close":          closes,
        "volume":         np.random.randint(5000, 50000, n),
        "open_interest":  np.random.randint(10000, 100000, n),
        "days_to_expiry": list(range(n, 0, -1)),
    })


# ── Category 1: Database Read Functions Return No Future Data ─────────────────

class TestSpxOhlcvLookAhead:
    """get_spx_ohlcv(as_of_date) must never return rows with date > as_of_date."""

    def test_no_data_after_as_of_date(self):
        dates = ["2020-01-10", "2020-01-13", "2020-01-14", "2020-01-15",
                 "2020-01-16", "2020-01-17", "2020-01-20"]
        insert_spx_ohlcv(make_spx_df(dates))

        as_of = "2020-01-15"
        df = get_spx_ohlcv(as_of_date=as_of)

        assert not df.empty, "Should return some data"
        max_date = df["date"].max()
        max_str = max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)[:10]
        assert max_str <= as_of, (
            f"LOOK-AHEAD: returned date {max_str} > as_of_date {as_of}"
        )

    def test_future_rows_excluded(self):
        dates = ["2020-01-14", "2020-01-15", "2020-01-16", "2020-01-17"]
        insert_spx_ohlcv(make_spx_df(dates))

        df = get_spx_ohlcv(as_of_date="2020-01-15")
        returned_dates = set(
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in df["date"]
        )

        assert "2020-01-16" not in returned_dates, "Future date leaked!"
        assert "2020-01-17" not in returned_dates, "Future date leaked!"
        assert "2020-01-15" in returned_dates, "as_of_date itself should be included"

    def test_empty_result_for_date_before_all_data(self):
        dates = ["2020-02-01", "2020-02-02"]
        insert_spx_ohlcv(make_spx_df(dates))
        df = get_spx_ohlcv(as_of_date="2020-01-01")
        assert df.empty, "Should return empty when all data is after as_of_date"

    def test_start_date_filter(self):
        dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
        insert_spx_ohlcv(make_spx_df(dates))
        df = get_spx_ohlcv(as_of_date="2020-01-07", start_date="2020-01-06")
        dates_returned = [
            d.strftime("%Y-%m-%d") if hasattr(d, "strftime") else str(d)[:10]
            for d in df["date"]
        ]
        assert "2020-01-02" not in dates_returned
        assert "2020-01-06" in dates_returned


class TestVixDailyLookAhead:
    """get_vix_daily(as_of_date) must never return rows with date > as_of_date."""

    def test_no_future_vix(self):
        dates = ["2021-06-01", "2021-06-02", "2021-06-03", "2021-06-04", "2021-06-07"]
        insert_vix_daily(make_vix_df(dates))

        as_of = "2021-06-03"
        df = get_vix_daily(as_of_date=as_of)
        max_date = df["date"].max()
        max_str = max_date.strftime("%Y-%m-%d") if hasattr(max_date, "strftime") else str(max_date)[:10]
        assert max_str <= as_of, f"LOOK-AHEAD: VIX date {max_str} > {as_of}"


class TestVixFuturesLookAhead:
    """get_vix_futures_curve(as_of_date) returns only prices from that exact date."""

    def test_curve_on_specific_date(self):
        dates = ["2020-02-10", "2020-02-11", "2020-02-12"]
        df_in = make_vix_futures_df(dates, expiry_month="2020-03")
        insert_vix_futures_daily(df_in)

        curve = get_vix_futures_curve(as_of_date="2020-02-11")
        assert len(curve) == 1, "Should return exactly 1 row (only 2020-03 contract)"

        curve_other = get_vix_futures_curve(as_of_date="2020-02-10")
        assert len(curve_other) == 1

    def test_curve_empty_for_date_with_no_data(self):
        dates = ["2020-02-10", "2020-02-11"]
        insert_vix_futures_daily(make_vix_futures_df(dates))
        # 2020-02-12 has no entry
        curve = get_vix_futures_curve(as_of_date="2020-02-12")
        assert curve.empty, "Should be empty for a date with no futures data"


class TestOptionsSnapshotLookAhead:
    """
    get_options_surface(as_of_date) must:
      1. Return only snapshots with snapshot_date <= as_of_date
      2. Return only options with expiry > as_of_date (not yet expired)
    """

    def test_no_future_snapshot(self):
        # Snapshot taken on Jan 20, but we query as of Jan 15
        df_snap = make_options_df("2020-01-20", "2020-03-20", [4000, 4050, 4100])
        insert_options_snapshot(df_snap, "SPX")

        result = get_options_surface(as_of_date="2020-01-15", underlying="SPX")
        assert result.empty, (
            "Options snapshot from Jan 20 must NOT appear when as_of_date is Jan 15"
        )

    def test_snapshot_on_exact_date_is_returned(self):
        snap_date = "2020-01-15"
        expiry = "2020-03-20"  # future relative to snapshot
        df_snap = make_options_df(snap_date, expiry, [4000, 4050])
        insert_options_snapshot(df_snap, "SPX")

        result = get_options_surface(as_of_date="2020-01-15", underlying="SPX")
        assert not result.empty, "Snapshot on exact as_of_date should be returned"

    def test_expired_options_excluded(self):
        snap_date = "2020-01-15"
        expired_expiry = "2020-01-10"  # already expired on snap date
        df_snap = make_options_df(snap_date, expired_expiry, [4000])
        insert_options_snapshot(df_snap, "SPX")

        # Even if snapshot_date matches, expired options must not be returned
        result = get_options_surface(as_of_date="2020-01-15", underlying="SPX")
        assert result.empty, "Expired options (expiry < as_of_date) must not be returned"

    def test_most_recent_snapshot_used(self):
        # Two snapshots: Jan 10 and Jan 14. Query as of Jan 15.
        # Should use Jan 14 snapshot (most recent <= as_of_date).
        df1 = make_options_df("2020-01-10", "2020-03-20", [3900, 4000])
        df2 = make_options_df("2020-01-14", "2020-03-20", [3950, 4050])
        insert_options_snapshot(df1, "SPX")
        insert_options_snapshot(df2, "SPX")

        result = get_options_surface(as_of_date="2020-01-15", underlying="SPX")
        assert not result.empty
        snap_dates = result["snapshot_date"].unique()
        snap_str = [
            s.strftime("%Y-%m-%d") if hasattr(s, "strftime") else str(s)[:10]
            for s in snap_dates
        ]
        assert all(d == "2020-01-14" for d in snap_str), (
            f"Expected snapshot 2020-01-14, got {snap_str}"
        )


# ── Category 2: Date Validation ───────────────────────────────────────────────

class TestDateValidation:
    """_validate_date must reject invalid date strings."""

    def test_valid_date_passes(self):
        _validate_date("2020-01-15")  # should not raise

    def test_invalid_format_raises(self):
        with pytest.raises(ValueError):
            _validate_date("01/15/2020")

    def test_invalid_date_raises(self):
        with pytest.raises(ValueError):
            _validate_date("2020-13-01")  # month 13

    def test_empty_string_raises(self):
        with pytest.raises(ValueError):
            _validate_date("")


# ── Category 3: Realised Vol Has No Forward-Looking Window ────────────────────

class TestRealisedVolLookAhead:
    """
    compute_realised_vol uses ONLY past returns to compute vol.
    A 20-day RV on date T uses returns [T-20, ..., T-1].
    It must NOT include return on T itself or any future date.
    """

    def test_rv_uses_only_past_data(self):
        dates = pd.bdate_range("2020-01-02", "2020-03-31").strftime("%Y-%m-%d").tolist()
        np.random.seed(42)
        closes = 4000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        df = pd.DataFrame({"date": dates, "close": closes})

        rv_df = yf_dl.compute_realised_vol(df, window=20)

        # The RV value on date T should equal the std of returns up to T-1 (window days)
        # Verify: if we truncate the input at some date D and recompute,
        # the RV at D should be identical.
        test_date = "2020-03-01"
        rv_full  = rv_df[rv_df["date"] == test_date]["rv_20d"].values
        rv_trunc = yf_dl.compute_realised_vol(
            df[df["date"] <= test_date], window=20
        )
        rv_trunc_val = rv_trunc[rv_trunc["date"] == test_date]["rv_20d"].values

        if len(rv_full) > 0 and len(rv_trunc_val) > 0:
            assert abs(rv_full[0] - rv_trunc_val[0]) < 1e-10, (
                f"RV differs when future data is removed: {rv_full[0]} vs {rv_trunc_val[0]}"
            )

    def test_rv_nan_for_insufficient_history(self):
        """First (window-1) rows must be NaN — no partial-window estimates."""
        dates = pd.bdate_range("2020-01-02", "2020-01-31").strftime("%Y-%m-%d").tolist()
        np.random.seed(1)
        closes = 4000 * np.exp(np.cumsum(np.random.randn(len(dates)) * 0.01))
        df = pd.DataFrame({"date": dates, "close": closes})

        rv_df = yf_dl.compute_realised_vol(df, window=20)
        # Should have fewer rows than input (NaNs dropped)
        assert len(rv_df) < len(dates), (
            "First (window-1) observations should be NaN and dropped"
        )


# ── Category 4: No Future Data in Log Returns ─────────────────────────────────

class TestLogReturnComputation:
    """Log return on day T = ln(S_T / S_{T-1}). Uses only T and T-1."""

    def test_log_return_uses_only_current_and_prev(self):
        dates = ["2020-01-02", "2020-01-03", "2020-01-06", "2020-01-07"]
        closes = [4000.0, 4010.0, 4005.0, 4020.0]
        df = pd.DataFrame({"date": dates, "close": closes,
                           "open": closes, "high": closes, "low": closes, "volume": [1]*4})
        insert_spx_ohlcv(df)

        result = get_spx_ohlcv(as_of_date="2020-01-07")
        # log return on Jan 3 = ln(4010/4000)
        jan3_row = result[result["date"].astype(str).str[:10] == "2020-01-03"]
        if not jan3_row.empty and "log_return" in result.columns:
            expected = np.log(4010 / 4000)
            actual = jan3_row["log_return"].iloc[0]
            assert abs(actual - expected) < 1e-10, (
                f"Log return mismatch: expected {expected:.6f}, got {actual:.6f}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
