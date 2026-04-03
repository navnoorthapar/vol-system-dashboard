"""
test_rates.py — C10-Step4: Tests for per-date T-bill rate integration.

18 tests covering:
  - database schema: tbill_rates table exists
  - insert_tbill_rates() round-trip
  - get_tbill_rate() look-ahead guard
  - get_tbill_rate() fallback when no data
  - get_tbill_rates_series() date range filter
  - yf_downloader.download_tbill_rate() structure (mock)
  - BacktestEngine._simulate_straddle_pnl accepts Series r
  - Zero look-ahead: rate series never returns future dates
"""

import sqlite3
from datetime import date, timedelta
from unittest.mock import patch, MagicMock

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.data import database as db
from joint_vol_calibration.data.database import (
    insert_tbill_rates,
    get_tbill_rate,
    get_tbill_rates_series,
    init_database,
)
from joint_vol_calibration.config import DB_PATH


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(autouse=True)
def ensure_schema():
    """Make sure the DB schema includes tbill_rates table."""
    init_database()


@pytest.fixture
def sample_rates() -> pd.DataFrame:
    """Small set of test rates: 2020-01-02 through 2020-01-10 (weekdays)."""
    dates = pd.bdate_range("2020-01-02", "2020-01-10").strftime("%Y-%m-%d").tolist()
    rates = np.linspace(0.015, 0.020, len(dates))
    return pd.DataFrame({"date": dates, "rate": rates})


@pytest.fixture
def loaded_rates(sample_rates) -> pd.DataFrame:
    """Insert sample rates into DB and return the DataFrame."""
    insert_tbill_rates(sample_rates)
    return sample_rates


# ── Schema tests ──────────────────────────────────────────────────────────────

class TestSchema:
    def test_tbill_rates_table_exists(self):
        """tbill_rates table must be created by init_database()."""
        with sqlite3.connect(str(DB_PATH)) as conn:
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='tbill_rates'"
            ).fetchone()
        assert row is not None, "tbill_rates table not found in DB"

    def test_tbill_rates_columns(self):
        """tbill_rates must have 'date' and 'rate' columns."""
        with sqlite3.connect(str(DB_PATH)) as conn:
            cursor = conn.execute("PRAGMA table_info(tbill_rates)")
            cols = {row[1] for row in cursor.fetchall()}
        assert "date" in cols, "'date' column missing from tbill_rates"
        assert "rate" in cols, "'rate' column missing from tbill_rates"


# ── Insert tests ──────────────────────────────────────────────────────────────

class TestInsertTbillRates:
    def test_insert_returns_row_count(self, sample_rates):
        n = insert_tbill_rates(sample_rates)
        assert n == len(sample_rates), f"Expected {len(sample_rates)} rows inserted, got {n}"

    def test_insert_idempotent(self, sample_rates):
        """Second insert replaces rows (INSERT OR REPLACE) without raising."""
        insert_tbill_rates(sample_rates)
        n2 = insert_tbill_rates(sample_rates)
        assert n2 == len(sample_rates)

    def test_insert_rate_decimal(self, sample_rates):
        """Rates must be stored as decimal (< 1.0), not percent."""
        insert_tbill_rates(sample_rates)
        with sqlite3.connect(str(DB_PATH)) as conn:
            rows = conn.execute(
                "SELECT rate FROM tbill_rates WHERE date >= '2020-01-02' AND date <= '2020-01-10'"
            ).fetchall()
        rates = [r[0] for r in rows]
        assert all(r < 1.0 for r in rates), f"Rates should be decimal, got: {rates[:3]}"
        assert all(r > 0.0 for r in rates), "All rates should be positive"


# ── get_tbill_rate() tests ─────────────────────────────────────────────────────

class TestGetTbillRate:
    def test_returns_most_recent_on_exact_date(self, loaded_rates):
        """get_tbill_rate on a date in DB returns the stored rate."""
        row = loaded_rates[loaded_rates["date"] == "2020-01-06"].iloc[0]
        r = get_tbill_rate("2020-01-06")
        assert abs(r - row["rate"]) < 1e-8

    def test_look_ahead_guard_no_future_data(self, loaded_rates):
        """get_tbill_rate never returns data from after as_of_date."""
        r = get_tbill_rate("2020-01-02")  # earliest date in fixture
        # The rate should be the one for 2020-01-02, not a later one
        first_rate = loaded_rates[loaded_rates["date"] == "2020-01-02"]["rate"].iloc[0]
        assert abs(r - first_rate) < 1e-8

    def test_fallback_when_no_data_before_date(self):
        """Returns fallback rate when no data exists before query date."""
        r = get_tbill_rate("2001-01-15", fallback=0.055)
        assert abs(r - 0.055) < 1e-8

    def test_fallback_default_is_0045(self):
        """Default fallback is 0.045 (historical baseline)."""
        r = get_tbill_rate("2001-01-15")
        assert abs(r - 0.045) < 1e-8, f"Default fallback should be 0.045, got {r}"

    def test_uses_most_recent_not_exact_match(self, loaded_rates):
        """Query on weekend (no market data) returns prior Friday rate."""
        # 2020-01-04 is Saturday — no DB row; should return 2020-01-03 rate
        r = get_tbill_rate("2020-01-04")
        fri_rate = loaded_rates[loaded_rates["date"] == "2020-01-03"]["rate"].iloc[0]
        assert abs(r - fri_rate) < 1e-8


# ── get_tbill_rates_series() tests ────────────────────────────────────────────

class TestGetTbillRatesSeries:
    def test_returns_dataframe(self, loaded_rates):
        df = get_tbill_rates_series("2020-01-10")
        assert isinstance(df, pd.DataFrame)
        assert "date" in df.columns
        assert "rate" in df.columns

    def test_no_future_dates(self, loaded_rates):
        """Zero look-ahead: all rows must have date <= as_of_date."""
        cutoff = "2020-01-06"
        df = get_tbill_rates_series(cutoff)
        if not df.empty:
            max_date = df["date"].max()
            if hasattr(max_date, "strftime"):
                max_date_str = max_date.strftime("%Y-%m-%d")
            else:
                max_date_str = str(max_date)[:10]
            assert max_date_str <= cutoff, f"Look-ahead bias: {max_date_str} > {cutoff}"

    def test_start_date_filter(self, loaded_rates):
        """start_date parameter correctly limits lower bound."""
        df = get_tbill_rates_series("2020-01-10", start_date="2020-01-07")
        if not df.empty:
            min_date = df["date"].min()
            min_str = min_date.strftime("%Y-%m-%d") if hasattr(min_date, "strftime") else str(min_date)[:10]
            assert min_str >= "2020-01-07", f"start_date filter failed: {min_str}"

    def test_sorted_ascending(self, loaded_rates):
        """Series must be sorted ascending by date."""
        df = get_tbill_rates_series("2020-01-10")
        if len(df) > 1:
            dates = pd.to_datetime(df["date"])
            assert (dates.diff().dropna() >= pd.Timedelta(0)).all(), "Dates not ascending"


# ── BacktestEngine integration tests ──────────────────────────────────────────

class TestBacktestRateIntegration:
    def test_simulate_straddle_accepts_series_r(self):
        """_simulate_straddle_pnl must accept a date-indexed Series for r."""
        from joint_vol_calibration.backtest.backtest_engine import (
            _simulate_straddle_pnl, R_BACKTEST,
        )

        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        position   = pd.Series([1, 1, 1, 0, 0], index=dates, dtype=float)
        kelly      = pd.Series([0.1]*5, index=dates, dtype=float)
        nav_series = pd.Series([1_000_000.0]*5, index=dates, dtype=float)
        spx_close  = pd.Series([3200.0, 3210.0, 3205.0, 3220.0, 3215.0], index=dates)
        vix_wide   = pd.DataFrame({
            "^VIX":   [15.0]*5,
            "^VIX3M": [16.0]*5,
        }, index=dates)

        # Scalar r — baseline
        pnl_scalar, _ = _simulate_straddle_pnl(
            position, kelly, nav_series, spx_close, vix_wide,
            r=R_BACKTEST, q=0.013,
        )
        # Series r — per-date
        r_series = pd.Series([R_BACKTEST]*5, index=dates)
        pnl_series, _ = _simulate_straddle_pnl(
            position, kelly, nav_series, spx_close, vix_wide,
            r=r_series, q=0.013,
        )
        # Constant Series should give identical result to scalar
        pd.testing.assert_series_equal(pnl_scalar, pnl_series, check_names=False)

    def test_simulate_straddle_varying_r(self):
        """Varying per-date rates produce different P&L from fixed rate."""
        from joint_vol_calibration.backtest.backtest_engine import _simulate_straddle_pnl

        dates = pd.date_range("2020-01-02", periods=5, freq="B")
        position   = pd.Series([1, 1, 1, 0, 0], index=dates, dtype=float)
        kelly      = pd.Series([0.15]*5, index=dates, dtype=float)
        nav_series = pd.Series([1_000_000.0]*5, index=dates, dtype=float)
        spx_close  = pd.Series([3200.0, 3210.0, 3205.0, 3220.0, 3215.0], index=dates)
        vix_wide   = pd.DataFrame({
            "^VIX":   [20.0]*5,
            "^VIX3M": [22.0]*5,
        }, index=dates)

        r_fixed  = pd.Series([0.045]*5, index=dates)
        r_varied = pd.Series([0.01, 0.02, 0.03, 0.04, 0.05], index=dates)

        pnl_fixed,  _ = _simulate_straddle_pnl(position, kelly, nav_series, spx_close, vix_wide, r=r_fixed)
        pnl_varied, _ = _simulate_straddle_pnl(position, kelly, nav_series, spx_close, vix_wide, r=r_varied)

        # Results should differ (rate affects option pricing)
        assert not pnl_fixed.equals(pnl_varied), "Varying rates should produce different P&L"


class TestTbillPipelineIntegration:
    def test_rate_stored_as_decimal(self, sample_rates):
        """Rates inserted via insert_tbill_rates must always be decimal (0.0–0.2)."""
        insert_tbill_rates(sample_rates)
        r = get_tbill_rate("2020-01-10")
        assert 0.0 < r < 0.2, f"Rate out of expected decimal range: {r}"

    def test_get_tbill_rate_validates_date_format(self):
        """get_tbill_rate() must raise ValueError on invalid date format."""
        from joint_vol_calibration.data.database import get_tbill_rate
        with pytest.raises(ValueError, match="Invalid date format"):
            get_tbill_rate("20200101")  # wrong format
