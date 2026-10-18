"""
test_lookahead.py — Zero look-ahead bias enforcement tests.

These are the most important tests in the system.
A single look-ahead bug invalidates the entire backtest.

Tests are grouped into these categories:
  1. Database layer — verify SQL queries respect as_of_date
  2. Pipeline layer — verify the DataPipeline enforces the guarantee
  3. Realised vol    — verify RV computation has no forward-looking windows
  4. Log returns     — r_t uses only S_t and S_{t-1}
  5. Signals, regime labels, the R2 gate and the HMM — the trading path.
     Each test is mutation-checked: removing the 1-day feature or regime lag,
     a full-sample z-score, a forward-looking R2 gate, a t+1 return in the
     label's realised vol, or HMM smoothing makes at least one of them fail.

Run with:
  pytest tests/test_lookahead.py -v

All tests use synthetic in-memory data so no real download is required.
"""


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
        # MUST be a business day: the fixture is a bdate_range, so a weekend
        # date yields no row and the guarded assertion below silently never
        # runs. 2020-03-01 was a Sunday, which made this test vacuous — it
        # passed even against a deliberately centred (forward-looking) window.
        test_date = "2020-03-02"
        rv_full  = rv_df[rv_df["date"] == test_date]["rv_20d"].values
        rv_trunc = yf_dl.compute_realised_vol(
            df[df["date"] <= test_date], window=20
        )
        rv_trunc_val = rv_trunc[rv_trunc["date"] == test_date]["rv_20d"].values

        assert len(rv_full) == 1 and len(rv_trunc_val) == 1, (
            f"no RV row for {test_date} — the fixture stopped producing it, so "
            "this test would silently assert nothing"
        )
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


# ── Category 5: Signals, regime labels, R2 gate, HMM ─────────────────────────

from joint_vol_calibration.signals.regime_classifier import (  # noqa: E402
    HMMRegimeClassifier,
    build_features,
    build_regime_labels,
    r2_gate_threshold,
)
from joint_vol_calibration.signals.signal_engine import SignalEngine  # noqa: E402


def _signal_market(n: int = 800, seed: int = 11):
    """GARCH-like returns; VIX tracks realised vol, VVIX tracks VIX, so the
    labels move through R0/R1/R2 and every signal has something to do."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2016-01-04", periods=n)
    s2 = np.empty(n)
    r = np.empty(n)
    s2[0], r[0] = 1e-4, 0.0
    for t in range(1, n):
        s2[t] = 3e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
        r[t] = rng.normal(0.0, np.sqrt(s2[t]))
    rv = np.sqrt(s2 * 252)
    vix = np.clip(15 + 35 * (rv - rv.mean()) / rv.std() + rng.normal(0, 1.5, n), 9, 60)
    vvix = np.clip(90 + 12 * (vix - vix.mean()) / vix.std() + rng.normal(0, 4, n), 75, 160)
    spx = pd.DataFrame({"date": dates, "close": 3000 * np.exp(np.cumsum(r)), "log_return": r})
    vixw = pd.DataFrame({"date": dates, "^VIX": vix,
                         "^VIX3M": vix + rng.normal(1.5, 1.0, n), "^VVIX": vvix})
    return spx, vixw


def _poison(spx, vix, from_date, ret, v, v3m, vv, inclusive=True):
    sp, vp = spx.copy(), vix.copy()
    ds, dv = pd.to_datetime(sp["date"]), pd.to_datetime(vp["date"])
    ms = ds >= from_date if inclusive else ds > from_date
    mv = dv >= from_date if inclusive else dv > from_date
    sp.loc[ms, "log_return"] = ret
    sp.loc[ms, "close"] = sp.loc[ms, "close"] * 5.0
    vp.loc[mv, "^VIX"] = v
    vp.loc[mv, "^VIX3M"] = v3m
    vp.loc[mv, "^VVIX"] = vv
    return sp, vp


# Two opposite poisons so a dependency cannot hide behind a saturated label.
POISONS = [
    dict(ret=0.25, v=5.0, v3m=90.0, vv=999.0),   # realised >> implied, VVIX huge
    dict(ret=0.0, v=90.0, v3m=10.0, vv=50.0),    # implied >> realised, VVIX tiny
]
_SIG_WINDOW = ("2016-01-01", "2020-01-01")


@pytest.fixture(scope="module")
def clean_signals():
    spx, vix = _signal_market()
    return spx, vix, SignalEngine().generate(spx, vix, *_SIG_WINDOW)


class TestSignalEngineSameDayPoison:
    """Signals at session t trade at the close of t on information dated
    ≤ t−1. So poisoning everything dated ≥ t (the cutoff session INCLUDED)
    must leave every numeric column of every row dated ≤ t bit-identical —
    positions, strengths, z-scores, features and the regime. The older
    end-to-end test poisoned strictly AFTER the cutoff and compared positions
    only, so a missing 1-day lag or a full-sample z-score went unnoticed."""

    @pytest.mark.parametrize("frac", [0.3, 0.45, 0.6, 0.75])
    @pytest.mark.parametrize("pz", POISONS, ids=["hot", "cold"])
    def test_rows_up_to_cutoff_unchanged(self, clean_signals, frac, pz):
        spx, vix, clean = clean_signals
        # a cutoff whose lagged regime is R1, so a forward-looking label flips
        start = clean.index[int(len(clean) * frac)]
        cutoff = clean.index[(clean["regime"] == 1) & (clean.index >= start)][0]
        sp, vp = _poison(spx, vix, cutoff, **pz)
        pois = SignalEngine().generate(sp, vp, *_SIG_WINDOW)
        cols = [c for c in clean.columns if clean[c].dtype.kind in "fiub"]
        for z in ("s2_z", "s3_z", "s4_z", "regime", "vvix", "pdv_iv_spread"):
            assert z in cols
        pd.testing.assert_frame_equal(clean.loc[:cutoff, cols], pois.loc[:cutoff, cols],
                                      check_exact=True)
        after = clean.index[clean.index > cutoff][:20]
        assert not clean.loc[after, cols].equals(pois.loc[after, cols]), "poison is vacuous"


class TestSignalEngineLags:
    def test_features_are_previous_session_values(self, clean_signals):
        spx, vix, clean = clean_signals
        # SignalEngine's default σ̂ is the walk-forward PDV-21 forecast
        from joint_vol_calibration.signals.regime_classifier import PDV_WALK_FORWARD
        raw = build_features(spx, vix, pdv_vol_series=PDV_WALK_FORWARD)
        prev = raw.shift(1).reindex(clean.index)
        for c in ("vix", "ts_slope", "fear_premium", "vvix", "pdv_iv_spread"):
            pd.testing.assert_series_equal(clean[c], prev[c], check_names=False)

    def test_regime_is_previous_session_label(self, clean_signals):
        spx, vix, clean = clean_signals
        labels = build_regime_labels(spx, vix)
        prev = labels.reindex(pd.to_datetime(spx["date"])).shift(1).ffill().reindex(clean.index)
        assert (clean["regime"] == prev).all()

    def test_precomputed_label_change_moves_only_later_rows(self, clean_signals):
        """Changing the label dated t may only move rows dated > t."""
        spx, vix, _ = clean_signals
        labels = build_regime_labels(spx, vix)
        t = labels.index[500]
        alt = labels.copy()
        alt.loc[t] = 2 if labels.loc[t] != 2 else 1
        a = SignalEngine().generate(spx, vix, *_SIG_WINDOW, precomputed_regimes=labels)
        b = SignalEngine().generate(spx, vix, *_SIG_WINDOW, precomputed_regimes=alt)
        cols = [c for c in a.columns if a[c].dtype.kind in "fiub"]
        pd.testing.assert_frame_equal(a.loc[:t, cols], b.loc[:t, cols], check_exact=True)
        assert a.loc[a.index > t, "regime"].iloc[0] != b.loc[b.index > t, "regime"].iloc[0]


class TestRegimeLabelCausality:
    """A label is a function of observables dated ≤ t: poisoning everything
    strictly AFTER t must not change any label (or gate) dated ≤ t."""

    @pytest.mark.parametrize("thr", [None, 100.0], ids=["adaptive", "fixed100"])
    @pytest.mark.parametrize("pz", POISONS, ids=["hot", "cold"])
    def test_labels_up_to_t_unchanged(self, pz, thr):
        spx, vix = _signal_market()
        clean = build_regime_labels(spx, vix, vvix_threshold=thr)
        for t in clean.index[[300, 450, 600]]:
            sp, vp = _poison(spx, vix, t, inclusive=False, **pz)
            pois = build_regime_labels(sp, vp, vvix_threshold=thr)
            pd.testing.assert_series_equal(clean.loc[:t], pois.loc[:t])

    def test_label_at_t_uses_return_at_t(self):
        """Non-vacuity: the label's realised vol includes r_t (same-day), so
        poisoning t itself must be able to move the label at t."""
        spx, vix = _signal_market()
        clean = build_regime_labels(spx, vix, vvix_threshold=1e9)   # no R2
        moved = 0
        for t in clean.index[[300, 450, 600]]:
            sp, vp = _poison(spx, vix, t, ret=0.25, v=15.0, v3m=16.0, vv=90.0)
            moved += int(build_regime_labels(sp, vp, vvix_threshold=1e9).loc[t] != clean.loc[t])
        assert moved > 0

    def test_adaptive_gate_uses_only_past_vvix(self):
        spx, vix = _signal_market()
        vvix = vix.set_index(pd.to_datetime(vix["date"]))["^VVIX"]
        clean = r2_gate_threshold(vvix)
        for k in (100, 400, 700):
            pois = vvix.copy()
            pois.iloc[k + 1:] = 999.0
            pd.testing.assert_series_equal(clean.iloc[:k + 1], r2_gate_threshold(pois).iloc[:k + 1])


class TestHMMIsFiltered:
    """HMM inference must be forward filtering: P(state_t | x_1..x_t)."""

    def test_truncation_invariance(self):
        rng = np.random.default_rng(42)
        n = 400
        block = n // 3
        vix = np.concatenate([rng.uniform(0.10, 0.18, block), rng.uniform(0.18, 0.28, block),
                              rng.uniform(0.28, 0.45, n - 2 * block)])
        X = pd.DataFrame({"vix": vix, "vvix": vix * rng.uniform(3.5, 5.0, n),
                          "ts_slope": rng.uniform(-0.005, 0.005, n),
                          "fear_premium": rng.uniform(0.95, 1.15, n)},
                         index=pd.bdate_range("2010-01-04", periods=n))
        hmm = HMMRegimeClassifier(n_components=3, n_iter=50, random_state=42).fit(X)
        full_p = hmm.hmm_predict_proba(X)
        full_l = hmm.predict(X)
        for k in (60, 130, 134, 140, 270, 300):
            part = X.iloc[:k]
            np.testing.assert_allclose(hmm.hmm_predict_proba(part).iloc[-1].values,
                                       full_p.iloc[k - 1].values, rtol=0, atol=1e-12)
            assert hmm.predict(part)[-1] == full_l[k - 1]

    def test_labels_do_not_use_later_observations(self):
        """Hand-set model: after a long state-1 spell, four rows sit marginally
        closer to state 0. The filter needs several such rows before it
        switches; a full-sequence decoder (Viterbi) relabels the FIRST of them
        using the rows that follow."""
        from hmmlearn.hmm import GaussianHMM

        means = np.array([[0.20, 0.95, 0.002, 0.85],    # → R0 (fear < 1)
                          [0.13, 0.85, 0.015, 1.45],    # → R1
                          [0.30, 1.35, -0.010, 1.20]])  # → R2 (highest vvix)
        sd = np.array([0.01, 0.03, 0.002, 0.04])
        model = GaussianHMM(n_components=3, covariance_type="full", init_params="", params="")
        model.n_features = 4
        model.startprob_ = np.full(3, 1 / 3)
        model.transmat_ = np.full((3, 3), 0.01) + np.eye(3) * 0.97
        model.means_ = means
        model.covars_ = np.array([np.diag(sd ** 2)] * 3)
        hmm = HMMRegimeClassifier()
        hmm._model, hmm.is_fitted = model, True
        hmm._state_to_regime = hmm.map_states_to_regimes(means)

        ambiguous = 0.505 * means[0] + 0.495 * means[1]
        rows = [means[1]] * 30 + [ambiguous] * 4 + [means[0]] * 30
        X = pd.DataFrame(rows, columns=hmm.HMM_FEATURES,
                         index=pd.bdate_range("2015-01-01", periods=len(rows)))
        full = hmm.predict(X)
        assert list(full[29:32]) == [1, 1, 1], "first ambiguous rows relabelled from the future"
        np.testing.assert_array_equal(full, hmm.hmm_predict_proba(X).values.argmax(axis=1))
        for k in range(28, 40):
            assert hmm.predict(X.iloc[:k])[-1] == full[k - 1]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
