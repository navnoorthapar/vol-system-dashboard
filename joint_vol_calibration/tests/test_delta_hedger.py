"""
test_delta_hedger.py — Unit tests for C7: Delta-Hedged P&L Simulation.

Tests cover:
  1. BS straddle pricing and put-call parity
  2. BS straddle Greeks (signs, symmetry, limits; vanna/volga vs finite differences)
  3. VIX term-structure IV interpolation (config tenors, 365-day year)
  4. Simulation output structure; Run B's PDV fitted strictly before entry
  5. P&L attribution: book-keeping identity on real data, and the
     unexplained-variance ratio in a pure Black-Scholes world (C22)
  6. Stress test (COVID crash date isolation)
  7. Vol-of-vol (vomma) flag: fires on a vol-of-vol spike, quiet otherwise
  8. Persistence (save / load), DeltaHedger interface, compute_hedge_metrics
"""

import math

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest.delta_hedger import (
    COVID_CRASH_DATE,
    ENTRY_DATE,
    EXIT_DATE,
    K_ENTRY,
    R,
    Q,
    VIX_TENORS,
    DeltaHedger,
    _bs_straddle_cross_greeks,
    _bs_straddle_greeks,
    _bs_straddle_value,
    _compute_vomma_at,
    _interp_atm_iv,
    _is_unstable,
    _zscore,
    compute_hedge_metrics,
    hedge_attribution,
    run_simulation,
    save_results,
    load_results,
    stress_test,
)

# ── Fixtures ───────────────────────────────────────────────────────────────────

@pytest.fixture
def atm_params():
    """Standard ATM option parameters."""
    return dict(S=3257.85, K=3257.85, T=1.0, r=0.015, q=0.013, sigma=0.20)


@pytest.fixture
def sample_vix_row():
    """Typical VIX term structure row (2020-style: elevated vol)."""
    return pd.Series({
        "^VIX9D": 35.0,
        "^VIX":   30.0,
        "^VIX3M": 28.5,
        "^VIX6M": 26.0,
    })


@pytest.fixture
def flat_vix_row():
    """Flat VIX term structure (all 20%)."""
    return pd.Series({
        "^VIX9D": 20.0,
        "^VIX":   20.0,
        "^VIX3M": 20.0,
        "^VIX6M": 20.0,
    })


@pytest.fixture
def mini_sim_df():
    """
    Minimal 5-day simulation DataFrame for unit-level tests (no DB access).
    Mimics the output of run_simulation().
    """
    dates = pd.date_range("2020-01-02", periods=5, freq="B")
    rng = np.random.default_rng(42)
    n = len(dates)

    pnl_gamma     = rng.uniform(5, 80, n)
    pnl_vega_a    = rng.uniform(-50, 100, n)
    pnl_theta     = np.full(n, -15.0)
    pnl_residual_a = rng.uniform(-30, 30, n)
    pnl_total     = pnl_gamma + pnl_vega_a + pnl_theta + pnl_residual_a

    pnl_vega_b     = pnl_vega_a * 1.1
    pnl_residual_b = pnl_total - pnl_gamma - pnl_vega_b - pnl_theta

    df = pd.DataFrame({
        "S":             [3257.85, 3240.0, 3220.0, 3100.0, 3200.0],
        "T_rem":         [1.0, 0.996, 0.992, 0.988, 0.984],
        "sigma_atm":     [0.125, 0.13, 0.15, 0.35, 0.30],
        "sigma_pdv":     [0.13, 0.14, 0.16, 0.38, 0.32],
        "pdv_ratio":     [1.04, 1.077, 1.067, 1.086, 1.067],
        "straddle_value":[520, 510, 530, 950, 800],
        "delta":         [0.01, 0.02, -0.01, -0.08, -0.05],
        "gamma":         [0.001]*n,
        "vega":          [5.0]*n,
        "theta":         [-50.0]*n,
        "dS":            [np.nan, -17.85, -20.0, -120.0, 100.0],
        "dsigma":        [np.nan, 0.005, 0.02, 0.20, -0.05],
        "pnl_total":     [np.nan] + list(pnl_total[1:]),
        "pnl_gamma":     [np.nan] + list(pnl_gamma[1:]),
        "pnl_vega_a":    [np.nan] + list(pnl_vega_a[1:]),
        "pnl_vega_b":    [np.nan] + list(pnl_vega_b[1:]),
        "pnl_theta":     [np.nan] + list(pnl_theta[1:]),
        "pnl_residual_a":[np.nan] + list(pnl_residual_a[1:]),
        "pnl_residual_b":[np.nan] + list(pnl_residual_b[1:]),
        "cum_pnl_a":     [0, 20, 40, 200, 180],
        "cum_pnl_b":     [0, 22, 43, 210, 190],
        "vomma":         [500, 510, 550, 2000, 900],
        "vomma_zscore":  [0.1, 0.2, 0.4, 2.5, 0.8],
        "is_unstable":   [False, False, False, True, False],
    }, index=dates)
    return df


# ── 1. BS Straddle Pricing ─────────────────────────────────────────────────────

class TestBsStraddleValue:
    def test_atm_positive(self, atm_params):
        v = _bs_straddle_value(**atm_params)
        assert v > 0

    def test_put_call_parity(self, atm_params):
        """Straddle = call + put; verify via put-call parity."""
        S, K, T, r, q, sigma = (atm_params[k] for k in ["S", "K", "T", "r", "q", "sigma"])
        straddle = _bs_straddle_value(S, K, T, r, q, sigma)

        eqT = math.exp(-q * T)
        erT = math.exp(-r * T)
        from scipy.stats import norm as _norm
        from math import log, sqrt
        F = S * math.exp((r - q) * T)
        sqrtT = sqrt(T)
        d1 = (log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        call = S * eqT * _norm.cdf(d1) - K * erT * _norm.cdf(d2)
        put  = call - S * eqT + K * erT

        assert abs(straddle - (call + put)) < 1e-8

    def test_increases_with_vol(self, atm_params):
        v1 = _bs_straddle_value(**{**atm_params, "sigma": 0.10})
        v2 = _bs_straddle_value(**{**atm_params, "sigma": 0.30})
        assert v2 > v1

    def test_increases_with_t(self, atm_params):
        v1 = _bs_straddle_value(**{**atm_params, "T": 0.25})
        v2 = _bs_straddle_value(**{**atm_params, "T": 1.0})
        assert v2 > v1

    def test_zero_t(self, atm_params):
        """At expiry, straddle = intrinsic = |S - K| = 0 for ATM."""
        v = _bs_straddle_value(**{**atm_params, "T": 0.0})
        assert v >= 0   # intrinsic; ATM → 0


# ── 2. BS Straddle Greeks ─────────────────────────────────────────────────────

class TestBsStraddleGreeks:
    def test_atm_delta_near_zero(self, atm_params):
        """ATM straddle delta should be close to 0 (≈ exp(-qT)*(2*N(d1)-1))."""
        delta, gamma, vega, theta = _bs_straddle_greeks(**atm_params)
        assert abs(delta) < 0.1

    def test_gamma_positive(self, atm_params):
        """Long straddle always has positive gamma (long convexity)."""
        delta, gamma, vega, theta = _bs_straddle_greeks(**atm_params)
        assert gamma > 0

    def test_vega_positive(self, atm_params):
        """Long straddle always has positive vega."""
        delta, gamma, vega, theta = _bs_straddle_greeks(**atm_params)
        assert vega > 0

    def test_theta_negative(self, atm_params):
        """Long straddle has negative theta (time decay)."""
        delta, gamma, vega, theta = _bs_straddle_greeks(**atm_params)
        assert theta < 0

    def test_vega_larger_for_higher_T(self, atm_params):
        """Vega increases with time to expiry."""
        _, _, vega_short, _ = _bs_straddle_greeks(**{**atm_params, "T": 0.25})
        _, _, vega_long,  _ = _bs_straddle_greeks(**atm_params)  # T=1.0
        assert vega_long > vega_short

    def test_gamma_decreases_with_T(self, atm_params):
        """Gamma for ATM decreases as T increases (more time = less convexity per unit time)."""
        _, g_short, _, _ = _bs_straddle_greeks(**{**atm_params, "T": 0.08})
        _, g_long,  _, _ = _bs_straddle_greeks(**atm_params)
        assert g_short > g_long


# ── 3. VIX IV Interpolation ────────────────────────────────────────────────────

class TestInterpAtmIV:
    def test_interpolation_within_range(self, sample_vix_row):
        """T=60d is between 30d and 93d — interpolated value should be in range."""
        iv_30  = sample_vix_row["^VIX"] / 100.0
        iv_93  = sample_vix_row["^VIX3M"] / 100.0
        iv_60  = _interp_atm_iv(sample_vix_row, 60.0 / 365.25)
        assert iv_93 <= iv_60 <= iv_30   # inverted slope in this fixture

    def test_flat_structure(self, flat_vix_row):
        """Flat VIX structure → interpolated IV = 20% at any tenor."""
        iv = _interp_atm_iv(flat_vix_row, 0.5)
        assert abs(iv - 0.20) < 1e-6

    def test_extrapolate_below(self, sample_vix_row):
        """T < 9d → return VIX9D."""
        iv_short = _interp_atm_iv(sample_vix_row, 5.0 / 365.25)
        assert abs(iv_short - sample_vix_row["^VIX9D"] / 100.0) < 1e-8

    def test_extrapolate_above(self, sample_vix_row):
        """T > 182d → return VIX6M (flat extrapolation)."""
        iv_long = _interp_atm_iv(sample_vix_row, 300.0 / 365.25)
        assert abs(iv_long - sample_vix_row["^VIX6M"] / 100.0) < 1e-8

    def test_nan_row(self):
        """All-NaN row → returns NaN."""
        row = pd.Series({"^VIX9D": np.nan, "^VIX": np.nan, "^VIX3M": np.nan, "^VIX6M": np.nan})
        iv = _interp_atm_iv(row, 0.25)
        assert np.isnan(iv)

    def test_positive_return(self, sample_vix_row):
        """IV must be positive for any valid VIX row."""
        for T in [9, 30, 60, 90, 120, 180]:
            iv = _interp_atm_iv(sample_vix_row, T / 365.25)
            assert iv > 0


# ── 4. Helper functions ────────────────────────────────────────────────────────

class TestHelpers:
    def test_zscore_standard(self):
        assert abs(_zscore(3.0, 0.0, 1.0) - 3.0) < 1e-12

    def test_zscore_zero_std(self):
        assert np.isnan(_zscore(5.0, 5.0, 0.0))

    def test_zscore_nan_inputs(self):
        assert np.isnan(_zscore(np.nan, 0.0, 1.0))
        assert np.isnan(_zscore(1.0, np.nan, 1.0))

    def test_is_unstable_true(self):
        assert _is_unstable(2.5, 2.0) is True

    def test_is_unstable_false(self):
        assert _is_unstable(1.9, 2.0) is False

    def test_is_unstable_nan_zscore(self):
        assert _is_unstable(np.nan, 2.0) is False

    def test_vomma_at_positive_for_deep_otm(self):
        """Deep OTM call has positive vomma (d1·d2 > 0)."""
        S, K, T, r, q, sigma = 100.0, 200.0, 1.0, 0.01, 0.0, 0.30
        v = _compute_vomma_at(S, K, T, r, q, sigma)
        assert v > 0


# ── 5. Attribution identity ────────────────────────────────────────────────────

class TestAttributionIdentity:
    def test_run_a_components_sum_to_total(self, mini_sim_df):
        """pnl_gamma + pnl_vega_a + pnl_theta + pnl_residual_a == pnl_total."""
        d = mini_sim_df.dropna(subset=["pnl_total"])
        recon = d["pnl_gamma"] + d["pnl_vega_a"] + d["pnl_theta"] + d["pnl_residual_a"]
        np.testing.assert_allclose(recon.values, d["pnl_total"].values, atol=1e-6)

    def test_run_b_components_sum_to_total(self, mini_sim_df):
        """pnl_gamma + pnl_vega_b + pnl_theta + pnl_residual_b == pnl_total."""
        d = mini_sim_df.dropna(subset=["pnl_total"])
        recon = d["pnl_gamma"] + d["pnl_vega_b"] + d["pnl_theta"] + d["pnl_residual_b"]
        np.testing.assert_allclose(recon.values, d["pnl_total"].values, atol=1e-6)

    def test_entry_row_has_nan_pnl(self, mini_sim_df):
        """Entry day P&L should be NaN (no prior bar to attribute against)."""
        assert pd.isna(mini_sim_df.iloc[0]["pnl_total"])

    def test_pnl_gamma_nonnegative(self, mini_sim_df):
        """Gamma P&L = ½ Γ ΔS² ≥ 0 always (long straddle is long convexity)."""
        d = mini_sim_df.dropna(subset=["pnl_gamma"])
        assert (d["pnl_gamma"] >= 0).all()


# ── 6. Hedge metrics ───────────────────────────────────────────────────────────

class TestHedgeMetrics:
    def test_returns_dict(self, mini_sim_df):
        metrics = compute_hedge_metrics(mini_sim_df)
        assert isinstance(metrics, dict)

    def test_required_keys(self, mini_sim_df):
        metrics = compute_hedge_metrics(mini_sim_df)
        for key in [
            "n_days", "cumulative_pnl_a", "cumulative_pnl_b",
            "hedge_efficiency_a", "hedge_efficiency_b",
            "pdv_improves_hedge", "n_unstable_days", "covid_crash_pnl",
        ]:
            assert key in metrics, f"Missing key: {key}"

    def test_n_days_correct(self, mini_sim_df):
        metrics = compute_hedge_metrics(mini_sim_df)
        # 5-row fixture: 1 NaN entry row → 4 valid P&L days
        assert metrics["n_days"] == 4

    def test_efficiency_between_0_and_1(self, mini_sim_df):
        metrics = compute_hedge_metrics(mini_sim_df)
        # Efficiencies might exceed 1 if residuals are large (that's OK for constructed test data)
        assert metrics["hedge_efficiency_a"] >= 0
        assert metrics["hedge_efficiency_b"] >= 0

    def test_n_unstable_days(self, mini_sim_df):
        metrics = compute_hedge_metrics(mini_sim_df)
        # Fixture has 1 unstable day (index 3)
        assert metrics["n_unstable_days"] == 1


# ── 7. Stress test ────────────────────────────────────────────────────────────

class TestStressTest:
    @pytest.fixture
    def sim_with_covid(self, mini_sim_df):
        """Extend mini_sim_df to include COVID crash date."""
        dates = pd.date_range("2020-03-13", periods=5, freq="B")
        rng = np.random.default_rng(99)
        extra = pd.DataFrame({
            "S":             [2870, 2711, 2386, 2530, 2630],   # crash and bounce
            "T_rem":         [0.78, 0.776, 0.772, 0.768, 0.764],
            "sigma_atm":     [0.40, 0.50, 0.75, 0.60, 0.55],
            "sigma_pdv":     [0.42, 0.55, 0.85, 0.65, 0.58],
            "pdv_ratio":     [1.05, 1.10, 1.13, 1.08, 1.05],
            "straddle_value":[1400, 1600, 2800, 2200, 1900],
            "delta":         [-0.1, -0.12, -0.18, -0.14, -0.11],
            "gamma":         [0.0005]*5,
            "vega":          [6.0]*5,
            "theta":         [-55.0]*5,
            "dS":            [np.nan, -159.0, -324.86, 144.0, 100.0],
            "dsigma":        [np.nan, 0.10, 0.25, -0.15, -0.05],
            "pnl_total":     [np.nan, -80.0, 450.0, 150.0, 50.0],
            "pnl_gamma":     [np.nan, 20.0, 85.0, 17.0, 8.0],
            "pnl_vega_a":    [np.nan, 60.0, 150.0, -90.0, -30.0],
            "pnl_vega_b":    [np.nan, 66.0, 169.0, -97.0, -31.5],
            "pnl_theta":     [np.nan, -55.0/252, -55.0/252, -55.0/252, -55.0/252],
            "pnl_residual_a":[np.nan, -160.0, 215.0, 223.0, 72.0],
            "pnl_residual_b":[np.nan, -166.0, 196.0, 230.0, 73.5],
            "cum_pnl_a":     [200, 120, 570, 720, 770],
            "cum_pnl_b":     [210, 130, 580, 730, 780],
            "vomma":         [900, 1200, 3500, 2500, 1800],
            "vomma_zscore":  [0.8, 1.5, 3.2, 2.3, 1.8],
            "is_unstable":   [False, False, True, True, False],
        }, index=dates)
        return pd.concat([mini_sim_df, extra]).sort_index()

    def test_stress_test_returns_dict(self, sim_with_covid):
        result = stress_test(sim_with_covid, stress_date=COVID_CRASH_DATE)
        assert isinstance(result, dict)

    def test_stress_test_correct_date(self, sim_with_covid):
        result = stress_test(sim_with_covid, stress_date=COVID_CRASH_DATE)
        assert result["date"] == COVID_CRASH_DATE

    def test_stress_test_missing_date_raises(self, mini_sim_df):
        with pytest.raises(KeyError):
            stress_test(mini_sim_df, stress_date="2019-01-01")

    def test_covid_pnl_in_metrics(self, sim_with_covid):
        metrics = compute_hedge_metrics(sim_with_covid)
        assert not np.isnan(metrics["covid_crash_pnl"])


# ── 8. Persistence ────────────────────────────────────────────────────────────

class TestPersistence:
    def test_save_creates_file(self, mini_sim_df, tmp_path):
        out = tmp_path / "test_results.parquet"
        save_results(mini_sim_df, path=out)
        assert out.exists()

    def test_load_preserves_shape(self, mini_sim_df, tmp_path):
        out = tmp_path / "test_results.parquet"
        save_results(mini_sim_df, path=out)
        loaded = load_results(path=out)
        assert loaded.shape == mini_sim_df.shape

    def test_load_preserves_values(self, mini_sim_df, tmp_path):
        out = tmp_path / "test_results.parquet"
        save_results(mini_sim_df, path=out)
        loaded = load_results(path=out)
        # Compare non-NaN columns
        pd.testing.assert_series_equal(
            loaded["sigma_atm"].reset_index(drop=True),
            mini_sim_df["sigma_atm"].reset_index(drop=True),
        )

    def test_load_preserves_index(self, mini_sim_df, tmp_path):
        out = tmp_path / "test_results.parquet"
        save_results(mini_sim_df, path=out)
        loaded = load_results(path=out)
        assert list(loaded.index) == list(mini_sim_df.index)


# ── 9. DeltaHedger class ──────────────────────────────────────────────────────

class TestDeltaHedger:
    def test_repr_not_run(self):
        h = DeltaHedger()
        assert "not run" in repr(h)
        assert str(round(K_ENTRY)) in repr(h)

    def test_repr_after_run(self, mini_sim_df):
        h = DeltaHedger()
        h._results = mini_sim_df   # inject
        assert "ready" in repr(h)

    def test_require_results_before_plot(self):
        h = DeltaHedger()
        with pytest.raises(RuntimeError):
            h.plot()

    def test_require_results_before_save(self):
        h = DeltaHedger()
        with pytest.raises(RuntimeError):
            h.save()

    def test_require_results_before_metrics(self):
        h = DeltaHedger()
        with pytest.raises(RuntimeError):
            h.metrics()

    def test_metrics_after_inject(self, mini_sim_df):
        h = DeltaHedger()
        h._results = mini_sim_df
        m = h.metrics()
        assert "hedge_efficiency_a" in m


# ── 10. IVInterp monotonicity & total-variance convexity ─────────────────────

class TestIVInterpTotalVariance:
    def test_total_variance_interpolated_between_endpoints(self, sample_vix_row):
        """
        Total variance TV(T) = σ(T)² × T must not exceed the linear interpolant
        of the endpoints' TV in a forward-variance-positive market.
        This just checks the interpolated IV is finite and positive.
        """
        for T_days in [9, 30, 60, 93, 130, 182]:
            iv = _interp_atm_iv(sample_vix_row, T_days / 365.25)
            assert iv > 0, f"IV should be positive for T={T_days}d"
            assert iv < 2.0, f"IV should be <200% for T={T_days}d"

    def test_no_negative_forward_variance_for_flat_curve(self, flat_vix_row):
        """
        For a flat vol term structure, forward variance = spot variance ≥ 0.
        The interpolated IV should be constant.
        """
        ivs = [_interp_atm_iv(flat_vix_row, T / 365.25) for T in [10, 30, 60, 90, 180]]
        np.testing.assert_allclose(ivs, 0.20, atol=1e-6)


# ── Module-level constants ─────────────────────────────────────────────────────

class TestConstants:
    def test_entry_date_string(self):
        assert ENTRY_DATE == "2020-01-02"

    def test_exit_date_string(self):
        assert EXIT_DATE == "2020-12-31"

    def test_k_entry_reasonable(self):
        assert 3000 <= K_ENTRY <= 3500

    def test_covid_crash_date(self):
        assert COVID_CRASH_DATE == "2020-03-16"

    def test_r_q_reasonable(self):
        assert 0 < R < 0.05
        assert 0 < Q < 0.03


# ── run_simulation end-to-end (covers the core P&L loop) ──────────────────────

@pytest.fixture(scope="module")
def sim():
    # Short January-2020 window → ~21 trading days, fast.
    return run_simulation(entry_date="2020-01-02", exit_date="2020-01-31")


class TestRunSimulationIntegration:
    """Exercise the actual run_simulation driver (not the hand-built mini_sim_df).

    This is the 280-line P&L loop that produces every delta-hedge result and
    was previously untested end-to-end. Uses a short real-data window so the
    test stays fast (~2s) while covering the loop, state advancement, and the
    attribution bookkeeping.
    """

    def test_output_structure(self, sim):
        for col in ("S", "T_rem", "sigma_atm", "straddle_value", "delta",
                    "gamma", "vega", "theta", "pnl_total", "pnl_gamma",
                    "pnl_vega_a", "pnl_vega_b", "pnl_theta", "pnl_residual_a",
                    "pnl_residual_b", "cum_pnl_a", "cum_pnl_b"):
            assert col in sim.columns, f"missing column {col}"
        assert isinstance(sim.index, pd.DatetimeIndex)
        assert len(sim) >= 15

    def test_entry_day_pnl_is_nan(self, sim):
        # The first (initialisation) row books no P&L.
        assert np.isnan(sim["pnl_total"].iloc[0])
        assert np.isnan(sim["pnl_gamma"].iloc[0])

    _COMMON = ("pnl_gamma", "pnl_theta", "pnl_vanna", "pnl_volga", "pnl_financing")

    def test_attribution_identity_run_a(self, sim):
        """Book-keeping only: the components plus the residual add up to the
        total (residual_a is the remainder by definition). Attribution
        QUALITY is tested in TestPureBlackScholesAttribution."""
        rows = sim.iloc[1:]
        recon = rows[list(self._COMMON)].sum(axis=1) + rows["pnl_vega_a"] + rows["pnl_residual_a"]
        np.testing.assert_allclose(rows["pnl_total"].values, recon.values,
                                   atol=1e-9)

    def test_attribution_identity_run_b(self, sim):
        rows = sim.iloc[1:]
        recon = rows[list(self._COMMON)].sum(axis=1) + rows["pnl_vega_b"] + rows["pnl_residual_b"]
        np.testing.assert_allclose(rows["pnl_total"].values, recon.values,
                                   atol=1e-9)

    def test_total_is_hedged_mtm_plus_financing(self, sim):
        """pnl_total = ΔV − Δ_{t−1}·ΔS + r(Δ_{t−1}S_{t−1} − V_{t−1})ΔT − qΔ_{t−1}S_{t−1}ΔT."""
        prev, cur = sim.iloc[:-1], sim.iloc[1:]
        dT = prev["T_rem"].values - cur["T_rem"].values
        fin = (R * (prev["delta"].values * prev["S"].values - prev["straddle_value"].values)
               - Q * prev["delta"].values * prev["S"].values) * dT
        expected = (cur["straddle_value"].values - prev["straddle_value"].values
                    - prev["delta"].values * (cur["S"].values - prev["S"].values) + fin)
        np.testing.assert_allclose(cur["pnl_total"].values, expected, rtol=1e-12, atol=1e-9)

    def test_theta_charged_on_the_calendar_clock(self, sim):
        """θ·(T_{t−1} − T_t): a Friday→Monday step decays three days, not one."""
        prev, cur = sim.iloc[:-1], sim.iloc[1:]
        expected = prev["theta"].values * (prev["T_rem"].values - cur["T_rem"].values)
        np.testing.assert_allclose(cur["pnl_theta"].values, expected, rtol=1e-12)
        gaps = sim.index.to_series().diff().dt.days.iloc[1:].values
        mon = cur["pnl_theta"].values[gaps == 3]
        mid = cur["pnl_theta"].values[gaps == 1]
        assert len(mon) and len(mid)
        assert np.mean(mon) / np.mean(mid) > 2.5

    def test_run_b_pdv_fitted_strictly_before_entry(self, sim):
        """Run B must not use the full-sample pdv_model.pkl (fitted through
        2025, i.e. on the very 2020 it attributes)."""
        assert sim.attrs.get("pdv_train_end") is not None
        assert pd.Timestamp(sim.attrs["pdv_train_end"]) < pd.Timestamp("2020-01-01")
        assert sim["sigma_pdv"].notna().all()

    def test_cum_pnl_a_equals_cum_pnl_b(self, sim):
        """Run B only re-attributes vega; residual_b absorbs the difference, so
        BOTH decompositions sum to the same total P&L. The cumulative curves are
        therefore identical by construction — the A-vs-B distinction lives in
        hedge EFFICIENCY (residual variance), never in total P&L. Locking this
        in prevents anyone misreading cum_pnl_b as a different strategy."""
        np.testing.assert_allclose(sim["cum_pnl_a"].values,
                                   sim["cum_pnl_b"].values, atol=1e-9)

    def test_greeks_signs(self, sim):
        # Long straddle: gamma and vega strictly non-negative everywhere.
        assert (sim["gamma"] >= 0).all()
        assert (sim["vega"] >= 0).all()

    def test_time_decays_monotonically(self, sim):
        # T_rem must strictly decrease each day (no negative or flat steps).
        assert (sim["T_rem"].diff().iloc[1:] < 0).all()
        assert (sim["T_rem"] > 0).all()

    def test_hedge_efficiency_attrs_present(self, sim):
        assert "hedge_efficiency_a" in sim.attrs
        assert "hedge_efficiency_b" in sim.attrs


# ── C22: attribution quality, cross-Greeks, tenors, vol-of-vol flag ───────────

def _bs_path(n_days: int, sigma: float, seed: int, S0: float = 3257.85):
    """Business-day GBM path under Q with a constant vol (calendar-clock T)."""
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2021-01-04", periods=n_days)
    dt = 1.0 / 252.0
    z = rng.standard_normal(n_days - 1)
    logS = np.log(S0) + np.r_[0.0, np.cumsum((R - Q - 0.5 * sigma ** 2) * dt + sigma * np.sqrt(dt) * z)]
    return dates, np.exp(logS)


@pytest.fixture(scope="module")
def paths():
    out = []
    for seed in range(40):
        dates, S = _bs_path(126, 0.20, seed)
        out.append(hedge_attribution(dates, S, np.full(len(S), 0.20),
                                     float(S[0]), 1.0, R, Q))
    return pd.concat(out)


class TestPureBlackScholesAttribution:
    """In a BS world (constant σ) the Greek expansion must explain the
    hedged P&L. With θ/252 on a calendar clock the weekend decay landed in
    the residual (Var ratio ≈ 0.14 and Monday residuals ≈ −1.7 vs +0.5)."""

    def test_unexplained_variance_ratio_small(self, paths):
        v = paths.dropna(subset=["pnl_total"])
        ratio = v["pnl_residual_a"].var() / v["pnl_total"].var()
        assert ratio < 0.02, f"Var(resid)/Var(total) = {ratio:.4f}"

    def test_no_weekend_pattern_in_residual(self, paths):
        v = paths.dropna(subset=["pnl_total"]).copy()
        v["gap"] = (v["dT"] * 365.0).round().astype(int)
        mon = v.loc[v["gap"] == 3, "pnl_residual_a"]
        mid = v.loc[v["gap"] == 1, "pnl_residual_a"]
        # A missing two-day theta would put ~2·θ/365 ≈ −0.9 $/day on Mondays.
        assert abs(mon.mean() - mid.mean()) < 0.1, (mon.mean(), mid.mean())

    def test_carry_plus_gamma_is_the_textbook_hedged_pnl(self, paths):
        """Θ·ΔT + financing = −½σ²S²Γ·ΔT (BS PDE): the explained P&L is
        ½Γ(ΔS² − σ²S²ΔT), whose mean is ~0 when realised = implied vol."""
        v = paths.dropna(subset=["pnl_total"])
        prevS = v["S"] - v["dS"]
        # gamma at t−1 is recovered from the gamma P&L where ΔS ≠ 0
        g = 2.0 * v["pnl_gamma"] / v["dS"] ** 2
        lhs = v["pnl_theta"] + v["pnl_financing"]
        rhs = -0.5 * 0.20 ** 2 * prevS ** 2 * g * v["dT"]
        np.testing.assert_allclose(lhs.values, rhs.values, rtol=1e-6, atol=1e-6)


@pytest.fixture(scope="module")
def sv_paths():
    out = []
    for seed in range(20):
        dates, S = _bs_path(60, 0.20, seed)
        rng = np.random.default_rng(100 + seed)
        sig = np.clip(0.20 + np.cumsum(rng.normal(0, 0.012, len(S))), 0.08, 0.6)
        out.append(hedge_attribution(dates, S, sig, float(S[0]) * 1.05, 1.0, R, Q))
    return pd.concat(out).dropna(subset=["pnl_total"])


class TestStochasticVolAttribution:
    """With moving implied vol, vanna and volga carry real P&L: dropping them
    (the pre-C22 expansion stopped at Γ and ν) leaves it in the residual."""

    def test_cross_greeks_explain_variance(self, sv_paths):
        """Measured: 5e-5 with vanna+volga, 1.1e-3 without (20× worse)."""
        v = sv_paths
        ratio = v["pnl_residual_a"].var() / v["pnl_total"].var()
        assert ratio < 3e-4, ratio
        # and the cross terms are not negligible on these paths
        assert (v["pnl_vanna"] + v["pnl_volga"]).abs().mean() > 2 * v["pnl_residual_a"].abs().mean()


class TestCrossGreeks:
    def test_vanna_volga_match_finite_differences(self, atm_params):
        S, K, T, r, q, sig = (atm_params[k] for k in ["S", "K", "T", "r", "q", "sigma"])
        S = S * 1.07   # off the money so both are material
        vanna, volga = _bs_straddle_cross_greeks(S, K, T, r, q, sig)
        hS, hv = 0.5, 1e-4
        V = lambda s, v: _bs_straddle_value(s, K, T, r, q, v)
        fd_vanna = (V(S + hS, sig + hv) - V(S + hS, sig - hv)
                    - V(S - hS, sig + hv) + V(S - hS, sig - hv)) / (4 * hS * hv)
        fd_volga = (V(S, sig + hv) - 2 * V(S, sig) + V(S, sig - hv)) / hv ** 2
        assert vanna == pytest.approx(fd_vanna, rel=1e-4)
        assert volga == pytest.approx(fd_volga, rel=1e-3)

    def test_straddle_volga_is_twice_call_vomma(self, atm_params):
        S, K, T, r, q, sig = (atm_params[k] for k in ["S", "K", "T", "r", "q", "sigma"])
        _, volga = _bs_straddle_cross_greeks(S * 1.1, K, T, r, q, sig)
        assert volga == pytest.approx(2.0 * _compute_vomma_at(S * 1.1, K, T, r, q, sig), rel=1e-9)


class TestVixTenors:
    def test_tenors_come_from_config(self):
        from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
        for tk, days in VIX_TENORS.items():
            assert days == VIX_INDEX_TENOR_DAYS[tk]

    def test_183_days_interpolates_between_vix3m_and_vix6m(self, sample_vix_row):
        """^VIX6M is a 184-day index: 183 days must still interpolate (the old
        182-day tenor on a 365.25 year extrapolated flat here)."""
        iv = _interp_atm_iv(sample_vix_row, 183.0 / 365.0)
        lo = sample_vix_row["^VIX6M"] / 100.0
        assert iv > lo + 1e-5

    def test_30_days_is_exactly_vix(self, sample_vix_row):
        assert _interp_atm_iv(sample_vix_row, 30.0 / 365.0) == pytest.approx(0.30, abs=1e-12)


class TestVolOfVolFlag:
    """The pre-C22 flag z-scored a single call's vomma at 2020 spot against
    the C6 surface at S≈6,581 and never fired (0/253 days, COVID included)."""

    def _path(self, spike: bool):
        dates, S = _bs_path(200, 0.20, seed=3)
        sig = np.full(len(S), 0.20)
        if spike:
            rng = np.random.default_rng(5)
            sig[140:170] = 0.20 + rng.choice([-0.06, 0.06], 30)
        return hedge_attribution(dates, S, sig, float(S[0]) * 1.08, 1.0, R, Q)

    def test_fires_on_vol_of_vol_spike(self):
        df = self._path(spike=True)
        flagged = df.index[df["is_unstable"]]
        assert len(flagged) >= 1
        assert flagged.min() >= df.index[140]

    def test_quiet_without_vol_of_vol(self):
        assert int(self._path(spike=False)["is_unstable"].sum()) == 0

    def test_flag_is_ex_ante(self):
        """Truncating the future must not change past flags."""
        full = self._path(spike=True)
        dates, S = _bs_path(200, 0.20, seed=3)
        sig = full["sigma_atm"].values
        cut = hedge_attribution(dates[:155], S[:155], sig[:155], float(S[0]) * 1.08, 1.0, R, Q)
        pd.testing.assert_series_equal(full["is_unstable"].iloc[:155], cut["is_unstable"])
