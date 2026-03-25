"""
test_risk_monitor.py — Test suite for C6: Second-Order Greeks Risk Monitor.

Coverage (20 tests across 6 classes):
  1. Analytical BS Greeks — shapes, signs, magnitudes
  2. Vomma sign analysis — positive far-OTM, negative near-ATM
  3. Volga numerical == Vomma analytical (BS cross-check)
  4. QV convexity MC — non-negative, zero σ, monotone in σ and T
  5. Greeks surface — shape, columns, value ranges
  6. Flagging, persistence, RiskMonitor class
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from joint_vol_calibration.greeks.risk_monitor import (
    RiskMonitor,
    _bs_call_price,
    _bs_d1d2,
    _bs_delta,
    _bs_vanna,
    _bs_vega,
    _bs_volga_numerical,
    _bs_vomma,
    compute_greeks_surface,
    flag_unstable_hedges,
    save_greeks_surface,
    simulate_qv_convexity,
)

# ── Shared fixtures ────────────────────────────────────────────────────────────

# Calibrated 2026-03-24 Heston params
_PARAMS = {
    "kappa": 4.62, "theta": 0.0764, "sigma": 0.84, "rho": -0.99, "v0": 0.0561,
}
_S, _r, _q = 6581.0, 0.045, 0.013

# Small grid for fast surface tests
_FAST_MATURITIES = np.array([30, 91]) / 365.0
_FAST_STRIKES    = np.array([5800.0, 6200.0, 6581.0, 7000.0, 7400.0])
_FAST_MC         = dict(mc_n_paths=500, mc_n_steps=30)


def _make_surface() -> pd.DataFrame:
    """Build a small surface for reuse across test classes (fast)."""
    return compute_greeks_surface(
        S=_S, r=_r, q=_q, params=_PARAMS,
        strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES,
        verbose=False, **_FAST_MC,
    )


# ── 1. Analytical BS Greeks ───────────────────────────────────────────────────

class TestAnalyticBsGreeks:
    """Verify correctness of vectorised BS Greek formulas."""

    _S = np.array([100.0, 100.0, 100.0, 100.0])
    _K = np.array([85.0,  100.0, 115.0, 130.0])
    _T = np.array([0.25,  0.25,  0.25,  0.25])
    _r = np.array([0.05,  0.05,  0.05,  0.05])
    _q = np.array([0.02,  0.02,  0.02,  0.02])
    _s = np.array([0.25,  0.25,  0.25,  0.25])

    def test_vega_always_positive(self):
        """BS vega = S·exp(-qT)·N'(d1)·√T > 0 for all valid inputs."""
        v = _bs_vega(self._S, self._K, self._T, self._r, self._q, self._s)
        assert (v > 0).all(), f"Vega must be positive, got: {v}"

    def test_delta_between_0_and_exp_neg_qt(self):
        """Call delta = exp(-qT)·N(d1) ∈ (0, exp(-qT))."""
        d = _bs_delta(self._S, self._K, self._T, self._r, self._q, self._s)
        upper = np.exp(-self._q * self._T)
        assert (d > 0).all(),      f"Delta must be positive: {d}"
        assert (d < upper).all(),  f"Delta must be < exp(-qT): {d} vs {upper}"

    def test_delta_atm_approx_half(self):
        """ATM call delta ≈ 0.5 (with small dividend adjustment)."""
        S_ = np.array([100.0]); K_ = np.array([100.0])
        T_ = np.array([0.25]);  r_ = np.array([0.05]); q_ = np.array([0.0])
        s_ = np.array([0.25])
        d  = _bs_delta(S_, K_, T_, r_, q_, s_)[0]
        assert abs(d - 0.5) < 0.10, f"ATM delta {d:.4f} too far from 0.5"

    def test_itm_delta_greater_than_otm(self):
        """ITM call delta > OTM call delta (call delta monotone decreasing in K)."""
        S_ = np.full(2, 100.0); T_ = np.full(2, 0.25)
        r_ = np.full(2, 0.05);  q_ = np.full(2, 0.02)
        s_ = np.full(2, 0.25)
        K_itm = np.array([80.0]); K_otm = np.array([120.0])
        d_itm = _bs_delta(np.array([100.0]), K_itm, np.array([0.25]),
                          np.array([0.05]), np.array([0.02]), np.array([0.25]))[0]
        d_otm = _bs_delta(np.array([100.0]), K_otm, np.array([0.25]),
                          np.array([0.05]), np.array([0.02]), np.array([0.25]))[0]
        assert d_itm > d_otm, f"ITM delta {d_itm:.4f} must exceed OTM delta {d_otm:.4f}"

    def test_vanna_sign_otm_call(self):
        """OTM call (K > F): d2 < 0 → Vanna = -exp(-qT)·N'(d1)·d2/σ > 0."""
        # K = 120 > F ≈ 100·exp(0.03·0.25) ≈ 100.75
        S_ = np.array([100.0]); K_ = np.array([120.0])
        T_ = np.array([0.25]);  r_ = np.array([0.05]); q_ = np.array([0.02])
        s_ = np.array([0.25])
        van = _bs_vanna(S_, K_, T_, r_, q_, s_)[0]
        assert van > 0, f"Vanna for OTM call must be positive, got {van:.6f}"

    def test_vanna_sign_itm_call(self):
        """Deep ITM call (K << F): d2 > 0 → Vanna < 0."""
        S_ = np.array([100.0]); K_ = np.array([70.0])
        T_ = np.array([0.25]);  r_ = np.array([0.05]); q_ = np.array([0.02])
        s_ = np.array([0.25])
        van = _bs_vanna(S_, K_, T_, r_, q_, s_)[0]
        assert van < 0, f"Vanna for deep ITM call must be negative, got {van:.6f}"


# ── 2. Vomma Sign Analysis ────────────────────────────────────────────────────

class TestVommaSign:
    """Validate the key property: Vomma > 0 for far OTM/ITM calls."""

    def _vomma(self, S, K, T, r, q, sigma):
        return _bs_vomma(
            np.array([S]), np.array([K]), np.array([T]),
            np.array([r]), np.array([q]), np.array([sigma]),
        )[0]

    def test_vomma_positive_deep_otm_call(self):
        """Deep OTM call (K/F ≈ 1.25): d1 < 0, d2 < d1 → d1·d2 > 0 → Vomma > 0."""
        # F ≈ 100, K = 130 (log_m ≈ +0.26)
        v = self._vomma(S=100, K=130, T=0.5, r=0.05, q=0.01, sigma=0.25)
        assert v > 0, f"Deep OTM call Vomma should be positive, got {v:.6f}"

    def test_vomma_positive_deep_itm_call(self):
        """Deep ITM call (K/F ≈ 0.75): d1 >> 0, d2 >> 0 → d1·d2 > 0 → Vomma > 0."""
        v = self._vomma(S=100, K=75, T=0.5, r=0.05, q=0.01, sigma=0.25)
        assert v > 0, f"Deep ITM call Vomma should be positive, got {v:.6f}"

    def test_vomma_negative_near_atm(self):
        """Near-ATM (K=F, short T): d1 > 0, d2 < 0 → d1·d2 < 0 → Vomma < 0.
        This is expected and NOT a bug: ATM options are locally concave in vol."""
        # ATM (K=S, r=q=0): d1=σ√T/2 > 0, d2=-σ√T/2 < 0
        v = self._vomma(S=100, K=100, T=0.10, r=0.0, q=0.0, sigma=0.20)
        assert v < 0, f"ATM call Vomma expected negative near ATM, got {v:.6f}"

    def test_vomma_full_surface_far_otm_positive(self):
        """For the calibrated Heston surface, all far-OTM nodes have Vomma > 0."""
        df = _make_surface()
        far_otm = df[df["log_moneyness"] > 0.10]
        if len(far_otm) > 0:
            assert (far_otm["vomma"] > 0).all(), (
                f"Far-OTM nodes should have Vomma > 0:\n"
                f"{far_otm[['K', 'T', 'log_moneyness', 'vomma']]}"
            )


# ── 3. Volga Numerical ≈ Vomma Analytical ────────────────────────────────────

class TestVolgaVommaConsistency:
    """Numerical Volga (central diff on price) must match analytical Vomma."""

    def _single(self, S, K, T, r, q, sigma):
        S_ = np.array([S]); K_ = np.array([K])
        T_ = np.array([T]); r_ = np.array([r]); q_ = np.array([q])
        s_ = np.array([sigma])
        vomma = _bs_vomma(S_, K_, T_, r_, q_, s_)[0]
        volga = _bs_volga_numerical(S_, K_, T_, r_, q_, s_)[0]
        return vomma, volga

    def test_volga_matches_vomma_atm(self):
        """ATM: numerical Volga ≈ analytical Vomma within 1%."""
        vomma, volga = self._single(100, 100, 0.5, 0.05, 0.02, 0.25)
        rel_err = abs(volga - vomma) / (abs(vomma) + 1e-6)
        assert rel_err < 0.01, (
            f"ATM: Vomma={vomma:.6f}, Volga={volga:.6f}, rel_err={rel_err:.4f}"
        )

    def test_volga_matches_vomma_otm(self):
        """OTM: numerical Volga ≈ analytical Vomma within 1%."""
        vomma, volga = self._single(100, 120, 0.5, 0.05, 0.02, 0.25)
        rel_err = abs(volga - vomma) / (abs(vomma) + 1e-6)
        assert rel_err < 0.01, (
            f"OTM: Vomma={vomma:.6f}, Volga={volga:.6f}, rel_err={rel_err:.4f}"
        )

    def test_volga_matches_vomma_across_surface(self):
        """Surface-wide: |Volga_num - Vomma_analytical| < 5% + 1e-4 for all nodes."""
        df = _make_surface()
        rel_err = np.abs(df["volga"] - df["vomma"]) / (np.abs(df["vomma"]) + 1e-4)
        assert (rel_err < 0.05).all(), (
            f"Max relative error: {rel_err.max():.4f} (threshold 5%)"
        )


# ── 4. QV Convexity (MC) ──────────────────────────────────────────────────────

class TestQvConvexity:
    """MC simulation of CIR variance process."""

    _BASE = dict(kappa=4.62, theta=0.0764, sigma=0.84, v0=0.0561)

    def test_nonnegative(self):
        """Var[QV] ≥ 0 always (variance is a non-negative quantity)."""
        qv = simulate_qv_convexity(**self._BASE, T=0.25, n_paths=1000, n_steps=30)
        assert qv >= 0, f"QV convexity must be non-negative, got {qv}"

    def test_zero_for_zero_sigma(self):
        """When σ=0, variance is deterministic → Var[QV] ≈ 0."""
        qv = simulate_qv_convexity(
            kappa=4.62, theta=0.0764, sigma=0.0, v0=0.0561,
            T=0.5, n_paths=2000, n_steps=50,
        )
        assert qv < 1e-6, f"σ=0 should give Var[QV]≈0, got {qv:.2e}"

    def test_increases_with_sigma(self):
        """Higher vol-of-vol σ → larger Var[QV] (more spread in realised variance)."""
        rng = np.random.default_rng(0)
        qv_lo = simulate_qv_convexity(
            kappa=4.62, theta=0.0764, sigma=0.20, v0=0.0561,
            T=0.5, n_paths=5000, n_steps=50, rng=np.random.default_rng(0),
        )
        qv_hi = simulate_qv_convexity(
            kappa=4.62, theta=0.0764, sigma=1.20, v0=0.0561,
            T=0.5, n_paths=5000, n_steps=50, rng=np.random.default_rng(0),
        )
        assert qv_hi > qv_lo, (
            f"Higher σ should increase QV convexity: σ=0.20→{qv_lo:.4e}, σ=1.20→{qv_hi:.4e}"
        )

    def test_increases_with_T(self):
        """Longer T → more accumulated uncertainty → larger Var[QV]."""
        rng0 = np.random.default_rng(42)
        qv_short = simulate_qv_convexity(**self._BASE, T=0.1,
                                          n_paths=3000, n_steps=50, rng=rng0)
        rng1 = np.random.default_rng(42)
        qv_long  = simulate_qv_convexity(**self._BASE, T=1.0,
                                          n_paths=3000, n_steps=50, rng=rng1)
        assert qv_long > qv_short, (
            f"Longer T should increase QV convexity: T=0.1→{qv_short:.4e}, T=1.0→{qv_long:.4e}"
        )


# ── 5. Greeks Surface ─────────────────────────────────────────────────────────

class TestGreeksSurface:
    """Validate the full surface output."""

    def test_returns_dataframe(self):
        df = _make_surface()
        assert isinstance(df, pd.DataFrame), "Surface must be a DataFrame"
        assert len(df) > 0, "Surface must be non-empty"

    def test_required_columns(self):
        """All downstream (C7) columns must be present."""
        df = _make_surface()
        required = {"K", "T", "T_days", "log_moneyness",
                    "iv", "delta", "vega", "vomma", "volga", "vanna", "qv_convexity"}
        missing = required - set(df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_row_count(self):
        """2 maturities × 5 strikes = 10 rows (all valid IVs)."""
        df = _make_surface()
        assert len(df) <= 10, f"Expected ≤10 rows for 2×5 grid, got {len(df)}"
        assert len(df) >= 6,  f"Expected ≥6 valid rows, got {len(df)}"

    def test_vega_strictly_positive(self):
        """BS vega > 0 everywhere — fundamental sanity check."""
        df = _make_surface()
        assert (df["vega"] > 0).all(), f"All vega must be positive:\n{df['vega']}"

    def test_iv_realistic_range(self):
        """IV ∈ (0.01, 3.0) — catches unit bugs (e.g., IV returned as %)."""
        df = _make_surface()
        assert (df["iv"] > 0.01).all(), f"IV too small:\n{df['iv'].describe()}"
        assert (df["iv"] < 3.0).all(),  f"IV too large:\n{df['iv'].describe()}"

    def test_delta_in_valid_range(self):
        """Call delta ∈ (0, 1) for all nodes."""
        df = _make_surface()
        assert (df["delta"] > 0).all() and (df["delta"] < 1).all(), (
            f"Delta out of (0, 1):\n{df['delta'].describe()}"
        )

    def test_as_of_date_column_added(self):
        """as_of_date column is added when provided."""
        df = compute_greeks_surface(
            S=_S, r=_r, q=_q, params=_PARAMS,
            strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES,
            as_of_date="2026-03-25", verbose=False, **_FAST_MC,
        )
        assert "as_of_date" in df.columns
        assert (df["as_of_date"] == "2026-03-25").all()

    def test_qv_convexity_nonneg(self):
        """QV convexity must be ≥ 0 (variance is non-negative)."""
        df = _make_surface()
        assert (df["qv_convexity"] >= 0).all(), (
            f"QV convexity must be non-negative:\n{df['qv_convexity']}"
        )


# ── 6. Flag Unstable Hedges ───────────────────────────────────────────────────

class TestFlagUnstableHedges:
    """Validate the flagging logic."""

    def test_adds_required_columns(self):
        """flag_unstable_hedges adds 'is_unstable' and 'vomma_zscore'."""
        df = _make_surface()
        flagged = flag_unstable_hedges(df)
        assert "is_unstable"  in flagged.columns
        assert "vomma_zscore" in flagged.columns

    def test_is_unstable_is_bool(self):
        """is_unstable must be a boolean column."""
        df = _make_surface()
        flagged = flag_unstable_hedges(df)
        assert flagged["is_unstable"].dtype == bool, (
            f"is_unstable dtype: {flagged['is_unstable'].dtype}"
        )

    def test_zscore_matches_threshold(self):
        """Flagged rows must have |zscore| > threshold_sigma."""
        df = _make_surface()
        flagged = flag_unstable_hedges(df, threshold_sigma=1.5)
        unstable = flagged[flagged["is_unstable"]]
        assert (unstable["vomma_zscore"].abs() > 1.5).all(), (
            "Flagged rows must all have |zscore| > threshold"
        )

    def test_injected_extreme_value_gets_flagged(self):
        """Manually inject a giant vomma → that node must be flagged."""
        df = _make_surface().copy()
        # Inject extreme vomma 100× the current max
        df.loc[df.index[0], "vomma"] = df["vomma"].abs().max() * 100
        flagged = flag_unstable_hedges(df, threshold_sigma=2.0)
        assert flagged.loc[flagged.index[0], "is_unstable"], (
            "Injected extreme vomma must be flagged as unstable"
        )

    def test_original_df_not_mutated(self):
        """flag_unstable_hedges returns a copy and does not modify the original."""
        df = _make_surface()
        df_original_cols = set(df.columns)
        _ = flag_unstable_hedges(df)
        assert set(df.columns) == df_original_cols, (
            "Original DataFrame must not be mutated"
        )


# ── 7. Persistence ────────────────────────────────────────────────────────────

class TestPersistence:
    """Verify parquet save/load round-trip."""

    def test_save_creates_parquet(self, tmp_path):
        df   = _make_surface()
        path = tmp_path / "test_surface.parquet"
        out  = save_greeks_surface(df, path)
        assert out == path
        assert path.exists(), "save_greeks_surface must create the parquet file"

    def test_load_preserves_row_count(self, tmp_path):
        df   = _make_surface()
        path = tmp_path / "surface.parquet"
        save_greeks_surface(df, path)
        df2  = pd.read_parquet(path)
        assert len(df2) == len(df), (
            f"Row count mismatch after load: {len(df2)} != {len(df)}"
        )

    def test_load_preserves_columns(self, tmp_path):
        df   = _make_surface()
        path = tmp_path / "surface2.parquet"
        save_greeks_surface(df, path)
        df2  = pd.read_parquet(path)
        assert set(df.columns) == set(df2.columns), (
            f"Column mismatch: {set(df.columns) ^ set(df2.columns)}"
        )

    def test_load_values_identical(self, tmp_path):
        df   = _make_surface()
        path = tmp_path / "surface3.parquet"
        save_greeks_surface(df, path)
        df2  = pd.read_parquet(path)
        pd.testing.assert_frame_equal(
            df.reset_index(drop=True),
            df2.reset_index(drop=True),
            check_like=True,
        )


# ── 8. RiskMonitor Class ──────────────────────────────────────────────────────

class TestRiskMonitor:
    """End-to-end tests for the RiskMonitor orchestration class."""

    def _monitor(self):
        m = RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS, as_of_date="2026-03-25")
        m.build(strikes=_FAST_STRIKES, maturities=_FAST_MATURITIES,
                verbose=False, **_FAST_MC)
        return m

    def test_build_returns_dataframe(self):
        m = self._monitor()
        assert isinstance(m.surface, pd.DataFrame)
        assert len(m.surface) > 0

    def test_flag_adds_columns(self):
        m = self._monitor()
        flagged = m.flag()
        assert "is_unstable" in flagged.columns

    def test_validate_all_pass(self):
        m = self._monitor()
        checks = m.validate()
        assert isinstance(checks, dict)
        failures = {k: v for k, v in checks.items() if not v}
        assert not failures, f"Validate() failures: {failures}"

    def test_repr(self):
        m = self._monitor()
        r = repr(m)
        assert "RiskMonitor" in r
        assert "2026-03-25" in r
        assert str(len(m.surface)) in r

    def test_plot_returns_figure(self, tmp_path):
        import matplotlib.pyplot as plt
        m = self._monitor()
        m.flag()
        fig = m.plot(save_path=tmp_path / "heatmap.png")
        assert isinstance(fig, plt.Figure)
        assert (tmp_path / "heatmap.png").exists()
        plt.close(fig)

    def test_save_creates_file(self, tmp_path):
        m = self._monitor()
        path = m.save(path=tmp_path / "surface.parquet")
        assert path.exists()

    def test_build_before_flag_raises(self):
        m = RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS)
        with pytest.raises(RuntimeError):
            m.flag()

    def test_build_before_plot_raises(self):
        m = RiskMonitor(S=_S, r=_r, q=_q, params=_PARAMS)
        with pytest.raises(RuntimeError):
            m.plot()
