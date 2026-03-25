"""
test_nn_pricer.py — Test suite for C5: Neural Network Acceleration Layer.

Coverage:
  1. HestonNet architecture (shape, positivity, input dims)
  2. _bs_iv_vectorized (round-trip, NaN on bad inputs, valid range)
  3. _sample_heston_params (bounds, ±30% constraint, rho sign)
  4. NNPricer.load() (loads from disk, repr, missing-path error)
  5. NNPricer.spx_iv / spx_iv_batch (shape, positivity, monotonicity)
  6. NNPricer.vix_prices (shape, positivity, mean reversion direction)
  7. NNPricer.benchmark (keys, MAE target, speedup)
  8. Training data cache loading (schema, filter bounds)
"""

import math
import numpy as np
import pytest
import torch
import torch.nn as nn
from pathlib import Path
from scipy.special import ndtr  # standard normal CDF (no circular import)

from joint_vol_calibration.config import HESTON_BOUNDS, RANDOM_SEED
from joint_vol_calibration.models.nn_pricer import (
    HestonNet,
    NNPricer,
    _bs_iv_vectorized,
    _load_baseline_params,
    _sample_heston_params,
    generate_spx_training_data,
    generate_vix_training_data,
    _SPX_MODEL_PATH,
    _VIX_MODEL_PATH,
    _SPX_DATA_PATH,
    _VIX_DATA_PATH,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _bs_call_price(F: float, K: float, T: float, sigma: float) -> float:
    """Undiscounted BS call in forward measure — used to construct test inputs."""
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    from scipy.stats import norm
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def _make_pricer() -> NNPricer:
    """Load the NNPricer; skip tests if model files are missing."""
    if not _SPX_MODEL_PATH.exists() or not _VIX_MODEL_PATH.exists():
        pytest.skip("Trained model files not found — run train_spx()/train_vix() first")
    return NNPricer.load(retrain_if_missing=False)


# ── 1. HestonNet Architecture ─────────────────────────────────────────────────

class TestHestonNetArchitecture:
    """Verify network topology, output shape, and strict-positivity guarantee."""

    def test_output_shape_single(self):
        """forward(x) on a single sample returns scalar tensor, not (1, 1)."""
        net = HestonNet(input_dim=7)
        x = torch.randn(1, 7)
        out = net(x)
        assert out.shape == (1,), f"Expected shape (1,), got {out.shape}"

    def test_output_shape_batch(self):
        """forward(x) on N samples returns shape (N,)."""
        net = HestonNet(input_dim=5)
        x = torch.randn(64, 5)
        out = net(x)
        assert out.shape == (64,), f"Expected shape (64,), got {out.shape}"

    def test_softplus_enforces_positivity(self):
        """All outputs are strictly positive even for extreme negative inputs."""
        net = HestonNet(input_dim=7)
        with torch.no_grad():
            x = torch.full((100, 7), -100.0)
            out = net(x)
        assert (out > 0).all(), "Softplus output must be strictly positive"

    def test_spx_architecture_7d_input(self):
        """SPX network accepts 7-dimensional input without errors."""
        net = HestonNet(input_dim=7, hidden_dim=256, n_layers=4)
        x = torch.randn(32, 7)
        out = net(x)
        assert out.shape == (32,)

    def test_vix_architecture_5d_input(self):
        """VIX network accepts 5-dimensional input without errors."""
        net = HestonNet(input_dim=5, hidden_dim=64, n_layers=4)
        x = torch.randn(32, 5)
        out = net(x)
        assert out.shape == (32,)

    def test_parameter_count_spx(self):
        """SPX network has expected parameter count: 4 × (in×256 + 256) + (256×1 + 1)."""
        net = HestonNet(input_dim=7, hidden_dim=256, n_layers=4)
        n_params = sum(p.numel() for p in net.parameters())
        # Layer 1: 7*256 + 256 = 2048; Layers 2-4: 256*256+256 each; Out: 256+1
        expected = (7*256 + 256) + 3*(256*256 + 256) + (256*1 + 1)
        assert n_params == expected, f"Expected {expected} params, got {n_params}"


# ── 2. _bs_iv_vectorized ───────────────────────────────────────────────────────

class TestBsIvVectorized:
    """Validate the Newton-Raphson Black-Scholes IV inversion."""

    def _make_inputs(self, n: int = 20, sigma_range=(0.05, 1.5)):
        """Return (call_prices, F, K, T, r, true_sigmas) for round-trip tests."""
        rng = np.random.default_rng(0)
        S = 6000.0; r = 0.045; q = 0.013
        T_vals = rng.uniform(0.05, 1.0, n)
        log_m  = rng.uniform(-0.25, 0.20, n)
        true_sigmas = rng.uniform(*sigma_range, n)

        F_vals = S * np.exp((r - q) * T_vals)
        K_vals = F_vals * np.exp(log_m)

        # Undiscounted forward call price
        sqrt_T = np.sqrt(T_vals)
        d1 = (np.log(F_vals / K_vals) + 0.5 * true_sigmas**2 * T_vals) / (true_sigmas * sqrt_T)
        d2 = d1 - true_sigmas * sqrt_T
        from scipy.stats import norm
        C_fwd = F_vals * norm.cdf(d1) - K_vals * norm.cdf(d2)
        C_disc = C_fwd * np.exp(-r * T_vals)   # discounted (as returned by heston_call_batch)

        return C_disc, F_vals, K_vals, T_vals, np.full(n, r), true_sigmas

    def test_roundtrip_atm_accuracy(self):
        """IV inversion of BS prices recovers true sigma to < 0.1% error for ATM."""
        C, F, K, T, r, true_sigma = self._make_inputs(n=50, sigma_range=(0.10, 0.60))
        # Make all ATM
        K_atm = F.copy()
        sqrt_T = np.sqrt(T)
        from scipy.stats import norm
        C_atm = F * norm.cdf(0.5 * true_sigma * sqrt_T) - K_atm * norm.cdf(-0.5 * true_sigma * sqrt_T)
        C_disc_atm = C_atm * np.exp(-r * T)
        recovered = _bs_iv_vectorized(C_disc_atm, F, K_atm, T, r)
        valid = ~np.isnan(recovered)
        assert valid.sum() >= 45, f"Too many NaNs: {(~valid).sum()} out of 50"
        err = np.abs(recovered[valid] - true_sigma[valid])
        assert err.max() < 0.001, f"Max IV error {err.max():.4f} > 0.001"

    def test_roundtrip_skew_range(self):
        """IV inversion recovers true sigma across full skew range (OTM puts/calls).
        95% of valid samples must have error < 0.01 vol (Newton-Raphson may struggle
        for extreme deep-OTM options with σ > 1.0 and short T in ≤50 iterations)."""
        C, F, K, T, r, true_sigma = self._make_inputs(n=100, sigma_range=(0.05, 1.20))
        recovered = _bs_iv_vectorized(C, F, K, T, r)
        valid = ~np.isnan(recovered)
        assert valid.sum() >= 80, f"Too many NaNs: {(~valid).sum()} / 100"
        err = np.abs(recovered[valid] - true_sigma[valid])
        pct_accurate = (err < 0.01).mean()
        assert pct_accurate >= 0.95, (
            f"Only {pct_accurate:.0%} of IVs within 0.01 tolerance (max err={err.max():.4f})"
        )

    def test_nan_for_negative_prices(self):
        """Prices below intrinsic value (arbitrage) → NaN, not garbage numbers."""
        F = np.array([100.0, 100.0, 100.0])
        K = np.array([90.0,  100.0, 110.0])
        T = np.array([0.25,  0.25,  0.25])
        r = np.array([0.05,  0.05,  0.05])
        # Set prices well below intrinsic
        bad_prices = np.array([-1.0, -0.5, -1.0])
        result = _bs_iv_vectorized(bad_prices, F, K, T, r)
        assert np.all(np.isnan(result)), "Negative prices should produce NaN"

    def test_nan_for_zero_time(self):
        """T ≈ 0 → NaN (IV undefined at expiry)."""
        F = np.array([100.0])
        K = np.array([100.0])
        T = np.array([0.0])
        r = np.array([0.0])
        call = np.array([1.0])
        result = _bs_iv_vectorized(call, F, K, T, r)
        assert np.isnan(result[0]), "Zero time should produce NaN"

    def test_output_clipped_to_valid_range(self):
        """Recovered IVs are clipped to [1e-4, 5.0] — no wild outliers."""
        C, F, K, T, r, _ = self._make_inputs(n=50, sigma_range=(0.15, 0.60))
        recovered = _bs_iv_vectorized(C, F, K, T, r)
        valid = recovered[~np.isnan(recovered)]
        assert (valid >= 1e-5).all(), "IVs should not be zero or negative"
        assert (valid <= 5.0).all(), "IVs should not exceed 5.0"


# ── 3. _sample_heston_params ──────────────────────────────────────────────────

class TestSampleHestonParams:
    """Validate parameter sampling around the calibrated centre."""

    CENTER = {
        "kappa": 4.62, "theta": 0.0764, "sigma": 0.8407, "rho": -0.99, "v0": 0.0561
    }

    def test_output_shape(self):
        """Returns dict of arrays each of length n."""
        n = 1000
        samples = _sample_heston_params(n, self.CENTER, perturb=0.30)
        for name in ["kappa", "theta", "sigma", "rho", "v0"]:
            assert name in samples, f"Missing key '{name}'"
            assert len(samples[name]) == n, f"{name}: expected {n} samples"

    def test_all_within_heston_bounds(self):
        """No sample should violate the hard parameter bounds."""
        samples = _sample_heston_params(5000, self.CENTER, perturb=0.30)
        for name, arr in samples.items():
            lo, hi = HESTON_BOUNDS[name]
            assert (arr >= lo).all(), f"{name}: sample below bound {lo}"
            assert (arr <= hi).all(), f"{name}: sample above bound {hi}"

    def test_within_30_pct_of_center(self):
        """Positive-valued params are within ±30% of centre (before bound clipping)."""
        samples = _sample_heston_params(5000, self.CENTER, perturb=0.30)
        for name in ["kappa", "theta", "sigma", "v0"]:
            p0 = self.CENTER[name]
            lo = max(p0 * 0.70, HESTON_BOUNDS[name][0])
            hi = min(p0 * 1.30, HESTON_BOUNDS[name][1])
            arr = samples[name]
            assert (arr >= lo - 1e-9).all(), f"{name}: below ±30% lower bound {lo:.4f}"
            assert (arr <= hi + 1e-9).all(), f"{name}: above ±30% upper bound {hi:.4f}"

    def test_rho_always_negative(self):
        """rho must remain in (-0.99, 0) — never positive."""
        samples = _sample_heston_params(2000, self.CENTER, perturb=0.30)
        assert (samples["rho"] < 0).all(), "rho should always be negative"
        assert (samples["rho"] >= -0.99 - 1e-9).all(), "rho should not exceed lower bound -0.99"

    def test_reproducible_with_rng(self):
        """Same RNG seed produces identical samples."""
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        s1 = _sample_heston_params(100, self.CENTER, perturb=0.30, rng=rng1)
        s2 = _sample_heston_params(100, self.CENTER, perturb=0.30, rng=rng2)
        for name in ["kappa", "theta", "sigma", "rho", "v0"]:
            np.testing.assert_array_equal(s1[name], s2[name])


# ── 4. NNPricer Load ──────────────────────────────────────────────────────────

class TestNNPricerLoad:
    """Test model loading from disk."""

    def test_load_returns_nnpricer(self):
        pricer = _make_pricer()
        assert isinstance(pricer, NNPricer)

    def test_repr_shows_loaded(self):
        pricer = _make_pricer()
        r = repr(pricer)
        assert "loaded" in r.lower(), f"repr should say 'loaded': {r}"
        assert "NN-1" in r and "NN-2" in r

    def test_missing_path_raises(self):
        """Loading from a nonexistent path without retrain should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError):
            NNPricer.load(
                spx_path="/tmp/nonexistent_spx_99999.pt",
                vix_path="/tmp/nonexistent_vix_99999.pt",
                retrain_if_missing=False,
            )

    def test_spx_model_in_eval_mode(self):
        """Loaded SPX model must be in eval mode (no dropout randomness)."""
        pricer = _make_pricer()
        assert not pricer._spx_model.training, "SPX model should be in eval() mode"

    def test_vix_model_in_eval_mode(self):
        """Loaded VIX model must be in eval mode."""
        pricer = _make_pricer()
        assert not pricer._vix_model.training, "VIX model should be in eval() mode"


# ── 5. NNPricer SPX Inference ─────────────────────────────────────────────────

class TestNNPricerSpxInference:
    """Test SPX implied vol predictions."""

    # Calibrated params (2026-03-24 baseline)
    PARAMS = dict(kappa=4.62, theta=0.0764, sigma=0.8407, rho=-0.99, v0=0.0561)

    def test_spx_iv_returns_float(self):
        pricer = _make_pricer()
        iv = pricer.spx_iv(log_moneyness=-0.05, T=0.25, **self.PARAMS)
        assert isinstance(iv, float), f"Expected float, got {type(iv)}"

    def test_spx_iv_strictly_positive(self):
        """Softplus guarantees positive output — even for extreme inputs."""
        pricer = _make_pricer()
        iv = pricer.spx_iv(log_moneyness=-0.05, T=0.25, **self.PARAMS)
        assert iv > 0.0, f"IV must be positive, got {iv}"

    def test_spx_iv_reasonable_atm_range(self):
        """ATM 3-month SPX IV for calibrated params should be in (0.10, 0.80)."""
        pricer = _make_pricer()
        iv = pricer.spx_iv(log_moneyness=0.0, T=0.25, **self.PARAMS)
        assert 0.05 < iv < 1.0, f"ATM IV {iv:.4f} outside reasonable range"

    def test_spx_iv_batch_shape(self):
        """spx_iv_batch returns shape (N,) for N input rows."""
        pricer = _make_pricer()
        n = 50
        feats = np.column_stack([
            np.linspace(-0.20, 0.15, n),   # log-moneyness
            np.full(n, np.sqrt(0.25)),       # sqrt_T
            np.full(n, self.PARAMS["kappa"]),
            np.full(n, self.PARAMS["theta"]),
            np.full(n, self.PARAMS["sigma"]),
            np.full(n, self.PARAMS["rho"]),
            np.full(n, self.PARAMS["v0"]),
        ]).astype(np.float32)
        ivs = pricer.spx_iv_batch(feats)
        assert ivs.shape == (n,), f"Expected shape ({n},), got {ivs.shape}"

    def test_spx_iv_batch_all_positive(self):
        """All batch predictions are positive (Softplus guarantee)."""
        pricer = _make_pricer()
        n = 100
        feats = np.column_stack([
            np.linspace(-0.30, 0.20, n),
            np.full(n, np.sqrt(0.25)),
            np.full(n, self.PARAMS["kappa"]),
            np.full(n, self.PARAMS["theta"]),
            np.full(n, self.PARAMS["sigma"]),
            np.full(n, self.PARAMS["rho"]),
            np.full(n, self.PARAMS["v0"]),
        ]).astype(np.float32)
        ivs = pricer.spx_iv_batch(feats)
        assert (ivs > 0).all(), "All batch IVs must be positive"

    def test_spx_iv_negative_skew(self):
        """With rho=-0.99, deep OTM puts (x << 0) have higher IV than OTM calls (x >> 0).
        This is the fundamental SPX skew property — if this fails, the model learned
        the wrong region of parameter space."""
        pricer = _make_pricer()
        n = 20
        feats_put = np.column_stack([
            np.full(n, -0.20),             # deep OTM put log-moneyness
            np.full(n, np.sqrt(0.25)),
            np.full(n, self.PARAMS["kappa"]),
            np.full(n, self.PARAMS["theta"]),
            np.full(n, self.PARAMS["sigma"]),
            np.full(n, self.PARAMS["rho"]),
            np.full(n, self.PARAMS["v0"]),
        ]).astype(np.float32)
        feats_call = feats_put.copy()
        feats_call[:, 0] = 0.20            # deep OTM call log-moneyness
        iv_put  = pricer.spx_iv_batch(feats_put).mean()
        iv_call = pricer.spx_iv_batch(feats_call).mean()
        assert iv_put > iv_call, (
            f"SPX negative skew expected: IV(put)={iv_put:.4f} should > IV(call)={iv_call:.4f}"
        )


# ── 6. NNPricer VIX Inference ─────────────────────────────────────────────────

class TestNNPricerVixInference:
    """Test VIX futures price predictions."""

    PARAMS = dict(kappa=4.62, theta=0.0764, sigma=0.8407, v0=0.0561)

    def test_vix_prices_shape(self):
        """vix_prices returns array of correct length."""
        pricer = _make_pricer()
        tenors = np.array([1/12, 3/12, 6/12, 9/12, 12/12])
        prices = pricer.vix_prices(**self.PARAMS, tenors=tenors)
        assert prices.shape == (len(tenors),), f"Expected shape ({len(tenors)},), got {prices.shape}"

    def test_vix_prices_all_positive(self):
        """VIX futures prices must always be positive."""
        pricer = _make_pricer()
        tenors = np.linspace(0.05, 1.5, 20)
        prices = pricer.vix_prices(**self.PARAMS, tenors=tenors)
        assert (prices > 0).all(), f"VIX prices must be positive, got {prices.min():.4f} min"

    def test_vix_prices_realistic_range(self):
        """VIX futures in (5, 100) VIX points for calibrated params."""
        pricer = _make_pricer()
        tenors = np.array([1/12, 3/12, 6/12, 12/12])
        prices = pricer.vix_prices(**self.PARAMS, tenors=tenors)
        assert (prices > 5).all(),  f"VIX price too low: {prices.min():.2f}"
        assert (prices < 100).all(), f"VIX price too high: {prices.max():.2f}"

    def test_vix_mean_reversion_direction(self):
        """With v0 < theta (backwardation → contango), longer tenors → higher VIX price.
        Calibrated: v0=0.0561, theta=0.0764, so v0 < theta → VIX curve in contango."""
        pricer = _make_pricer()
        assert self.PARAMS["v0"] < self.PARAMS["theta"], "Test requires v0 < theta"
        short_tenor = np.array([0.05])
        long_tenor  = np.array([1.0])
        p_short = pricer.vix_prices(**self.PARAMS, tenors=short_tenor)[0]
        p_long  = pricer.vix_prices(**self.PARAMS, tenors=long_tenor)[0]
        assert p_long > p_short, (
            f"v0<theta → contango expected: short={p_short:.2f}, long={p_long:.2f}"
        )


# ── 7. NNPricer Benchmark ─────────────────────────────────────────────────────

class TestNNPricerBenchmark:
    """Verify benchmark meets production targets."""

    def test_benchmark_keys_present(self):
        """All expected keys appear in benchmark result."""
        pricer = _make_pricer()
        result = pricer.benchmark(n_options=100)
        expected_keys = {"n_options", "heston_ms", "nn_ms", "speedup",
                         "mae_vol_pts", "rmse_vol_pts", "target_passed"}
        assert expected_keys <= result.keys(), (
            f"Missing keys: {expected_keys - result.keys()}"
        )

    def test_benchmark_target_passed(self):
        """NN-1 must achieve MAE < 0.1 vol point on the benchmark set."""
        pricer = _make_pricer()
        result = pricer.benchmark(n_options=500)
        assert result["target_passed"], (
            f"MAE {result['mae_vol_pts']:.4f} vol pts exceeds 0.1 vp target"
        )

    def test_benchmark_speedup_positive(self):
        """NN must be faster than Heston — speedup > 1×."""
        pricer = _make_pricer()
        result = pricer.benchmark(n_options=500)
        assert result["speedup"] > 1.0, (
            f"NN is slower than Heston! speedup={result['speedup']:.2f}×"
        )

    def test_benchmark_mae_numerically_reasonable(self):
        """MAE in (0, 5.0) vol points — catches unit/scaling bugs."""
        pricer = _make_pricer()
        result = pricer.benchmark(n_options=200)
        assert 0 < result["mae_vol_pts"] < 5.0, (
            f"MAE {result['mae_vol_pts']:.3f} outside (0, 5.0) — scaling bug?"
        )


# ── 8. Training Data Cache ────────────────────────────────────────────────────

class TestTrainingDataCache:
    """Verify cached training data has correct schema and filter properties."""

    def test_spx_data_loads_from_cache(self):
        """SPX training data loads from disk (not regenerated) with correct schema."""
        if not _SPX_DATA_PATH.exists():
            pytest.skip("SPX training data cache not found — run generate_spx_training_data() first")
        df = generate_spx_training_data(force_regen=False)
        expected_cols = {"x", "sqrt_T", "kappa", "theta", "sigma", "rho", "v0", "iv_target"}
        assert expected_cols <= set(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

    def test_spx_iv_filter_bounds(self):
        """All iv_target values are in [0.02, 3.0] — IV filter was applied."""
        if not _SPX_DATA_PATH.exists():
            pytest.skip("SPX training data cache not found")
        df = generate_spx_training_data(force_regen=False)
        assert df["iv_target"].notna().all(), "No NaN allowed in iv_target after filter"
        assert (df["iv_target"] >= 0.02).all(), "iv_target must be >= 0.02 (training filter)"
        assert (df["iv_target"] <= 3.0).all(),  "iv_target must be <= 3.0 (sanity bound)"

    def test_spx_data_size(self):
        """SPX training set has at least 200K samples (expected ~424K)."""
        if not _SPX_DATA_PATH.exists():
            pytest.skip("SPX training data cache not found")
        df = generate_spx_training_data(force_regen=False)
        assert len(df) >= 200_000, f"Expected >= 200K samples, got {len(df)}"

    def test_vix_data_loads_from_cache(self):
        """VIX training data loads from disk with correct schema."""
        if not _VIX_DATA_PATH.exists():
            pytest.skip("VIX training data cache not found — run generate_vix_training_data() first")
        df = generate_vix_training_data(force_regen=False)
        expected_cols = {"kappa", "theta", "sigma", "v0", "tenor_years", "vix_target"}
        assert expected_cols <= set(df.columns), f"Missing columns: {expected_cols - set(df.columns)}"

    def test_vix_target_filter_bounds(self):
        """All vix_target values are in (2, 100) VIX points."""
        if not _VIX_DATA_PATH.exists():
            pytest.skip("VIX training data cache not found")
        df = generate_vix_training_data(force_regen=False)
        assert (df["vix_target"] > 2.0).all(),  "vix_target must be > 2.0 (sanity filter)"
        assert (df["vix_target"] < 100.0).all(), "vix_target must be < 100.0 (sanity filter)"

    def test_vix_data_size(self):
        """VIX training set has at least 400K samples (expected 500K)."""
        if not _VIX_DATA_PATH.exists():
            pytest.skip("VIX training data cache not found")
        df = generate_vix_training_data(force_regen=False)
        assert len(df) >= 400_000, f"Expected >= 400K samples, got {len(df)}"


# ── 9. Load Baseline Params ───────────────────────────────────────────────────

class TestLoadBaselineParams:
    """Verify baseline parameter loading."""

    def test_returns_dict_with_all_params(self):
        """_load_baseline_params returns dict with all 5 Heston params."""
        params = _load_baseline_params()
        expected_keys = {"kappa", "theta", "sigma", "rho", "v0"}
        assert expected_keys <= params.keys(), f"Missing params: {expected_keys - params.keys()}"

    def test_params_within_heston_bounds(self):
        """Loaded params are within HESTON_BOUNDS."""
        params = _load_baseline_params()
        for name, val in params.items():
            lo, hi = HESTON_BOUNDS[name]
            assert lo <= val <= hi, f"{name}={val} outside bounds [{lo}, {hi}]"

    def test_rho_negative(self):
        """rho from baseline should be negative (SPX always has negative skew)."""
        params = _load_baseline_params()
        assert params["rho"] < 0, f"Expected rho < 0, got {params['rho']}"
