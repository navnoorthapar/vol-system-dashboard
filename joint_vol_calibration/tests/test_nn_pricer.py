"""
test_nn_pricer.py — Test suite for C5: neural-network surrogate (research prototype).

Coverage:
  1. HestonNet / SPXImpliedVolNet architecture
  2. _bs_iv_vectorized re-export (strict round trips; the solver itself is
     covered in test_implied_vol.py)
  3. _sample_heston_params (legacy ±perturb sampler)
  4. Training domain: archived fits inside, out-of-domain detection
  5. Exact labels and training-data generation (small, in tmp_path)
  6. Training: runs on the installed torch, seeded and deterministic, split
     by parameter set, metrics and targets in the right units, checkpoint
     loads with weights_only=True
  7. NNPricer on a tiny freshly trained model: load, domain guard, benchmark
     scoring every well-posed option (never filtered on NN output)
  8. NNPricer on the shipped weights (skipped when data_store/nn_*_model.pt
     are absent — they are not tracked by git)
  9. Cached training data (skipped when absent) and baseline params
"""

import math
import sys
import warnings

import numpy as np
import pandas as pd
import pytest
import torch
from scipy import integrate

from joint_vol_calibration.config import HESTON_BOUNDS, NN_VOL_ERROR_TARGET
from joint_vol_calibration.models import implied_vol
from joint_vol_calibration.models import nn_pricer as nnp
from joint_vol_calibration.models.heston import characteristic_function
from joint_vol_calibration.models.nn_pricer import (
    NN_LABEL_MIN_OTM_PRICE_REL,
    SPX_FEATURES,
    SPX_TRAINING_DOMAIN,
    VIX_FEATURES,
    VIX_TRAINING_DOMAIN,
    HestonNet,
    NNPricer,
    SPXImpliedVolNet,
    _SPX_DATA_PATH,
    _SPX_MODEL_PATH,
    _VIX_DATA_PATH,
    _bs_iv_vectorized,
    _load_baseline_params,
    _sample_domain_params,
    _sample_heston_params,
    _split_by_group,
    generate_spx_training_data,
    generate_vix_training_data,
    heston_iv_exact,
    spx_out_of_domain,
    train_network,
    train_spx,
)

# ── Helpers ────────────────────────────────────────────────────────────────────

def _bs_call_price(F: float, K: float, T: float, sigma: float) -> float:
    """Undiscounted BS call in forward measure — used to construct test inputs."""
    from scipy.stats import norm
    sqrt_T = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T
    return float(F * norm.cdf(d1) - K * norm.cdf(d2))


def _features(x, T, p):
    x = np.atleast_1d(np.asarray(x, float))
    return np.column_stack([x, np.full(len(x), np.sqrt(T))]
                           + [np.full(len(x), p[k]) for k in ("kappa", "theta", "sigma", "rho", "v0")])


def _ref_iv(x, T, p):
    """Independent reference: Lewis by scipy.quad + scipy brentq on BS."""
    from scipy import optimize
    from scipy.stats import norm

    def integrand(u):
        cf = characteristic_function(np.array([u - 0.5j]), 1.0, T, 0.0, 0.0, **p)[0]
        return np.real(np.exp(-1j * u * x) * cf) / (u * u + 0.25)

    val, _ = integrate.quad(integrand, 0, np.inf, limit=2000, epsabs=1e-15, epsrel=1e-13)
    c = 1.0 - np.exp(x / 2) / np.pi * val
    otm = c if x >= 0 else c - (1.0 - np.exp(x))

    def f(s):
        d1 = (-x + 0.5 * s * s * T) / (s * np.sqrt(T))
        d2 = d1 - s * np.sqrt(T)
        model = (norm.cdf(d1) - np.exp(x) * norm.cdf(d2) if x >= 0
                 else np.exp(x) * norm.cdf(-d2) - norm.cdf(-d1))
        return model - otm
    if otm < 1e-12:                       # numerically zero: no implied vol
        return np.nan, otm
    return optimize.brentq(f, 1e-4, 5.0, xtol=1e-14), otm


_BASE = dict(kappa=4.62, theta=0.0764, sigma=0.8407, rho=-0.99, v0=0.0561)

# Extremes of the archived joint_cal_*.pkl fits (2026-03-24 .. 2026-10-09)
_ARCHIVED_EXTREMES = [
    dict(kappa=4.6213, theta=0.0764, sigma=0.8407, rho=-0.99, v0=0.0561),   # 03-24 (legacy)
    dict(kappa=1.7284, theta=0.0654, sigma=0.4754, rho=-0.95, v0=0.0185),   # 06-16 (min kappa)
    dict(kappa=12.8895, theta=0.0269, sigma=0.8332, rho=-0.95, v0=1e-4),    # 08-25 (max kappa)
    dict(kappa=12.0888, theta=0.0218, sigma=0.7257, rho=-0.95, v0=1e-4),    # 08-28 (min theta)
    dict(kappa=3.3002, theta=0.0471, sigma=0.2360, rho=-0.95, v0=0.0092),   # 10-02 (min sigma)
    dict(kappa=4.1442, theta=0.0516, sigma=0.6540, rho=-0.727, v0=0.022),   # 07-20 (max rho)
]


# ── 1. Architecture ───────────────────────────────────────────────────────────

class TestHestonNetArchitecture:

    def test_output_shape_single(self):
        assert HestonNet(input_dim=7)(torch.randn(1, 7)).shape == (1,)

    def test_output_shape_batch(self):
        assert HestonNet(input_dim=5)(torch.randn(64, 5)).shape == (64,)

    def test_softplus_enforces_positivity(self):
        with torch.no_grad():
            out = HestonNet(input_dim=7)(torch.full((100, 7), -100.0))
        assert (out > 0).all()

    def test_parameter_count_spx(self):
        net = HestonNet(input_dim=7, hidden_dim=256, n_layers=4)
        expected = (7 * 256 + 256) + 3 * (256 * 256 + 256) + (256 * 1 + 1)
        assert sum(p.numel() for p in net.parameters()) == expected

    def test_spx_net_scales_with_reference_vol(self):
        """IV = σ_ref·MLP(...): positive, shape (N,), and σ_ref is the √ of the
        expected average variance."""
        net = SPXImpliedVolNet(hidden_dim=16, n_layers=2)
        f = torch.tensor(_features([-0.1, 0.0, 0.1], 0.25, _BASE), dtype=torch.float32)
        with torch.no_grad():
            out = net(f)
        assert out.shape == (3,) and (out > 0).all()
        _, sref = SPXImpliedVolNet.engineer(f)
        expected = nnp._sigma_ref(0.25, _BASE["kappa"], _BASE["theta"], _BASE["v0"])
        np.testing.assert_allclose(sref.numpy(), expected, rtol=1e-5)


# ── 2. IV helper re-export ───────────────────────────────────────────────────

class TestBsIvVectorized:
    """nn_pricer keeps exporting the solver under its old private name."""

    def _make_inputs(self, n, sigma_range, seed=0):
        rng = np.random.default_rng(seed)
        S, r, q = 6000.0, 0.045, 0.013
        T = rng.uniform(0.05, 1.0, n)
        log_m = rng.uniform(-0.25, 0.20, n)
        sig = rng.uniform(*sigma_range, n)
        F = S * np.exp((r - q) * T)
        K = F * np.exp(log_m)
        C = np.array([_bs_call_price(f, k, t, s) for f, k, t, s in zip(F, K, T, sig)])
        return C * np.exp(-r * T), F, K, T, np.full(n, r), sig

    def test_same_object_as_implied_vol(self):
        assert _bs_iv_vectorized is implied_vol._bs_iv_vectorized

    def test_roundtrip_atm_accuracy(self):
        C, F, K, T, r, sig = self._make_inputs(50, (0.10, 0.60))
        C_atm = np.array([_bs_call_price(f, f, t, s) for f, t, s in zip(F, T, sig)]) * np.exp(-r * T)
        rec = _bs_iv_vectorized(C_atm, F, F.copy(), T, r)
        assert np.abs(rec - sig).max() < 1e-9

    def test_roundtrip_skew_range_all_exact(self):
        """Every one of 400 OTM/ITM calls inverts to 1e-8. The old test let 5% be
        wrong by > 0.01 vol and its NaN check was vacuous: the old solver never
        returned NaN, it returned 1e-4 (3 of 100 samples here)."""
        C, F, K, T, r, sig = self._make_inputs(400, (0.05, 1.20))
        well_posed = (np.where(K >= F, C, C - (F - K) * np.exp(-r * T)) * np.exp(r * T)) > 1e-8 * F
        rec = _bs_iv_vectorized(C, F, K, T, r)
        assert not np.isnan(rec[well_posed]).any()
        assert np.abs(rec[well_posed] - sig[well_posed]).max() < 1e-8

    def test_nan_for_negative_prices(self):
        F = np.array([100.0, 100.0, 100.0]); K = np.array([90.0, 100.0, 110.0])
        T = np.full(3, 0.25); r = np.full(3, 0.05)
        assert np.all(np.isnan(_bs_iv_vectorized(np.array([-1.0, -0.5, -1.0]), F, K, T, r)))

    def test_nan_for_zero_time(self):
        out = _bs_iv_vectorized(np.array([1.0]), np.array([100.0]), np.array([100.0]),
                                np.array([0.0]), np.array([0.0]))
        assert np.isnan(out[0])

    def test_output_in_valid_range(self):
        C, F, K, T, r, _ = self._make_inputs(50, (0.15, 0.60))
        rec = _bs_iv_vectorized(C, F, K, T, r)
        valid = rec[~np.isnan(rec)]
        assert (valid > 1e-4).all() and (valid < 5.0).all()


# ── 3. _sample_heston_params (legacy sampler) ────────────────────────────────

class TestSampleHestonParams:

    CENTER = {"kappa": 4.62, "theta": 0.0764, "sigma": 0.8407, "rho": -0.95, "v0": 0.0561}

    def test_output_shape(self):
        samples = _sample_heston_params(1000, self.CENTER, perturb=0.30)
        assert all(len(samples[k]) == 1000 for k in self.CENTER)

    def test_all_within_heston_bounds(self):
        samples = _sample_heston_params(5000, self.CENTER, perturb=0.30)
        for name, arr in samples.items():
            lo, hi = HESTON_BOUNDS[name]
            assert (arr >= lo).all() and (arr <= hi).all()

    def test_reproducible_with_rng(self):
        s1 = _sample_heston_params(100, self.CENTER, rng=np.random.default_rng(42))
        s2 = _sample_heston_params(100, self.CENTER, rng=np.random.default_rng(42))
        for name in self.CENTER:
            np.testing.assert_array_equal(s1[name], s2[name])


# ── 4. Training domain ───────────────────────────────────────────────────────

class TestTrainingDomain:

    def test_archived_fits_inside_domain(self):
        """The old ±30% box around one day's fit contained 0 of the archived
        calibrations; the domain must contain all of them."""
        for p in _ARCHIVED_EXTREMES:
            feats = _features([0.0], 0.25, p)
            assert not spx_out_of_domain(feats)[0], p
            assert all(VIX_TRAINING_DOMAIN[k][0] <= p[k] <= VIX_TRAINING_DOMAIN[k][1]
                       for k in ("kappa", "theta", "sigma", "v0"))

    @pytest.mark.parametrize("bad", [
        dict(rho=-0.2), dict(v0=0.5), dict(kappa=0.2), dict(theta=0.5), dict(sigma=1.5),
    ])
    def test_parameter_outside(self, bad):
        assert spx_out_of_domain(_features([0.0], 0.25, {**_BASE, **bad}))[0]

    def test_T_x_and_z_outside(self):
        assert spx_out_of_domain(_features([0.0], 3.0, _BASE))[0]          # T > 2y
        assert spx_out_of_domain(_features([0.3], 1.0, _BASE))[0]          # x > 0.25
        low_vol = {**_BASE, "v0": 1e-4, "theta": 0.01, "kappa": 0.5}       # σ_ref ≈ 1.2% at 7d
        assert spx_out_of_domain(_features([-0.2], 7 / 365, low_vol))[0]   # z ≈ −60
        assert not spx_out_of_domain(_features([-0.2], 1.0, _BASE))[0]

    def test_sampler_stays_in_domain(self):
        rng = np.random.default_rng(0)
        draws = _sample_domain_params(5000, SPX_TRAINING_DOMAIN, rng)
        for k, arr in draws.items():
            lo, hi = SPX_TRAINING_DOMAIN[k]
            assert arr.min() >= lo and arr.max() <= hi


# ── 5. Labels and data generation ────────────────────────────────────────────

class TestExactLabels:

    @pytest.mark.parametrize("p, T", [
        (_BASE, 0.25),
        (dict(kappa=12.86, theta=0.0278, sigma=0.846, rho=-0.95, v0=1e-4), 14 / 365),
    ])
    def test_matches_independent_reference(self, p, T):
        lo, hi = nnp._x_interval(T, p, SPX_TRAINING_DOMAIN)
        x = np.linspace(lo, hi, 7)
        iv, otm = heston_iv_exact(x, T, p)
        for xi, ivi, oi in zip(x, iv, otm):
            ref, ref_otm = _ref_iv(xi, T, p)
            assert abs(oi - ref_otm) < 1e-13
            if ref_otm >= NN_LABEL_MIN_OTM_PRICE_REL:
                assert abs(ivi - ref) < 1e-9
            else:
                assert np.isnan(ivi)

    def test_auditor_examples_otm_call_wing(self):
        """Baseline params, T=0.25 (the old benchmark slice): the options at
        x ≈ 0.091, 0.117, 0.154 (worth ~$48, $16, $1.3) have reference IVs
        0.1773, 0.1572, 0.1292 (audit, x rounded to 3 dp). The old pipeline
        returned 1e-4 for all three and the benchmark excluded them."""
        iv, _ = heston_iv_exact(np.array([0.091, 0.117, 0.154]), 0.25, _load_baseline_params())
        np.testing.assert_allclose(iv, [0.1773, 0.1572, 0.1292], atol=1e-3)


@pytest.fixture(scope="module")
def small_data(tmp_path_factory):
    path = tmp_path_factory.mktemp("nn") / "spx_small.parquet"
    return generate_spx_training_data(1200, path=path, n_T=4, n_K=10), path


class TestDataGeneration:

    def test_schema_and_labels(self, small_data):
        df, _ = small_data
        assert {"set_id", *SPX_FEATURES, "iv_target", "otm_rel"} <= set(df.columns)
        assert df["set_id"].nunique() == 30
        assert np.isfinite(df["iv_target"]).all() and (df["iv_target"] > 0).all()
        assert (df["otm_rel"] >= NN_LABEL_MIN_OTM_PRICE_REL).all()
        assert not spx_out_of_domain(df[SPX_FEATURES].to_numpy()).any()

    def test_labels_are_exact(self, small_data):
        df, _ = small_data
        sample = df.iloc[:: max(len(df) // 12, 1)]
        for row in sample.itertuples():
            p = {k: getattr(row, k) for k in ("kappa", "theta", "sigma", "rho", "v0")}
            ref, _ = _ref_iv(row.x, row.sqrt_T**2, p)
            assert abs(row.iv_target - ref) < 1e-9

    def test_otm_call_wing_labelled(self, small_data):
        """Only options below the price floor go unlabelled — the OTM-call wing
        is no longer dropped by solver failures (51-77% missing before)."""
        df, _ = small_data
        assert len(df) >= 0.9 * 1200
        assert (df["x"] > 0.10).sum() > 0.05 * len(df)
        # Re-run 5 parameter sets with no floor: every option missing from the
        # data is one whose OTM value is below the floor, nothing else.
        ids = np.arange(5)
        params = _sample_domain_params(30, SPX_TRAINING_DOMAIN,
                                       np.random.default_rng(nnp.RANDOM_SEED))
        args = (ids, {k: v[ids] for k, v in params.items()}, SPX_TRAINING_DOMAIN, 4, 10,
                nnp.RANDOM_SEED)
        unfloored = nnp._spx_rows_for_sets(args + (0.0, None))
        assert len(unfloored) == 5 * 4 * 10          # without a floor every option inverts
        kept = df[df["set_id"].isin(ids)]
        missing = len(unfloored) - len(kept)
        assert missing == (unfloored["otm_rel"] < NN_LABEL_MIN_OTM_PRICE_REL).sum()

    def test_deterministic_and_cached(self, small_data, tmp_path):
        df, path = small_data
        again = generate_spx_training_data(1200, path=tmp_path / "again.parquet", n_T=4, n_K=10)
        pd.testing.assert_frame_equal(df, again)
        cached = generate_spx_training_data(1200, path=path, n_T=4, n_K=10)
        pd.testing.assert_frame_equal(df, cached)

    def test_interrupted_generation_resumes(self, small_data, tmp_path, monkeypatch):
        """Finished blocks survive a crash and are reused, not re-priced."""
        df, _ = small_data
        path = tmp_path / "resume.parquet"

        def crash(*a, **k):
            raise KeyboardInterrupt
        monkeypatch.setattr(nnp, "_write_cached", crash)
        with pytest.raises(KeyboardInterrupt):
            generate_spx_training_data(1200, path=path, n_T=4, n_K=10)
        monkeypatch.undo()

        def no_pricing(*a, **k):
            raise AssertionError("block was re-priced")
        monkeypatch.setattr(nnp, "_heston_otm_rel", no_pricing)
        resumed = generate_spx_training_data(1200, path=path, n_T=4, n_K=10)
        pd.testing.assert_frame_equal(df, resumed)
        assert not list(tmp_path.glob("resume.parts-*"))

    def test_cache_with_other_settings_is_not_reused(self, small_data):
        df, path = small_data
        other = generate_spx_training_data(600, path=path, n_T=4, n_K=10)
        assert other["set_id"].nunique() == 15

    def test_vix_data_has_groups(self, tmp_path):
        dv = generate_vix_training_data(400, path=tmp_path / "vix.parquet")
        assert {"set_id", *VIX_FEATURES, "vix_target"} <= set(dv.columns)
        assert (dv.groupby("set_id").size() == 10).all()


# ── 6. Training ───────────────────────────────────────────────────────────────

class TestTraining:

    def test_group_split_has_no_shared_sets(self):
        groups = np.repeat(np.arange(200), 10)
        tr, va, te = _split_by_group(groups, 0.1, 0.1, seed=1)
        g = [set(groups[i]) for i in (tr, va, te)]
        assert not (g[0] & g[1]) and not (g[0] & g[2]) and not (g[1] & g[2])
        assert len(tr) + len(va) + len(te) == len(groups) and len(g[2]) == 20

    def test_train_spx_runs_seeded_and_reports_units(self, small_data, tmp_path):
        """train_network crashed on torch >= 2.6 (ReduceLROnPlateau verbose=);
        weights were unseeded; passed_target accepted 10 vol points."""
        df, _ = small_data
        kw = dict(data=df, max_epochs=2, hidden_dim=16, n_layers=2, batch_size=256, verbose=False)
        r1 = train_spx(model_path=tmp_path / "a.pt", **kw)
        r2 = train_spx(model_path=tmp_path / "b.pt", **kw)
        a = torch.load(tmp_path / "a.pt", weights_only=True)
        b = torch.load(tmp_path / "b.pt", weights_only=True)
        assert all(torch.equal(a["state_dict"][k], b["state_dict"][k]) for k in a["state_dict"])
        assert r1["test_mae_vp"] == r2["test_mae_vp"]
        # a 2-epoch, 16-unit network is nowhere near 0.1 vol point
        assert r1["test_mae_vp"] > NN_VOL_ERROR_TARGET * 100 and r1["passed_target"] is False
        assert r1["n_test"] > 0 and r1["mae_vol_pts"] == r1["test_mae_vp"]
        meta = a["training"]
        assert meta["seed"] == nnp.RANDOM_SEED and len(meta["data_sha256"]) == 64
        assert a["domain"] == SPX_TRAINING_DOMAIN and a["format"] == "spx_iv_net_v2"

    def test_train_network_native_units_and_target(self, tmp_path):
        rng = np.random.default_rng(0)
        n_sets = 60
        df = pd.DataFrame({"set_id": np.repeat(np.arange(n_sets), 10),
                           "a": rng.normal(size=n_sets * 10), "b": rng.normal(size=n_sets * 10)})
        df["y"] = 20.0 + 2.0 * df["a"] - df["b"]
        common = dict(hidden_dim=8, n_layers=2, max_epochs=2, batch_size=64, verbose=False,
                      target_units="VIX points")
        loose = train_network(df, ["a", "b"], "y", tmp_path / "m1.pt", target_mae=1e9, **common)
        tight = train_network(df, ["a", "b"], "y", tmp_path / "m2.pt", target_mae=1e-9, **common)
        assert loose["passed_target"] is True and tight["passed_target"] is False
        assert loose["test_mae"] == tight["test_mae"]
        assert loose["target_units"] == "VIX points" and loose["n_test"] == 60
        ckpt = torch.load(tmp_path / "m1.pt", weights_only=True)
        assert ckpt["test_metrics"]["mae"] == pytest.approx(loose["test_mae"])
        # the split is by parameter set: the recorded test groups hold exactly the
        # test rows (train_vix used to row-shuffle, leaking 82.6% of val sets)
        split = ckpt["training"]["split"]
        assert split["group_col"] == "set_id"
        assert df["set_id"].isin(split["test_groups"]).sum() == loose["n_test"]
        assert df["set_id"].isin(split["val_groups"]).sum() == loose["n_val"]
        assert not set(split["test_groups"]) & set(split["val_groups"])

    def test_train_network_without_groups_falls_back(self, tmp_path):
        df = pd.DataFrame({"a": np.linspace(0, 1, 3000), "y": np.linspace(1, 2, 3000)})
        res = train_network(df, ["a"], "y", tmp_path / "m.pt", hidden_dim=8, n_layers=1,
                            max_epochs=1, verbose=False)
        assert res["n_train"] + res["n_val"] + res["n_test"] == 3000


# ── 7. NNPricer on a tiny model ──────────────────────────────────────────────

class _Smuggled:
    """Stand-in for an arbitrary pickled object inside a checkpoint."""


@pytest.fixture(scope="module")
def tiny_pricer(small_data, tmp_path_factory):
    df, _ = small_data
    path = tmp_path_factory.mktemp("nnmodel") / "spx.pt"
    train_spx(data=df, model_path=path, max_epochs=3, hidden_dim=16, n_layers=2,
              batch_size=256, verbose=False)
    return NNPricer.load(spx_path=path), path


class TestNNPricerTiny:

    def test_checkpoint_loading_refuses_pickled_code(self, tiny_pricer, tmp_path):
        """Checkpoints hold only tensors and plain data, so they are loaded with
        torch.load(weights_only=True); a checkpoint smuggling an arbitrary
        object must be rejected instead of unpickled."""
        _, path = tiny_pricer
        ckpt = torch.load(path, weights_only=True)
        ckpt["payload"] = _Smuggled()
        bad = tmp_path / "bad.pt"
        torch.save(ckpt, bad)
        with pytest.raises(Exception, match="(?i)weights.only|unsupported global"):
            NNPricer.load(spx_path=bad)

    def test_load_does_not_train_by_default(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            NNPricer.load(spx_path=tmp_path / "missing.pt")

    def test_missing_explicit_vix_raises(self, tiny_pricer, tmp_path):
        _, path = tiny_pricer
        with pytest.raises(FileNotFoundError):
            NNPricer.load(spx_path=path, vix_path=tmp_path / "missing_vix.pt")

    def test_spx_iv_batch_shape_and_positive(self, tiny_pricer):
        pricer, _ = tiny_pricer
        ivs = pricer.spx_iv_batch(_features(np.linspace(-0.2, 0.15, 50), 0.25, _BASE))
        assert ivs.shape == (50,) and (ivs > 0).all()
        assert isinstance(pricer.spx_iv(-0.05, 0.25, **_BASE), float)

    def test_out_of_domain_raises_by_default(self, tiny_pricer):
        pricer, _ = tiny_pricer
        feats = _features([0.0, 0.0], 0.25, _BASE)
        feats[1, 5] = -0.2                       # rho outside [-0.99, -0.30]
        with pytest.raises(ValueError, match="outside NN-1's training domain"):
            pricer.spx_iv_batch(feats)
        out = pricer.spx_iv_batch(feats, out_of_domain="nan")
        assert np.isfinite(out[0]) and np.isnan(out[1])
        exact = pricer.spx_iv_batch(feats, out_of_domain="exact")
        ref, _ = _ref_iv(0.0, 0.25, {**_BASE, "rho": -0.2})
        assert exact[1] == pytest.approx(ref, abs=1e-9)

    def test_benchmark_scores_every_well_posed_option(self, tiny_pricer, monkeypatch):
        """The old benchmark dropped options where the NN predicted < 2% IV (and
        every OTM call the reference solver failed on). Predicting 0.01% IV
        everywhere must therefore count against the NN on every option."""
        pricer, _ = tiny_pricer
        monkeypatch.setattr(pricer, "spx_iv_batch",
                            lambda f, out_of_domain="raise": np.full(len(f), 1e-4))
        res = pricer.benchmark(n_options=24, n_param_sets=2, repeats=1, verbose=False)
        assert res["n_well_posed"] >= 0.9 * res["n_options"]
        assert res["mae_vol_pts"] > 5.0
        assert res["target_passed"] is False

    def test_benchmark_covers_otm_call_wing_and_reports_machine(self, tiny_pricer):
        pricer, _ = tiny_pricer
        res = pricer.benchmark(n_options=60, n_param_sets=3, repeats=1, verbose=False)
        legacy = {"n_options", "heston_ms", "nn_ms", "speedup", "mae_vol_pts",
                  "rmse_vol_pts", "target_passed"}
        assert legacy <= res.keys()
        wing_mae, wing_n = res["mae_by_moneyness_vp"]["[+0.10,+0.25]"]
        assert wing_n > 0 and wing_mae is not None
        assert res["machine"]["torch"] and res["speedup_iqr"][0] <= res["speedup"]

    def test_vix_optional_and_deprecated(self, tiny_pricer, tmp_path):
        pricer, _ = tiny_pricer
        assert "NN-2=not loaded" in repr(pricer)
        dv = generate_vix_training_data(600, path=tmp_path / "vix.parquet")
        vpath = tmp_path / "vix.pt"
        train_network(dv, VIX_FEATURES, "vix_target", vpath, hidden_dim=8, n_layers=2,
                      max_epochs=1, verbose=False, domain=VIX_TRAINING_DOMAIN)
        p2 = NNPricer.load(spx_path=tiny_pricer[1], vix_path=vpath)
        with pytest.warns(DeprecationWarning):
            out = p2.vix_prices(4.62, 0.0764, 0.84, 0.0561, np.array([0.1, 0.5]))
        assert out.shape == (2,)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            with pytest.raises(ValueError):
                p2.vix_prices(4.62, 0.0764, 0.84, 0.0561, np.array([3.0]))


# ── 8. Shipped weights (skipped when absent) ─────────────────────────────────

def _make_pricer() -> NNPricer:
    if not _SPX_MODEL_PATH.exists():
        pytest.skip("Trained model files not found — run train_spx() first")
    return NNPricer.load()


class TestShippedModel:

    PARAMS = dict(kappa=4.62, theta=0.0764, sigma=0.8407, rho=-0.99, v0=0.0561)

    def test_load_and_repr(self):
        pricer = _make_pricer()
        assert "NN-1=loaded" in repr(pricer)
        assert not pricer._spx_model.training

    def test_checkpoint_records_domain_and_test_metrics(self):
        pricer = _make_pricer()
        meta = pricer._spx_meta
        assert meta.get("format") == "spx_iv_net_v2", "legacy checkpoint: retrain with train_spx()"
        assert meta["domain"] == SPX_TRAINING_DOMAIN
        assert meta["test_metrics_vp"]["n"] > 10_000

    def test_reasonable_atm_level(self):
        pricer = _make_pricer()
        iv = pricer.spx_iv(log_moneyness=0.0, T=0.25, **self.PARAMS)
        ref, _ = _ref_iv(0.0, 0.25, self.PARAMS)
        assert abs(iv - ref) < 0.01

    def test_negative_skew(self):
        pricer = _make_pricer()
        iv_put = pricer.spx_iv(-0.20, 0.25, **self.PARAMS)
        iv_call = pricer.spx_iv(0.20, 0.25, **self.PARAMS)
        assert iv_put > iv_call

    def test_benchmark_consistent_with_recorded_test_error(self):
        """Fresh parameter sets from the domain score close to the checkpoint's
        held-out test MAE (both are out-of-sample estimates of the same thing)."""
        pricer = _make_pricer()
        res = pricer.benchmark(n_options=120, n_param_sets=8, repeats=1, verbose=False)
        recorded = pricer._spx_meta["test_metrics_vp"]["mae"]
        assert 0 < res["mae_vol_pts"] < 3 * recorded + 0.05
        assert res["target_passed"] == (res["mae_vol_pts"] < NN_VOL_ERROR_TARGET * 100)

    def test_benchmark_speedup_positive(self):
        """Wall-clock: best of 3 samples, so transient CPU contention does not
        flip a genuine order-of-magnitude advantage."""
        if sys.gettrace() is not None:
            # coverage/debugger tracing taxes the Python-heavy reference pricer
            # and the NN path unequally; the ratio would measure the tracer.
            pytest.skip("wall-clock benchmark is not meaningful under a tracer (pytest --cov)")
        pricer = _make_pricer()
        speedups = []
        for _ in range(3):
            speedups.append(pricer.benchmark(n_options=300, n_param_sets=3, repeats=3,
                                             verbose=False)["speedup"])
            if speedups[-1] > 1.0:
                break
        assert max(speedups) > 1.0, speedups


# ── 9. Cached training data (skipped when absent) and baseline params ────────

class TestTrainingDataCache:

    def test_spx_cache_schema(self):
        if not _SPX_DATA_PATH.exists():
            pytest.skip("SPX training data cache not found")
        df = pd.read_parquet(_SPX_DATA_PATH)
        assert {"set_id", *SPX_FEATURES, "iv_target", "otm_rel"} <= set(df.columns)
        assert np.isfinite(df["iv_target"]).all()
        assert (df["otm_rel"] >= NN_LABEL_MIN_OTM_PRICE_REL).all()

    def test_vix_cache_schema(self):
        if not _VIX_DATA_PATH.exists():
            pytest.skip("VIX training data cache not found")
        df = pd.read_parquet(_VIX_DATA_PATH)
        assert {"set_id", *VIX_FEATURES, "vix_target"} <= set(df.columns)


class TestLoadBaselineParams:

    def test_returns_dict_with_all_params(self):
        assert {"kappa", "theta", "sigma", "rho", "v0"} <= _load_baseline_params().keys()

    def test_params_within_heston_bounds(self):
        for name, val in _load_baseline_params().items():
            lo, hi = HESTON_BOUNDS[name]
            assert lo <= val <= hi

    def test_rho_negative(self):
        assert _load_baseline_params()["rho"] < 0
