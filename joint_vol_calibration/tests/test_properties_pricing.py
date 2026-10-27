"""
test_properties_pricing.py: property-based tests (Hypothesis) of the
no-arbitrage and invariance properties of the Heston/Bates engine, over the
calibrator's whole parameter box.

The example-based tests (test_heston.py, test_heston_reference.py) check the
engine at chosen points against independent references. These check
properties that must hold at EVERY point of HESTON_BOUNDS, at any maturity
from one day to three years, any carry, on strike ladders out to
four standard deviations:

  static arbitrage in strike   intrinsic <= C <= S e^{-qT}; C non-increasing in K
                               with slope >= -DF; C convex in K (butterflies >= 0)
  calendar arbitrage           at fixed forward moneyness, C / (DF F_T) is
                               non-decreasing in T (S_t/F_t is a martingale)
  put-call parity              the call from heston._lewis_call_prices and the
                               OTM put from implied_vol.fourier_otm_price (a
                               second Lewis engine, opposite x convention, its
                               own truncation and refinement) satisfy
                               C - P = DF (F - K)
  martingale condition         log phi(0) = log phi(-i) = 0 for the Heston and
                               Bates characteristic functions (what makes parity
                               and E[S_T] = F_T hold in a Fourier engine)
  Black-Scholes limit          sigma -> 0: Heston = Black-76 with the deterministic
                               total variance E[int v dt], for any rho, v0, theta
  nesting                      Bates with lam = 0 is Heston, to the last bits
  time change                  (kappa, theta, sigma, v0, r, q; T) and
                               (kappa/c, theta/c, sigma/c, v0/c, r/c, q/c; cT) give
                               the same prices (the variance clock rescales)
  homogeneity                  C(lam S, lam K) = lam C(S, K)
  VIX                          E[VIX_T] <= sqrt(E[VIX_T^2]) (Jensen); the spot
                               index lies between 100 sqrt(v0) and 100 sqrt(theta)

They run on the UNCLIPPED engine (heston._lewis_call_prices). The public
heston_call_batch clips every price into [DF max(F-K, 0), DF F], so a bound
test on it is true by construction and proves nothing; one property here
checks instead that the clip moves no price by more than the engine's
quadrature tolerance, i.e. that it only removes rounding noise, as its
docstring says.

Tolerances are in units of DF*F (the engine's error target is 1e-10 DF F per
price; the docstring states 5e-9 F at the slowest-decaying corners of the box,
where the 2**20-node cap binds). They are set from that target, not from the
observed errors, which are much smaller (worst bound excess 1.4e-13 DF F over
the 64 corners of the box).

Example budgets come from the Hypothesis profile (HYPOTHESIS_PROFILE: ci, the
default, is derandomized, so CI tests the same examples on every run; dev;
thorough = 10x the examples, random); see _hypothesis_profiles below.
"""
from __future__ import annotations

import math
import os
import warnings

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import HealthCheck, assume, event, given, note, settings
from hypothesis import strategies as st

from joint_vol_calibration.config import HESTON_BOUNDS
from joint_vol_calibration.models import heston as H
from joint_vol_calibration.models.implied_vol import (
    black_price,
    fourier_otm_price,
    implied_vol_black,
)

pytestmark = pytest.mark.property


# ── Hypothesis profiles ──────────────────────────────────────────────────────

def _hypothesis_profiles() -> None:
    """ci (default): derandomized, so a CI run is reproducible and a failure
    reproduces locally with the same command; no example database (nothing
    is written into the checkout). dev: random examples. thorough: random,
    10x the examples (weekly CI run, or by hand)."""
    common = settings(deadline=None, database=None, print_blob=True,
                      suppress_health_check=[HealthCheck.too_slow])
    settings.register_profile("ci", common, max_examples=60, derandomize=True)
    settings.register_profile("dev", common, max_examples=60)
    settings.register_profile("thorough", common, max_examples=600)
    settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "ci"))


_hypothesis_profiles()


def budget(fraction: float = 1.0) -> int:
    """Examples for one test: a fraction of the active profile's budget."""
    return max(5, int(settings().max_examples * fraction))


# ── Strategies ───────────────────────────────────────────────────────────────

def log_uniform(lo: float, hi: float) -> st.SearchStrategy[float]:
    """Floats in [lo, hi], uniform in log (Hypothesis still tries both ends)."""
    return st.floats(math.log(lo), math.log(hi)).map(lambda u: min(max(math.exp(u), lo), hi))


B = HESTON_BOUNDS
heston_params = st.fixed_dictionaries({
    "kappa": log_uniform(*B["kappa"]),
    "theta": log_uniform(*B["theta"]),
    "sigma": log_uniform(*B["sigma"]),
    "rho": st.floats(*B["rho"]),
    "v0": log_uniform(*B["v0"]),
})
# Bates jumps on the calibrator's own jump box (joint_calibrator._BATES_JUMP_BOUNDS).
jump_params = st.fixed_dictionaries({
    "lam": st.floats(0.0, 10.0),
    "mu_j": st.floats(-0.30, 0.10),
    "sigma_j": st.floats(0.0, 0.30),
})
market = st.fixed_dictionaries({
    "T": log_uniform(1.0 / 365.0, 3.0),
    "r": st.floats(-0.01, 0.08),
    "q": st.floats(-0.08, 0.08),       # parity-implied q_T spans about -8% .. +8%
})

TOL = 2e-9      # per price, in units of DF*F (documented corner accuracy 5e-9 F / 2.5)
TOL_DIFF = 4e-9  # for a difference of two prices / a butterfly of three


def _log_cf(p: dict, jumps: dict | None, T: float):
    if jumps is None:
        return lambda z: H._heston_log_cf(z, T, p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])
    return lambda z: (H._heston_log_cf(z, T, p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])
                      + H._bates_jump_log_cf(z, T, jumps["lam"], jumps["mu_j"], jumps["sigma_j"]))


def raw_calls(S: float, K: np.ndarray, T: float, r: float, q: float, p: dict,
              jumps: dict | None = None) -> np.ndarray:
    """Unclipped discounted calls from the engine, with the same total-variance
    hint and jump extent heston_call_batch / bates_call_batch pass it.
    Engine warnings (quadrature not confirmed at the node cap) are recorded as
    a Hypothesis event: the properties must hold there too."""
    w = float(H._expected_integrated_variance(T, p["kappa"], p["theta"], p["v0"]))
    extent = 0.0
    if jumps is not None:
        w += jumps["lam"] * T * (jumps["mu_j"] ** 2 + jumps["sigma_j"] ** 2)
        if jumps["lam"] * T > 1e-14:
            extent = abs(jumps["mu_j"]) + 10.0 * jumps["sigma_j"]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", RuntimeWarning)
        calls, _ = H._lewis_call_prices(S, K, T, r, q, _log_cf(p, jumps, T), w, jump_extent=extent)
    if caught:
        event("engine warned: quadrature not confirmed within the node cap")
    return calls


def strike_ladder(F: float, T: float, p: dict, n: int = 33, width: float = 4.0) -> np.ndarray:
    """Equally spaced strikes (so second differences are butterflies) over
    ln(K/F) in [-width, +width] standard deviations of ln S_T, clipped to
    keep every strike positive."""
    sd = math.sqrt(max(float(H._expected_integrated_variance(T, p["kappa"], p["theta"], p["v0"])), 1e-12))
    lo, hi = F * math.exp(-width * sd), F * math.exp(min(width * sd, 3.0))
    return np.linspace(lo, hi, n)


# ── Static arbitrage in strike ───────────────────────────────────────────────

@settings(max_examples=budget())
@given(p=heston_params, m=market, S=log_uniform(1.0, 1e4),
       jumps=st.one_of(st.none(), jump_params))
def test_calls_are_free_of_static_arbitrage_in_strike(p, m, S, jumps):
    T, r, q = m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    K = strike_ladder(F, T, p)
    c = raw_calls(S, K, T, r, q, p, jumps)
    unit = DF * F
    note(f"F={F:.6g} DF={DF:.6g} K=[{K[0]:.6g}..{K[-1]:.6g}] c={c / unit}")
    assert np.all(np.isfinite(c))
    # bounds: intrinsic <= C <= S e^{-qT} (= DF F)
    assert np.all(c >= DF * np.maximum(F - K, 0.0) - TOL * unit), "below intrinsic"
    assert np.all(c <= unit + TOL * unit), "above S e^{-qT}"
    # call spreads: 0 >= C(K2) - C(K1) >= -DF (K2 - K1)
    dc, dk = np.diff(c), np.diff(K)
    assert np.all(dc <= TOL_DIFF * unit), "call price increases with strike"
    assert np.all(dc >= -DF * dk - TOL_DIFF * unit), "call spread worth more than DF x width"
    # butterflies on the equally spaced ladder: C(K-h) - 2C(K) + C(K+h) >= 0
    fly = c[:-2] - 2.0 * c[1:-1] + c[2:]
    assert np.all(fly >= -TOL_DIFF * unit), "butterfly with negative value"


@settings(max_examples=budget())
@given(p=heston_params, m=market)
def test_public_clip_only_removes_rounding_noise(p, m):
    """heston_call_batch = clip(raw engine price): the clip may move a price
    by no more than the engine's tolerance, so it never hides a real error."""
    S, T, r, q = 100.0, m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    K = strike_ladder(F, T, p, n=17)
    raw = raw_calls(S, K, T, r, q, p)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        public = H.heston_call_batch(S, K, T, r, q, p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"])
    assert np.max(np.abs(public - raw)) <= TOL * DF * F


# ── Calendar arbitrage ───────────────────────────────────────────────────────

@settings(max_examples=budget())
@given(p=heston_params, r=st.floats(-0.01, 0.08), q=st.floats(-0.08, 0.08),
       T1=log_uniform(1.0 / 365.0, 2.0), ratio=st.floats(1.05, 4.0),
       k=st.lists(st.floats(-1.5, 1.0), min_size=1, max_size=8))
def test_no_calendar_arbitrage_at_fixed_forward_moneyness(p, r, q, T1, ratio, k):
    """c(k, T) = C(F_T e^k, T) / (DF_T F_T) = E[(S_T/F_T - e^k)^+] is
    non-decreasing in T: S_t/F_t is a martingale and (x - e^k)^+ is convex."""
    T2 = T1 * ratio
    S = 100.0
    kk = np.asarray(k)
    norm = []
    for T in (T1, T2):
        F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
        norm.append(raw_calls(S, F * np.exp(kk), T, r, q, p) / (DF * F))
    note(f"c(T1)={norm[0]} c(T2)={norm[1]}")
    assert np.all(norm[1] >= norm[0] - TOL_DIFF)


# ── Put-call parity across two independent Lewis engines ─────────────────────

@settings(max_examples=budget(0.5))
@given(p=heston_params, m=market,
       x=st.lists(st.floats(-2.0, 1.0), min_size=1, max_size=6))
def test_put_call_parity_between_the_two_lewis_engines(p, m, x):
    """heston._lewis_call_prices (x = ln(F/K), BS control variate on the
    expected integrated variance) prices the call; implied_vol.fourier_otm_price
    (x = ln(K/F), control variate on the CF's second cumulant, its own
    truncation and refinement) prices the OTM option at the same strike.
    For K < F that is the put, so C - P = DF (F - K) checks the two engines
    against each other through parity; for K >= F both price the call."""
    S, T, r, q = 100.0, m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    w = float(H._expected_integrated_variance(T, p["kappa"], p["theta"], p["v0"]))
    xs = np.asarray(x) * math.sqrt(max(w, 1e-12)) * 2.0          # ln(K/F), within +-4 sd
    K = F * np.exp(xs)
    call = raw_calls(S, K, T, r, q, p)
    log_cf = _log_cf(p, None, T)
    with np.errstate(over="ignore", invalid="ignore"):
        otm_over_f, ok = fourier_otm_price(lambda u: np.exp(log_cf(u)), xs)
    assume(ok)                       # its own convergence report; not the engine under test
    event("fourier_otm_price converged")
    otm = DF * F * otm_over_f
    put_side = xs < 0
    lhs = np.where(put_side, call - otm, call)                      # C - P, or C
    rhs = np.where(put_side, DF * (F - K), otm)                     # DF (F - K), or C
    note(f"K/F={np.exp(xs)} lhs-rhs={(lhs - rhs) / (DF * F)}")
    assert np.max(np.abs(lhs - rhs)) <= TOL_DIFF * DF * F


# ── Martingale condition of the characteristic functions ─────────────────────

@settings(max_examples=budget(2.0))
@given(p=heston_params, jumps=jump_params, T=log_uniform(1.0 / 365.0, 5.0))
def test_characteristic_functions_are_martingale_normalised(p, jumps, T):
    """phi(0) = 1 (a probability measure) and phi(-i) = E[S_T/F_T] = 1 (the
    forward is the risk-neutral mean), for Heston and for Bates, whose jump
    compensator is what makes the second hold."""
    for name, log_cf in (("heston", _log_cf(p, None, T)), ("bates", _log_cf(p, jumps, T))):
        vals = log_cf(np.array([0.0 + 0.0j, -1.0j]))
        note(f"{name}: log phi(0)={vals[0]!r} log phi(-i)={vals[1]!r}")
        assert abs(vals[0]) <= 1e-12, name
        assert abs(vals[1]) <= 1e-10, name


# ── Black-Scholes limit ──────────────────────────────────────────────────────

@settings(max_examples=budget())
@given(p=heston_params, m=market, x=st.lists(st.floats(-3.0, 3.0), min_size=1, max_size=8))
def test_vanishing_vol_of_vol_gives_black_with_the_deterministic_variance(p, m, x):
    """sigma -> 0 makes v_t deterministic, v_t = theta + (v0 - theta) e^{-kappa t},
    so Heston must price as Black-76 with total variance E[int_0^T v_t dt],
    for any rho and any v0 != theta (rho multiplies sigma, so it drops out)."""
    q = {**p, "sigma": 1e-10}
    S, T, r, qq = 100.0, m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - qq) * T), math.exp(-r * T)
    w = float(H._expected_integrated_variance(T, q["kappa"], q["theta"], q["v0"]))
    K = F * np.exp(np.asarray(x) * math.sqrt(w))
    heston = raw_calls(S, K, T, r, qq, q)
    black = black_price(F, K, T, math.sqrt(w / T), True, DF)
    note(f"w={w!r} diff={(heston - black) / (DF * F)}")
    assert np.max(np.abs(heston - black)) <= TOL * DF * F
    # and the implied vol of the Heston price is sqrt(w/T) wherever the time
    # value is resolvable (an OTM price of at least 1e-8 DF F)
    otm = np.where(K >= F, heston, heston - DF * (F - K))
    resolvable = otm >= 1e-8 * DF * F
    if resolvable.any():
        iv = implied_vol_black(heston[resolvable], F, K[resolvable], T, True, DF)
        assert np.max(np.abs(iv - math.sqrt(w / T))) <= 1e-5 * math.sqrt(w / T)


# ── Nesting, time change, homogeneity ───────────────────────────────────────

@settings(max_examples=budget())
@given(p=heston_params, m=market, mu_j=st.floats(-0.30, 0.10), sigma_j=st.floats(0.0, 0.30))
def test_bates_without_jumps_is_heston(p, m, mu_j, sigma_j):
    S, T, r, q = 100.0, m["T"], m["r"], m["q"]
    F = S * math.exp((r - q) * T)
    K = strike_ladder(F, T, p, n=9)
    hes = raw_calls(S, K, T, r, q, p)
    bat = raw_calls(S, K, T, r, q, p, {"lam": 0.0, "mu_j": mu_j, "sigma_j": sigma_j})
    np.testing.assert_allclose(bat, hes, rtol=0, atol=1e-14 * F)


@settings(max_examples=budget())
@given(p=heston_params, m=market, c=st.floats(0.25, 4.0))
def test_time_change_invariance(p, m, c):
    """With t' = c t the variance process v' = v/c is again Heston with
    (kappa/c, theta/c, sigma/c, v0'= v0/c) and rho unchanged, and the carry
    becomes (r/c, q/c): the price of the same claim over cT is unchanged."""
    S, T, r, q = 100.0, m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    K = strike_ladder(F, T, p, n=9)
    a = raw_calls(S, K, T, r, q, p)
    pc = {"kappa": p["kappa"] / c, "theta": p["theta"] / c, "sigma": p["sigma"] / c,
          "rho": p["rho"], "v0": p["v0"] / c}
    b = raw_calls(S, K, c * T, r / c, q / c, pc)
    note(f"a={a} b={b}")
    assert np.max(np.abs(a - b)) <= TOL_DIFF * DF * F


@settings(max_examples=budget())
@given(p=heston_params, m=market, lam=log_uniform(1e-3, 1e3))
def test_prices_are_homogeneous_of_degree_one_in_spot_and_strike(p, m, lam):
    S, T, r, q = 100.0, m["T"], m["r"], m["q"]
    F, DF = S * math.exp((r - q) * T), math.exp(-r * T)
    K = strike_ladder(F, T, p, n=9)
    a = raw_calls(S, K, T, r, q, p)
    b = raw_calls(lam * S, lam * K, T, r, q, p) / lam
    assert np.max(np.abs(a - b)) <= TOL_DIFF * DF * F


# ── VIX index and futures ────────────────────────────────────────────────────

@settings(max_examples=budget(2.0))
@given(p=heston_params, T=st.lists(log_uniform(1.0 / 365.0, 2.0), min_size=1, max_size=6),
       tau=log_uniform(1.0 / 365.0, 1.0))
def test_vix_future_is_below_its_jensen_bound_and_the_index_is_a_variance_average(p, T, tau):
    k, th, sg, v0 = p["kappa"], p["theta"], p["sigma"], p["v0"]
    Ts = np.asarray(T)
    exact = np.atleast_1d(H.heston_vix_future_exact(k, th, sg, v0, Ts))
    upper = np.atleast_1d(H.heston_vix_futures_upper_bound(k, th, sg, v0, Ts))
    note(f"exact={exact} upper={upper}")
    assert np.all(np.isfinite(exact))
    assert np.all(exact <= upper * (1.0 + 1e-10))                    # Jensen: E[sqrt X] <= sqrt(E X)
    # the spot index is 100 sqrt(average of a variance path between v0 and theta)
    idx = H.heston_vix_term_index(k, th, sg, v0, tau)
    lo, hi = 100.0 * math.sqrt(min(v0, th)), 100.0 * math.sqrt(max(v0, th))
    assert lo * (1 - 1e-12) <= idx <= hi * (1 + 1e-12)
