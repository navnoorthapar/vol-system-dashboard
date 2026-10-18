"""
test_joint_calibrator.py — Unit and integration tests for the C4 Joint Calibrator.

Two kinds of test:
  * synthetic-market tests: a complete SPX chain priced by Heston at KNOWN
    parameters off a known forward/discount, plus the matching VIX spot
    indices, fed through the real cleaning + anchor selection. They check
    what the C22 audit found broken: the VIX leg's instrument, parity carry,
    anchor coverage of the put wing, unique expiry buckets, T measured from
    the quote session, the Feller mode, weight normalisation, the quality
    gate, and that the leg is ~0 at the generating parameters.
  * real-snapshot tests on the committed 2026-03-24 chain (DB).

Run with:
  pytest tests/test_joint_calibrator.py -v
"""

import math

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.calibration import joint_calibrator as jc
from joint_vol_calibration.calibration.joint_calibrator import (
    JointCalibrator,
    active_constraints,
    calibration_is_legacy,
    classify_feller,
    count_pinned_params,
    heston_vix_call_price,
    heston_vix_put_price,
    is_acceptable_calibration,
)
from joint_vol_calibration.config import HESTON_BOUNDS, JOINT_W1, JOINT_W2, VIX_INDEX_TENOR_DAYS
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.models.heston import (
    heston_call_batch,
    heston_call_price,
    heston_vix_future_exact,
    heston_vix_futures_curve,
    heston_vix_index,
    heston_vix_term_index,
    implied_vol_from_price,
)

# ── Shared fixtures ───────────────────────────────────────────────────────────

_P = dict(kappa=3.0, theta=0.04, sigma=0.5, v0=0.04)  # stable Heston params
_R = 0.045
_Q = 0.013
_S = 5000.0

# Synthetic market (session Wed 2026-09-23)
_TRUE = dict(kappa=3.0, theta=0.05, sigma=0.7, rho=-0.75, v0=0.025)
_SESSION = "2026-09-23"
_SPOT = 7600.0
_R_TRUE, _Q_TRUE = 0.042, 0.011
_EXPIRIES = ["2026-10-07", "2026-10-23", "2026-11-20", "2026-12-18", "2027-03-19",
             "2027-09-17", "2026-10-09", "2026-10-30"]


def _heston_chain(params=_TRUE, session=_SESSION, S=_SPOT, r=_R_TRUE, q=_Q_TRUE,
                  expiries=_EXPIRIES, spread=0.2):
    """Two-sided SPXW chain priced by Heston (bid/ask = model -/+ spread/2)."""
    rows = []
    for exp in expiries:
        T = oc.year_fraction(session, exp, "PM")
        F = S * math.exp((r - q) * T)
        K = np.arange(round(0.6 * F / 25) * 25, 1.4 * F, 25.0)
        calls = heston_call_batch(S, K, T, r, q, params["kappa"], params["theta"],
                                  params["sigma"], params["rho"], params["v0"])
        puts = calls - S * math.exp(-q * T) + K * math.exp(-r * T)
        sym = exp[2:4] + exp[5:7] + exp[8:10]
        for right, px in (("C", calls), ("P", puts)):
            for k, p in zip(K, px):
                rows.append({"contractSymbol": f"SPXW{sym}{right}{int(k * 1000):08d}",
                             "strike": k, "bid": p - spread / 2, "ask": p + spread / 2,
                             "lastPrice": 1.0, "impliedVolatility": 0.3,
                             "expiration": exp,
                             "option_type": "call" if right == "C" else "put"})
    return pd.DataFrame(rows)


def _index_row(params=_TRUE, session=_SESSION, tickers=None):
    tickers = tickers or list(VIX_INDEX_TENOR_DAYS)
    row = {"date": session}
    for t in tickers:
        row[t] = heston_vix_term_index(params["kappa"], params["theta"], params["sigma"],
                                       params["v0"], VIX_INDEX_TENOR_DAYS[t] / 365.0)
    return pd.DataFrame([row])


def _patch_db(mp, session=_SESSION, S=_SPOT, vix_row=None, spx_surface=None,
              vix_surface=None, spx_dates=None):
    dates = spx_dates or ["2026-09-21", "2026-09-22", session]
    spx = pd.DataFrame({"date": dates, "close": [S * 0.99] * (len(dates) - 1) + [S]})
    mp.setattr(jc.db, "get_spx_ohlcv", lambda as_of_date, start_date=None:
               spx[spx["date"] <= as_of_date].reset_index(drop=True))
    mp.setattr(jc.db, "get_vix_term_structure_wide", lambda as_of_date, start_date=None:
               _index_row() if vix_row is None else vix_row)
    # tbill_rates stores the ^IRX bank-discount yield AND its cc conversion;
    # get_tbill_rate returns the cc rate (4.03% discount -> 4.10% cc)
    mp.setattr(jc.db, "get_tbill_rate", lambda as_of_date, fallback=0.045: 0.0410)
    mp.setattr(jc.db, "get_tbill_rates_series", lambda as_of_date, start_date=None:
               pd.DataFrame({"date": [dates[-1]], "rate": [0.0410], "discount_yield": [0.0403]}))
    mp.setattr(jc.db, "get_vix_futures_curve", lambda as_of_date:
               pd.DataFrame(columns=["expiry_month", "settle", "close", "price", "days_to_expiry"]))

    def _surface(as_of_date, underlying):
        if underlying == "VIX":
            return vix_surface if vix_surface is not None else pd.DataFrame()
        return spx_surface if spx_surface is not None else pd.DataFrame()
    mp.setattr(jc.db, "get_options_surface", _surface)


@pytest.fixture(scope="module")
def synth():
    """Calibrator on the synthetic Heston market (built once)."""
    with pytest.MonkeyPatch.context() as mp:
        _patch_db(mp)
        cal = JointCalibrator(as_of_date=_SESSION, spx_chain=_heston_chain())
    return cal


def _tvec(p=_TRUE):
    return [p["kappa"], p["theta"], p["sigma"], p["rho"], p["v0"]]


# ── VIX option pricing (CIR law) ──────────────────────────────────────────────

class TestHestonVixOptionPricing:
    """VIX options priced by integrating the payoff against the CIR law."""

    def test_atm_call_positive(self):
        assert heston_vix_call_price(20.0, 0.25, **_P) > 0

    def test_call_monotone_decreasing_in_strike(self):
        prices = [heston_vix_call_price(K, 0.25, **_P) for K in (15, 18, 20, 22, 25, 30)]
        assert all(a >= b for a, b in zip(prices, prices[1:]))

    def test_put_monotone_increasing_in_strike(self):
        prices = [heston_vix_put_price(K, 0.25, **_P) for K in (15, 18, 20, 22, 25)]
        assert all(a <= b for a, b in zip(prices, prices[1:]))

    @pytest.mark.parametrize("T", [30 / 365, 60 / 365, 0.25])
    @pytest.mark.parametrize("params", [
        _P, dict(kappa=3.28, theta=0.0462, sigma=0.516, v0=0.0116),
        dict(kappa=5.2, theta=0.06, sigma=1.7, v0=0.013)])          # d < 2: singular density
    def test_put_call_parity_with_exact_vix_forward(self, T, params):
        """C - P = DF (E[VIX_T] - K) with E[VIX_T] the exact futures price.

        Pre-C22 the put used parity with sqrt(E[VIX_T^2]) — above E[VIX_T] by
        the Jensen gap (0.7 pts at 30d), so puts were biased low by that much
        and could go negative; the old test hid it behind a 3-point tolerance.
        """
        F = heston_vix_future_exact(params["kappa"], params["theta"], params["sigma"],
                                    params["v0"], T)
        for K in (12.0, 15.0, 18.0, 25.0):
            c = heston_vix_call_price(K, T, **params, r=_R)
            p = heston_vix_put_price(K, T, **params, r=_R)
            assert c - p == pytest.approx(math.exp(-_R * T) * (F - K), abs=2e-3)

    def test_put_nonnegative_across_strikes(self):
        params = dict(kappa=3.28, theta=0.0462, sigma=0.516, v0=0.0116)
        for T in (30 / 365, 60 / 365):
            for K in np.arange(5.0, 61.0, 5.0):
                assert heston_vix_put_price(K, T, **params) >= 0.0
        # The old parity-with-sqrt(E[VIX^2]) put gave -0.38 here. Reference:
        # exact CIR sampling (scipy ncx2, 4M draws, seed 0) gives 0.5737
        # (standard error ~0.001).
        assert heston_vix_put_price(12.0, 60 / 365, **params, r=0.0) == pytest.approx(0.5737, abs=0.003)

    def test_zero_maturity_is_intrinsic_on_model_spot_vix(self):
        """At T=0 the call is max(model spot VIX - K, 0); the old code used
        100*sqrt(theta) and returned 13.80 instead of 0.80 at the showcase."""
        k, th, s, v0 = 1.7695, 0.0665, 0.46, 0.0127
        vix_now = heston_vix_index(k, th, v0)
        assert heston_vix_call_price(12.0, 0.0, k, th, s, v0) == pytest.approx(vix_now - 12.0)
        assert heston_vix_put_price(14.0, 0.0, k, th, s, v0) == pytest.approx(14.0 - vix_now)

    def test_high_sigma_higher_otm_call(self):
        c_lo = heston_vix_call_price(30.0, 0.5, kappa=3.0, theta=0.04, sigma=0.2, v0=0.04)
        c_hi = heston_vix_call_price(30.0, 0.5, kappa=3.0, theta=0.04, sigma=1.5, v0=0.04)
        assert c_hi > c_lo


# ── Batch Pricer Accuracy ──────────────────────────────────────────────────────

class TestHestonCallBatch:
    """Batch pricer agrees with the scalar pricer (both are the same engine)."""

    @pytest.mark.parametrize("T", [0.1, 0.25, 0.5, 1.0])
    def test_batch_vs_scalar(self, T):
        """|dIV| < 0.05 vp (the old test allowed 5.0 price units)."""
        kappa, theta, sigma, rho, v0 = 2.0, 0.04, 0.5, -0.7, 0.04
        strikes = np.array([4500., 4750., 5000., 5250., 5500.])
        batch = heston_call_batch(_S, strikes, T, _R, _Q, kappa, theta, sigma, rho, v0)
        for i, K in enumerate(strikes):
            scalar = heston_call_price(_S, K, T, _R, _Q, kappa, theta, sigma, rho, v0)
            iv_b = implied_vol_from_price(batch[i], _S, K, T, _R, _Q, "C")
            iv_s = implied_vol_from_price(scalar, _S, K, T, _R, _Q, "C")
            assert abs(iv_b - iv_s) < 5e-4

    def test_batch_monotone_in_strike(self):
        strikes = np.array([4000., 4500., 5000., 5500., 6000.])
        result = heston_call_batch(_S, strikes, 0.25, _R, _Q, 2.0, 0.04, 0.5, -0.7, 0.04)
        assert all(a >= b for a, b in zip(result, result[1:]))


# ── Synthetic market: legs, carry, anchors ────────────────────────────────────

class TestSyntheticMarket:

    def test_spx_leg_zero_at_generating_params(self, synth):
        """With mids priced by Heston off (F, D), the SPX leg vanishes at the
        generating parameters — carry, T and IVs are all consistent."""
        assert synth._spx_leg(**_TRUE) < 1e-10
        assert synth._spx_leg(**{**_TRUE, "rho": -0.6}) > 1e-6

    def test_vix_index_leg_zero_at_generating_params(self, synth):
        assert synth._vix_index_leg(_TRUE["kappa"], _TRUE["theta"], _TRUE["sigma"],
                                    _TRUE["v0"]) < 1e-20

    def test_vix_leg_prices_spot_indices_not_futures(self, synth):
        """The model ^VIX must be heston_vix_term_index(30d); the old leg used
        the futures curve (forward window [T, T+30d]) at 9/30/91/182 days."""
        ts = synth.vix_ts.set_index("ticker")
        assert ts.loc["^VIX3M", "tenor_years"] == pytest.approx(93 / 365)
        assert ts.loc["^VIX6M", "tenor_years"] == pytest.approx(184 / 365)
        p = _TRUE
        idx = heston_vix_term_index(p["kappa"], p["theta"], p["sigma"], p["v0"], 30 / 365)
        assert idx == pytest.approx(heston_vix_index(p["kappa"], p["theta"], p["v0"]))
        # Feeding the futures curve as "market" leaves a clearly non-zero leg.
        orig = synth.vix_ts.copy()
        try:
            synth.vix_ts["market_price"] = heston_vix_futures_curve(
                p["kappa"], p["theta"], p["sigma"], p["v0"], synth.vix_ts["tenor_years"].values)
            assert synth._vix_index_leg(p["kappa"], p["theta"], p["sigma"], p["v0"]) > 1e-5
        finally:
            synth.vix_ts = orig

    def test_published_showcase_misses_the_spot_indices(self, synth):
        """Regression: the 2026-05-31 showcase parameters (fitted with the
        futures mapping, published 'VIX RMSE 0.53') miss the 2026-05-29 spot
        indices by ~3 VIX points under the correct index model."""
        k, th, s_, v0 = 1.7684851, 0.0665629, 0.4602925, 0.0126525
        market = np.array([12.59, 15.32, 18.66, 21.60])
        T = np.array([9, 30, 93, 184]) / 365
        model = heston_vix_term_index(k, th, s_, v0, T)
        assert np.sqrt(np.mean((model - market) ** 2)) > 2.0
        assert model[1] == pytest.approx(12.80, abs=0.02)

    def test_carry_recovered_from_parity(self, synth):
        """Per-expiry q_T, r_T recovered from the chain (not assumed)."""
        carry = synth.spx_carry
        assert len(carry) == len(_EXPIRIES)
        long = carry[carry["T"] > 60 / 365]
        assert np.allclose(long["q"], _Q_TRUE, atol=1e-4)
        assert np.allclose(long["r"], _R_TRUE, atol=1e-4)

    def test_anchors_cover_the_put_wing(self, synth):
        """>= 3 puts with K/F < 0.97 in every expiry (pre-C22: 0 puts beyond
        the 15d slice on the 05-31 showcase; anchors were ATM calls)."""
        s = synth.spx_surface
        for exp, g in s.groupby("expiry"):
            kf = g["strike"] / g["forward"]
            assert ((g["right"] == "P") & (kf < 0.97)).sum() >= 3, exp
            assert ((g["right"] == "C") & (kf > 1.02)).sum() >= 2, exp
            # OTM against the FORWARD
            assert ((g["right"] == "C") == (g["strike"] >= g["forward"])).all()
            span = kf.max() / kf.min()
            T = g["time_to_expiry"].iloc[0]
            assert math.log(span) > 3 * g["implied_vol"].median() * math.sqrt(T)

    def test_one_expiry_per_bucket_no_duplicates(self, synth):
        s = synth.spx_surface
        assert s["expiry"].nunique() == 6
        assert not s.duplicated(["expiry", "strike", "right"]).any()
        assert len(s) <= 6 * 9

    def test_weights_normalised_independent_of_disabled_leg(self, synth):
        assert synth.w3 == 0.0
        assert synth.w1 + synth.w2 == pytest.approx(1.0)
        assert synth.w1 / synth.w2 == pytest.approx(JOINT_W1 / JOINT_W2)

    def test_reference_rate_is_the_stored_cc_rate(self, synth):
        """tbill_rates.rate is already continuously compounded. The first C22
        version converted it a second time (4.10% -> 4.18%, labelled as a
        discount yield)."""
        assert synth.r_reference == 0.0410
        assert synth.r_reference_source == (
            f"^IRX 13-week bill {_SESSION}: 4.0300% bank discount = 4.1000% cc")


def _vix_chain(n, snapshot_date=_SESSION):
    rows = []
    for i in range(n):
        for right in ("C", "P"):
            rows.append({"snapshot_date": snapshot_date, "expiry": "2026-10-21",
                         "strike": 12.0 + i, "right": right,
                         "bid": 1.0 + (right == "C") * (8 - i) * 0.3,
                         "ask": 1.2 + (right == "C") * (8 - i) * 0.3,
                         "mid_price": 1.1, "implied_vol": 0.9})
    return pd.DataFrame(rows)


class TestWeightsWithVixChain:
    def test_stale_vix_chain_leaves_the_leg_empty(self):
        """Pre-C22 every fit read the only VIX chain in the DB (2026-06-20)
        with its June time_to_expiry, and its row count toggled the weights."""
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp, vix_surface=_vix_chain(8, snapshot_date="2026-09-16"),
                      spx_dates=["2026-09-15", "2026-09-16", "2026-09-22", _SESSION])
            cal = JointCalibrator(as_of_date=_SESSION, spx_chain=_heston_chain(
                expiries=_EXPIRIES[:6]))
        assert cal.vix_options.empty

    def test_weights_identical_with_0_or_8_vix_options(self):
        """Pre-C22: 0.50/0.60 with >= 5 (stale) VIX options, 0.4545/0.5455
        otherwise — the disabled leg changed the effective Feller strength."""
        vix_chain = _vix_chain
        got = []
        for n in (0, 8):
            with pytest.MonkeyPatch.context() as mp:
                _patch_db(mp, vix_surface=vix_chain(n) if n else None)
                cal = JointCalibrator(as_of_date=_SESSION, spx_chain=_heston_chain(
                    expiries=_EXPIRIES[:6]))
            got.append((cal.w1, cal.w2, cal.w3, len(cal.vix_options)))
        assert got[1][3] >= 5, "the 8-strike chain must yield a non-sparse VIX leg"
        assert got[0][:3] == pytest.approx(got[1][:3])


class TestFellerMode:
    _VIOLATE = [0.5, 0.01, 0.3, -0.7, 0.01]      # 2*kappa*theta = 0.01 < sigma^2 = 0.09

    def test_off_adds_no_penalty(self, synth):
        """Mutation guard: with feller='off' the loss is exactly the data legs."""
        k, th, s, rho, v0 = self._VIOLATE
        legs = (synth.w1 * synth._spx_leg(k, th, s, rho, v0)
                + synth.w2 * synth._vix_index_leg(k, th, s, v0))
        assert synth.feller == "off"
        assert synth.joint_loss(self._VIOLATE) == legs

    def test_soft_adds_exactly_the_penalty(self, synth):
        k, th, s, rho, v0 = self._VIOLATE
        legs = (synth.w1 * synth._spx_leg(k, th, s, rho, v0)
                + synth.w2 * synth._vix_index_leg(k, th, s, v0))
        synth.feller = "soft"
        try:
            loss = synth.joint_loss(self._VIOLATE)
        finally:
            synth.feller = "off"
        assert loss - legs == pytest.approx(jc._FELLER_PENALTY * (s ** 2 - 2 * k * th) ** 2,
                                            rel=1e-12)

    def test_invalid_mode_rejected(self):
        with pytest.raises(ValueError, match="feller"):
            JointCalibrator("2026-09-23", feller="hard")


class TestBatesNestsHeston:
    def test_bates_loss_equals_heston_at_zero_intensity(self, synth):
        n0 = synth._n_evals
        b = synth._bates_joint_loss(np.array(_tvec() + [0.0, -0.05, 0.05]))
        assert synth._n_evals == n0 + 1          # evaluations are counted
        assert b == pytest.approx(synth.joint_loss(np.array(_tvec())), abs=1e-14)

    def test_bounds_contain_heston(self):
        lam, mu_j, sigma_j = JointCalibrator._BATES_JUMP_BOUNDS
        assert lam[0] == 0.0 and mu_j[0] < 0.0 < mu_j[1] and sigma_j[0] == 0.0


# ── Time to expiry, staleness, empty surfaces ────────────────────────────────

def _db_rows(chain: pd.DataFrame, snapshot_date: str) -> pd.DataFrame:
    """A cleaned chain as stored in options_snapshots (no root column, stored
    time_to_expiry measured from the snapshot's wall-clock date)."""
    df = chain.rename(columns={"expiration": "expiry"}).copy()
    df["right"] = df.pop("option_type").str[0].str.upper()
    df["snapshot_date"] = snapshot_date
    df["mid_price"] = 0.5 * (df["bid"] + df["ask"])
    df["implied_vol"] = df.pop("impliedVolatility")
    df["time_to_expiry"] = (pd.to_datetime(df["expiry"]) - pd.Timestamp(snapshot_date)).dt.days / 365
    return df.drop(columns=["contractSymbol", "lastPrice"])


class TestSessionTiming:
    def test_sunday_snapshot_measures_T_from_friday(self):
        """The 05-31 showcase snapshot was pulled on a Sunday with Friday's
        quotes; T must run from Friday's close (stored T was 2 days short)."""
        fri, sun = "2026-09-18", "2026-09-20"
        chain = _heston_chain(session=fri, expiries=["2026-10-02", "2026-10-16", "2026-11-20",
                                                     "2026-12-18", "2027-03-19", "2027-09-17"])
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp, session=fri, vix_row=_index_row(session=fri),
                      spx_surface=_db_rows(chain, sun),
                      spx_dates=["2026-09-16", "2026-09-17", fri])
            cal = JointCalibrator(as_of_date=sun)
        assert cal.spot_date == fri and cal.spx_snapshot_session == fri
        g = cal.spx_surface[cal.spx_surface["expiry"] == "2026-10-02"]
        assert g["time_to_expiry"].iloc[0] == pytest.approx(14 / 365)
        assert cal._spx_leg(**_TRUE) < 1e-10

    def test_stale_snapshot_rejected_by_shared_gate(self):
        old = "2026-09-16"
        chain = _heston_chain(session=old)
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp, spx_surface=_db_rows(chain, old),
                      spx_dates=["2026-09-15", old, "2026-09-22", _SESSION])
            cal = JointCalibrator(as_of_date=_SESSION)
        assert cal.spx_snapshot_session == old != cal.spot_date
        res = _fake_result(spx_snapshot_session=old, spx_snapshot_date=old)
        ok, reason = is_acceptable_calibration(res)
        assert not ok and "stale" in reason

    def test_empty_surface_raises_value_error(self):
        """Pre-C22: KeyError('time_to_expiry') from a log line."""
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp)
            with pytest.raises(ValueError, match="SPX anchors"):
                JointCalibrator(as_of_date=_SESSION)

    def test_spx_leg_never_returns_zero_without_anchors(self, synth):
        """A weighted data leg with no data must raise, not return 0 (which
        would turn the 'joint' fit into a VIX-only fit)."""
        orig = synth.spx_surface
        try:
            synth.spx_surface = orig.iloc[0:0]
            with pytest.raises(ValueError, match="no anchors"):
                synth._spx_leg(**_TRUE)
        finally:
            synth.spx_surface = orig

    def test_empty_book_raises_value_error(self):
        chain = _heston_chain()
        chain["bid"] = chain["ask"] = 0.0
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp)
            with pytest.raises(ValueError, match="no live two-sided quotes"):
                JointCalibrator(as_of_date=_SESSION, spx_chain=chain)


# ── Full calibration on the synthetic market ─────────────────────────────────

@pytest.fixture(scope="module")
def synth_fit():
    with pytest.MonkeyPatch.context() as mp:
        _patch_db(mp)
        cal = JointCalibrator(as_of_date=_SESSION, spx_chain=_heston_chain())
        res = cal.calibrate(de_maxiter=40, de_popsize=8, verbose=False)
    return cal, res


class TestVxFuturesOutOfSample:
    """Real VX futures are scored at their SETTLEMENT (database ``price`` =
    COALESCE(settle, close)), not the last trade: on 2026-03-24 the front
    closes sat 0.4-0.6 pts from the settles, and a contract with no trade that
    day (no close) was dropped."""

    def _cal(self, fut):
        with pytest.MonkeyPatch.context() as mp:
            _patch_db(mp)
            mp.setattr(jc.db, "get_vix_futures_curve", lambda as_of_date: fut)
            return JointCalibrator(as_of_date=_SESSION, spx_chain=_heston_chain())

    def test_scored_at_settlement_and_untraded_contracts_kept(self):
        fut = pd.DataFrame({
            "expiry_month": ["2026-10", "2026-11", "2026-12"],
            "settle": [24.94, 24.15, 23.85],
            "close": [24.32, 23.62, np.nan],          # December did not trade
            "days_to_expiry": [22, 57, 85],
        })
        fut["price"] = fut["settle"].fillna(fut["close"])
        cal = self._cal(fut)
        assert len(cal.vx_futures) == 3
        k, th, sg, v0 = 3.0, 0.04, 0.6, 0.03
        model = heston_vix_future_exact(k, th, sg, v0, fut["days_to_expiry"].to_numpy(float) / 365.0)
        want = float(np.sqrt(np.mean((np.atleast_1d(model) - fut["settle"].to_numpy()) ** 2)))
        assert cal._vx_futures_oos(k, th, sg, v0) == pytest.approx(want, rel=1e-12)

    def test_frame_without_price_column_falls_back_to_settle_then_close(self):
        fut = pd.DataFrame({"expiry_month": ["2026-10", "2026-11"], "settle": [np.nan, 24.15],
                            "close": [24.32, 23.62], "days_to_expiry": [22, 57]})
        cal = self._cal(fut)
        assert cal.vx_futures["price"].tolist() == [24.32, 24.15]


class TestSyntheticCalibration:
    def test_recovers_generating_parameters(self, synth_fit):
        _, res = synth_fit
        p = res["params"]
        assert res["leg_losses"]["spx_iv_rmse"] < 0.05
        assert res["leg_losses"]["vix_index_rmse"] < 0.05
        for k, tol in (("kappa", 0.15), ("theta", 0.05), ("sigma", 0.05), ("rho", 0.03),
                       ("v0", 0.05)):
            assert abs(p[k] - _TRUE[k]) <= tol * abs(_TRUE[k]), (k, p[k])

    def test_out_of_sample_reported_and_small(self, synth_fit):
        _, res = synth_fit
        ll = res["leg_losses"]
        assert res["oos"]["n_spx_oos"] > 500
        assert ll["spx_oos_iv_rmse"] < 0.2
        assert np.isfinite(ll["spx_oos_put_wing_rmse"])

    def test_result_provenance(self, synth_fit):
        cal, res = synth_fit
        assert res["model_version"] == 2 and not calibration_is_legacy(res)
        assert res["n_spx_expiries"] == 6 and res["n_spx_options"] >= 48
        assert res["leg_losses"]["vix_futures_rmse"] == \
            res["leg_losses"]["vix_index_rmse"]           # back-compat alias
        assert res["r"] == pytest.approx(_R_TRUE, abs=2e-3)
        assert res["q"] == pytest.approx(_Q_TRUE, abs=2e-3)
        # r stored == r used: medians of the per-expiry discount rates the leg priced off
        used = cal.spx_surface.groupby("expiry")["discount"].first()
        T = cal.spx_surface.groupby("expiry")["time_to_expiry"].first()
        assert res["r"] == pytest.approx(float(np.median(-np.log(used) / T)))
        assert res["feller"] == "off" and "feller" not in res["pinned_params"]
        assert res["methods"]["optimizer"]["scipy_polish"] is False
        assert res["spx_snapshot_session"] == res["spot_date"] == _SESSION
        ok, reason = is_acceptable_calibration(res)
        assert ok, reason

    def test_validate_matches_loss(self, synth_fit):
        """validate() is measured on the same IVs as the objective."""
        cal, res = synth_fit
        rep = cal.validate()
        assert rep["smile_rmse_vol_pts"] == pytest.approx(res["leg_losses"]["spx_iv_rmse"], abs=0.05)
        assert rep["vix_index_rmse_pts"] == pytest.approx(res["leg_losses"]["vix_index_rmse"])

    def test_save_persists_the_result_dict(self, synth_fit, tmp_path):
        cal, res = synth_fit
        path = cal.save(str(tmp_path / "c.pkl"))
        loaded = JointCalibrator.load_params(path)
        assert loaded["params"] == res["params"]
        assert loaded["leg_losses"] == res["leg_losses"]
        assert loaded["model_version"] == 2


# ── Quality gate ──────────────────────────────────────────────────────────────

def _fake_result(**over):
    res = {
        "model_version": 2,
        "params": {"kappa": 5.2, "theta": 0.05, "sigma": 0.7, "rho": -0.9, "v0": 0.015},
        "leg_losses": {"spx_iv_rmse": 1.4, "vix_index_rmse": 0.6, "vix_futures_rmse": 0.6},
        "n_spx_expiries": 6, "n_spx_options": 54, "n_put_wing_expiries": 6,
        "n_vix_tenors": 4, "spot_date": "2026-09-23",
        "spx_snapshot_date": "2026-09-23", "spx_snapshot_session": "2026-09-23",
    }
    res.update(over)
    return res


class TestCalibrationQualityGate:
    """is_acceptable_calibration() rejects degenerate, under-identified or
    stale fits."""

    def test_full_result_accepted(self):
        assert is_acceptable_calibration(_fake_result()) == (True, "ok")

    def test_single_expiry_rejected(self):
        """Every production fit 2026-09-08..09-23 used ONE expiry (9 options)."""
        ok, reason = is_acceptable_calibration(_fake_result(n_spx_expiries=1, n_spx_options=9))
        assert not ok and "term structure under-identified" in reason

    def test_too_few_options_rejected(self):
        ok, reason = is_acceptable_calibration(_fake_result(n_spx_options=20))
        assert not ok and "options" in reason

    def test_missing_put_wing_rejected(self):
        ok, reason = is_acceptable_calibration(_fake_result(n_put_wing_expiries=0))
        assert not ok and "put wing" in reason

    def test_stale_surface_rejected(self):
        ok, reason = is_acceptable_calibration(_fake_result(spx_snapshot_session="2026-06-18",
                                                            spx_snapshot_date="2026-06-20"))
        assert not ok and "stale" in reason

    def test_staleness_is_the_reason_reported_first(self):
        """A months-old chain seen from a later session has few unexpired
        slices left (the 2026-06-18 chain at 2026-10-16: 3 expiries); the
        reason must name the staleness, not the symptom."""
        ok, reason = is_acceptable_calibration(_fake_result(
            spx_snapshot_session="2026-06-18", spx_snapshot_date="2026-06-18",
            spot_date="2026-10-16", n_spx_expiries=3, n_spx_options=26))
        assert not ok and "stale" in reason

    def test_legacy_result_rejected_unless_allowed(self):
        legacy = {"params": {"kappa": 5.0, "theta": 0.0431, "sigma": 0.657, "rho": -0.95,
                             "v0": 0.01088},
                  "leg_losses": {"spx_iv_rmse": 0.215, "vix_futures_rmse": 0.27},
                  "n_spx_options": 9, "n_vix_tenors": 4, "spot_date": "2026-09-11",
                  "spx_snapshot_date": "2026-09-11"}
        assert calibration_is_legacy(legacy)
        ok, reason = is_acceptable_calibration(legacy)
        assert not ok and "legacy" in reason
        ok, reason = is_acceptable_calibration(legacy, allow_legacy=True)
        assert not ok and "options" in reason      # the 9-option 09-11 headline
        # 05-31 showcase: Sunday snapshot holding Friday's session — fresh
        showcase = dict(legacy, n_spx_options=54, spot_date="2026-05-29",
                        spx_snapshot_date="2026-05-31",
                        params={"kappa": 1.77, "theta": 0.0666, "sigma": 0.46,
                                "rho": -0.95, "v0": 0.0127})
        assert is_acceptable_calibration(showcase, allow_legacy=True)[0]
        no_prov = {k: v for k, v in showcase.items() if k != "spx_snapshot_date"}
        assert not is_acceptable_calibration(no_prov, allow_legacy=True)[0]

    # Positional (pre-C22) call form stays backward compatible
    def test_good_local_fits_pass(self):
        assert is_acceptable_calibration({"kappa": 1.77, "sigma": 0.46, "rho": -0.95},
                                         spx_iv_rmse=0.52)[0]

    def test_vol_of_vol_collapse_rejected(self):
        ok, reason = is_acceptable_calibration(
            {"kappa": 20.0, "sigma": 0.0015, "rho": -0.0003}, spx_iv_rmse=5.74)
        assert not ok and "sigma" in reason

    def test_no_skew_rejected(self):
        ok, reason = is_acceptable_calibration({"kappa": 2.0, "sigma": 0.4, "rho": -0.02}, 1.0)
        assert not ok and "skew" in reason

    def test_kappa_pinned_rejected(self):
        ok, reason = is_acceptable_calibration({"kappa": 0.10, "sigma": 0.3, "rho": -0.5}, 1.0)
        assert not ok and "kappa" in reason

    def test_v0_at_floor_rejected(self):
        ok, reason = is_acceptable_calibration(
            {"kappa": 2.0, "sigma": 0.4, "rho": -0.6, "theta": 0.05, "v0": 1.5e-4}, 1.0)
        assert not ok and "v0" in reason

    def test_high_rmse_rejected(self):
        ok, reason = is_acceptable_calibration({"kappa": 2.0, "sigma": 0.4, "rho": -0.6}, 4.5)
        assert not ok and "RMSE" in reason

    def test_single_vix_tenor_rejected(self):
        """C20: a 1-tenor VIX leg cannot identify the joint fit (07-23: 0.049vp)."""
        healthy = {"kappa": 2.26, "sigma": 0.474, "rho": -0.95}
        ok, reason = is_acceptable_calibration(healthy, spx_iv_rmse=0.049, n_vix_tenors=1)
        assert not ok and "tenor" in reason.lower()
        assert is_acceptable_calibration(healthy, spx_iv_rmse=0.049, n_vix_tenors=4)[0]

    def test_tenor_check_skipped_when_unknown(self):
        assert is_acceptable_calibration({"kappa": 1.77, "sigma": 0.46, "rho": -0.95},
                                         spx_iv_rmse=0.52, n_vix_tenors=None)[0]


class TestFellerClassification:
    """Feller is a diagnostic: PASS / BINDING / VIOLATED."""

    def test_clear_pass(self):
        state, margin = classify_feller(1.77, 0.0666, 0.460)
        assert state == "PASS" and margin > 0

    def test_binding_not_reported_as_failure(self):
        kappa, theta, sigma = 4.022199139627185, 0.04630341824281701, 0.6103173509096278
        assert 2 * kappa * theta < sigma**2
        state, margin = classify_feller(kappa, theta, sigma)
        assert state == "BINDING" and -1e-4 < margin < 0

    def test_genuine_violation_still_flagged(self):
        state, margin = classify_feller(4.6217, 0.07637, 0.84115)
        assert state == "VIOLATED" and margin < -1e-4

    def test_states_are_exhaustive_and_ordered(self):
        kappa, theta = 2.0, 0.05
        ceiling = (2 * kappa * theta) ** 0.5
        seen = [classify_feller(kappa, theta, s)[0] for s in (ceiling * 0.9, ceiling, ceiling * 1.1)]
        assert seen == ["PASS", "BINDING", "VIOLATED"]


class TestActiveConstraints:
    def test_rho_at_floor_is_pinned(self):
        assert "rho" in count_pinned_params(
            {"kappa": 2.2, "theta": 0.058, "sigma": 0.43, "rho": -0.95, "v0": 0.016})

    def test_interior_params_not_pinned(self):
        assert count_pinned_params(
            {"kappa": 2.2, "theta": 0.058, "sigma": 0.43, "rho": -0.60, "v0": 0.016}) == []

    def test_v0_tolerance_is_relative_to_the_bound(self):
        """Pre-C22 the tolerance was 1e-3 x the [1e-4, 1] span, so v0 = 1.04e-3
        (10x the floor) was 'pinned' on the 08-07/08-29 fits."""
        base = {"kappa": 2.2, "theta": 0.058, "sigma": 0.43, "rho": -0.60}
        assert count_pinned_params({**base, "v0": 1.0366e-3}) == []
        assert count_pinned_params({**base, "v0": 1.0001e-4}) == ["v0"]

    def test_feller_binding_counts_as_one_constraint_not_sigma(self):
        params = {"kappa": 4.02, "theta": 0.0463, "sigma": 0.6103, "rho": -0.60, "v0": 0.0053}
        assert count_pinned_params(params) == []
        assert count_pinned_params(params, feller_state="BINDING") == ["feller"]
        assert count_pinned_params(params, "BINDING", feller_imposed=False) == []

    def test_active_constraint_messages(self):
        p = {"kappa": 4.02, "theta": 0.0463, "sigma": 0.6103, "rho": -0.95, "v0": 0.0053}
        msgs = active_constraints(p, "BINDING", feller_imposed=True)
        assert msgs == ["rho = -0.95 (box floor)",
                        "2*kappa*theta >= sigma^2 (Feller, imposed, binding)"]
        assert active_constraints(p, "BINDING", feller_imposed=False) == ["rho = -0.95 (box floor)"]


# ── Real snapshot (DB): 2026-03-24 ────────────────────────────────────────────

@pytest.fixture(scope="module")
def real_cal():
    return JointCalibrator(as_of_date="2026-03-24")


class TestRealSnapshot:
    @pytest.fixture
    def cal(self, real_cal):
        return real_cal

    def test_surface_filters(self, cal):
        s = cal.spx_surface
        kf = s["strike"] / s["forward"]
        assert kf.between(0.70, 1.30).all()
        assert s["implied_vol"].between(0.01, 2.0).all()
        assert (s["bs_vega"] > 0).all() and (s["market_price_f"] > 0).all()
        assert s["expiry"].nunique() == 6
        assert cal.spx_snapshot_session == cal.spot_date == "2026-03-24"

    def test_put_wing_present_in_every_expiry(self, cal):
        s = cal.spx_surface
        for exp, g in s.groupby("expiry"):
            assert ((g["right"] == "P") & (g["strike"] / g["forward"] < 0.97)).sum() >= 3, exp

    def test_vix_indices_on_cboe_windows(self, cal):
        assert len(cal.vix_ts) == 4
        assert sorted(np.round(cal.vix_ts["tenor_years"] * 365)) == [9, 30, 93, 184]

    def test_loss_ranks_params(self, cal):
        p_bad = np.array([0.2, 0.002, 0.1, -0.1, 0.002])
        p_good = np.array([5.0, 0.065, 1.5, -0.8, 0.07])
        assert np.isfinite(cal.joint_loss(p_good)) and cal.joint_loss(p_good) < cal.joint_loss(p_bad)
        assert cal.joint_loss([25.0, 0.04, 0.5, -0.7, 0.04]) == 1e6
        assert cal.joint_loss([2.0, 0.04, 0.5, 0.1, 0.04]) == 1e6


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
