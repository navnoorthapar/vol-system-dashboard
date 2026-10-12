"""
test_backtest_accounting.py — C22 regression tests for the backtest engine.

Each test pins one defect from the C22 audit and exercises the engine code
itself (no source greps, no re-implemented logic):

  hedge costs     opening hedge + closing unwind, no phantom exit rebalance
  financing       r on the cash account (−V + ΔS), q on the stock leg;
                  hedged P&L is unbiased in a Black-Scholes world
  expiry / roll   straddles settle at expiry and roll; no T_rem floor
  rate gaps       previous ^IRX print, negative yields kept
  bookkeeping     open trade logged; Σ trade net P&L == sleeve P&L;
                  sizing off the sleeve's running NAV
  S3              proxy exit day; index leg priced as a short straddle;
                  random-walk null test of the timing rule
  sizing overlay  correlation netting reduces size when sleeves co-move
  metrics         Sharpe on excess P&L (rf = 0)
  ATM vol         σ_ATM = β·VIX; β estimator recovers a known 30-day vol
  sub-periods     no classifier re-training
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.backtest.backtest_engine import (
    CONTRACT_MULT,
    SLIPPAGE_PCT,
    TRADING_DAYS,
    BacktestEngine,
    C22Corrections,
    _simulate_s3_pnl,
    _simulate_straddle_pnl,
)
from joint_vol_calibration.backtest.delta_hedger import (
    _bs_straddle_greeks,
    _bs_straddle_value,
)
from joint_vol_calibration.tests.test_backtest_engine import _make_spx, _make_vix

FIXTURES = Path(__file__).parent / "fixtures"
T30 = 30 / 365.0
Q = 0.013
VIX_COLS = ["^VIX9D", "^VIX", "^VIX3M", "^VIX6M"]


def _market(S, vix: float = 20.0, start: str = "2021-01-04"):
    S = np.asarray(S, dtype=float)
    dates = pd.bdate_range(start, periods=len(S))
    spx = pd.Series(S, index=dates)
    vix_df = pd.DataFrame({c: vix for c in VIX_COLS}, index=dates)
    return dates, spx, vix_df


def _series(dates, values) -> pd.Series:
    return pd.Series(np.asarray(values, dtype=float), index=dates)


def _sim(pos, kelly, spx, vix_df, nav=1_000_000.0, **kw):
    dates = spx.index
    return _simulate_straddle_pnl(
        _series(dates, pos), _series(dates, kelly), _series(dates, [nav] * len(dates)),
        spx, vix_df, **kw,
    )


def _rw_market(n: int = 700, seed: int = 21, start: str = "2017-01-02"):
    """Synthetic SPX/VIX where log(VIX/VVIX) is a random walk (as on real
    data, 3%/day) and VVIX stays below the fixed R2 gate."""
    spx = _make_spx(n, start=start, seed=seed)
    rng = np.random.default_rng(seed + 1)
    vvix = np.clip(88 * np.exp(np.cumsum(rng.normal(0, 0.01, n))), 70, 99)
    vix = np.clip(vvix * np.exp(np.log(0.2) + np.cumsum(rng.normal(0, 0.03, n))), 9, 60)
    return spx, pd.DataFrame({"date": spx["date"].values, "^VIX": vix, "^VIX9D": vix * 0.95,
                              "^VIX3M": vix * 1.05, "^VIX6M": vix * 1.08, "^VVIX": vvix})


def _path(T_sched, S, r, sigma, K):
    """BS value and delta along a T schedule (independent of the engine loop)."""
    V = [_bs_straddle_value(s, K, T, r, Q, sigma) for s, T in zip(S, T_sched)]
    D = [_bs_straddle_greeks(s, K, T, r, Q, sigma)[0] for s, T in zip(S, T_sched)]
    return np.array(V), np.array(D)


# ── hedge costs ────────────────────────────────────────────────────────────────

class TestHedgeCosts:
    def test_open_rebalance_and_unwind_charged_once_each(self):
        """5 bp × S × n × 100 × (|Δ0| + Σ|ΔΔ| + |Δ_last|), by hand.

        Pre-C22 the opening hedge and the closing unwind were free and the
        exit day paid a rebalance to Δ_t that a closing trader never makes."""
        S = np.full(10, 4000.0)
        dates, spx, vix = _market(S)
        pos = [0, 1, 1, 1, 1, 1, 0, 0, 0, 0]          # entry day 1, exit day 6
        pnl, trades = _sim(pos, [0.05] * 10, spx, vix, r=0.03, q=Q, atm_to_vix_ratio=1.0)
        (t,) = trades
        T_sched = [T30 - j / TRADING_DAYS for j in range(5)]   # entry + 4 holding marks
        _, D = _path(T_sched, S[:5], 0.03, 0.20, 4000.0)
        shares = abs(D[0]) + np.abs(np.diff(D)).sum() + abs(D[-1])
        expected = SLIPPAGE_PCT * shares * 4000.0 * t.n_contracts * CONTRACT_MULT
        assert t.slippage_cost == pytest.approx(expected, rel=1e-12)
        assert t.exit_reason == "signal"

    def test_entry_day_pays_opening_hedge(self):
        S = np.full(6, 4000.0)
        dates, spx, vix = _market(S)
        pnl, (t,) = _sim([0, 1, 1, 0, 0, 0], [0.05] * 6, spx, vix, r=0.03, q=Q)
        D0 = _bs_straddle_greeks(4000.0, 4000.0, T30, 0.03, Q, bte.ATM_TO_VIX_RATIO * 0.20)[0]
        hedge0 = SLIPPAGE_PCT * abs(D0) * 4000.0 * t.n_contracts * CONTRACT_MULT
        assert pnl.iloc[1] == pytest.approx(-(t.entry_cost + hedge0), rel=1e-12)


# ── financing ──────────────────────────────────────────────────────────────────

class TestFinancing:
    def test_trade_financing_matches_formula(self):
        S = np.array([4000, 4000, 4040, 3990, 4010, 4010, 4010], dtype=float)
        dates, spx, vix = _market(S)
        r = _series(dates, [0.02, 0.03, 0.04, 0.05, 0.05, 0.05, 0.05])
        pos = [0, -1, -1, -1, -1, 0, 0]               # short: entry day 1, exit day 5
        pnl, (t,) = _sim(pos, [0.05] * 7, spx, vix, r=r, q=Q, atm_to_vix_ratio=1.0)
        idx = range(1, 5)                              # days whose close feeds the next day
        T_sched = [T30 - j / TRADING_DAYS for j in range(4)]
        V, D = [], []
        for j, i in enumerate(idx):
            V.append(_bs_straddle_value(S[i], 4000.0, T_sched[j], r.iloc[i], Q, 0.20))
            D.append(_bs_straddle_greeks(S[i], 4000.0, T_sched[j], r.iloc[i], Q, 0.20)[0])
        fin = sum(-1 * (r.iloc[i] * (D[j] * S[i] - V[j]) - Q * D[j] * S[i]) / TRADING_DAYS
                  for j, i in enumerate(idx)) * t.n_contracts * CONTRACT_MULT
        assert t.financing == pytest.approx(fin, rel=1e-10)
        assert t.financing != 0.0
        assert pnl.sum() == pytest.approx(t.net_pnl, rel=1e-12)

    def test_hedged_pnl_unbiased_in_black_scholes_world(self, monkeypatch):
        """Realised vol = implied vol under Q, no trading costs: the mean
        hedged P&L of a self-financing straddle book is ~0. Without the
        carry term it drifts by about −r(V − ΔS) per day (here −$900/path)."""
        for name in ("BID_ASK_VOL_PTS", "COMMISSION_PER_LEG", "SLIPPAGE_PCT"):
            monkeypatch.setattr(bte, name, 0.0)
        r, sig, n_days, n_paths = 0.30, 0.20, 16, 400
        rng = np.random.default_rng(20261012)
        dt = 1 / TRADING_DAYS
        totals = []
        pos = [1] * (n_days - 1) + [0]
        for _ in range(n_paths):
            z = rng.standard_normal(n_days - 1)
            S = 4000 * np.exp(np.r_[0, np.cumsum((r - 0.5 * sig ** 2) * dt + sig * np.sqrt(dt) * z)])
            dates, spx, vix = _market(S)
            pnl, _ = _sim(pos, [0.05] * n_days, spx, vix, r=r, q=0.0, atm_to_vix_ratio=1.0)
            totals.append(pnl.sum())
        totals = np.array(totals)
        se = totals.std(ddof=1) / np.sqrt(n_paths)
        assert abs(totals.mean()) < 3 * se, (totals.mean(), se)


# ── expiry / roll ──────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def held_60():
    rng = np.random.default_rng(4)
    S = 4000 * np.exp(np.cumsum(rng.normal(0, 0.01, 70)))
    dates, spx, vix = _market(S)
    pos = [0] + [-1] * 60 + [0] * 9
    pnl, trades = _sim(pos, [0.05] * 70, spx, vix, r=0.03, q=Q)
    return dates, spx, pnl, trades


class TestExpiryRoll:
    def test_rolls_into_fresh_atm_straddles(self, held_60):
        dates, spx, pnl, trades = held_60
        assert len(trades) >= 3
        assert [t.exit_reason for t in trades[:2]] == ["expiry", "expiry"]
        assert len({t.K for t in trades}) == len(trades), "each roll re-strikes ATM"
        for prev, nxt in zip(trades, trades[1:]):
            assert nxt.entry_date == prev.exit_date
            assert nxt.K == pytest.approx(float(spx.loc[nxt.entry_date]))

    def test_no_trade_outlives_its_expiry(self, held_60):
        dates, _, _, trades = held_60
        n_expiry = int(np.ceil(T30 * TRADING_DAYS))
        for t in trades:
            sessions = dates.get_loc(t.exit_date) - dates.get_loc(t.entry_date)
            assert sessions <= n_expiry

    def test_expiry_settles_at_intrinsic_without_exit_cost(self, held_60):
        """ΔV telescopes, so the first trade's gross P&L is
        −n·100·[(|S_exp − K| − V0) − Σ Δ_{j−1}·ΔS_j]: settlement at intrinsic."""
        dates, spx, pnl, trades = held_60
        t = trades[0]
        assert t.exit_cost == 0.0
        i0, i1 = dates.get_loc(t.entry_date), dates.get_loc(t.exit_date)
        S = spx.values[i0:i1 + 1]
        T_sched = [T30 - j / TRADING_DAYS for j in range(len(S) - 1)]
        V, D = _path(T_sched, S[:-1], 0.03, t.sigma_entry, t.K)
        hedge = (D * np.diff(S)).sum()
        expected = -t.n_contracts * CONTRACT_MULT * ((abs(S[-1] - t.K) - V[0]) - hedge)
        # σ is re-read from the (flat) VIX row on each mark; with β·20% flat it
        # equals sigma_entry, so the hand path is exact.
        assert t.gross_pnl == pytest.approx(expected, rel=1e-9)

    def test_trades_sum_to_sleeve_pnl(self, held_60):
        _, _, pnl, trades = held_60
        assert sum(t.net_pnl for t in trades) == pytest.approx(pnl.sum(), rel=1e-12, abs=1e-6)


# ── rates ──────────────────────────────────────────────────────────────────────

class TestRateGaps:
    def test_gap_uses_previous_print_and_keeps_negative_yields(self):
        dates = pd.bdate_range("2020-03-16", periods=8)
        r = pd.Series([0.003, -0.0005, np.nan, np.nan, 0.001, np.nan, 0.002, 0.002], index=dates).dropna()
        out = bte._resolve_rates(r, dates, ffill=True)
        np.testing.assert_allclose(out, [0.003, -0.0005, -0.0005, -0.0005, 0.001, 0.001, 0.002, 0.002])

    def test_simulation_on_gappy_series_equals_filled_series(self):
        S = np.array([4000, 4010, 3990, 4005, 4020, 4000, 4000, 4000], dtype=float)
        dates, spx, vix = _market(S)
        full = _series(dates, [0.01, 0.01, 0.002, 0.002, 0.002, 0.03, 0.03, 0.03])
        gappy = full.drop(dates[[3, 4]])
        pos = [1, 1, 1, 1, 1, 1, 0, 0]
        a, _ = _sim(pos, [0.05] * 8, spx, vix, r=full, q=Q)
        b, _ = _sim(pos, [0.05] * 8, spx, vix, r=gappy, q=Q)
        pd.testing.assert_series_equal(a, b)


# ── bookkeeping and sizing ─────────────────────────────────────────────────────

class TestBookkeeping:
    def test_open_trade_logged_at_end(self):
        S = np.full(8, 4000.0)
        dates, spx, vix = _market(S)
        pnl, trades = _sim([0, 0, 0, 0, 1, 1, 1, 1], [0.05] * 8, spx, vix, r=0.03, q=Q)
        (t,) = trades
        assert t.is_open and t.exit_reason == "open" and t.exit_date == dates[-1]
        assert t.exit_cost == 0.0
        assert t.net_pnl == pytest.approx(pnl.sum(), rel=1e-12)

    def test_sizing_and_margin_use_running_sleeve_nav(self):
        S = np.array([4000, 4000, 4000, 4400, 4400, 4400, 4400, 4400, 4400, 4400], dtype=float)
        dates, spx, vix = _market(S)
        pos = [0, -1, -1, -1, 0, 0, -1, -1, 0, 0]
        k = 0.2
        pnl, trades = _sim(pos, [k] * 10, spx, vix, r=0.03, q=Q)
        first, second = trades
        nav_at_second = 1_000_000.0 + pnl.iloc[:6].sum()
        assert nav_at_second < 900_000.0, "the first short lost on the 10% jump"
        sigma = bte.ATM_TO_VIX_RATIO * 0.20
        V0 = _bs_straddle_value(4400.0, 4400.0, T30, 0.03, Q, sigma)
        n_kelly = int(k * nav_at_second / (V0 * CONTRACT_MULT))
        n_margin = int(nav_at_second / (bte.MARGIN_RATIO * 4400.0 * CONTRACT_MULT))
        assert second.n_contracts == max(1, min(n_kelly, n_margin))
        assert second.n_contracts < first.n_contracts


# ── S3 ─────────────────────────────────────────────────────────────────────────

@pytest.fixture(scope="module")
def engine():
    spx, vix = _rw_market()
    eng = BacktestEngine(start_date="2018-01-02", end_date="2019-09-30")
    eng.run(spx, vix)
    eng.compute_metrics()
    return eng


class TestS3:
    def test_proxy_exit_day_books_dz_and_cost_at_entry_size(self):
        dates = pd.bdate_range("2021-01-04", periods=8)
        pos = _series(dates, [0, 1, 1, 1, 0, 0, 0, 0])
        kelly = _series(dates, [0, 0.10, 0.10, 0.10, 0, 0, 0, 0])   # signal engine: 0 once flat
        nav = _series(dates, [1_000_000.0] * 8)
        z = _series(dates, [-1.2, -1.2, -1.1, -1.0, 0.0, 0.0, 0.0, 0.0])  # +1 on the exit day
        pnl, (t,) = _simulate_s3_pnl(pos, kelly, nav, z)
        notional = 0.10 * 1_000_000.0
        assert pnl.iloc[4] == pytest.approx(notional * 1.0 * bte.S3_Z_SCALE
                                            - bte.S3_PROXY_COST * notional)
        assert t.exit_cost == pytest.approx(bte.S3_PROXY_COST * notional)
        assert t.kelly_frac == pytest.approx(0.10)

    def test_index_leg_is_a_short_spx_straddle(self, engine):
        s3 = [t for t in engine.all_trades if t.signal == "s3"]
        assert engine.signals_df["s3_position"].abs().sum() > 0
        assert s3, "S3 traded in the synthetic sample"
        for t in s3:
            assert t.direction == -1 and t.K > 0 and t.n_contracts >= 1
        assert sum(t.net_pnl for t in s3) == pytest.approx(engine.equity_df["pnl_s3"].sum(), rel=1e-9)

    def test_null_test_exposes_the_z_proxy_bias(self, engine):
        """Random walks carry no information, yet the pre-C22 z-unit proxy
        books positive P&L on them: its mean is significantly > 0."""
        res = bte.s3_timing_null_test(engine, n_paths=30, seed=7)
        z = res["z_proxy"]
        assert z["null_mean"] > 3 * z["null_std"] / np.sqrt(res["n_paths"]), z
        assert z["null_frac_positive"] >= 0.6
        for key in ("index_leg", "z_proxy"):
            assert 0 < res[key]["p_value"] <= 1
        assert res["null_mean_trades"] > 0

    def test_null_test_is_reproducible(self, engine):
        a = bte.s3_timing_null_test(engine, n_paths=4, seed=3, method="permutation")
        b = bte.s3_timing_null_test(engine, n_paths=4, seed=3, method="permutation")
        assert a == b
        with pytest.raises(ValueError):
            bte.s3_timing_null_test(engine, n_paths=1, method="bootstrap")


# ── sizing overlay ─────────────────────────────────────────────────────────────

class TestCorrelationNetting:
    def _mat(self, cols):
        idx = pd.bdate_range("2021-01-04", periods=len(next(iter(cols.values()))))
        return pd.DataFrame(cols, index=idx)

    def test_identical_sleeves_are_cut(self):
        x = np.random.default_rng(1).normal(size=200)
        m = bte._corr_netting_multipliers(self._mat({"a": x, "b": x, "c": x}))
        assert m.iloc[-1].values == pytest.approx([1 / np.sqrt(3)] * 3)

    def test_anticorrelated_sleeves_keep_full_size(self):
        x = np.random.default_rng(2).normal(size=200)
        m = bte._corr_netting_multipliers(self._mat({"a": x, "b": -x}))
        assert (m.iloc[25:] == 1.0).all().all()

    def test_flat_sleeve_counts_as_uncorrelated(self):
        x = np.random.default_rng(3).normal(size=200)
        m = bte._corr_netting_multipliers(self._mat({"a": x, "b": np.zeros(200)}))
        assert (m == 1.0).all().all()

    def test_legacy_multiplier_could_never_reduce_size(self):
        x = np.random.default_rng(4).normal(size=200)
        for y in (x, -x, np.random.default_rng(5).normal(size=200)):
            legacy = bte._legacy_corr_kelly_mult(pd.Series(x), pd.Series(y))
            assert (legacy.clip(0.5, 1.0) == 1.0).all()


# ── metrics ────────────────────────────────────────────────────────────────────

class TestSharpeConvention:
    def _nav(self, rets):
        idx = pd.bdate_range("2021-01-04", periods=len(rets) + 1)
        return pd.DataFrame({"nav": 1e6 * np.cumprod(np.r_[1.0, 1.0 + np.asarray(rets)])}, index=idx)

    def test_zero_mean_pnl_has_zero_sharpe(self):
        rets = np.tile([0.001, -0.001], 126)
        assert abs(bte.compute_metrics(self._nav(rets))["sharpe"]) < 0.1

    def test_engine_does_not_charge_tbills_on_excess_pnl(self):
        rets = np.random.default_rng(9).normal(0.0002, 0.002, 300)
        df = self._nav(rets)
        eng = BacktestEngine()
        eng.equity_df = df
        eng._rate_series = pd.Series(0.05, index=df.index)
        m = eng.compute_metrics()
        assert m["sharpe"] == pytest.approx(bte.compute_metrics(df, rf=0.0)["sharpe"])
        assert m["sharpe"] > bte.compute_metrics(df, rf_series=eng._rate_series)["sharpe"] + 1.0

    def test_crisis_sharpe_uses_the_same_convention(self):
        rets = np.random.default_rng(10).normal(0.0, 0.003, 600)
        df = self._nav(rets)
        df.index = pd.bdate_range("2019-06-03", periods=len(df))
        cp = bte.compute_crisis_performance(df)["COVID_2020"]
        sub = df.loc["2020-02-19":"2020-04-30"]
        assert cp["sharpe"] == pytest.approx(bte.compute_metrics(sub)["sharpe"])


# ── ATM vol = β · VIX ──────────────────────────────────────────────────────────

class TestAtmVol:
    def test_entry_vol_is_beta_times_vix(self):
        dates, spx, vix = _market(np.full(5, 4000.0), vix=25.0)
        _, (t,) = _sim([0, 1, 1, 0, 0], [0.05] * 5, spx, vix, r=0.03, q=Q, atm_to_vix_ratio=0.8)
        assert t.sigma_entry == pytest.approx(0.8 * 0.25)
        _, (t_default,) = _sim([0, 1, 1, 0, 0], [0.05] * 5, spx, vix, r=0.03, q=Q)
        assert t_default.sigma_entry == pytest.approx(bte.ATM_TO_VIX_RATIO * 0.25)

    def test_default_beta_is_below_one(self):
        """VIX is a variance-swap rate (skew premium included); ATM vol is lower."""
        assert 0.70 <= bte.ATM_TO_VIX_RATIO <= 0.90
        assert BacktestEngine().atm_to_vix_ratio == bte.ATM_TO_VIX_RATIO


def _synthetic_chain(spot, r, as_of, expiries_days, sigma_of_T):
    rows = []
    for d in expiries_days:
        expiry = (as_of + pd.Timedelta(days=d)).normalize()
        T = ((expiry + pd.Timedelta(hours=16)) - as_of).total_seconds() / (365 * 86400)
        DF, F = np.exp(-r * T), spot * np.exp((r - Q) * T)
        for K in np.arange(round(F, -1) - 100, round(F, -1) + 105, 5.0):
            for right in ("C", "P"):
                p = bte._black76_price(F, K, T, DF, sigma_of_T(T), right == "C")
                rows.append({"expiry": str(expiry.date()), "strike": K, "right": right,
                             "bid": p - 0.05, "ask": p + 0.05})
    return pd.DataFrame(rows)


class TestBetaEstimator:
    def test_recovers_known_30d_atm_vol(self):
        as_of = pd.Timestamp("2026-06-01 16:00")
        sig = lambda T: 0.12 + 0.30 * T                     # upward term structure, no smile
        chain = _synthetic_chain(7000.0, 0.04, as_of, [21, 27, 34, 41], sig)
        term = bte.atm_forward_term_structure(chain, as_of, 0.04)
        np.testing.assert_allclose(term["atm_iv"], [sig(T) for T in term["T"]], atol=2e-5)
        a = term.loc[term["T"] <= T30].iloc[-1]
        b = term.loc[term["T"] > T30].iloc[0]
        w = a["atm_iv"] ** 2 * a["T"] + (b["atm_iv"] ** 2 * b["T"] - a["atm_iv"] ** 2 * a["T"]) \
            * (T30 - a["T"]) / (b["T"] - a["T"])
        est = bte.estimate_atm_to_vix_ratio(chain, as_of, vix=20.0, r=0.04, q=Q)
        assert est["atm30"] == pytest.approx(np.sqrt(w / T30), abs=1e-6)
        assert est["ratio"] == pytest.approx(est["atm30"] / 0.20)
        assert est["implied_spot"] == pytest.approx(7000.0, rel=1e-4)

    def test_live_chain_fixture(self):
        """Yahoo ^SPX chain captured 2026-10-12 04:39 ET (Cboe GTH quotes)."""
        meta = json.loads((FIXTURES / "spx_chain_live_20261012.json").read_text())
        chain = pd.read_csv(FIXTURES / "spx_chain_live_20261012.csv")
        est = bte.estimate_atm_to_vix_ratio(chain, pd.Timestamp(meta["as_of_et"]),
                                            meta["vix"], meta["irx_rate"], Q)
        assert est["ratio"] == pytest.approx(meta["ratio_full_chain"], abs=1e-6)
        assert est["ratio"] == pytest.approx(0.797, abs=0.001)
        assert 7_800 < est["implied_spot"] < 7_870


# ── sub-period stability ───────────────────────────────────────────────────────

class TestSubPeriods:
    def test_no_classifier_is_trained(self, monkeypatch):
        """Pre-C22 every window re-trained the C8 classifier (~6 min) and the
        trading path ignored it."""
        from joint_vol_calibration.signals import regime_classifier as rc
        calls = []
        monkeypatch.setattr(rc.RegimeClassifier, "fit",
                            lambda self, *a, **k: calls.append(1) or self)
        spx = _make_spx(1000, start="2016-01-04")
        vix = _make_vix(spx)
        eng = BacktestEngine(start_date="2018-01-02", end_date="2019-12-31")
        wf = eng.walk_forward_validation(spx, vix, n_windows=1, window_months=6)
        assert calls == []
        assert len(wf) == 1 and np.isfinite(wf["sharpe"].iloc[0])
