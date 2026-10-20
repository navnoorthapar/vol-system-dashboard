"""
Cboe benchmark analysis (stats/benchmarks.py): parsing of the real file
format (captured fixtures, no network), roll calendars, the put-write
replication's algebra and the implied-β round trip.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm

from joint_vol_calibration.stats import benchmarks as bm

FIX = Path(__file__).parent / "fixtures"


def test_parse_the_captured_cboe_files():
    put = bm.parse_cboe_index_csv((FIX / "cboe_PUT_History_2018Q1.csv").read_text(), "PUT")
    st = bm.parse_cboe_index_csv((FIX / "cboe_SET_History_2018Q1.csv").read_text(), "SET")
    assert list(put.columns) == ["date", "close"]
    assert put["date"].iloc[0] == "2018-01-02"
    assert put["date"].is_monotonic_increasing
    assert len(put) == 61
    assert (put["close"] > 0).all()
    assert st.set_index("date").loc["2018-01-19", "close"] == pytest.approx(2806.67)


def test_parse_ohlc_duplicates_and_garbage():
    text = ("DATE,OPEN,HIGH,LOW,CLOSE\n01/03/2018,1,2,0.5,1.5\n01/02/2018,1,2,0.5,1.4\n"
            "01/03/2018,1,2,0.5,1.6\nnot a date,1,1,1,1\n01/04/2018,1,1,1,0\n")
    df = bm.parse_cboe_index_csv(text)
    assert df["date"].tolist() == ["2018-01-02", "2018-01-03"]
    assert df["close"].tolist() == [1.4, 1.6]                     # last duplicate kept, zero dropped
    with pytest.raises(ValueError, match="no DATE column"):
        bm.parse_cboe_index_csv("A,B,C\n1,2,3\n")


def test_roll_calendars():
    sessions = pd.bdate_range("2019-01-01", "2019-12-31").drop(pd.Timestamp("2019-04-19"))  # Good Friday
    m = bm.monthly_roll_dates(sessions)
    assert len(m) == 12
    assert pd.Timestamp("2019-04-18") in m                      # moved to Thursday
    assert pd.Timestamp("2019-01-18") in m
    assert pd.Timestamp("2019-03-15") in m
    w = bm.weekly_roll_dates(sessions)
    assert pd.Timestamp("2019-04-18") in w
    assert pd.Timestamp("2019-04-12") in w
    assert all(d.weekday() in (3, 4) for d in w[1:-1])


def test_bs_put_parity_and_limits():
    S, K, T, r, q, s = 100.0, 95.0, 0.25, 0.03, 0.01, 0.2
    p = float(bm.bs_put(S, K, T, r, q, s))
    d1 = (np.log(S / K) + (r - q + 0.5 * s * s) * T) / (s * np.sqrt(T))
    c = S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d1 - s * np.sqrt(T))
    assert c - p == pytest.approx(S * np.exp(-q * T) - K * np.exp(-r * T), abs=1e-10)
    assert float(bm.bs_put(90.0, 100.0, 0.0, r, q, s)) == pytest.approx(10.0)
    assert float(bm.bs_put(100.0, 100.0, 0.25, r, q, 0.3)) > p


def _periods(n=24, seed=0):
    rng = np.random.default_rng(seed)
    rolls = pd.bdate_range("2018-01-19", periods=n + 1, freq="21B")
    K = 2500 * np.exp(np.cumsum(rng.normal(0, 0.04, n)))
    return pd.DataFrame({"roll": rolls[:-1], "expiry": rolls[1:], "K": K,
                         "S_settle": K * np.exp(rng.normal(0, 0.05, n)), "settle_source": "primary",
                         "T": 30 / 365, "r": 0.02, "q": 0.013, "growth": np.exp(0.02 * 30 / 365)})


def test_period_returns_increase_with_vol_and_reduce_to_cash_when_the_put_is_worthless():
    per = _periods()
    lo = bm.period_returns(per, np.full(len(per), 0.10))
    hi = bm.period_returns(per, np.full(len(per), 0.20))
    assert np.all(hi > lo)
    otm = per.assign(S_settle=per["K"] * 1.2)
    np.testing.assert_allclose(bm.period_returns(otm, np.full(len(per), 1e-6)), per["growth"] - 1, atol=1e-12)


def test_implied_beta_round_trip():
    per = _periods(36, seed=1)
    vt = np.random.default_rng(2).uniform(0.12, 0.35, len(per))
    beta_true = np.r_[np.full(12, 0.70), np.full(12, 0.85), np.full(12, 1.00)]
    R = bm.period_returns(per, beta_true * vt)
    years = np.repeat(["a", "b", "c"], 12)
    ib = bm.implied_beta(per, vt, R, groups={g: years == g for g in "abc"})
    np.testing.assert_allclose(ib["per_period"], beta_true, atol=1e-7)
    assert ib["groups"]["a"] == pytest.approx(0.70, abs=1e-7)
    assert ib["groups"]["c"] == pytest.approx(1.00, abs=1e-7)
    assert 0.70 < ib["overall"] < 1.00
    # a target outside the bracket is NaN, not a clipped value
    assert np.isnan(bm.implied_beta(per.iloc[:1], vt[:1], np.array([0.5]))["per_period"][0])


def test_replication_daily_matches_the_period_returns_at_the_rolls():
    rng = np.random.default_rng(3)
    sessions = pd.bdate_range("2018-01-02", "2018-12-31")
    S = pd.Series(2700 * np.exp(np.cumsum(rng.normal(0, 0.01, len(sessions)))), index=sessions)
    rates = pd.Series(0.02, index=sessions)
    vix = pd.DataFrame({"^VIX": 15 + rng.random(len(sessions))}, index=sessions)
    rolls = bm.monthly_roll_dates(sessions)
    per = bm.put_write_periods(rolls, S, S, vix, rates, q=0.013, expiry_hour_et=16.0)
    assert len(per) == len(rolls) - 1
    assert (per["settle_source"] == "primary").all()
    vt = np.full(len(per), 0.16)
    lvl = bm.replication_daily(per, sessions, S, lambda d, T: 0.16, rates, beta=0.8, q=0.013,
                               expiry_hour_et=16.0)
    expected = np.cumprod(1 + bm.period_returns(per, 0.8 * vt))
    np.testing.assert_allclose(lvl.reindex(per["expiry"]).to_numpy(), expected, rtol=1e-12)
    # the T-bill growth of a period is exp(r × actual days / 365)
    days = (per["expiry"] - per["roll"]).dt.days.to_numpy()
    np.testing.assert_allclose(per["growth"].to_numpy(), np.exp(0.02 * days / 365), rtol=1e-12)


def test_put_write_periods_fall_back_and_record_it():
    sessions = pd.bdate_range("2018-01-02", "2018-04-30")
    S = pd.Series(2700.0, index=sessions)
    vix = pd.DataFrame({"^VIX": 15.0}, index=sessions)
    rolls = bm.monthly_roll_dates(sessions)
    settle = S.drop(rolls[2])
    per = bm.put_write_periods(rolls, S, settle, vix, pd.Series(0.02, index=sessions), 0.013,
                               settle_fallback=S * 1.01)
    assert per.set_index("expiry").loc[rolls[2], "settle_source"] == "fallback"
    assert per["T"].iloc[0] == pytest.approx(((rolls[1] - rolls[0]).days - 6.5 / 24) / 365)


def test_pair_stats_beta_and_partial_correlation():
    rng = np.random.default_rng(4)
    idx = pd.bdate_range("2019-01-01", periods=700)
    m = pd.Series(rng.normal(0, 0.01, 700), index=idx)
    b = m + pd.Series(rng.normal(0, 0.002, 700), index=idx)
    s_lin = 0.5 * b + pd.Series(rng.normal(0, 0.001, 700), index=idx)
    st = bm.pair_stats(s_lin, b, m)
    assert st["beta"] == pytest.approx(0.5, abs=0.02)
    assert st["corr_daily"] > 0.9
    # both driven by the market only: high raw correlation, ~zero partial correlation
    s_mkt = m + pd.Series(rng.normal(0, 0.002, 700), index=idx)
    st2 = bm.pair_stats(s_mkt, b, m)
    assert st2["corr_daily"] > 0.9
    assert abs(st2["partial_corr_ex_spx"]) < 0.1
    assert st["sharpe_diff"] == pytest.approx(st["sharpe_sleeve"] - st["sharpe_bench"])


def test_benchmark_excess_returns_and_table():
    idx = pd.bdate_range("2019-12-02", periods=60)
    levels = pd.DataFrame({"PUT": np.linspace(100, 106, 60)}, index=idx).drop(idx[10])
    cash = pd.Series(0.0001, index=idx)
    ex = bm.benchmark_excess_returns(levels, idx, cash)
    assert np.isnan(ex["PUT"].iloc[10])
    assert np.isnan(ex["PUT"].iloc[11])
    assert ex["PUT"].iloc[5] == pytest.approx(levels["PUT"].iloc[5] / levels["PUT"].iloc[4] - 1 - 0.0001)
    sl = pd.DataFrame({"s3": np.random.default_rng(5).normal(0, 0.01, 60)}, index=idx)
    tab = bm.strategy_benchmark_stats(sl, ex)
    assert set(tab["period"]) == {"all", "2019", "2020"}
    assert tab.set_index("period").loc["all", "n_days"] == 57
