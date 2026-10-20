"""
StraddleReplay must be backtest_engine._simulate_straddle_pnl to the last bit
on any position/size series: entries, signal exits, expiry settlement and
rolls, direct sign flips, margin-capped and wiped-out sleeves, rate gaps.
"""

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.stats.straddle_replay import StraddleReplay


def _market(n=320, seed=0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    S = 2800.0 * np.exp(np.cumsum(rng.normal(0, 0.012, n)))
    base = 16 + np.cumsum(rng.normal(0, 0.6, n)).clip(-8, 30)
    vix = pd.DataFrame({"^VIX9D": base - 1 + rng.normal(0, 0.5, n), "^VIX": base,
                        "^VIX3M": base + 1.5, "^VIX6M": base + 2.5}, index=dates).clip(lower=9)
    vix.loc[dates[40], "^VIX9D"] = np.nan                 # a missing tenor
    rates = pd.Series(0.02 + 0.02 * rng.random(n), index=dates)
    rates = rates.drop(dates[[5, 6, 100]])                # gaps → previous print
    return dates, pd.Series(S, index=dates), vix, rates


def _positions(dates, rng, flips=True):
    n = len(dates)
    pos = np.zeros(n)
    i = 0
    while i < n:
        L = int(rng.integers(1, 45))                      # some runs outlive the 21-session tenor
        v = rng.choice([-1, 0, 1, 0])
        pos[i:i + L] = v
        i += L
    if flips:
        pos[100:110], pos[110:118] = 1, -1                 # a direct +1 → −1 flip
    kel = rng.uniform(0.02, 0.2, n)
    return pd.Series(pos, index=dates), pd.Series(kel, index=dates)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_replay_equals_engine_bit_for_bit(seed):
    dates, S, vix, rates = _market(seed=seed)
    rng = np.random.default_rng(100 + seed)
    pos, kel = _positions(dates, rng)
    nav = pd.Series(1_000_000.0, index=dates)
    ref, _ = bte._simulate_straddle_pnl(pos, kel, nav, S, vix, r=rates, q=0.013,
                                        atm_to_vix_ratio=0.82)
    rep = StraddleReplay(dates, S, vix, rates, q=0.013, atm_to_vix_ratio=0.82)
    got = rep.simulate(pos.to_numpy(), kel.to_numpy(), nav.to_numpy())
    assert np.array_equal(got, ref.to_numpy())
    assert np.abs(ref.to_numpy()).sum() > 0                 # the market was actually traded


def test_replay_equals_engine_when_the_sleeve_is_margin_capped_or_wiped_out():
    dates, S, vix, rates = _market(seed=5)
    rng = np.random.default_rng(7)
    pos, _ = _positions(dates, rng, flips=False)
    kel = pd.Series(5.0, index=dates)                       # absurd size: NAV goes negative
    for base in (300_000.0, 50_000.0):                      # 50k: a short cannot pass the margin check
        nav = pd.Series(base, index=dates)
        ref, _ = bte._simulate_straddle_pnl(pos, kel, nav, S, vix, r=0.03, atm_to_vix_ratio=1.0)
        rep = StraddleReplay(dates, S, vix, 0.03, atm_to_vix_ratio=1.0)
        assert np.array_equal(rep.simulate(pos.to_numpy(), kel.to_numpy(), base), ref.to_numpy())


def test_replay_refuses_what_it_does_not_model():
    dates, S, vix, rates = _market(seed=3)
    with pytest.raises(NotImplementedError):
        StraddleReplay(dates, S, vix.drop(dates[10]), rates)
    with pytest.raises(NotImplementedError):
        StraddleReplay(dates, S, vix, rates, corrections=bte.C22Corrections.legacy())
    rep = StraddleReplay(dates, S, vix, rates)
    with pytest.raises(ValueError):
        rep.simulate(np.zeros(len(dates) - 1), np.zeros(len(dates) - 1))
