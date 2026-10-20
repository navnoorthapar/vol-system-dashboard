"""
Data-snooping tests (stats/multiple_testing.py): SPA and StepM reject only
when something really beats cash; PBO is ~0 for genuine persistent skill,
around one half for noise and high for configurations whose in-sample
winner is the out-of-sample loser by construction.
"""

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.stats import multiple_testing as mt


def _noise(T, N, rng, drift=None):
    X = rng.normal(0.0, 0.01, (T, N))
    if drift is not None:
        X = X + np.asarray(drift)[None, :]
    return pd.DataFrame(X, columns=[f"c{i}" for i in range(N)])


def test_spa_does_not_reject_for_skill_less_strategies():
    pvals = [mt.spa_vs_cash(_noise(1000, 8, np.random.default_rng(s), drift=np.zeros(8) - 0.0002),
                            block_size=5, reps=300, seed=s)["p_consistent"] for s in range(6)]
    assert min(pvals) > 0.05


def test_spa_rejects_when_one_strategy_beats_cash():
    rng = np.random.default_rng(10)
    drift = np.zeros(8)
    drift[3] = 0.0025                 # Sharpe ≈ 4 annualised
    res = mt.spa_vs_cash(_noise(1000, 8, rng, drift), block_size=5, reps=500, seed=1)
    assert res["p_consistent"] < 0.01
    assert res["best"] == "c3"
    assert res["p_lower"] <= res["p_consistent"] <= res["p_upper"]


def test_stepm_finds_exactly_the_superior_strategy():
    rng = np.random.default_rng(11)
    drift = np.full(6, -0.0005)
    drift[2] = 0.003
    res = mt.stepm_vs_cash(_noise(1200, 6, rng, drift), block_size=5, reps=500, seed=2)
    assert res["superior"] == ["c2"]


def test_pbo_near_zero_for_persistent_skill():
    rng = np.random.default_rng(12)
    drift = np.zeros(20)
    drift[0] = 0.003
    assert mt.pbo_cscv(_noise(1600, 20, rng, drift), n_splits=10)["pbo"] == pytest.approx(0.0, abs=0.02)


def test_pbo_about_one_half_for_noise():
    vals = [mt.pbo_cscv(_noise(1200, 25, np.random.default_rng(s)), n_splits=10)["pbo"] for s in range(8)]
    assert 0.3 < np.mean(vals) < 0.7


def test_pbo_high_when_in_sample_winners_reverse():
    """Each configuration earns +m on a random half of the blocks and −m on the
    rest: the best in-sample configuration is the worst out of sample."""
    rng = np.random.default_rng(13)
    S, T, N = 10, 2000, 30
    edges = np.linspace(0, T, S + 1).astype(int)
    X = rng.normal(0.0, 0.002, (T, N))
    for j in range(N):
        good = rng.permutation(S)[: S // 2]
        for k in range(S):
            X[edges[k]:edges[k + 1], j] += 0.004 if k in good else -0.004
    res = mt.pbo_cscv(X, n_splits=S)
    assert res["pbo"] > 0.9
    assert res["oos_on_is_slope"] < 0


def test_pbo_input_validation_and_subsampling():
    X = np.random.default_rng(0).normal(size=(400, 5))
    with pytest.raises(ValueError, match="n_splits must be an even number"):
        mt.pbo_cscv(X, n_splits=7)
    with pytest.raises(ValueError, match="at least two configurations"):
        mt.pbo_cscv(X[:, :1])
    res = mt.pbo_cscv(X, n_splits=12, max_combinations=100)
    assert res["n_combinations"] == 100
