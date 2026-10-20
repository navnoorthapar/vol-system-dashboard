"""
Stationary bootstrap (stats/bootstrap.py): index paths have the Politis-Romano
structure, intervals cover at close to their nominal rate on dependent data,
and everything is reproducible from the seed.
"""

import numpy as np
import pytest

from joint_vol_calibration.stats import bootstrap as bs


def test_indices_have_stationary_bootstrap_structure():
    rng = np.random.default_rng(0)
    n, b = 2000, 12.0
    idx = bs.stationary_bootstrap_indices(n, b, 300, rng)
    assert idx.shape == (300, n)
    assert idx.min() >= 0 and idx.max() < n
    cont = (np.diff(idx, axis=1) % n) == 1
    # a block continues with probability 1 - 1/b (plus the chance a fresh draw lands next)
    assert cont.mean() == pytest.approx(1 - 1 / b + 1 / (b * n), abs=0.003)
    # block lengths are geometric with mean b
    lengths = []
    for row in cont[:50]:
        run = 1
        for c in row:
            if c:
                run += 1
            else:
                lengths.append(run)
                run = 1
    assert np.mean(lengths) == pytest.approx(b, rel=0.05)
    # every row is uniform over the sample
    counts = np.bincount(idx.ravel(), minlength=n) / idx.size
    assert counts.std() / counts.mean() < 0.1


def test_indices_reproducible_and_validated():
    a = bs.stationary_bootstrap_indices(100, 5, 10, np.random.default_rng(7))
    b = bs.stationary_bootstrap_indices(100, 5, 10, np.random.default_rng(7))
    c = bs.stationary_bootstrap_indices(100, 5, 10, np.random.default_rng(8))
    assert np.array_equal(a, b) and not np.array_equal(a, c)
    with pytest.raises(ValueError):
        bs.stationary_bootstrap_indices(100, 0.5, 10, np.random.default_rng(0))


def _ar1(n, phi, rng, mu=0.0):
    e = rng.normal(size=n)
    x = np.empty(n)
    x[0] = e[0] / np.sqrt(1 - phi ** 2)
    for i in range(1, n):
        x[i] = phi * x[i - 1] + e[i]
    return x + mu


def test_block_length_grows_with_persistence():
    rng = np.random.default_rng(1)
    iid = rng.normal(size=3000)
    ar = _ar1(3000, 0.8, rng)
    assert bs.block_length(ar, include_squares=False) > 3 * bs.block_length(iid, include_squares=False)
    # volatility clustering without autocorrelation in levels: r² drives the choice
    v = np.ones(3000)
    for i in range(1, 3000):
        v[i] = 0.05 + 0.1 * (iid[i - 1] * np.sqrt(v[i - 1])) ** 2 + 0.85 * v[i - 1]
    g = iid * np.sqrt(v)
    assert bs.block_length(g) >= bs.block_length(g, include_squares=False)


def test_mean_interval_covers_on_ar1_data():
    """95% percentile intervals of an AR(1) mean (φ = 0.5) cover the true mean
    in close to 95% of samples; an IID bootstrap (block 1) under-covers."""
    rng = np.random.default_rng(2)
    sims, n = 150, 600
    cover_sb, cover_iid = 0, 0
    for k in range(sims):
        x = _ar1(n, 0.5, rng, mu=0.3)
        res = bs.bootstrap_series(x, x, reps=600, seed=k)
        lo, hi = res["ci"]["mean_daily_pnl"]
        cover_sb += lo <= 0.3 <= hi
        res1 = bs.bootstrap_series(x, x, reps=600, seed=k, mean_block=1.0)
        lo1, hi1 = res1["ci"]["mean_daily_pnl"]
        cover_iid += lo1 <= 0.3 <= hi1
    assert 0.87 <= cover_sb / sims <= 0.99
    assert cover_iid / sims < cover_sb / sims - 0.1


def test_bootstrap_series_point_values_and_reproducibility():
    rng = np.random.default_rng(3)
    r = rng.normal(0.0004, 0.01, 1500)
    pnl = r * 1e6
    a = bs.bootstrap_series(r, pnl, reps=400, seed=11)
    b = bs.bootstrap_series(r, pnl, reps=400, seed=11)
    assert a["point"]["sharpe"] == pytest.approx(r.mean() / r.std(ddof=1) * np.sqrt(252))
    assert a["point"]["mean_daily_pnl"] == pytest.approx(pnl.mean())
    for k in bs.STATISTICS:
        assert np.array_equal(a["replicates"][k], b["replicates"][k])
        lo, hi = a["ci"][k]
        assert lo < a["point"][k] < hi
    with pytest.raises(ValueError):
        bs.bootstrap_series(r, pnl[:-1])


def test_bootstrap_table_seeds_each_series_independently():
    rng = np.random.default_rng(4)
    x, y = rng.normal(0.001, 0.01, 800), rng.normal(-0.001, 0.01, 800)
    s1, reps1 = bs.bootstrap_table({"x": {"returns": x, "pnl": x}}, reps=200, seed=5)
    s2, reps2 = bs.bootstrap_table({"x": {"returns": x, "pnl": x}, "y": {"returns": y, "pnl": y}},
                                   reps=200, seed=5)
    assert np.array_equal(reps1["sharpe"].to_numpy(), reps2[reps2["series"] == "x"]["sharpe"].to_numpy())
    assert set(s2["statistic"]) == set(bs.STATISTICS)
    row = s2[(s2["series"] == "y") & (s2["statistic"] == "sharpe")].iloc[0]
    assert row["excludes_zero"] == bool(row["ci_hi"] < 0 or row["ci_lo"] > 0)
