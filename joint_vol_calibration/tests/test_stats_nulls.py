"""
Null-path generators (stats/nulls.py): they preserve the exposure they claim
to preserve, the p-value is the standard (1 + k)/(B + 1), and the null finds
timing skill that is really there and none where there is none.
"""

from collections import Counter

import numpy as np
import pytest

from joint_vol_calibration.stats import nulls


def test_circular_shift_paths_enumerate_every_admissible_shift():
    paths = nulls.circular_shift_paths(100, min_shift=10)
    fwd = [p.shift for p in paths if not p.reverse]
    rev = [p.shift for p in paths if p.reverse]
    assert fwd == list(range(10, 91)) and rev == fwd
    assert len({p.path_id for p in paths}) == len(paths) == 162
    assert len(nulls.circular_shift_paths(1816)) >= 2000         # the canonical window
    with pytest.raises(ValueError):
        nulls.circular_shift_paths(100, min_shift=60)


def _runs(x):
    s = np.sign(x).astype(int)
    out, start = [], 0
    for i in range(1, len(s) + 1):
        if i == len(s) or s[i] != s[start]:
            out.append((int(s[start]), i - start))
            start = i
    return out


def test_shift_and_reverse_keep_the_exposure():
    rng = np.random.default_rng(0)
    pos = np.repeat(rng.choice([-1, 0, 1], 40), rng.integers(1, 15, 40)).astype(float)
    kel = rng.random(pos.size)
    for shift, rev in [(17, False), (33, True), (0, True)]:
        p2, k2 = nulls.apply_path(nulls.NullPath("circular", 0, shift, rev), [pos, kel], pos)
        assert Counter(p2.tolist()) == Counter(pos.tolist())
        assert sorted(k2.tolist()) == sorted(kel.tolist())
        assert np.array_equal(p2, nulls.shift_series(pos, shift, rev))
        # a shift splits at most one run in two
        assert abs(nulls.n_entries(p2) - nulls.n_entries(pos)) <= 1
    assert np.array_equal(nulls.shift_series(np.arange(5), 2), np.array([2, 3, 4, 0, 1]))
    assert np.array_equal(nulls.shift_series(np.arange(5), 1, reverse=True), np.array([3, 2, 1, 0, 4]))


def test_trade_shuffle_permutes_whole_trades_and_gaps():
    rng = np.random.default_rng(1)
    pos = np.repeat(rng.choice([-1, 0, 1], 60), rng.integers(1, 12, 60)).astype(float)
    kel = np.arange(pos.size, dtype=float)                  # distinct values tag each row
    p2, k2 = nulls.trade_shuffle([pos, kel], pos, np.random.default_rng(5))
    assert not np.array_equal(p2, pos)
    # the same runs (direction, length) — adjacent equal-sign runs may merge after
    # shuffling, so compare the row content of each original run instead
    for start, stop in nulls._runs(np.sign(pos).astype(int)):
        seg = kel[start:stop]
        j = int(np.flatnonzero(k2 == seg[0])[0])
        assert np.array_equal(k2[j:j + seg.size], seg)          # run kept intact, in order
        assert np.all(p2[j:j + seg.size] == pos[start])
    assert sum(L for s, L in _runs(p2) if s != 0) == sum(L for s, L in _runs(pos) if s != 0)
    # trades stay separated by gaps: the entry count is kept (exposure-matched)
    gappy = np.repeat(np.tile([1.0, 0.0], 40), rng.integers(1, 9, 80))
    for seed in range(5):
        g2, = nulls.trade_shuffle([gappy], gappy, np.random.default_rng(seed))
        assert nulls.n_entries(g2) == nulls.n_entries(gappy) == 40
        assert sorted(L for s, L in _runs(g2)) == sorted(L for s, L in _runs(gappy))


def test_holm_adjustment():
    adj = nulls.holm({"a": 0.01, "b": 0.04, "c": 0.03, "d": 0.5})
    assert adj["a"] == pytest.approx(0.04)
    assert adj["c"] == pytest.approx(0.09)
    assert adj["b"] == pytest.approx(0.09)              # monotone: never below a smaller p's value
    assert adj["d"] == pytest.approx(0.5)


def test_n_entries_counts_starts_and_flips():
    assert nulls.n_entries(np.array([0, 1, 1, 0, -1, 1, 1, 0, 0, 1])) == 4
    assert nulls.n_entries(np.zeros(5)) == 0


def test_pvalue_and_summary():
    null = np.array([1.0, 2.0, 3.0, 4.0])
    assert nulls.pvalue_greater(3.0, null) == pytest.approx(3 / 5)
    assert nulls.pvalue_greater(10.0, null) == pytest.approx(1 / 5)
    s = nulls.summarise_null(5.0, null)
    assert s["timing_component"] == pytest.approx(5.0 - 2.5)
    assert s["n_paths"] == 4 and s["null_frac_positive"] == 1.0


def _toy_pnl(pos, f):
    return float(np.sum(pos * f))


def test_circular_null_detects_real_timing_and_not_luck():
    """Toy market f_t = P&L per unit long exposure. A position with foresight
    is significant; an independent position is not (in most samples)."""
    rng = np.random.default_rng(2)
    n = 1500
    f = rng.standard_t(4, n)
    fut = np.convolve(f, np.ones(5), "full")[4:4 + n]     # f_t + … + f_{t+4}
    skilled = np.where(np.abs(fut) > 3.0, np.sign(fut), 0.0)
    paths = nulls.circular_shift_paths(n, min_shift=21)
    null = [_toy_pnl(nulls.apply_path(p, [skilled], skilled)[0], f) for p in paths]
    assert nulls.pvalue_greater(_toy_pnl(skilled, f), np.array(null)) < 0.01
    ps = []
    for s in range(10):
        r = np.random.default_rng(50 + s)
        pos = np.repeat(r.choice([-1, 0, 1], 400), r.integers(1, 15, 400))[:n].astype(float)
        null = [_toy_pnl(nulls.apply_path(p, [pos], pos)[0], f) for p in paths[::7]]
        ps.append(nulls.pvalue_greater(_toy_pnl(pos, f), np.array(null)))
    assert np.mean(np.array(ps) < 0.05) <= 0.3
    assert 0.2 < np.mean(ps) < 0.8
