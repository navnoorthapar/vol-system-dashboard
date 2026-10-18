"""
test_backtest_reproducibility.py — the committed backtest artifacts must
reproduce from the code.

Re-runs the canonical backtest (exactly what
``python -m joint_vol_calibration.backtest.run_backtest`` does, minus the
S3 null test and the sub-period windows) on the shipped database and
compares it with data_store/backtest/full_results.{parquet,pkl}, the β = 0.82
row of beta_sensitivity.csv and the last row of c22_bridge.csv — the files
README's headline block, β grid and waterfall are read from. A code change
that moves any of these numbers fails here until the artifacts are
regenerated (README "Regenerating every artifact") and the published copy is
updated.

Also pins that every entry point gives the same run: BacktestEngine().run()
with no data must equal run_backtest's (data from 2015, PDV-21 history from
2010). Before C22 the no-argument run gave NAV $729,697 against the
published $738,794. And that the engine's default β is the estimate from
the real SPX chains (run_backtest --estimate-beta).

The S3 null-test p-value (200 paths, ~4 min) is not re-run here; its
pipeline is pinned in test_backtest_accounting.py.
"""

import pickle

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest.backtest_engine import (
    ATM_TO_VIX_RATIO,
    BACKTEST_DIR,
    RESULTS_PATH,
    BacktestEngine,
)

SLEEVES = ("s1c", "s3", "s4", "s1", "s2", "combined")


@pytest.fixture(scope="module")
def committed():
    eq = pd.read_parquet(RESULTS_PATH)
    with open(RESULTS_PATH.with_suffix(".pkl"), "rb") as fh:
        state = pickle.load(fh)
    return eq, state


@pytest.fixture(scope="module")
def canonical():
    from joint_vol_calibration.backtest.run_backtest import _load_data, load_pdv_vol
    spx, vix = _load_data()
    eng = BacktestEngine(pdv_vol_series=load_pdv_vol())
    eq = eng.run(spx, vix)
    return eng, eq, eng.compute_metrics()


def test_committed_run_used_the_canonical_settings(committed):
    _, state = committed
    eng = BacktestEngine()
    assert state["atm_to_vix_ratio"] == ATM_TO_VIX_RATIO
    assert state["corrections"] == {k: True for k in state["corrections"]}
    assert state["pdv_horizon"] == eng.pdv_horizon
    assert state["s3_leg_direction"] == eng.s3_leg_direction
    assert (state["start_date"], state["end_date"]) == (eng.start_date, eng.end_date)


def test_equity_curves_reproduce(committed, canonical):
    eq_c, _ = committed
    _, eq, _ = canonical
    assert list(eq.columns) == list(eq_c.columns)
    pd.testing.assert_index_equal(eq.index, eq_c.index)
    num = [c for c in eq_c.columns if pd.api.types.is_float_dtype(eq_c[c])]
    for col in num:
        np.testing.assert_allclose(eq[col].to_numpy(), eq_c[col].to_numpy(),
                                   rtol=0, atol=1e-6, err_msg=col)
    for col in set(eq_c.columns) - set(num):
        assert (eq[col].fillna(-99) == eq_c[col].fillna(-99)).all(), col


def test_metrics_and_trade_log_reproduce(committed, canonical):
    _, state = committed
    eng, _, m = canonical
    stored = state["metrics"]
    for k, v in stored.items():
        if k == "s3_null_p_value":            # 200-path null test, not re-run here
            continue
        if isinstance(v, (float, np.floating)):
            assert m[k] == pytest.approx(v, rel=1e-9, abs=1e-9), k
        else:
            assert m[k] == v, k
    assert len(eng.all_trades) == len(state["all_trades"])
    for a, b in zip(eng.all_trades, state["all_trades"]):
        assert (a.signal, a.entry_date, a.exit_date, a.n_contracts, a.is_open) == \
               (b.signal, b.entry_date, b.exit_date, b.n_contracts, b.is_open)
        assert a.net_pnl == pytest.approx(b.net_pnl, rel=1e-9, abs=1e-6)


def test_every_entry_point_gives_the_same_run(canonical):
    _, eq, _ = canonical
    eng = BacktestEngine()
    eq2 = eng.run()
    np.testing.assert_allclose(eq2["nav"].to_numpy(), eq["nav"].to_numpy(), rtol=0, atol=1e-6)


def test_beta_grid_and_waterfall_end_on_the_canonical_run(canonical):
    _, eq, m = canonical
    grid = pd.read_csv(BACKTEST_DIR / "beta_sensitivity.csv")
    bridge = pd.read_csv(BACKTEST_DIR / "c22_bridge.csv")
    row = grid.loc[np.isclose(grid["beta"], ATM_TO_VIX_RATIO)].iloc[0]
    last = bridge.iloc[-1]
    nav = float(eq["nav"].iloc[-1])
    for r in (row, last):
        # the CSVs are written with 6 significant digits
        assert r["nav"] == pytest.approx(nav, rel=2e-6)
        assert r["sharpe"] == pytest.approx(m["sharpe"], rel=2e-6)
        assert r["max_dd"] == pytest.approx(m["max_drawdown"], rel=2e-6)
        for s in ("s1c", "s3", "s4", "s1", "s2"):
            assert r[f"{s}_pnl"] == pytest.approx(m[f"{s}_total_pnl"], rel=2e-6), s
            assert r[f"{s}_trades"] == m[f"{s}_n_trades"], s
    assert last["step"] == "pdv_h21"


def test_default_beta_is_the_estimate_from_the_real_chains():
    """ATM_TO_VIX_RATIO is the mean β over every real SPX chain (2 dp): the
    four time-matched DB sessions plus the captured live GTH chain."""
    import json
    from pathlib import Path
    from joint_vol_calibration.backtest.run_backtest import estimate_beta

    table = estimate_beta(include_live=False)
    inc = table[table["status"].str.startswith("included")]
    assert list(inc["session"]) == ["2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18"]
    assert ((inc["ratio"] > 0.80) & (inc["ratio"] < 0.85)).all()
    meta = json.loads((Path(__file__).parent / "fixtures" / "spx_chain_live_20261012.json").read_text())
    ratios = list(inc["ratio"]) + [meta["ratio_full_chain"]]
    assert round(float(np.mean(ratios)), 2) == ATM_TO_VIX_RATIO
