"""
test_backtest_reproducibility.py — the committed backtest artifacts must
reproduce from the code.

Re-runs the canonical backtest (exactly what
``python -m joint_vol_calibration.backtest.run_backtest`` does, minus the
S3 null test and the sub-period windows) on the shipped database once, and
compares it with the files README's headline block, β grid and waterfall
are read from. What is checked, exactly:

  full_results.parquet   every column of the equity curve: same columns,
                         order and index; float columns within 1e-6 (not
                         a byte comparison of the file), the others equal
  full_results.pkl       every stored metric (rel 1e-9) except
                         s3_null_p_value, and no metric missing or extra;
                         the trade log (signal, dates, contracts, open
                         flag, net P&L); the run settings
  beta_sensitivity.csv   the β = 0.82 row only, rebuilt by run_backtest's
                         own row builder and CSV format: the line and the
                         header must equal the committed text byte for
                         byte. The other five β rows are NOT re-run here
                         (five more full backtests)
  c22_bridge.csv         the last row (pdv_h21 = the canonical run) only,
                         rebuilt by c22_bridge's own row builder: every
                         field byte for byte except d_pnl, which needs row
                         13's unrounded P&L and is checked to the CSV's 6
                         significant digits. Rows 0-13 are NOT re-run here
  delta_hedge_2020.parquet  the C7 study re-run (DeltaHedger, ~2 s): same
                         columns, index and dtypes; values to 1e-10 of each
                         column's scale (artifact_tolerance.py: a CPU without
                         AVX-512 moves the last bits), and attrs; also re-run
                         with numpy's AVX-512 kernels disabled
  s3_null_p_value        NOT re-run (200 paths, ~4 min); only checked to be
                         a valid 200-path p-value, (1 + k)/201. Its pipeline
                         is pinned in test_backtest_accounting.py

A code change that moves any of these numbers fails here until the
artifacts are regenerated (README "Regenerating every artifact") and the
published copy is updated.

Also pins that every entry point gives the same run: BacktestEngine().run()
with no data must equal run_backtest's (data from 2015, PDV-21 history from
2010). Before C22 the no-argument run gave NAV $729,697 against the
published $738,794. And that the engine's default β is the estimate from
the real SPX chains (run_backtest --estimate-beta).
"""

import csv
import io
import pickle

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest.backtest_engine import (
    ARTIFACT_CSV_FLOAT_FORMAT,
    ATM_TO_VIX_RATIO,
    BETA_ESTIMATION_SESSIONS,
    BACKTEST_DIR,
    RESULTS_PATH,
    BacktestEngine,
)
from joint_vol_calibration.tests.artifact_tolerance import (
    ARTIFACT_RTOL,
    assert_frame_matches_artifact,
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
    assert set(stored) == set(m) | {"s3_null_p_value"}
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


def _csv_rows(text: str) -> list:
    return list(csv.reader(io.StringIO(text)))


def test_beta_grid_canonical_row_is_byte_identical(canonical):
    """The β = 0.82 line of beta_sensitivity.csv, every column, as the
    --beta-sensitivity writer would print it from the canonical run."""
    from joint_vol_calibration.backtest.run_backtest import beta_row
    eng, eq, m = canonical
    committed = (BACKTEST_DIR / "beta_sensitivity.csv").read_text().splitlines()
    rebuilt = pd.DataFrame([beta_row(ATM_TO_VIX_RATIO, eng, eq, m)]).to_csv(
        index=False, float_format=ARTIFACT_CSV_FLOAT_FORMAT).splitlines()
    assert rebuilt[0] == committed[0]                                   # header
    (line,) = [ln for ln in committed[1:] if ln.split(",")[0] == f"{ATM_TO_VIX_RATIO:g}"]
    assert rebuilt[1] == line


def test_bridge_last_row_is_byte_identical(canonical):
    """The last c22_bridge.csv row (pdv_h21, the canonical run), every field
    as c22_bridge's writer prints it; d_pnl to the CSV's precision."""
    from joint_vol_calibration.backtest import c22_bridge
    eng, eq, m = canonical
    rows = _csv_rows((BACKTEST_DIR / "c22_bridge.csv").read_text())
    header, prev, last = rows[0], rows[-2], rows[-1]
    step = c22_bridge.build_steps()[-1]
    rec = {"step": step["key"], "label": step["label"], "note": step.get("note", ""),
           **c22_bridge.result_row(eng, eq, m), "d_pnl": np.nan}
    rebuilt = pd.DataFrame([rec], index=[int(last[0])]).to_csv(
        index_label="row", float_format=ARTIFACT_CSV_FLOAT_FORMAT)
    (h2, r2) = _csv_rows(rebuilt)
    assert h2 == header
    i = header.index("d_pnl")
    assert r2[:i] + r2[i + 1:] == last[:i] + last[i + 1:]
    pnl = rec["pnl"] - float(prev[header.index("pnl")])
    assert float(last[i]) == pytest.approx(pnl, abs=1.0)    # row 13's P&L is rounded to 6 digits


def _assert_c7_matches(df: pd.DataFrame, committed: pd.DataFrame) -> None:
    assert_frame_matches_artifact(df, committed)
    assert df.attrs.keys() == committed.attrs.keys()
    for k, v in committed.attrs.items():
        assert df.attrs[k] == (pytest.approx(v, rel=ARTIFACT_RTOL) if isinstance(v, float) else v), k


def test_c7_delta_hedge_study_reproduces():
    """delta_hedge_2020.parquet is DeltaHedger().run() (run_backtest --c7)."""
    from joint_vol_calibration.backtest.delta_hedger import OUTPUT_PARQUET, DeltaHedger
    _assert_c7_matches(DeltaHedger().run(), pd.read_parquet(OUTPUT_PARQUET))


_C7_IN_CHILD = """
import sys
from joint_vol_calibration.backtest.delta_hedger import DeltaHedger
DeltaHedger().run().to_pickle(sys.argv[1])
"""


def test_c7_reproduces_without_avx512(tmp_path):
    """CI run 39166521027: GitHub's runner (no AVX-512) recomputed the study
    1.1e-15 off the committed file, and the old exact comparison failed it.
    The same run with numpy's AVX-512 kernels disabled must still match (on a
    CPU without them the variable changes nothing). The child only reads
    data_store; the suite's guard fails the run if it wrote there."""
    import os
    import subprocess
    import sys
    from pathlib import Path
    from joint_vol_calibration.backtest.delta_hedger import OUTPUT_PARQUET
    out = tmp_path / "c7.pkl"
    root = Path(__file__).resolve().parents[2]
    env = {**os.environ, "NPY_DISABLE_CPU_FEATURES": "X86_V4",
           "PYTHONPATH": os.pathsep.join(filter(None, [str(root), os.environ.get("PYTHONPATH")]))}
    subprocess.run([sys.executable, "-c", _C7_IN_CHILD, str(out)], cwd=tmp_path, env=env,
                   check=True, timeout=300)
    df = pd.read_pickle(out)
    _assert_c7_matches(df, pd.read_parquet(OUTPUT_PARQUET))


def test_artifact_comparison_accepts_rounding_and_rejects_a_real_change():
    """The tolerance passes last-bit noise (3x the worst measured, on every
    float column at once) and fails a change a code edit would make."""
    from joint_vol_calibration.backtest.delta_hedger import OUTPUT_PARQUET
    committed = pd.read_parquet(OUTPUT_PARQUET)
    num = committed.select_dtypes("float").columns
    noisy = committed.copy()
    noisy[num] = noisy[num] * (1 + 1e-14)
    noisy["pnl_residual_a"] += 3 * 2.8e-13 * committed["pnl_residual_a"].abs().max()
    noisy.attrs = {k: (v * (1 + 1e-14) if isinstance(v, float) else v)
                   for k, v in committed.attrs.items()}
    _assert_c7_matches(noisy, committed)

    for col, bump in (("sigma_pdv", 1e-8), ("pnl_total", 1e-6), ("pnl_residual_a", 1e-6)):
        moved = committed.copy()
        moved.iloc[100, moved.columns.get_loc(col)] += bump
        with pytest.raises(AssertionError, match=col):
            _assert_c7_matches(moved, committed)
    flipped = committed.copy()
    flipped["is_unstable"] = ~flipped["is_unstable"]
    with pytest.raises(AssertionError, match="is_unstable"):
        _assert_c7_matches(flipped, committed)
    moved = committed.copy()
    moved.attrs = {**committed.attrs, "cum_residual_a": committed.attrs["cum_residual_a"] + 1e-6}
    with pytest.raises(AssertionError, match="cum_residual_a"):
        _assert_c7_matches(moved, committed)


def test_stored_null_p_value_is_a_200_path_p_value(committed):
    """Not re-run (~4 min): the stored p must at least be (1 + k)/201."""
    _, state = committed
    p = state["metrics"]["s3_null_p_value"]
    k = p * 201 - 1
    assert 0 <= k <= 200 and k == pytest.approx(round(k), abs=1e-9)


def test_default_beta_is_the_estimate_from_the_real_chains():
    """ATM_TO_VIX_RATIO is the mean β over the frozen estimation set (2 dp): the
    four time-matched DB sessions plus the captured live GTH chain."""
    import json
    from pathlib import Path
    from joint_vol_calibration.backtest.run_backtest import estimate_beta

    table = estimate_beta(include_live=False)
    inc = table[table["status"].str.startswith("included")]
    assert list(inc["session"]) == list(BETA_ESTIMATION_SESSIONS)
    assert ((inc["ratio"] > 0.80) & (inc["ratio"] < 0.85)).all()
    # Chains the bot stores later are reported, never folded into the default.
    later = table[table["session"].fillna("") > max(BETA_ESTIMATION_SESSIONS)]
    assert later["status"].str.startswith(("out-of-sample", "excluded")).all()
    meta = json.loads((Path(__file__).parent / "fixtures" / "spx_chain_live_20261012.json").read_text())
    ratios = list(inc["ratio"]) + [meta["ratio_full_chain"]]
    assert round(float(np.mean(ratios)), 2) == ATM_TO_VIX_RATIO
