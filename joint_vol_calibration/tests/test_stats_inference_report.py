"""
The committed inference artifacts (data_store/backtest/inference/, written by
scripts/inference_report.py) are internally consistent and the parts that can
be rebuilt in seconds rebuild exactly: the bootstrap from the committed equity
curve, a sample of null paths from the canonical engine, and the PUT-implied
β from the committed benchmark levels.
"""

import json

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.stats import benchmarks as bm
from joint_vol_calibration.stats import bootstrap as bs
from joint_vol_calibration.stats import nulls as nl
from joint_vol_calibration.stats import sharpe as sh

OUT = bte.BACKTEST_DIR / "inference"
SLEEVES = {"s1": ("nav_s1", "pnl_s1"), "s1c": ("nav_s1c", "pnl_s1c"), "s2": ("nav_s2", "pnl_s2"),
           "s3": ("nav_s3", "pnl_s3"), "s4": ("nav_s4", "pnl_s4"), "portfolio": ("nav", "daily_pnl")}


@pytest.fixture(scope="module")
def summary():
    return json.loads((OUT / "summary.json").read_text())


@pytest.fixture(scope="module")
def committed():
    return pd.read_parquet(bte.RESULTS_PATH)


def test_every_artifact_is_present_and_well_formed(summary):
    for name in ("summary.json", "nulls.parquet", "bootstrap.parquet", "forecast_eval.json",
                 "benchmarks.json"):
        assert (OUT / name).exists(), name
    assert summary["schema"] == "inference_summary/1"
    assert set(summary["bootstrap"]) == set(SLEEVES)
    reps = pd.read_parquet(OUT / "bootstrap.parquet")
    assert list(reps.columns) == ["series", "rep", "sharpe", "sortino", "mean_daily_pnl"]
    assert len(reps) == summary["bootstrap_settings"]["reps"] * len(SLEEVES)
    nulls = pd.read_parquet(OUT / "nulls.parquet")
    assert set(nulls["sleeve"]) == set(SLEEVES)
    for s in SLEEVES:
        c = summary["nulls"]["sleeves"][s]["circular"]["total_pnl"]
        assert c["n_paths"] >= 2000
        assert (nulls[(nulls["sleeve"] == s) & (nulls["kind"] == "circular")].shape[0] == c["n_paths"])


def test_internal_consistency(summary):
    for s in SLEEVES:
        for k in bs.STATISTICS:
            x = summary["bootstrap"][s][k]
            lo, hi = x["ci95"]
            assert lo <= x["point"] <= hi
            assert x["excludes_zero"] == (lo > 0 or hi < 0)
        cf = summary["closed_form"][s]
        for dsr in summary["trials"]["dsr"][s].values():
            assert dsr <= cf["psr_gt_0"] + 1e-12            # deflating never raises the probability
        for kind in ("circular", "shuffle"):
            n = summary["nulls"]["sleeves"][s][kind]["total_pnl"]
            assert 0 < n["p_value"] <= 1
            assert n["timing_component"] == pytest.approx(n["real"] - n["null_mean"], rel=1e-8, abs=1e-6)
        rng = summary["trials"]["dsr_range"][s]
        grid = summary["trials"]["dsr_sensitivity"]["values"][s]
        for v in summary["trials"]["dsr"][s].values():
            assert rng["min"] - 1e-12 <= v <= rng["max"] + 1e-12
        assert rng["max"] <= cf["psr_gt_0"] + 1e-12
        # the grid holds the extremes of every (N, V) choice
        for v in (rng["min"], rng["max"]):
            assert min(abs(v - g) for g in grid.values()) <= 1e-9 * max(1.0, abs(v))
    t = summary["trials"]
    assert 1 <= t["family"]["n_eff_c16_c22"] <= t["ledger"]["n_trials_c16_c22"]
    nc = t["n_choices"]
    assert 1 <= nc["n_eigen_participation"] <= t["family"]["n_series"]
    assert nc["n_raw_c16_c22"] == t["ledger"]["n_trials_c16_c22"] <= nc["n_raw_all_rounds"] \
        <= nc["n_upper_with_evaluations"]
    # the expected maximum Sharpe of N skill-less trials rises with N and with V
    v = t["variance_choices_per_period"]
    for vname in v:
        sr0 = [t["sr0_annual_all_choices"][f"{vname}|{k}"] for k in sorted(nc, key=nc.get)]
        assert sr0 == sorted(sr0)
    assert v["null_var"] == pytest.approx(1.0 / (summary["n_sessions"] - 1))


def _script():
    import importlib.util

    from joint_vol_calibration.config import ROOT_DIR

    spec = importlib.util.spec_from_file_location("inference_report_under_test",
                                                  ROOT_DIR / "scripts" / "inference_report.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_headline_is_generated_from_the_stored_numbers(summary):
    """No hand-typed numbers: the stored headline is what the script's own
    headline() makes of the stored summary, forecast and benchmark files."""
    fe = json.loads((OUT / "forecast_eval.json").read_text())
    bj = json.loads((OUT / "benchmarks.json").read_text())
    assert _script().headline(summary, fe, bj) == summary["headline"]


def test_bootstrap_rebuilds_exactly_from_the_committed_curve(summary, committed):
    st = summary["bootstrap_settings"]
    reps = pd.read_parquet(OUT / "bootstrap.parquet")
    for s in ("s3", "portfolio"):
        i = st["series_order"].index(s)
        nav, pnl = SLEEVES[s]
        r = sh.returns_from_nav(committed[nav].to_numpy(float), bte.INITIAL_CAPITAL)
        res = bs.bootstrap_series(r, committed[pnl].to_numpy(float), reps=st["reps"], seed=st["seed"] + i)
        assert res["block"] == pytest.approx(summary["bootstrap"][s]["block"], rel=1e-9)
        assert res["ci"]["sharpe"] == pytest.approx(summary["bootstrap"][s]["sharpe"]["ci95"], rel=1e-8)
        np.testing.assert_array_equal(res["replicates"]["sharpe"],
                                      reps[reps["series"] == s]["sharpe"].to_numpy())


@pytest.fixture(scope="module")
def canonical_null():
    from joint_vol_calibration.backtest.run_backtest import _load_data, load_pdv_vol
    from joint_vol_calibration.stats.straddle_replay import StraddleReplay

    spx, vix = _load_data()
    eng = bte.BacktestEngine(pdv_vol_series=load_pdv_vol())
    eq = eng.run(spx, vix)
    mk = eng._market
    rep = StraddleReplay(eng.signals_df.index, mk["spx_close"], mk["vix_idx"], mk["r"],
                         q=eng.q, atm_to_vix_ratio=eng.atm_to_vix_ratio)
    return nl.PortfolioNull(eng, rep), eq


def test_replay_and_portfolio_null_are_the_engine(canonical_null):
    pn, eq = canonical_null
    real = pn.run_path(nl.NullPath("real", -1))
    for s, (_, pcol) in SLEEVES.items():
        np.testing.assert_array_equal(real[s], eq[pcol].to_numpy(float))


def test_committed_null_paths_rebuild(canonical_null):
    pn, _ = canonical_null
    nulls = pd.read_parquet(OUT / "nulls.parquet")
    rows = nulls.drop_duplicates(["kind", "path_id"])
    pick = pd.concat([rows[rows["kind"] == "circular"].iloc[[0, 700]],
                      rows[(rows["kind"] == "circular") & rows["reverse"]].iloc[[3]],
                      rows[rows["kind"] == "shuffle"].iloc[[5]]])
    for _, r in pick.iterrows():
        path = nl.NullPath(r["kind"], int(r["path_id"]), int(r["shift"]), bool(r["reverse"]),
                           None if r["seed"] < 0 else int(r["seed"]))
        got = pn.statistics(pn.run_path(path))
        ref = nulls[(nulls["kind"] == r["kind"]) & (nulls["path_id"] == r["path_id"])].set_index("sleeve")
        for s in SLEEVES:
            assert got[s]["total_pnl"] == pytest.approx(float(ref.loc[s, "total_pnl"]), rel=1e-12, abs=1e-6)
            assert got[s]["sharpe"] == pytest.approx(float(ref.loc[s, "sharpe"]), rel=1e-12, abs=1e-12)


def test_put_implied_beta_rebuilds_from_the_committed_levels():
    from joint_vol_calibration.backtest.delta_hedger import _interp_atm_iv
    from joint_vol_calibration.data.database import (
        get_spx_ohlcv,
        get_tbill_rates_series,
        get_vix_term_structure_wide,
    )

    bj = json.loads((OUT / "benchmarks.json").read_text())
    lv = bm.load_levels(bte.DATA_DIR / "benchmarks", ("PUT", "SET"))
    spx = get_spx_ohlcv(as_of_date=bte.BT_END, start_date="2017-01-01")
    spx = spx.set_index(pd.to_datetime(spx["date"]))
    vw = get_vix_term_structure_wide(as_of_date=bte.BT_END, start_date="2017-01-01")
    vw = vw.set_index(pd.to_datetime(vw["date"]))
    tb = get_tbill_rates_series(as_of_date=bte.BT_END, start_date="2017-01-01")
    rates = pd.Series(tb["rate"].to_numpy(float), index=pd.to_datetime(tb["date"]))
    sessions = spx.index[(spx.index >= "2017-12-01") & (spx.index <= bte.BT_END)]
    rolls = bm.monthly_roll_dates(sessions)
    rolls = rolls[(rolls >= "2017-12-15") & (rolls <= bte.BT_END)]
    per = bm.put_write_periods(rolls, spx["close"], lv["SET"], vw, rates, bte.Q_BACKTEST,
                               expiry_hour_et=9.5, settle_fallback=spx["open"])
    R = lv["PUT"].reindex(per["expiry"]).to_numpy() / lv["PUT"].reindex(per["roll"]).to_numpy() - 1
    vt = np.array([_interp_atm_iv(vw.loc[a], T) for a, T in zip(per["roll"], per["T"])])
    yrs = per["expiry"].dt.year.to_numpy()
    ib = bm.implied_beta(per, vt, R, groups={str(y): yrs == y for y in sorted(set(yrs))})
    ref = bj["implied_beta"]["PUT"]
    assert ib["overall"] == pytest.approx(ref["overall"], rel=1e-8)
    for y, b in ref["by_year"].items():
        assert ib["groups"][y] == pytest.approx(b, rel=1e-8)
    assert set(ref["by_year"]) >= {str(y) for y in range(2018, 2025)}
    pp = ib["per_period"]
    per_year = ref["by_year_per_period"]
    assert (sum(v["n_periods"] for v in per_year.values())
            == ref["n_periods"] - ref["per_period"]["n_unsolved"])
    for y, v in per_year.items():
        sel = pp[(yrs == int(y)) & np.isfinite(pp)]
        assert v["per_period_median"] == pytest.approx(float(np.median(sel)), rel=1e-8)
        assert v["se_approx"] == pytest.approx(float(np.std(sel, ddof=1) / np.sqrt(sel.size)), rel=1e-8)


def test_forecast_eval_is_complete():
    fe = json.loads((OUT / "forecast_eval.json").read_text())
    for sample in ("full_sample", "backtest_window", "non_overlapping"):
        ev = fe[sample]
        assert "pdv" in ev["models"]
        assert len(ev["models"]) >= 6
        for loss in ("qlike", "mse"):
            assert all(v >= 0 for v in ev["losses"][loss].values())
            assert set(ev["dm"][loss]) == set(ev["models"]) - {"pdv"}
            pv = ev["mcs"][loss]["90"]["pvalues"]
            assert all(0 <= v <= 1 for v in pv.values())
            assert max(pv.values()) == 1.0
    assert fe["full_sample"]["n"] > 3000
