#!/usr/bin/env python3
"""Statistical inference on the committed backtest, forecasts and benchmarks.

Writes data_store/backtest/inference/:

    summary.json        bootstrap CIs, PSR / DSR / MinTRL, Hansen SPA and
                        Romano-Wolf StepM, CSCV PBO, exposure-matched timing
                        nulls for every sleeve, the timing-vs-premium split,
                        and the provenance of every input
    bootstrap.parquet   the stationary-bootstrap replicates
                        (series, rep, sharpe, sortino, mean_daily_pnl)
    nulls.parquet       every null path (kind, path_id, shift, reverse, seed,
                        sleeve, total_pnl, sharpe, n_entries)
    forecast_eval.json  PDV vs HAR / GARCH / GJR / EWMA / VIX baselines:
                        QLIKE and MSE, Diebold-Mariano, Model Confidence Set
    benchmarks.json     sleeves vs Cboe PUT / WPUT / BXM / BFLY / CNDR / VXTH
                        by year, and the β that PUT / WPUT imply

Inputs: the committed equity curve (data_store/backtest/full_results.parquet),
the committed database (market data), docs/trials_ledger.yaml and
data_store/benchmarks/ (scripts/fetch_cboe_benchmarks.py). The canonical
backtest is re-run first and must reproduce the committed curve (1e-6), and
the trial family must reproduce the committed β grid and C22 bridge, or the
script stops: nothing is inferred from numbers it cannot rebuild.

Every random element is seeded (--seed); the same inputs give the same files.
Runtime: 536 s measured on a 4-CPU machine shared with other jobs (load
average ~12); the 5,550 null paths take 334 s of it and the 20-run trial
family 110 s.

Usage:  python scripts/inference_report.py [--out DIR] [--reps 10000] [--shuffles 2000] [--seed 42]
"""
from __future__ import annotations

import argparse
import contextlib
import logging
import pathlib
import subprocess
import sys
import time
from typing import Dict, List, Tuple

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np                                                    # noqa: E402
import pandas as pd                                                   # noqa: E402

from joint_vol_calibration.backtest import backtest_engine as bte      # noqa: E402
from joint_vol_calibration.stats import artifacts, benchmarks as bm   # noqa: E402
from joint_vol_calibration.stats import bootstrap as bs               # noqa: E402
from joint_vol_calibration.stats import forecast as fc                # noqa: E402
from joint_vol_calibration.stats import multiple_testing as mt        # noqa: E402
from joint_vol_calibration.stats import nulls as nl                   # noqa: E402
from joint_vol_calibration.stats import sharpe as sh                  # noqa: E402
from joint_vol_calibration.stats import trials as tr                  # noqa: E402
from joint_vol_calibration.stats.regression import ols_hac            # noqa: E402
from joint_vol_calibration.stats.straddle_replay import StraddleReplay  # noqa: E402

log = logging.getLogger("inference_report")

OUT_DIR = bte.BACKTEST_DIR / "inference"
BENCH_DIR = bte.DATA_DIR / "benchmarks"
SLEEVES: Dict[str, Tuple[str, str]] = {
    "s1": ("nav_s1", "pnl_s1"), "s1c": ("nav_s1c", "pnl_s1c"), "s2": ("nav_s2", "pnl_s2"),
    "s3": ("nav_s3", "pnl_s3"), "s4": ("nav_s4", "pnl_s4"), "portfolio": ("nav", "daily_pnl"),
}
CAP = float(bte.INITIAL_CAPITAL)
VRP_KELLY = 0.10
HAC_LAGS = 21


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:                                   # noqa: BLE001 — provenance is best effort
        return "unknown"


def code_sha256() -> Dict[str, str]:
    """SHA-256 of every source file whose code determines the numbers."""
    files = [ROOT / "scripts" / "inference_report.py",
             *sorted((ROOT / "joint_vol_calibration" / "stats").glob("*.py")),
             ROOT / "joint_vol_calibration" / "backtest" / "backtest_engine.py",
             ROOT / "joint_vol_calibration" / "backtest" / "c22_bridge.py",
             ROOT / "joint_vol_calibration" / "signals" / "signal_engine.py",
             ROOT / "joint_vol_calibration" / "models" / "pdv.py"]
    return {str(f.relative_to(ROOT)): artifacts.sha256_file(f) for f in files}


@contextlib.contextmanager
def quiet():
    lv = logging.root.manager.disable
    logging.disable(logging.INFO)
    try:
        yield
    finally:
        logging.disable(lv)


# ── inputs ────────────────────────────────────────────────────────────────────

def load_market():
    from joint_vol_calibration.backtest.run_backtest import _load_data, load_pdv_vol
    with quiet():
        spx, vix = _load_data()
        pdv = load_pdv_vol()
    return spx, vix, pdv


def rerun_canonical(spx, vix, pdv, committed: pd.DataFrame) -> bte.BacktestEngine:
    with quiet():
        eng = bte.BacktestEngine(pdv_vol_series=pdv)
        eq = eng.run(spx, vix)
    if list(eq.columns) != list(committed.columns) or not eq.index.equals(committed.index):
        raise SystemExit("canonical re-run: columns or dates differ from full_results.parquet")
    for c in eq.columns:
        a, b = eq[c].to_numpy(float), committed[c].to_numpy(float)
        if not np.allclose(a, b, rtol=0, atol=1e-6, equal_nan=True):
            raise SystemExit(f"canonical re-run differs from full_results.parquet in {c}")
    return eng


def sleeve_returns(eq: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({s: sh.returns_from_nav(eq[nav].to_numpy(float), CAP)
                         for s, (nav, _) in SLEEVES.items()}, index=eq.index)


# ── 1. bootstrap and closed forms ────────────────────────────────────────────

def section_bootstrap(eq: pd.DataFrame, reps: int, seed: int):
    R = sleeve_returns(eq)
    series = {s: {"returns": R[s].to_numpy(), "pnl": eq[p].to_numpy(float)} for s, (_, p) in SLEEVES.items()}
    table, replicates = bs.bootstrap_table(series, reps=reps, seed=seed)
    fixed, _ = bs.bootstrap_table(series, reps=reps, seed=seed, fixed_block=21.0)
    out = {}
    for s in SLEEVES:
        rows = table[table["series"] == s].set_index("statistic")
        frows = fixed[fixed["series"] == s].set_index("statistic")
        out[s] = {
            "block": float(rows["block"].iloc[0]),
            **{k: {"point": float(rows.loc[k, "point"]), "ci95": [float(rows.loc[k, "ci_lo"]),
                                                                  float(rows.loc[k, "ci_hi"])],
                   "se": float(rows.loc[k, "se"]), "frac_gt_0": float(rows.loc[k, "frac_gt_0"]),
                   "excludes_zero": bool(rows.loc[k, "excludes_zero"]),
                   "ci95_block21": [float(frows.loc[k, "ci_lo"]), float(frows.loc[k, "ci_hi"])],
                   "excludes_zero_block21": bool(frows.loc[k, "excludes_zero"])}
               for k in bs.STATISTICS},
        }
    settings = {"reps": reps, "seed": seed, "level": 0.95, "method": "stationary bootstrap, percentile",
                "block_rule": "max of Politis-White(-Patton) for r and r² (arch); seed + i for series i",
                "series_order": list(SLEEVES)}
    return out, settings, replicates


def section_closed_form(eq: pd.DataFrame) -> Dict:
    R = sleeve_returns(eq)
    out = {}
    for s in SLEEVES:
        r = R[s].to_numpy()
        inf = sh.sharpe_inference(r).to_dict()
        inf["newey_west_mean_return_t_lag21"] = sh.newey_west_tstat(r, HAC_LAGS)["t"]
        inf["newey_west_mean_return_t_auto"] = sh.newey_west_tstat(r)["t"]
        out[s] = inf
    return out


# ── 2. the trial family, DSR, SPA/StepM, PBO ─────────────────────────────────

def trial_family(spx, vix, pdv) -> Tuple[pd.DataFrame, List[Dict]]:
    """Daily returns of every reproducible trial: the C22 bridge rows and the
    β grid, each × six sleeves; checked against the committed CSVs."""
    from joint_vol_calibration.backtest import c22_bridge as cb

    cols, meta, checks, index = {}, [], [], []
    bridge_csv = pd.read_csv(bte.BACKTEST_DIR / "c22_bridge.csv")
    beta_csv = pd.read_csv(bte.BACKTEST_DIR / "beta_sensitivity.csv")

    def add(tag, eq, extra):
        if not index:
            index.append(eq.index)
        elif not eq.index.equals(index[0]):
            raise SystemExit(f"trial {tag}: session index differs from the first trial")
        R = sleeve_returns(eq)
        for s in SLEEVES:
            cols[f"{tag}:{s}"] = R[s].to_numpy()
            meta.append({"trial": f"{tag}:{s}", "sleeve": s, **extra})

    state = {"beta": 1.0, "gate": cb.LEGACY_VVIX_GATE, "pdv_h": 1, "legacy_interp": True}
    corr = bte.C22Corrections.legacy()
    for i, step in enumerate(cb.build_steps()):
        state.update(step.get("set", {}))
        if step.get("corr"):
            corr = corr.with_(**step["corr"])
        ctx = cb.legacy_vix_interpolation() if state["legacy_interp"] else contextlib.nullcontext()
        with quiet(), ctx:
            eng = bte.BacktestEngine(atm_to_vix_ratio=state["beta"], corrections=corr,
                                     vvix_threshold=state["gate"], pdv_horizon=state["pdv_h"],
                                     pdv_vol_series=pdv)
            eq = eng.run(spx, vix)
        nav = float(eq["nav"].iloc[-1])
        checks.append(("bridge", i, nav, float(bridge_csv["nav"].iloc[i])))
        add(f"bridge{i:02d}", eq, {"family": "c22_bridge", "row": i, "step": step["key"]})
    for b in tr.load()["reproducible"]["beta_grid"]["betas"]:
        if abs(b - bte.ATM_TO_VIX_RATIO) < 1e-12:
            continue                                      # = bridge row 14 (canonical)
        with quiet():
            eng = bte.BacktestEngine(atm_to_vix_ratio=b, pdv_vol_series=pdv)
            eq = eng.run(spx, vix)
        nav = float(eq["nav"].iloc[-1])
        ref = float(beta_csv.loc[(beta_csv["beta"] - b).abs() < 1e-9, "nav"].iloc[0])
        checks.append(("beta", b, nav, ref))
        add(f"beta{b:.2f}", eq, {"family": "beta_grid", "beta": b})
    # the committed CSVs keep 10 significant digits: a NAV must agree to 1e-8 relative
    bad = [c for c in checks if abs(c[2] - c[3]) > 1e-8 * abs(c[3])]
    if bad:
        raise SystemExit(f"trial family does not reproduce the committed CSVs: {bad[:3]}")
    log.info("trial family reproduces %d committed NAVs (max rel diff %.1e)", len(checks),
             max(abs(c[2] - c[3]) / abs(c[3]) for c in checks))
    return pd.DataFrame(cols, index=index[0]), meta


def beta_estimated_columns(fam: pd.DataFrame) -> List[str]:
    """Trial series priced at an estimated β: C22 bridge rows 11-14 (β = 0.82)
    and the β grid below 1.00. Bridge rows 0-10 and the β = 1.00 grid row price
    ATM options at the VIX itself (the mis-specified pre-C22 mark)."""
    return [c for c in fam.columns
            if (c.startswith("bridge") and int(c[6:8]) >= 11)
            or (c.startswith("beta") and not c.startswith("beta1.00"))]


def section_trials(eq, fam: pd.DataFrame, meta: List[Dict]) -> Dict:
    led = tr.load()
    n_all = tr.count_trials(led, include_evaluations=True)
    n_c16 = tr.count_trials(led, from_round="C16")
    srs = fam.apply(lambda c: sh.sharpe(c.to_numpy()))
    var_sr = float(srs.var(ddof=1))
    est = beta_estimated_columns(fam)
    var_sr_est = float(srs[est].var(ddof=1))
    C = np.corrcoef(fam.to_numpy().T)
    iu = np.triu_indices_from(C, k=1)
    rho = float(np.nanmean(C[iu]))
    n_eff = sh.effective_trials(n_c16["n_trials"], rho)
    lam = np.clip(np.linalg.eigvalsh(np.nan_to_num(C)), 0.0, None)
    participation = float(lam.sum() ** 2 / (lam ** 2).sum())
    # N choices. The average-correlation N_eff is close to the raw count because
    # the family mixes sleeves whose returns are uncorrelated or opposite (S1C is
    # S1's contrarian), which pulls ρ̄ towards 0 even though the configurations
    # of one sleeve are near-copies; the eigenvalue participation ratio of the
    # family's correlation matrix measures that clustering and is the low end.
    choices = {"n_eigen_participation": participation, "n_eff_c16_c22": n_eff,
               "n_raw_c16_c22": float(n_c16["n_trials"]),
               "n_raw_all_rounds": float(n_all["n_trials"]),
               "n_upper_with_evaluations": float(n_all["n_trials_upper"])}
    R = sleeve_returns(eq)
    T = len(R)
    v_null = 1.0 / (T - 1)
    # Trial-variance choices: the whole family's cross-trial Sharpe variance (the
    # paper's V[{SR_n}], which here includes the β = 1 rows' mis-specified
    # mark), the same over the configurations priced at an estimated β, and
    # the variance of SR̂ under H0 for every trial (1/(T − 1)), the most lenient.
    variances = {"family_var": var_sr, "family_var_beta_estimated": var_sr_est, "null_var": v_null}
    dsr = {s: {k: sh.deflated_sharpe_ratio(R[s].to_numpy(), n, var_sr)["dsr"] for k, n in choices.items()}
           for s in SLEEVES}
    sr0 = {k: sh.annualise(sh.expected_max_sharpe(n, var_sr)) for k, n in choices.items()}
    sr0_all = {f"{vname}|{k}": sh.annualise(sh.expected_max_sharpe(n, v))
               for vname, v in variances.items() for k, n in choices.items()}
    grid_n = sorted({1.0, 5.0, 10.0, 30.0, *choices.values()})
    dsr_grid = {s: {f"{vname}|N={n:.6g}": sh.deflated_sharpe_ratio(R[s].to_numpy(), n, v)["dsr"]
                    for vname, v in variances.items() for n in grid_n}
                for s in SLEEVES}
    dsr_range = {}
    for s in SLEEVES:
        vals = {f"{vname}|{k}": sh.deflated_sharpe_ratio(R[s].to_numpy(), n, v)["dsr"]
                for vname, v in variances.items() for k, n in choices.items()}
        lo, hi = min(vals, key=vals.get), max(vals, key=vals.get)
        dsr_range[s] = {"min": vals[lo], "argmin": lo, "max": vals[hi], "argmax": hi}
    return {
        "ledger": {"path": "docs/trials_ledger.yaml", "n_trials_all_rounds": n_all["n_trials"],
                   "n_trials_c16_c22": n_c16["n_trials"], "by_round": n_all["by_round"],
                   "n_evaluations_not_trials": n_all["n_evaluations"]},
        "family": {"n_series": int(fam.shape[1]), "families": sorted({m["family"] for m in meta}),
                   "var_sharpe_per_period": var_sr,
                   "sd_sharpe_annual": float(np.sqrt(var_sr) * np.sqrt(sh.TRADING_DAYS)),
                   "n_series_beta_estimated": len(est),
                   "var_sharpe_per_period_beta_estimated": var_sr_est,
                   "sd_sharpe_annual_beta_estimated": float(np.sqrt(var_sr_est) * np.sqrt(sh.TRADING_DAYS)),
                   "sd_sharpe_annual_under_h0": float(np.sqrt(v_null) * np.sqrt(sh.TRADING_DAYS)),
                   "mean_pairwise_correlation": rho, "n_eff_c16_c22": n_eff,
                   "eigen_participation_ratio": participation},
        "n_choices": choices,
        "variance_choices_per_period": variances,
        "sr0_annual": sr0,
        "sr0_annual_all_choices": sr0_all,
        "dsr": dsr,
        "dsr_range": dsr_range,
        "dsr_sensitivity": {"n_grid": grid_n, "var_null_per_period": v_null,
                            "participation_ratio_family": participation, "values": dsr_grid},
        "note": ("DSR = PSR(SR > SR0), SR0 = expected maximum Sharpe of N skill-less trials with trial "
                 "Sharpe variance V. 'dsr' uses the whole family's V; 'dsr_range' spans every N choice "
                 "(eigenvalue participation of the re-run family, average-correlation N_eff, raw ledger "
                 "counts) and every V choice. N from the committed trials ledger"),
    }


def section_multiple_testing(eq, fam: pd.DataFrame, block: int, reps: int, seed: int) -> Dict:
    R = sleeve_returns(eq)
    port_cols = [c for c in fam.columns if c.endswith(":portfolio")]
    est = beta_estimated_columns(fam)
    return {
        "spa_beta_estimated": mt.spa_vs_cash(fam[est], block, reps, seed),
        "stepm_beta_estimated": mt.stepm_vs_cash(fam[est], block, 0.05, reps, seed),
        "pbo_beta_estimated": mt.pbo_cscv(fam[est], n_splits=16),
        "beta_estimated_series": est,
        "block_size": block,
        "spa_published": mt.spa_vs_cash(R, block, reps, seed),
        "stepm_published": mt.stepm_vs_cash(R, block, 0.05, reps, seed),
        "spa_trial_family": mt.spa_vs_cash(fam, block, reps, seed),
        "stepm_trial_family": mt.stepm_vs_cash(fam, block, 0.05, reps, seed),
        "pbo_portfolio_configs": mt.pbo_cscv(fam[port_cols], n_splits=16),
        "pbo_all_trials": mt.pbo_cscv(fam, n_splits=16),
        "note": ("H0 of SPA: no strategy beats cash (zero excess return); StepM lists strategies that do "
                 "at FWER 5%; PBO = share of CSCV splits whose in-sample best is at or below the "
                 "out-of-sample median. The full trial family contains configurations that price ATM "
                 "options at the VIX (β = 1: bridge rows 0-10 and the β = 1.00 grid row); a rejection "
                 "or a low PBO there that comes from those rows measures the mis-specified mark, not "
                 "a strategy — compare the beta_estimated subset. A low PBO says the in-sample ranking "
                 "persists out of sample; it is not evidence of skill: the configurations differ mostly by "
                 "persistent level effects (the β of the mark, a sleeve's net short- or long-vol "
                 "exposure) that move every sub-period alike and keep ranks stable with or without skill. "
                 "Read it with oos_sharpe_of_selected_mean and oos_on_is_slope (< 0: the in-sample "
                 "winners' out-of-sample Sharpe falls as their in-sample Sharpe rises)"),
    }


# ── 3. timing nulls and the premium/timing split ─────────────────────────────

def section_nulls(eng: bte.BacktestEngine, eq: pd.DataFrame, n_shuffle: int, seed: int):
    mk = eng._market
    replay = StraddleReplay(eng.signals_df.index, mk["spx_close"], mk["vix_idx"], mk["r"],
                            q=eng.q, atm_to_vix_ratio=eng.atm_to_vix_ratio)
    pn = nl.PortfolioNull(eng, replay)
    real = pn.run_path(nl.NullPath("real", -1))
    for s, (_, pcol) in SLEEVES.items():
        if not np.array_equal(real[s], eq[pcol].to_numpy(float)):
            raise SystemExit(f"null machinery does not reproduce the engine's {s} P&L")
    paths = nl.circular_shift_paths(len(eng.signals_df), nl.DEFAULT_MIN_SHIFT, include_reversed=True)
    paths += nl.shuffle_paths(n_shuffle, seed=seed)
    t0 = time.time()
    df = pn.run(paths)
    log.info("null paths: %d in %.0fs", len(paths), time.time() - t0)
    real_stats = pn.statistics(real)
    summary = {}
    for s in real_stats:
        summary[s] = {"real_entries": int(sum(real["_entries"][x] for x in nl.PORTFOLIO))
                      if s == "portfolio" else int(real["_entries"][s])}
        for kind in ("circular", "shuffle"):
            sub = df[(df["sleeve"] == s) & (df["kind"] == kind)]
            summary[s][kind] = {
                "total_pnl": nl.summarise_null(real_stats[s]["total_pnl"], sub["total_pnl"].to_numpy()),
                "sharpe": nl.summarise_null(real_stats[s]["sharpe"], sub["sharpe"].to_numpy()),
                "mean_entries": float(sub["n_entries"].mean()),
            }
    # six timing tests on one window: Holm-adjust each null family across them
    for kind in ("circular", "shuffle"):
        for stat in ("total_pnl", "sharpe"):
            adj = nl.holm({s: summary[s][kind][stat]["p_value"] for s in summary})
            for s, v in adj.items():
                summary[s][kind][stat]["p_holm_6_sleeves"] = v
    # premium factor: an always-on straddle on the same engine and costs
    short = replay.simulate(np.full(len(eq), -1.0), np.full(len(eq), VRP_KELLY), pn.nav_base)
    long_ = replay.simulate(np.full(len(eq), 1.0), np.full(len(eq), VRP_KELLY), pn.nav_base)
    attribution = {}
    for s, (_, pcol) in SLEEVES.items():
        reg = ols_hac(eq[pcol].to_numpy(float), short, lags=HAC_LAGS)
        attribution[s] = {"alpha_per_year": reg["coef"][0] * sh.TRADING_DAYS, "alpha_t": reg["t"][0],
                          "beta_to_always_short": reg["coef"][1], "beta_t": reg["t"][1], "r2": reg["r2"]}
    factor = {"always_short_total_pnl": float(short.sum()), "always_long_total_pnl": float(long_.sum()),
              "always_short_sharpe": sh.annualise(sh.sharpe(sh.returns_from_pnl(short, CAP))),
              "kelly": VRP_KELLY}
    settings = {"circular_min_shift": nl.DEFAULT_MIN_SHIFT,
                "n_circular": int(sum(p.kind == "circular" for p in paths)),
                "n_shuffle": int(n_shuffle), "shuffle_seed": int(seed),
                "statistic": "total P&L (primary) and annualised Sharpe, after the pass-2 overlay for "
                             "S1C/S3/S4/portfolio (re-run per path), base size for S1/S2"}
    return {"settings": settings, "sleeves": summary, "vrp_factor": factor,
            "attribution_vs_always_short": attribution}, df, short


# ── 4. forecasts ──────────────────────────────────────────────────────────────

def section_forecast(reps: int, seed: int) -> Dict:
    from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
    with quiet():
        spx = get_spx_ohlcv(as_of_date="2100-01-01", start_date="2010-01-01")
        vw = get_vix_term_structure_wide(as_of_date="2100-01-01", start_date="2010-01-01")
    close = pd.Series(spx["close"].to_numpy(float), index=pd.to_datetime(spx["date"]))
    r = np.log(close).diff().dropna()
    vix = pd.Series(vw["^VIX"].to_numpy(float), index=pd.to_datetime(vw["date"]))
    with quiet():
        F = fc.build_forecasts(r, vix)
    y2 = fc.target_variance(r)
    out = {"data": {"spx_first": str(close.index[0].date()), "spx_last": str(close.index[-1].date()),
                    "n_returns": int(len(r))},
           "target": "y²(t) = 252/21 Σ_{i=1..21} r²_{t+i}, annualised realised variance of the next 21 sessions",
           "models": {
               "pdv": "PDVLinear walk-forward (models.pdv.walk_forward_vol_forecast), σ̂², the S1/S1C input",
               "har": "HAR-RV on daily r², expanding OLS, purged (labels i ≤ t − 21)",
               "garch": "GARCH(1,1) zero mean, Gaussian QML (arch), re-estimated every 21 sessions",
               "gjr": "GJR-GARCH(1,1,1), as garch",
               "ewma": "RiskMetrics λ = 0.94",
               "vix2": "(VIX/100)², risk-neutral (contains the variance premium)",
               "vix_mz": "walk-forward OLS of y² on [1, (VIX/100)²], purged"}}
    def _strip(ev):
        ev = dict(ev)
        ev.pop("_loss_frames", None)
        return ev
    out["full_sample"] = _strip(fc.evaluate(F, y2, mcs_reps=reps, seed=seed))
    win = (F.index >= pd.Timestamp(bte.BT_START)) & (F.index <= pd.Timestamp(bte.BT_END))
    out["backtest_window"] = _strip(fc.evaluate(F[win], y2, mcs_reps=reps, seed=seed))
    common = pd.concat([F, y2.rename("_y")], axis=1).dropna()
    sub = fc.non_overlapping(common, fc.HORIZON)
    out["non_overlapping"] = _strip(fc.evaluate(sub.drop(columns="_y"), sub["_y"], horizon=1,
                                                mcs_reps=reps, seed=seed, mcs_block=1))
    return out


# ── 5. Cboe benchmarks ────────────────────────────────────────────────────────

def section_benchmarks(eq: pd.DataFrame, vrp_short: np.ndarray) -> Dict:
    from joint_vol_calibration.backtest.delta_hedger import _interp_atm_iv
    from joint_vol_calibration.data.database import (get_spx_ohlcv, get_tbill_rates_series,
                                                     get_vix_term_structure_wide)

    levels = bm.load_levels(BENCH_DIR, bm.BENCHMARK_INDICES)
    if levels.empty:
        raise SystemExit(f"no benchmark files in {BENCH_DIR}: run scripts/fetch_cboe_benchmarks.py")
    setv = bm.load_levels(BENCH_DIR, ("SET",))["SET"]
    dates = eq.index
    tb = eq["nav_tbill"].to_numpy(float)
    cash = pd.Series(tb / np.r_[CAP, tb[:-1]] - 1.0, index=dates)
    ex = bm.benchmark_excess_returns(levels, dates, cash)
    with quiet():
        spx = get_spx_ohlcv(as_of_date=bte.BT_END, start_date="2017-01-01")
        vw = get_vix_term_structure_wide(as_of_date=bte.BT_END, start_date="2017-01-01")
        tbr = get_tbill_rates_series(as_of_date=bte.BT_END, start_date="2017-01-01")
    spx = spx.set_index(pd.to_datetime(spx["date"]))
    vw = vw.set_index(pd.to_datetime(vw["date"]))
    rates = pd.Series(tbr["rate"].to_numpy(float), index=pd.to_datetime(tbr["date"]))
    spx_ret = spx["close"].reindex(dates) / spx["close"].shift(1).reindex(dates) - 1.0
    S = sleeve_returns(eq)
    S["always_short_straddle"] = sh.returns_from_pnl(vrp_short, CAP)
    years = list(range(pd.Timestamp(bte.BT_START).year, pd.Timestamp(bte.BT_END).year + 1))
    tab = bm.strategy_benchmark_stats(S, ex, spx_ret, years)
    nested: Dict = {}
    for _, row in tab.iterrows():
        nested.setdefault(row["sleeve"], {}).setdefault(row["benchmark"], {})[row["period"]] = {
            k: row[k] for k in tab.columns if k not in ("sleeve", "benchmark", "period")}
    out = {"window": [bte.BT_START, bte.BT_END], "indices": list(levels.columns),
           "excess_returns": "index return over the backtest's own session intervals minus the T-bill "
                             "accrual of the backtest's cash benchmark (nav_tbill)",
           "co_movement": nested, "years": years,
           "partial_years": {str(pd.Timestamp(bte.BT_END).year):
                             f"{pd.Timestamp(bte.BT_END).year}-01-01..{bte.BT_END}"},
           "fields": {"corr_daily": "Pearson correlation of daily excess returns",
                      "corr_weekly": "same on W-FRI compounded weeks",
                      "partial_corr_ex_spx": "correlation after a linear regression of both on the SPX "
                                             "daily return (removes linear delta, not gamma)",
                      "beta": "OLS slope of sleeve on benchmark daily excess returns",
                      "alpha_ann / alpha_t": "intercept × 252 (excess return per year) and its "
                                             "Newey-West t",
                      "sharpe_diff": "sleeve Sharpe minus benchmark Sharpe (annualised, rf = 0 on "
                                     "excess returns)"},
           "always_short_straddle": "the backtest engine's delta-hedged 30-day ATM straddle held short "
                                    "every session at 0.10 size: the mark-to-model premium factor"}

    # implied β from PUT (monthly, AM-settled at SET) and WPUT (weekly, PM)
    sessions = spx.index[(spx.index >= "2017-12-01") & (spx.index <= bte.BT_END)]
    def _implied(index_name, rolls, settle, fallback, hour):
        per = bm.put_write_periods(rolls, spx["close"], settle, vw, rates, bte.Q_BACKTEST,
                                   expiry_hour_et=hour, settle_fallback=fallback)
        lv = levels[index_name]
        R = lv.reindex(per["expiry"]).to_numpy() / lv.reindex(per["roll"]).to_numpy() - 1.0
        ok = np.isfinite(R)
        per, R = per[ok].reset_index(drop=True), R[ok]
        vt = np.array([_interp_atm_iv(vw.loc[a], T) for a, T in zip(per["roll"], per["T"])])
        yrs = per["expiry"].dt.year.to_numpy()
        ib = bm.implied_beta(per, vt, R, groups={str(y): yrs == y for y in sorted(set(yrs))})
        pp = ib["per_period"]
        rep = bm.period_returns(per, bte.ATM_TO_VIX_RATIO * vt)
        full_years = {k: v for k, v in ib["groups"].items() if int(k) < pd.Timestamp(bte.BT_END).year}
        vals = np.array(list(full_years.values()), dtype=float)
        # How much of the year-to-year spread is estimation noise: the within-year
        # spread of the per-period β over √(periods) approximates a year's
        # standard error (a year's β is close to a vega-weighted average of them)
        per_year = {}
        for y in sorted(set(yrs)):
            v = pp[(yrs == y) & np.isfinite(pp)]
            per_year[str(y)] = {"n_periods": int(v.size), "per_period_median": float(np.median(v)),
                                "se_approx": float(np.std(v, ddof=1) / np.sqrt(v.size)) if v.size > 1
                                else float("nan")}
        se_full = np.array([per_year[k]["se_approx"] for k in full_years], dtype=float)
        vfn = lambda d, T: _interp_atm_iv(vw.loc[d], T)         # noqa: E731
        daily = {}
        for label, b in (("beta_0.82", bte.ATM_TO_VIX_RATIO), ("beta_implied_overall", ib["overall"])):
            lvl = bm.replication_daily(per, sessions, spx["close"], vfn, rates, b, bte.Q_BACKTEST,
                                       expiry_hour_et=hour)
            dr = lvl.pct_change().dropna()
            ir = lv.reindex(lvl.index).pct_change().reindex(dr.index)
            st = bm.pair_stats(dr, ir, spx["close"].pct_change().reindex(dr.index))
            daily[label] = {k: st[k] for k in ("n_days", "corr_daily", "corr_weekly", "partial_corr_ex_spx",
                                               "beta", "sharpe_sleeve", "sharpe_bench")}
        return {
            "n_periods": int(len(per)), "first_roll": str(per["roll"].iloc[0].date()),
            "last_expiry": str(per["expiry"].iloc[-1].date()),
            "settle_sources": per["settle_source"].value_counts().to_dict(),
            "overall": ib["overall"], "by_year": ib["groups"], "by_year_per_period": per_year,
            "full_year_dispersion": {"min": float(np.nanmin(vals)), "max": float(np.nanmax(vals)),
                                     "sd": float(np.nanstd(vals, ddof=1)), "n_years": int(len(vals)),
                                     "median_se_approx": float(np.nanmedian(se_full))},
            "per_period": {"median": float(np.nanmedian(pp)), "q25": float(np.nanpercentile(pp, 25)),
                           "q75": float(np.nanpercentile(pp, 75)), "sd": float(np.nanstd(pp, ddof=1)),
                           "n_unsolved": int(np.isnan(pp).sum())},
            "replication_at_0.82": {"period_return_corr": float(np.corrcoef(rep, R)[0, 1]),
                                    "mean_period_diff_bp": float((rep - R).mean() * 1e4),
                                    "tracking_error_bp_per_period": float((rep - R).std(ddof=1) * 1e4)},
            "daily": daily,
        }

    mrolls = bm.monthly_roll_dates(sessions)
    mrolls = mrolls[(mrolls >= "2017-12-15") & (mrolls <= bte.BT_END)]
    wrolls = bm.weekly_roll_dates(sessions)
    wrolls = wrolls[(wrolls >= "2017-12-29") & (wrolls <= bte.BT_END)]
    out["implied_beta"] = {
        "PUT": _implied("PUT", mrolls, setv, spx["open"], 9.5),
        "WPUT": _implied("WPUT", wrolls, spx["close"], None, 16.0),
        "canonical_beta": bte.ATM_TO_VIX_RATIO,
        "method": ("cash-secured put-write replication with puts at σ = β·VIX-TS(T): strike = SPX close "
                   "on the roll date, sold at that close; monthly puts settle at Cboe's SET value "
                   "(Yahoo open where SET has no row), weekly puts at the close; collateral earns ^IRX. "
                   "β solves replication return = index return per period, per expiry year and overall. "
                   "The indices sell at a VWAP and PUT's collateral earns 1- and 3-month bills, so this "
                   "is an approximation; 2025 covers January-March only."),
    }
    return out


# ── assembly ──────────────────────────────────────────────────────────────────

def _ci_verdict(lo: float, hi: float) -> str:
    return "excludes zero (negative)" if hi < 0 else "excludes zero (positive)" if lo > 0 else "includes zero"


def _years(x) -> str:
    return "n/a (SR <= 0)" if x is None or not np.isfinite(x) else f"{x:.0f} years"


def headline(summary: Dict, fe: Dict, bmk: Dict) -> List[str]:
    """Plain statements generated from the numbers (no hand-typed values)."""
    lines = []
    b = summary["bootstrap"]
    for s in SLEEVES:
        x, mp = b[s]["sharpe"], b[s]["mean_daily_pnl"]
        lo, hi = x["ci95"]
        plo, phi = mp["ci95"]
        lines.append(f"{s}: Sharpe {x['point']:.2f}, 95% stationary-bootstrap CI [{lo:.2f}, {hi:.2f}] "
                     f"{_ci_verdict(lo, hi)}; mean daily P&L ${mp['point']:,.0f} [{plo:,.0f}, {phi:,.0f}] "
                     f"{_ci_verdict(plo, phi)}")
    t = summary["trials"]
    cf = summary["closed_form"]
    nc = t["n_choices"]
    lines.append("PSR(SR>0) and the largest DSR over every N and trial-variance choice: "
                 + "; ".join(f"{s} {cf[s]['psr_gt_0']:.2f} / {t['dsr_range'][s]['max']:.3f}" for s in SLEEVES)
                 + f" (N from {nc['n_eigen_participation']:.1f}, the eigenvalue participation of the "
                   f"{t['family']['n_series']}-series re-run family, to {nc['n_upper_with_evaluations']:.0f}; "
                   f"ledger C16-C22 {t['ledger']['n_trials_c16_c22']}, all rounds {t['ledger']['n_trials_all_rounds']})")
    s3 = t["dsr_sensitivity"]["values"]["s3"]
    lines.append(f"s3: DSR {t['dsr']['s3']['n_raw_c16_c22']:.3f} at the ledger's N "
                 f"{t['ledger']['n_trials_c16_c22']} with the family's trial-Sharpe sd "
                 f"{t['family']['sd_sharpe_annual']:.2f}/yr (SR0 {t['sr0_annual']['n_raw_c16_c22']:.2f}); "
                 f"{s3['null_var|N=' + format(nc['n_raw_c16_c22'], '.6g')]:.3f} with the H0 sd "
                 f"{t['family']['sd_sharpe_annual_under_h0']:.2f}/yr; "
                 f"{s3['null_var|N=' + format(nc['n_eigen_participation'], '.6g')]:.2f} at N "
                 f"{nc['n_eigen_participation']:.1f} with the H0 sd; MinTRL {_years(cf['s3']['min_trl_years'])} "
                 f"at the observed SR and moments")
    m = summary["multiple_testing"]
    lines.append(f"Hansen SPA (the 6 published series vs cash): consistent p {m['spa_published']['p_consistent']:.2f}; "
                 f"Romano-Wolf StepM superior: {m['stepm_published']['superior'] or 'none'}")
    lines.append(f"Over the {m['spa_beta_estimated']['n_strategies']} trial series priced at an estimated beta: "
                 f"SPA p {m['spa_beta_estimated']['p_consistent']:.2f}, StepM superior "
                 f"{m['stepm_beta_estimated']['superior'] or 'none'}; over all "
                 f"{m['spa_trial_family']['n_strategies']} (incl. beta = 1, the mis-specified mark): "
                 f"SPA p {m['spa_trial_family']['p_consistent']:.3f}, StepM superior "
                 f"{m['stepm_trial_family']['superior'] or 'none'}")
    for key, label in (("pbo_beta_estimated", "beta-estimated trials"), ("pbo_portfolio_configs", "portfolio configs"),
                       ("pbo_all_trials", "all trials")):
        p = m[key]
        lines.append(f"CSCV PBO ({label}, {p['n_configurations']} series, {p['n_combinations']} splits): "
                     f"{p['pbo']:.2f}; OOS-on-IS Sharpe slope {p['oos_on_is_slope']:.2f}; selected config's "
                     f"OOS Sharpe {p['oos_sharpe_of_selected_mean'] * np.sqrt(sh.TRADING_DAYS):.2f}/yr "
                     f"(IS {p['is_sharpe_of_selected_mean'] * np.sqrt(sh.TRADING_DAYS):.2f})")
    for s in SLEEVES:                     # fixed order: the JSON file sorts its keys
        v = summary["nulls"]["sleeves"][s]
        c = v["circular"]["total_pnl"]
        sf = v["shuffle"]["total_pnl"]
        lines.append(f"{s}: timing vs {c['n_paths']} circular shifts p {c['p_value']:.3f} (Holm over 6: "
                     f"{c['p_holm_6_sleeves']:.2f}); real ${c['real']:,.0f} = exposure at random timing "
                     f"${c['null_mean']:,.0f} + timing ${c['timing_component']:,.0f}; trade shuffle "
                     f"({sf['n_paths']} paths) p {sf['p_value']:.3f} (Holm {sf['p_holm_6_sleeves']:.2f})")
    at = summary["nulls"]["attribution_vs_always_short"]
    lines.append("Daily P&L on the always-short straddle (HAC t, 21 lags): "
                 + "; ".join(f"{s} alpha ${at[s]['alpha_per_year']:,.0f}/yr (t {at[s]['alpha_t']:.2f}), "
                             f"beta {at[s]['beta_to_always_short']:.2f} (t {at[s]['beta_t']:.2f}, "
                             f"R2 {at[s]['r2']:.2f})" for s in SLEEVES))
    for sample in ("full_sample", "backtest_window"):
        q = fe[sample]
        ql = q["losses"]["qlike"]
        best = min(ql, key=ql.get)
        dm = q["dm"]["qlike"]
        sig = sorted(k for k, v in dm.items() if v["rectangular_hln"]["p_two_sided"] < 0.05)
        lines.append(f"21-day variance forecasts, {sample} {q['start']}..{q['end']} (n {q['n']}): QLIKE PDV "
                     f"{ql['pdv']:.3f}, rank {q['rank']['qlike']['pdv']}/{len(ql)} (best {best} {ql[best]:.3f}); "
                     f"MSE rank {q['rank']['mse']['pdv']}/{len(ql)}; DM (HLN) vs PDV significant at 5%: "
                     f"{sig or 'none'}; 90% MCS (QLIKE) {q['mcs']['qlike']['90']['included']}")
    for name in ("PUT", "WPUT"):
        ib = bmk["implied_beta"][name]
        fy = ib["full_year_dispersion"]
        lines.append(f"{name}-implied beta overall {ib['overall']:.3f}; full years {fy['min']:.2f}-{fy['max']:.2f} "
                     f"(sd {fy['sd']:.2f}; typical within-year se {fy['median_se_approx']:.2f}); "
                     f"replication at beta {bmk['implied_beta']['canonical_beta']}: daily corr "
                     f"{ib['daily']['beta_0.82']['corr_daily']:.3f} with the index")
    cm = bmk["co_movement"]["always_short_straddle"]
    lines.append("Always-short straddle (the mark's premium factor) daily corr / corr ex-SPX with "
                 + "; ".join(f"{bn}: " + ", ".join(f"{p} {cm[bn][p]['corr_daily']:.2f}/{cm[bn][p]['partial_corr_ex_spx']:.2f}"
                                                  for p in ("all", "2018", "2020", "2022"))
                             for bn in ("WPUT", "PUT", "BFLY")))
    return lines


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=pathlib.Path, default=OUT_DIR)
    ap.add_argument("--reps", type=int, default=10_000, help="bootstrap / SPA / MCS replications")
    ap.add_argument("--shuffles", type=int, default=2_000, help="trade-shuffle null paths")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
    t_all = time.time()

    committed = pd.read_parquet(bte.RESULTS_PATH)
    spx, vix, pdv = load_market()
    eng = rerun_canonical(spx, vix, pdv, committed)
    log.info("canonical backtest reproduced")

    boot, boot_settings, replicates = section_bootstrap(committed, args.reps, args.seed)
    closed = section_closed_form(committed)
    log.info("bootstrap done")
    fam, meta = trial_family(spx, vix, pdv)
    # drop exact duplicates (none expected: β = 0.82 is skipped above)
    fam = fam.loc[:, ~fam.T.duplicated()]
    log.info("trial family: %d series", fam.shape[1])
    trials_sec = section_trials(committed, fam, meta)
    block = int(round(np.median([boot[s]["block"] for s in SLEEVES])))
    mtest = section_multiple_testing(committed, fam, block, min(args.reps, 5000), args.seed)
    log.info("SPA / StepM / PBO done")
    nulls_sec, null_df, vrp_short = section_nulls(eng, committed, args.shuffles, args.seed)
    log.info("nulls done")
    fe = section_forecast(min(args.reps, 5000), args.seed)
    log.info("forecast evaluation done")
    bmk = section_benchmarks(committed, vrp_short)
    log.info("benchmarks done")

    inputs = {"full_results.parquet": artifacts.sha256_file(bte.RESULTS_PATH),
              "vol_system.db": artifacts.sha256_file(bte.DATA_DIR / "vol_system.db"),
              "trials_ledger.yaml": artifacts.sha256_file(tr.LEDGER_PATH)}
    for p in sorted(BENCH_DIR.glob("*.csv")):
        inputs[f"benchmarks/{p.name}"] = artifacts.sha256_file(p)
    summary = {
        "schema": "inference_summary/1",
        "code_base_commit": git_head(),
        "code_sha256": code_sha256(),
        "inputs_sha256": inputs,
        "window": [str(committed.index[0].date()), str(committed.index[-1].date())],
        "n_sessions": int(len(committed)),
        "caveats": [
            "Every P&L here is the backtest's mark-to-model P&L (Black-Scholes at beta x VIX-TS, constant "
            "beta 0.82); the inference is about that proxy, not about executable option P&L",
            "One window, 2018-01-02..2025-03-24; nothing here is out of sample for the rules, which were "
            "revised on this window over the rounds in docs/trials_ledger.yaml",
            "The closed forms (PSR, DSR, MinTRL) assume IID returns; the stationary bootstrap is the "
            "primary interval and the closed forms sit beside it",
            "Timing nulls condition on each sleeve's own exposure (position and size series): they test "
            "when it trades, not whether the short- or long-vol exposure itself earns a premium",
        ],
        "conventions": {"returns": "daily NAV returns from the $1M base (backtest_engine.compute_metrics); "
                                   "excess over T-bills, rf = 0",
                        "annualisation": "× √252", "pnl": "mark-to-model (β·VIX), as the backtest"},
        "bootstrap_settings": boot_settings,
        "bootstrap": boot,
        "closed_form": closed,
        "trials": trials_sec,
        "multiple_testing": mtest,
        "nulls": nulls_sec,
    }
    summary["headline"] = headline(summary, fe, bmk)
    args.out.mkdir(parents=True, exist_ok=True)
    artifacts.write_json(summary, args.out / "summary.json")
    artifacts.write_parquet(replicates[["series", "rep", "sharpe", "sortino", "mean_daily_pnl"]],
                            args.out / "bootstrap.parquet")
    artifacts.write_parquet(null_df.sort_values(["kind", "path_id", "sleeve"]), args.out / "nulls.parquet")
    artifacts.write_json(fe, args.out / "forecast_eval.json")
    artifacts.write_json(bmk, args.out / "benchmarks.json")
    for line in summary["headline"]:
        print(line)
    log.info("wrote %s in %.0fs", args.out, time.time() - t_all)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
