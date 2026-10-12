"""
run_backtest.py — C10: Live Backtest Runner

Loads real 2018-2025 data, runs BacktestEngine, runs the S3 timing null
test and the sub-period stability windows, generates full_report.html, and
saves results to parquet.

Usage (from the repository root):
  python -m joint_vol_calibration.backtest.run_backtest
  python -m joint_vol_calibration.backtest.run_backtest --no-wf     # skip sub-period windows
  python -m joint_vol_calibration.backtest.run_backtest --no-save   # print only, write nothing
  python -m joint_vol_calibration.backtest.run_backtest --estimate-beta [--live]
  python -m joint_vol_calibration.backtest.run_backtest --beta-sensitivity
  python -m joint_vol_calibration.backtest.run_backtest --pseudo-oos
  python -m joint_vol_calibration.backtest.run_backtest --c7        # re-run the C7 delta-hedge study
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import time
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("run_backtest")

# ── Imports ───────────────────────────────────────────────────────────────────
from joint_vol_calibration.config import DB_PATH
from joint_vol_calibration.data.database import (
    get_spx_ohlcv,
    get_vix_term_structure_wide,
)
from joint_vol_calibration.backtest.backtest_engine import (
    ATM_TO_VIX_RATIO,
    Q_BACKTEST,
    BacktestEngine,
    estimate_atm_to_vix_ratio,
    s3_timing_null_test,
)
from joint_vol_calibration.backtest.report_generator import ReportGenerator


# ── Data range ────────────────────────────────────────────────────────────────
# Load from 2015 so signal rolling windows (252-day lookbacks) are warm by 2018
DATA_START = "2015-01-01"
BT_END     = "2025-03-24"

# A chain's parity-implied spot must match its session close this closely
# before its quotes are compared with that session's VIX close.
SNAPSHOT_SPOT_TOL = 0.005


def _load_data(start: str = DATA_START, end: str = BT_END) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load SPX OHLCV and VIX wide from SQLite."""
    logger.info("Loading SPX OHLCV (%s → %s) …", start, end)
    spx_df = get_spx_ohlcv(as_of_date=end, start_date=start)
    logger.info("  SPX rows: %d  (%.4f → %.4f)", len(spx_df),
                spx_df["close"].iloc[0], spx_df["close"].iloc[-1])

    logger.info("Loading VIX term structure (%s → %s) …", start, end)
    vix_wide = get_vix_term_structure_wide(as_of_date=end, start_date=start)
    logger.info("  VIX rows: %d  cols: %s", len(vix_wide), list(vix_wide.columns))

    return spx_df, vix_wide


# ── β = ATM vol / VIX estimation ──────────────────────────────────────────────

def _read_only_db() -> sqlite3.Connection:
    """Read-only connection (database._connect switches the file to WAL mode)."""
    return sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)


def _db_chain_snapshots() -> List[Tuple[str, str, pd.DataFrame]]:
    """Every SPX chain stored in the DB: (source, snapshot_date, quotes)."""
    out = []
    with _read_only_db() as con:
        tables = {r[0] for r in con.execute("select name from sqlite_master where type='table'")}
        if "options_snapshots" in tables:
            for (d,) in con.execute("select distinct snapshot_date from options_snapshots "
                                    "where underlying='SPX' order by 1"):
                q = pd.read_sql("select expiry, strike, right, bid, ask from options_snapshots "
                                "where underlying='SPX' and snapshot_date=?", con, params=[d])
                out.append(("options_snapshots", d, q))
        # spx_options is the table the daily bot used to write by mistake
        # (C21); nothing reads it, but its quotes are real.
        if "spx_options" in tables:
            for (d,) in con.execute("select distinct snapshot_date from spx_options order by 1"):
                q = pd.read_sql("select expiry, strike, right, bid, ask from spx_options "
                                "where snapshot_date=?", con, params=[d])
                out.append(("spx_options", d, q))
    return out


def _session_market(snapshot_date: str) -> Optional[Dict]:
    """Last session on/before snapshot_date with its ^VIX, SPX close and ^IRX."""
    with _read_only_db() as con:
        row = con.execute(
            "select v.date, v.close, s.close from vix_term_structure v "
            "join spx_ohlcv s on s.date = v.date "
            "where v.ticker='^VIX' and v.date <= ? order by v.date desc limit 1",
            [snapshot_date]).fetchone()
        if row is None:
            return None
        r = con.execute("select rate from tbill_rates where date <= ? order by date desc limit 1",
                        [row[0]]).fetchone()
    return {"session": row[0], "vix": float(row[1]), "spx": float(row[2]),
            "r": float(r[0]) if r else 0.04}


def fetch_live_spx_chain(min_days: int = 7, max_days: int = 75) -> Tuple[pd.DataFrame, pd.Timestamp, float, float, str]:
    """
    Live ^SPX chain (SPXW, PM-settled) and the ^VIX print closest in time.

    Returns (quotes, as_of [naive US/Eastern], vix, r, note). Outside regular
    hours Yahoo serves Cboe global-trading-hours quotes for both SPXW options
    and VIX, so the pair is still time-matched; the note records the gap.
    """
    import yfinance as yf

    now = pd.Timestamp.now(tz="America/New_York")
    t = yf.Ticker("^SPX")
    rows = []
    for e in t.options:
        dte = (pd.Timestamp(e).tz_localize("America/New_York") - now).days
        if dte < min_days or dte > max_days:
            continue
        ch = t.option_chain(e)
        for right, df in (("C", ch.calls), ("P", ch.puts)):
            d = df[df["contractSymbol"].str.startswith("SPXW")]
            rows.append(pd.DataFrame({"expiry": e, "strike": d["strike"], "right": right,
                                      "bid": d["bid"], "ask": d["ask"]}))
    chain = pd.concat(rows, ignore_index=True)
    v = yf.Ticker("^VIX").history(period="1d", interval="1m")
    vix_ts = v.index[-1].tz_convert("America/New_York")
    irx = yf.Ticker("^IRX").history(period="5d")["Close"]
    note = f"VIX print {vix_ts:%Y-%m-%d %H:%M} ET, chain {now:%Y-%m-%d %H:%M} ET"
    return chain, now.tz_localize(None), float(v["Close"].iloc[-1]), float(irx.iloc[-1]) / 100.0, note


def estimate_beta(include_live: bool = True) -> pd.DataFrame:
    """
    β = σ_ATM,30d / VIX for every real SPX chain available.

    DB chains are matched to the last session on/before their snapshot date
    (weekend snapshots hold the prior session's closing quotes) and kept only
    if the chain's parity-implied spot is within SNAPSHOT_SPOT_TOL of that
    session's SPX close — otherwise the quotes are from another time and the
    session's VIX close is not the same-time VIX.
    """
    rows = []
    for source, sd, quotes in _db_chain_snapshots():
        mk = _session_market(sd)
        rec = {"source": f"{source}:{sd}"}
        if mk is None:
            rows.append({**rec, "status": "excluded: no VIX session"})
            continue
        try:
            est = estimate_atm_to_vix_ratio(
                quotes, pd.Timestamp(mk["session"]) + pd.Timedelta(hours=16),
                mk["vix"], mk["r"], Q_BACKTEST)
        except ValueError as exc:
            rows.append({**rec, "session": mk["session"], "status": f"excluded: {exc}"})
            continue
        gap = est["implied_spot"] / mk["spx"] - 1.0
        ok = abs(gap) <= SNAPSHOT_SPOT_TOL
        rows.append({**rec, "session": mk["session"], **est, "session_close": mk["spx"],
                     "spot_gap": gap,
                     "status": "included" if ok else
                     f"excluded: implied spot {est['implied_spot']:.0f} vs close "
                     f"{mk['spx']:.0f} ({gap:+.1%}) — quotes not from this session"})
    if include_live:
        try:
            chain, as_of, vix, r, note = fetch_live_spx_chain()
            est = estimate_atm_to_vix_ratio(chain, as_of, vix, r, Q_BACKTEST)
            rows.append({"source": "yahoo-live ^SPX", "session": str(as_of)[:16], **est,
                         "status": f"included ({note})"})
        except Exception as exc:          # network / closed market
            rows.append({"source": "yahoo-live ^SPX", "status": f"unavailable: {exc}"})
    return pd.DataFrame(rows)


def _print_beta(table: pd.DataFrame) -> None:
    cols = [c for c in ["source", "session", "atm30", "vix", "ratio", "implied_spot",
                        "session_close", "status"] if c in table.columns]
    with pd.option_context("display.width", 200, "display.max_colwidth", 90):
        print(table[cols].round(4).to_string(index=False))
    inc = table[table["status"].str.startswith("included")]
    if len(inc):
        print(f"\n  β estimate: mean {inc['ratio'].mean():.4f}  median {inc['ratio'].median():.4f}"
              f"  sd {inc['ratio'].std(ddof=1):.4f}  n={len(inc)}"
              f"  (engine default ATM_TO_VIX_RATIO = {ATM_TO_VIX_RATIO:.2f})")


# ── Sensitivity / pseudo-OOS runners ──────────────────────────────────────────

def run_beta_sensitivity(
    spx_df: pd.DataFrame,
    vix_wide: pd.DataFrame,
    betas: Sequence[float] = (1.00, 0.90, ATM_TO_VIX_RATIO),
) -> pd.DataFrame:
    """Re-run the backtest for each β; one row per β with headline and per-signal stats."""
    rows = []
    for b in betas:
        eng = BacktestEngine(atm_to_vix_ratio=b)
        eq = eng.run(spx_df, vix_wide)
        m = eng.compute_metrics()
        row = {"beta": b, "nav": float(eq["nav"].iloc[-1]),
               "pnl": float(eq["nav"].iloc[-1] - eng.initial_capital),
               "sharpe": m["sharpe"], "max_dd": m["max_drawdown"]}
        for s in ("s1c", "s3", "s4", "s1", "s2"):
            row[f"{s}_pnl"] = m.get(f"{s}_total_pnl", 0.0)
            row[f"{s}_sharpe"] = m.get(f"{s}_sharpe")
        rows.append(row)
    return pd.DataFrame(rows)


def run_pseudo_oos(start: str = "2013-01-01", end: str = "2017-12-31") -> Dict:
    """
    S1C on 2013-2017 (before the 2018-2025 window it was discovered on).

    Confounded: 2013-2017 was the strongest systematic short-vol era on
    record, so a profit here may be short-vol beta, not contrarian alpha.
    """
    spx, vix = _load_data(start="2010-01-01", end=end)
    eng = BacktestEngine(start_date=start, end_date=end)
    eng.run(spx, vix)
    m = eng.compute_metrics()
    return {k: m.get(f"s1c_{k}") for k in ("total_pnl", "sharpe", "win_rate", "n_trades")}


def run_c7(save: bool = True) -> Dict:
    """Re-run the C7 delta-hedge study and (optionally) save parquet + chart."""
    from joint_vol_calibration.backtest.delta_hedger import DeltaHedger
    h = DeltaHedger()
    df = h.run()
    if save:
        h.save()
        h.plot()
    m = h.metrics()
    m["pdv_train_end"] = df.attrs.get("pdv_train_end")
    return m


# ── Printing ──────────────────────────────────────────────────────────────────

def _print_summary(metrics: dict, crisis: dict) -> None:
    """Print key metrics to stdout."""
    m = metrics
    sep = "─" * 60

    def _p(x):  return f"{x*100:.2f}%"   if x is not None and pd.notna(x) else "—"
    def _f(x):  return f"{x:.3f}"         if x is not None and pd.notna(x) else "—"
    def _d(x):  return f"${x:,.0f}"       if x is not None and pd.notna(x) else "—"
    def _i(x):  return str(int(x))         if x is not None and pd.notna(x) else "—"

    print(f"\n{sep}")
    print("  C10 BACKTEST RESULTS — 2018-01-01 → 2025-03-24")
    print(f"  β (ATM vol / VIX) = {m.get('atm_to_vix_ratio', float('nan')):.2f}")
    print(sep)
    print(f"  Cumulative return :  {_p(m.get('cumulative_return'))}")
    print(f"  T-bill cash       :  {_p(m.get('cash_cum_return'))}  (collateral benchmark)")
    print(f"  Ann. return       :  {_p(m.get('ann_return'))}")
    print(f"  Sharpe (rf=0)     :  {_f(m.get('sharpe'))}  excess P&L; collateral earns T-bills")
    print(f"  Sortino (rf=0)    :  {_f(m.get('sortino'))}")
    print(f"  Max Drawdown      :  {_p(m.get('max_drawdown'))}")
    print(f"  DD Duration       :  {_i(m.get('dd_duration_days'))} days")
    print(f"  Calmar            :  {_f(m.get('calmar'))}")
    print(f"  Win Rate (S1C/S3/S4 trades) :  {_p(m.get('win_rate'))}")
    print(f"  Avg P&L / trade   :  {_d(m.get('avg_pnl_per_trade'))}")
    print(f"  Best day          :  {_p(m.get('best_day'))}")
    print(f"  Worst day         :  {_p(m.get('worst_day'))}")
    print(f"  N trades          :  {_i(m.get('n_trades'))}")
    print(f"  N days            :  {_i(m.get('n_days'))}")
    print(sep)

    print("\n  Per-signal breakdown:")
    hdr = f"  {'Signal':<12} {'Sharpe':>8} {'AnnRet':>8} {'WinRate':>9} {'NTrades':>9} {'TotalP&L':>12}"
    print(hdr)
    print("  " + "─" * (len(hdr) - 2))
    for sig, lbl in [("s1c","S1C Contr"),("s3","S3 Disp"),("s4","S4 VRP"),("s1","S1 ref"),("s2","S2 ref"),("combined","Combined")]:
        sh = m.get(f"{sig}_sharpe")
        ar = m.get(f"{sig}_ann_return")
        wr = m.get(f"{sig}_win_rate")
        nt = m.get(f"{sig}_n_trades")
        tp = m.get(f"{sig}_total_pnl")
        print(f"  {lbl:<12} {_f(sh):>8} {_p(ar):>8} {_p(wr):>9} {_i(nt):>9} {_d(tp):>12}")

    print("\n  Crisis performance (Sharpe rf=0):")
    for name, d in crisis.items():
        if d.get("n_days", 0) == 0:
            print(f"  {name:<20} — no data")
            continue
        tr = d.get("total_return", 0) * 100
        dd = d.get("max_drawdown", 0) * 100
        sh = d.get("sharpe", 0)
        sign = "+" if tr >= 0 else ""
        print(f"  {name:<20}  return={sign}{tr:.1f}%  maxDD={dd:.1f}%  Sharpe={sh:.2f}")
    print(sep)


def _print_null(null: Dict) -> None:
    print(f"\n  S3 timing null test ({null['method']}, {null['n_paths']} paths, seed {null['seed']}):")
    print(f"  real trades {null['real_trades']}  vs null mean {null['null_mean_trades']:.1f}")
    for key, lbl in (("index_leg", "S3 index leg (short SPX straddle)"),
                     ("z_proxy", "pre-C22 z-unit proxy")):
        s = null[key]
        print(f"  {lbl:<36} real ${s['real']:>10,.0f} | null mean ${s['null_mean']:>10,.0f} "
              f"sd ${s['null_std']:>9,.0f} | P(null>0) {s['null_frac_positive']:.2f} "
              f"| p = {s['p_value']:.3f}")


def main(run_wf: bool = True, save: bool = True, null_paths: int = 200) -> None:
    t0 = time.time()

    # ── 1. Load data ──────────────────────────────────────────────────────────
    spx_df, vix_wide = _load_data()

    # ── 2. Run full backtest ──────────────────────────────────────────────────
    logger.info("Running BacktestEngine (2018-01-01 → 2025-03-24) …")
    engine    = BacktestEngine()
    equity_df = engine.run(spx_df, vix_wide)
    logger.info("Backtest done: %d days | NAV $%.0f → $%.0f | trades: %d",
                len(equity_df),
                engine.initial_capital,
                float(equity_df["nav"].iloc[-1]),
                len(engine.all_trades))

    # ── 3. Metrics + crisis ───────────────────────────────────────────────────
    metrics = engine.compute_metrics()
    crisis  = engine.crisis_performance()
    _print_summary(metrics, crisis)

    # ── 4. S3 timing null test ────────────────────────────────────────────────
    null = None
    if null_paths > 0:
        logger.info("Running S3 timing null test (%d paths) …", null_paths)
        null = s3_timing_null_test(engine, n_paths=null_paths)
        _print_null(null)
        metrics["s3_null_p_value"] = null["index_leg"]["p_value"]

    # ── 5. Sub-period stability ───────────────────────────────────────────────
    wf_df: pd.DataFrame | None = None
    if run_wf:
        logger.info("Running sub-period stability windows (12 × 6 months) …")
        t_wf = time.time()
        wf_df = engine.sub_period_stability(spx_df, vix_wide)
        logger.info("Sub-periods done in %.1fs", time.time() - t_wf)
        n_neg = int((wf_df["below_zero_sharpe"]).sum())
        n_tot = len(wf_df)
        print(f"\n  Sub-period stability ({n_neg}/{n_tot} windows with Sharpe < 0; "
              "same rules re-run per window — NOT out-of-sample):")
        print(f"  {'Win':<5} {'Period':<25} {'Sharpe':>8} {'AnnRet':>8} {'Flag'}")
        print("  " + "─" * 60)
        for _, r in wf_df.iterrows():
            sh = r.get("sharpe", float("nan"))
            ar = r.get("ann_return", float("nan"))
            flag = "⚠ NEG" if r.get("below_zero_sharpe") else "OK"
            sh_s = f"{sh:.2f}" if pd.notna(sh) else "—"
            ar_s = f"{ar*100:.1f}%" if pd.notna(ar) else "—"
            period = f"{r['test_start']} → {r['test_end']}"
            print(f"  {int(r['window']):<5} {period:<25} {sh_s:>8} {ar_s:>8}  {flag}")

    if not save:
        print(f"\n  Total elapsed: {time.time() - t0:.1f}s  (--no-save: nothing written)")
        return

    # ── 6. Save results to disk ───────────────────────────────────────────────
    logger.info("Saving results to disk …")
    engine.save()
    logger.info("Saved: full_results.parquet + full_results.pkl")

    # ── 7. Generate HTML report ───────────────────────────────────────────────
    logger.info("Generating HTML report …")
    rg   = ReportGenerator(
        engine      = engine,
        equity_df   = equity_df,
        metrics     = metrics,
        crisis_perf = crisis,
        wf_df       = wf_df,
        signals_df  = engine.signals_df,
        s3_null     = null,
    )
    report_path = rg.generate(auto_open=True)
    logger.info("Report saved → %s", report_path)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"  Report: {report_path}")
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="C10: Run full backtest")
    parser.add_argument("--no-wf", dest="no_wf", action="store_true",
                        help="Skip the sub-period stability windows")
    parser.add_argument("--no-save", dest="no_save", action="store_true",
                        help="Print results only; write no parquet/pkl/report")
    parser.add_argument("--null-paths", type=int, default=200,
                        help="Paths for the S3 timing null test (0 = skip)")
    parser.add_argument("--estimate-beta", action="store_true",
                        help="Estimate β = ATM vol / VIX from every SPX chain in the DB")
    parser.add_argument("--live", action="store_true",
                        help="With --estimate-beta: also fetch a live Yahoo ^SPX chain")
    parser.add_argument("--beta-sensitivity", action="store_true",
                        help="Re-run the backtest for β ∈ {1.00, 0.90, default}")
    parser.add_argument("--pseudo-oos", action="store_true",
                        help="S1C on 2013-2017 (confounded pseudo out-of-sample)")
    parser.add_argument("--c7", action="store_true",
                        help="Re-run the C7 delta-hedge study and save its parquet/chart")
    args = parser.parse_args()

    if args.estimate_beta:
        _print_beta(estimate_beta(include_live=args.live))
    elif args.beta_sensitivity:
        spx, vix = _load_data()
        with pd.option_context("display.width", 250, "display.max_columns", 30):
            print(run_beta_sensitivity(spx, vix).round(3).to_string(index=False))
    elif args.pseudo_oos:
        print(run_pseudo_oos())
    elif args.c7:
        print(run_c7(save=not args.no_save))
    else:
        main(run_wf=not args.no_wf, save=not args.no_save, null_paths=args.null_paths)
