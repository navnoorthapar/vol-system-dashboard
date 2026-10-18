"""
c22_bridge.py — C22 correction waterfall from the published C21 headline.

Re-runs BacktestEngine applying the C22 corrections ONE AT A TIME, in the
fixed order below, starting from the engine exactly as published in C21
(NAV $738,794 / P&L −$261,206 / Sharpe −1.80 / MaxDD −31.5%), and prints
how each correction moves the headline and the sleeves.

Order (cumulative; each row includes every row above it):
   0  published      C21 engine: β=1 (VIX as ATM vol), fixed VVIX>100 gate,
                     1-day PDV spread, legacy interpolation and accounting
   1  interp         VIX tenors from config (^VIX6M 184d) on a 365-day year
   2  sharpe         Sharpe/Sortino on excess P&L with rf = 0 (metrics only)
   3  rates          missing ^IRX dates use the previous print, not 4.5%
   4  s3_exit        S3 proxy holds its entry size through the exit day
   5  hedge_costs    opening hedge + closing unwind; no phantom exit rebalance
   6  financing      carry of premium and hedge (r on cash, q on the stock leg)
   7  expiry         straddles settle at expiry and roll if the signal is on
   8  open_trade     the trade still open on the last date is logged
   9  sleeve_nav     size and margin-check off each sleeve's running NAV
  10  corr_netting   correlation netting that can actually reduce size
  11  beta           ATM vol = ATM_TO_VIX_RATIO × VIX term structure
  12  s3_index_leg   S3 priced as the short-SPX-straddle index leg
  13  r2_gate        R2 gate = regime_classifier's default gate      [signals chain]
  14  pdv_h21        PDV spread on a 21-day forward-RMS target       [PDV chain]
  --  s2_stop        S2 stop-loss direction (reference sleeve only)  [signals chain]

Rows 13-14 depend on other C22 chains, which are merged; both rows are
always applied, and a behavioural probe of each upstream change guards it
(the bridge raises if the probe fails, i.e. if the upstream fix regressed):
  • r2_gate : SignalEngine.generate no longer overwrites
              precomputed regimes with its own VVIX>100 rule (otherwise an
              adaptive label set would be unioned with the fixed gate). The
              row then uses build_regime_labels' DEFAULT gate — whatever single
              gate regime_classifier defines.
  • pdv_h21 : models.pdv.build_xy(horizon=h) targets returns over
              [t+1, t+h] (C22 PDV fix; before it, h>1 leaked). The row
              switches the spread from the yearly 1-day |r| PDVLinear to
              walk_forward_pdv_vol (21-session RMS vol, daily purged refit).
  • s2_stop : lives inside generate_signal2 and cannot be toggled from here.
              The probe reports whether stops fire on adverse moves; S2 is a
              research reference and never enters the portfolio, so the
              portfolio rows are unaffected either way.

Row 0 reproduces the published numbers exactly while the signal and PDV
code is the C21 version (checked: NAV $738,794.35, Sharpe −1.799). Once
upstream changes that cannot be switched off from here are merged (S2 stop
direction, z-score warm-up from history, EWM start-up, …) row 0 runs the
legacy ENGINE on the new signals; its ΔP&L against the published P&L is
then the size of those changes, and the reconciliation says so.

The canonical run (run_backtest) uses BacktestEngine's defaults
(PDV_SPREAD_HORIZON = 21, build_regime_labels' default gate) with the
PDV-21 series from run_backtest.load_pdv_vol(); the bridge passes the same
series, so its last row is the canonical headline.

Usage (repository root):
  python -m joint_vol_calibration.backtest.c22_bridge
  python -m joint_vol_calibration.backtest.c22_bridge --json out.json
  python -m joint_vol_calibration.backtest.c22_bridge --csv data_store/backtest/c22_bridge.csv
"""

from __future__ import annotations

import argparse
import contextlib
import json
import logging
import time
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.backtest import delta_hedger as dh

logger = logging.getLogger("c22_bridge")

# The C21 numbers as published (README / dashboard / substack).
PUBLISHED = {
    "nav": 738_794.0, "pnl": -261_206.0, "sharpe": -1.80, "max_dd": -0.315,
    "s1c_pnl": -599_995.0, "s3_pnl": 23_212.0, "s4_pnl": -206_437.0,
    "s1_pnl": 693_938.0, "s2_pnl": -76_064.0,
}
LEGACY_VVIX_GATE = 100.0
SLEEVES = ("s1c", "s3", "s4", "s1", "s2", "combined")


# ── Legacy interpolation (row 0) ──────────────────────────────────────────────

@contextlib.contextmanager
def legacy_vix_interpolation():
    """C21 interpolation: ^VIX6M at 182 days and T converted on 365.25 days."""
    old_days, old_tenors = dh.DAYS_PER_YEAR, dict(dh.VIX_TENORS)
    dh.DAYS_PER_YEAR = 365.25
    dh.VIX_TENORS["^VIX6M"] = 182.0
    try:
        yield
    finally:
        dh.DAYS_PER_YEAR = old_days
        dh.VIX_TENORS.clear()
        dh.VIX_TENORS.update(old_tenors)


# ── Upstream probes ───────────────────────────────────────────────────────────

def _synthetic_market(n: int = 320, seed: int = 11, vvix: float = 150.0):
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range("2019-01-02", periods=n)
    close = 3000.0 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    spx = pd.DataFrame({"date": dates, "close": close,
                        "log_return": np.r_[np.nan, np.diff(np.log(close))]})
    vix = pd.DataFrame({"date": dates, "^VIX": 18 + rng.normal(0, 1, n),
                        "^VIX3M": 20.0, "^VIX6M": 21.0, "^VIX9D": 17.0, "^VVIX": vvix})
    return spx, vix


def probe_r2_override_removed() -> bool:
    """True when SignalEngine.generate keeps precomputed regimes as given."""
    from joint_vol_calibration.signals.signal_engine import SignalEngine
    spx, vix = _synthetic_market()
    regimes = pd.Series(1, index=pd.to_datetime(spx["date"]))
    try:
        out = SignalEngine().generate(spx, vix, start_date="2019-03-01", end_date="2020-12-31",
                                      precomputed_regimes=regimes)
    except Exception as exc:                       # pragma: no cover - defensive
        logger.warning("R2 probe failed: %s", exc)
        return False
    return bool((out["regime"] != 2).all())


def default_gate_description() -> str:
    """Human-readable default gate of build_regime_labels."""
    import inspect
    from joint_vol_calibration.signals.regime_classifier import build_regime_labels
    p = inspect.signature(build_regime_labels).parameters.get("vvix_threshold")
    if p is None:
        return "default"
    d = p.default
    return "adaptive (rolling 80th pct)" if d is None else f"fixed VVIX>{d:g}"


def probe_pdv_forward_window() -> bool:
    """True when build_xy(horizon=h) averages r² over [t+1, t+h] (no leak)."""
    from joint_vol_calibration.models.pdv import build_xy, extract_pdv_features
    n, k, h = 120, 80, 5
    idx = pd.bdate_range("2015-01-02", periods=n)
    r = pd.Series(0.001, index=idx)
    r.iloc[k] = 0.10
    try:
        _, y = build_xy(r, extract_pdv_features(r), horizon=h)
    except Exception as exc:                       # pragma: no cover - defensive
        logger.warning("PDV probe failed: %s", exc)
        return False
    hit = y[y > 0.2].index
    want = idx[k - h:k]
    return len(hit) > 0 and set(hit) <= set(want) and idx[k - h] in set(hit)


def probe_s2_stop_direction() -> str:
    """'fixed' if a long-vol S2 trade is stopped on an ADVERSE move only.

    Features are coherent with VIX3M pinned at 20: slope = (20 − VIX)/100,
    so the probe works whether the stop is expressed on the slope (pre-C22)
    or on the VIX level (C22 signals chain).
    """
    from joint_vol_calibration.signals.signal_engine import generate_signal2
    n, e = 300, 280
    idx = pd.bdate_range("2019-01-02", periods=n)

    def _stopped(vix_after_entry) -> bool:
        vix = np.full(n, 18.0)                        # flat contango: no trades before e
        vix[e] = 21.0                                 # backwardation → long front vol
        vix[e + 1:e + 1 + len(vix_after_entry)] = vix_after_entry
        feats = pd.DataFrame({"ts_slope": (20.0 - vix) / 100.0, "vix": vix / 100.0}, index=idx)
        out = generate_signal2(feats, pd.Series(1, index=idx), max_hold=50)
        pos = out["s2_position"].values
        return bool(pos[e] == 1 and (pos[e + 1:e + 1 + len(vix_after_entry)] == 0).any())

    # A long straddle gains when VIX rises and loses when it falls.
    try:
        favourable = _stopped([25.0, 26.0, 27.0])
        adverse = _stopped([17.0, 16.8, 16.5])
    except Exception as exc:                       # pragma: no cover - defensive
        logger.warning("S2 probe failed: %s", exc)
        return "indeterminate"
    if favourable and not adverse:
        return "inverted (stops winning trades)"
    if adverse and not favourable:
        return "fixed (stops losing trades)"
    return "indeterminate"


# ── Runner ────────────────────────────────────────────────────────────────────

def _run(spx, vix, *, beta: float, corr: bte.C22Corrections, gate, pdv_h: int,
         legacy_interp: bool, pdv_vol: Optional[pd.Series] = None) -> Dict:
    ctx = legacy_vix_interpolation() if legacy_interp else contextlib.nullcontext()
    with ctx:
        eng = bte.BacktestEngine(atm_to_vix_ratio=beta, corrections=corr,
                                 vvix_threshold=gate, pdv_horizon=pdv_h,
                                 pdv_vol_series=pdv_vol)
        eq = eng.run(spx, vix)
        m = eng.compute_metrics()
    out = {"nav": float(eq["nav"].iloc[-1]), "pnl": float(eq["nav"].iloc[-1] - eng.initial_capital),
           "sharpe": float(m["sharpe"]), "max_dd": float(m["max_drawdown"])}
    for s in SLEEVES:
        out[f"{s}_pnl"] = float(m.get(f"{s}_total_pnl", 0.0))
        out[f"{s}_sharpe"] = float(m.get(f"{s}_sharpe", np.nan))
        out[f"{s}_win"] = m.get(f"{s}_win_rate")
        out[f"{s}_trades"] = m.get(f"{s}_n_trades")
    return out


def build_steps() -> List[Dict]:
    """The ordered correction list with the settings each row adds."""
    beta = bte.ATM_TO_VIX_RATIO
    steps: List[Dict] = [
        {"key": "published", "label": "C21 engine as published", "set": {}},
        {"key": "interp", "label": "VIX tenors 9/30/93/184d, 365-day year", "set": {"legacy_interp": False}},
        {"key": "sharpe", "label": "Sharpe on excess P&L, rf=0", "corr": {"sharpe_rf_zero": True}},
        {"key": "rates", "label": "^IRX gaps: previous print", "corr": {"rate_ffill": True}},
        {"key": "s3_exit", "label": "S3 proxy exit-day P&L + cost", "corr": {"s3_exit_day": True}},
        {"key": "hedge_costs", "label": "hedge open/unwind costs", "corr": {"hedge_costs": True}},
        {"key": "financing", "label": "premium + hedge financing", "corr": {"financing": True}},
        {"key": "expiry", "label": "expiry settle / roll", "corr": {"expiry_roll": True}},
        {"key": "open_trade", "label": "open trade in trade log", "corr": {"open_trade_log": True}},
        {"key": "sleeve_nav", "label": "size off sleeve NAV", "corr": {"sleeve_nav_sizing": True}},
        {"key": "corr_netting", "label": "real correlation netting", "corr": {"corr_netting": True}},
        {"key": "beta", "label": f"ATM vol = {beta:.2f} × VIX", "set": {"beta": beta}},
        {"key": "s3_index_leg", "label": "S3 = short SPX straddle (index leg)", "corr": {"s3_index_leg": True}},
    ]
    # Both upstream chains are merged: the rows are always applied. The probes
    # stay as guards — if either fails, an upstream fix has regressed and the
    # waterfall would silently mislabel a no-op as a correction.
    if not probe_r2_override_removed():
        raise RuntimeError("r2_gate row: SignalEngine overrides precomputed regimes "
                           "again (signals-chain fix regressed)")
    if not probe_pdv_forward_window():
        raise RuntimeError("pdv_h21 row: build_xy(horizon>1) no longer targets "
                           "[t+1, t+h] (PDV-chain fix regressed)")
    gate_desc = default_gate_description()
    steps.append({
        "key": "r2_gate", "upstream": "signals",
        "label": f"R2 gate = upstream default ({gate_desc})",
        "set": {"gate": bte._DEFAULT_GATE},
    })
    steps.append({
        "key": "pdv_h21", "upstream": "pdv",
        "label": "PDV spread: 21-day RMS-vol forecast (daily purged refit)",
        "set": {"pdv_h": bte.PDV_SPREAD_HORIZON},
    })
    return steps


def run_bridge(spx: Optional[pd.DataFrame] = None, vix: Optional[pd.DataFrame] = None,
               pdv_vol: Optional[pd.Series] = None) -> pd.DataFrame:
    """Run every step cumulatively; one row per step.

    ``pdv_vol`` is the walk-forward PDV-21 σ̂ used once the pdv_h21 row
    switches the spread to the 21-day target (rows with pdv_h = 1 ignore
    it). Left None with the DB data, it is run_backtest.load_pdv_vol() —
    the same series the canonical run uses, so the last row reproduces it.
    """
    if spx is None or vix is None:
        from joint_vol_calibration.backtest.run_backtest import _load_data, load_pdv_vol
        spx, vix = _load_data()
        if pdv_vol is None:
            pdv_vol = load_pdv_vol()
    state = {"beta": 1.0, "gate": LEGACY_VVIX_GATE, "pdv_h": 1, "legacy_interp": True}
    corr = bte.C22Corrections.legacy()
    rows = []
    for step in build_steps():
        state.update(step.get("set", {}))
        if step.get("corr"):
            corr = corr.with_(**step["corr"])
        t0 = time.time()
        res = _run(spx, vix, beta=state["beta"], corr=corr, gate=state["gate"],
                   pdv_h=state["pdv_h"], legacy_interp=state["legacy_interp"],
                   pdv_vol=pdv_vol)
        logger.info("step %-13s done in %.1fs", step["key"], time.time() - t0)
        rows.append({"step": step["key"], "label": step["label"],
                     "note": step.get("note", ""), **res})
    df = pd.DataFrame(rows)
    df["d_pnl"] = df["pnl"].diff().fillna(df["pnl"] - PUBLISHED["pnl"])
    return df


def _fmt_table(df: pd.DataFrame) -> str:
    lines = []
    hdr = (f"{'#':>2}  {'step':<13} {'NAV':>11} {'P&L':>11} {'ΔP&L':>10} {'Sharpe':>7} {'MaxDD':>7}"
           f" {'S1C':>10} {'S3':>10} {'S4':>10} | {'S1 ref':>10} {'S2 ref':>10}")
    lines.append(hdr)
    lines.append("─" * len(hdr))
    for i, r in df.iterrows():
        lines.append(
            f"{i:>2}  {r['step']:<13} {r['nav']:>11,.0f} {r['pnl']:>11,.0f} {r['d_pnl']:>10,.0f}"
            f" {r['sharpe']:>7.3f} {r['max_dd']:>7.1%} {r['s1c_pnl']:>10,.0f} {r['s3_pnl']:>10,.0f}"
            f" {r['s4_pnl']:>10,.0f} | {r['s1_pnl']:>10,.0f} {r['s2_pnl']:>10,.0f}"
        )
    return "\n".join(lines)


def _reconcile(df: pd.DataFrame) -> List[str]:
    """Compare row 0 with the published C21 numbers."""
    r0 = df.iloc[0]
    out = []
    for k, v in PUBLISHED.items():
        tol = 0.006 if k == "sharpe" else (0.0006 if k == "max_dd" else 1.0)
        ok = abs(r0[k] - v) <= tol
        out.append(f"  {k:<8} published {v:>12,.3f}  recomputed {r0[k]:>12,.3f}  {'OK' if ok else 'DIFFERS'}")
    return out


def main(argv: Optional[List[str]] = None) -> pd.DataFrame:
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(name)s %(message)s")
    logger.setLevel(logging.INFO)
    ap = argparse.ArgumentParser(description="C22 correction waterfall")
    ap.add_argument("--json", help="write the rows as JSON to this path")
    ap.add_argument("--csv", help="write the rows as CSV to this path "
                                  "(canonical: data_store/backtest/c22_bridge.csv)")
    args = ap.parse_args(argv)

    df = run_bridge()
    print("\nC22 correction waterfall (cumulative; P&L in $ on $1M; sleeves at full size)\n")
    print(_fmt_table(df))
    print("\nStep notes:")
    for i, r in df.iterrows():
        print(f"  {i:>2} {r['step']:<13} {r['label']}" + (f"  — {r['note']}" if r["note"] else ""))
    print(f"\n  S2 stop-loss (signals chain, reference only): {probe_s2_stop_direction()}")
    print("\nRow 0 vs published C21 headline:")
    rec = _reconcile(df)
    print("\n".join(rec))
    if any(line.endswith("DIFFERS") for line in rec):
        print("  Row 0 runs the legacy ENGINE on the signal/PDV code installed now. A gap\n"
              "  here is the effect of upstream changes the bridge cannot switch off (e.g.\n"
              "  S2 stop direction, z-score warm-up, EWM start-up); row 0's ΔP&L is that gap.")
    last = df.iloc[-1]
    print("\nFinal per-signal stats:")
    for s in SLEEVES:
        w = last[f"{s}_win"]
        print(f"  {s:<9} P&L {last[f'{s}_pnl']:>11,.0f}  Sharpe {last[f'{s}_sharpe']:>7.3f}"
              f"  win {'—' if w is None or pd.isna(w) else f'{w:.1%}'}  trades {last[f'{s}_trades']}")
    if args.csv:
        df.to_csv(args.csv, index_label="row", float_format="%.6g")
    if args.json:
        with open(args.json, "w") as fh:
            json.dump(json.loads(df.to_json(orient="records")), fh, indent=1)
    return df


if __name__ == "__main__":
    main()
