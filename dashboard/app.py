"""
dashboard/app.py — Joint SPX/VIX Volatility System — Live Dashboard
C-Suite grade display layer. Reads exclusively from data_store/.

Run:  python dashboard/app.py
URL:  http://localhost:5000
"""
from __future__ import annotations

import json
import pickle
import sqlite3
import sys
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Any

import numpy as np
import pandas as pd
from flask import Flask, render_template

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data_store"
sys.path.insert(0, str(ROOT))

app = Flask(
    __name__,
    template_folder=str(Path(__file__).resolve().parent / "templates"),
    static_folder=str(Path(__file__).resolve().parent / "static"),
)


# ── Utilities ──────────────────────────────────────────────────────────────────
def _db() -> sqlite3.Connection:
    return sqlite3.connect(DATA / "vol_system.db")


def _clean(obj: Any) -> Any:
    """Recursively convert numpy / pandas scalars to plain Python types."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, pd.Timestamp):
        return str(obj.date())
    return obj


def _vomma_color(z: float) -> str:
    """Map vomma z-score to a CSS color string for the heatmap."""
    if z is None or (isinstance(z, float) and np.isnan(z)):
        return "#1a1a1a"
    if abs(z) > 2.0:
        return "#cc1111"   # danger red
    if z > 0:
        t = min(z / 2.0, 1.0)
        r = int(17 + (0 - 17) * t)
        g = int(17 + (190 - 17) * t)
        b = int(17 + (90 - 17) * t)
        return f"#{r:02x}{g:02x}{b:02x}"
    else:
        t = min(abs(z) / 2.0, 1.0)
        r = int(17 + (40 - 17) * t)
        g = int(17 + (90 - 17) * t)
        b = int(17 + (200 - 17) * t)
        return f"#{r:02x}{g:02x}{b:02x}"


# ── PAGE 1 DATA: Live Market Overview ─────────────────────────────────────────
def load_page1() -> dict[str, Any]:
    d: dict[str, Any] = {}

    # ── SPX latest ──────────────────────────────────────────────────────────
    try:
        con = _db()
        spx = pd.read_sql(
            "SELECT date, close FROM spx_ohlcv ORDER BY date DESC LIMIT 2", con
        )
        d["spx"]       = round(float(spx["close"].iloc[0]), 2)
        d["spx_prev"]  = round(float(spx["close"].iloc[1]), 2)
        d["spx_date"]  = str(spx["date"].iloc[0])
        d["spx_chg"]   = round((d["spx"] - d["spx_prev"]) / d["spx_prev"] * 100, 2)
        d["spx_up"]    = d["spx"] >= d["spx_prev"]

        # VIX term structure — latest value per ticker (may differ in date)
        vts_sql = """
            SELECT t.ticker, t.tenor_days, t.close, t.date, t.is_vvix
            FROM vix_term_structure t
            INNER JOIN (
                SELECT ticker, MAX(date) as max_date
                FROM vix_term_structure
                GROUP BY ticker
            ) m ON t.ticker = m.ticker AND t.date = m.max_date
            ORDER BY t.is_vvix, t.tenor_days
        """
        vts = pd.read_sql(vts_sql, con)
        con.close()

        # Build term structure dict
        vix_curve: dict[str, float] = {}
        for _, row in vts.iterrows():
            v = float(row["close"])
            if v < 5:
                v *= 100  # fraction → points
            if row["is_vvix"]:
                d["vvix"] = round(v, 2)
                d["vvix_date"] = str(row["date"])
            else:
                vix_curve[int(row["tenor_days"])] = round(v, 2)

        # Named references
        d["vix"]   = vix_curve.get(30, vix_curve.get(9, 0))
        d["vix9d"] = vix_curve.get(9,  None)
        d["vix3m"] = vix_curve.get(93, None)
        d["vix6m"] = vix_curve.get(182, None)
        d.setdefault("vvix", 0)

        # Chart data — term structure
        ts_pts = [(t, v) for t, v in sorted(vix_curve.items())]
        d["ts_tenors"] = [f"{t}d" for t, _ in ts_pts]
        d["ts_values"] = [v for _, v in ts_pts]

        # Contango check
        if len(ts_pts) >= 2:
            d["ts_regime"] = "CONTANGO" if ts_pts[-1][1] > ts_pts[0][1] else "BACKWARDATION"
            d["ts_regime_color"] = "#00ff88" if d["ts_regime"] == "CONTANGO" else "#ff3333"
        else:
            d["ts_regime"] = "N/A"
            d["ts_regime_color"] = "#666"

    except Exception as e:
        d["db_error"] = str(e)
        for k in ["spx", "vix", "vvix", "spx_chg"]:
            d.setdefault(k, 0)
        d["ts_tenors"] = []
        d["ts_values"] = []
        d["ts_regime"] = "N/A"
        d["ts_regime_color"] = "#666"

    # ── Regime ──────────────────────────────────────────────────────────────
    try:
        rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
        latest = rl.iloc[-1]
        reg = int(latest["regime"])
        names  = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
        colors = ["#3388ff", "#00ff88", "#ff3333"]
        labels = ["R0", "R1", "R2"]
        d["regime"]       = reg
        d["regime_name"]  = names[reg]
        d["regime_color"] = colors[reg]
        d["regime_label"] = labels[reg]
        d["regime_conf"]  = round(float(latest["confidence"]) * 100, 1)
        d["regime_date"]  = str(rl.index[-1].date())
        d["prob_r0"]      = round(float(latest["prob_0"]) * 100, 1)
        d["prob_r1"]      = round(float(latest["prob_1"]) * 100, 1)
        d["prob_r2"]      = round(float(latest["prob_2"]) * 100, 1)
    except Exception as e:
        d["regime_error"] = str(e)
        d["regime"] = -1
        d["regime_name"] = "UNKNOWN"
        d["regime_color"] = "#666"
        d["regime_label"] = "—"
        d["regime_conf"]  = 0
        d["prob_r0"] = d["prob_r1"] = d["prob_r2"] = 33.3

    # ── PDV Forecast vs ATM ──────────────────────────────────────────────────
    try:
        with open(DATA / "pdv_model.pkl", "rb") as f:
            pdv = pickle.load(f)
        feat    = pdv._features.iloc[-1]
        coefs   = pdv.linear_.coef_
        intcpt  = float(pdv.linear_.intercept_)
        s1  = float(feat["sigma1"])
        s2  = float(feat["sigma2"])
        lev = float(feat["lev"])
        pdv_forecast = float(coefs[0] * s1 + coefs[1] * s2 + coefs[2] * lev + intcpt) * 100
        d["pdv_forecast"] = round(pdv_forecast, 2)
        d["pdv_spread"]   = round((d.get("vix", 0) or 0) - pdv_forecast, 2)
        d["pdv_date"]     = str(pdv._features.index[-1].date())
        d["pdv_sigma1"]   = round(s1 * 100, 2)
        d["pdv_sigma2"]   = round(s2 * 100, 2)
        d["pdv_spread_positive"] = d["pdv_spread"] > 0
    except Exception as e:
        d["pdv_error"] = str(e)
        d["pdv_forecast"] = 0
        d["pdv_spread"]   = 0
        d["pdv_date"]     = "N/A"
        d["pdv_sigma1"]   = 0
        d["pdv_sigma2"]   = 0
        d["pdv_spread_positive"] = False

    return _clean(d)


# ── PAGE 2 DATA: Calibration Results ─────────────────────────────────────────
def load_page2() -> dict[str, Any]:
    d: dict[str, Any] = {}

    try:
        with open(DATA / "calibrations" / "joint_cal_2026-03-24.pkl", "rb") as f:
            cal = pickle.load(f)

        p = cal["params"]
        d["kappa"]     = round(p["kappa"], 4)
        d["theta"]     = round(p["theta"], 5)
        d["theta_vol"] = round(np.sqrt(p["theta"]) * 100, 2)
        d["sigma"]     = round(p["sigma"], 4)
        d["rho"]       = round(p["rho"], 4)
        d["v0"]        = round(p["v0"], 5)
        d["v0_vol"]    = round(np.sqrt(p["v0"]) * 100, 2)
        d["S"]         = round(float(cal.get("S", 0)), 1)
        d["r_pct"]     = round(float(cal.get("r", 0)) * 100, 2)
        d["q_pct"]     = round(float(cal.get("q", 0)) * 100, 2)
        d["as_of"]     = str(cal.get("as_of_date", "2026-03-24"))
        d["fit_time"]  = round(float(cal.get("fit_time", 0)), 1)
        d["n_evals"]   = int(cal.get("n_evals", 0))

        losses = cal.get("losses", {})
        d["spx_rmse"]     = round(float(losses.get("spx_iv_rmse", 0)), 3)
        d["vix_fut_rmse"] = round(float(losses.get("vix_futures_rmse", 0)), 3)
        d["vix_opt_rmse"] = round(float(losses.get("vix_options_rmse", 0)), 2)
        d["total_loss"]   = round(float(losses.get("total_loss", 0)), 6)

        # Feller condition: 2κθ > σ²
        feller_lhs = 2 * p["kappa"] * p["theta"]
        feller_rhs = p["sigma"] ** 2
        d["feller_lhs"]    = round(feller_lhs, 5)
        d["feller_rhs"]    = round(feller_rhs, 5)
        d["feller_margin"] = round(feller_lhs - feller_rhs, 6)
        d["feller_pass"]   = bool(feller_lhs > feller_rhs)

        # Rho boundary flag
        d["rho_at_boundary"] = abs(p["rho"]) >= 0.99

    except Exception as e:
        d["cal_error"] = str(e)
        for k in ["kappa", "theta", "sigma", "rho", "v0", "spx_rmse",
                  "vix_fut_rmse", "vix_opt_rmse", "total_loss",
                  "theta_vol", "v0_vol", "S", "r_pct", "q_pct",
                  "fit_time", "n_evals", "feller_lhs", "feller_rhs", "feller_margin"]:
            d.setdefault(k, 0)
        d.setdefault("feller_pass", False)
        d.setdefault("rho_at_boundary", False)
        d.setdefault("as_of", "N/A")
        d.setdefault("mat_smiles", [])

    # SPX smile from greeks surface (91-day, closest to standard 3M)
    try:
        gdf = pd.read_parquet(DATA / "greeks" / "greeks_surface.parquet")
        smile = gdf[gdf["T_days"] == 91].sort_values("log_moneyness")
        if len(smile) == 0:
            smile = gdf[gdf["T_days"] == gdf["T_days"].unique()[3]].sort_values("log_moneyness")
        d["smile_lm"] = smile["log_moneyness"].round(4).tolist()
        d["smile_iv"] = (smile["iv"] * 100).round(3).tolist()
        d["smile_mat"] = f"{int(smile['T_days'].iloc[0])}d (≈3M)"
    except Exception as e:
        d["smile_error"] = str(e)
        d["smile_lm"] = []
        d["smile_iv"] = []
        d["smile_mat"] = ""

    # Multi-maturity smile overlay
    try:
        gdf = pd.read_parquet(DATA / "greeks" / "greeks_surface.parquet")
        mat_smiles: list[dict] = []
        colors_map = {14: "#ff3333", 30: "#ff8844", 60: "#ffcc00",
                      91: "#00ff88", 180: "#44aaff", 365: "#aa44ff"}
        for T in sorted(gdf["T_days"].unique()):
            sub = gdf[gdf["T_days"] == T].sort_values("log_moneyness")
            mat_smiles.append({
                "T": int(T),
                "lm": sub["log_moneyness"].round(4).tolist(),
                "iv": (sub["iv"] * 100).round(3).tolist(),
                "color": colors_map.get(int(T), "#888"),
            })
        d["mat_smiles"] = mat_smiles
    except Exception as e:
        d["mat_smiles"] = []

    return _clean(d)


# ── PAGE 3 DATA: Greeks Risk Monitor ─────────────────────────────────────────
def load_page3() -> dict[str, Any]:
    d: dict[str, Any] = {}

    try:
        gdf = pd.read_parquet(DATA / "greeks" / "greeks_surface.parquet")
        maturities = sorted(gdf["T_days"].unique().tolist())

        # Bin log_moneyness into 0.05-wide buckets for the heatmap grid
        gdf = gdf.copy()
        gdf["lm_bin"] = (np.floor(gdf["log_moneyness"] / 0.05) * 0.05).round(2)
        lm_bins = sorted(gdf["lm_bin"].unique().tolist())

        # Build heatmap: rows = maturity, cols = lm_bin
        heatmap_rows: list[dict] = []
        unstable_nodes: list[dict] = []

        for T in maturities:
            sub = gdf[gdf["T_days"] == T]
            cells: list[dict | None] = []
            for lm in lm_bins:
                cell_df = sub[sub["lm_bin"] == lm]
                if len(cell_df) == 0:
                    cells.append(None)
                else:
                    r = cell_df.iloc[0]
                    z    = float(r["vomma_zscore"])
                    v    = float(r["vomma"])
                    iv   = float(r["iv"]) * 100
                    K    = float(r["K"])
                    unstable = bool(r["is_unstable"])
                    cell = {
                        "vomma":    round(v, 1),
                        "z":        round(z, 2),
                        "iv":       round(iv, 1),
                        "K":        round(K, 0),
                        "unstable": unstable,
                        "bg":       _vomma_color(z),
                        "fg":       "#fff" if unstable or abs(z) > 1 else "#aaa",
                    }
                    if unstable:
                        unstable_nodes.append({
                            "T_days": int(T),
                            "lm":     round(lm, 2),
                            "K":      round(K, 0),
                            "vomma":  round(v, 1),
                            "z":      round(z, 2),
                            "iv":     round(iv, 1),
                        })
                    cells.append(cell)
            heatmap_rows.append({
                "T":     int(T),
                "label": f"{int(T)}d",
                "cells": cells,
            })

        d["heatmap_rows"] = heatmap_rows
        d["heatmap_cols"] = [f"{b:+.2f}" for b in lm_bins]
        d["unstable_nodes"] = sorted(unstable_nodes, key=lambda x: x["z"], reverse=True)
        d["n_unstable"]     = len(unstable_nodes)

        # Vomma surface statistics
        d["vomma_max"]  = round(float(gdf["vomma"].max()), 1)
        d["vomma_min"]  = round(float(gdf["vomma"].min()), 1)
        d["vomma_mean"] = round(float(gdf["vomma"].mean()), 1)
        d["n_cells"]    = len(gdf)

        # QV convexity per maturity
        qv = gdf.groupby("T_days")["qv_convexity"].agg(["mean", "max"]).reset_index()
        d["qv_labels"]    = [f"{int(t)}d" for t in qv["T_days"]]
        d["qv_mean"]      = (qv["mean"] * 1e4).round(4).tolist()  # ×1e4 for display
        d["qv_max"]       = (qv["max"]  * 1e4).round(4).tolist()

        # Vanna per maturity
        vanna = gdf.groupby("T_days")["vanna"].mean().reset_index()
        d["vanna_labels"] = [f"{int(t)}d" for t in vanna["T_days"]]
        d["vanna_values"] = vanna["vanna"].round(4).tolist()

        # Vega per maturity
        vega = gdf.groupby("T_days")["vega"].mean().reset_index()
        d["vega_labels"]  = [f"{int(t)}d" for t in vega["T_days"]]
        d["vega_values"]  = vega["vega"].round(2).tolist()

        # Vomma z-score scatter (for chart)
        d["scatter_lm"]    = gdf["log_moneyness"].round(4).tolist()
        d["scatter_z"]     = gdf["vomma_zscore"].round(3).tolist()
        d["scatter_T"]     = gdf["T_days"].astype(int).tolist()
        d["scatter_iv"]    = (gdf["iv"] * 100).round(2).tolist()

    except Exception as e:
        d["greeks_error"] = str(e)
        d["heatmap_rows"] = []
        d["heatmap_cols"] = []
        d["unstable_nodes"] = []
        d["n_unstable"] = 0
        d["n_cells"] = 0
        d["vomma_max"] = 0
        d["vomma_min"] = 0
        d["vomma_mean"] = 0
        d["qv_labels"] = []
        d["qv_mean"] = []
        d["qv_max"] = []
        d["vanna_labels"] = []
        d["vanna_values"] = []
        d["vega_labels"] = []
        d["vega_values"] = []
        d["scatter_lm"] = []
        d["scatter_z"] = []
        d["scatter_T"] = []
        d["scatter_iv"] = []

    return _clean(d)


# ── PAGE 4 DATA: Backtest Results ─────────────────────────────────────────────
def load_page4() -> dict[str, Any]:
    d: dict[str, Any] = {}

    # ── Equity curve ────────────────────────────────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)

        # Downsample to ≤600 points
        step = max(1, len(eq) // 600)
        eq_ds = eq.iloc[::step]

        # Normalise each NAV series to % gain from 100
        def _norm(col: str) -> list[float]:
            s = eq_ds[col]
            base = float(s.iloc[0])
            return ((s / base * 100) - 100).round(2).tolist()

        d["eq_dates"]    = [dt.strftime("%Y-%m-%d") for dt in eq_ds.index]
        d["eq_nav"]      = _norm("nav")
        d["eq_nav_s1"]   = _norm("nav_s1")
        d["eq_nav_s2"]   = _norm("nav_s2")
        d["eq_nav_s3"]   = _norm("nav_s3")
        d["eq_nav_comb"] = _norm("nav_combined")

        # Final NAV for display
        d["final_nav"]   = round(float(eq["nav"].iloc[-1]), 0)
        d["start_nav"]   = round(float(eq["nav"].iloc[0]), 0)
        d["total_pnl"]   = round(d["final_nav"] - d["start_nav"], 0)

    except Exception as e:
        d["eq_error"] = str(e)
        d["eq_dates"] = []
        d["eq_nav"]   = []
        d["final_nav"] = 0

    # ── Metrics from pickle ──────────────────────────────────────────────────
    try:
        with open(DATA / "backtest" / "full_results.pkl", "rb") as f:
            bt = pickle.load(f)
        m = bt["metrics"]

        def _pct(x): return f"{x*100:.2f}%" if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"
        def _f2(x):  return f"{x:.3f}"       if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"
        def _dol(x): return f"${x:,.0f}"     if x is not None and not (isinstance(x, float) and np.isnan(x)) else "—"

        d["metrics_rows"] = [
            {"label": "Cumulative Return",  "value": _pct(m.get("cumulative_return")),
             "positive": (m.get("cumulative_return") or 0) > 0},
            {"label": "Ann. Return",        "value": _pct(m.get("ann_return")),
             "positive": (m.get("ann_return") or 0) > 0},
            {"label": "Sharpe (rf=5%)",     "value": _f2(m.get("sharpe")),
             "positive": (m.get("sharpe") or 0) > 0},
            {"label": "Sortino",            "value": _f2(m.get("sortino")),
             "positive": (m.get("sortino") or 0) > 0},
            {"label": "Max Drawdown",       "value": _pct(m.get("max_drawdown")),
             "positive": False},
            {"label": "DD Duration",        "value": f"{int(m.get('dd_duration_days', 0))} days",
             "positive": False},
            {"label": "Calmar",             "value": _f2(m.get("calmar")),
             "positive": (m.get("calmar") or 0) > 0},
            {"label": "Win Rate",           "value": _pct(m.get("win_rate")),
             "positive": (m.get("win_rate") or 0) > 0.5},
            {"label": "Avg P&L / Trade",    "value": _dol(m.get("avg_pnl_per_trade")),
             "positive": (m.get("avg_pnl_per_trade") or 0) > 0},
            {"label": "Best Day",           "value": _pct(m.get("best_day")),  "positive": True},
            {"label": "Worst Day",          "value": _pct(m.get("worst_day")), "positive": False},
            {"label": "N Trades",           "value": str(int(m.get("n_trades", 0))), "positive": True},
            {"label": "N Days",             "value": str(int(m.get("n_days",   0))), "positive": True},
        ]

        # Signal P&L bar chart
        signal_pnl = [
            round(float(m.get("s1_total_pnl",        0)) / 1000, 1),
            round(float(m.get("s2_total_pnl",        0)) / 1000, 1),
            round(float(m.get("s3_total_pnl",        0)) / 1000, 1),
            round(float(m.get("combined_total_pnl",  0)) / 1000, 1),
        ]
        d["signal_labels"] = ["S1  IVR / PDV", "S2  VIX TS", "S3  Dispersion", "Combined"]
        d["signal_pnl"]    = signal_pnl
        d["signal_colors"] = ["#00c870" if v > 0 else "#ff3333" for v in signal_pnl]

        # Per-signal metrics table
        d["signal_metrics"] = []
        for sig, lbl in [("s1","S1 IVR"),("s2","S2 VIX TS"),("s3","S3 Disp"),("combined","Combined")]:
            pnl = float(m.get(f"{sig}_total_pnl", 0))
            d["signal_metrics"].append({
                "label":    lbl,
                "sharpe":   _f2(m.get(f"{sig}_sharpe")),
                "ann_ret":  _pct(m.get(f"{sig}_ann_return")),
                "win_rate": _pct(m.get(f"{sig}_win_rate")),
                "n_trades": str(int(m.get(f"{sig}_n_trades", 0))),
                "total_pnl": _dol(pnl),
                "green": pnl > 0,
            })

    except Exception as e:
        d["pkl_error"] = str(e)
        d["metrics_rows"] = []
        d["signal_metrics"] = []
        d["signal_pnl"] = []
        d["signal_labels"] = []
        d["signal_colors"] = []

    # ── Annual returns ───────────────────────────────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        ann_years: list[int] = []
        ann_rets:  list[float] = []
        ann_colors: list[str] = []
        ann_labels: list[str] = []
        for yr, g in eq.groupby(eq.index.year):
            ret = round((float(g["nav"].iloc[-1]) - float(g["nav"].iloc[0]))
                        / float(g["nav"].iloc[0]) * 100, 2)
            ann_years.append(int(yr))
            ann_rets.append(ret)
            ann_colors.append("#00c870" if ret > 0 else "#ff3333")
            ann_labels.append(f"{ret:+.2f}%")
        d["ann_years"]  = ann_years
        d["ann_rets"]   = ann_rets
        d["ann_colors"] = ann_colors
        d["ann_labels"] = ann_labels
    except Exception as e:
        d["ann_error"] = str(e)
        d["ann_years"] = []
        d["ann_rets"]  = []

    # ── Walk-forward (computed from equity curve) ────────────────────────────
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        wf_periods: list[str]  = []
        wf_sharpes: list[float] = []
        wf_colors:  list[str]  = []
        for year in range(2019, 2026):
            for half, (s, e) in enumerate(
                [(f"{year}-01-01", f"{year}-06-30"),
                 (f"{year}-07-01", f"{year}-12-31")], 1
            ):
                sub = eq.loc[s:e, "nav"]
                if len(sub) < 5:
                    continue
                rets = sub.pct_change().dropna()
                if rets.abs().max() < 1e-10 or rets.std() < 1e-10:
                    sh = 0.0
                else:
                    ex = rets - 0.05 / 252
                    sh = float(ex.mean() / ex.std() * np.sqrt(252))
                sh = round(max(min(sh, 4.0), -4.0), 2)
                wf_periods.append(f"{year}-H{half}")
                wf_sharpes.append(sh)
                wf_colors.append("#00c870" if sh >= 0 else "#ff3333")
        d["wf_periods"]   = wf_periods
        d["wf_sharpes"]   = wf_sharpes
        d["wf_colors"]    = wf_colors
        d["wf_neg"]       = sum(1 for s in wf_sharpes if s < 0)
        d["wf_total"]     = len(wf_sharpes)
        d["wf_pos"]       = d["wf_total"] - d["wf_neg"]
    except Exception as e:
        d["wf_error"] = str(e)
        d["wf_periods"] = []
        d["wf_sharpes"] = []
        d["wf_colors"]  = []
        d["wf_neg"] = d["wf_total"] = d["wf_pos"] = 0

    return _clean(d)


# ── Routes ─────────────────────────────────────────────────────────────────────
@app.route("/")
def page_market():
    return render_template("index.html", d=load_page1())


@app.route("/calibration")
def page_calibration():
    return render_template("calibration.html", d=load_page2())


@app.route("/greeks")
def page_greeks():
    return render_template("greeks.html", d=load_page3())


@app.route("/backtest")
def page_backtest():
    return render_template("backtest.html", d=load_page4())


# ── Entry point ────────────────────────────────────────────────────────────────
def _find_free_port(preferred: int = 5000) -> int:
    import socket
    for port in [preferred, 5001, 5050, 8080, 8888]:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex(("localhost", port)) != 0:
                return port
    return preferred


if __name__ == "__main__":
    # Railway / Render / Fly.io inject PORT — fall back to local auto-detect
    import os as _os
    port = int(_os.environ.get("PORT", _find_free_port(5000)))
    local = not _os.environ.get("PORT")           # True when running locally
    url   = f"http://localhost:{port}"

    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │   Joint SPX/VIX Volatility System — Dashboard    │")
    print(f"  │   {url:<44}  │")
    print("  └──────────────────────────────────────────────────┘\n")

    if local:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
