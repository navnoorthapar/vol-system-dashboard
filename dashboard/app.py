"""
dashboard/app.py — Joint SPX/VIX Volatility System — Live Dashboard
C-Suite grade display layer. Reads exclusively from data_store/.

Run:  python dashboard/app.py
URL:  http://localhost:5000
"""
from __future__ import annotations

import json
import logging
import pickle
import sqlite3
import sys
import threading
import webbrowser
from pathlib import Path
from threading import Timer
from typing import Any

_logger = logging.getLogger("vol_dashboard")

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

    # ── Regime — live inference from current market data ───────────────────
    try:
        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as f:
            clf = pickle.load(f)

        # SPX history for realized-vol features
        con2 = _db()
        spx_hist = pd.read_sql(
            "SELECT date, close FROM spx_ohlcv ORDER BY date DESC LIMIT 30", con2
        )
        con2.close()
        spx_hist = spx_hist.sort_values("date").reset_index(drop=True)
        spx_rets = spx_hist["close"].pct_change().dropna()

        rv_20d    = float(spx_rets.iloc[-20:].std()  * np.sqrt(252)) if len(spx_rets) >= 20 else 0.15
        rv_5d     = float(spx_rets.iloc[-5:].std()   * np.sqrt(252)) if len(spx_rets) >= 5  else rv_20d
        rv_5d_lag = float(spx_rets.iloc[-10:-5].std()* np.sqrt(252)) if len(spx_rets) >= 10 else rv_5d

        # Features in same units as training data (VIX/100, VVIX/100)
        vix_dec  = (d.get("vix")  or 0) / 100
        vix3m_dec= (d.get("vix3m") or d.get("vix") or 0) / 100
        vvix_dec = (d.get("vvix") or 0) / 100
        pdv_dec  = (d.get("pdv_forecast") or 0) / 100

        feat_live = pd.DataFrame([{
            "vix":          vix_dec,
            "ts_slope":     vix3m_dec - vix_dec,          # (VIX3M - VIX)/100; < 0 = backwardation
            "fear_premium": vix_dec / rv_20d if rv_20d > 0 else 1.0,
            "rv_change_5d": rv_5d - rv_5d_lag,
            "pdv_iv_spread":pdv_dec - vix_dec,            # PDV - IV; < 0 = IV > PDV
            "vvix":         vvix_dec,                     # > 1.0 triggers R2
        }])

        proba = clf.predict_proba(feat_live)[0]
        reg   = int(np.argmax(proba))

        names  = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
        colors = ["#3388ff", "#00ff88", "#ff3333"]
        labels = ["R0", "R1", "R2"]
        d["regime"]       = reg
        d["regime_name"]  = names[reg]
        d["regime_color"] = colors[reg]
        d["regime_label"] = labels[reg]
        d["regime_conf"]  = round(float(proba[reg]) * 100, 1)
        d["regime_date"]  = d.get("spx_date", "live")
        d["prob_r0"]      = round(float(proba[0]) * 100, 1)
        d["prob_r1"]      = round(float(proba[1]) * 100, 1)
        d["prob_r2"]      = round(float(proba[2]) * 100, 1)
    except Exception as e:
        d["regime_error"] = str(e)
        _logger.warning("Live regime inference failed (%s); falling back to cached labels.", e)
        # Fallback: read last known regime from regime_labels.parquet so the
        # page always shows a valid regime and a non-blank "As of" date.
        _cached = False
        try:
            rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
            if "regime" in rl.columns and len(rl) > 0:
                reg = int(rl["regime"].iloc[-1])
                cache_dt = (
                    str(rl.index[-1].date())
                    if hasattr(rl.index[-1], "date")
                    else str(rl.index[-1])[:10]
                )
                if 0 <= reg <= 2:
                    _names  = ["LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE"]
                    _colors = ["#3388ff", "#00ff88", "#ff3333"]
                    _labels = ["R0", "R1", "R2"]
                    d["regime"]       = reg
                    d["regime_name"]  = _names[reg]
                    d["regime_color"] = _colors[reg]
                    d["regime_label"] = _labels[reg]
                    d["regime_conf"]  = 0
                    d["regime_date"]  = f"{cache_dt} (cached)"
                    d["prob_r0"] = 100.0 if reg == 0 else 0.0
                    d["prob_r1"] = 100.0 if reg == 1 else 0.0
                    d["prob_r2"] = 100.0 if reg == 2 else 0.0
                    _cached = True
                    _logger.info("Regime fallback: R%d from %s", reg, cache_dt)
        except Exception as fe:
            _logger.error("Regime cache fallback also failed: %s", fe)

        if not _cached:
            d["regime"]       = -1
            d["regime_name"]  = "UNKNOWN"
            d["regime_color"] = "#666"
            d["regime_label"] = "—"
            d["regime_conf"]  = 0
            d["regime_date"]  = d.get("spx_date", "N/A")   # never leave blank
            d["prob_r0"] = d["prob_r1"] = d["prob_r2"] = 33.3

    # Calibration date for timestamp (Bug 7)
    try:
        with open(DATA / "calibrations" / "joint_cal_2026-03-24.pkl", "rb") as f:
            _cal_tmp = pickle.load(f)
        d["calib_date"] = str(_cal_tmp.get("as_of_date", "2026-03-24"))
    except Exception:
        d["calib_date"] = "2026-03-24"

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

    # ── PDV spread historical context (Bug 3) ────────────────────────────
    try:
        rl = pd.read_parquet(DATA / "signals" / "regime_labels.parquet")
        # pdv_iv_spread in parquet = PDV − VIX (decimal) → negate to get implied−realised (pp)
        hist_spread_pp = -rl["pdv_iv_spread"] * 100
        cur_spread = d.get("pdv_spread", 0) or 0
        d["pdv_spread_mean"] = round(float(hist_spread_pp.mean()), 2)
        d["pdv_spread_pct"]  = round(
            float((hist_spread_pp < cur_spread).mean() * 100), 1
        )
        # elevated / cheap label
        pct = d["pdv_spread_pct"]
        if pct >= 80:
            d["pdv_spread_label"] = f"Top {100-pct:.0f}% — premium selling regime"
            d["pdv_spread_label_color"] = "#ff8833"
        elif pct <= 20:
            d["pdv_spread_label"] = f"Bottom {pct:.0f}% — vol cheap, long vol regime"
            d["pdv_spread_label_color"] = "#3388ff"
        else:
            d["pdv_spread_label"] = f"{pct:.0f}th percentile — neutral"
            d["pdv_spread_label_color"] = "#888"
        # Sparkline: last 60 trading days of spread
        recent = hist_spread_pp.iloc[-60:]
        d["pdv_sparkline"]       = [round(float(x), 2) for x in recent.tolist()]
        d["pdv_sparkline_dates"] = [str(dt.date()) for dt in recent.index]
    except Exception:
        d.setdefault("pdv_spread_mean", 0)
        d.setdefault("pdv_spread_pct", 50)
        d.setdefault("pdv_spread_label", "")
        d.setdefault("pdv_spread_label_color", "#888")
        d.setdefault("pdv_sparkline", [])
        d.setdefault("pdv_sparkline_dates", [])

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
        d["eq_dates"]    = []
        d["eq_nav"]      = []
        d["eq_nav_s1"]   = []
        d["eq_nav_s2"]   = []
        d["eq_nav_s3"]   = []
        d["eq_nav_comb"] = []
        d["final_nav"]   = 0
        d["start_nav"]   = 0
        d["total_pnl"]   = 0

    # ── Metrics — computed entirely from parquet (no pickle import needed) ───
    # This avoids pickle loading joint_vol_calibration.backtest.TradeRecord
    # objects which require the full codebase installed on Railway.
    try:
        eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
        eq.index = pd.to_datetime(eq.index)
        n_days = len(eq)
        ann_factor = np.sqrt(252)

        def _pct(x): return f"{x*100:.2f}%" if x is not None and np.isfinite(float(x)) else "—"
        def _f2(x):  return f"{x:.3f}"       if x is not None and np.isfinite(float(x)) else "—"
        def _dol(x): return f"${x:,.0f}"     if x is not None and np.isfinite(float(x)) else "—"

        def _metrics_for(nav_col: str, pnl_col: str, pos_col: str):
            nav  = eq[nav_col]
            rets = nav.pct_change().fillna(0)
            pnl  = eq[pnl_col]
            # Sharpe (excess over 5% rf)
            ex   = rets - 0.05 / 252
            sh   = float(ex.mean() / ex.std() * ann_factor) if rets.std() > 1e-10 else 0.0
            # Sortino (downside only)
            down = ex[ex < 0]
            srt  = float(ex.mean() / down.std() * ann_factor) if len(down) > 1 else 0.0
            # Max drawdown
            roll_max = nav.cummax()
            dd_series = (nav / roll_max) - 1
            mdd  = float(dd_series.min())
            # DD duration (longest consecutive drawdown)
            in_dd = (dd_series < 0).astype(int)
            max_dur = 0
            cur_dur = 0
            for v in in_dd:
                if v: cur_dur += 1
                else: cur_dur = 0
                max_dur = max(max_dur, cur_dur)
            # Calmar
            years = n_days / 252
            ann_ret = float((nav.iloc[-1] / nav.iloc[0]) ** (1 / years) - 1) if years > 0 else 0
            calmar  = float(ann_ret / abs(mdd)) if abs(mdd) > 1e-10 else 0.0
            # Win rate (by trading day P&L)
            active_pnl = pnl[pnl != 0]
            win_rate = float((active_pnl > 0).mean()) if len(active_pnl) > 0 else 0.0
            # N trades (position changes)
            pos = eq[pos_col]
            n_trades = int((pos.diff().fillna(0) != 0).sum())
            # Total P&L
            total_pnl = float(nav.iloc[-1] - nav.iloc[0])
            cum_ret = float((nav.iloc[-1] - nav.iloc[0]) / nav.iloc[0])
            avg_pnl = total_pnl / n_trades if n_trades > 0 else 0.0
            return dict(
                cum_ret=cum_ret, ann_ret=ann_ret, sharpe=sh, sortino=srt,
                mdd=mdd, dd_dur=max_dur, calmar=calmar, win_rate=win_rate,
                avg_pnl=avg_pnl, best_day=float(rets.max()), worst_day=float(rets.min()),
                n_trades=n_trades, total_pnl=total_pnl,
            )

        m  = _metrics_for("nav",          "daily_pnl",  "s1_position")  # portfolio
        m1 = _metrics_for("nav_s1",       "pnl_s1",     "s1_position")
        m2 = _metrics_for("nav_s2",       "pnl_s2",     "s2_position")
        m3 = _metrics_for("nav_s3",       "pnl_s3",     "s3_position")
        mc = _metrics_for("nav_combined",  "pnl_combined","combined_pos")

        d["metrics_rows"] = [
            {"label": "Cumulative Return", "value": _pct(m["cum_ret"]),
             "positive": m["cum_ret"] > 0},
            {"label": "Ann. Return",       "value": _pct(m["ann_ret"]),
             "positive": m["ann_ret"] > 0},
            {"label": "Sharpe (rf=5%)",    "value": _f2(m["sharpe"]),
             "positive": m["sharpe"] > 0},
            {"label": "Sortino",           "value": _f2(m["sortino"]),
             "positive": m["sortino"] > 0},
            {"label": "Max Drawdown",      "value": _pct(m["mdd"]),     "positive": False},
            {"label": "DD Duration",       "value": f"{m['dd_dur']} days","positive": False},
            {"label": "Calmar",            "value": _f2(m["calmar"]),
             "positive": m["calmar"] > 0},
            {"label": "Win Rate",          "value": _pct(m["win_rate"]),
             "positive": m["win_rate"] > 0.5},
            {"label": "Avg P&L / Trade",   "value": _dol(m["avg_pnl"]),
             "positive": m["avg_pnl"] > 0},
            {"label": "Best Day",          "value": _pct(m["best_day"]),  "positive": True},
            {"label": "Worst Day",         "value": _pct(m["worst_day"]), "positive": False},
            {"label": "N Trades",          "value": str(m["n_trades"]),   "positive": True},
            {"label": "N Days",            "value": str(n_days),          "positive": True},
        ]

        # Signal P&L bar chart
        signal_pnl = [
            round(m1["total_pnl"] / 1000, 1),
            round(m2["total_pnl"] / 1000, 1),
            round(m3["total_pnl"] / 1000, 1),
            round(mc["total_pnl"] / 1000, 1),
        ]
        d["signal_labels"] = ["S1  IVR / PDV", "S2  VIX TS", "S3  Dispersion", "Combined"]
        d["signal_pnl"]    = signal_pnl
        d["signal_colors"] = ["#00c870" if v > 0 else "#ff3333" for v in signal_pnl]

        # Per-signal metrics table
        d["signal_metrics"] = []
        for mx, lbl in [(m1,"S1 IVR"),(m2,"S2 VIX TS"),(m3,"S3 Disp"),(mc,"Combined")]:
            d["signal_metrics"].append({
                "label":     lbl,
                "sharpe":    _f2(mx["sharpe"]),
                "ann_ret":   _pct(mx["ann_ret"]),
                "win_rate":  _pct(mx["win_rate"]),
                "n_trades":  str(mx["n_trades"]),
                "total_pnl": _dol(mx["total_pnl"]),
                "green":     mx["total_pnl"] > 0,
            })

    except Exception as e:
        d["pkl_error"] = str(e)
        d["metrics_rows"]   = []
        d["signal_metrics"] = []
        d["signal_pnl"]     = []
        d["signal_labels"]  = []
        d["signal_colors"]  = []

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


# ── Startup data refresh (Bug 2 fix) ──────────────────────────────────────────

def _refresh_regime_labels() -> None:
    """
    Re-run the regime classifier on the latest DB data and overwrite
    data_store/signals/regime_labels.parquet.

    Called after daily_refresh() completes so that the cached labels
    used by the live-regime fallback (Bug 1) reflect today's data.
    """
    try:
        from joint_vol_calibration.data.database import (
            get_spx_ohlcv,
            get_vix_term_structure_wide,
        )
        from joint_vol_calibration.signals.regime_classifier import (
            build_features,
        )

        today       = pd.Timestamp.today().strftime("%Y-%m-%d")
        spx_df      = get_spx_ohlcv(as_of_date=today)
        vix_wide_df = get_vix_term_structure_wide(as_of_date=today)

        # Use the PDV model for pdv_iv_spread if available
        pdv_model = None
        try:
            with open(DATA / "pdv_model.pkl", "rb") as _f:
                pdv_model = pickle.load(_f)
        except Exception:
            pass

        feats = build_features(spx_df, vix_wide_df, pdv_model=pdv_model)

        with open(DATA / "signals" / "regime_classifier.pkl", "rb") as _f:
            clf = pickle.load(_f)

        labels = clf.predict_series(feats.dropna())
        rl     = feats.reindex(labels.index).copy()
        rl["regime"] = labels

        out_path = DATA / "signals" / "regime_labels.parquet"
        rl.to_parquet(out_path)
        _logger.info(
            "[startup_refresh] regime_labels.parquet updated: %d rows, last date %s",
            len(rl),
            str(rl.index[-1].date()),
        )
    except Exception as exc:
        _logger.error("[startup_refresh] regime_labels.parquet refresh failed: %s", exc)


def _startup_data_refresh() -> None:
    """
    Background thread entry-point — called once when the app starts.

    Checks whether SPX data is stale (> 3 calendar days behind today).
    If stale: runs DataPipeline.daily_refresh() then rebuilds regime_labels.parquet.
    Logs every step so Railway logs show exactly which dates were added.
    """
    try:
        con = _db()
        row = pd.read_sql("SELECT MAX(date) AS last_date FROM spx_ohlcv", con)
        con.close()
        last_str  = str(row["last_date"].iloc[0])[:10]
        last_date = pd.Timestamp(last_str)
        today     = pd.Timestamp.today().normalize()

        # 3-day lag covers weekends + public holidays without false triggers
        if last_date >= today - pd.Timedelta(days=3):
            _logger.info(
                "[startup_refresh] Data is current (last SPX: %s). No refresh needed.",
                last_str,
            )
            return

        _logger.info(
            "[startup_refresh] Stale data: last SPX=%s, today=%s. Starting refresh...",
            last_str,
            today.strftime("%Y-%m-%d"),
        )

        from joint_vol_calibration.data.pipeline import DataPipeline
        pipe    = DataPipeline()
        results = pipe.daily_refresh()

        for src, n_rows in results.items():
            if n_rows:
                _logger.info("[startup_refresh] %-30s  +%d rows", src, n_rows)

        # Rebuild regime labels so the cache reflects the freshest data
        _refresh_regime_labels()

        # Confirm new high-water mark
        con2     = _db()
        row2     = pd.read_sql("SELECT MAX(date) AS last_date FROM spx_ohlcv", con2)
        con2.close()
        new_last = str(row2["last_date"].iloc[0])[:10]
        _logger.info(
            "[startup_refresh] Done. SPX now current to %s (was %s).",
            new_last,
            last_str,
        )

    except Exception as exc:
        _logger.error("[startup_refresh] Startup refresh failed: %s", exc)


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
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    port  = int(_os.environ.get("PORT", _find_free_port(5000)))
    local = not _os.environ.get("PORT")           # True when running locally
    url   = f"http://localhost:{port}"

    print("\n  ┌──────────────────────────────────────────────────┐")
    print("  │   Joint SPX/VIX Volatility System — Dashboard    │")
    print(f"  │   {url:<44}  │")
    print("  └──────────────────────────────────────────────────┘\n")

    # Kick off data refresh in background so the app binds to its port
    # immediately (Railway health-checks require fast startup).
    _refresh_thread = threading.Thread(
        target=_startup_data_refresh,
        name="startup-refresh",
        daemon=True,
    )
    _refresh_thread.start()
    _logger.info("Startup refresh thread launched (daemon=True).")

    if local:
        Timer(1.2, lambda: webbrowser.open(url)).start()

    app.run(host="0.0.0.0", port=port, debug=False, use_reloader=False)
