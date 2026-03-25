"""
report_generator.py — C10: Automated HTML Report Generator

Generates a self-contained HTML report from backtest results.
All charts embedded as base64 PNG; no external file dependencies.

Sections
--------
  1. Header + KPI strip (ann_return, Sharpe, Sortino, max_drawdown, Calmar, win_rate)
  2. Equity curve with drawdown overlay, crisis shading, Regime 2 markers
  3. Monthly returns heatmap (year × month)
  4. Signal contribution breakdown (bar + cumulative line)
  5. Walk-forward Sharpe distribution (bar chart + summary table)
  6. Performance metrics table
  7. Crisis performance table
  8. Per-signal statistics table
  9. Greeks P&L attribution from C7
 10. Honest failure documentation (PDV, Heston rho, C6 vomma nodes)
 11. Assumptions list

Output: data_store/backtest/reports/full_report.html (auto-opens in browser)
"""

from __future__ import annotations

import base64
import io
import logging
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd

from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.backtest.backtest_engine import (
    CRISIS_PERIODS,
    REPORTS_DIR,
    REPORT_PATH,
    BacktestEngine,
    TradeRecord,
)

logger = logging.getLogger(__name__)

# ── Visual constants ───────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    "s1": "#1565C0", "s2": "#2E7D32",
    "s3": "#6A1B9A", "combined": "#B71C1C",
}
CRISIS_COLOR = "#B0BEC5"   # grey shading for crisis periods
REGIME2_COLOR = "#FF9800"  # orange markers for Regime 2 days

# ── Embedded CSS ───────────────────────────────────────────────────────────────

_CSS = """
<style>
*{box-sizing:border-box;margin:0;padding:0}
body{font-family:'Segoe UI',Arial,sans-serif;background:#f5f7fa;color:#212121;font-size:14px}
.header{background:#1a237e;color:#fff;padding:1.5rem 2.5rem}
.header h1{font-size:1.6rem;margin-bottom:.3rem}
.header .sub{opacity:.8;font-size:.9rem}
.kpi-row{display:flex;flex-wrap:wrap;gap:.8rem;padding:1rem 2.5rem}
.kpi{background:#fff;border-radius:6px;padding:.8rem 1.2rem;box-shadow:0 1px 3px rgba(0,0,0,.1);flex:1;min-width:130px}
.kpi-val{font-size:1.4rem;font-weight:700;color:#1a237e}
.kpi-lbl{font-size:.7rem;color:#757575;text-transform:uppercase;letter-spacing:.5px;margin-top:.2rem}
.section{background:#fff;margin:.8rem 2.5rem;padding:1.2rem 1.5rem;border-radius:6px;box-shadow:0 1px 3px rgba(0,0,0,.1)}
.section h2{font-size:1.05rem;color:#1a237e;border-bottom:2px solid #e3f2fd;padding-bottom:.4rem;margin-bottom:.8rem}
.chart{width:100%;height:auto}
table{border-collapse:collapse;width:100%;font-size:.85rem}
th{background:#e3f2fd;color:#1a237e;padding:.5rem .8rem;text-align:left;font-weight:600}
td{padding:.4rem .8rem;border-bottom:1px solid #f0f0f0}
tr:hover td{background:#fafafa}
.pos{color:#2e7d32;font-weight:600}
.neg{color:#c62828;font-weight:600}
.warn{background:#fff3e0;border-left:4px solid #ff9800;padding:.8rem 1rem;border-radius:4px;margin:.5rem 0}
.fail{background:#fce4ec;border-left:4px solid #e53935;padding:.8rem 1rem;border-radius:4px;margin:.5rem 0}
.fail-title{font-weight:600;color:#c62828;margin-bottom:.3rem;font-size:.9rem}
.fail-body{color:#555;font-size:.82rem;line-height:1.6}
.tag-neg{display:inline-block;background:#ffebee;color:#c62828;padding:1px 6px;border-radius:3px;font-size:.75rem;font-weight:600}
.tag-pos{display:inline-block;background:#e8f5e9;color:#2e7d32;padding:1px 6px;border-radius:3px;font-size:.75rem;font-weight:600}
footer{text-align:center;color:#9e9e9e;font-size:.75rem;padding:1.5rem}
ul{padding-left:1.5rem;line-height:1.8;color:#555;font-size:.85rem}
</style>
"""


class ReportGenerator:
    """
    Generates a self-contained HTML backtest report from BacktestEngine results.

    Usage
    -----
      rg   = ReportGenerator(engine, equity_df, metrics, crisis_perf, wf_df)
      path = rg.generate()   # saves HTML and opens in browser
    """

    def __init__(
        self,
        engine:      BacktestEngine,
        equity_df:   pd.DataFrame,
        metrics:     Dict,
        crisis_perf: Dict[str, Dict],
        wf_df:       Optional[pd.DataFrame] = None,
        signals_df:  Optional[pd.DataFrame] = None,
    ):
        self.engine      = engine
        self.equity_df   = equity_df
        self.metrics     = metrics
        self.crisis_perf = crisis_perf
        self.wf_df       = wf_df
        self.signals_df  = signals_df

    # ── Entry point ───────────────────────────────────────────────────────────

    def generate(
        self,
        save_path: Optional[Path] = None,
        auto_open: bool           = True,
    ) -> Path:
        """Build HTML, save to disk, and optionally open in the default browser."""
        path = Path(save_path or REPORT_PATH)
        path.parent.mkdir(parents=True, exist_ok=True)

        html = self._build_html()
        path.write_text(html, encoding="utf-8")
        logger.info("Report saved → %s", path)

        if auto_open:
            try:
                webbrowser.open(path.as_uri())
            except Exception:
                logger.info("Auto-open failed; open manually: %s", path)

        return path

    # ── HTML assembly ─────────────────────────────────────────────────────────

    def _build_html(self) -> str:
        m    = self.metrics
        cap  = self.engine.initial_capital
        nav1 = float(self.equity_df["nav"].iloc[-1]) if len(self.equity_df) > 0 else cap

        def _pct(x, d=1):
            return f"{x * 100:.{d}f}%" if (x is not None and pd.notna(x)) else "—"

        def _f(x, d=2):
            return f"{x:.{d}f}" if (x is not None and pd.notna(x)) else "—"

        def _dol(x):
            return f"${x:,.0f}" if (x is not None and pd.notna(x)) else "—"

        eq_b64  = self._equity_chart()
        mhm_b64 = self._monthly_heatmap()
        sc_b64  = self._signal_contribution()
        wf_b64  = self._walk_forward_chart() if self.wf_df is not None else None

        wf_section = ""
        if wf_b64:
            wf_section = f"""
<div class="section">
  <h2>Walk-Forward Sharpe Distribution (12 × 6-Month OOS Windows, 2019–2025)</h2>
  <img class="chart" src="data:image/png;base64,{wf_b64}" alt="Walk-Forward Sharpe">
  {self._wf_table_html()}
</div>"""

        return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1.0">
<title>Joint SPX/VIX Vol Strategy — Full Backtest Report</title>
{_CSS}
</head>
<body>

<div class="header">
  <h1>Joint SPX/VIX Vol Strategy — Full Backtest Report</h1>
  <div class="sub">Period: {self.engine.start_date} → {self.engine.end_date}
  &nbsp;|&nbsp; Capital: {_dol(cap)} → {_dol(nav1)}
  &nbsp;|&nbsp; Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</div>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{_pct(m.get('ann_return'))}</div>
    <div class="kpi-lbl">Ann. Return</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('sharpe'))}</div>
    <div class="kpi-lbl">Sharpe (5% rf)</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('sortino'))}</div>
    <div class="kpi-lbl">Sortino</div></div>
  <div class="kpi"><div class="kpi-val">{_pct(m.get('max_drawdown'))}</div>
    <div class="kpi-lbl">Max Drawdown</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('calmar'))}</div>
    <div class="kpi-lbl">Calmar</div></div>
  <div class="kpi"><div class="kpi-val">{_pct(m.get('win_rate'))}</div>
    <div class="kpi-lbl">Win Rate</div></div>
</div>

<div class="section">
  <h2>Equity Curve &amp; Drawdown</h2>
  <img class="chart" src="data:image/png;base64,{eq_b64}" alt="Equity Curve">
  <p style="font-size:.78rem;color:#757575;margin-top:.5rem">
    Grey shading = crisis periods (COVID 2020, Fed Hikes 2022, Tariffs 2025).
    Orange triangles = Regime 2 (VOMMA_ACTIVE) days — new entries blocked.
  </p>
</div>

<div class="section">
  <h2>Monthly Returns Heatmap</h2>
  <img class="chart" src="data:image/png;base64,{mhm_b64}" alt="Monthly Returns">
</div>

<div class="section">
  <h2>Signal Contribution Breakdown</h2>
  <img class="chart" src="data:image/png;base64,{sc_b64}" alt="Signal Contribution">
</div>

{wf_section}

<div class="section">
  <h2>Performance Metrics</h2>
  {self._metrics_table_html()}
</div>

<div class="section">
  <h2>Crisis Performance</h2>
  {self._crisis_table_html()}
</div>

<div class="section">
  <h2>Per-Signal Statistics</h2>
  {self._signal_stats_table_html()}
</div>

<div class="section">
  <h2>Greeks P&amp;L Attribution (C7 Framework — 2020 ATM Straddle)</h2>
  {self._greeks_section()}
</div>

<div class="section">
  <h2>⚠ Honest Failure Documentation</h2>
  <p style="color:#555;font-size:.83rem;margin-bottom:.8rem">
    Known model limitations confirmed during development. Documented in full — not hidden.
  </p>
  {self._failures_html()}
</div>

<div class="section">
  <h2>Modelling Assumptions</h2>
  <ul>{"".join(f"<li>{a}</li>" for a in BacktestEngine.ASSUMPTIONS)}</ul>
</div>

<footer>
  Joint SPX/VIX Smile Calibration System — C10 Backtest Report &nbsp;|&nbsp;
  Zero look-ahead · Fixed seed={self.engine.seed} &nbsp;|&nbsp;
  {datetime.now().strftime("%Y-%m-%d %H:%M UTC")}
</footer>

</body>
</html>"""

    # ── Charts ────────────────────────────────────────────────────────────────

    def _equity_chart(self) -> str:
        eq  = self.equity_df
        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(14, 8), gridspec_kw={"height_ratios": [3, 1]}, sharex=True,
        )
        fig.patch.set_facecolor("#fafafa")

        # Crisis period shading
        for _, (cs, ce) in CRISIS_PERIODS.items():
            try:
                for ax in (ax1, ax2):
                    ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                               alpha=0.15, color=CRISIS_COLOR, zorder=0)
            except Exception:
                pass

        # Per-signal NAVs
        sig_map = [("nav_s1","S1 IVR","s1"),("nav_s2","S2 VIX TS","s2"),
                   ("nav_s3","S3 Disp","s3"),("nav_combined","Combined","combined")]
        for col, lbl, sig in sig_map:
            if col in eq.columns:
                ax1.plot(eq.index, eq[col], lw=1.0, alpha=0.55,
                         color=SIGNAL_COLORS[sig], label=lbl)

        ax1.plot(eq.index, eq["nav"], lw=2.2, color="#212121", label="Portfolio NAV", zorder=5)
        ax1.axhline(self.engine.initial_capital, color="#9e9e9e", lw=0.7, ls="--", alpha=0.6)

        # Regime 2 markers
        if "regime" in eq.columns:
            r2 = eq.index[eq["regime"] == 2]
            if len(r2):
                ax1.scatter(r2, eq.loc[r2, "nav"], marker="^", s=16,
                            color=REGIME2_COLOR, zorder=6, alpha=0.7, label="Regime 2")

        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
        ax1.set_ylabel("Portfolio NAV")
        ax1.legend(fontsize=7.5, loc="upper left", framealpha=0.9, ncol=3)
        ax1.grid(True, alpha=0.25)
        ax1.set_title("Delta-Hedged Vol Strategy — Equity Curve", fontsize=11)

        nav = eq["nav"]
        dd  = (nav - nav.cummax()) / nav.cummax() * 100.0
        ax2.fill_between(eq.index, dd, 0, color="#F44336", alpha=0.4)
        ax2.plot(eq.index, dd, color="#F44336", lw=0.7)
        ax2.set_ylabel("Drawdown (%)")
        ax2.grid(True, alpha=0.25)
        ax2.set_ylim(top=3)

        plt.tight_layout()
        return self._fig_b64(fig)

    def _monthly_heatmap(self) -> str:
        nav     = self.equity_df["nav"]
        monthly = (nav.resample("ME").last().pct_change().dropna() * 100.0)
        monthly.index = monthly.index.to_period("M")

        years  = sorted(set(monthly.index.year))
        months = list(range(1, 13))
        MABB   = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]

        grid = pd.DataFrame(np.nan, index=years, columns=months)
        for period, val in monthly.items():
            if period.year in grid.index and period.month in grid.columns:
                grid.loc[period.year, period.month] = val

        arr   = grid.values.astype(float)
        valid = arr[~np.isnan(arr)]
        vmax  = max(abs(valid).max() if len(valid) else 5.0, 3.0)

        fig, ax = plt.subplots(figsize=(14, max(3.5, len(years) * 0.55 + 1.5)))
        fig.patch.set_facecolor("#fafafa")

        cmap = mcolors.LinearSegmentedColormap.from_list(
            "rg", ["#c62828","#ffcdd2","#ffffff","#c8e6c9","#2e7d32"]
        )
        im = ax.imshow(arr, cmap=cmap, vmin=-vmax, vmax=vmax, aspect="auto")

        ax.set_xticks(range(12))
        ax.set_xticklabels(MABB, fontsize=9)
        ax.set_yticks(range(len(years)))
        ax.set_yticklabels(years, fontsize=9)

        for i, y in enumerate(years):
            for j, mo in enumerate(months):
                v = grid.loc[y, mo]
                if not np.isnan(v):
                    fc = "white" if abs(v) > vmax * 0.55 else "#212121"
                    ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                            fontsize=7.5, color=fc, fontweight="bold")

        plt.colorbar(im, ax=ax, label="Monthly Return (%)", shrink=0.55, pad=0.02)
        ax.set_title("Monthly Returns Heatmap (%)", fontsize=11, pad=8)
        plt.tight_layout()
        return self._fig_b64(fig)

    def _signal_contribution(self) -> str:
        eq      = self.equity_df
        sig_map = [("S1 IVR","pnl_s1","s1"),("S2 VIX TS","pnl_s2","s2"),
                   ("S3 Disp","pnl_s3","s3"),("Combined","pnl_combined","combined")]

        labels, totals, cols_used = [], [], []
        for lbl, col, sig in sig_map:
            if col in eq.columns:
                labels.append(lbl)
                totals.append(float(eq[col].sum()))
                cols_used.append((col, sig))

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#fafafa")

        colors = [SIGNAL_COLORS[sig] for _, sig in cols_used]
        bars   = ax1.bar(labels, [v / 1e3 for v in totals], color=colors, edgecolor="white")
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_ylabel("Total Gross P&L ($K)")
        ax1.set_title("Cumulative P&L by Signal")
        ax1.grid(True, alpha=0.25, axis="y")
        for bar, val in zip(bars, totals):
            y_off = 0.5 if val >= 0 else -1.5
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + y_off,
                     f"${val/1e3:.1f}K", ha="center", va="bottom", fontsize=8.5)

        for col, sig in cols_used:
            lbl = {"pnl_s1":"S1","pnl_s2":"S2","pnl_s3":"S3","pnl_combined":"Combined"}.get(col, col)
            ax2.plot(eq.index, eq[col].cumsum() / 1e3,
                     color=SIGNAL_COLORS[sig], label=lbl, lw=1.5)
        ax2.axhline(0, color="black", lw=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax2.set_title("Cumulative P&L Over Time by Signal")
        ax2.legend(fontsize=8.5)
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        return self._fig_b64(fig)

    def _walk_forward_chart(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#fafafa")
            ax.text(0.5, 0.5, "Walk-forward not computed", ha="center", va="center",
                    transform=ax.transAxes, color="#757575")
            return self._fig_b64(fig)

        fig, ax = plt.subplots(figsize=(14, 5))
        fig.patch.set_facecolor("#fafafa")

        sharpes = self.wf_df["sharpe"].fillna(0).values
        x       = np.arange(len(sharpes))
        colors  = ["#c62828" if s < 0 else "#1565C0" for s in sharpes]

        ax.bar(x, sharpes, color=colors, edgecolor="white", linewidth=0.4, width=0.7)
        ax.axhline(0, color="black", lw=1.0)
        ax.axhline(1, color="#2e7d32", lw=1.0, ls="--", label="Sharpe = 1.0", alpha=0.7)
        ax.axhline(-0.5, color="#e53935", lw=0.8, ls=":", alpha=0.5)

        tick_labels = [
            f"W{int(r['window'])}\n{str(r['test_start'])[:7]}"
            for _, r in self.wf_df.iterrows()
        ]
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("OOS Sharpe Ratio (annualised)")
        ax.set_title("Walk-Forward Sharpe Distribution — 12 × 6-Month OOS Windows")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.25, axis="y")

        n_neg = int((np.array(sharpes) < 0).sum())
        clr   = "#c62828" if n_neg > 0 else "#2e7d32"
        ax.text(0.02, 0.96, f"Negative Sharpe windows: {n_neg}/{len(sharpes)}",
                transform=ax.transAxes, fontsize=9, color=clr, va="top")

        plt.tight_layout()
        return self._fig_b64(fig)

    # ── HTML table helpers ─────────────────────────────────────────────────────

    def _metrics_table_html(self) -> str:
        m = self.metrics

        def _p(x, d=2): return f"{x*100:.{d}f}%" if pd.notna(x) else "—"
        def _f(x, d=3): return f"{x:.{d}f}" if pd.notna(x) else "—"
        def _d(x):      return f"${x:,.0f}" if pd.notna(x) else "—"
        def _i(x):      return str(int(x)) if pd.notna(x) else "—"

        rows = [
            ("Cumulative Return",        _p(m.get("cumulative_return"))),
            ("Annualised Return",         _p(m.get("ann_return"))),
            ("Sharpe Ratio (rf = 5%)",    _f(m.get("sharpe"))),
            ("Sortino Ratio",             _f(m.get("sortino"))),
            ("Max Drawdown",              _p(m.get("max_drawdown"))),
            ("Max Drawdown Duration",     f"{_i(m.get('dd_duration_days'))} days"),
            ("Calmar Ratio",              _f(m.get("calmar"))),
            ("Win Rate",                  _p(m.get("win_rate"))),
            ("Avg P&L per Trade",         _d(m.get("avg_pnl_per_trade"))),
            ("Best Single Day",           _p(m.get("best_day"))),
            ("Worst Single Day",          _p(m.get("worst_day"))),
            ("Total Trades",              _i(m.get("n_trades"))),
            ("Total Trading Days",        _i(m.get("n_days"))),
        ]
        body = "".join(f"<tr><td>{k}</td><td><b>{v}</b></td></tr>" for k, v in rows)
        return (f"<table><thead><tr><th>Metric</th><th>Value</th></tr></thead>"
                f"<tbody>{body}</tbody></table>")

    def _crisis_table_html(self) -> str:
        rows = []
        for name, d in self.crisis_perf.items():
            if d.get("n_days", 0) == 0:
                rows.append(f"<tr><td><b>{name}</b></td><td colspan='5'>No data</td></tr>")
                continue
            tr  = d.get("total_return", 0.0) * 100
            dd  = d.get("max_drawdown", 0.0) * 100
            sh  = d.get("sharpe", 0.0)
            wd  = d.get("worst_day", 0.0) * 100
            nd  = d.get("n_days", 0)
            cls = "pos" if tr >= 0 else "neg"
            rows.append(
                f"<tr><td><b>{name.replace('_',' ')}</b></td>"
                f'<td class="{cls}">{tr:+.1f}%</td>'
                f"<td>{dd:.1f}%</td>"
                f"<td>{sh:.2f}</td>"
                f"<td>{wd:.1f}%</td>"
                f"<td>{nd}</td></tr>"
            )
        hdr = ("<thead><tr>"
               "<th>Period</th><th>Total Return</th><th>Max DD</th>"
               "<th>Sharpe</th><th>Worst Day</th><th>Days</th>"
               "</tr></thead>")
        return f"<table>{hdr}<tbody>{''.join(rows)}</tbody></table>"

    def _signal_stats_table_html(self) -> str:
        m    = self.metrics
        rows = []
        for sig, lbl in [("s1","S1 IVR"),("s2","S2 VIX TS"),("s3","S3 Disp"),("combined","Combined")]:
            def _v(key):
                v = m.get(f"{sig}_{key}")
                return v
            wr  = _v("win_rate")
            ap  = _v("avg_pnl")
            nt  = _v("n_trades")
            sh  = _v("sharpe")
            ar  = _v("ann_return")
            tp  = _v("total_pnl")
            cls = "pos" if (sh is not None and pd.notna(sh) and sh >= 0) else "neg"
            rows.append(
                f"<tr><td><b>{lbl}</b></td>"
                f"<td>{f'{wr*100:.1f}%' if wr is not None and pd.notna(wr) else '—'}</td>"
                f"<td>{f'${ap:,.0f}' if ap is not None and pd.notna(ap) else '—'}</td>"
                f"<td>{int(nt) if nt is not None else '—'}</td>"
                f'<td class="{cls}">{f"{sh:.2f}" if sh is not None and pd.notna(sh) else "—"}</td>'
                f"<td>{f'{ar*100:.1f}%' if ar is not None and pd.notna(ar) else '—'}</td>"
                f"<td>{f'${tp:,.0f}' if tp is not None and pd.notna(tp) else '—'}</td></tr>"
            )
        hdr = ("<thead><tr>"
               "<th>Signal</th><th>Win Rate</th><th>Avg P&L/Trade</th><th>N Trades</th>"
               "<th>Sharpe</th><th>Ann Return</th><th>Total P&L</th>"
               "</tr></thead>")
        return f"<table>{hdr}<tbody>{''.join(rows)}</tbody></table>"

    def _greeks_section(self) -> str:
        c7_path = DATA_DIR / "backtest" / "delta_hedge_2020.parquet"
        try:
            c7 = pd.read_parquet(c7_path)
            cum_a   = float(c7["cum_pnl_a"].iloc[-1]) if "cum_pnl_a" in c7.columns else None
            cum_b   = float(c7["cum_pnl_b"].iloc[-1]) if "cum_pnl_b" in c7.columns else None
            pnl_col = "pnl_total" if "pnl_total" in c7.columns else None
            eff_a = (float(c7["pnl_residual_a"].var() / c7[pnl_col].var())
                     if "pnl_residual_a" in c7.columns and pnl_col else None)
            eff_b = (float(c7["pnl_residual_b"].var() / c7[pnl_col].var())
                     if "pnl_residual_b" in c7.columns and pnl_col else None)
            n_unstable = int(c7["is_unstable"].sum()) if "is_unstable" in c7.columns else 0

            def _dol(x): return f"${x:,.2f}" if x is not None and pd.notna(x) else "—"
            def _ef(x):  return f"{x:.4f}"   if x is not None and pd.notna(x) else "—"

            ca_cls = "pos" if cum_a and cum_a >= 0 else "neg"
            cb_cls = "pos" if cum_b and cum_b >= 0 else "neg"
            ea_cls = "pos" if eff_a and eff_a < 0.1 else "neg"
            eb_cls = "neg"

            table = f"""<table>
  <thead><tr><th>Metric</th><th>Run A (Market IV Greeks)</th><th>Run B (PDV-Adjusted Vega)</th></tr></thead>
  <tbody>
    <tr><td>Straddle</td><td colspan="2">ATM SPX, K=3257.85, T=1yr, 2020-01-02 → 2020-12-31</td></tr>
    <tr><td>Cumulative P&L</td>
        <td class="{ca_cls}"><b>{_dol(cum_a)}</b></td>
        <td class="{cb_cls}"><b>{_dol(cum_b)}</b></td></tr>
    <tr><td>Hedge Efficiency (Var(residual) / Var(total P&L))</td>
        <td class="{ea_cls}"><b>{_ef(eff_a)}</b></td>
        <td class="{eb_cls}"><b>{_ef(eff_b)}</b></td></tr>
    <tr><td>C6 Unstable Vomma Days (|z|&gt;2)</td>
        <td colspan="2">{n_unstable} days (2020 straddle at S~3257 vs C6 surface at S~6581)</td></tr>
  </tbody>
</table>
<div class="warn" style="margin-top:.7rem">
  <b>C7 finding:</b> Run A (VIX-interpolated market Greeks) achieves hedge efficiency 0.016 —
  only 1.6% of P&L variance unexplained. Run B (PDV-adjusted) worsens to 0.303 because
  PDV over-forecasts realized vol in 2020 (σ_PDV=91.6% vs σ_ATM=55.4% on 2020-03-16).
</div>"""
        except Exception as exc:
            table = f"<p style='color:#757575'>C7 parquet not found — run DeltaHedger first ({exc})</p>"

        return table

    def _failures_html(self) -> str:
        failures = [
            (
                "PDV Over-Forecast During COVID-19 (C7)",
                "On 2020-03-16, PDV predicted σ = 91.6% vs ATM IV 55.4%. "
                "PDV EWMA features track recent realized vol — they over-respond to the COVID spike. "
                "This caused Run B (PDV-adjusted vega) to over-hedge by 1.65×, worsening efficiency "
                "from 0.016 → 0.303. PDV systematically over-forecasts in sharp vol spikes.",
            ),
            (
                "Heston rho = −0.99 Boundary Convergence (C4)",
                "Joint calibration on 2026-03-24 hits the lower bound rho = −0.99. "
                "Heston cannot reproduce the steep 2026 SPX put skew. Deep OTM option prices "
                "and Greeks from this calibration are less accurate. Impact: straddle vega "
                "under-estimates true vega convexity; S1/S2 vega P&L attribution is approximate.",
            ),
            (
                "C6 Unstable Vomma Nodes — Delta Hedge Insufficient",
                "Three nodes in the C6 Greeks surface have |vomma_z| > 2σ: "
                "1Y deep OTM puts (log_m = −0.30, −0.26) and 1Y far OTM call (log_m = +0.13). "
                "On these nodes, standard delta-hedging is insufficient — vomma (∂Vega/∂σ) "
                "hedges are required but are NOT implemented in this backtest. "
                "P&L residuals are elevated on days matching these conditions.",
            ),
            (
                "Walk-Forward Windows with Sharpe < 0",
                self._wf_failures_text(),
            ),
            (
                "S3 Dispersion Signal is a Proxy (No Real Options Data)",
                "Signal 3 uses VIX/VVIX z-score as proxy for implied correlation because "
                "no historical single-stock option data is in the database. "
                "P&L is modelled as 2% of kelly-weighted NAV per z-unit improvement (S3_Z_SCALE=0.02). "
                "Real dispersion involves long single-stock straddles + short SPX variance; "
                "actual transaction costs are higher than the 0.1% proxy used here.",
            ),
        ]
        parts = []
        for title, body in failures:
            parts.append(
                f'<div class="fail">'
                f'<div class="fail-title">{title}</div>'
                f'<div class="fail-body">{body}</div>'
                f'</div>'
            )
        return "\n".join(parts)

    def _wf_failures_text(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            return "Walk-forward results not computed."
        neg = self.wf_df[self.wf_df["below_zero_sharpe"] == True]
        if len(neg) == 0:
            return "All walk-forward windows show positive Sharpe — no negative windows."
        lines = [
            f"Window {int(r['window'])} ({r['test_start']} → {r['test_end']}): Sharpe = {r['sharpe']:.2f}"
            for _, r in neg.iterrows()
        ]
        return (
            f"{len(neg)}/{len(self.wf_df)} windows with Sharpe < 0. "
            "These represent OOS periods where the strategy underperformed the risk-free rate. "
            + " | ".join(lines) + "."
        )

    def _wf_table_html(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            return ""
        rows = []
        for _, r in self.wf_df.iterrows():
            sh = r.get("sharpe", np.nan)
            ar = r.get("ann_return", np.nan)
            dd = r.get("max_drawdown", np.nan)
            sh_s = f"{sh:.2f}" if pd.notna(sh) else "—"
            ar_s = f"{ar*100:.1f}%" if pd.notna(ar) else "—"
            dd_s = f"{dd*100:.1f}%" if pd.notna(dd) else "—"
            cls  = "neg" if (pd.notna(sh) and sh < 0) else "pos"
            flag = '<span class="tag-neg">⚠ Negative</span>' if r.get("below_zero_sharpe") else '<span class="tag-pos">OK</span>'
            rows.append(
                f"<tr><td>{int(r['window'])}</td>"
                f"<td>{r['train_end']}</td>"
                f"<td>{r['test_start']} → {r['test_end']}</td>"
                f'<td class="{cls}">{sh_s}</td>'
                f"<td>{ar_s}</td>"
                f"<td>{dd_s}</td>"
                f"<td>{flag}</td></tr>"
            )
        hdr = ("<thead><tr>"
               "<th>Win.</th><th>Train End</th><th>Test Period</th>"
               "<th>Sharpe</th><th>Ann Ret</th><th>Max DD</th><th>Flag</th>"
               "</tr></thead>")
        return f"<table style='margin-top:.8rem'>{hdr}<tbody>{''.join(rows)}</tbody></table>"

    # ── Utility ───────────────────────────────────────────────────────────────

    @staticmethod
    def _fig_b64(fig: plt.Figure) -> str:
        """Save matplotlib figure to base64-encoded PNG string."""
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        plt.close(fig)
        buf.seek(0)
        return base64.b64encode(buf.read()).decode("utf-8")
