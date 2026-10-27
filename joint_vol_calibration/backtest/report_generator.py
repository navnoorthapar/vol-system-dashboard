"""
report_generator.py — C10: Automated HTML Report Generator

Generates a self-contained HTML report from backtest results.
All charts embedded as base64 PNG; no external file dependencies.

Sections
--------
  1. Header + KPI strip (ann_return, Sharpe, Sortino, max_drawdown, Calmar,
     win_rate, T-bill cash benchmark)
  2. Equity curve: portfolio + its sleeves (S1C/S3/S4), research references
     (S1/S2/Combined) greyed, T-bill cash line, crisis shading, R2 markers
  3. Monthly returns heatmap (year × month, first month included)
  4. Signal contribution breakdown (net P&L, portfolio vs references)
  5. Sub-period stability (6-month windows; not out-of-sample)
  6. Performance metrics table
  7. Crisis performance table
  8. Per-signal statistics table
  9. S3 timing null test (if run)
 10. Greeks P&L attribution from C7 (read from its parquet)
 11. Honest failure documentation (BacktestEngine.HONEST_FAILURES + data)
 12. Assumptions list

Every number in the text is read from the results passed in or from the
C7 parquet; nothing is hard-coded, so a re-run cannot leave the prose stale.

Output: data_store/backtest/reports/full_report.html (auto-opens in browser)
"""

from __future__ import annotations

import base64
import html as _html
import io
import logging
import webbrowser
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

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
    PORTFOLIO_SIGNALS,
    REPORT_PATH,
    BacktestEngine,
)

logger = logging.getLogger(__name__)

# ── Visual constants ───────────────────────────────────────────────────────────

SIGNAL_COLORS = {
    # Portfolio sleeves
    "s1c": "#1565C0", "s3": "#6A1B9A", "s4": "#2E7D32",
    # Research references (drawn grey/dashed)
    "s1": "#9E9E9E", "s2": "#BDBDBD", "combined": "#757575",
}
SIGNAL_LABELS = {
    "s1c": "S1C Contrarian PDV", "s3": "S3 Dispersion (index leg)", "s4": "S4 VRP",
    "s1": "S1 PDV–IV (ref)", "s2": "S2 VIX TS (ref)", "combined": "Combined (ref)",
}
REFERENCE_ORDER = ("s1", "s2", "combined")
CRISIS_COLOR = "#B0BEC5"   # grey shading for crisis periods
REGIME2_COLOR = "#FF9800"  # orange markers for Regime 2 days
CASH_COLOR = "#00897B"

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
tr.ref td{color:#9e9e9e}
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

SHARPE_LABEL = "Sharpe (rf=0 on excess P&L)"


def _esc(x) -> str:
    return _html.escape(str(x))


class ReportGenerator:
    """
    Generates a self-contained HTML backtest report from BacktestEngine results.

    Usage
    -----
      rg   = ReportGenerator(engine, equity_df, metrics, crisis_perf, wf_df,
                             s3_null=s3_timing_null_test(engine))
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
        s3_null:     Optional[Dict] = None,
        c7_path:     Optional[Path] = None,
    ):
        self.engine      = engine
        self.equity_df   = equity_df
        self.metrics     = metrics
        self.crisis_perf = crisis_perf
        self.wf_df       = wf_df
        self.signals_df  = signals_df
        self.s3_null     = s3_null
        self.c7_path     = Path(c7_path) if c7_path else DATA_DIR / "backtest" / "delta_hedge_2020.parquet"

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
        beta = getattr(self.engine, "atm_to_vix_ratio", None)

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
  <h2>{_esc(self._wf_heading())}</h2>
  <p style="font-size:.8rem;color:#555;margin-bottom:.5rem">
    Each window re-runs the same rules from a flat book with fresh capital. Nothing
    is re-fitted per window (each reads the main run's causal PDV forecast), and
    the rules were chosen on the whole sample — this is a stability check, not
    an out-of-sample test. Sharpe uses rf = 0 on the excess P&L.
  </p>
  <img class="chart" src="data:image/png;base64,{wf_b64}" alt="Sub-period Sharpe">
  {self._wf_table_html()}
</div>"""

        null_section = ""
        if self.s3_null:
            null_section = f"""
<div class="section">
  <h2>S3 Timing Null Test</h2>
  {self._null_html()}
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
  &nbsp;|&nbsp; ATM vol = {_f(beta)} × VIX term structure
  &nbsp;|&nbsp; Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}</div>
</div>

<div class="kpi-row">
  <div class="kpi"><div class="kpi-val">{_pct(m.get('cumulative_return'))}</div>
    <div class="kpi-lbl">Cumulative Return</div></div>
  <div class="kpi"><div class="kpi-val">{_pct(m.get('cash_cum_return'))}</div>
    <div class="kpi-lbl">T-bill Cash (benchmark)</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('sharpe'))}</div>
    <div class="kpi-lbl">{SHARPE_LABEL}</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('sortino'))}</div>
    <div class="kpi-lbl">Sortino (rf=0)</div></div>
  <div class="kpi"><div class="kpi-val">{_pct(m.get('max_drawdown'))}</div>
    <div class="kpi-lbl">Max Drawdown</div></div>
  <div class="kpi"><div class="kpi-val">{_f(m.get('calmar'))}</div>
    <div class="kpi-lbl">Calmar</div></div>
  <div class="kpi"><div class="kpi-val">{_pct(m.get('win_rate'))}</div>
    <div class="kpi-lbl">Win Rate (S1C/S3/S4 trades)</div></div>
</div>

<div class="section">
  <h2>Equity Curve &amp; Drawdown</h2>
  <img class="chart" src="data:image/png;base64,{eq_b64}" alt="Equity Curve">
  <p style="font-size:.78rem;color:#757575;margin-top:.5rem">
    Portfolio = (S1C + S3 + S4) / 3, each sleeve sized off its own running NAV.
    Grey dashed lines are research references with no portfolio weight.
    The teal line is the $1M collateral compounding at ^IRX — the strategy P&L
    is measured in excess of it.
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

{null_section}

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
  <ul>{"".join(f"<li>{_esc(a)}</li>" for a in BacktestEngine.ASSUMPTIONS)}</ul>
</div>

<footer>
  Joint SPX/VIX Calibration System — C10 Backtest Report &nbsp;|&nbsp;
  Signals use t−1 information; rules, thresholds, β and S1C's direction were
  chosen with knowledge of the whole sample &nbsp;|&nbsp; Fixed seed={self.engine.seed} &nbsp;|&nbsp;
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
            for ax in (ax1, ax2):
                ax.axvspan(pd.Timestamp(cs), pd.Timestamp(ce),
                           alpha=0.15, color=CRISIS_COLOR, zorder=0)

        for sig in PORTFOLIO_SIGNALS:
            col = f"nav_{sig}"
            if col in eq.columns:
                ax1.plot(eq.index, eq[col], lw=1.2, alpha=0.8,
                         color=SIGNAL_COLORS[sig], label=SIGNAL_LABELS[sig])
        for sig in REFERENCE_ORDER:
            col = f"nav_{sig}"
            if col in eq.columns:
                ax1.plot(eq.index, eq[col], lw=0.9, alpha=0.6, ls="--",
                         color=SIGNAL_COLORS[sig], label=SIGNAL_LABELS[sig])
        if "nav_tbill" in eq.columns:
            ax1.plot(eq.index, eq["nav_tbill"], lw=1.4, color=CASH_COLOR,
                     label="T-bill cash ($1M at ^IRX)")

        ax1.plot(eq.index, eq["nav"], lw=2.2, color="#212121", label="Portfolio NAV", zorder=5)
        ax1.axhline(self.engine.initial_capital, color="#9e9e9e", lw=0.7, ls="--", alpha=0.6)

        # Regime 2 markers
        if "regime" in eq.columns:
            r2 = eq.index[eq["regime"] == 2]
            if len(r2):
                ax1.scatter(r2, eq.loc[r2, "nav"], marker="^", s=16,
                            color=REGIME2_COLOR, zorder=6, alpha=0.7, label="Regime 2")

        ax1.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x/1e6:.2f}M"))
        ax1.set_ylabel("NAV")
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

    def _monthly_returns(self) -> pd.Series:
        """Monthly NAV returns (%), the first month measured from initial capital."""
        nav = self.equity_df["nav"]
        month_end = nav.resample("ME").last()
        prev = month_end.shift(1)
        prev.iloc[0] = float(self.engine.initial_capital)
        monthly = (month_end / prev - 1.0).dropna() * 100.0
        monthly.index = monthly.index.to_period("M")
        return monthly

    def _monthly_heatmap(self) -> str:
        monthly = self._monthly_returns()

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
        eq   = self.equity_df
        sigs = [s for s in (*PORTFOLIO_SIGNALS, *REFERENCE_ORDER) if f"pnl_{s}" in eq.columns]

        labels = [SIGNAL_LABELS[s].replace(" (ref)", "\n(ref)") for s in sigs]
        totals = [float(eq[f"pnl_{s}"].sum()) for s in sigs]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.patch.set_facecolor("#fafafa")

        colors = [SIGNAL_COLORS[s] for s in sigs]
        bars   = ax1.bar(labels, [v / 1e3 for v in totals], color=colors, edgecolor="white")
        for bar, s in zip(bars, sigs):
            if s in REFERENCE_ORDER:
                bar.set_hatch("//")
        ax1.axhline(0, color="black", lw=0.8)
        ax1.set_ylabel("Total Net P&L ($K, full-size sleeve)")
        ax1.set_title("Net P&L by Signal (portfolio sleeves | references)")
        ax1.grid(True, alpha=0.25, axis="y")
        ax1.tick_params(axis="x", labelsize=7.5)
        for bar, val in zip(bars, totals):
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                     f"${val/1e3:.1f}K", ha="center",
                     va="bottom" if val >= 0 else "top", fontsize=8.5)

        for s in sigs:
            ax2.plot(eq.index, eq[f"pnl_{s}"].cumsum() / 1e3,
                     color=SIGNAL_COLORS[s], label=SIGNAL_LABELS[s],
                     lw=1.5 if s in PORTFOLIO_SIGNALS else 1.0,
                     ls="-" if s in PORTFOLIO_SIGNALS else "--")
        ax2.axhline(0, color="black", lw=0.8)
        ax2.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"${x:.0f}K"))
        ax2.set_title("Cumulative Net P&L Over Time by Signal")
        ax2.legend(fontsize=7.5)
        ax2.grid(True, alpha=0.25)

        plt.tight_layout()
        return self._fig_b64(fig)

    def _wf_heading(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            return "Sub-Period Stability"
        n = len(self.wf_df)
        first = str(self.wf_df["test_start"].iloc[0])[:7]
        last = str(self.wf_df["test_end"].iloc[-1])[:7]
        return f"Sub-Period Stability ({n} × 6-Month Windows, {first} → {last}; not out-of-sample)"

    def _walk_forward_chart(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            fig, ax = plt.subplots(figsize=(10, 3))
            fig.patch.set_facecolor("#fafafa")
            ax.text(0.5, 0.5, "Sub-period windows not computed", ha="center", va="center",
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

        tick_labels = [
            f"W{int(r['window'])}\n{str(r['test_start'])[:7]}"
            for _, r in self.wf_df.iterrows()
        ]
        ax.set_xticks(list(x))
        ax.set_xticklabels(tick_labels, fontsize=8)
        ax.set_ylabel("Window Sharpe (rf=0, annualised)")
        ax.set_title(self._wf_heading())
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

        def _p(x, d=2): return f"{x*100:.{d}f}%" if x is not None and pd.notna(x) else "—"
        def _f(x, d=3): return f"{x:.{d}f}" if x is not None and pd.notna(x) else "—"
        def _d(x):      return f"${x:,.0f}" if x is not None and pd.notna(x) else "—"
        def _i(x):      return str(int(x)) if x is not None and pd.notna(x) else "—"

        rows = [
            ("Cumulative Return",          _p(m.get("cumulative_return"))),
            ("T-bill Cash, same dates",    _p(m.get("cash_cum_return"))),
            ("Annualised Return",          _p(m.get("ann_return"))),
            (SHARPE_LABEL,                 _f(m.get("sharpe"))),
            ("Sortino Ratio (rf=0)",       _f(m.get("sortino"))),
            ("Max Drawdown",               _p(m.get("max_drawdown"))),
            ("Max Drawdown Duration",      f"{_i(m.get('dd_duration_days'))} days"),
            ("Calmar Ratio",               _f(m.get("calmar"))),
            ("Win Rate (S1C/S3/S4 trades)", _p(m.get("win_rate"))),
            ("Avg P&L per Trade (full-size sleeve)", _d(m.get("avg_pnl_per_trade"))),
            ("Best Single Day",            _p(m.get("best_day"))),
            ("Worst Single Day",           _p(m.get("worst_day"))),
            ("Total Trades (S1C/S3/S4)",   _i(m.get("n_trades"))),
            ("Total Trading Days",         _i(m.get("n_days"))),
            ("ATM vol / VIX (β)",          _f(m.get("atm_to_vix_ratio"), 2)),
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
               "<th>Sharpe (rf=0)</th><th>Worst Day</th><th>Days</th>"
               "</tr></thead>")
        return f"<table>{hdr}<tbody>{''.join(rows)}</tbody></table>"

    def _signal_stats_table_html(self) -> str:
        m    = self.metrics
        rows = []
        for sig in (*PORTFOLIO_SIGNALS, *REFERENCE_ORDER):
            wr  = m.get(f"{sig}_win_rate")
            ap  = m.get(f"{sig}_avg_pnl")
            nt  = m.get(f"{sig}_n_trades")
            no  = m.get(f"{sig}_n_open")
            sh  = m.get(f"{sig}_sharpe")
            ar  = m.get(f"{sig}_ann_return")
            tp  = m.get(f"{sig}_total_pnl")
            cls = "pos" if (sh is not None and pd.notna(sh) and sh >= 0) else "neg"
            nt_s = "—" if nt is None else (f"{int(nt)} ({int(no)} open)" if no else str(int(nt)))
            row_cls = ' class="ref"' if sig in REFERENCE_ORDER else ""
            rows.append(
                f"<tr{row_cls}><td><b>{_esc(SIGNAL_LABELS[sig])}</b></td>"
                f"<td>{f'{wr*100:.1f}%' if wr is not None and pd.notna(wr) else '—'}</td>"
                f"<td>{f'${ap:,.0f}' if ap is not None and pd.notna(ap) else '—'}</td>"
                f"<td>{nt_s}</td>"
                f'<td class="{cls}">{f"{sh:.2f}" if sh is not None and pd.notna(sh) else "—"}</td>'
                f"<td>{f'{ar*100:.1f}%' if ar is not None and pd.notna(ar) else '—'}</td>"
                f"<td>{f'${tp:,.0f}' if tp is not None and pd.notna(tp) else '—'}</td></tr>"
            )
        hdr = ("<thead><tr>"
               "<th>Signal</th><th>Win Rate</th><th>Avg Net P&L/Trade</th><th>N Trades</th>"
               f"<th>{SHARPE_LABEL}</th><th>Ann Return</th><th>Total Net P&L</th>"
               "</tr></thead>")
        note = ("<p style='font-size:.78rem;color:#757575;margin-top:.4rem'>Each sleeve is "
                "shown at full size on its own $1M; the portfolio holds one third of each. "
                "S3 is the short-SPX-straddle index leg of the dispersion trade (single-stock "
                "leg not modelled). Trades still open on the last date are included at their "
                "mark.</p>")
        return f"<table>{hdr}<tbody>{''.join(rows)}</tbody></table>{note}"

    def _null_html(self) -> str:
        n = self.s3_null
        rows = []
        for key, lbl in (("index_leg", "S3 index leg (short SPX straddle) at base size, "
                                       "before the sizing overlay"),
                         ("z_proxy", "Pre-C22 z-unit proxy P&amp;L at base size")):
            s = n[key]
            rows.append(
                f"<tr><td>{lbl}</td><td>${s['real']:,.0f}</td><td>${s['null_mean']:,.0f}</td>"
                f"<td>${s['null_std']:,.0f}</td><td>{s['null_frac_positive']:.0%}</td>"
                f"<td><b>{s['p_value']:.3f}</b></td></tr>"
            )
        overlay = n.get("engine_overlay_pnl", self.metrics.get("s3_total_pnl"))
        overlay_txt = (f" (${overlay:,.0f} for the real S3)"
                       if overlay is not None and pd.notna(overlay) else "")
        return (
            f"<p style='font-size:.82rem;color:#555;margin-bottom:.5rem'>The S3 entry/exit rule "
            f"was re-run on {n['n_paths']} synthetic VIX/VVIX ratio paths ({_esc(n['method'])}, "
            f"seed {n['seed']}) that carry no information, keeping the real regime gate and "
            f"market. Every path, the real one included, is sized at S3's base kelly (the "
            f"engine's first pass): the tested statistic is the base-size P&amp;L, not the S3 "
            f"P&amp;L reported in the signal table, which is after the ex-ante sizing overlay"
            f"{overlay_txt}. The overlay rescales the same entries and exits and is not re-run "
            f"on the null paths. Real S3: {n['real_trades']} entries; null mean "
            f"{n['null_mean_trades']:.1f}. p = (1 + #null ≥ real)/(N + 1).</p>"
            "<table><thead><tr><th>Statistic</th><th>Real</th><th>Null mean</th><th>Null sd</th>"
            "<th>Null paths &gt; 0</th><th>p-value</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table>"
            "<div class='warn' style='margin-top:.6rem'>A z-score drifts back toward 0 through "
            "its own rolling normalisation, so the pre-C22 proxy books positive 'P&amp;L' on "
            "most random walks; its positivity was never evidence of edge.</div>"
        )

    def _c7_frame(self) -> Optional[pd.DataFrame]:
        try:
            return pd.read_parquet(self.c7_path)
        except Exception as exc:
            logger.info("C7 parquet not readable (%s)", exc)
            return None

    def _greeks_section(self) -> str:
        c7 = self._c7_frame()
        if c7 is None:
            return (f"<p style='color:#757575'>C7 parquet not found at {_esc(self.c7_path)} — "
                    "run `run_backtest --c7` first.</p>")

        from joint_vol_calibration.backtest.delta_hedger import compute_hedge_metrics
        hm = compute_hedge_metrics(c7)
        v = c7.dropna(subset=["pnl_total"])
        has_cross = all(c in c7.columns for c in ("pnl_vanna", "pnl_volga", "pnl_financing"))
        flagged = c7["is_unstable"].astype(bool) if "is_unstable" in c7.columns else pd.Series(False, index=c7.index)
        nxt_res = v["pnl_residual_a"].abs().shift(-1)
        res_flag = nxt_res[flagged.reindex(v.index, fill_value=False)].mean()
        res_rest = nxt_res[~flagged.reindex(v.index, fill_value=False)].mean()

        def _dol(x): return f"${x:,.2f}" if x is not None and pd.notna(x) else "—"
        def _ef(x):  return f"{x:.4f}"   if x is not None and pd.notna(x) else "—"

        start, end = c7.index[0].date(), c7.index[-1].date()
        K = float(c7["S"].iloc[0])
        covid = pd.Timestamp("2020-03-16")
        covid_txt = ""
        if covid in c7.index and pd.notna(c7.loc[covid, "pdv_ratio"]):
            row = c7.loc[covid]
            covid_txt = (f" On 2020-03-16 σ_PDV = {row['sigma_pdv']:.1%} vs σ_ATM = "
                         f"{row['sigma_atm']:.1%} (ratio {row['pdv_ratio']:.2f}).")

        lineage = ("Attribution: Γ + ν + Θ·ΔT (calendar clock) + vanna + volga + financing."
                   if has_cross else
                   "<b>This parquet predates C22</b> (θ/252 on a calendar clock, no vanna/volga/"
                   "financing, full-sample PDV): re-run `run_backtest --c7`.")
        table = f"""<table>
  <thead><tr><th>Metric</th><th>Run A (VIX-TS Greeks)</th><th>Run B (PDV-scaled vega line)</th></tr></thead>
  <tbody>
    <tr><td>Straddle</td><td colspan="2">Long 1 ATM SPX straddle, K={K:,.2f}, T=1yr, {start} → {end}, delta-hedged daily</td></tr>
    <tr><td>Cumulative hedged P&amp;L (same hedge in both runs)</td>
        <td colspan="2"><b>{_dol(hm['cumulative_pnl_a'])}</b></td></tr>
    <tr><td>Unexplained-variance ratio Var(residual)/Var(total)</td>
        <td><b>{_ef(hm['unexplained_var_ratio_a'])}</b></td><td><b>{_ef(hm['unexplained_var_ratio_b'])}</b></td></tr>
    <tr><td>Cumulative residual</td>
        <td>{_dol(hm['cum_residual_a'])}</td><td>{_dol(hm['cum_residual_b'])}</td></tr>
    <tr><td>Vol-of-vol (vomma) flagged days</td>
        <td colspan="2">{hm['n_unstable_days']} days; next-day mean |residual A| {_dol(res_flag)} on
        flagged days vs {_dol(res_rest)} otherwise</td></tr>
  </tbody>
</table>
<div class="warn" style="margin-top:.7rem">
  {lineage} Run B changes only how much of the same P&amp;L is labelled "vega"
  (ν·σ_PDV/σ_ATM·Δσ); the hedge and the P&amp;L are identical to Run A, so Run B
  cannot over- or under-hedge. A higher Run B ratio means the PDV-scaled vega line
  explains less of the daily variance.{covid_txt}
  PDV fitted on data through {_esc(c7.attrs.get('pdv_train_end', 'unknown (see run log)'))}.
</div>"""
        return table

    def _failures_html(self) -> str:
        m = self.metrics
        failures = []
        for f in BacktestEngine.HONEST_FAILURES:
            title, sep, body = f.partition(": ")
            failures.append((title, body) if sep else ("Limitation", f))

        cum, cash = m.get("cumulative_return"), m.get("cash_cum_return")
        if cum is not None and cash is not None and pd.notna(cum) and pd.notna(cash):
            failures.append((
                "Result vs cash",
                f"The portfolio's excess P&L over the window is {cum:+.1%} of capital "
                f"(Sharpe {m.get('sharpe', float('nan')):.2f}, rf=0); the collateral "
                f"alone earned {cash:+.1%} in T-bills.",
            ))
        if self.s3_null:
            p = self.s3_null["index_leg"]["p_value"]
            failures.append((
                "S3 timing vs random timing",
                f"S3's timing has a null-test p-value of {p:.3f} against "
                f"{self.s3_null['n_paths']} random-timing paths. The statistic is S3's "
                f"index-leg P&L at base size, before the sizing overlay that produces "
                f"the S3 P&L in the signal table.",
            ))
        failures.append(("Sub-period windows with Sharpe < 0", self._wf_failures_text()))

        parts = []
        for title, body in failures:
            parts.append(
                f'<div class="fail">'
                f'<div class="fail-title">{_esc(title)}</div>'
                f'<div class="fail-body">{_esc(body)}</div>'
                f'</div>'
            )
        return "\n".join(parts)

    def _wf_failures_text(self) -> str:
        if self.wf_df is None or len(self.wf_df) == 0:
            return "Sub-period windows not computed."
        neg = self.wf_df[self.wf_df["below_zero_sharpe"].astype(bool)]
        if len(neg) == 0:
            return "All sub-period windows show positive Sharpe — no negative windows."
        lines = [
            f"Window {int(r['window'])} ({r['test_start']} → {r['test_end']}): Sharpe = {r['sharpe']:.2f}"
            for _, r in neg.iterrows()
        ]
        return (
            f"{len(neg)}/{len(self.wf_df)} windows with Sharpe < 0 (rf=0: the strategy "
            "lost money in excess of cash). " + " | ".join(lines) + "."
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
                f"<td>{r['test_start']} → {r['test_end']}</td>"
                f'<td class="{cls}">{sh_s}</td>'
                f"<td>{ar_s}</td>"
                f"<td>{dd_s}</td>"
                f"<td>{int(r.get('n_trades', 0) or 0)}</td>"
                f"<td>{flag}</td></tr>"
            )
        hdr = ("<thead><tr>"
               "<th>Win.</th><th>Window</th>"
               "<th>Sharpe (rf=0)</th><th>Ann Ret</th><th>Max DD</th><th>Trades</th><th>Flag</th>"
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
