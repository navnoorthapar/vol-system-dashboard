"""
desk_trades.py — the backtest's round trips and worst drawdowns (B12; pure
functions, no Flask).

A round trip is one TradeRecord of data_store/backtest/full_results.pkl
all_trades: a straddle from its entry to its exit (or to the last session,
still open), marked to model at β·VIX; its P&L is the record's net_pnl (gross
P&L less slippage, plus financing). The open trade carries only its entry
cost, so it is counted apart from the closed ones. The trade log's 'combined'
records are the research-only agreement rule, not a portfolio sleeve; they are
never read here.

A drawdown period of a NAV series (pyfolio's convention): the peak is the last
session at the running maximum before the NAV falls below it, the valley the
lowest NAV before it regains the peak, and the recovery the first session at or
above the peak again (none: 'not recovered', the period runs to the last
session). Depth = valley / peak − 1.
"""
from __future__ import annotations

import itertools
from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np
import pandas as pd

TOP_N = 5
PNL_BINS_K = (-60, -40, -20, -10, 0, 10, 20, 40, 60)        # $000s; the outer bins are open
DAY_BINS = (0, 5, 10, 15, 20, 25, 30)                       # calendar days held; the last bin is open


def round_trips(trades: Iterable[Any], sleeves: Sequence[str]) -> list[dict]:
    """The round trips of ``sleeves``, in entry order."""
    out = []
    for t in trades:
        sig = getattr(t, "signal", None)
        if sig not in sleeves:
            continue
        entry, exit_ = pd.Timestamp(t.entry_date), pd.Timestamp(t.exit_date)
        out.append({"signal": sig, "entry": str(entry.date()), "exit": str(exit_.date()),
                    "days": int((exit_ - entry).days), "pnl": float(t.net_pnl),
                    "direction": int(getattr(t, "direction", 0) or 0),
                    "open": bool(getattr(t, "is_open", False))})
    return sorted(out, key=lambda r: (r["entry"], r["signal"]))


def sleeve_stats(rts: Sequence[dict], top: int = TOP_N) -> dict:
    """One sleeve's round-trip statistics: closed trades only, except ``total``
    (every trade, the open one's entry cost included) and ``n_open``."""
    closed = np.array([r["pnl"] for r in rts if not r["open"]], dtype=float)
    days = np.array([r["days"] for r in rts if not r["open"]], dtype=float)
    gains, losses = closed[closed > 0], closed[closed < 0]
    best = np.sort(gains)[::-1][:top]
    worst = np.sort(losses)[:top]
    return {
        "n": len(rts), "n_closed": int(closed.size), "n_open": sum(1 for r in rts if r["open"]),
        "total": float(sum(r["pnl"] for r in rts)),
        "p_profit": float((closed > 0).mean()) if closed.size else None,
        "mean": float(closed.mean()) if closed.size else None,
        "median": float(np.median(closed)) if closed.size else None,
        "best": float(closed.max()) if closed.size else None,
        "worst": float(closed.min()) if closed.size else None,
        # are the losses a few large trades or many small ones?
        "worst_share": float(worst.sum() / losses.sum()) if losses.size else None,
        "best_share": float(best.sum() / gains.sum()) if gains.size else None,
        "median_days": float(np.median(days)) if days.size else None,
    }


def histogram(values: Sequence[float], edges: Sequence[float], open_low: bool = True) -> list[int]:
    """Counts in the bins of ``edges``: (−∞, e0) when ``open_low``, [e_i, e_i+1),
    and [e_last, ∞)."""
    v = np.asarray(values, dtype=float)
    out = [int((v < edges[0]).sum())] if open_low else []
    out += [int(((v >= a) & (v < b)).sum()) for a, b in itertools.pairwise(edges)]
    out.append(int((v >= edges[-1]).sum()))
    return out


def drawdown_periods(nav: pd.Series) -> list[dict]:
    """Every drawdown period of ``nav`` (see the module docstring), deepest first."""
    nav = nav.astype(float).dropna()
    if nav.empty:
        return []
    v = nav.to_numpy()
    idx = nav.index
    peak_val, peak_i = v[0], 0
    out: list[dict] = []
    cur: dict | None = None
    for i in range(1, len(v)):
        if v[i] >= peak_val:
            if cur is not None:
                cur["recovery_i"] = i
                out.append(cur)
                cur = None
            peak_val, peak_i = v[i], i
        else:
            if cur is None:
                cur = {"peak_i": peak_i, "peak": peak_val, "valley_i": i, "valley": v[i], "recovery_i": None}
            elif v[i] < cur["valley"]:
                cur["valley_i"], cur["valley"] = i, v[i]
    if cur is not None:
        out.append(cur)
    rows = []
    for p in out:
        end_i = p["recovery_i"] if p["recovery_i"] is not None else len(v) - 1
        rows.append({
            "peak": str(pd.Timestamp(idx[p["peak_i"]]).date()),
            "valley": str(pd.Timestamp(idx[p["valley_i"]]).date()),
            "recovery": (str(pd.Timestamp(idx[p["recovery_i"]]).date()) if p["recovery_i"] is not None
                         else None),
            "depth": float(p["valley"] / p["peak"] - 1.0),
            "to_valley": int(p["valley_i"] - p["peak_i"]),
            "sessions": int(end_i - p["peak_i"])})
    return sorted(rows, key=lambda r: r["depth"])


def worst_drawdowns(nav: pd.Series, top: int = TOP_N) -> list[dict]:
    return drawdown_periods(nav)[:top]
