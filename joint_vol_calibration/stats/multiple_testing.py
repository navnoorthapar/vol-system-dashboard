"""
multiple_testing.py — data-snooping-robust tests across many strategies.

``spa_vs_cash``     Hansen (2005) Superior Predictive Ability test, the
                    studentised and recentred refinement of White's (2000)
                    Reality Check, via arch.bootstrap.SPA (arch, NCSA licence).
                    H0: no strategy has a positive expected excess return.
``stepm_vs_cash``   Romano & Wolf (2005) stepwise multiple testing (StepM) via
                    arch.bootstrap.StepM: which strategies, if any, beat cash
                    with family-wise error rate ≤ size.
``pbo_cscv``        Probability of Backtest Overfitting by combinatorially
                    symmetric cross-validation (Bailey, Borwein, López de Prado
                    & Zhu 2017, J. Computational Finance 20(4); SSRN 2326253),
                    written from the paper (the common Python implementation,
                    pypbo, is AGPL and is not used).

The backtest P&L is an excess return over T-bills (collateral assumed to earn
them), so "cash" is the zero-return benchmark: arch's tests take LOSSES, here
loss = −daily return, and the benchmark loss is 0.
"""

from __future__ import annotations

from itertools import combinations
from typing import Dict, Optional

import numpy as np
import pandas as pd


def _as_frame(returns: "pd.DataFrame | np.ndarray") -> pd.DataFrame:
    if isinstance(returns, pd.DataFrame):
        return returns.astype(float)
    a = np.asarray(returns, dtype=float)
    if a.ndim == 1:
        a = a[:, None]
    return pd.DataFrame(a, columns=[f"m{i}" for i in range(a.shape[1])])


def spa_vs_cash(returns: "pd.DataFrame | np.ndarray", block_size: int, reps: int = 5000,
                seed: int = 42) -> Dict:
    """Hansen SPA of every column against a zero excess return.

    Returns the lower / consistent / upper p-values (Hansen's three
    recentring choices; 'consistent' is the headline), the number of
    strategies and the best one's mean daily return.
    """
    from arch.bootstrap import SPA

    df = _as_frame(returns)
    bench = np.zeros(len(df))
    spa = SPA(bench, -df.to_numpy(), block_size=int(block_size), reps=int(reps),
              bootstrap="stationary", studentize=True, seed=int(seed))
    spa.compute()
    pv = spa.pvalues
    best = df.mean().idxmax()
    return {"p_lower": float(pv["lower"]), "p_consistent": float(pv["consistent"]),
            "p_upper": float(pv["upper"]), "n_strategies": int(df.shape[1]),
            "best": str(best), "best_mean_daily_return": float(df[best].mean()),
            "block_size": int(block_size), "reps": int(reps), "seed": int(seed)}


def stepm_vs_cash(returns: "pd.DataFrame | np.ndarray", block_size: int, size: float = 0.05,
                  reps: int = 5000, seed: int = 42) -> Dict:
    """Romano-Wolf StepM: the strategies that beat cash at FWER ≤ ``size``."""
    from arch.bootstrap import StepM

    df = _as_frame(returns)
    bench = pd.Series(np.zeros(len(df)), index=df.index, name="cash")
    st = StepM(bench, -df, size=float(size), block_size=int(block_size), reps=int(reps),
               bootstrap="stationary", studentize=True, seed=int(seed))
    st.compute()
    sup = [str(c) for c in st.superior_models]
    return {"superior": sup, "n_superior": len(sup), "n_strategies": int(df.shape[1]),
            "size": float(size), "block_size": int(block_size), "reps": int(reps), "seed": int(seed)}


def pbo_cscv(returns: "pd.DataFrame | np.ndarray", n_splits: int = 16,
             max_combinations: Optional[int] = None, seed: int = 42) -> Dict:
    """Probability of Backtest Overfitting by CSCV.

    The T × N matrix of per-period returns of N candidate configurations is
    cut into S (even) contiguous blocks. For every choice of S/2 blocks as the
    in-sample set (the complement is out of sample) the configuration with
    the best in-sample Sharpe ratio is "selected"; ω is its relative rank out
    of sample (rank / (N + 1), 1 = best) and λ = log(ω / (1 − ω)). PBO is the
    share of splits with λ ≤ 0: the selected configuration lands at or below
    the out-of-sample median. Also reported: the mean out-of-sample Sharpe of
    the selection, and the slope of OOS on IS Sharpe of the selection
    (performance degradation).

    All C(S, S/2) splits are used unless ``max_combinations`` caps them (a
    seeded random subset). Sharpe ratios are per period.
    """
    df = _as_frame(returns)
    X = df.to_numpy()
    T, N = X.shape
    if N < 2:
        raise ValueError("PBO needs at least two configurations")
    if n_splits % 2 or n_splits < 2:
        raise ValueError("n_splits must be an even number >= 2")
    edges = np.linspace(0, T, n_splits + 1).astype(int)
    s1 = np.stack([X[a:b].sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])        # (S, N)
    s2 = np.stack([(X[a:b] ** 2).sum(axis=0) for a, b in zip(edges[:-1], edges[1:])])
    cnt = np.diff(edges).astype(float)

    def _sr(mask: np.ndarray) -> np.ndarray:
        n = cnt[mask].sum()
        m = s1[mask].sum(axis=0) / n
        v = (s2[mask].sum(axis=0) - n * m ** 2) / (n - 1.0)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(v > 0, m / np.sqrt(np.maximum(v, 0.0)), np.nan)

    combos = list(combinations(range(n_splits), n_splits // 2))
    if max_combinations is not None and len(combos) > max_combinations:
        rng = np.random.default_rng(seed)
        pick = rng.choice(len(combos), size=int(max_combinations), replace=False)
        combos = [combos[i] for i in sorted(pick)]
    lam, is_best, oos_best = [], [], []
    for c in combos:
        m = np.zeros(n_splits, dtype=bool)
        m[list(c)] = True
        sr_is, sr_oos = _sr(m), _sr(~m)
        if not np.isfinite(sr_is).any():
            continue
        j = int(np.nanargmax(sr_is))
        oos = np.where(np.isfinite(sr_oos), sr_oos, -np.inf)
        rank = float((oos < oos[j]).sum() + 1 + 0.5 * ((oos == oos[j]).sum() - 1))
        w = rank / (N + 1.0)
        lam.append(np.log(w / (1.0 - w)))
        is_best.append(float(sr_is[j]))
        oos_best.append(float(sr_oos[j]))
    lam = np.asarray(lam)
    is_best, oos_best = np.asarray(is_best), np.asarray(oos_best)
    ok = np.isfinite(is_best) & np.isfinite(oos_best)
    slope = (float(np.polyfit(is_best[ok], oos_best[ok], 1)[0])
             if ok.sum() > 2 and np.ptp(is_best[ok]) > 0 else float("nan"))
    return {
        "pbo": float((lam <= 0).mean()) if lam.size else float("nan"),
        "n_configurations": int(N), "n_splits": int(n_splits), "n_combinations": int(lam.size),
        "logit_median": float(np.median(lam)) if lam.size else float("nan"),
        "oos_sharpe_of_selected_mean": float(np.nanmean(oos_best)) if oos_best.size else float("nan"),
        "is_sharpe_of_selected_mean": float(np.nanmean(is_best)) if is_best.size else float("nan"),
        "oos_on_is_slope": slope,
        "frac_selected_oos_negative": float((oos_best[ok] < 0).mean()) if ok.any() else float("nan"),
    }
