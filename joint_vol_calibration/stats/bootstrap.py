"""
bootstrap.py — stationary-bootstrap confidence intervals for P&L statistics.

The stationary bootstrap (Politis & Romano 1994, JASA 89(428)) resamples a
time series in blocks of random, geometrically distributed length with mean
``b``, wrapping around the end, so the resampled series is stationary and
keeps the dependence within each block. The mean block length comes from the
Politis & White (2004) automatic selector with the Patton, Politis & White
(2009) correction, as implemented by ``arch.bootstrap.optimal_block_length``
(arch, NCSA licence).

A Sharpe ratio is a smooth function of the means of r and r², so its
sampling distribution depends on the dependence of both: daily straddle P&L
has little autocorrelation in r but strong volatility clustering. The block
length used is therefore the larger of the selector's values for r and r²
(``block_length``). Every statistic of one series is computed on the same
resampled index paths, and the paths are seeded, so the intervals are
reproducible bit for bit.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import numpy as np
import pandas as pd

from joint_vol_calibration.stats.sharpe import TRADING_DAYS


def block_length(x: Sequence[float], include_squares: bool = True) -> float:
    """Politis-White(-Patton) mean block length for the stationary bootstrap.

    The larger of the values for x and (if ``include_squares``) x², floored
    at 1. Delegates the selector itself to arch.
    """
    from arch.bootstrap import optimal_block_length

    x = np.asarray(x, dtype=float)
    vals = [float(optimal_block_length(x)["stationary"].iloc[0])]
    if include_squares:
        vals.append(float(optimal_block_length(x ** 2)["stationary"].iloc[0]))
    vals = [v for v in vals if np.isfinite(v)]
    return max([1.0, *vals])


def stationary_bootstrap_indices(n: int, mean_block: float, reps: int,
                                 rng: np.random.Generator) -> np.ndarray:
    """(reps, n) int array of stationary-bootstrap row indices.

    Row 0 of each path starts at a uniform index; every later row starts a new
    block with probability p = 1/mean_block (a fresh uniform index) and
    otherwise continues the block at the next index, modulo n.
    """
    if n < 1 or reps < 1:
        raise ValueError("n and reps must be positive")
    if not mean_block >= 1.0:
        raise ValueError(f"mean_block must be >= 1, got {mean_block!r}")
    p = 1.0 / float(mean_block)
    starts = rng.integers(0, n, size=(reps, n))
    new_block = rng.random((reps, n)) < p
    new_block[:, 0] = True
    # Position inside the current block: 0 at a block start, +1 per step.
    step = np.arange(n)
    last_start = np.where(new_block, step, 0)
    np.maximum.accumulate(last_start, axis=1, out=last_start)
    offset = step[None, :] - last_start
    first = np.take_along_axis(starts, last_start, axis=1)
    return ((first + offset) % n).astype(np.int64)


# ── Statistics evaluated on a (reps, n) matrix of resampled series ─────────────

def _sharpe_rows(r: np.ndarray) -> np.ndarray:
    sd = r.std(axis=1, ddof=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(sd > 0, r.mean(axis=1) / sd, np.nan) * np.sqrt(TRADING_DAYS)


def _sortino_rows(r: np.ndarray) -> np.ndarray:
    dd = np.sqrt(np.mean(np.minimum(r, 0.0) ** 2, axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(dd > 0, r.mean(axis=1) / dd, np.nan) * np.sqrt(TRADING_DAYS)


STATISTICS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    # name: f(resampled returns, resampled daily P&L) → one value per row
    "sharpe": lambda r, p: _sharpe_rows(r),
    "sortino": lambda r, p: _sortino_rows(r),
    "mean_daily_pnl": lambda r, p: p.mean(axis=1),
}


def bootstrap_series(
    returns: Sequence[float],
    pnl: Sequence[float] | None = None,
    reps: int = 5000,
    seed: int = 42,
    mean_block: float | None = None,
    level: float = 0.95,
    chunk: int = 500,
) -> dict:
    """Stationary-bootstrap distribution of Sharpe, Sortino and mean daily P&L.

    Parameters
    ----------
    returns    : daily returns of the NAV (the Sharpe/Sortino input)
    pnl        : daily P&L in $ on the same rows (mean_daily_pnl); defaults
                 to ``returns``
    reps       : bootstrap replications
    seed       : seed of the index paths (numpy default_rng)
    mean_block : mean block length; None → ``block_length(returns)``
    level      : two-sided percentile-interval level

    Returns
    -------
    dict with 'block', 'reps', 'seed', 'level', 'point' {stat: value},
    'ci' {stat: [lo, hi]}, 'se' {stat}, 'frac_gt_0' {stat} and
    'replicates' {stat: ndarray (reps,)}.
    """
    r = np.asarray(returns, dtype=float)
    p = r if pnl is None else np.asarray(pnl, dtype=float)
    if p.shape != r.shape:
        raise ValueError("returns and pnl must have the same length")
    b = block_length(r) if mean_block is None else float(mean_block)
    rng = np.random.default_rng(seed)
    reps_out = {k: np.empty(reps) for k in STATISTICS}
    done = 0
    while done < reps:
        m = min(chunk, reps - done)
        idx = stationary_bootstrap_indices(r.size, b, m, rng)
        rr, pp = r[idx], p[idx]
        for k, f in STATISTICS.items():
            reps_out[k][done:done + m] = f(rr, pp)
        done += m
    point = {k: float(f(r[None, :], p[None, :])[0]) for k, f in STATISTICS.items()}
    a = (1.0 - level) / 2.0
    ci, se, gt0 = {}, {}, {}
    for k, v in reps_out.items():
        fin = v[np.isfinite(v)]
        ci[k] = ([float(np.quantile(fin, a)), float(np.quantile(fin, 1.0 - a))] if fin.size
                 else [np.nan, np.nan])
        se[k] = float(fin.std(ddof=1)) if fin.size > 1 else float("nan")
        gt0[k] = float((fin > 0).mean()) if fin.size else float("nan")
    return {"block": b, "reps": int(reps), "seed": int(seed), "level": float(level),
            "point": point, "ci": ci, "se": se, "frac_gt_0": gt0, "replicates": reps_out}


def bootstrap_table(
    series: Mapping[str, dict[str, Sequence[float]]],
    reps: int = 5000,
    seed: int = 42,
    level: float = 0.95,
    fixed_block: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """bootstrap_series for several named series.

    ``series`` maps a name to {'returns': ..., 'pnl': ...}. Each series gets
    its own seed (seed + position in the mapping) so adding a series does not
    change the others' paths. Returns (summary, replicates): summary has one
    row per (series, statistic) with point, ci_lo, ci_hi, se, frac_gt_0,
    excludes_zero, block; replicates is long-form (series, rep, statistic
    columns).
    """
    rows, reps_frames = [], []
    for i, (name, d) in enumerate(series.items()):
        res = bootstrap_series(d["returns"], d.get("pnl"), reps=reps, seed=seed + i,
                               mean_block=fixed_block, level=level)
        for k in STATISTICS:
            lo, hi = res["ci"][k]
            rows.append({"series": name, "statistic": k, "point": res["point"][k],
                         "ci_lo": lo, "ci_hi": hi, "se": res["se"][k],
                         "frac_gt_0": res["frac_gt_0"][k],
                         "excludes_zero": bool(lo > 0 or hi < 0),
                         "block": res["block"], "reps": res["reps"], "seed": res["seed"],
                         "level": res["level"]})
        reps_frames.append(pd.DataFrame({"series": name, "rep": np.arange(reps),
                                         **{k: res["replicates"][k] for k in STATISTICS}}))
    return pd.DataFrame(rows), pd.concat(reps_frames, ignore_index=True)
