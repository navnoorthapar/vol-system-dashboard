"""
forecast.py — out-of-sample comparison of 21-session volatility forecasts.

Target (the quantity S1/S1C trade on, horizon-matched with the 30-day VIX):
    y²(t) = 252/h · Σ_{i=1..h} r²_{t+i},   h = 21,
the annualised realised variance of the next h sessions, known at t + h.

Forecasters, each a forecast of y²(t) made at the close of t from data dated
≤ t only (walk-forward; labels used in a fit are fully observed by t):
    pdv        models.pdv.walk_forward_vol_forecast (PDVLinear refit daily,
               purged, embargo h) — the σ̂ behind pdv_iv_spread; squared
    har        HAR-RV (Corsi 2009) on daily squared returns: OLS of y² on
               [1, RV_d, RV_w, RV_m] (1/5/22-session means of 252·r²),
               expanding window, purged (labels i ≤ t − h), refit daily
    garch      GARCH(1,1), zero mean, Gaussian QML (arch), re-estimated
               every 21 sessions on returns ≤ the refit origin, filtered
               daily with those parameters; analytic h-step variance path
    gjr        GJR-GARCH(1,1,1), otherwise as garch
    ewma       RiskMetrics λ = 0.94: σ²_{t+1|t}·252, flat over the horizon
    vix2       (VIX_t / 100)²: the risk-neutral 30-day variance (it contains
               the variance risk premium, so it is biased high by design)
    vix_mz     walk-forward OLS of y² on [1, (VIX_t/100)²] (purged), i.e.
               the VIX rescaled with past data only

Losses (Patton 2011: both are robust to the noise in a realised-variance
proxy, so they rank forecasts consistently):
    QLIKE(y², f) = y²/f − log(y²/f) − 1       MSE(y², f) = (y² − f)²
on annualised variance.

Tests:
    dm_test   Diebold & Mariano (1995) on the loss differential, with the
              Harvey, Leybourne & Newbold (1997) small-sample correction
              and t_{T−1} p-values. Consecutive targets overlap by h − 1
              sessions, so the long-run variance uses the rectangular
              kernel with h − 1 lags (the MA(h−1) structure of optimal
              h-step errors); a Bartlett kernel with 2h lags is reported as
              a robustness check.
    mcs       Model Confidence Set (Hansen, Lunde & Nason 2011) via
              arch.bootstrap.MCS (range statistic, stationary bootstrap).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Literal

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.stats import t as student_t

HORIZON: int = 21
ANNUALISE: int = 252
VAR_FLOOR: float = 1e-6        # annualised variance floor for QLIKE (0.1% vol)


# ── target and losses ─────────────────────────────────────────────────────────

def target_variance(log_returns: pd.Series, horizon: int = HORIZON,
                    annualise: int = ANNUALISE) -> pd.Series:
    """y²(t) = annualise/h · Σ_{i=1..h} r²_{t+i} (NaN for the last h rows)."""
    r = log_returns.astype(float)
    # the mean over rows t+1..t+h is the trailing h-row mean at row t+h
    return (r ** 2).rolling(horizon).mean().shift(-horizon) * annualise


def qlike(y2: np.ndarray, f: np.ndarray) -> np.ndarray:
    y2 = np.asarray(y2, dtype=float)
    f = np.maximum(np.asarray(f, dtype=float), VAR_FLOOR)
    ratio = np.maximum(y2, 1e-12) / f
    return ratio - np.log(ratio) - 1.0


def mse(y2: np.ndarray, f: np.ndarray) -> np.ndarray:
    return (np.asarray(y2, dtype=float) - np.asarray(f, dtype=float)) ** 2


LOSSES = {"qlike": qlike, "mse": mse}


# ── forecasters ───────────────────────────────────────────────────────────────

def pdv_forecast(log_returns: pd.Series, horizon: int = HORIZON) -> pd.Series:
    """The production PDV σ̂ (walk-forward, purged), squared → variance."""
    from joint_vol_calibration.models.pdv import walk_forward_vol_forecast

    s = walk_forward_vol_forecast(log_returns, horizon=horizon)
    return (s ** 2).rename("pdv")


def ewma_forecast(log_returns: pd.Series, lam: float = 0.94,
                  annualise: int = ANNUALISE) -> pd.Series:
    """RiskMetrics σ²_{t+1|t} = λσ²_{t|t−1} + (1 − λ)r_t², annualised (the
    same recursion as models.pdv.forecast_skill_comparison)."""
    r = log_returns.astype(float)
    return ((r ** 2).ewm(alpha=1.0 - lam, adjust=False).mean() * annualise).rename("ewma")


def _walk_forward_ols(X: np.ndarray, y: np.ndarray, horizon: int, train_min: int) -> np.ndarray:
    """Expanding-window OLS forecasts: row t uses labels i ≤ t − horizon.

    X rows with NaN are never used; the normal equations are accumulated one
    label at a time, so each origin costs one small solve.
    """
    n, k = X.shape
    xtx = np.zeros((k, k))
    xty = np.zeros(k)
    used = 0
    out = np.full(n, np.nan)
    for t in range(n):
        i = t - horizon
        if i >= 0 and np.isfinite(X[i]).all() and np.isfinite(y[i]):
            xtx += np.outer(X[i], X[i])
            xty += X[i] * y[i]
            used += 1
        if used >= train_min and np.isfinite(X[t]).all():
            try:
                b = np.linalg.solve(xtx, xty)
            except np.linalg.LinAlgError:
                b = np.linalg.lstsq(xtx, xty, rcond=None)[0]
            out[t] = float(X[t] @ b)
    return out


def har_forecast(log_returns: pd.Series, horizon: int = HORIZON, train_min: int = 504,
                 annualise: int = ANNUALISE) -> pd.Series:
    """HAR-RV on daily squared returns, walk-forward and purged."""
    r = log_returns.astype(float)
    rv = (r ** 2) * annualise
    X = np.column_stack([np.ones(len(r)), rv.to_numpy(),
                         rv.rolling(5).mean().to_numpy(), rv.rolling(22).mean().to_numpy()])
    y = target_variance(r, horizon, annualise).to_numpy()
    f = _walk_forward_ols(X, y, horizon, train_min)
    return pd.Series(np.maximum(f, VAR_FLOOR), index=r.index, name="har").where(np.isfinite(f))


def vix_mz_forecast(log_returns: pd.Series, vix: pd.Series, horizon: int = HORIZON,
                    train_min: int = 504, annualise: int = ANNUALISE) -> pd.Series:
    """Walk-forward OLS of y² on [1, (VIX/100)²], purged."""
    r = log_returns.astype(float)
    v2 = (vix.reindex(r.index).astype(float) / 100.0) ** 2
    X = np.column_stack([np.ones(len(r)), v2.to_numpy()])
    y = target_variance(r, horizon, annualise).to_numpy()
    f = _walk_forward_ols(X, y, horizon, train_min)
    return pd.Series(np.maximum(f, VAR_FLOOR), index=r.index, name="vix_mz").where(np.isfinite(f))


def vix2_forecast(vix: pd.Series) -> pd.Series:
    return ((vix.astype(float) / 100.0) ** 2).rename("vix2")


def garch_forecast(log_returns: pd.Series, horizon: int = HORIZON, o: int = 0,
                   refit_every: int = 21, train_min: int = 504,
                   annualise: int = ANNUALISE, name: str | None = None) -> pd.Series:
    """Walk-forward GARCH(1,1) (o=0) or GJR-GARCH(1,1,1) (o=1) forecasts.

    Every ``refit_every`` sessions the model is estimated (arch, zero mean,
    Gaussian QML, returns in percent) on returns up to and including the
    refit origin s; the forecasts at origins s .. s + refit_every − 1 come
    from that parameter vector filtered over returns up to the last origin
    of the block (arch's fixed-parameter model on the truncated sample, so
    no return after an origin can reach its forecast). The forecast is the
    analytic h-step variance path summed: annualise/h · Σ_{i=1..h} σ²_{t+i|t}.
    """
    from arch import arch_model

    r = log_returns.astype(float).dropna()
    x = r.to_numpy() * 100.0
    n = len(x)
    out = np.full(n, np.nan)
    for s in range(int(train_min), n, int(refit_every)):
        end = min(s + int(refit_every), n)
        fit = arch_model(x[:s + 1], mean="Zero", vol="GARCH", p=1, o=int(o), q=1,
                         dist="normal").fit(disp="off", show_warning=False)
        fixed = arch_model(x[:end], mean="Zero", vol="GARCH", p=1, o=int(o), q=1,
                           dist="normal").fix(fit.params.to_numpy())
        fc = fixed.forecast(horizon=int(horizon), start=s, reindex=False, method="analytic")
        var = fc.variance.to_numpy()          # rows: origins s..end-1, cols: steps 1..h
        out[s:end] = var.sum(axis=1)[: end - s] / 1e4 * annualise / horizon
    return pd.Series(out, index=r.index, name=name or ("gjr" if o else "garch"))


def build_forecasts(log_returns: pd.Series, vix: pd.Series, horizon: int = HORIZON,
                    models: Iterable[str] | None = None) -> pd.DataFrame:
    """Every forecaster on the same date index (NaN where one has no forecast)."""
    r = log_returns.astype(float).dropna()
    builders = {
        "pdv": lambda: pdv_forecast(r, horizon),
        "har": lambda: har_forecast(r, horizon),
        "garch": lambda: garch_forecast(r, horizon, o=0),
        "gjr": lambda: garch_forecast(r, horizon, o=1),
        "ewma": lambda: ewma_forecast(r),
        "vix2": lambda: vix2_forecast(vix),
        "vix_mz": lambda: vix_mz_forecast(r, vix, horizon),
    }
    names = list(builders) if models is None else list(models)
    return pd.DataFrame({m: builders[m]().reindex(r.index) for m in names}, index=r.index)


# ── tests ─────────────────────────────────────────────────────────────────────

def _autocov(d: np.ndarray, k: int) -> float:
    c = d - d.mean()
    return float(c[k:] @ c[:len(c) - k]) / len(c)


def dm_test(loss_a: npt.ArrayLike, loss_b: npt.ArrayLike, horizon: int = 1,
            kernel: str = "rectangular", lags: int | None = None,
            harvey: bool = True) -> dict[str, Any]:
    """Diebold-Mariano test of E[loss_a − loss_b] = 0.

    kernel 'rectangular' uses lags = horizon − 1 unweighted autocovariances
    (falls back to Bartlett if that variance is not positive); 'bartlett' uses
    Newey-West weights with ``lags`` (default 2·horizon). Negative statistic
    ⇒ model a has the lower loss. Returns stat, p_two_sided, p_a_better
    (one-sided, H1: a has lower loss), mean_diff, lags, kernel used.
    """
    d = np.asarray(loss_a, dtype=float) - np.asarray(loss_b, dtype=float)
    d = d[np.isfinite(d)]
    T = d.size
    if T < 10:
        raise ValueError("too few observations for a DM test")
    used = kernel
    if kernel == "rectangular":
        L = max(int(horizon) - 1, 0) if lags is None else int(lags)
        lrv = _autocov(d, 0) + 2.0 * sum(_autocov(d, k) for k in range(1, L + 1))
        if not lrv > 0:
            used = "bartlett"
    if used == "bartlett":
        L = 2 * int(horizon) if lags is None or kernel == "rectangular" else int(lags)
        lrv = _autocov(d, 0) + 2.0 * sum((1.0 - k / (L + 1.0)) * _autocov(d, k) for k in range(1, L + 1))
    elif used != "rectangular":
        raise ValueError(f"unknown kernel {kernel!r}")
    stat = float(d.mean() / np.sqrt(lrv / T))
    h = int(horizon)
    if harvey:
        stat *= float(np.sqrt((T + 1 - 2 * h + h * (h - 1) / T) / T))
        dist: Any = student_t(df=T - 1)
    else:
        from scipy.stats import norm as dist
    return {"stat": stat, "p_two_sided": float(2.0 * dist.sf(abs(stat))),
            "p_a_better": float(dist.cdf(stat)), "mean_diff": float(d.mean()),
            "lags": int(L), "kernel": used, "n": int(T)}


def mcs(losses: pd.DataFrame, size: float = 0.10, block_size: int = HORIZON,
        reps: int = 5000, seed: int = 42, method: Literal["R", "max"] = "R") -> dict:
    """Model Confidence Set via arch: p-value per model and the set at ``size``."""
    from arch.bootstrap import MCS

    m = MCS(losses.astype(float), size=float(size), reps=int(reps), block_size=int(block_size),
            method=method, bootstrap="stationary", seed=int(seed))
    m.compute()
    pv = m.pvalues["Pvalue"].astype(float)
    return {"pvalues": {str(k): float(v) for k, v in pv.items()},
            "included": sorted(str(x) for x in m.included),
            "excluded": sorted(str(x) for x in m.excluded),
            "size": float(size), "block_size": int(block_size), "reps": int(reps),
            "seed": int(seed), "method": method}


def evaluate(forecasts: pd.DataFrame, y2: pd.Series, horizon: int = HORIZON,
             reference: str = "pdv", mcs_reps: int = 5000, seed: int = 42,
             mcs_block: int | None = None) -> dict:
    """Losses, DM tests of ``reference`` against every other model, and the
    MCS at 90% and 75% on the rows where every model and the target exist."""
    df = pd.concat([forecasts, y2.rename("_y2")], axis=1).dropna()
    if df.empty:
        raise ValueError("no common rows")
    y = df.pop("_y2").to_numpy()
    out: dict = {"n": len(df), "start": str(df.index[0].date()), "end": str(df.index[-1].date()),
                 "horizon": int(horizon), "models": list(df.columns), "losses": {}, "dm": {}, "mcs": {},
                 "r2_vol": {}, "mean_forecast_vol": {}, "mean_realised_vol": float(np.sqrt(y).mean())}
    loss_frames = {}
    for lname, lf in LOSSES.items():
        L = pd.DataFrame({m: lf(y, df[m].to_numpy()) for m in df.columns}, index=df.index)
        loss_frames[lname] = L
        out["losses"][lname] = {m: float(L[m].mean()) for m in L.columns}
        out["dm"][lname] = {}
        for m in L.columns:
            if m == reference:
                continue
            rect = dm_test(L[reference], L[m], horizon=horizon, kernel="rectangular")
            bart = dm_test(L[reference], L[m], horizon=horizon, kernel="bartlett")
            out["dm"][lname][m] = {"rectangular_hln": rect, "bartlett_2h": bart}
        blk = int(mcs_block or horizon)
        out["mcs"][lname] = {
            "90": mcs(L, size=0.10, block_size=blk, reps=mcs_reps, seed=seed),
            "75": mcs(L, size=0.25, block_size=blk, reps=mcs_reps, seed=seed),
        }
    vol = np.sqrt(y)
    for m in df.columns:
        fv = np.sqrt(np.maximum(df[m].to_numpy(), 0.0))
        ss_tot = float(((vol - vol.mean()) ** 2).sum())
        out["r2_vol"][m] = 1.0 - float(((vol - fv) ** 2).sum()) / ss_tot
        out["mean_forecast_vol"][m] = float(fv.mean())
    ranks = {}
    for lname in LOSSES:
        order = sorted(out["losses"][lname], key=out["losses"][lname].get)
        ranks[lname] = {m: i + 1 for i, m in enumerate(order)}
    out["rank"] = ranks
    out["_loss_frames"] = loss_frames
    return out


def non_overlapping(df: pd.DataFrame, horizon: int = HORIZON, offset: int = 0) -> pd.DataFrame:
    """Every ``horizon``-th row (targets that do not overlap)."""
    return df.iloc[offset::horizon]
