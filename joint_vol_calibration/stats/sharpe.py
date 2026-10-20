"""
sharpe.py — sampling theory of the Sharpe ratio.

Everything here works on PER-PERIOD (daily) Sharpe ratios; annualise only for
display (``annualise``). The conventions match backtest_engine.compute_metrics:
daily simple returns of the NAV, measured from the NAV before the first
session, sample standard deviation (ddof = 1), and a zero hurdle (the backtest
P&L is already an excess return over T-bills).

Skewness and kurtosis are the plain sample moments (biased estimators, Pearson
kurtosis: 3 for a normal), which is what the formulas below were derived with.

References
----------
Mertens, E. (2002). Comments on variance of the IID estimator in Lo (2002).
    Var(SR̂) ≈ (1 − γ3·SR + (γ4 − 1)/4·SR²) / T for IID non-normal returns.
Lo, A. (2002). The statistics of Sharpe ratios. Financial Analysts Journal 58(4).
Bailey, D. & López de Prado, M. (2012). The Sharpe ratio efficient frontier.
    Journal of Risk 15(2). Probabilistic Sharpe ratio (PSR) and the minimum
    track record length (MinTRL).
Bailey, D. & López de Prado, M. (2014). The deflated Sharpe ratio: correcting
    for selection bias, backtest overfitting and non-normality. Journal of
    Portfolio Management 40(5). SSRN 2460551. DSR and the expected maximum
    Sharpe ratio of N trials. For M correlated trials the independent-trial
    count is approximated by N̂ = ρ̄ + (1 − ρ̄)·M (ρ̄ = average pairwise
    correlation; an approximation, reported beside the raw M).

Caveat that applies to every closed form here: they assume IID returns (the
non-normality is handled, serial dependence is not). Delta-hedged straddle
P&L is serially dependent and has kurtosis near 100 in this backtest, so the
stationary-bootstrap intervals in ``bootstrap.py`` are the primary inference
and these numbers are reported beside them, not instead of them.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np
from scipy.stats import norm

TRADING_DAYS: int = 252
EULER_GAMMA: float = 0.5772156649015329


# ── Basic statistics ──────────────────────────────────────────────────────────

def returns_from_nav(nav: Sequence[float], base_nav: float) -> np.ndarray:
    """Daily simple returns of a NAV path, the first measured from ``base_nav``
    (the NAV at the close before the first session), as compute_metrics does."""
    nav = np.asarray(nav, dtype=float)
    if not (np.isfinite(base_nav) and base_nav > 0):
        raise ValueError(f"base_nav must be a positive NAV, got {base_nav!r}")
    prev = np.concatenate([[float(base_nav)], nav[:-1]])
    return nav / prev - 1.0


def returns_from_pnl(pnl: Sequence[float], capital: float) -> np.ndarray:
    """Daily returns of NAV = capital + cumulative P&L (the backtest's sleeves)."""
    pnl = np.asarray(pnl, dtype=float)
    return returns_from_nav(capital + np.cumsum(pnl), capital)


def sharpe(r: Sequence[float]) -> float:
    """Per-period Sharpe ratio mean / std (ddof = 1); NaN if undefined."""
    r = np.asarray(r, dtype=float)
    if r.size < 2:
        return float("nan")
    sd = float(r.std(ddof=1))
    if not sd > 0:
        return float("nan")
    return float(r.mean() / sd)


def sortino(r: Sequence[float]) -> float:
    """Per-period Sortino ratio: mean / sqrt(mean(min(r, 0)²)) over ALL periods
    (the downside deviation about a zero target, as compute_metrics)."""
    r = np.asarray(r, dtype=float)
    dd = float(np.sqrt(np.mean(np.minimum(r, 0.0) ** 2))) if r.size else 0.0
    if not dd > 0:
        return float("nan")
    return float(r.mean() / dd)


def annualise(sr_per_period: float, periods: int = TRADING_DAYS) -> float:
    """Per-period Sharpe/Sortino → annual (× √periods)."""
    return float(sr_per_period) * float(np.sqrt(periods))


def sample_moments(r: Sequence[float]) -> dict[str, float]:
    """Mean, sd (ddof=1), sample skewness and Pearson kurtosis (normal = 3)."""
    r = np.asarray(r, dtype=float)
    mu = float(r.mean())
    c = r - mu
    m2 = float(np.mean(c ** 2))
    if m2 <= 0:
        return {"mean": mu, "sd": 0.0, "skew": float("nan"), "kurtosis": float("nan"), "n": int(r.size)}
    return {
        "mean": mu,
        "sd": float(r.std(ddof=1)),
        "skew": float(np.mean(c ** 3) / m2 ** 1.5),
        "kurtosis": float(np.mean(c ** 4) / m2 ** 2),
        "n": int(r.size),
    }


# ── Sampling distribution of the Sharpe ratio ────────────────────────────────

def sharpe_variance(sr: float, n: int, skew: float, kurtosis: float) -> float:
    """Var(SR̂) for IID returns with the given skewness and Pearson kurtosis
    (Mertens 2002): (1 − γ3·SR + (γ4 − 1)/4·SR²) / (n − 1), per-period units."""
    if n < 2:
        return float("nan")
    return float((1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2) / (n - 1))


def probabilistic_sharpe_ratio(
    sr: float, n: int, skew: float, kurtosis: float, sr_benchmark: float = 0.0,
) -> float:
    """PSR(SR*) = Φ((SR̂ − SR*)·√(n − 1) / √(1 − γ3·SR̂ + (γ4 − 1)/4·SR̂²)).

    The probability that the true per-period Sharpe ratio exceeds
    ``sr_benchmark`` given the observed SR̂ (Bailey & López de Prado 2012).
    """
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    if n < 2 or not denom > 0:
        return float("nan")
    z = (sr - sr_benchmark) * np.sqrt(n - 1.0) / np.sqrt(denom)
    return float(norm.cdf(z))


def min_track_record_length(
    sr: float, skew: float, kurtosis: float, sr_benchmark: float = 0.0, alpha: float = 0.05,
) -> float:
    """MinTRL: periods needed for PSR(SR*) ≥ 1 − alpha at the observed moments,
        1 + (1 − γ3·SR̂ + (γ4 − 1)/4·SR̂²)·(z_{1−α} / (SR̂ − SR*))².
    Infinite when SR̂ ≤ SR* (no track record can reject SR ≤ SR* then)."""
    if not sr > sr_benchmark:
        return float("inf")
    denom = 1.0 - skew * sr + (kurtosis - 1.0) / 4.0 * sr ** 2
    z = norm.ppf(1.0 - alpha)
    return float(1.0 + denom * (z / (sr - sr_benchmark)) ** 2)


def expected_max_sharpe(n_trials: float, var_trial_sr: float) -> float:
    """E[max of N IID N(0, V) Sharpe estimates] (Bailey & López de Prado 2014):
        √V·((1 − γ)·Φ⁻¹(1 − 1/N) + γ·Φ⁻¹(1 − 1/(N·e))),  γ = Euler-Mascheroni.
    The DSR benchmark SR*: what the best of N skill-less trials scores."""
    if n_trials < 1 or not var_trial_sr >= 0:
        return float("nan")
    if n_trials == 1:
        return 0.0
    n = float(n_trials)
    return float(np.sqrt(var_trial_sr) * ((1.0 - EULER_GAMMA) * norm.ppf(1.0 - 1.0 / n)
                                          + EULER_GAMMA * norm.ppf(1.0 - 1.0 / (n * np.e))))


def effective_trials(n_trials: int, mean_correlation: float) -> float:
    """N̂ = ρ̄ + (1 − ρ̄)·M: the average-correlation approximation of the
    independent-trial count of M trials whose returns have average pairwise
    correlation ρ̄ (1 for perfectly correlated trials, M for uncorrelated
    ones); ρ̄ is clipped to [0, 1]."""
    rho = float(np.clip(mean_correlation, 0.0, 1.0))
    return float(rho + (1.0 - rho) * n_trials)


@dataclass(frozen=True)
class SharpeInference:
    """Closed-form Sharpe inference for one return series (per-period units,
    annual figures alongside)."""
    n: int
    sr: float
    sr_annual: float
    skew: float
    kurtosis: float
    se: float
    se_annual: float
    psr_gt_0: float
    min_trl_periods: float
    min_trl_years: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def sharpe_inference(r: Sequence[float], alpha: float = 0.05,
                     periods: int = TRADING_DAYS) -> SharpeInference:
    """SR̂, its IID non-normal standard error, PSR(SR > 0) and MinTRL."""
    r = np.asarray(r, dtype=float)
    mom = sample_moments(r)
    sr = sharpe(r)
    var = sharpe_variance(sr, mom["n"], mom["skew"], mom["kurtosis"])
    se = float(np.sqrt(var)) if var >= 0 else float("nan")
    mtrl = min_track_record_length(sr, mom["skew"], mom["kurtosis"], 0.0, alpha)
    return SharpeInference(
        n=mom["n"], sr=sr, sr_annual=annualise(sr, periods),
        skew=mom["skew"], kurtosis=mom["kurtosis"],
        se=se, se_annual=annualise(se, periods),
        psr_gt_0=probabilistic_sharpe_ratio(sr, mom["n"], mom["skew"], mom["kurtosis"], 0.0),
        min_trl_periods=mtrl, min_trl_years=mtrl / periods,
    )


def deflated_sharpe_ratio(
    r: Sequence[float], n_trials: float, var_trial_sr: float,
) -> dict[str, float]:
    """DSR = PSR(SR̂ ≥ SR*) with SR* = expected_max_sharpe(N, V).

    ``var_trial_sr`` is the cross-sectional variance of the per-period Sharpe
    ratios of the trials (the paper's V[{SR_n}]); ``n_trials`` their
    independent-equivalent count. Returns the DSR with its inputs.
    """
    r = np.asarray(r, dtype=float)
    mom = sample_moments(r)
    sr = sharpe(r)
    sr0 = expected_max_sharpe(n_trials, var_trial_sr)
    return {
        "dsr": probabilistic_sharpe_ratio(sr, mom["n"], mom["skew"], mom["kurtosis"], sr0),
        "sr": sr, "sr_annual": annualise(sr),
        "sr0": sr0, "sr0_annual": annualise(sr0),
        "n_trials": float(n_trials), "var_trial_sr": float(var_trial_sr),
        "n": mom["n"], "skew": mom["skew"], "kurtosis": mom["kurtosis"],
    }


# ── HAC t-statistic of the mean ───────────────────────────────────────────────

def newey_west_variance(x: Sequence[float], lags: int) -> float:
    """Bartlett-kernel (Newey-West) long-run variance of x (demeaned)."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = x.size
    lrv = float(x @ x) / n
    for k in range(1, int(lags) + 1):
        if k >= n:
            break
        w = 1.0 - k / (lags + 1.0)
        lrv += 2.0 * w * float(x[k:] @ x[:-k]) / n
    return lrv


def newey_west_tstat(x: Sequence[float], lags: int | None = None) -> dict[str, float]:
    """t-statistic of the mean of x with a Newey-West standard error.

    lags default: floor(4·(n/100)^(2/9)) (Newey & West 1994 rule of thumb).
    """
    x = np.asarray(x, dtype=float)
    n = x.size
    if lags is None:
        lags = int(np.floor(4.0 * (n / 100.0) ** (2.0 / 9.0)))
    lrv = newey_west_variance(x, lags)
    se = float(np.sqrt(lrv / n)) if lrv > 0 else float("nan")
    mean = float(x.mean())
    t = mean / se if se and np.isfinite(se) else float("nan")
    return {"mean": mean, "se": se, "t": t, "lags": int(lags),
            "p_two_sided": float(2.0 * norm.sf(abs(t))) if np.isfinite(t) else float("nan")}
