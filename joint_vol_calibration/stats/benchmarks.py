"""
benchmarks.py — Cboe strategy indices as an external check on the β·VIX mark.

The backtest marks every straddle with Black-Scholes at σ_ATM = β·VIX-TS and
has no option prices inside 2018-2025. Cboe publishes daily levels of
systematic option-writing strategies computed from real SPX option prices:

    PUT    one-month ATM SPX put written monthly, cash-secured in T-bills
    WPUT   the same with one-week puts written weekly
    BXM    S&P 500 buy-write (long index, short one-month ATM call)
    BFLY   iron butterfly: short ATM straddle, long ~5% OTM wings, monthly
    CNDR   iron condor: short ~5% OTM strangle, long further OTM wings
    VXTH   S&P 500 with a VIX-call tail hedge

Two uses:
1. Co-movement (``strategy_benchmark_stats``): correlation, OLS beta, Jensen
   alpha and Sharpe comparisons of each backtest sleeve (and of an
   always-on short straddle on the same engine) against the indices' excess
   returns, by calendar year. A mark that is roughly right should make the
   short-straddle P&L co-move with BFLY/PUT/WPUT in stress years.
2. Implied β (``put_write_replication`` / ``implied_beta``): replicate PUT
   (and WPUT) with puts priced at β·VIX-TS and solve, per roll period and per
   year, for the β that reproduces the index's return. Its spread across
   years is direct evidence on the constant-β assumption.

Approximations (stated wherever the numbers are shown): the indices sell
their options at a VWAP (PUT: 11:30-12:00 ET on the roll date) and settle the
expiring monthly put at the special opening quotation; the replication sells
at the roll-date close, settles a monthly (AM) put at Cboe's published SET
value (the SPX exercise-settlement value; Yahoo's SPX open where SET has no
row) and a weekly put at the close, strikes
at the SPX level itself (no strike grid), and earns ^IRX on the collateral.
"""

from __future__ import annotations

import io
import itertools
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd
from scipy.optimize import brentq
from scipy.stats import norm

from joint_vol_calibration.stats.regression import ols_hac
from joint_vol_calibration.stats.sharpe import TRADING_DAYS

BENCHMARK_INDICES: tuple = ("PUT", "WPUT", "BXM", "BFLY", "CNDR", "VXTH")
CBOE_INDEX_URL = "https://cdn.cboe.com/api/global/us_indices/daily_prices/{index}_History.csv"


# ── levels ────────────────────────────────────────────────────────────────────

def parse_cboe_index_csv(text: str, name: str | None = None) -> pd.DataFrame:
    """A Cboe ``<INDEX>_History.csv`` (DATE + one value column, or OHLC) →
    DataFrame [date (YYYY-MM-DD), close], sorted, de-duplicated, positive."""
    df = pd.read_csv(io.StringIO(text))
    df.columns = [str(c).strip() for c in df.columns]
    cols = {c.lower(): c for c in df.columns}
    if "date" not in cols:
        raise ValueError(f"no DATE column: {df.columns.tolist()}")
    dcol = cols["date"]
    if "close" in cols:
        vcol = cols["close"]
    else:
        vals = [c for c in df.columns if c != dcol]
        if name is not None and name in df.columns:
            vcol = name
        elif len(vals) == 1:
            vcol = vals[0]
        else:
            raise ValueError(f"cannot identify the value column: {df.columns.tolist()}")
    out = pd.DataFrame({
        "date": pd.to_datetime(df[dcol], format="%m/%d/%Y", errors="coerce"),
        "close": pd.to_numeric(df[vcol], errors="coerce"),
    }).dropna()
    out = out[out["close"] > 0]
    out = out.drop_duplicates("date", keep="last").sort_values("date").reset_index(drop=True)
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out


def load_levels(directory: Path, names: Iterable[str] = BENCHMARK_INDICES) -> pd.DataFrame:
    """Wide date-indexed frame of the stored index closes (one column per index)."""
    frames = {}
    for nm in names:
        p = Path(directory) / f"{nm}.csv"
        if p.exists():
            d = pd.read_csv(p)
            frames[nm] = pd.Series(d["close"].to_numpy(float), index=pd.to_datetime(d["date"]), name=nm)
    if not frames:
        return pd.DataFrame()
    return pd.DataFrame(frames).sort_index()


# ── co-movement ───────────────────────────────────────────────────────────────

def _weekly(r: pd.Series) -> pd.Series:
    return (1.0 + r).resample("W-FRI").prod() - 1.0


def _resid(y: np.ndarray, x: np.ndarray) -> np.ndarray:
    X = np.column_stack([np.ones(len(x)), x])
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    return y - X @ b


def _sharpe_ann(r: np.ndarray) -> float:
    sd = r.std(ddof=1)
    return float(r.mean() / sd * np.sqrt(TRADING_DAYS)) if sd > 0 else float("nan")


def pair_stats(s: pd.Series, b: pd.Series, spx: pd.Series | None = None) -> dict[str, float]:
    """Co-movement of sleeve excess returns s with benchmark excess returns b
    (same index). Weekly figures compound the daily returns into W-FRI weeks."""
    df = pd.concat([s.rename("s"), b.rename("b")] + ([spx.rename("m")] if spx is not None else []),
                   axis=1).dropna()
    n = len(df)
    out: dict[str, float] = {"n_days": int(n)}
    if n < 20 or df["s"].std() == 0 or df["b"].std() == 0:
        out.update({k: float("nan") for k in ("corr_daily", "corr_weekly", "beta", "alpha_ann",
                                               "alpha_t", "sharpe_sleeve", "sharpe_bench",
                                               "sharpe_diff", "partial_corr_ex_spx", "r2")})
        return out
    sv, bv = df["s"].to_numpy(), df["b"].to_numpy()
    reg = ols_hac(sv, bv)
    wk = pd.concat([_weekly(df["s"]), _weekly(df["b"])], axis=1).dropna()
    out.update({
        "corr_daily": float(np.corrcoef(sv, bv)[0, 1]),
        "corr_weekly": (float(np.corrcoef(wk.iloc[:, 0], wk.iloc[:, 1])[0, 1]) if len(wk) > 3
                        else float("nan")),
        "beta": float(reg["coef"][1]),
        "alpha_ann": float(reg["coef"][0] * TRADING_DAYS),
        "alpha_t": float(reg["t"][0]),
        "r2": float(reg["r2"]),
        "sharpe_sleeve": _sharpe_ann(sv),
        "sharpe_bench": _sharpe_ann(bv),
    })
    out["sharpe_diff"] = out["sharpe_sleeve"] - out["sharpe_bench"]
    if spx is not None:
        m = df["m"].to_numpy()
        out["partial_corr_ex_spx"] = float(np.corrcoef(_resid(sv, m), _resid(bv, m))[0, 1])
    else:
        out["partial_corr_ex_spx"] = float("nan")
    return out


def benchmark_excess_returns(levels: pd.DataFrame, dates: pd.DatetimeIndex,
                             cash_daily: pd.Series) -> pd.DataFrame:
    """Index returns over the backtest's own session intervals minus the T-bill
    accrual of the same interval. A session missing from an index's file makes
    that day's return NaN rather than spanning two sessions."""
    lv = levels.reindex(dates)
    r = lv / lv.shift(1) - 1.0
    return r.sub(cash_daily.reindex(dates), axis=0)


def strategy_benchmark_stats(
    sleeve_returns: pd.DataFrame,
    bench_excess: pd.DataFrame,
    spx_returns: pd.Series | None = None,
    years: Sequence[int] | None = None,
) -> pd.DataFrame:
    """pair_stats for every (sleeve, benchmark) over the whole window and each
    calendar year."""
    idx = cast(pd.DatetimeIndex, sleeve_returns.index)
    if years is None:
        years = sorted(set(idx.year))
    rows = []
    periods = [("all", np.ones(len(idx), dtype=bool))] + [(str(y), idx.year == y) for y in years]
    for sname in sleeve_returns.columns:
        for bname in bench_excess.columns:
            for label, mask in periods:
                st = pair_stats(sleeve_returns[sname][mask], bench_excess[bname][mask],
                                None if spx_returns is None else spx_returns.reindex(idx)[mask])
                rows.append({"sleeve": sname, "benchmark": bname, "period": label, **st})
    return pd.DataFrame(rows)


# ── PUT / WPUT replication and the implied β ──────────────────────────────────

def bs_put(S, K, T, r, q, sigma):
    """Black-Scholes European put (vectorised; intrinsic at T ≤ 0)."""
    S, K, T, r, q, sigma = (np.asarray(a, dtype=float) for a in (S, K, T, r, q, sigma))
    T_ = np.maximum(T, 1e-12)
    sq = sigma * np.sqrt(T_)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma ** 2) * T_) / sq
    d2 = d1 - sq
    val = K * np.exp(-r * T_) * norm.cdf(-d2) - S * np.exp(-q * T_) * norm.cdf(-d1)
    return np.where(T > 0, val, np.maximum(K - S, 0.0))


def monthly_roll_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The session on or before each month's third Friday (the SPX monthly
    expiry, moved to Thursday when Friday is an exchange holiday)."""
    sessions = pd.DatetimeIndex(sessions).sort_values()
    out = []
    for (y, m), _ in pd.Series(0, index=sessions).groupby([sessions.year, sessions.month]):
        first = pd.Timestamp(year=int(y), month=int(m), day=1)  # type: ignore[call-overload]  # int keys
        fri = first + pd.Timedelta(days=(4 - first.weekday()) % 7) + pd.Timedelta(days=14)
        on_before = sessions[sessions <= fri]
        if len(on_before) and on_before[-1].month == m:
            out.append(on_before[-1])
    return pd.DatetimeIndex(out)


def weekly_roll_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """The last session of each W-FRI week (Friday, or Thursday on a holiday)."""
    s = pd.Series(sessions, index=sessions)
    weeks = cast(pd.DatetimeIndex, s.index).to_period("W-FRI")
    return pd.DatetimeIndex(s.groupby(weeks).max().to_numpy())


def put_write_periods(
    roll_dates: pd.DatetimeIndex,
    spx_close: pd.Series,
    settle_prices: pd.Series,
    vix_rows: pd.DataFrame,
    rates: pd.Series,
    q: float,
    expiry_hour_et: float = 9.5,
    settle_fallback: pd.Series | None = None,
) -> pd.DataFrame:
    """One row per roll period [d_i, d_{i+1}] with what the replication needs.

    Strike K = SPX close at d_i; T to the expiry at d_{i+1} at
    ``expiry_hour_et`` (9.5 = the 09:30 ET special opening quotation of an
    AM-settled monthly, 16 = the close for a PM-settled weekly); the
    settlement price at d_{i+1} from ``settle_prices`` (Cboe's SET series for
    AM, the close for PM), else ``settle_fallback`` (recorded in
    settle_source); the T-bill growth of the collateral over the period
    (continuously compounded ^IRX, each print earning until the next, actual
    days / 365); and the rate at d_i.
    """
    rd = pd.DatetimeIndex(roll_dates)
    rows = []
    rate = rates.sort_index()
    for a, b in itertools.pairwise(rd):
        if a not in spx_close.index or a not in vix_rows.index:
            continue
        S0 = float(spx_close.loc[a])
        ST, src = float(settle_prices.get(b, np.nan)), "primary"
        if not np.isfinite(ST) and settle_fallback is not None:
            ST, src = float(settle_fallback.get(b, np.nan)), "fallback"
        if not (np.isfinite(S0) and np.isfinite(ST) and S0 > 0 and ST > 0):
            continue
        days = (b - a).days
        T = (days - (16.0 - float(expiry_hour_et)) / 24.0) / 365.0
        seg = rate[(rate.index >= a) & (rate.index < b)]
        r0 = float(rate[rate.index <= a].iloc[-1]) if (rate.index <= a).any() else float("nan")
        pts = [*seg.index, b]
        g = 0.0
        for k in range(len(seg)):
            g += float(seg.iloc[k]) * (pts[k + 1] - pts[k]).days / 365.0
        rows.append({"roll": a, "expiry": b, "K": S0, "S_settle": ST, "settle_source": src,
                     "T": T, "r": r0, "q": float(q), "growth": float(np.exp(g))})
    return pd.DataFrame(rows)


def period_returns(periods: pd.DataFrame, sigma: np.ndarray) -> np.ndarray:
    """Cash-secured put-write return per period at put vols ``sigma``:
        R = (1 + P0/K)·G − max(K − S_T, 0)/K − 1
    (N = M/K puts per unit of collateral M; premium invested at T-bills)."""
    K = periods["K"].to_numpy(float)
    P0 = bs_put(K, K, periods["T"].to_numpy(float), periods["r"].to_numpy(float),
                periods["q"].to_numpy(float), sigma)
    pay = np.maximum(K - periods["S_settle"].to_numpy(float), 0.0)
    return (1.0 + P0 / K) * periods["growth"].to_numpy(float) - pay / K - 1.0


def implied_beta(
    periods: pd.DataFrame,
    vix_tenor_vol: np.ndarray,
    index_return: np.ndarray,
    groups: Mapping[str, np.ndarray] | None = None,
    bracket: tuple = (0.05, 3.0),
) -> dict:
    """β with σ = β·vix_tenor_vol that reproduces the index's return.

    Per period: R_rep(β) = R_index (R_rep is increasing in β through the
    premium). Per group (e.g. calendar year) and overall: the compounded
    return. ``vix_tenor_vol`` is VIX-TS(T) at each roll (decimal). Periods
    whose target lies outside the bracket give NaN.
    """
    v = np.asarray(vix_tenor_vol, dtype=float)
    R = np.asarray(index_return, dtype=float)

    def _solve(mask: np.ndarray) -> float:
        target = float(np.prod(1.0 + R[mask]) - 1.0)
        sub = periods[mask]

        def f(b):
            return float(np.prod(1.0 + period_returns(sub, b * v[mask])) - 1.0) - target
        lo, hi = bracket
        flo, fhi = f(lo), f(hi)
        if not (np.isfinite(flo) and np.isfinite(fhi)) or flo * fhi > 0:
            return float("nan")
        return float(brentq(f, lo, hi, xtol=1e-10))

    n = len(periods)
    per = np.array([_solve(np.arange(n) == i) for i in range(n)])
    out: dict[str, Any] = {"per_period": per, "overall": _solve(np.ones(n, dtype=bool)), "groups": {}}
    for g, m in (groups or {}).items():
        mask = np.asarray(m, dtype=bool)
        out["groups"][g] = _solve(mask) if mask.any() else float("nan")
    return out


def replication_daily(
    periods: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    spx_close: pd.Series,
    vix_tenor_vol_fn,
    rates: pd.Series,
    beta: float,
    q: float,
    expiry_hour_et: float = 9.5,
) -> pd.Series:
    """Daily level of the β·VIX put-write replication (starts at 1).

    Between rolls the account is collateral·T-bill growth − N·P_t with P_t
    the put at σ = β·vix_tenor_vol_fn(date, T_rem); at each roll the put
    settles and a new one is sold at the close. ``vix_tenor_vol_fn(date, T)``
    returns VIX-TS(T) in decimal on that session.
    """
    sessions = pd.DatetimeIndex(sessions)
    rate = rates.sort_index()
    level: dict[Any, float] = {}
    M = 1.0
    for _, p in periods.iterrows():
        a, b, K = p["roll"], p["expiry"], float(p["K"])
        N = M / K
        T0 = float(p["T"])
        P0 = float(bs_put(K, K, T0, p["r"], q, beta * vix_tenor_vol_fn(a, T0)))
        cash = M + N * P0
        level.setdefault(a, M)
        prev = a
        for d in sessions[(sessions > a) & (sessions <= b)]:
            rp = rate[rate.index <= prev]
            r_prev = float(rp.iloc[-1]) if len(rp) else 0.0
            cash *= float(np.exp(r_prev * (d - prev).days / 365.0))
            prev = d
            if d == b:
                M = cash - N * max(K - float(p["S_settle"]), 0.0)
                level[d] = M
            else:
                T_rem = ((b - d).days - (16.0 - expiry_hour_et) / 24.0) / 365.0
                rd = rate[rate.index <= d]
                r_d = float(rd.iloc[-1]) if len(rd) else 0.0
                Pt = float(bs_put(float(spx_close.loc[d]), K, T_rem, r_d, q,
                                  beta * vix_tenor_vol_fn(d, T_rem)))
                level[d] = cash - N * Pt
    return pd.Series(level).sort_index()
