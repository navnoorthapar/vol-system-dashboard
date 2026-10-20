"""
nulls.py — exposure-matched timing nulls for every sleeve of the backtest.

Question asked of each sleeve: does WHEN it holds straddles matter, given HOW
MUCH and in which direction it holds them? The null keeps the sleeve's own
position and size series (the number, length, direction and sizing of its
trades, its time in the market and the clustering of its trades) and only
moves it in time against the fixed market, then re-prices it on the backtest
engine's own straddle P&L (``StraddleReplay``, bit-exact).

Two families of null paths:

circular shift   position_t → position_{(t+τ) mod n}, every admissible shift
                 |τ| ≥ min_shift, plus the time-reversed series shifted the
                 same way (``circular_shift_paths``). The forward shifts are
                 an exhaustive, exact enumeration: with n = 1,816 sessions
                 there are only n − 1 distinct shifts, so a ≥ 2,000-path null
                 needs the reversed set as well. Reversal keeps every block
                 and gap length (a stationary position process has the same
                 autocovariance reversed).
trade shuffle    the sleeve's trades (maximal runs of the same direction) and
                 the flat gaps between them are put in a random order
                 (``trade_shuffle``): the same trades and gaps, with clustering
                 broken; any number of seeded paths.

A shift splits the trade that crosses the wrap point in two, so a path may
have one entry more than the real series; the per-path entry count is
recorded.

For the three portfolio sleeves the published P&L comes after the engine's
pass-2 sizing overlay (correlation netting × trailing net-Sharpe scale), so
the portfolio null shifts S1C, S3 and S4 by the same τ (their co-timing is
part of the exposure) and re-runs that overlay on each path
(``PortfolioNull``); the real path through the same code is the published
portfolio to the last bit (asserted by inference_report).

The split this gives, per sleeve:
    real P&L = E_null[P&L]               (the exposure's P&L at random timing:
                                          the short- or long-vol premium "beta")
             + (real − E_null[P&L])      (timing alpha)
with p = (1 + #{null ≥ real}) / (B + 1) for the timing alpha.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.stats.sharpe import TRADING_DAYS, returns_from_pnl, sharpe

# Shifts shorter than a straddle's life mostly re-trade the real timing.
DEFAULT_MIN_SHIFT: int = 21


@dataclass(frozen=True)
class NullPath:
    """One null transformation of a position series."""
    kind: str            # 'real' | 'circular' | 'shuffle'
    path_id: int
    shift: int = 0
    reverse: bool = False
    seed: int | None = None


def circular_shift_paths(n: int, min_shift: int = DEFAULT_MIN_SHIFT,
                         include_reversed: bool = True) -> list[NullPath]:
    """Every shift τ in [min_shift, n − min_shift], forward then reversed."""
    if not 1 <= min_shift <= n // 2:
        raise ValueError(f"min_shift must be in [1, n/2], got {min_shift}")
    shifts = range(int(min_shift), int(n - min_shift) + 1)
    out = [NullPath("circular", i, s, False) for i, s in enumerate(shifts)]
    if include_reversed:
        base = len(out)
        out += [NullPath("circular", base + i, s, True) for i, s in enumerate(shifts)]
    return out


def shuffle_paths(n_paths: int, seed: int = 42) -> list[NullPath]:
    """``n_paths`` trade-shuffle paths with seeds seed, seed+1, ..."""
    return [NullPath("shuffle", i, 0, False, int(seed + i)) for i in range(int(n_paths))]


def shift_series(x: np.ndarray, shift: int, reverse: bool = False) -> np.ndarray:
    """x reversed (optionally) then circularly shifted: y_t = x'_{(t+shift) mod n}."""
    x = np.asarray(x)
    if reverse:
        x = x[::-1]
    return np.roll(x, -int(shift))


def _runs(sign: np.ndarray) -> list[tuple[int, int]]:
    """Maximal runs of equal sign as (start, stop) half-open intervals."""
    if sign.size == 0:
        return []
    cut = np.flatnonzero(np.diff(sign)) + 1
    starts = np.r_[0, cut]
    stops = np.r_[cut, sign.size]
    return list(zip(starts.tolist(), stops.tolist()))


def trade_shuffle(arrays: Sequence[np.ndarray], key: np.ndarray,
                  rng: np.random.Generator) -> list[np.ndarray]:
    """Put the trades and the flat gaps of sign(key) in a random order.

    The series is cut into maximal runs of equal sign; the sequence of run
    TYPES (trade / gap) is kept and the trade slots are refilled with a random
    permutation of the trades, the gap slots with a random permutation of the
    gaps. Trades therefore stay separated by gaps exactly as often as in the
    real series (an unconstrained shuffle would join neighbouring trades of
    the same direction into one, roughly halving the entries). Every array in
    ``arrays`` is cut at the same boundaries and reassembled in the same
    order, so a position series and its size series stay aligned; runs keep
    their internal order.
    """
    sign = np.sign(np.asarray(key, dtype=float)).astype(int)
    runs = _runs(sign)
    is_trade = [sign[a] != 0 for a, _ in runs]
    trades = [k for k, t in enumerate(is_trade) if t]
    gaps = [k for k, t in enumerate(is_trade) if not t]
    t_perm = iter(rng.permutation(trades).tolist() if trades else [])
    g_perm = iter(rng.permutation(gaps).tolist() if gaps else [])
    order = [next(t_perm) if t else next(g_perm) for t in is_trade]
    out = []
    for arr in arrays:
        a = np.asarray(arr)
        out.append(np.concatenate([a[runs[k][0]:runs[k][1]] for k in order]))
    return out


def holm(pvalues: dict[str, float]) -> dict[str, float]:
    """Holm (1979) step-down adjusted p-values (family-wise error control)."""
    items = sorted(pvalues.items(), key=lambda kv: kv[1])
    m = len(items)
    out, running = {}, 0.0
    for i, (k, p) in enumerate(items):
        running = max(running, min(1.0, (m - i) * float(p)))
        out[k] = running
    return out


def apply_path(path: NullPath, arrays: Sequence[np.ndarray], key: np.ndarray) -> list[np.ndarray]:
    """The null transformation ``path`` applied to aligned arrays."""
    if path.kind == "real":
        return [np.asarray(a) for a in arrays]
    if path.kind == "circular":
        return [shift_series(a, path.shift, path.reverse) for a in arrays]
    if path.kind == "shuffle":
        return trade_shuffle(arrays, key, np.random.default_rng(path.seed))
    raise ValueError(f"unknown null kind {path.kind!r}")


def n_entries(position: np.ndarray) -> int:
    """Flat → on transitions (and direction flips) of a position series."""
    s = np.sign(np.asarray(position, dtype=float)).astype(int)
    prev = np.r_[0, s[:-1]]
    return int(((s != 0) & (s != prev)).sum())


def pvalue_greater(real: float, null: np.ndarray) -> float:
    """One-sided (1 + #{null ≥ real}) / (B + 1)."""
    null = np.asarray(null, dtype=float)
    null = null[np.isfinite(null)]
    return float((1 + (null >= real).sum()) / (null.size + 1))


def summarise_null(real: float, null: np.ndarray) -> dict[str, float]:
    """Real value, null moments/quantiles, p-value and the timing component."""
    null = np.asarray(null, dtype=float)
    fin = null[np.isfinite(null)]
    return {
        "real": float(real),
        "null_mean": float(fin.mean()), "null_sd": float(fin.std(ddof=1)) if fin.size > 1 else float("nan"),
        "null_q05": float(np.quantile(fin, 0.05)), "null_q50": float(np.quantile(fin, 0.50)),
        "null_q95": float(np.quantile(fin, 0.95)),
        "null_frac_positive": float((fin > 0).mean()),
        "p_value": pvalue_greater(real, fin),
        "p_value_binomial_se": float(np.sqrt(pvalue_greater(real, fin) * (1 - pvalue_greater(real, fin))
                                             / (fin.size + 1))),
        "timing_component": float(real - fin.mean()),
        "n_paths": int(fin.size),
    }


# ── Sleeves and the portfolio on the canonical engine ────────────────────────

SLEEVE_SPECS: dict[str, dict[str, str]] = {
    # name: signals_df columns; S3's straddle direction is engine.s3_leg_direction × position
    "s1": {"position": "s1_position", "kelly": "s1_kelly"},
    "s2": {"position": "s2_position", "kelly": "s2_kelly"},
    "s1c": {"position": "s1c_position", "kelly": "s1c_kelly"},
    "s3": {"position": "s3_position", "kelly": "s3_kelly"},
    "s4": {"position": "s4_position", "kelly": "s4_kelly"},
}
PORTFOLIO = tuple(bte.PORTFOLIO_SIGNALS)       # ('s1c', 's3', 's4')


class PortfolioNull:
    """Null paths for every sleeve and the portfolio of a finished engine run.

    ``engine`` must have run (BacktestEngine.run); ``replay`` is a
    StraddleReplay on the same market and β.
    """

    def __init__(self, engine: bte.BacktestEngine, replay) -> None:
        if engine.signals_df is None or not engine._market:
            raise RuntimeError("run the engine first")
        self.engine = engine
        self.replay = replay
        sig = engine.signals_df
        self.capital = float(engine.initial_capital)
        self.nav_base = engine._market["nav_base"].to_numpy(dtype=float)
        self.dirn = {s: (engine.s3_leg_direction if s == "s3" else 1) for s in SLEEVE_SPECS}
        self.pos = {s: sig[c["position"]].to_numpy(dtype=float) for s, c in SLEEVE_SPECS.items()}
        self.kel = {s: sig[c["kelly"]].to_numpy(dtype=float) for s, c in SLEEVE_SPECS.items()}
        self.index = sig.index

    def _straddle(self, s: str, pos: np.ndarray, kel: np.ndarray) -> np.ndarray:
        return self.replay.simulate(self.dirn[s] * pos, kel, self.nav_base)

    def overlay(self, p1: dict[str, np.ndarray], pos: dict[str, np.ndarray],
                kel: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """The engine's pass 2: size multipliers from pass-1 P&L, shifted one
        session, then the three portfolio sleeves re-simulated."""
        pnl_mat = pd.DataFrame({s: p1[s] for s in PORTFOLIO}, index=self.index).fillna(0.0)
        m_corr = bte._corr_netting_multipliers(pnl_mat)
        out = {}
        for s in PORTFOLIO:
            k_mult = (m_corr[s].clip(0.5, 1.0) * bte._net_kelly_scale(pnl_mat[s])).shift(1).fillna(1.0)
            out[s] = self._straddle(s, pos[s], kel[s] * k_mult.to_numpy())
        return out

    def run_path(self, path: NullPath) -> dict[str, np.ndarray]:
        """Daily P&L of every sleeve and the portfolio on one null path.

        Every sleeve is transformed by the same path. Trade shuffles permute
        each sleeve's own runs (one seeded generator per sleeve), so the
        portfolio's co-timing is kept only by the circular paths.
        """
        pos, kel = {}, {}
        for k, s in enumerate(SLEEVE_SPECS):
            if path.kind == "shuffle":
                p = NullPath("shuffle", path.path_id, 0, False, int(path.seed) * 16 + k)
            else:
                p = path
            pos[s], kel[s] = apply_path(p, [self.pos[s], self.kel[s]], self.pos[s])
        p1 = {s: self._straddle(s, pos[s], kel[s]) for s in SLEEVE_SPECS}
        p2 = self.overlay(p1, pos, kel)
        out = {"s1": p1["s1"], "s2": p1["s2"], **p2}
        out["portfolio"] = (p2["s1c"] + p2["s3"] + p2["s4"]) / 3.0
        out["_entries"] = {s: n_entries(pos[s]) for s in SLEEVE_SPECS}
        return out

    def statistics(self, pnl: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        out = {}
        for s in ("s1", "s2", "s1c", "s3", "s4", "portfolio"):
            p = pnl[s]
            r = returns_from_pnl(p, self.capital)
            out[s] = {"total_pnl": float(p.sum()), "sharpe": float(sharpe(r) * np.sqrt(TRADING_DAYS))}
        return out

    def run(self, paths: Iterable[NullPath]) -> pd.DataFrame:
        """Long-form table: one row per (path, sleeve)."""
        rows = []
        for path in paths:
            res = self.run_path(path)
            stats = self.statistics(res)
            for s, st in stats.items():
                ent = (res["_entries"].get(s) if s != "portfolio"
                       else int(sum(res["_entries"][x] for x in PORTFOLIO)))
                rows.append({"kind": path.kind, "path_id": path.path_id, "shift": path.shift,
                             "reverse": path.reverse, "seed": -1 if path.seed is None else path.seed,
                             "sleeve": s, "total_pnl": st["total_pnl"], "sharpe": st["sharpe"],
                             "n_entries": ent})
        return pd.DataFrame(rows)


def iter_chunks(seq: Sequence, size: int) -> Iterator[Sequence]:
    for i in range(0, len(seq), size):
        yield seq[i:i + size]
