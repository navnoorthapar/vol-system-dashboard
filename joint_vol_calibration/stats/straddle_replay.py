"""
straddle_replay.py — the backtest's straddle P&L, replayed fast for null tests.

backtest_engine._simulate_straddle_pnl prices a delta-hedged 30-day ATM
straddle on every session it is held: ~0.5-0.8 s per sleeve on 2018-2025, so
thousands of null paths per sleeve would take hours. Every price it computes
depends only on the ENTRY session (the strike is that day's close) and the
number of sessions held, never on the position path or the sizing, so
``StraddleReplay`` prices each (entry session, holding day) pair once, by
calling the engine's own helpers (_atm_sigma, _bs_straddle_value,
_bs_straddle_greeks, _straddle_vega) with the engine's own arguments, and then
replays any position/size series with scalar arithmetic in the engine's
operation order (contract counts, margin cap, entry/exit costs, hedge
slippage, financing, expiry settlement and roll, the sign-flip guard).

The result is the engine's P&L to the last bit (tests assert exact equality
against _simulate_straddle_pnl on synthetic markets; inference_report checks
it on every sleeve of the canonical run before using it). It supports the
default (corrected) C22Corrections only and a market in which every session
has an SPX close and a VIX row; anything else raises, so a silent divergence
from the engine is not possible.
"""

from __future__ import annotations

from typing import Optional, Sequence, Union

import numpy as np
import pandas as pd

from joint_vol_calibration.backtest import backtest_engine as bte


class StraddleReplay:
    """Exact, precomputed replay of backtest_engine._simulate_straddle_pnl.

    Parameters mirror _simulate_straddle_pnl's market arguments; ``dates``
    is the position index (the backtest sessions).
    """

    def __init__(
        self,
        dates: Sequence,
        spx_close: pd.Series,
        vix_wide_idx: pd.DataFrame,
        r: "float | pd.Series",
        q: float = bte.Q_BACKTEST,
        atm_to_vix_ratio: float = bte.ATM_TO_VIX_RATIO,
        tenor_days: int = bte.STRADDLE_TENOR_DAYS,
        corrections: Optional[bte.C22Corrections] = None,
    ):
        c = corrections or bte.C22Corrections()
        if c != bte.C22Corrections():
            raise NotImplementedError("StraddleReplay replays the default (corrected) engine only")
        self.dates = pd.DatetimeIndex(dates)
        if not self.dates.is_unique:
            raise ValueError("dates must be unique")
        if not vix_wide_idx.index.is_unique:
            raise ValueError("the VIX frame index must be unique")
        missing = [d for d in self.dates if d not in spx_close.index or d not in vix_wide_idx.index]
        if missing:
            raise NotImplementedError(
                f"{len(missing)} sessions lack an SPX close or VIX row (first {missing[0].date()}); "
                "the engine skips those and the replay does not model the skip")
        self.q = float(q)
        self.beta = float(atm_to_vix_ratio)
        self.T_entry = tenor_days / 365.0
        self.n_expiry = max(1, int(np.ceil(self.T_entry * bte.TRADING_DAYS - 1e-9)))
        n = len(self.dates)
        self.S = np.array([float(spx_close.loc[d]) for d in self.dates])
        self.rates = bte._resolve_rates(r, self.dates, ffill=c.rate_ffill)
        rows = [vix_wide_idx.loc[d] for d in self.dates]

        # Entry quantities on every session (an entry or an expiry roll).
        self.sig0 = np.empty(n)
        self.V0 = np.empty(n)
        self.vega0 = np.empty(n)
        self.delta0 = np.empty(n)
        for i in range(n):
            S, r_t = self.S[i], float(self.rates[i])
            sigma = bte._atm_sigma(rows[i], self.T_entry, self.beta)
            self.sig0[i] = sigma
            self.V0[i] = max(float(bte._bs_straddle_value(S, S, self.T_entry, r_t, self.q, sigma)), 1e-4)
            self.vega0[i] = bte._straddle_vega(S, S, self.T_entry, r_t, self.q, sigma)
            self.delta0[i] = float(bte._bs_straddle_greeks(S, S, self.T_entry, r_t, self.q, sigma)[0])

        # Holding-day quantities: entry session e, j sessions later (j = 1..n_expiry).
        J = self.n_expiry
        self.V = np.full((n, J + 1), np.nan)
        self.D = np.full((n, J + 1), np.nan)
        self.vega_exit = np.full((n, J + 1), np.nan)
        self.V[:, 0] = self.V0
        self.D[:, 0] = self.delta0
        for e in range(n):
            K = self.S[e]
            for j in range(1, J + 1):
                t = e + j
                if t >= n:
                    break
                T_raw = self.T_entry - j / bte.TRADING_DAYS
                expired = j >= J
                T_rem = max(T_raw, 0.0)
                S, r_t = self.S[t], float(self.rates[t])
                if expired:
                    self.V[e, j] = abs(S - K)
                    self.D[e, j] = 0.0
                else:
                    sigma = bte._atm_sigma(rows[t], T_rem, self.beta)
                    self.V[e, j] = float(bte._bs_straddle_value(S, K, T_rem, r_t, self.q, sigma))
                    self.D[e, j] = float(bte._bs_straddle_greeks(S, K, T_rem, r_t, self.q, sigma)[0])
                    self.vega_exit[e, j] = bte._straddle_vega(S, K, T_rem, r_t, self.q, sigma)

    # ── replay ─────────────────────────────────────────────────────────────────

    def simulate(
        self,
        position: Union[Sequence[float], pd.Series],
        kelly: Union[Sequence[float], pd.Series],
        nav_base: Union[float, Sequence[float]] = bte.INITIAL_CAPITAL,
    ) -> np.ndarray:
        """Daily P&L of the straddle sleeve for this position/size series.

        ``position``'s sign is the straddle direction (+1 long, −1 short, 0
        flat); ``kelly`` the size fraction; ``nav_base`` the sizing capital
        (the sleeve's running P&L is added to it, as in the engine).
        """
        pos_arr = np.asarray(position, dtype=float)
        k_arr = np.asarray(kelly, dtype=float)
        n = len(self.dates)
        if pos_arr.shape != (n,) or k_arr.shape != (n,):
            raise ValueError("position and kelly must have one value per session")
        base = np.broadcast_to(np.asarray(nav_base, dtype=float), (n,))
        S_arr, rates, q = self.S, self.rates, self.q
        J = self.n_expiry
        MULT = bte.CONTRACT_MULT
        TD = bte.TRADING_DAYS
        hedge_cost, one_way = bte._hedge_cost, bte._one_way_cost
        out = np.zeros(n)
        cum = 0.0

        # open-position state
        have = False
        dirn = 0
        e = 0            # entry session index
        held = 0         # sessions held so far (days_biz)
        nc = 0           # contracts
        V = dl = Sp = rp = 0.0

        for i in range(n):
            pos = int(np.sign(pos_arr[i]))
            k = float(k_arr[i])
            nav = float(base[i]) + cum
            if have and pos != 0 and pos != dirn:
                pos = 0
            S = float(S_arr[i])
            r_t = float(rates[i])
            day = 0.0
            if have:
                j = held + 1
                expired = j >= J
                held = j
                mult = nc * MULT
                V_t = float(self.V[e, j])
                delta_t = float(self.D[e, j])
                raw = dirn * (V_t - V - dl * (S - Sp)) * mult
                fin = (dirn * (rp * (dl * Sp - V) - q * dl * Sp) / TD * mult)
                closing = expired or pos == 0
                if closing:
                    slip = hedge_cost(dl, S, nc)
                else:
                    slip = hedge_cost(delta_t - dl, S, nc)
                day += raw + fin - slip
                V, dl, Sp, rp = V_t, delta_t, S, r_t
                if closing:
                    exit_cost = 0.0 if expired else one_way(float(self.vega_exit[e, j]), nc)
                    day -= exit_cost
                    have = False
                    if expired and pos != 0:
                        ok, cost = self._open(pos, i, k, nav)
                        if ok:
                            have, dirn, e, held, nc, V, dl, Sp, rp = ok
                        day -= cost
            elif pos != 0:
                ok, cost = self._open(pos, i, k, nav)
                if ok:
                    have, dirn, e, held, nc, V, dl, Sp, rp = ok
                day -= cost
            out[i] += day
            cum += day
        return out

    def _open(self, direction: int, i: int, k: float, nav: float):
        """Engine's _open at session i: (state tuple | None, cash cost)."""
        if nav <= 0:
            return None, 0.0
        S = float(self.S[i])
        r_t = float(self.rates[i])
        V0 = float(self.V0[i])
        n = max(1, int(k * nav / (V0 * bte.CONTRACT_MULT)))
        if direction < 0:
            n_margin_cap = int(nav / (bte.MARGIN_RATIO * S * bte.CONTRACT_MULT))
            if n_margin_cap < 1:
                return None, 0.0
            n = min(n, n_margin_cap)
        entry_cost = bte._one_way_cost(float(self.vega0[i]), n)
        delta0 = float(self.delta0[i])
        hedge0 = bte._hedge_cost(delta0, S, n)
        state = (True, int(direction), i, 0, n, V0, delta0, S, r_t)
        return state, entry_cost + hedge0
