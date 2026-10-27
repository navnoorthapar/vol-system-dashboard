"""
scenario_grid.py — spot × vol scenario P&L of a reference book on the headline fit.

The Risk page's scenario panel (dashboard/ARTIFACTS.md §5). One reference
book, revalued in full (Black-76 on the shocked forward and the shocked
implied vol; no Greek expansion) on a grid of instantaneous shocks:

    book        long one 30-day straddle struck at the 30-day forward, on one
                unit of the index, delta-hedged at inception in the index at
                its Black-Scholes (sticky-strike) delta: the instrument every
                backtest sleeve trades (a 30-day ATM straddle, delta-hedged)
    smile       the headline Heston fit's implied vols at 30 days (Fourier OTM
                prices inverted with implied_vol_black), with its carry
                (risk_monitor.calibration_carry: the fit's parity forwards and
                the Treasury discount curve)
    spot shock  S1 = S0·(1 + s/100); F1 = F0·S1/S0 (carry unchanged)
    vol shock   a parallel shift of the whole smile by v vol points
    dynamics    how the smile moves with spot:
                  sticky_strike     σ1(K) = σ0(K) + v
                  sticky_moneyness  σ1(K) = σ0(K·F0/F1) + v (the smile is a
                                    function of ln(K/F); at v = 0 this is the
                                    Heston revaluation at S1 with v0 held,
                                    since the model is homogeneous in (S, K))
    P&L         V(F1, σ1(K)) − V(F0, σ0(K)) − Δ0·(S1 − S0), index points per
                straddle on one unit of the index (×100 for one SPX contract)

Time does not pass (no theta) and rates and dividends are held. The shocked
vols are floored at VOL_FLOOR; how many cells hit the floor is recorded.

Output: data_store/greeks/scenario_grid.parquet, one row per (spot_shock_pct,
vol_shock_vp, dynamics), with the provenance JSON in the parquet key-value
metadata under b"scenario_provenance" (book, as_of_date, units, the source
calibration and every input). Written atomically (temporary file + rename).

    python -m joint_vol_calibration.greeks.scenario_grid \\
        --calibration data_store/calibrations/joint_cal_2026-10-16.pkl
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import ndtr

from joint_vol_calibration.config import DATA_DIR, ROOT_DIR
from joint_vol_calibration.greeks.risk_monitor import (
    _git_revision,
    calibration_carry,
    heston_otm_prices,
)
from joint_vol_calibration.models.implied_vol import black_price, bs_d1d2, implied_vol_black

SCENARIO_GRID_PATH = DATA_DIR / "greeks" / "scenario_grid.parquet"
PROVENANCE_KEY = b"scenario_provenance"
SCHEMA_VERSION = 1

TENOR_DAYS = 30
SPOT_SHOCKS_PCT: tuple[float, ...] = (-10.0, -5.0, -3.0, -1.0, 0.0, 1.0, 3.0, 5.0, 10.0)
VOL_SHOCKS_VP: tuple[float, ...] = (-5.0, -2.0, 0.0, 2.0, 5.0, 10.0)
DYNAMICS: tuple[str, ...] = ("sticky_strike", "sticky_moneyness")
VOL_FLOOR = 0.01

BOOK = (f"Long one {TENOR_DAYS}-day SPX straddle struck at the {TENOR_DAYS}-day forward, on one unit "
        "of the index, delta-hedged at inception in the index at its Black-Scholes (sticky-strike) "
        "delta; instantaneous shocks, full Black-76 revaluation on the headline Heston fit's smile")
UNITS = "index points per straddle on one unit of the index (×100 for one SPX contract)"
DYNAMICS_DOC = {
    "sticky_strike": "each strike keeps its implied vol (plus the parallel shock)",
    "sticky_moneyness": ("the smile is a function of ln(K/F): a fixed strike takes the vol of its new "
                         "moneyness (plus the parallel shock); at a zero vol shock this equals the "
                         "Heston revaluation at the shocked spot with v0 held"),
}


def model_smile(S: float, T: float, r: float, q: float, params: dict) -> Callable:
    """σ0(K) at maturity T: the Heston fit's Black implied vols (OTM Fourier
    prices, implied_vol_black). Raises if the pricer did not converge or a
    vol cannot be inverted."""
    D = float(np.exp(-r * T))

    def smile(K) -> np.ndarray:
        K = np.atleast_1d(np.asarray(K, dtype=float))
        otm, F, ok = heston_otm_prices(S, K, T, r, q, params)
        if not ok:
            raise RuntimeError(f"Heston Fourier pricer did not converge at T={T:.4f}y")
        iv = implied_vol_black(otm / D, F, K, T, is_call=K >= F, discount=1.0,
                               sigma_min=1e-4, sigma_max=10.0)
        if not np.isfinite(iv).all():
            raise ValueError(f"model implied vol not invertible at K={K[~np.isfinite(iv)]}")
        return np.asarray(iv, dtype=float)
    return smile


def straddle_value(F, K, T, sigma, D) -> np.ndarray:
    """Black-76 call + put at strike K (discount factor D)."""
    return (black_price(F, K, T, sigma, True, D) + black_price(F, K, T, sigma, False, D))


def straddle_spot_delta(S, F, K, T, sigma, D) -> float:
    """∂(C + P)/∂S at fixed σ: D·(F/S)·(2N(d1) − 1)."""
    d1, _ = bs_d1d2(F, K, T, sigma)
    return float(D * (F / S) * (2.0 * ndtr(d1) - 1.0))


def compute_scenario_grid(cal: dict, *, spot_shocks_pct: Sequence[float] = SPOT_SHOCKS_PCT,
                          vol_shocks_vp: Sequence[float] = VOL_SHOCKS_VP,
                          dynamics: Sequence[str] = DYNAMICS, tenor_days: int = TENOR_DAYS,
                          source: dict | None = None) -> pd.DataFrame:
    """The scenario grid of a joint_cal_*.pkl result (module docstring).
    Returns the frame with df.attrs["provenance"]."""
    unknown = set(dynamics) - set(DYNAMICS_DOC)
    if unknown:
        raise ValueError(f"unknown dynamics {sorted(unknown)}; choose from {sorted(DYNAMICS_DOC)}")
    for key in ("S", "params"):
        if cal.get(key) is None:
            raise ValueError(f"calibration result has no {key!r}")
    S0 = float(cal["S"])
    p = {k: float(cal["params"][k]) for k in ("kappa", "theta", "sigma", "rho", "v0")}
    T = tenor_days / 365.0
    r_fn, q_fn, carry = calibration_carry(cal)
    r, q = float(r_fn(T)), float(q_fn(T))
    D = float(np.exp(-r * T))
    F0 = S0 * float(np.exp((r - q) * T))
    K = F0
    smile = model_smile(S0, T, r, q, p)
    s_shocks = np.asarray(spot_shocks_pct, dtype=float)
    v_shocks = np.asarray(vol_shocks_vp, dtype=float) / 100.0
    ratio = 1.0 + s_shocks / 100.0
    if (ratio <= 0).any():
        raise ValueError("a spot shock of -100% or below")
    S1, F1 = S0 * ratio, F0 * ratio       # (not F0·S1/S0: exact at a zero shock)
    sig0 = float(smile(K)[0])
    V0 = float(straddle_value(F0, K, T, sig0, D))
    delta0 = straddle_spot_delta(S0, F0, K, T, sig0, D)
    # the unshifted vol of the fixed strike after each spot shock
    mon = smile(K / ratio)
    mon[ratio == 1.0] = sig0   # the same strike: the same number (a batch can differ in the last bit)
    base = {"sticky_strike": np.full(len(S1), sig0), "sticky_moneyness": mon}
    rows, n_floored = [], 0
    for dyn in dynamics:
        for v in v_shocks:
            sig = base[dyn] + v
            floored = sig < VOL_FLOOR
            n_floored += int(floored.sum())
            sig = np.where(floored, VOL_FLOOR, sig)
            V1 = straddle_value(F1, K, T, sig, D)
            hedge = -delta0 * (S1 - S0)
            pnl = V1 - V0 + hedge
            for i, s in enumerate(s_shocks):
                rows.append({"spot_shock_pct": float(s), "vol_shock_vp": float(round(v * 100.0, 10)),
                             "dynamics": dyn, "pnl": float(pnl[i]), "spot": float(S1[i]),
                             "iv_strike": float(sig[i]), "straddle_value": float(V1[i]),
                             "hedge_pnl": float(hedge[i]), "vol_floored": bool(floored[i])})
    df = pd.DataFrame(rows)
    as_of = str(cal.get("as_of_date") or cal.get("spot_date") or "")[:10] or None
    df.attrs["provenance"] = {
        "schema_version": SCHEMA_VERSION, "book": BOOK, "as_of_date": as_of, "units": UNITS,
        "built_at_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "code_revision": _git_revision(),
        "source": source or {},
        "params": p, "S": S0, "tenor_days": tenor_days, "T": T, "r": r, "q": q, "discount": D,
        "forward": F0, "strike": K, "iv_atm_forward": sig0, "straddle_value": V0,
        "hedge_delta": delta0, "carry": carry,
        "spot_shocks_pct": [float(x) for x in s_shocks],
        "vol_shocks_vp": [float(x) for x in np.asarray(vol_shocks_vp, dtype=float)],
        "dynamics": {d: DYNAMICS_DOC[d] for d in dynamics},
        "vol_floor": VOL_FLOOR, "n_vol_floored": n_floored,
        "notes": ["instantaneous shocks: no time decay; r, q and the hedge held",
                  "P&L at (0, 0) is zero: the unshocked book revalues to itself"],
    }
    return df


def save_scenario_grid(df: pd.DataFrame, path: Path | None = None) -> Path:
    """Write the grid with its provenance (parquet metadata b"scenario_provenance"),
    atomically: a temporary file in the same directory, then os.replace."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    out = Path(path) if path else SCENARIO_GRID_PATH
    out.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df, preserve_index=False)
    meta = dict(table.schema.metadata or {})
    meta[PROVENANCE_KEY] = json.dumps(df.attrs.get("provenance") or {}, default=str).encode()
    tmp = out.with_name(f".{out.name}.tmp")
    pq.write_table(table.replace_schema_metadata(meta), tmp)
    os.replace(tmp, out)
    return out


def load_scenario_grid(path: Path | None = None) -> pd.DataFrame:
    """Read a saved grid; its provenance goes to attrs["provenance"]."""
    import pyarrow.parquet as pq

    p = Path(path) if path else SCENARIO_GRID_PATH
    df = pd.read_parquet(p)
    raw = (pq.read_schema(p).metadata or {}).get(PROVENANCE_KEY)
    df.attrs["provenance"] = json.loads(raw.decode()) if raw else None
    return df


def build_from_calibration(calibration_path, out_path: Path | None = None) -> pd.DataFrame:
    """compute_scenario_grid on one pickle, saved with its source recorded."""
    path = Path(calibration_path)
    with open(path, "rb") as f:
        cal = pickle.load(f)  # noqa: S301 (trusted: the calibrator's own joint_cal_*.pkl)
    try:
        rel = str(path.resolve().relative_to(ROOT_DIR.resolve()))
    except ValueError:
        rel = str(path)
    df = compute_scenario_grid(cal, source={"calibration_file": rel,
                                            "spot_date": cal.get("spot_date"),
                                            "model_version": cal.get("model_version")})
    save_scenario_grid(df, out_path)
    return df


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spot x vol scenario grid of the reference book "
                                             "from a joint_cal_*.pkl calibration.")
    ap.add_argument("--calibration", required=True, help="path to a joint_cal_*.pkl")
    ap.add_argument("--out", default=str(SCENARIO_GRID_PATH), help="output parquet path")
    args = ap.parse_args(argv)
    df = build_from_calibration(args.calibration, Path(args.out))
    prov = df.attrs["provenance"]
    g = df[df["dynamics"] == "sticky_strike"]
    print(f"{args.out}: {len(df)} cells ({len(prov['spot_shocks_pct'])} spot x "
          f"{len(prov['vol_shocks_vp'])} vol x {len(prov['dynamics'])} dynamics), as_of "
          f"{prov['as_of_date']}; sticky-strike P&L {g['pnl'].min():+.1f} .. {g['pnl'].max():+.1f} "
          f"index points; {prov['n_vol_floored']} vol(s) floored")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
