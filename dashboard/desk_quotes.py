"""
desk_quotes.py — the quotes behind a smile chart, packed for the quote inspector
(pure functions; no Flask, no I/O).

A smile chart (Surface #chart-slices, the Calibration smiles) carries one block
per expiry in ``spec.quotes.groups``. desk.js unpacks it to draw the bid–ask
bars, the mids, every model at the quoted strikes and the residual strip, and
to say, for the quote under the pointer, its strike, right, ln(K/F), forward
delta, bid / mid / ask IV and each model's IV and z. ``decode`` below is the
same arithmetic in Python: the tests check it against residuals.parquet, and
the browser tests check desk.js against it.

Packing (``encode``), per expiry:
  * F (the parity forward, full precision), T (calendar days), step (the
    strikes' grid step, their greatest common divisor), K0 and dK (the first
    strike and the gaps to the next one in steps), r (one 'P' / 'C' per quote);
  * mid: mid IV in units of 10⁻⁵ vol pts (IV % × 10⁵, an integer); db, da: mid
    − bid and ask − mid in 10⁻⁸ vol pts (the half-spread divides every z, so its
    relative error multiplies |z|: at 10⁻⁶ a 1,456-half-spread quote decoded
    0.06 away from its error_hs); m[model]: model − mid in 10⁻⁴ vol pts (None
    where the model has no value);
  * s[model]: one digit per quote, a bit mask: 1 = the model was fitted to it
    (in_sample), 2 = the model lies inside its bid–ask (inside_spread; the
    producer's flag, not recomputed from the packed IVs, which could flip it
    within 10⁻⁴ vol pts of an edge);
  * f: one digit per quote, a bit mask: 1 = a leg of a constraint row violated
    at bid / ask (arb_exec), 2 = moved by the l1 repair (|repair_dprice| > 1e-9
    index points; the producer's n_perturbed counts the same quotes), 4 = moved
    outside its bid–ask by it (arb_flag: left out of the surfaces' fitted set).
At these units the forward delta from the packed mid differs from the one on
the parquet's own values by about 10⁻⁷, and z from its error_hs by under 0.01
for any |z| up to 10⁴ (model − mid contributes ≤ 0.005, the half-spread
≤ |z| · 5·10⁻⁷; tested on every stored session).

Conventions (the producer's, joint_vol_calibration.surface.quotes / metrics):
k = ln(K/F); T in years = days / 365; forward delta = N(d₁) for a call, N(d₁) −
1 for a put, d₁ = −k/√w + √w/2, w = σ²T on the quote's mid IV (undiscounted, not
premium-adjusted); half-spread h = (ask IV − bid IV)/2; z = (model IV − mid
IV)/max(h, 0.01 vol pts); a model is inside the spread iff bid IV ≤ model IV ≤
ask IV (the producer's inside_spread).
"""
from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.special import ndtr

IV_SCALE = 100_000                  # packed mid IV: vol pts × 10⁵ (10⁻⁵ vol pts)
SPREAD_SCALE = 100_000_000          # packed mid − bid, ask − mid: vol pts × 10⁸
MODEL_SCALE = 10_000                # packed model − mid: vol pts × 10⁴ (10⁻⁴ vol pts)
DAYS_PER_YEAR = 365.0               # the producer's clock (quotes.py: T_days = T · 365)
H_FLOOR_VP = 0.01                   # metrics.H_FLOOR_VP: z's half-spread floor (a test checks they agree)
REPAIR_TOL = 1e-9                   # index points: a repair move that is not float noise
FLAG_EXEC, FLAG_REPAIRED, FLAG_OUTSIDE = 1, 2, 4
MODEL_FIT, MODEL_INSIDE = 1, 2
Z_CLIP = 10.0                       # the residual strip's y range: |z| beyond it is drawn at the edge


def _ints(x: Any, scale: int) -> list[int | None]:
    return [None if not math.isfinite(v) else round(float(v) * 100.0 * scale)
            for v in np.asarray(x, dtype=float)]


def encode_group(q: Any, model_rows: Mapping[str, Any]) -> dict:
    """One expiry's quotes (``q``: its market rows, any model's) and each model's
    rows (``model_rows``: model -> its rows of the same expiry), packed."""
    q = q.sort_values("k", kind="mergesort")
    K = q["strike"].to_numpy(float)
    Ki = np.round(K).astype(np.int64)
    if not np.allclose(K, Ki):
        raise ValueError("strikes are not whole index points")
    step = int(np.gcd.reduce(Ki)) or 1
    mid = q["mid_iv"].to_numpy(float)
    g: dict[str, Any] = {
        "F": float(q["forward"].iloc[0]), "T": float(q["T_days"].iloc[0]), "step": step,
        "K0": int(Ki[0]), "dK": [int(d) for d in np.diff(Ki) // step],
        "r": "".join(str(x)[0].upper() for x in q["right"]),
        "mid": _ints(mid, IV_SCALE),
        "db": _ints(mid - q["bid_iv"].to_numpy(float), SPREAD_SCALE),
        "da": _ints(q["ask_iv"].to_numpy(float) - mid, SPREAD_SCALE),
        "m": {}, "s": {},
    }
    keys = list(zip(q["strike"].to_numpy(float), q["right"].astype(str)))
    for m, rows in model_rows.items():
        r = rows.drop_duplicates(["strike", "right"]).set_index(["strike", "right"])
        iv = r["model_iv"].reindex(keys).to_numpy(float)
        g["m"][m] = _ints(iv - mid, MODEL_SCALE)
        st = np.zeros(len(keys), dtype=int)
        for col, bit in (("in_sample", MODEL_FIT), ("inside_spread", MODEL_INSIDE)):
            if col in r.columns:
                st |= np.where(r[col].reindex(keys).fillna(False).astype(bool).to_numpy(), bit, 0)
        g["s"][m] = "".join(str(int(x)) for x in st)
    flags = np.zeros(len(q), dtype=int)
    if "arb_exec" in q.columns:
        flags |= np.where(q["arb_exec"].fillna(False).astype(bool), FLAG_EXEC, 0)
    if "repair_dprice" in q.columns:
        dq = np.abs(q["repair_dprice"].to_numpy(float))
        flags |= np.where(np.isfinite(dq) & (dq > REPAIR_TOL), FLAG_REPAIRED, 0)
    if "arb_flag" in q.columns:
        flags |= np.where(q["arb_flag"].fillna(False).astype(bool), FLAG_OUTSIDE, 0)
    g["f"] = "".join(str(int(x)) for x in flags)
    return g


def encode(res: Any, models: Sequence[str], expiries: Sequence[str] | None = None) -> dict[str, dict]:
    """Every expiry of a residual table (or ``expiries``), packed; the market
    columns are read from the first model's rows (they repeat per model)."""
    models = [m for m in models if (res["model"] == m).any()]
    base = res[res["model"] == models[0]]
    exps = list(expiries) if expiries is not None else sorted(base["expiry"].unique())
    out = {}
    for e in exps:
        q = base[base["expiry"] == e]
        if q.empty:
            continue
        out[str(e)] = encode_group(q, {m: res[(res["model"] == m) & (res["expiry"] == e)] for m in models})
    return out


def strikes(g: Mapping) -> list[int]:
    out = [int(g["K0"])]
    for d in g["dK"]:
        out.append(out[-1] + int(d) * int(g["step"]))
    return out


def forward_delta(k: float, iv_pct: float, T_days: float, right: str) -> float:
    """Undiscounted forward Black delta on the quote's own vol: N(d₁) for a call,
    N(d₁) − 1 for a put, d₁ = −k/√w + √w/2 with w = σ²T."""
    w = (iv_pct / 100.0) ** 2 * T_days / DAYS_PER_YEAR
    d1 = -k / math.sqrt(w) + math.sqrt(w) / 2.0
    n = float(ndtr(d1))
    return n - 1.0 if right == "P" else n


def delta_coord(k: float, iv_pct: float, T_days: float) -> float:
    """The forward-delta axis: N(−d₁), so an OTM put at |Δ| = 0.10 sits at 0.10,
    the forward near 0.5 and an OTM call at Δ = 0.10 at 0.90 (puts left, calls right)."""
    w = (iv_pct / 100.0) ** 2 * T_days / DAYS_PER_YEAR
    return float(ndtr(k / math.sqrt(w) - math.sqrt(w) / 2.0))


def decode(g: Mapping) -> list[dict]:
    """The quotes of one packed expiry, each with every value the inspector shows."""
    out = []
    Ks = strikes(g)
    F, T = float(g["F"]), float(g["T"])
    for i, K in enumerate(Ks):
        mid = g["mid"][i] / IV_SCALE
        db, da = g["db"][i] / SPREAD_SCALE, g["da"][i] / SPREAD_SCALE
        hs = (db + da) / 2.0
        bid, ask = mid - db, mid + da
        k = math.log(K / F)
        right = g["r"][i]
        fl = int(g["f"][i]) if g.get("f") else 0
        row = {"strike": K, "right": right, "k": k, "T_days": T, "mid": mid, "bid": bid, "ask": ask,
               "half_spread": hs, "delta": forward_delta(k, mid, T, right), "x_delta": delta_coord(k, mid, T),
               "exec": bool(fl & FLAG_EXEC), "repaired": bool(fl & FLAG_REPAIRED),
               "outside": bool(fl & FLAG_OUTSIDE), "models": {}}
        for m, errs in g["m"].items():
            e = errs[i]
            st = int((g.get("s") or {}).get(m, "0" * len(Ks))[i])
            row["models"][m] = {"iv": None if e is None else mid + e / MODEL_SCALE,
                                "z": None if e is None else (e / MODEL_SCALE) / max(hs, H_FLOOR_VP),
                                "inside": None if e is None else bool(st & MODEL_INSIDE),
                                "fit": bool(st & MODEL_FIT)}
        out.append(row)
    return out


def _r(x: float | None, n: int) -> float | None:
    return None if x is None else round(x, n)


def series_data(rows: Sequence[Mapping], field: str, model: str | None = None, x: str = "k") -> list[list]:
    """The points a quote series draws (desk.js quotePoints does the same):
    'ba' [x, bid, ask], 'mid' [x, mid], 'model' [x, model IV], 'fit' / 'oos' [x,
    mid] of the model's fitted / other quotes, 'z' / 'zfit' [x, z clipped at
    ±Z_CLIP, z] of every quote / the fitted ones; x is ln(K/F) ('k') or the
    forward-delta axis N(−d₁) ('delta')."""
    out: list[list] = []
    for q in rows:
        xv = round(q["k"] if x == "k" else q["x_delta"], 6)
        mm = q["models"].get(model) if model else None
        if field == "ba":
            out.append([xv, _r(q["bid"], 6), _r(q["ask"], 6)])
        elif field == "mid":
            out.append([xv, _r(q["mid"], 5)])
        elif field == "model":
            out.append([xv, _r(mm["iv"], 5) if mm else None])
        elif field in ("fit", "oos"):
            if mm and mm["fit"] == (field == "fit"):
                out.append([xv, _r(q["mid"], 5)])
        elif field in ("z", "zfit") and mm and mm["z"] is not None and (field == "z" or mm["fit"]):
            z = mm["z"]
            out.append([xv, round(max(-Z_CLIP, min(Z_CLIP, z)), 3), round(z, 3)])
    return out
