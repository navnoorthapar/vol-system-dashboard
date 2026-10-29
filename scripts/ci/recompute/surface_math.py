"""
surface_math.py: the Surface page's numbers rebuilt from data_store/surface and
the market DB by this audit's own code, from the definitions the page states.

Conventions (the page's captions): k = ln(K/F); T = calendar days / 365; total
variance w = σ²T; the forward delta is the undiscounted Black call delta N(d₁),
d₁ = −k/√w + √w/2, on the slice's own vol (a put's is N(d₁) − 1); RR = σ_call −
σ_put, BF = (σ_call + σ_put)/2 − ATMF, in vol pts; a constant maturity is linear
in total variance at fixed delta between the two bracketing listed expiries,
never extrapolated; the risk-neutral density of ln(S_T/F) is Breeden–Litzenberger
on the slice, p(k) = g(k)/√(2πw)·exp(−d₋²/2) with Durrleman's g; realised vol over
h sessions is 100·√(252/h·Σr²) of close-to-close log returns.

Nothing here imports dashboard/ or joint_vol_calibration/.
"""
from __future__ import annotations

import itertools
import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm

from .data import Store, strike_at_delta, svi_total_variance

DAYS = 365.0
PILLARS = (("10P", -0.10), ("25P", -0.25), ("ATM", None), ("25C", 0.25), ("10C", 0.10))
CM_TENORS = ((30, "1M"), (61, "2M"), (91, "3M"), (182, "6M"), (365, "1Y"))
SKEW_UNIT_K = 0.1                     # the grid's ATM skew: vol pts per 0.1 of ln-moneyness
FWD_MIN_DAYS = 7.0                    # a forward period spans at least this many days
K_EDGES = (-0.30, -0.20, -0.14, -0.09, -0.05, -0.02, 0.0, 0.02, 0.05, 0.09, 0.14, 0.20, 0.30)
N_BUCKETS = len(K_EDGES) + 1          # an open wing bucket on each side
Z_FLOOR_VP = 0.01                     # z = (model − mid) / max(half-spread, 0.01 vol pts)
REPAIR_TOL = 1e-9                     # index points: a repair move that is not float noise
SPX_MULTIPLIER = 100.0
HEADLINE_SURFACE = "svi_cal"
# the tail thresholds (K/F) and the band of the density table
THRESHOLDS = (("lt80", 0.80, "below"), ("lt90", 0.90, "below"), ("gt110", 1.10, "above"))
BAND = (0.95, 1.05)
RND_TENORS = (30, 91, 182)            # the density chart: listed expiries nearest these days
RND_TENOR_TOL = 0.35                  # ... within 35% of them
RND_CHART_STEP = 0.005                # the density chart's grid step in k
RND_TABLE_K = (-0.30, -0.25, -0.20, -0.15, -0.10, -0.05, 0.0, 0.05, 0.10, 0.15)
# the realised-vol cone
CONE_H = (10, 21, 42, 63, 126, 252)
CONE_PCTS = (5, 25, 50, 75, 95)
CONE_SAMPLES = (("since-2010", "Since 2010", "Since 2010"), ("last-3y", "Last 3 years", "Last 3 y"))
CONE_START = "2010-01-04"
CONE_SINCE = 2010                     # ... the first sample's year
CONE_YEARS = 3                        # the second sample: the last 3 years
# the backtest's β estimator: 30-day ATM-forward Black-76 vol / VIX on a DB chain
BETA_Q = 0.013
BETA_DAYS = (7.0, 75.0)
BETA_N_FORWARD = 3
BETA_SPOT_TOL = 0.005

MODEL_LABELS = {"heston_v2": "Heston", "heston": "Heston", "svi_cal": "SVI-cal", "svi": "SVI",
                "essvi": "eSSVI", "rough_heston": "Rough Heston"}
ARB_LABELS = {"C1_outright": "Outright bounds", "C2_vertical_bound": "Vertical bound",
              "C2_vertical_spread": "Vertical spread", "C3_vertical_butterfly": "Butterfly",
              "C4_calendar_spread": "Calendar spread", "C5_calendar_vertical_spread": "Calendar vertical",
              "C6_calendar_butterfly": "Calendar butterfly"}
VIX_FAMILY = (("VIX9D", 9), ("VIX", 30), ("VIX3M", 93), ("VIX6M", 184))


def label(m: str) -> str:
    """A model's name as the page prints it."""
    return MODEL_LABELS.get(m, m.replace("_", " ").title())


def num(x: Any) -> float | None:
    """A finite float, or None."""
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


# ── one stored session ────────────────────────────────────────────────────────
@dataclass
class Session:
    name: str
    res: pd.DataFrame                  # every (model, quote) row
    models: list[str]                  # in the file's order
    sj: dict | None = None             # surface.json
    sc: dict | None = None             # scorecard.json
    vx: dict | None = None             # vix_replication.json
    _cache: dict = field(default_factory=dict)

    @property
    def mkt(self) -> pd.DataFrame:
        """One row per quote: the first model's rows (the market columns repeat per model)."""
        return self.res[self.res["model"] == self.models[0]]

    def model_rows(self, m: str) -> pd.DataFrame:
        return self.res[self.res["model"] == m]


def surface_sessions(store: Store) -> list[str]:
    """Every data_store/surface/YYYY-MM-DD/ that holds residuals.parquet (the page's sessions)."""
    root = store.root / "surface"
    return sorted(p.name for p in root.iterdir() if p.is_dir() and len(p.name) == 10
                  and p.name[4] == "-" and (p / "residuals.parquet").exists())


def load_session(store: Store, name: str) -> Session:
    key = "surface-session:" + name
    if key not in store._cache:
        res = store.parquet(f"surface/{name}/residuals.parquet").copy()
        res["expiry"] = res["expiry"].astype(str).str[:10]
        res["model"] = res["model"].astype(str)
        models = list(dict.fromkeys(res["model"]))

        def opt(fname: str) -> dict | None:
            p = store.root / "surface" / name / fname
            return store.json(f"surface/{name}/{fname}") if p.exists() else None
        store._cache[key] = Session(name, res, models, opt("surface.json"), opt("scorecard.json"),
                                    opt("vix_replication.json"))
    return store._cache[key]


def expiries(s: Session) -> dict[str, float]:
    """{expiry: its quotes' median T_days} (the residual table's clock), shortest first."""
    med = s.mkt.groupby("expiry")["T_days"].median().sort_values()
    return {str(e): float(t) for e, t in med.items()}


def oos_rows(s: Session, m: str) -> pd.DataFrame:
    """A model's out-of-sample quotes: not fitted to (in_sample False), with a model IV."""
    q = s.model_rows(m)
    if "in_sample" in q.columns:
        q = q[~q["in_sample"].fillna(False).astype(bool)]
    return q[pd.to_numeric(q["model_iv"], errors="coerce").notna()]


def half_spread_vp(q: pd.DataFrame) -> pd.Series:
    """(ask IV − bid IV)/2 in vol pts, from the quote's own bid and ask IVs."""
    return (q["ask_iv"] - q["bid_iv"]) * 50.0


def z_hs(q: pd.DataFrame) -> pd.Series:
    """(model IV − mid IV) / max(half-spread, 0.01 vol pts), from the quote's IVs."""
    return (q["model_iv"] - q["mid_iv"]) * 100.0 / np.maximum(half_spread_vp(q), Z_FLOOR_VP)


# ── ln(K/F) buckets ───────────────────────────────────────────────────────────
def bucket(k: Any) -> np.ndarray:
    """Right-closed buckets on K_EDGES: 0 is k ≤ −0.30, N_BUCKETS − 1 is k > +0.30."""
    return np.searchsorted(np.asarray(K_EDGES), np.asarray(k, float), side="left")


def buckets_used(k: Any) -> list[int]:
    """Every inner bucket, and an open wing only where some quote lies in it."""
    present = set(bucket(k).tolist())
    return [b for b in range(N_BUCKETS) if 0 < b < N_BUCKETS - 1 or b in present]


def _signed(x: float, spec: str) -> str:
    body = format(abs(x), spec)
    return ("" if float(body) == 0.0 else ("+" if x > 0 else "−")) + body


def bucket_label(b: int, compact: bool) -> str:
    """'≤ −0.30', '> +0.30'; inside, the midpoint ('−0.25') or the range ('−0.30…−0.20')."""
    if b == 0:
        return f"≤ {_signed(K_EDGES[0], '.2f')}"
    if b == N_BUCKETS - 1:
        return f"> {_signed(K_EDGES[-1], '.2f')}"
    lo, hi = K_EDGES[b - 1], K_EDGES[b]
    if not compact:
        return f"{format(lo, '+.2f')}…{format(hi, '+.2f')}".replace("-", "−")
    mid = (lo + hi) / 2.0
    return "0" if abs(mid) < 1e-12 else format(mid, "+.3f").rstrip("0").rstrip(".").replace("-", "−")


# ── SVI-cal in desk coordinates ───────────────────────────────────────────────
def svi_w1(k: Any, sl: dict) -> Any:
    """∂w/∂k of a raw-SVI slice."""
    x = np.asarray(k, float) - sl["m"]
    return sl["b"] * (sl["rho"] + x / np.sqrt(x * x + sl["sigma"] ** 2))


def svi_w2(k: Any, sl: dict) -> Any:
    """∂²w/∂k² of a raw-SVI slice."""
    x = np.asarray(k, float) - sl["m"]
    return sl["b"] * sl["sigma"] ** 2 / (x * x + sl["sigma"] ** 2) ** 1.5


def slices(sj: dict | None, model: str = HEADLINE_SURFACE) -> list[dict]:
    """A surface's stored slices (surface.json), shortest first."""
    sl = ((((sj or {}).get("surfaces") or {}).get(model) or {}).get("params") or {}).get("slices") or []
    return sorted(sl, key=lambda s: float(s["T"]))


def quoted_k(s: Session) -> dict[str, tuple[float, float]]:
    """Each expiry's quoted ln(K/F) range: the SVI-cal rows (every model is scored on the same quotes)."""
    q = s.res[s.res["model"] == HEADLINE_SURFACE] if (s.res["model"] == HEADLINE_SURFACE).any() else s.res
    g = q.groupby("expiry")["k"].agg(["min", "max"])
    return {str(e): (float(lo), float(hi)) for e, lo, hi in zip(g.index, g["min"], g["max"], strict=True)}


def inside(k: float, rng: tuple[float, float] | None) -> bool:
    """k lies in a quoted range (never, without one)."""
    return rng is not None and rng[0] <= k <= rng[1]


def _rr_bf(vol: dict, xp: dict, wing: str) -> tuple[float | None, float | None]:
    c, p, a = vol.get(wing + "C"), vol.get(wing + "P"), vol.get("ATM")
    if c is None or p is None or a is None or xp.get(wing + "C") or xp.get(wing + "P"):
        return None, None
    return c - p, 0.5 * (c + p) - a


def listed_grid(s: Session) -> list[dict]:
    """One row per SVI-cal slice, shortest first: pillar k and vol, extrapolation flags
    (a pillar outside the quoted ln(K/F) range), RR/BF, ATM skew, and the strip's VS
    vol and strike count."""
    if "listed" in s._cache:
        return s._cache["listed"]
    kr = quoted_k(s)
    strip = {str(r.get("expiry"))[:10]: r for r in (s.vx or {}).get("strip") or []}
    rows = []
    for sl in slices(s.sj):
        e, T = str(sl["expiry"])[:10], float(sl["T"])
        ks, vol, xp = {}, {}, {}
        for key, delta in PILLARS:
            k = 0.0 if delta is None else strike_at_delta(sl, delta)
            ks[key], vol[key] = k, 100.0 * math.sqrt(float(svi_total_variance(k, sl)) / T)
            xp[key] = not inside(k, kr.get(e))
        w0, w0p = float(svi_total_variance(0.0, sl)), float(svi_w1(0.0, sl))
        st = strip.get(e)
        vs = num((st or {}).get("var_swap_vol"))
        r = {"expiry": e, "T": T, "T_days": T * DAYS, "k": ks, "vol": vol, "xp": xp, "atmf": vol["ATM"],
             "w0": w0, "w0p": w0p, "skew_vp": w0p / (2.0 * math.sqrt(w0 * T)) * 100.0 * SKEW_UNIT_K,
             "vs": vs, "vs_minus_atmf": None if vs is None else vs - vol["ATM"],
             "n_strikes": ((st.get("n_puts") or 0) + (st.get("n_calls") or 0)) if st else None}
        r["rr25"], r["bf25"] = _rr_bf(vol, xp, "25")
        r["rr10"], r["bf10"] = _rr_bf(vol, xp, "10")
        rows.append(r)
    s._cache["listed"] = rows
    return rows


def _lin_w(v1: float | None, T1: float, v2: float | None, T2: float, T: float) -> float | None:
    """A vol (%) at T, linear in total variance between (T1, v1) and (T2, v2)."""
    if v1 is None or v2 is None:
        return None
    w1, w2 = (v1 / 100.0) ** 2 * T1, (v2 / 100.0) ** 2 * T2
    x = 0.0 if T2 == T1 else (T - T1) / (T2 - T1)
    return 100.0 * math.sqrt(((1.0 - x) * w1 + x * w2) / T)


def cm_grid(s: Session) -> list[dict]:
    """The constant maturities 1M…1Y from the listed grid (never extrapolated)."""
    if "cm" in s._cache:
        return s._cache["cm"]
    listed = listed_grid(s)
    out = []
    for D, lab in CM_TENORS:
        T = D / DAYS
        r: dict[str, Any] = {
            "days": D, "label": lab, "T": T, "available": False, "lower": None, "upper": None,
            "vol": dict.fromkeys(k for k, _ in PILLARS), "xp": dict.fromkeys((k for k, _ in PILLARS), False),
            "atmf": None, "rr25": None, "bf25": None, "rr10": None, "bf10": None, "skew_vp": None,
            "vs": None, "vs_minus_atmf": None, "strikes": (None, None)}
        if listed and listed[0]["T"] - 1e-12 <= T <= listed[-1]["T"] + 1e-12:
            lo = max((x for x in listed if x["T"] <= T + 1e-12), key=lambda x: x["T"])
            hi = min((x for x in listed if x["T"] >= T - 1e-12), key=lambda x: x["T"])
            r.update(available=True, lower=lo["expiry"], upper=hi["expiry"],
                     strikes=(lo["n_strikes"], hi["n_strikes"]))
            for key, _ in PILLARS:
                r["vol"][key] = _lin_w(lo["vol"][key], lo["T"], hi["vol"][key], hi["T"], T)
                r["xp"][key] = bool(lo["xp"][key] or hi["xp"][key])
            r["atmf"] = r["vol"]["ATM"]
            r["rr25"], r["bf25"] = _rr_bf(r["vol"], r["xp"], "25")
            r["rr10"], r["bf10"] = _rr_bf(r["vol"], r["xp"], "10")
            x = 0.0 if hi["T"] == lo["T"] else (T - lo["T"]) / (hi["T"] - lo["T"])
            w0, w0p = (1 - x) * lo["w0"] + x * hi["w0"], (1 - x) * lo["w0p"] + x * hi["w0p"]
            r["skew_vp"] = w0p / (2.0 * math.sqrt(w0 * T)) * 100.0 * SKEW_UNIT_K
            r["vs"] = _lin_w(lo["vs"], lo["T"], hi["vs"], hi["T"], T)
            r["vs_minus_atmf"] = None if r["vs"] is None else r["vs"] - r["atmf"]
        out.append(r)
    s._cache["cm"] = out
    return out


def state_values(cm: list[dict]) -> dict[str, float | None]:
    """The surface-state tiles of one session: 1M ATMF, RR25, BF25, VS − ATMF and 1Y − 1M ATMF."""
    by = {r["label"]: r for r in cm}
    m1, y1 = by["1M"], by["1Y"]
    slope = (y1["atmf"] - m1["atmf"]) if (m1["atmf"] is not None and y1["atmf"] is not None) else None
    return {"atmf": m1["atmf"], "rr25": m1["rr25"], "bf25": m1["bf25"], "vs": m1["vs_minus_atmf"],
            "slope": slope}


def forward_vols(cm: list[dict]) -> list[dict]:
    """Forward ATM vol between consecutive constant maturities at least 7 days apart."""
    pts = [p for p in cm if p["available"] and p["atmf"] is not None]
    out, prev = [], None
    for p in pts:
        if prev is not None and (p["T"] - prev["T"]) * DAYS < FWD_MIN_DAYS:
            continue
        if prev is not None:
            w1, w2 = (prev["atmf"] / 100) ** 2 * prev["T"], (p["atmf"] / 100) ** 2 * p["T"]
            fvar = (w2 - w1) / (p["T"] - prev["T"])
            out.append({"from": prev["label"], "to": p["label"], "t1": prev["T"] * DAYS, "t2": p["T"] * DAYS,
                        "atm1": prev["atmf"], "fwd_vol": 100.0 * math.sqrt(fvar) if fvar >= 0 else None})
        prev = p
    return out


# ── risk-neutral density of one slice ─────────────────────────────────────────
def durrleman_g(k: Any, sl: dict) -> Any:
    """Durrleman's g(k) = (1 − kw′/2w)² − w′²/4·(1/w + 1/4) + w″/2; g ≥ 0 is no butterfly arbitrage."""
    k = np.asarray(k, float)
    w, w1, w2 = np.asarray(svi_total_variance(k, sl)), svi_w1(k, sl), svi_w2(k, sl)
    return (1.0 - k * w1 / (2.0 * w)) ** 2 - (w1 * w1 / 4.0) * (1.0 / w + 0.25) + w2 / 2.0


def density(k: Any, sl: dict) -> Any:
    """p(k) = g(k)/√(2πw)·exp(−d₋²/2), d₋ = −k/√w − √w/2."""
    k = np.asarray(k, float)
    w = np.asarray(svi_total_variance(k, sl))
    dm = -k / np.sqrt(w) - 0.5 * np.sqrt(w)
    return durrleman_g(k, sl) / np.sqrt(2.0 * math.pi * w) * np.exp(-0.5 * dm * dm)


def prob_below(k: float, sl: dict) -> float:
    """P(S_T < F·eᵏ) = N(−d₋) + φ(d₋)·w′/(2√w)."""
    w = float(svi_total_variance(k, sl))
    dm = -k / math.sqrt(w) - 0.5 * math.sqrt(w)
    return float(norm.cdf(-dm) + norm.pdf(dm) * float(svi_w1(k, sl)) / (2.0 * math.sqrt(w)))


def _integral(f: Callable[[float], float], sl: dict) -> float:
    """∫f over the whole line: [−40, 40], split at ±6 and ±12 ATM total-vol units."""
    sd = math.sqrt(max(float(svi_total_variance(0.0, sl)), 1e-12))
    cuts = sorted({-40.0, -12 * sd, -6 * sd, 0.0, 6 * sd, 12 * sd, 40.0})
    return sum(quad(f, a, b, limit=400, epsabs=1e-13, epsrel=1e-11)[0] for a, b in itertools.pairwise(cuts))


def _p_tail(sl: dict, rng: tuple[float, float] | None, kf: float, side: str) -> float | None:
    """P(S_T < kf·F) ('below') or P(S_T > kf·F) ('above'); None beyond the quoted range."""
    k = math.log(kf)
    if not inside(k, rng):
        return None
    pb = prob_below(k, sl)
    return pb if side == "below" else 1.0 - pb


def _moments(sl: dict) -> tuple[float, float]:
    """∫p and E[S_T/F] = ∫eᵏp over the whole line."""
    def p(k: float) -> float:
        return float(density(k, sl))

    def kp(k: float) -> float:
        return math.exp(k) * p(k)
    return _integral(p, sl), _integral(kp, sl)


def tail_rows(s: Session) -> list[dict]:
    """Every SVI-cal slice: the threshold probabilities (None beyond the quoted strikes),
    the mass outside the quoted range, ∫p and E[S_T/F], and the certificate."""
    if "tails" in s._cache:
        return s._cache["tails"]
    kr = quoted_k(s)
    rows = []
    for sl in slices(s.sj):
        e = str(sl["expiry"])[:10]
        rng = kr.get(e)
        prob = {key: _p_tail(sl, rng, kf, side) for key, kf, side in THRESHOLDS}
        b_lo, b_hi = _p_tail(sl, rng, BAND[0], "below"), _p_tail(sl, rng, BAND[1], "above")
        prob["out5"] = None if b_lo is None or b_hi is None else b_lo + b_hi
        mass, mean = _moments(sl)
        rows.append({"expiry": e, "T_days": float(sl["T"]) * DAYS, "sl": sl,
                     "k_lo": rng[0] if rng else None, "k_hi": rng[1] if rng else None, "prob": prob,
                     "below": prob_below(rng[0], sl) if rng else None,
                     "above": 1.0 - prob_below(rng[1], sl) if rng else None,
                     "mass": mass, "mean": mean, "certified": bool(sl.get("certified")),
                     "g_min": num(sl.get("g_min"))})
    s._cache["tails"] = rows
    return rows


def rnd_picks(rows: list[dict]) -> list[dict]:
    """The certified listed expiry nearest 30, 91 and 182 days (within 35%), each once."""
    out, used = [], set()
    for D in RND_TENORS:
        c = [r for r in rows if r["expiry"] not in used and r["k_lo"] is not None and r["certified"]
             and abs(r["T_days"] - D) <= RND_TENOR_TOL * D]
        if c:
            r = min(c, key=lambda r: abs(r["T_days"] - D))
            used.add(r["expiry"])
            out.append({"tenor": D, "row": r})
    return out


# ── the realised-vol cone ─────────────────────────────────────────────────────
def spx_close(store: Store) -> pd.Series:
    c = store.spx()["close"].astype(float)
    c = c[~c.index.duplicated(keep="last")]
    c.index = pd.to_datetime(c.index)
    return c.sort_index()


@dataclass
class ConeSample:
    key: str
    label: str
    short: str
    start: str                         # the sample's first close
    r: np.ndarray                      # its close-to-close log returns up to the session


def cone_samples(store: Store, end: str) -> list[ConeSample]:
    """Since 2010-01-04, and from the first close on or after the date three years
    before ``end``; both end at ``end``, so no later close enters."""
    close = spx_close(store)
    e = pd.Timestamp(end)
    out = []
    for key, label, short in CONE_SAMPLES:
        want = pd.Timestamp(CONE_START) if key == "since-2010" else e - pd.DateOffset(years=CONE_YEARS)
        c = close[(close.index >= want) & (close.index <= e)]
        out.append(ConeSample(key, label, short, str(c.index[0].date()), np.diff(np.log(c.to_numpy(float)))))
    return out


def rv_windows(r: np.ndarray, h: int) -> np.ndarray:
    """RV_h (%) of every window of h consecutive returns."""
    cs = np.concatenate([[0.0], np.cumsum(r * r)])
    return 100.0 * np.sqrt((cs[h:] - cs[:-h]) * 252.0 / h) if len(r) >= h else np.array([])


def cone_rank(r: np.ndarray, h: int, value: float) -> float:
    rv = rv_windows(r, h)
    return float(100.0 * np.mean(rv < value))


def cone_h(T_days: float) -> tuple[float, int]:
    """An expiry's sessions (T·252/365) and the RV horizon it is ranked at (rounded, ≥ 2)."""
    hs = T_days * 252.0 / 365.0
    return hs, max(2, round(hs))


# ── static arbitrage: the largest mid violations ──────────────────────────────
Leg = tuple[str, str, float]                  # expiry, right, strike


@dataclass
class Violation:
    """One arbitrage.largest_mid_violations row and the bad quote that explains it."""
    type: str
    legs: list[tuple[str, str, float, float]]   # expiry, right, strike, weight
    shortfall: float | None                       # index points
    executable: bool
    bad: Leg | None = None

    def width(self) -> float:
        """The widest gap between adjacent strikes of the row's legs."""
        ks = sorted({lg[2] for lg in self.legs})
        return max((b - a for a, b in itertools.pairwise(ks)), default=0.0)

    def held(self) -> set[Leg]:
        return {lg[:3] for lg in self.legs}


def violations(arb: dict) -> list[Violation]:
    """arbitrage.largest_mid_violations, largest shortfall first, each with the bad
    quote that explains it: a row's shortfall above its widest adjacent strike gap
    is no spread's price, so one leg is wrong; the leg named is the one most such
    rows share (then the larger summed |weight|, then the larger repair move), and
    every listed row holding a named leg carries it."""
    rows = [Violation(str(v.get("type") or ""),
                      [(str(lg["expiry"])[:10], str(lg["right"]), float(lg["strike"]), float(lg["weight"]))
                       for lg in v.get("legs") or []],
                      num(v.get("shortfall_index_points")), bool(v.get("executable")))
            for v in arb.get("largest_mid_violations") or []]
    rows.sort(key=lambda r: -(r.shortfall or 0.0))
    repaired = (arb.get("quotes_repaired_outside_spread") or {}).get("largest") or []
    moved = {(str(q.get("expiry"))[:10], str(q.get("right")), float(q["strike"])):
             abs(num(q.get("repair_dprice")) or 0.0) for q in repaired if num(q.get("strike")) is not None}
    left = [r for r in rows if (r.shortfall or 0.0) > r.width() > 0.0]
    bad: list[Leg] = []
    while left:
        score: dict[Leg, tuple[int, float]] = {}
        for r in left:
            for e, rt, k, w in r.legs:
                n, sw = score.get((e, rt, k), (0, 0.0))
                score[(e, rt, k)] = (n + 1, sw + abs(w))
        leg = max(score, key=lambda g: (*score[g], moved.get(g, 0.0), g))
        bad.append(leg)
        left = [r for r in left if leg not in r.held()]
    for r in rows:
        r.bad = next((g for g in bad if g in r.held()), None)
    return rows


# ── the backtest's β estimator on the DB's SPX chains ─────────────────────────
def _black76(F: float, K: float, T: float, DF: float, s: float, call: bool) -> float:
    sd = s * math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sd * sd) / sd
    d2 = d1 - sd
    if call:
        return DF * (F * norm.cdf(d1) - K * norm.cdf(d2))
    return DF * (K * norm.cdf(-d2) - F * norm.cdf(-d1))


def _black76_iv(price: float, F: float, K: float, T: float, DF: float, call: bool) -> float:
    lo = DF * max((F - K) if call else (K - F), 0.0)
    if not math.isfinite(price) or price <= lo + 1e-10 or price >= DF * (F if call else K):
        return math.nan
    try:
        return float(brentq(lambda s: _black76(F, K, T, DF, s, call) - price, 1e-4, 5.0, xtol=1e-10))
    except ValueError:
        return math.nan


def _atm_term(chain: pd.DataFrame, session: str, r: float) -> list[dict]:
    """Per expiry 7–75 days out (16:00 ET to 16:00 ET): the parity forward from the three
    strikes with the smallest |C − P|, and the Black-76 ATM vol interpolated in ln K
    between the strikes bracketing it (call and put IVs averaged per strike)."""
    out = []
    for e, quotes in chain.groupby("expiry"):
        T = (pd.Timestamp(str(e)[:10]) - pd.Timestamp(session)).days / DAYS
        if not BETA_DAYS[0] / DAYS <= T <= BETA_DAYS[1] / DAYS:
            continue
        g = quotes[(quotes["bid"] > 0) & (quotes["ask"] >= quotes["bid"])]
        mid = (g["bid"] + g["ask"]) / 2.0
        calls = mid[g["right"] == "C"].groupby(g.loc[g["right"] == "C", "strike"]).mean()
        puts = mid[g["right"] == "P"].groupby(g.loc[g["right"] == "P", "strike"]).mean()
        ks = calls.index.intersection(puts.index).sort_values()
        if len(ks) < max(BETA_N_FORWARD, 2):
            continue
        DF = math.exp(-r * T)
        diff = calls[ks] - puts[ks]
        near = diff.abs().sort_values().index[:BETA_N_FORWARD]
        F = float(np.mean([k + diff[k] / DF for k in near]))
        below, above = ks[ks <= F], ks[ks > F]
        if not len(below) or not len(above):
            continue
        k1, k2 = float(below.max()), float(above.min())
        ivs = []
        for k in (k1, k2):
            pair = [_black76_iv(float(calls[k]), F, k, T, DF, True),
                    _black76_iv(float(puts[k]), F, k, T, DF, False)]
            ivs.append(float(np.nanmean(pair)) if np.isfinite(pair).any() else math.nan)
        if not all(math.isfinite(v) for v in ivs):
            continue
        x1, x2 = math.log(k1 / F), math.log(k2 / F)
        out.append({"T": T, "F": F, "atm": ivs[0] + (ivs[1] - ivs[0]) * (0.0 - x1) / (x2 - x1)})
    return out


def spx_chains(store: Store) -> dict[str, pd.DataFrame]:
    """Every stored SPX chain (expiry, strike, right, bid, ask) by snapshot date: the
    market DB's, and the committed record's options/spx_<date>.parquet that the site
    loads into its copy of the DB (a date the DB holds keeps the DB's chain)."""
    cols = ["expiry", "strike", "right", "bid", "ask"]
    db = store.sql("SELECT snapshot_date, expiry, strike, right, bid, ask FROM options_snapshots "
                   "WHERE underlying='SPX'")
    out = {str(d)[:10]: g[cols] for d, g in db.groupby("snapshot_date")}
    for p in sorted((store.root / "options").glob("spx_*.parquet")):
        d = p.stem.removeprefix("spx_")
        if len(d) == 10 and d not in out:
            out[d] = store.parquet(f"options/{p.name}", columns=cols)
    return dict(sorted(out.items()))


def beta_by_session(store: Store) -> dict[str, dict]:
    """{session: {ratio, spot_gap}} of every stored SPX chain, matched to the last
    session on or before its snapshot date with a VIX and an SPX close."""
    out: dict[str, dict] = {}
    for d, chain in spx_chains(store).items():
        mk = store.sql("SELECT v.date, v.close AS vix, s.close AS spx FROM vix_term_structure v "
                       "JOIN spx_ohlcv s ON s.date = v.date WHERE v.ticker='^VIX' AND v.date <= ? "
                       "ORDER BY v.date DESC LIMIT 1", (d,))
        if mk.empty:
            continue
        session = str(mk["date"].iloc[0])[:10]
        rr = store.sql("SELECT rate FROM tbill_rates WHERE date <= ? ORDER BY date DESC LIMIT 1", (session,))
        r = float(rr["rate"].iloc[0]) if len(rr) else 0.04
        term = _atm_term(chain, session, r)
        T30 = 30.0 / DAYS
        lo = [t for t in term if t["T"] <= T30]
        hi = [t for t in term if t["T"] > T30]
        if not lo or not hi:
            continue
        a, b = max(lo, key=lambda t: t["T"]), min(hi, key=lambda t: t["T"])
        w1, w2 = a["atm"] ** 2 * a["T"], b["atm"] ** 2 * b["T"]
        atm30 = math.sqrt((w1 + (w2 - w1) * (T30 - a["T"]) / (b["T"] - a["T"])) / T30)
        nearest = min(term, key=lambda t: abs(t["T"] - T30))
        spot = nearest["F"] * math.exp(-(r - BETA_Q) * nearest["T"])
        out[session] = {"ratio": atm30 / (float(mk["vix"].iloc[0]) / 100.0),
                        "spot_gap": spot / float(mk["spx"].iloc[0]) - 1.0}
    return out


# ── the page's number formats ─────────────────────────────────────────────────
def session_range(values: list[float | None]) -> tuple[int, float, float] | None:
    """(n, lo, hi) over the finite values, None with fewer than two."""
    xs = [v for v in map(num, values) if v is not None]
    if len(xs) < 2:
        return None
    return len(xs), min(xs), max(xs)
