"""
chrome.py: the status strip and the market ticker every page carries.
"""
from __future__ import annotations

import math
import re

import pandas as pd

from .core import Audit, norm, text_of
from .data import Store, rms_vol, sessions_between, strike_at_delta, svi_total_variance

PANEL_SB, PANEL_TK = "status strip", "ticker"
CM_1M_DAYS = 30
INDEX_TENORS = ("^VIX9D", "^VIX", "^VIX3M", "^VIX6M")


def headline_fit_session(store: Store) -> str | None:
    """The newest stored joint fit of the current schema (the job stores only fits
    its gate accepted, so the newest model_version-2 file is the headline)."""
    best = None
    for p in sorted((store.root / "calibrations").glob("joint_cal_*.pkl")):
        cal = store.pickle(f"calibrations/{p.name}")
        if not isinstance(cal, dict) or cal.get("model_version") != 2 or cal.get("success") is False:
            continue
        s = str(cal.get("spot_date") or cal.get("as_of_date") or p.stem.removeprefix("joint_cal_"))[:10]
        best = s if best is None or s > best else best
    return best


def labels(store: Store) -> pd.Series:
    df = store.parquet("signals/regime_labels.parquet")
    s = df["regime"]
    s.index = pd.to_datetime(s.index).strftime("%Y-%m-%d")
    return s


def greeks_date(store: Store) -> str | None:
    import json

    import pyarrow.parquet as pq
    md = pq.read_schema(store.root / "greeks" / "greeks_surface.parquet").metadata or {}
    raw = md.get(b"greeks_provenance")
    if not raw:
        return None
    prov = json.loads(raw)
    d = prov.get("spot_date") or prov.get("as_of_date")
    return str(d)[:10] if d else None


def backtest_end(store: Store) -> str:
    nav = store.parquet("backtest/full_results.parquet", columns=["nav"])
    return pd.Timestamp(nav.index[-1]).strftime("%Y-%m-%d")


def sources(store: Store) -> dict[str, str | None]:
    spx = store.spx()
    surf = store.surface_sessions()
    return {"SPX": spx.index[-1], "VIX": store.vix_close().index[-1], "LABELS": labels(store).index[-1],
            "FIT": headline_fit_session(store), "GREEKS": greeks_date(store),
            "SURFACE": surf[-1] if surf else None, "BACKTEST": backtest_end(store)}


def run(a: Audit, store: Store) -> None:
    src = sources(store)
    ref = max(str(src["SPX"]), str(src["LABELS"]))
    # ── status strip (desktop items, then the phone list) ─────────────────────
    for item in a.soup.select("header.statusbar > span.sb-item[title], header.statusbar span.sb-item"):
        k = item.select_one(".sb-k")
        v = item.select_one(".sb-v")
        if k is None or v is None:
            continue
        key = text_of(k)
        if key in src:
            a.date(PANEL_SB, f"{key} date", v, src[key])
    for li in a.soup.select("header.statusbar ul.sb-list li"):
        key = text_of(li.select_one(".sb-k"))
        if key not in src:
            continue
        a.date(PANEL_SB, f"{key} date (phone list)", li.select_one(".sb-v"), src[key])
        age_el, when = li.select_one(".sb-age"), src[key]
        if age_el is not None and when:
            a.num(PANEL_SB, f"{key} age, NYSE sessions (phone list)", age_el,
                  float(sessions_between(when, ref)))
    view = a.soup.select_one("#freshness")
    if view is not None:
        a.equal(PANEL_SB, "VIEW data-date (newest session)", view.get("data-date"), ref)
        a.date(PANEL_SB, "VIEW as-of date", view, ref)
    # ── ticker ────────────────────────────────────────────────────────────────
    items = {text_of(t.select_one(".tk-label")): t for t in a.soup.select("div.ticker span.tk")}
    spx = store.spx()["close"].astype(float)
    vix = store.vix_close().astype(float)
    fam = store.vix_family()

    def tick(label: str):
        t = items.get(norm(label))
        if t is None:
            a.fail(PANEL_TK, label, "present", "missing")
        return t

    if (t := tick("SPX")) is not None:
        a.num(PANEL_TK, "SPX close", t.select_one(".tk-val"), spx.iloc[-1])
        a.num(PANEL_TK, "SPX change on the previous close, %", t.select_one(".tk-chg"),
              (spx.iloc[-1] / spx.iloc[-2] - 1.0) * 100.0)
    if (t := tick("VIX")) is not None:
        a.num(PANEL_TK, "VIX close", t.select_one(".tk-val"), vix.iloc[-1])
        a.num(PANEL_TK, "VIX change, index pts", t.select_one(".tk-chg"), vix.iloc[-1] - vix.iloc[-2])
    if (t := tick("VVIX")) is not None and "^VVIX" in fam:
        vv = fam["^VVIX"].dropna()
        a.num(PANEL_TK, "VVIX close", t.select_one(".tk-val"), vv.iloc[-1])
        a.num(PANEL_TK, "VVIX change, index pts", t.select_one(".tk-chg"), vv.iloc[-1] - vv.iloc[-2])
    full = fam[[c for c in INDEX_TENORS if c in fam]].dropna()
    if (t := tick("9D→6M")) is not None and len(full):
        row = full.iloc[-1]
        a.num(PANEL_TK, "VIX6M − VIX9D, index pts", t.select_one(".tk-val"), row["^VIX6M"] - row["^VIX9D"])
        a.text(PANEL_TK, "term-structure state", t.select_one(".tk-chg"),
               "contango" if row["^VIX6M"] > row["^VIX9D"] else "backwardation")
    if (t := tick("VIX/VIX3M")) is not None and len(full):
        row = full.iloc[-1]
        a.num(PANEL_TK, "VIX / VIX3M", t.select_one(".tk-val"), row["^VIX"] / row["^VIX3M"])
    r = (spx / spx.shift(1)).dropna().map(math.log)
    rv21 = rms_vol(r.iloc[-21:].to_numpy())
    if (t := tick("RV21")) is not None:
        a.num(PANEL_TK, "RV21, % (RMS of 21 daily log returns ×252)", t.select_one(".tk-val"), rv21)
    if (t := tick("VIX−RV21")) is not None:
        a.num(PANEL_TK, "VIX − RV21, vol pts", t.select_one(".tk-val"), vix.loc[spx.index[-1]] - rv21)
    if (t := tick("VIX 1Y PCTL")) is not None:
        hist = vix.loc[:spx.index[-1]]
        prior, today = hist.iloc[-253:-1], hist.iloc[-1]
        below = int((prior < today).sum())
        pct = 100.0 * below / len(prior)
        val = text_of(t.select_one(".tk-val"))
        if pct >= 100.0 - 1e-12 or pct <= 1e-12:
            a.text(PANEL_TK, "VIX 1Y percentile", val, "1y high" if pct > 50 else "1y low")
        else:
            a.num(PANEL_TK, "VIX 1Y percentile (strictly below, 252 prior sessions)", t.select_one(".tk-val"),
                  float(round(pct)))
        title = t.get("title") or ""
        m = re.search(r"(\d+) of the (\d+) sessions", text_of(title))
        a.equal(PANEL_TK, "VIX 1Y percentile: sessions below / window", m.groups() if m else None,
                (str(below), str(len(prior))))
    if (t := tick("RULE")) is not None:
        lab = labels(store)
        a.text(PANEL_TK, "rule label", t.select_one(".tk-val"), f"R{int(lab.iloc[-1])}")
        a.date(PANEL_TK, "rule label date", t.select_one(".tk-note"), lab.index[-1])
    if (t := tick("1M ATMF")) is not None:
        atmf, rr25 = cm_1m(store)
        vals = t.select(".tk-val")
        a.num(PANEL_TK, "1M ATMF (SVI-cal, 30 days), %", vals[0] if vals else None, atmf)
        a.num(PANEL_TK, "1M RR25 (25Δ call − 25Δ put), vol pts", vals[1] if len(vals) > 1 else None, rr25)
        sess = store.surface_sessions()
        a.date(PANEL_TK, "surface session", t.select_one(".tk-note"), sess[-1] if sess else None)


def cm_1m(store: Store, days: int = CM_1M_DAYS) -> tuple[float | None, float | None]:
    """ATMF and RR25 (vol pts) of the newest SVI-cal surface at ``days`` calendar days,
    linear in total variance at fixed forward delta between the bracketing expiries."""
    sess = store.surface_sessions()[-1]
    sj = store.json(f"surface/{sess}/surface.json")
    slices = sorted(sj["surfaces"]["svi_cal"]["params"]["slices"], key=lambda s: float(s["T"]))
    T = days / 365.0
    lo = max((s for s in slices if float(s["T"]) <= T + 1e-12), key=lambda s: float(s["T"]), default=None)
    hi = min((s for s in slices if float(s["T"]) >= T - 1e-12), key=lambda s: float(s["T"]), default=None)
    if lo is None or hi is None:
        return None, None

    def w_at(sl: dict, delta: float | None) -> float:
        k = 0.0 if delta is None else strike_at_delta(sl, delta)
        return float(svi_total_variance(k, sl))

    def vol(delta: float | None) -> float:
        T1, T2 = float(lo["T"]), float(hi["T"])
        x = 0.0 if T2 == T1 else (T - T1) / (T2 - T1)
        w = (1 - x) * w_at(lo, delta) + x * w_at(hi, delta)
        return 100.0 * math.sqrt(w / T)
    return vol(None), vol(0.25) - vol(-0.25)
