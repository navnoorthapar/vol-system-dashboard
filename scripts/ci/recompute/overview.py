"""
overview.py: every number the Overview page prints, recomputed and diffed.

The page's figures come from three kinds of source, and each is checked the way
its kind allows:
  * market series the page derives itself (the rule inputs, the term structure,
    the PDV forecast and its history, VIX/VIX3M, the VX curve, RV21): recomputed
    from vol_system.db by overview_math's own code;
  * values the page derives from stored fits and surfaces (counts, ranges,
    ages, the Heston and SVI ATM skews, the VX legs, the 1M surface row):
    recomputed from the stored parameters and slices;
  * artifact reads (a scorecard median, a bootstrap interval, a stored RMSE):
    read from the artifact and checked to be printed as stored.
Every element that exists only in some data states (the stale flag, the age
note, a findings item, an overlay, a bound chip) is decided from the data: if
the data say it belongs and the page lacks it, the check fails.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import pandas as pd
from bs4 import Tag

from . import overview_math as om
from .core import DATE, Audit, norm, numbers_in, rounds_to, text_of
from .data import Store, chain_words, sessions_between

P_BADGE, P_FIND, P_KEY = "page badge", "findings", "key results"
P_REG, P_TS, P_PDV = "regime", "term structure", "implied vs forecast"
P_VX, P_RATIO, P_FIT, P_PIPE = "VX curve", "VIX/VIX3M", "headline fit", "pipeline"
SURFACE_MODEL = "svi_cal"
TS_OVERLAYS = ((1, "1 session earlier"), (5, "5 sessions earlier"), (21, "21 sessions earlier"))
SHORT_END_DAYS = 30.0        # 'the short end': listed expiries at most 30 calendar days out
CM_DAYS = 30                 # the surface line's constant maturity (1M)
PCT_MIN_HISTORY = 252        # fewer prior sessions: the page hides the spread's percentile
SPARK_N = 60                 # sessions in the spread's sparkline
STRESS_DAY = "2020-03-16"    # the COVID stress test's date, as the page states it
SKILL_EX_YEAR = "2020"       # the walk-forward R² is also given without this year
SPEC_SIG = 6                 # chart specs carry floats at 6 significant figures
VIX2_NOTE = " (risk-neutral, contains the variance premium)"
FORECAST_NAMES = {"vix2": "VIX²", "pdv": "PDV", "vix_mz": "VIX (MZ)"}

Parts = list[Any]
Facts = dict[str, Any]


# ── checking helpers ─────────────────────────────────────────────────────────
def said(a: Audit, panel: str, label: str, el: Tag | None, parts: Parts, note: str = "") -> bool:
    """Every number and every date ``el`` prints, in order. ``parts`` holds recomputed
    values (one printed number each) and text the page prints as given (a date, a
    stored timestamp, a stated constant), whose numbers and dates are read the same
    way as the page's."""
    if el is None:
        a.fail(panel, label, "present", "missing", note)
        return False
    want_n: list[float | None] = []
    want_d: list[str] = []
    for p in parts:
        if isinstance(p, str):
            want_n += [v for v, _, _ in numbers_in(p)]
            want_d += [m.group(0) for m in DATE.finditer(norm(p))]
        else:
            want_n.append(p)
    ok = a.nums(panel, label, el, want_n, note=note)
    got_d = [m.group(0) for m in DATE.finditer(text_of(el))]
    return a.equal(panel, f"{label}: dates", got_d, want_d, note) and ok


def present(a: Audit, panel: str, label: str, el: Tag | None, expected: bool, note: str = "") -> bool:
    """An element that exists only in some data states is there exactly when the data say so."""
    return a.equal(panel, f"{label}: shown", el is not None, expected, note)


def inline_text(el: Tag | None) -> str:
    """The element's text as it reads (no space added around inline tags such as a
    no-wrap date), normalised."""
    return norm(el.get_text("")) if el is not None else ""


def says(a: Audit, panel: str, label: str, el: Tag | None, phrase: str, expected: bool) -> None:
    """A conditional phrase is in the element's text exactly when the data call for it."""
    a.equal(panel, f"{label}: {'says' if expected else 'does not say'} {phrase!r}",
            norm(phrase) in inline_text(el), expected)


def by_text(root: Tag | None, selector: str, needle: str) -> Tag | None:
    for el in (root.select(selector) if root is not None else []):
        if needle in text_of(el):
            return el
    return None


def attr(el: Tag | None, name: str) -> str:
    """An attribute's text ('' when the element or the attribute is absent)."""
    v = el.get(name) if el is not None else None
    return " ".join(v) if isinstance(v, list) else str(v or "")


def classes(el: Tag | None) -> list[str]:
    return attr(el, "class").split()


def kpi_lookup(root: Tag) -> Callable[..., Tag | None]:
    """A panel's KPI tiles by label (normalised text): kv(label) is the tile's value,
    kv(label, '.kpi-sub') its sub-line."""
    tiles = {text_of(k.select_one(".kpi-label")): k for k in root.select(".kpis .kpi")}

    def kv(label: str, sel: str = ".kpi-value") -> Tag | None:
        k = tiles.get(norm(label))
        return k.select_one(sel) if k is not None else None
    return kv


def spec_json(a: Audit, chart_id: str, panel: str) -> dict[str, Any]:
    el = a.need(f"script#{chart_id}-spec", panel)
    return json.loads(el.string) if el is not None and el.string else {}


def spec_value_ok(got: Any, want: float | None, decimals: int | None = None) -> bool:
    """One chart-spec value against the recomputed one: the spec rounds every float to
    6 significant figures (after rounding to ``decimals`` places, for a series the page
    rounds first), so the value passes when the recomputed one rounds to it."""
    if want is None or got is None:
        return want is None and got is None
    if isinstance(got, bool) or not isinstance(got, (int, float)):
        return False
    g = float(got)
    half = 0.0 if g == 0 else 0.5 * 10.0 ** (math.floor(math.log10(abs(g))) - SPEC_SIG + 1)
    if decimals is not None:
        half += 0.5 * 10.0 ** -decimals
    return abs(g - want) <= half + 1e-12 * max(1.0, abs(g))


def series_check(a: Audit, panel: str, label: str, got: Any, want: Sequence[float | None],
                 decimals: int | None = None) -> bool:
    """A chart series value by value (spec_value_ok); the first mismatch is shown."""
    vals = list(got) if isinstance(got, list) else []
    bad = [i for i, (g, w) in enumerate(zip(vals, want)) if not spec_value_ok(g, w, decimals)]
    first = f"; first off [{bad[0]}]: {vals[bad[0]]} vs {want[bad[0]]}" if bad else ""
    return a.equal(panel, label, f"{len(vals)} values, {min(len(vals), len(want)) - len(bad)} match{first}",
                   f"{len(want)} values, {len(want)} match")


def fmt_signed(x: float, spec: str = ".2f") -> str:
    """The page's signed number: '+1.23', '−1.23', or '0.00' for one that rounds to zero."""
    body = format(abs(x), spec)
    return body if float(body) == 0.0 else ("+" if x > 0 else "−") + body


# ── the store, read once ─────────────────────────────────────────────────────
def scorecards(store: Store, sessions: list[str]) -> list[dict[str, Any]]:
    """(session, model) out-of-sample rows of every scorecard, each session's in its
    ranking order (by out-of-sample median |model − mid| in half-spreads)."""
    out = []
    for s in sessions:
        if not (store.root / "surface" / s / "scorecard.json").exists():
            continue
        sc = store.json(f"surface/{s}/scorecard.json")
        rank = [r.get("model") for r in sc.get("ranking_by_oos_median_abs_hs") or [] if isinstance(r, dict)]
        order = {m["model"]: i for i, m in enumerate(sc["models"])}
        for m in sorted(sc["models"], key=lambda m: (rank.index(m["model"]) if m["model"] in rank
                                                     else len(rank) + order[m["model"]])):
            o = m["metrics"]["out_of_sample"]
            dg = m.get("dense_grid") or {}
            out.append({"session": s, "model": m["model"], "n": int(o["n"]),
                        "mhs": om.finite(o["median_abs_hs"]), "inside": om.finite(o["inside_spread_pct"]),
                        "butterfly": dg.get("butterfly_violations"),
                        "calendar": dg.get("calendar_violations")})
    return out


def facts(store: Store) -> Facts:
    """Everything the checks compare against that more than one panel reads, recomputed once."""
    f: Facts = {}
    spx = om.spx_closes(store)
    vix = om.vix_daily(store)
    labels = om.stored_labels(store)
    f.update(spx=spx, vix=vix, fam=store.vix_family(), labels=labels, spx_date=spx.index[-1],
             vix_date=vix.index[-1], label_date=labels.index[-1])
    f["ref_mkt"] = max(f["spx_date"], f["label_date"])            # an artifact's age is counted to here
    f["ref_page"] = max(f["spx_date"], f["vix_date"], f["label_date"])
    arc = om.calibration_archive(store)
    gated = sorted((r for r in arc if not r["legacy"] and r["ok"] and r["session"]),
                   key=lambda r: (r["session"], str(r["cal"].get("run_date") or ""), r["file"]))
    f.update(arc=arc, gated=gated, head=gated[-1] if gated else None)
    root = store.root / "surface"
    f["surf_sessions"] = sorted(p.name for p in root.iterdir() if p.is_dir() and DATE.fullmatch(p.name)
                                and (p / "residuals.parquet").exists())
    f["scores"] = scorecards(store, f["surf_sessions"])
    diag = store.root / "calibrations" / "diagnostics"
    f["diag_rows"] = (sorted(store.json("calibrations/diagnostics/summary.json")["rows"],
                             key=lambda r: r["session"]) if (diag / "summary.json").exists() else [])
    f["profiles"] = {p.stem: store.json(f"calibrations/diagnostics/{p.name}")
                     for p in sorted(diag.glob("*.json")) if DATE.fullmatch(p.stem)}
    f["inf"], f["bt"] = inference(store), backtest(store)
    f["skew"], f["legs"] = skew_compare(store, f), vx_legs(store, f)
    return f


def pdv_facts(f: Facts, store: Store) -> Facts:
    """The PDV forecast, its spread to VIX and that spread's own history."""
    pdv = om.load_inert(store.root / "pdv_model.pkl")
    h = int(om.attr(pdv, "horizon", 1))
    hl = [float(om.attr(pdv, k, d)) for k, d in (("halflife_fast", 5.0), ("halflife_slow", 60.0),
                                                 ("halflife_lev", 10.0))]
    burn = int(om.attr(pdv, "burn_in", 60))
    r = f["spx"]["close_return"].dropna()
    r.index = list(r.index)
    feats = om.pdv_state(r, *hl)
    fc = om.pdv_walk_forward(r, feats, h, int(om.attr(pdv, "train_min_days", 504)), burn)
    spread = (f["vix"] - 100.0 * fc).dropna()
    cur = spread.index[-1]
    y = om.forward_rv(r, h)
    both = pd.concat([y.rename("y"), fc.rename("p")], axis=1, join="inner").dropna()
    ex = both[[not str(d).startswith(SKILL_EX_YEAR) for d in both.index]]
    return {"h": h, "hl": hl, "date": cur, "fc": float(fc[cur]) * 100.0, "vix": float(f["vix"][cur]),
            "spread": float(spread.iloc[-1]), "hist": spread.iloc[:-1], "spark": spread.iloc[-SPARK_N:],
            "sigma1": float(feats.loc[cur, "sigma1"]) * 100.0,
            "sigma2": float(feats.loc[cur, "sigma2"]) * 100.0,
            "skill": {"r2": om.r_squared(both["y"].to_numpy(), both["p"].to_numpy()),
                      "r2_ex": om.r_squared(ex["y"].to_numpy(), ex["p"].to_numpy()), "n": len(both),
                      "start": both.index[0], "end": both.index[-1]},
            "stress": om.pdv_stress(r, feats, h, burn, STRESS_DAY)}


def inference(store: Store) -> dict[str, Any] | None:
    """The portfolio Sharpe's bootstrap interval and its settings (inference/summary.json)."""
    if not (store.root / "backtest" / "inference" / "summary.json").exists():
        return None
    s = store.json("backtest/inference/summary.json")
    b = ((s.get("bootstrap") or {}).get("portfolio") or {}).get("sharpe")
    if not isinstance(b, dict) or "ci95" not in b:
        return None
    sett = s.get("bootstrap_settings") or {}
    return {"summary": s, "sharpe": b, "level": om.finite(sett.get("level")) or 0.95, "settings": sett}


def backtest(store: Store) -> dict[str, Any] | None:
    """Window and cumulative return of the stored backtest (NAV end / initial capital − 1)."""
    pkl = store.root / "backtest" / "full_results.pkl"
    if not pkl.exists():
        return None
    st = om.load_inert(pkl)
    nav = store.parquet("backtest/full_results.parquet", columns=["nav"])["nav"].astype(float)
    cap = om.finite(st.get("initial_capital")) or float(nav.iloc[0])
    m = st.get("metrics") or {}
    return {"cum_ret": float(nav.iloc[-1]) / cap - 1.0,
            "cum_ret_stored": om.finite(m.get("cumulative_return")),
            "start": pd.Timestamp(nav.index[0]).strftime("%Y-%m-%d"),
            "end": pd.Timestamp(nav.index[-1]).strftime("%Y-%m-%d")}


# ── page header ──────────────────────────────────────────────────────────────
def check_badge(a: Audit, f: Facts, pdv: Facts) -> None:
    """The header badge: last close, rule label, headline fit (and its age in NYSE
    sessions when it is older than the newest session), PDV forecast date."""
    badge = a.need(".page-head .page-badge", P_BADGE)
    if badge is None:
        return
    spans = {norm(DATE.sub("", text_of(s))): s for s in badge.select("span.nowrap")}
    head = f["head"]["session"] if f["head"] else None
    for key, want in (("Last close", f["spx_date"]), ("Rule label", f["label_date"]),
                      ("Headline fit", head), ("PDV forecast", pdv["date"])):
        a.date(P_BADGE, f"{key} date", spans.get(key), want)
    age = sessions_between(head, f["ref_mkt"]) if head else 0
    el = badge.select_one("span[class^='age-']")
    if present(a, P_BADGE, "headline fit age", el, age > 0) and el is not None:
        a.num(P_BADGE, "headline fit age, NYSE sessions to the newest session", el, float(age))
        a.text(P_BADGE, "headline fit age wording", el, f"({om.plural(age, 'session')} old)")
        a.equal(P_BADGE, "headline fit age level", classes(el), [f"age-{om.stale_level(age)}"])


# ── shared by the findings and the key results ───────────────────────────────
def surface_headline(f: Facts) -> dict[str, Any] | None:
    """The quote-grade surface's scorecard row on the headline fit's session (else the newest)."""
    rows = [r for r in f["scores"] if r["model"] == SURFACE_MODEL]
    if not rows:
        return None
    sessions = sorted({r["session"] for r in rows})
    head = f["head"]["session"] if f["head"] else None
    sess = head if head in sessions else sessions[-1]
    same = [r for r in f["scores"] if r["session"] == sess]
    hkey = next((r["model"] for r in same if r["model"].lower().startswith("heston")), None)
    return {"sess": sess, "cur": next(r for r in rows if r["session"] == sess), "rows": rows,
            "n_sessions": len(sessions), "heston": next((r for r in same if r["model"] == hkey), None)}


def joint_result(f: Facts) -> dict[str, Any] | None:
    """Heston against rough Heston out of sample, from diagnostics/summary.json rows: a
    'lower' rough fit beats Heston by more than 0.01 vol pts; one at H ≥ ½ is Heston."""
    rows = [r for r in f["diag_rows"] if r.get("sigma_cap2_oos") is not None
            and r.get("joint_rough_oos") is not None]
    if not rows:
        return None
    head = f["head"]["session"] if f["head"] else None
    return {"cur": next((r for r in rows if r["session"] == head), rows[-1]), "n": len(rows),
            "lower": sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in rows),
            "same": sum(r.get("joint_rough_H") is not None and r["joint_rough_H"] >= 0.5 - 1e-9
                        for r in rows),
            "h": [r["sigma_cap2_oos"] for r in rows], "rough": [r["joint_rough_oos"] for r in rows]}


def skew_compare(store: Store, f: Facts) -> dict[str, Any] | None:
    """|Heston| / |SVI-cal| ATM skew at each listed expiry of the headline surface
    session, and the SVI-cal power-law slope in T over every stored surface."""
    per = []
    for s in f["surf_sessions"]:
        if (store.root / "surface" / s / "surface.json").exists():
            sl = om.svi_slices(store.json(f"surface/{s}/surface.json"))
            if sl:
                per.append({"session": s, "slices": sl, "slope": om.power_slope(
                    [float(x["T"]) for x in sl], [om.svi_atm_skew(x) for x in sl])})
    if not per or f["head"] is None:
        return None
    head = next((p for p in per if p["session"] == f["head"]["session"]), per[-1])
    params = f["head"]["cal"]["params"]
    rows = []
    for sl in head["slices"]:
        T = float(sl["T"])
        hs, ms = om.heston_atm_skew(params, T), om.svi_atm_skew(sl)
        if hs is not None and ms:
            rows.append({"T_days": T * om.DAYS_PER_YEAR, "heston": hs, "svi": ms, "ratio": abs(hs) / abs(ms)})
    slopes = [p["slope"] for p in per if p["slope"] is not None]
    if not rows or not slopes:
        return None
    return {"head": head["session"], "rows": rows, "slopes": slopes, "lo": min(r["ratio"] for r in rows),
            "hi": max(r["ratio"] for r in rows), "steeper": all(r["ratio"] > 1 for r in rows),
            "flatter": all(r["ratio"] < 1 for r in rows)}


def vx_rows(store: Store, session: str) -> pd.DataFrame:
    """A session's live monthly VX contracts, front first: price = settlement, else the
    last trade; days = calendar days to the contract's expiry (from the dates)."""
    df = store.sql("SELECT expiry_date, expiry_month, is_weekly, settle, close "
                   "FROM vix_futures_daily WHERE date = ?", (session,)).copy()
    df["expiry_date"] = df["expiry_date"].astype(str).str[:10]
    df["price"] = df["settle"].where(df["settle"].notna(), df["close"]).astype(float)
    df["days"] = [(pd.Timestamp(e) - pd.Timestamp(session)).days for e in df["expiry_date"]]
    df = df[(df["is_weekly"] == 0) & (df["price"] > 0) & (df["days"] > 0)]
    return df.sort_values(["days", "expiry_date"]).reset_index(drop=True)


def vx_last(store: Store, before: str = "9999-12-31") -> str | None:
    """The futures table's newest session (before ``before``, when given)."""
    d = store.sql("SELECT MAX(date) AS d FROM vix_futures_daily WHERE date < ?", (before,))["d"].iloc[0]
    return str(d)[:10] if d else None


def vx_legs(store: Store, f: Facts) -> dict[str, Any]:
    """Each gated fit's VX futures leg: its own session's monthly settlements against
    the fit's E^Q[VIX_T], their RMSE and how many settle above the model."""
    views = []
    for g in f["gated"]:
        rows = vx_rows(store, g["session"])
        leg = None
        if not rows.empty:
            model = np.array([om.heston_vix_future(g["cal"]["params"], d / om.DAYS_PER_YEAR)
                              for d in rows["days"]])
            gap = model - rows["price"].to_numpy(float)
            leg = {"rmse": float(np.sqrt(np.mean(gap ** 2))), "n": len(gap), "below": int((gap < 0).sum())}
        views.append({"session": g["session"], "leg": leg})
    return {"last": vx_last(store), "views": views}


def own_vvix(f: Facts) -> tuple[float, float] | None:
    """The headline fit's model VVIX against the ^VVIX close, from its own diagnostics
    file when that file has a σ profile and names the headline pickle."""
    head = f["head"]
    d = f["profiles"].get(head["session"]) if head else None
    sp = (d or {}).get("sigma_profile")
    if not d or not isinstance(sp, dict) or not isinstance(sp.get("rows"), list):
        return None
    sf = d.get("stored_fit") or {}
    if sf.get("file") != head["file"] or sf.get("vvix_model") is None or sf.get("vvix_market") is None:
        return None
    return float(sf["vvix_model"]), float(sf["vvix_market"])


def classifier_eval(store: Store) -> tuple[float, float] | None:
    p = store.root / "signals" / "regime_classifier.pkl"
    if not p.exists():
        return None
    ev = (om.attr(om.load_inert(p), "metadata_") or {}).get("evaluation") or {}
    acc, base = om.finite(ev.get("oos_acc")), om.finite(ev.get("persistence_acc"))
    return (acc, base) if acc is not None and base is not None else None


def fit_box(f: Facts) -> dict[str, tuple[float, float]]:
    """The production box the headline fit records (methods.optimizer.bounds), else the gate's."""
    cal = f["head"]["cal"] if f["head"] else {}
    b = (((cal.get("methods") or {}).get("optimizer") or {}).get("bounds")) or om.GATE["box"]
    return {k: (float(v[0]), float(v[1])) for k, v in dict(b).items()}


# ── findings at a glance ─────────────────────────────────────────────────────
Item = tuple[str, str, Parts, str]          # (link, lead, parts, note)


def works_items(a: Audit, store: Store, f: Facts) -> list[Item]:
    """'Works': the surface on held-out strikes, the VIX replication (its stored gap
    checked against replicated − official close), the gate, and Heston's ATM skew
    when it is within 10% of the market's at every listed expiry."""
    items: list[Item] = []
    sh = surface_headline(f)
    if sh and sh["cur"]["mhs"] is not None:
        c = sh["cur"]
        items.append(("surface.html#scorecard", "Arbitrage-free SVI surface",
                      [c["mhs"], c["inside"], sh["sess"]], "scorecard.json out_of_sample"))
        rp = f"surface/{sh['sess']}/vix_replication.json"
        rep = store.json(rp) if (store.root / rp).exists() else {}
        ix = (rep.get("indices") or {}).get("VIX") or {}
        gap = om.finite(ix.get("gap_vs_official"))
        if gap is not None:
            off = ix.get("official")
            off = om.finite(off.get("close") if isinstance(off, dict) else off)
            rep_v = om.finite(ix.get("replicated"))
            if rep_v is not None and off is not None:
                a.num(P_FIND, "VIX gap as stored (6 dp) = replicated − official close", f"{gap:.6f}",
                      rep_v - off)
            n_dep = om.vix_departures(rep)
            items.append(("surface.html#vix-replication", "VIX replicated from the stored chain",
                          [gap, sh["sess"], *([float(n_dep)] if n_dep else [])],
                          "gap from vix_replication.json; departures counted from its notes"))
    legacy = sum(r["legacy"] for r in f["arc"])
    cur = [r for r in f["arc"] if not r["legacy"]]
    n_ok = sum(r["ok"] for r in cur)
    items.append(("calibration.html#history", "The gate keeps broken-data fits off the page",
                  [float(legacy), float(len(cur)), *([float(n_ok)] if n_ok != len(cur) else [])],
                  "every joint_cal_*.pkl, legacy flag and gate restated"))
    sk = skew_item(f)
    return items + ([sk[1]] if sk and sk[0] else [])


def skew_item(f: Facts) -> tuple[bool, Item] | None:
    """Heston's ATM skew against the SVI-cal market's: (within 10% at every listed
    expiry, the item)."""
    sk = f["skew"]
    if not sk:
        return None
    f0 = sk["rows"][0]
    parts = [sk["lo"], sk["hi"], *([] if sk["steeper"] or sk["flatter"] else [float(len(sk["rows"]))]),
             sk["head"], abs(f0["heston"]), abs(f0["svi"]), f0["T_days"], min(sk["slopes"]),
             max(sk["slopes"]), float(len(sk["slopes"]))]
    works = sk["lo"] >= 0.9 and sk["hi"] <= 1.1
    lead = ("Heston's ATM skew matches the market's" if works else
            "Heston's ATM skew is steeper than the market's" if sk["steeper"] else
            "Heston's ATM skew is flatter than the market's" if sk["flatter"] else
            "Heston's ATM skew against the market's")
    return works, ("calibration.html#skew-term", lead, parts,
                   "Heston skew by Lewis pricing; SVI-cal skew from the slices")


def short_end(f: Facts) -> tuple[str, float, float] | None:
    """The headline fit's worst out-of-sample expiry within SHORT_END_DAYS: (expiry, days, RMSE)."""
    cal = f["head"]["cal"] if f["head"] else {}
    t_by = {str(c.get("expiry")): float(c["T"]) * om.DAYS_PER_YEAR for c in cal.get("carry_by_expiry") or []
            if om.finite(c.get("T")) is not None}
    oos = (cal.get("oos") or {}).get("spx_oos_rmse_by_expiry") or {}
    short = sorted((e, t_by[e], float(v)) for e, v in oos.items()
                   if om.finite(v) is not None and e in t_by and t_by[e] <= SHORT_END_DAYS)
    return max(short, key=lambda x: x[2]) if short else None


def fails_items(store: Store, f: Facts) -> list[Item]:
    """'Does not work': Heston's short end, its ATM skew (unless it matches), the
    backtest (when its stored cumulative return exists), the classifier."""
    items: list[Item] = []
    head = f["head"]
    se = short_end(f)
    if head and se:
        vv = own_vvix(f)
        items.append(("calibration.html#oos", "One-factor Heston at the short end",
                      [se[2], se[0], se[1], *(vv or ()), head["session"]],
                      f"worst expiry ≤ {SHORT_END_DAYS:.0f} days of the headline pickle"))
    sk = skew_item(f)
    if sk and not sk[0]:
        items.append(sk[1])
    inf, bt = f["inf"], f["bt"]
    if bt and inf and bt["cum_ret_stored"] is not None:
        b = inf["sharpe"]
        items.append(("research.html#inference", "The mark-to-model backtest loses",
                      [bt["cum_ret"] * 100, bt["start"][:4], bt["end"][:4], b["point"], inf["level"] * 100,
                       b["ci95"][0], b["ci95"][1]],
                      "cumulative return recomputed from the stored NAV; Sharpe and CI from summary.json"))
    ev = classifier_eval(store)
    if ev:
        items.append(("research.html#research-classifier", "The ML regime classifier",
                      [ev[0] * 100, ev[1] * 100], "regime_classifier.pkl metadata_.evaluation"))
    return items


def next_items(store: Store, f: Facts) -> list[Item]:
    """'Next': the forward test, rough Heston, the VX leg, a wider σ box."""
    led = om.forward_ledger(store)
    items: list[Item] = [("research.html#forward", "Forward test at executable prices",
                          [float(led["n"] if led else 0)], "forward/ledger_*.json, hash chain checked")]
    jr = joint_result(f)
    if jr:
        on_floor = "H" in (jr["cur"].get("joint_rough_on_bound") or [])
        items.append(("calibration.html#rough-vs-heston", "Rough Heston in production",
                      [float(jr["lower"]), float(jr["n"]),
                       *([jr["cur"]["joint_rough_H"]] if on_floor else []), jr["cur"]["session"]],
                      "diagnostics/summary.json rows"))
    legs = f["legs"]
    with_leg = [v["leg"] for v in legs["views"] if v["leg"]]
    if with_leg:
        rm = [g["rmse"] for g in with_leg]
        items.append(("calibration.html#vixleg", "VX futures in the joint objective",
                      [min(rm), *([max(rm)] if len(with_leg) > 1 else []), float(len(with_leg)),
                       float(len(legs["views"])), float(sum(g["below"] for g in with_leg)),
                       float(sum(g["n"] for g in with_leg)), vx_empty(f)],
                      "E^Q[VIX_T] of each gated fit against its own session's settlements"))
    items.append(("calibration.html#sigma-tradeoff", "A wider production σ box", [fit_box(f)["sigma"][1]],
                  "σ ceiling of the box the headline fit records"))
    return items


def vx_empty(f: Facts) -> str:
    """Why the headline fit has no VX leg ('' when it has one)."""
    head = f["head"]["session"] if f["head"] else None
    hv = next((v for v in f["legs"]["views"] if v["session"] == head), None)
    if head is None or hv is None or hv["leg"] is not None:
        return ""
    last = f["legs"]["last"]
    if last is None:
        return f"The market database has no VX futures table; none stored for {head}."
    if head > last:
        return f"VX futures table ends {last}; none stored for {head}."
    return f"No monthly VX settlements are stored for {head}."


def check_findings(a: Audit, store: Store, f: Facts) -> None:
    """The three findings columns: which items each holds (decided from the data),
    each item's lead, and every number and date of each item."""
    items = {"works": works_items(a, store, f), "fails": fails_items(store, f), "next": next_items(store, f)}
    for key, want in items.items():
        col = a.need(f"#find-list-{key}", P_FIND)
        head_el = a.need(f"h3#find-{key}", P_FIND)
        if head_el is not None:
            a.equal(P_FIND, f"{key}: data-count", attr(head_el, "data-count"), str(len(want)))
        if col is None:
            continue
        lis = col.select("li.find-item")
        a.equal(P_FIND, f"{key}: items (by link)", [attr(li.a, "href") for li in lis], [w[0] for w in want])
        for href, lead, parts, note in want:
            li = next((x for x in lis if attr(x.a, "href") == href), None)
            if li is not None:
                a.text(P_FIND, f"{href}: lead", li.a, lead)
            said(a, P_FIND, href, li, parts, note)
    check_finding_words(a, f)


def check_finding_words(a: Audit, f: Facts) -> None:
    """The wording the data decide inside the items (σ on its cap, H on its floor, why
    the headline fit has no VX leg), and the backtest's stored cumulative return
    against the one its NAV gives."""
    head = f["head"]
    li = by_text(a.one("#find-list-fails"), "li", "One-factor Heston")
    if li is not None and head:
        pinned = head["cal"].get("pinned_params") or []
        says(a, P_FIND, "short end", li, "σ on its box cap", "sigma" in pinned)
    jr = joint_result(f)
    li = by_text(a.one("#find-list-next"), "li", "Rough Heston")
    if li is not None and jr:
        on_floor = "H" in (jr["cur"].get("joint_rough_on_bound") or [])
        a.text(P_FIND, "rough Heston: H wording", li,
               "but H sits on its 0.02 floor" if on_floor else "but H is weakly identified", contains=True)
    li = by_text(a.one("#find-list-next"), "li", "VX futures in the joint objective")
    if li is not None and vx_empty(f):
        a.text(P_FIND, "VX leg: why the headline fit has none", inline_text(li), vx_empty(f), contains=True)
    bt = f["bt"]
    if bt and bt["cum_ret_stored"] is not None:
        a.num(P_FIND, "backtest cumulative return: stored metric (9 dp) = NAV end / capital − 1",
              f"{bt['cum_ret_stored']:.9f}", bt["cum_ret"])


# ── key results ──────────────────────────────────────────────────────────────
def check_key_results(a: Audit, store: Store, f: Facts) -> None:
    """The three result cards: link, value, detail, interval or range, source path."""
    root = a.need("#key-results", P_KEY)
    if root is None:
        return
    cards = root.select("a.result")
    sh = surface_headline(f)
    jr = joint_result(f)
    want = [("surface.html#scorecard" if sh and sh["cur"]["mhs"] is not None else "surface.html"),
            ("calibration.html#rough-vs-heston" if jr else "calibration.html#oos"), "research.html#inference"]
    a.equal(P_KEY, "cards (by link)", [attr(c, "href") for c in cards], want)
    card = {attr(c, "href"): c for c in cards}
    if sh and sh["cur"]["mhs"] is not None:
        surface_card(a, store, sh, card.get("surface.html#scorecard"))
    if jr:
        joint_card(a, f, jr, card.get("calibration.html#rough-vs-heston"))
    if f["inf"] and f["bt"]:
        backtest_card(a, f["inf"], f["bt"], card.get("research.html#inference"))


def part(c: Tag | None, sel: str) -> Tag | None:
    return c.select_one(sel) if c is not None else None


def surface_card(a: Audit, store: Store, sh: dict[str, Any], c: Tag | None) -> None:
    """The surface's held-out median in half-spreads, its detail (quotes, inside share,
    dense-grid violations, Heston beside it, the quotes' time) and the range over sessions."""
    cur, hes = sh["cur"], sh["heston"]
    rel = f"surface/{sh['sess']}/surface.json"
    sj = store.json(rel) if (store.root / rel).exists() else {}
    rule = str((sj.get("fit_set") or {}).get("rule") or "")
    inputs = sj.get("inputs") or {}
    held = "alternate strikes held out" if rule.startswith("alternate strikes") else "held-out strikes"
    a.text(P_KEY, "surface card: what is held out", part(c, ".result-k"),
           f"Arbitrage-free SVI surface · {held}")
    said(a, P_KEY, "surface card: median |model − mid|, half-spreads", part(c, ".result-v"), [cur["mhs"]])
    words = chain_words(inputs)
    detail: Parts = [float(cur["n"]), sh["sess"], cur["inside"]]
    if cur["butterfly"] is not None and cur["calendar"] is not None:
        detail += [float(cur["butterfly"]), float(cur["calendar"])]
    if hes:
        detail += [hes["mhs"], hes["inside"]]
    detail += [f" {words[0]}, {words[1]}."] if words else []
    el = part(c, ".result-d")
    said(a, P_KEY, "surface card: detail", el, detail, "scorecard.json, surface.json inputs")
    if words:
        says(a, P_KEY, "surface card: detail", el, f"{words[0]}, {words[1]}.", True)
    rng = part(c, ".kpi-ci.kpi-range")
    if present(a, P_KEY, "surface card: range over sessions", rng, sh["n_sessions"] > 1) and rng is not None:
        mh = [r["mhs"] for r in sh["rows"] if r["mhs"] is not None]
        ins = [r["inside"] for r in sh["rows"] if r["inside"] is not None]
        said(a, P_KEY, "surface card: range over sessions", rng,
             [float(sh["n_sessions"]), min(mh), max(mh), min(ins), max(ins)])
    a.text(P_KEY, "surface card: source", part(c, ".src"), f"data_store/surface/{sh['sess']}/scorecard.json")


def joint_card(a: Audit, f: Facts, jr: dict[str, Any], c: Tag | None) -> None:
    """Heston's out-of-sample RMSE on the headline session, rough Heston beside it, the
    counts over diagnosed sessions and the ranges (with κ's curvature SE)."""
    cur, head = jr["cur"], f["head"]
    said(a, P_KEY, "joint fit card: Heston OOS RMSE", part(c, ".result-v"), [cur["sigma_cap2_oos"]])
    n_oos = (head["cal"].get("oos") or {}).get("n_spx_oos") if head else None
    detail: Parts = [float(n_oos)] if head and head["session"] == cur["session"] and n_oos is not None else []
    detail += [cur["session"], cur["joint_rough_oos"]]
    detail += [cur["joint_rough_H"]] if cur.get("joint_rough_H") is not None else []
    detail += [float(jr["lower"]), float(jr["n"]), float(jr["same"])]
    el = part(c, ".result-d")
    said(a, P_KEY, "joint fit card: detail", el, detail, "diagnostics/summary.json")
    says(a, P_KEY, "joint fit card", el, "(σ on its box cap)", "sigma" in (cur.get("on_bound") or []))
    says(a, P_KEY, "joint fit card", el, "on its floor", "H" in (cur.get("joint_rough_on_bound") or []))
    rse = om.finite((cur.get("relative_standard_errors") or {}).get("kappa"))
    said(a, P_KEY, "joint fit card: ranges", part(c, ".kpi-ci"),
         [float(jr["n"]), min(jr["h"]), max(jr["h"]), min(jr["rough"]), max(jr["rough"]),
          *([rse * 100] if rse is not None else [])])
    a.text(P_KEY, "joint fit card: source", part(c, ".src"),
           "data_store/calibrations/diagnostics/summary.json")


def backtest_card(a: Audit, inf: dict[str, Any], bt: dict[str, Any], c: Tag | None) -> None:
    """The portfolio Sharpe with its bootstrap interval, the window, the SPA p-value and
    the S3 timing null."""
    s, b = inf["summary"], inf["sharpe"]
    said(a, P_KEY, "backtest card: portfolio Sharpe", part(c, ".result-v"), [b["point"]])
    circ = ((((s.get("nulls") or {}).get("sleeves") or {}).get("s3") or {}).get("circular") or {}).get(
        "total_pnl") or {}
    spa = ((s.get("multiple_testing") or {}).get("spa_published") or {}).get("p_consistent")
    detail: Parts = [bt["start"][:4], bt["end"][:4], *([float(spa)] if spa is not None else [])]
    holm = next(((float(m.group(1)), v) for k, v in circ.items()
                 if (m := re.fullmatch(r"p_holm_(\d+)_sleeves", k))), None)
    if circ.get("p_value") is not None:
        detail += [float(circ["n_paths"]), float(circ["p_value"]), *(holm or (None, None))]
    said(a, P_KEY, "backtest card: detail", part(c, ".result-d"), detail, "inference/summary.json")
    sett = inf["settings"]
    reps = om.finite(sett.get("reps"))
    ci = part(c, ".kpi-ci")
    said(a, P_KEY, "backtest card: bootstrap interval", ci,
         [inf["level"] * 100, b["ci95"][0], b["ci95"][1], *([reps] if reps is not None else [])])
    a.text(P_KEY, "backtest card: bootstrap method", ci, f"({sett.get('method') or 'bootstrap'}, ",
           contains=True)
    a.text(P_KEY, "backtest card: source", part(c, ".src"), "data_store/backtest/inference/summary.json")


# ── regime ───────────────────────────────────────────────────────────────────
REGIME_NAMES = ("LONG GAMMA", "SHORT GAMMA", "VOMMA ACTIVE")


def check_regime(a: Audit, store: Store, f: Facts) -> None:
    """The rule label, its inputs recomputed from the market DB, the gate distance,
    the stored label of the inputs' session and the label-lag monitor chip."""
    sec = a.need("section#state-regime", P_REG)
    if sec is None:
        return
    lab = int(f["labels"].iloc[-1])
    a.date(P_REG, "panel date (rule label)", sec.select_one(".panel-meta"), f["label_date"])
    lag = sessions_between(f["spx_date"], f["label_date"])
    status = "ok" if lag <= 0 else ("watch" if lag <= 1 else "breach")
    chip = sec.select_one("a.chip[data-monitored='check-label-lag']")
    a.text(P_REG, "label-lag chip status (DB sessions behind the labels)", chip,
           f"· {'OK' if status == 'ok' else status.upper()}", contains=True)
    a.equal(P_REG, "label-lag chip class", f"chip-{status}" in classes(chip), True)
    kv = sec.select("div.kpi-value > span")
    a.text(P_REG, "rule label", kv[-1] if kv else None, f"R{lab} · {REGIME_NAMES[lab]}")
    a.date(P_REG, "label date", by_text(sec, "p.note", "Label as of"), f["label_date"])
    window, quant = f"{om.RULE['gate_window']}-day", float(round(om.RULE["gate_q"] * 100))
    said(a, P_REG, "rule definitions (window, quantile)", by_text(sec, "details dd", "rolling"),
         [window, quant])
    ri = om.rule_inputs(store)
    box = sec.select_one("#rule-inputs")
    if not present(a, P_REG, "rule inputs", box, ri is not None) or box is None or ri is None:
        return
    differs = f["label_date"] != ri["date"]
    said(a, P_REG, "rule inputs: session", box.select_one("p.note"),
         [ri["date"], *([f["label_date"]] if differs else [])])
    dd = {text_of(t): t.find_next_sibling("dd") for t in box.select("dl dt")}
    rv_dt = next((t for t in box.select("dl dt") if text_of(t).startswith("rv")), None)
    a.text(P_REG, "rv₂₀ label: the rule's 20-session window", rv_dt, "rv₂₀ (20-session)")
    said(a, P_REG, "rv₂₀, % (20 sessions, stored log returns)", dd.get("rv₂₀ (20-session)"),
         [ri["rv20"] * 100])
    said(a, P_REG, "VIX (vix_term_structure ^VIX)", dd.get("VIX"),
         [ri["vix"] * 100] if ri["vix"] is not None else [])
    said(a, P_REG, "VVIX on the session (not carried)", dd.get("VVIX"),
         [ri["vvix"]] if ri["vvix"] is not None else [])
    said(a, P_REG, "R2 gate (rolling 0.80 quantile of 252 VVIX prints)", dd.get("R2 gate"),
         [ri["gate"], window, quant])
    dist = dd.get("Gate distance")
    if present(a, P_REG, "gate distance", dist, ri["vvix"] is not None) and ri["vvix"] is not None:
        said(a, P_REG, "gate distance", dist, [ri["vvix"], ri["gate"], ri["vvix"] - ri["gate"]])
        side = "above" if ri["vvix"] > ri["gate"] else ("below" if ri["vvix"] < ri["gate"] else "at the gate")
        a.text(P_REG, "gate side", dist, side, contains=True)
    stored = f["labels"].get(ri["date"])
    want = f"R{ri['label']}"
    if stored is not None:
        want += f"; stored label for {ri['date']}: R{int(stored)}"
        want += " (differs)" if int(stored) != ri["label"] else ""
    a.text(P_REG, "rule gives / stored label", dd.get("Rule gives"), want)


# ── VIX term structure ───────────────────────────────────────────────────────
def check_term_structure(a: Audit, f: Facts) -> None:
    """The newest full VIX-index curve (tenors with a print in the last 90 days),
    its 1, 5 and 21 sessions earlier, the note's slope from the printed levels, the
    data table cell by cell and the chart series."""
    sec = a.need("section#state-ts", P_TS)
    if sec is None:
        return
    fam = f["fam"]
    by_days = sorted((c for c in om.VIX_TENOR_DAYS if c in fam), key=lambda c: om.VIX_TENOR_DAYS[c])
    last = pd.Timestamp(fam.index[-1])
    recent = fam[[pd.Timestamp(d) >= last - pd.Timedelta(days=90) for d in fam.index]]
    curve = [c for c in by_days if recent[c].notna().any()]
    full = fam[curve].dropna()
    if len(curve) < 2 or full.empty:
        present(a, P_TS, "term-structure chart", sec.select_one("#chart-ts"), False)
        return
    ts_date = full.index[-1]
    days = [om.VIX_TENOR_DAYS[c] for c in curve]
    vals = [float(full.iloc[-1][c]) for c in curve]
    all4 = fam[by_days].dropna()
    overlays = []
    if len(all4) and all4.index[-1] == ts_date:
        for k, what in TS_OVERLAYS:
            if len(all4) > k:
                d = all4.index[-1 - k]
                overlays.append({"date": d, "name": f"{d} ({what})",
                                 "days": [om.VIX_TENOR_DAYS[c] for c in by_days],
                                 "vals": [float(all4.iloc[-1 - k][c]) for c in by_days]})
    a.date(P_TS, "panel date (newest full curve)", sec.select_one(".panel-meta"), ts_date)
    a.equal(P_TS, "chart title (aria-label)", attr(sec.select_one("div#chart-ts"), "aria-label"),
            f"VIX index term structure on {ts_date}, with earlier sessions")
    names = [f"{ts_date} (latest)"] + [o["name"] for o in overlays]
    btns = sec.select("button[data-legend-for='chart-ts']")
    a.equal(P_TS, "legend series", [text_of(b) for b in btns], names)
    for b, nm in zip(btns, names):
        said(a, P_TS, f"legend {nm}", b, [nm])
    a.text(P_TS, "chart description", sec.select_one("#chart-ts-desc"),
           "^VIX9D, ^VIX, ^VIX3M and ^VIX6M on one session, and dashed on "
           + ", ".join(o["name"] for o in overlays) + "; tenor on a linear axis, straight segments.")
    tbl = sec.select_one("#chart-ts-data table")
    said(a, P_TS, "data table caption", tbl.caption if tbl else None, [ts_date])
    said(a, P_TS, "data table header", tbl.thead if tbl else None, [ts_date] + [o["date"] for o in overlays])
    body = tbl.select("tbody tr") if tbl else []
    a.equal(P_TS, "data table rows (tenors)", [text_of(tr.td) for tr in body], [f"{d}d" for d in days])
    for i, tr in enumerate(body[:len(days)]):
        said(a, P_TS, f"data table {days[i]}d", tr,
             [vals[i]] + [o["vals"][i] for o in overlays if i < len(o["vals"])])
    note = by_text(sec, "p.note", "→")
    tail = [f["vix_date"]] if ts_date != f["vix_date"] else []
    said(a, P_TS, "note: ends and slope", note,
         [vals[0], vals[-1], round(vals[-1], 2) - round(vals[0], 2), *tail],
         "slope from the two printed (2-decimal) levels")
    a.text(P_TS, "note: regime", note, "contango" if vals[-1] > vals[0] else "backwardation", contains=True)
    spec = spec_json(a, "chart-ts", P_TS)
    series = spec.get("series") or []
    a.equal(P_TS, "chart series (sessions)", len(series), 1 + len(overlays))
    for s, (nm, xs, ys) in zip(series, [(names[0], days, vals)] + [(o["name"], o["days"], o["vals"])
                                                                   for o in overlays]):
        pts = s.get("data") or []
        a.equal(P_TS, f"chart {nm}: tenors", [p[0] for p in pts], xs)
        series_check(a, P_TS, f"chart {nm}: levels (2 dp)", [p[1] for p in pts], ys, decimals=2)


# ── implied vs forecast vol ──────────────────────────────────────────────────
def check_pdv(a: Audit, store: Store, f: Facts, pdv: Facts) -> None:
    """VIX against the PDV walk-forward forecast: KPIs, stale flag, forecast ranks,
    the spread's percentile among its own history, the sparkline and VIX − RV21."""
    sec = a.need("section#state-pdv", P_PDV)
    if sec is None:
        return
    d = pdv["date"]
    a.date(P_PDV, "panel date (forecast origin)", sec.select_one(".panel-meta"), d)
    age = sessions_between(d, f["ref_page"])
    stale = age >= 1
    flag = sec.select_one("#pdv-stale-flag")
    if present(a, P_PDV, "stale-forecast flag", flag, stale) and stale:
        said(a, P_PDV, "stale flag", flag, [d, float(age), f["ref_page"]])
    kv = kpi_lookup(sec)
    said(a, P_PDV, "VIX (vix_daily) on the forecast date", kv("VIX"), [pdv["vix"]])
    said(a, P_PDV, "PDV forecast, % (purged walk-forward OLS)", kv("PDV forecast"), [pdv["fc"]])
    a.text(P_PDV, "forecast horizon", kv("PDV forecast", ".kpi-sub"),
           f"next {pdv['h']}-day realised" if pdv["h"] != 1 else "next-day |r|, annualised")
    said(a, P_PDV, "VIX − PDV, vol pts", kv("VIX − PDV"), [pdv["spread"]],
         "rounded once, from the unrounded spread")
    sp = round(pdv["spread"], 2)
    a.equal(P_PDV, "VIX − PDV tone (sign of the printed spread)", classes(kv("VIX − PDV")),
            ["kpi-value", "pos" if sp > 0 else ("neg" if sp < 0 else "flat")])
    check_forecast_rank(a, store, sec)
    check_spread_rank(a, sec, pdv, stale)
    spark = [round(float(x), 2) for x in pdv["spark"]]
    svg = sec.select_one("svg.spark")
    a.text(P_PDV, "sparkline label", attr(svg, "aria-label"),
           f"VIX minus PDV forecast, last {len(spark)} sessions to {d}: from {fmt_signed(spark[0])} to "
           f"{fmt_signed(spark[-1])} vol pts")
    check_sparkline(a, svg, spark)
    said(a, P_PDV, "sparkline note", by_text(sec, "p.note", "sessions of the spread"), [float(len(spark))])
    rv = vix_rv(f)
    said(a, P_PDV, "VIX − RV21 line", sec.select_one("#vix-rv-line"),
         [rv["vix"] - rv["rv"], rv["date"], rv["vix"], "21 sessions", rv["rv"]],
         "RV21 = RMS of 21 close-to-close returns")
    check_pdv_method(a, sec, pdv)


def check_spread_rank(a: Audit, sec: Tag, pdv: Facts, stale: bool) -> None:
    """Today's spread against its prior sessions: percent strictly below, worded as the
    page words a percentile, with the rich/cheap reading only when the forecast is fresh."""
    hist, cur = pdv["hist"], pdv["spread"]
    n = len(hist)
    pct = float((hist < cur).mean() * 100.0) if n else 0.0
    rk = om.rank_words(pct) if n >= PCT_MIN_HISTORY else None
    rich = (by_text(sec, "p.note", "spread's usual range")
            or by_text(sec, "p.note", "versus this spread's own history"))
    present(a, P_PDV, "rich/cheap interpretation", rich, not stale and rk is not None)
    if rich is not None and rk is not None:
        a.text(P_PDV, "rich/cheap interpretation", rich,
               f"Implied vol is rich versus this spread's own history ({rk['tail']})." if pct >= 80 else
               f"Implied vol is cheap versus this spread's own history ({rk['tail']})." if pct <= 20 else
               "Within this spread's usual range.")
    line = sec.select_one("#pdv-percentile-line")
    if present(a, P_PDV, "percentile line", line, rk is not None) and rk is not None:
        a.text(P_PDV, "percentile wording", sec.select_one("#pdv-percentile"), rk["big"])
        said(a, P_PDV, "percentile of the spread among prior sessions", line,
             [*([float(rk["k"])] if rk["k"] else []), float(n), hist.index[0], hist.index[-1], hist.mean()])
    else:
        said(a, P_PDV, "percentile hidden: valid history points", by_text(sec, "p.note", "Percentile hidden"),
             [float(n), float(PCT_MIN_HISTORY)])


def check_pdv_method(a: Audit, sec: Tag, pdv: Facts) -> None:
    """The method note: the forecast target, σ₁ and σ₂ on the forecast date, the
    walk-forward R² (pooled and ex-2020) and the ex-ante COVID stress test."""
    meth = sec.select_one("details.method .prose")
    paras = meth.select("p") if meth else []
    h, sk, st = pdv["h"], pdv["skill"], pdv["stress"]
    said(a, P_PDV, "method: horizon", paras[0] if paras else None,
         [float(h)] if h != 1 else ["|r|·√252, about 0.8σ"])
    said(a, P_PDV, "method: σ₁, σ₂ and walk-forward R²", paras[1] if len(paras) > 1 else None,
         [pdv["hl"][0], pdv["sigma1"], pdv["hl"][1], pdv["sigma2"], sk["r2"], float(SKILL_EX_YEAR),
          sk["r2_ex"], float(sk["n"]), sk["start"], sk["end"]],
         "R² of the forecast on its own target, pooled and ex-2020")
    stress_p = by_text(meth, "p", "COVID stress test")
    if present(a, P_PDV, "COVID stress test", stress_p, st is not None) and st is not None:
        horizon = ([float(st["n_window"]), STRESS_DAY, st["window_end"]] if h != 1
                   else [STRESS_DAY, "|r|·√252"])
        said(a, P_PDV, "COVID stress test", stress_p,
             [STRESS_DAY, st["origin"], float(st["n_train"]), st["forecast"], *horizon, st["actual"],
              st["error"]])


def check_sparkline(a: Audit, svg: Tag | None, ys: list[float]) -> None:
    """The polyline's points (2 dp): x spread evenly over [2, 98], y the 2-dp spread
    mapped onto [34, 2] over its range with zero included; the zero line at y(0)."""
    if svg is None:
        a.fail(P_PDV, "sparkline", "present", "missing")
        return
    lo, hi = min(*ys, 0.0), max(*ys, 0.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    n = len(ys) - 1

    def sy(y: float) -> float:
        return 34.0 - 32.0 * (y - lo) / (hi - lo)
    pts = [tuple(map(float, p.split(","))) for p in attr(svg.select_one("polyline"), "points").split()]
    want = [(2.0 + 96.0 * i / n, sy(y)) for i, y in enumerate(ys)]
    good = sum(rounds_to(wx, px, 2) and rounds_to(wy, py, 2) for (px, py), (wx, wy) in zip(pts, want))
    a.equal(P_PDV, "sparkline points (x evenly spaced, y = spread scaled with 0 in range)",
            f"{good} of {len(pts)} match", f"{len(want)} of {len(want)} match")
    y0 = attr(svg.select_one("line.spark-zero"), "y1")
    a.num(P_PDV, "sparkline zero line y", y0 or None, sy(0.0))


def vix_rv(f: Facts) -> dict[str, Any]:
    """VIX on the newest session with a trailing RV21 against that RV21 (RMS of the 21
    close-to-close log returns to it, ×√252)."""
    r = f["spx"]["close_return"]
    rv = np.sqrt((r ** 2).rolling(21).mean() * om.ANNUALISE) * 100.0
    fr = pd.DataFrame({"vix": f["vix"]}).join(rv.rename("rv")).dropna()
    return {"date": fr.index[-1], "vix": float(fr["vix"].iloc[-1]), "rv": float(fr["rv"].iloc[-1])}


def check_forecast_rank(a: Audit, store: Store, sec: Tag) -> None:
    """PDV's rank among the evaluated forecasters on each loss (recomputed from the
    stored losses), the window, 90% MCS membership and the best model."""
    p = store.root / "backtest" / "inference" / "forecast_eval.json"
    fe = store.json("backtest/inference/forecast_eval.json") if p.exists() else {}
    bw = fe.get("backtest_window") or {}
    labels = {"mse": f"{bw.get('horizon')}-day MSE", "qlike": "QLIKE"}
    losses = [k for k in labels if "pdv" in ((bw.get("losses") or {}).get(k) or {})]
    el = sec.select_one("#pdv-forecast-rank")
    if not present(a, P_PDV, "forecast rank", el, bool(losses)) or not losses:
        return
    parts: Parts = []
    best, inc = [], []
    for k in losses:
        lv = bw["losses"][k]
        parts += [float(1 + sum(v < lv["pdv"] for v in lv.values())), float(len(lv)), labels[k]]
        best.append(tuple(sorted(m for m in lv if lv[m] == min(lv.values()))))
        mcs = ((bw.get("mcs") or {}).get(k) or {}).get("90")
        members = mcs if isinstance(mcs, list) else (mcs or {}).get("included") if mcs else None
        inc.append("pdv" in members if isinstance(members, list) else None)
    parts += [bw.get("start"), bw.get("end"), "90%"]
    said(a, P_PDV, "forecast rank (from forecast_eval losses)", el, parts)
    if len(set(inc)) == 1 and inc[0] is not None:
        mcs_txt = f"model confidence set: {'yes' if inc[0] else 'no'}{' on both' if len(losses) > 1 else ''}"
    else:
        mcs_txt = "model confidence set: " + ", ".join(
            f"{'yes' if i else 'no' if i is not None else 'not recorded'} on {labels[k]}"
            for k, i in zip(losses, inc))
    a.text(P_PDV, "90% MCS membership", el, mcs_txt, contains=True)
    if len(set(best)) == 1:
        names = ", ".join(FORECAST_NAMES.get(m, m.upper()) + (VIX2_NOTE if m == "vix2" else "")
                          for m in best[0])
        where = "both" if len(losses) > 1 else labels[losses[0]]
        a.text(P_PDV, "best forecaster", el, f"best on {where}: {names}.", contains=True)


# ── VX futures curve ─────────────────────────────────────────────────────────
def check_vx_curve(a: Audit, store: Store, f: Facts) -> None:
    """The VX monthly curve on the futures table's own newest date, its age in NYSE
    sessions, the session before (dashed), M1 → M2, the table and the chart series."""
    sec = a.one("section#vx-curve")
    last = vx_last(store)
    cur = vx_rows(store, last) if last else pd.DataFrame()
    if not present(a, P_VX, "VX curve panel", sec, not cur.empty) or sec is None or last is None or cur.empty:
        return
    prev_date = vx_last(store, last)
    prev = vx_rows(store, prev_date) if prev_date else pd.DataFrame()
    pv = dict(zip(prev["expiry_date"], prev["price"])) if not prev.empty else {}
    dashed = [prev_date] if prev_date and not prev.empty else []
    a.date(P_VX, "panel date (futures table's newest)", sec.select_one(".panel-meta"), last)
    age = sessions_between(last, f["ref_mkt"])
    el = sec.select_one("#vx-curve-age")
    said(a, P_VX, "age of the futures table", el, [last, float(age), f["ref_mkt"]])
    a.equal(P_VX, "age level", classes(el.select_one("span[class^='age-']") if el else None),
            [f"age-{om.stale_level(age)}"])
    a.equal(P_VX, "chart title (aria-label)", attr(sec.select_one("div#chart-vx-curve"), "aria-label"),
            f"VX monthly futures settlements on {last}" + "".join(f", with {d}" for d in dashed))
    names = [f"{last} settlements"] + [f"{d} settlements" for d in dashed]
    btns = sec.select("button[data-legend-for='chart-vx-curve']")
    a.equal(P_VX, "legend series", [text_of(b) for b in btns], names)
    for b, nm in zip(btns, names):
        said(a, P_VX, f"legend {nm}", b, [nm])
    s = cur["price"].to_numpy(float)
    m1m2 = (s[1] / s[0] - 1.0) * 100.0 if len(s) >= 2 else None
    said(a, P_VX, "chart description", sec.select_one("#chart-vx-curve-desc"),
         [float(len(s)), last, *([m1m2] if m1m2 is not None else [])])
    tbl = sec.select_one("#chart-vx-curve-data table")
    said(a, P_VX, "table caption", tbl.caption if tbl else None, [last, *dashed])
    said(a, P_VX, "table header", tbl.thead if tbl else None, [last, *dashed])
    body = tbl.select("tbody tr") if tbl else []
    a.equal(P_VX, "table rows (contracts)", len(body), len(cur))
    for tr, (_, r) in zip(body, cur.iterrows()):
        parts: Parts = [str(r["expiry_month"]), r["expiry_date"], float(r["days"]), r["price"]]
        if dashed:
            p0 = pv.get(r["expiry_date"])
            parts += [p0, r["price"] - p0] if p0 is not None else []
        said(a, P_VX, f"table {r['expiry_date']}", tr, parts, "days = calendar days to the final settlement")
    said(a, P_VX, "M1 → M2", sec.select_one("#vx-m1m2"),
         [*([m1m2, s[0], s[1]] if m1m2 is not None else []), "VX2 / VX1 − 1", *dashed])
    spec = spec_json(a, "chart-vx-curve", P_VX)
    ser = spec.get("series") or []
    a.equal(P_VX, "chart categories (final settlement dates)", (spec.get("x") or {}).get("categories"),
            list(cur["expiry_date"]))
    series_check(a, P_VX, "chart series: settlements", ser[0].get("data") if ser else None, list(s))
    if dashed:
        series_check(a, P_VX, "chart series: previous settlements",
                     ser[1].get("data") if len(ser) > 1 else None, [pv.get(e) for e in cur["expiry_date"]])


# ── VIX / VIX3M ──────────────────────────────────────────────────────────────
def check_vix_ratio(a: Audit, f: Facts) -> None:
    """^VIX/^VIX3M on every session with both: share above 1, today's value and its
    percentile among all prior sessions, the yearly table, the chart series, its
    dates and the shaded runs above 1."""
    sec = a.one("section#vix-ratio")
    fam = f["fam"]
    both = {"^VIX", "^VIX3M"} <= set(fam)
    ratio = (fam["^VIX"] / fam["^VIX3M"]).dropna() if both else pd.Series(dtype=float)
    if not present(a, P_RATIO, "VIX/VIX3M panel", sec, len(ratio) >= 2) or sec is None or len(ratio) < 2:
        return
    above = ratio > 1.0
    today = float(ratio.iloc[-1])
    prior = ratio.iloc[:-1]
    rk = om.rank_words(float((prior < today).mean() * 100.0))
    first, last, n = ratio.index[0], ratio.index[-1], len(ratio)
    share = float(above.mean() * 100.0)
    said(a, P_RATIO, "panel title (first session)", sec.select_one(".panel-title"), [first])
    a.date(P_RATIO, "panel date", sec.select_one(".panel-meta"), last)
    a.equal(P_RATIO, "chart title (aria-label)", attr(sec.select_one("div#chart-vix-ratio"), "aria-label"),
            f"VIX / VIX3M, {first} to {last}, shaded above 1 (backwardation)")
    said(a, P_RATIO, "chart description", sec.select_one("#chart-vix-ratio-desc"),
         ["above 1 (the 30-day index over the 3-month one)", share, float(n), today])
    line = sec.select_one("#vix-ratio-line")
    said(a, P_RATIO, "summary line", line,
         ["Above 1 (backwardation: the 30-day index over the 3-month one)", share, float(n), today,
          *([float(rk["k"])] if rk["k"] else []), float(len(prior)), first,
          *([rk["tail"]] if rk["tail"] else [])])
    strong = line.select("strong") if line else []
    a.text(P_RATIO, "percentile wording", strong[-1] if strong else None,
           rk["big"] + ("" if rk["record"] else " percentile"))
    tbl = sec.select_one("#chart-vix-ratio-data table")
    a.text(P_RATIO, "table caption", tbl.caption if tbl else None,
           "VIX / VIX3M by calendar year; the CSV has every session")
    a.text(P_RATIO, "table header (the threshold is 1)", tbl.thead if tbl else None,
           "Year Sessions Above 1 (%) Mean")
    body = tbl.select("tbody tr") if tbl else []
    groups = list(ratio.groupby([d[:4] for d in ratio.index]))
    a.equal(P_RATIO, "table rows (years)", [text_of(tr.td) for tr in body], [y for y, _ in groups])
    for tr, (y, g) in zip(body, groups):
        said(a, P_RATIO, f"table {y}", tr,
             [float(y), float(len(g)), float((g > 1).mean() * 100), float(g.mean())])
    spec = spec_json(a, "chart-vix-ratio", P_RATIO)
    series_check(a, P_RATIO, "chart series: every session's ratio (3 dp)",
                 ((spec.get("series") or [{}])[0]).get("data"), [float(x) for x in ratio], decimals=3)
    xs = spec.get("xsDates") or {}
    steps = [(pd.Timestamp(b) - pd.Timestamp(a_)).days for a_, b in zip(ratio.index[:-1], ratio.index[1:])]
    a.equal(P_RATIO, "chart dates (start and day steps)", (xs.get("start"), xs.get("steps")), (first, steps))
    runs, start = [], None
    idx = list(ratio.index)
    for i, on in enumerate(above.to_numpy()):
        if on and start is None:
            start = i
        if start is not None and (not on or i == len(idx) - 1):
            runs.append([idx[start], idx[i if on else i - 1]])
            start = None
    got = [[r.get("from"), r.get("to")] for r in (spec.get("marks") or {}).get("areas") or []]
    a.equal(P_RATIO, "shaded runs above 1", got, runs)


# ── headline fit ─────────────────────────────────────────────────────────────
PARAMS = (("kappa", "κ", ".3f"), ("theta", "θ", ".4f"), ("sigma", "σ", ".3f"), ("rho", "ρ", ".3f"),
          ("v0", "v₀", ".4f"))
STRIKES_PER_EXPIRY = 9       # the calibrator's anchors per expiry: bounds an unrecorded expiry count
BOUND_REL_TOL = 1e-4         # 'at a bound': within this fraction of the box width from its edge


def check_fit(a: Audit, store: Store, f: Facts) -> None:
    """The headline joint Heston fit: its stored losses, counts and model VVIX, the
    five parameters and their bound chips, the 1M surface row and the constraints line."""
    sec = a.need("section#state-fit", P_FIT)
    head = f["head"]
    if sec is None or head is None:
        return
    cal = head["cal"]
    ll = cal.get("leg_losses") or cal.get("losses") or {}
    a.date(P_FIT, "panel date (fit session)", sec.select_one(".panel-meta"), head["session"])
    kv = kpi_lookup(sec)
    n_exp = cal.get("n_spx_expiries")
    n_exp = n_exp if n_exp is not None else int(cal["n_spx_options"]) // STRIKES_PER_EXPIRY
    n_oos = (cal.get("oos") or {}).get("n_spx_oos")
    said(a, P_FIT, "SPX RMSE in sample (stored)", kv("SPX RMSE in sample"), [ll.get("spx_iv_rmse")])
    said(a, P_FIT, "anchors and expiries", kv("SPX RMSE in sample", ".kpi-sub"),
         [float(cal["n_spx_options"]), float(n_exp)])
    said(a, P_FIT, "SPX RMSE out of sample (stored)", kv("Out of sample"), [ll.get("spx_oos_iv_rmse")])
    said(a, P_FIT, "out-of-sample quotes", kv("Out of sample", ".kpi-sub"),
         [float(n_oos) if n_oos is not None else None])
    said(a, P_FIT, "put-wing OOS RMSE (stored)", kv("Put wing OOS"), [ll.get("spx_oos_put_wing_rmse")])
    said(a, P_FIT, "VIX index RMSE (stored)", kv("VIX indices"),
         [ll.get("vix_index_rmse", ll.get("vix_futures_rmse"))])
    vv = own_vvix(f)
    if present(a, P_FIT, "VVIX model / market", kv("VVIX model / market"), vv is not None) and vv:
        said(a, P_FIT, "VVIX model / market (diagnostics stored_fit)", kv("VVIX model / market"),
             [vv[0], vv[1]])
        said(a, P_FIT, "VVIX tenor", kv("VVIX model / market", ".kpi-sub"), ["30-day"])
    box = fit_box(f)
    pinned = cal.get("pinned_params") or []
    pnote = by_text(sec, "p.note", "κ ")
    spans = pnote.select("span.nowrap") if pnote else []
    a.equal(P_FIT, "parameter tiles", len(spans), len(PARAMS))
    for sp, (key, sym, spec) in zip(spans, PARAMS):
        v = float(cal["params"][key])
        lo, hi = box[key]
        tol = BOUND_REL_TOL * ((hi - lo) or 1.0)
        side = ("upper" if abs(v - hi) <= tol else "lower" if abs(v - lo) <= tol
                else "bound" if key in pinned else None)
        chip = f" at {side if side != 'bound' else ''} bound" if side else ""
        a.text(P_FIT, f"{key} (and its bound chip)", sp, f"{sym} {format(v, spec)}{chip}")
        a.num(P_FIT, f"{key} value", sp, v)
    check_surface_line(a, store, f, sec)
    check_constraints(a, f, sec, box)


def check_surface_line(a: Audit, store: Store, f: Facts, sec: Tag) -> None:
    """The newest surface's 30-day constant-maturity row: ATMF, RR25, BF25 (None when
    a 25Δ strike is beyond the quoted strikes) and the surface's age."""
    sess = f["surf_sessions"][-1] if f["surf_sessions"] else None
    cm = None
    if sess and (store.root / "surface" / sess / "surface.json").exists():
        res = store.parquet(f"surface/{sess}/residuals.parquet", columns=["model", "expiry", "k"])
        kr = om.quoted_k_ranges(res)
        rows = [om.desk_row(s, kr.get(str(s["expiry"])[:10]))
                for s in om.svi_slices(store.json(f"surface/{sess}/surface.json"))]
        cm = om.constant_maturity(rows, CM_DAYS)
    el = sec.select_one("#surface-line")
    if not present(a, P_FIT, "surface line", el, cm is not None) or cm is None or el is None or sess is None:
        return
    age = sessions_between(sess, f["ref_mkt"])
    said(a, P_FIT, "1M ATMF, RR25, BF25 (SVI-cal, 30 days) and age", el,
         [f"at {CM_DAYS} days", cm["atmf"], *[x for x in (cm["rr25"], cm["bf25"]) if x is not None], sess,
          *([float(age)] if age else [])],
         "linear in total variance at fixed forward delta between the bracketing expiries")
    says(a, P_FIT, "surface line", el, "RR25 —", cm["rr25"] is None)
    a.text(P_FIT, "surface age wording", el,
           "the newest session" if age == 0 else f"{om.plural(age, 'session')} old", contains=True)


def sigma_ridge(f: Facts) -> dict[str, Any] | None:
    """κ along the σ profile of the headline fit's diagnostics: the lowest and highest
    caps, κ there, whether κ rises with the cap (Spearman ≥ 0.8 and the top κ at least
    1.5 × the fit's), the profile's κ ceiling and κ's relative SE (the pickle's
    identifiability, else the file's)."""
    head = f["head"]
    cal = head["cal"]
    d = f["profiles"].get(head["session"]) or {}
    ident = (cal["identifiability"] if isinstance(cal.get("identifiability"), dict)
             else d.get("identifiability"))
    sp = d["sigma_profile"] if isinstance(d.get("sigma_profile"), dict) else {}
    rows = sorted((r for r in sp.get("rows") or [] if om.finite(r.get("sigma_cap")) is not None
                   and om.finite((r.get("params") or {}).get("kappa")) is not None),
                  key=lambda r: float(r["sigma_cap"]))
    if not isinstance(ident, dict) or len(rows) < 3:
        return None
    caps = pd.Series([float(r["sigma_cap"]) for r in rows]).rank()
    kap = pd.Series([float(r["params"]["kappa"]) for r in rows]).rank()
    rho = float(np.corrcoef(caps, kap)[0, 1])
    k_fit = float(cal["params"]["kappa"])
    lo, top = rows[0], rows[-1]
    return {"moves": rho >= 0.8 and float(top["params"]["kappa"]) >= 1.5 * k_fit, "kappa_fit": k_fit,
            "cap_lo": float(lo["sigma_cap"]), "kappa_lo": float(lo["params"]["kappa"]),
            "cap_top": float(top["sigma_cap"]), "kappa_top": float(top["params"]["kappa"]),
            "kappa_max": om.finite(sp.get("kappa_max")),
            "rse": om.finite((ident.get("relative_standard_errors") or {}).get("kappa"))}


def check_constraints(a: Audit, f: Facts, sec: Tag, box: dict[str, tuple[float, float]]) -> None:
    """Active constraints, the Feller margin 2κθ − σ² and its state, and the refit in
    a wider box (diagnostics/summary.json) with the box it ran in."""
    el = sec.select_one("#fit-constraints")
    if el is None:
        a.fail(P_FIT, "constraints line", "present", "missing")
        return
    cal = f["head"]["cal"]
    p = cal["params"]
    ac = cal.get("active_constraints") or []
    rg = sigma_ridge(f)
    margin = 2.0 * float(p["kappa"]) * float(p["theta"]) - float(p["sigma"]) ** 2
    state = "PASS" if margin > om.FELLER_TOL else ("BINDING" if margin >= -om.FELLER_TOL else "VIOLATED")
    parts: Parts = []
    if ac:
        parts += ["; ".join(ac), float(5 - len(cal.get("pinned_params") or []))]
        if rg and rg["moves"]:
            parts += [rg["kappa_lo"], rg["cap_lo"], rg["kappa_fit"], rg["kappa_top"], rg["cap_top"],
                      *([rg["rse"] * 100] if rg["rse"] is not None else [])]
    parts.append(margin)
    row = next((r for r in f["diag_rows"] if r["session"] == f["head"]["session"]), None)
    fs = row if row and row.get("sigma_free") is not None else None
    present(a, P_FIT, "wider-box refit", el.select_one("#free-sigma"), fs is not None)
    wide = bool(rg and rg["kappa_max"] is not None)
    if fs:
        if rg and wide:
            parts += [rg["cap_top"], rg["kappa_max"], box["kappa"][1]]
        parts += [fs["sigma_free"], fs["kappa_free"],
                  *([fs["sigma_free_loss_rel"] * 100] if fs.get("sigma_free_loss_rel") is not None else []),
                  fs["sigma_cap2_oos"], fs["sigma_free_oos"]]
        a.text(P_FIT, "refit box wording", el,
               "Refitted in a wider box" if wide else "Refitted without the σ cap", contains=True)
    said(a, P_FIT, "constraints, Feller margin, wider-box refit", el, parts,
         "Feller margin recomputed from the stored parameters")
    a.text(P_FIT, "Feller state", el, f"Feller {state} (margin", contains=True)
    a.text(P_FIT, "bound statement", el, "On a bound:" if ac else "No parameter on a bound.", contains=True)


# ── pipeline ─────────────────────────────────────────────────────────────────
def check_pipeline(a: Audit, store: Store, f: Facts) -> None:
    """What the repository holds: the newest Cboe book per underlying (counts
    recomputed from its stored quotes), accepted fits, surface reports, diagnostics,
    the forward ledger and the backtest window."""
    tbl = a.need("table#pipeline", P_PIPE)
    if tbl is None:
        return
    rows = {text_of(tr.th): tr.td for tr in tbl.select("tbody tr") if tr.th is not None}
    newest: dict[str, dict[str, Any]] = {}
    for jp in sorted((store.root / "options" / "cboe").glob("*.json")):
        rec = store.json(f"options/cboe/{jp.name}")
        und = str(rec.get("underlying") or jp.name.split("_")[0]).upper()
        key = (str(rec.get("session")), str(rec.get("quote_time")))
        if und not in newest or key > (str(newest[und].get("session")), str(newest[und].get("quote_time"))):
            newest[und] = {**rec, "path": jp}
    parts: Parts = []
    for und in sorted(newest):
        rec = newest[und]
        book = rec["path"].with_suffix(".parquet")
        if book.exists():
            q = pd.read_parquet(book, columns=["expiry", "bid", "ask"])
            two_sided = (q["bid"] > 0) & (q["ask"] >= q["bid"])
            n, two, exp = len(q), int(two_sided.sum()), int(q["expiry"].nunique())
        else:
            n, two, exp = rec["rows"], rec["two_sided_rows"], rec["expiries"]
        parts += [rec["session"], float(n), float(two), float(exp),
                  str(rec.get("quote_time") or "").replace("T", " ").replace("Z", " UTC")]
    books = rows.get("Cboe option books")
    said(a, P_PIPE, "Cboe option books (counts from the stored quotes)", books, parts,
         "rows; two-sided = bid > 0 and ask ≥ bid; expiries")
    for und, seg in zip(sorted(newest), text_of(books).split("; ")):
        a.equal(P_PIPE, f"{und} book: overnight flag (book_state gth)", seg.endswith("(overnight book)"),
                newest[und].get("book_state") == "gth")
    said(a, P_PIPE, "accepted joint fits", rows.get("Accepted joint fits"),
         [float(len(f["gated"])), f"model_version {om.GATE['model_version']}",
          *([f["head"]["session"]] if f["head"] else [])])
    ss = f["surf_sessions"]
    said(a, P_PIPE, "surface reports", rows.get("Surface reports"),
         [float(len(ss)), *([ss[0], ss[-1]] if ss else [])])
    said(a, P_PIPE, "calibration diagnostics", rows.get("Calibration diagnostics"),
         [float(len(f["diag_rows"]))])
    led = om.forward_ledger(store)
    if led:
        said(a, P_PIPE, "forward ledger", rows.get("Forward ledger"),
             [float(led["n"]), led["first"], led["last"]])
    else:
        a.text(P_PIPE, "forward ledger: no session claimed", rows.get("Forward ledger"), "no entry yet:",
               contains=True)
    nav = store.parquet("backtest/full_results.parquet", columns=["nav"])
    said(a, P_PIPE, "backtest window", rows.get("Backtest window"),
         [pd.Timestamp(nav.index[-1]).strftime("%Y-%m-%d")])


# ── the page ─────────────────────────────────────────────────────────────────
def run(a: Audit, store: Store) -> None:
    f = facts(store)
    pdv = pdv_facts(f, store)
    check_badge(a, f, pdv)
    check_findings(a, store, f)
    check_key_results(a, store, f)
    check_regime(a, store, f)
    check_term_structure(a, f)
    check_pdv(a, store, f, pdv)
    check_vx_curve(a, store, f)
    check_vix_ratio(a, f)
    check_fit(a, store, f)
    check_pipeline(a, store, f)
