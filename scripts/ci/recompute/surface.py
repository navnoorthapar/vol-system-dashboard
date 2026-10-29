"""
surface.py: every number the Surface page prints, against this audit's own
recompute (surface_math.py) from data_store/surface, the market DB and the
committed option chain.

The page derives most of its numbers itself (the SVI-cal surface in desk
coordinates, constant maturities, the density and tails, the realised-vol
cone, heatmap medians, the β estimator): those are rebuilt from the slices, the
per-quote residual table and the DB. What the page only reads from an artifact
(the scorecard's metrics, the arbitrage screen's counts, the VIX replication)
is checked against that artifact and, where the per-quote table or the DB
holds the same number, against that too.
"""
from __future__ import annotations

import json
import math
import re
from collections.abc import Callable, Iterable, Sequence, Sized
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from bs4 import Tag

from . import surface_math as M
from .chrome import labels as regime_labels
from .core import DATE, Audit, Check, norm, numbers_in, rounds_to, text_of
from .data import Store, chain_words, sessions_between

STALE_WARN = 2                      # the page's amber banner from this age (NYSE sessions) on
STALE_ERROR = 5                     # ... and the red one
RESID_RAMP = 5.0                    # the residual heatmaps' colour scale: ±5 half-spreads
WITHIN_HS = 2.0                     # the scorecard's 'within 2 half-spreads' column
READOUT_PAGE = 10                   # the quote readout's Page Up / Down step (quotes)
LV_SHOWN = 8                        # largest violations listed in the panel; the rest below
GAP_SPLIT_TOL = 1e-6                # index points: a VIX gap split's parts add up to the stored gap
FAR_WING_IN_LINE = 0.005            # index points: a far wing closer than this prices 'in line'
SPEC_SIG = 6                        # chart specs carry 6 significant figures (desk_format.round_sig)
SPEC_ATOL = 1e-12                   # ... and float noise of a value that is zero
CONE_SIG = 4                        # the cone's series: 4 significant figures (app.vol_cone_view)
FULL_SIG = 10                       # a spec value stored unrounded
PACK_MID, PACK_SPREAD, PACK_MODEL = 1e5, 1e8, 1e4  # the packed quotes' units per vol pt
QUAD_NOISE = 1e-10                  # two quadratures of ∫p agree to about this
SUPER = str.maketrans("⁰¹²³⁴⁵⁶⁷⁸⁹⁻", "0123456789-")


# ── generic checks ────────────────────────────────────────────────────────────
def cell(a: Audit, panel: str, label: str, el: Tag | str | None, want: float | None,
         scale: float = 1.0, note: str = "") -> None:
    """A number, or '—' where the recompute has none."""
    if want is None or not math.isfinite(want):
        a.text(panel, label, el, "—", note=note)
    else:
        a.num(panel, label, el, want, scale=scale, note=note)


def nums(a: Audit, panel: str, label: str, el: Tag | None, want: Iterable[float | None],
         dates: Sequence[str] = (), note: str = "") -> None:
    """Every number of an element in order, and its dates in order."""
    a.nums(panel, label, el, want, note=note)
    if dates or DATE.search(text_of(el)):
        a.equal(panel, f"{label}: dates", ["-".join(m) for m in DATE.findall(text_of(el))], list(dates))


def days_nums(a: Audit, panel: str, label: str, el: Tag | str | None, want: Sequence[float],
              dates: Sequence[str] = ()) -> None:
    """Like nums, for a text whose day counts carry a bare unit ('2026-11-06 · 11d'),
    which the page's number grammar does not read as a number."""
    text_nums(a, panel, label, re.sub(r"(?<=\d)d\b", " d", text_of(el)), want, dates)
    if isinstance(el, Tag):
        a.cover(el)


def text_nums(a: Audit, panel: str, label: str, txt: str, want: Sequence[float | None],
              dates: Sequence[str] = ()) -> None:
    """Every number of a text in order, and its dates (nums on a text, not an element)."""
    got = numbers_in(txt)
    if len(got) != len(want):
        a.fail(panel, label, f"{len(want)} numbers", f"{len(got)} numbers: {txt[:160]}")
    else:
        for i, w in enumerate(want):
            a.num(panel, f"{label} [{i}]", txt, w, index=i)
    if dates or DATE.search(txt):
        a.equal(panel, f"{label}: dates", ["-".join(m) for m in DATE.findall(txt)], list(dates))


def statics(a: Audit, panel: str, label: str, el: Tag | None, allowed: Iterable[float],
            dates: Sequence[str] = ()) -> None:
    """Methodology prose: every number is one of the constants the recompute uses
    (horizons, thresholds, tenors), and its dates are the session's."""
    if el is None:
        a.fail(panel, label, "present", "missing")
        return
    ok = list(allowed)
    stray = [raw for v, _, raw in numbers_in(text_of(el)) if not any(abs(v - c) < 1e-9 for c in ok)]
    a.equal(panel, f"{label}: constants", ", ".join(stray) or "none", "none",
            note="numbers outside the recompute's constants")
    a.equal(panel, f"{label}: dates", ["-".join(m) for m in DATE.findall(text_of(el))], list(dates))
    a.cover(el)


def spec_ok(printed: Any, want: Any, sig: int = SPEC_SIG, decimals: int | None = None) -> bool:
    """A chart-spec value against the recompute: the spec rounds to ``sig`` significant
    figures (and, where the page rounds first, to ``decimals``), so the recomputed value
    must round to it at that precision; None (a gap) matches only None."""
    if printed is None or want is None:
        return printed is None and want is None
    p, w = float(printed), float(want)
    if not math.isfinite(w):
        return False
    half = 0.0 if p == 0.0 else 0.5 * 10.0 ** (math.floor(math.log10(abs(p))) - sig + 1)
    if decimals is not None:
        half = max(half, 0.5 * 10.0 ** -decimals)
    return abs(w - p) <= half * (1.0 + 1e-9) + SPEC_ATOL


def approx(a: Audit, panel: str, label: str, got: Any, want: Any, sig: int = SPEC_SIG,
           decimals: int | None = None) -> bool:
    """A chart spec's numbers (a list, or a list of [x, y] points, None for a gap) against
    the recompute, each at the precision the spec carries (spec_ok)."""
    def flat(x: Any) -> list:
        return [y for v in x for y in flat(v)] if isinstance(x, (list, tuple)) else [x]
    g, w = flat(got), flat(want)
    bad = [f"{x} vs {y}" for x, y in zip(g, w) if not spec_ok(x, y, sig, decimals)]
    ok = len(g) == len(w) and not bad
    first = f"{len(g)} vs {len(w)} values" if len(g) != len(w) else (bad[0] if bad else "")
    prec = f"{sig} significant figures" + (f" or {decimals} decimals" if decimals is not None else "")
    printed = f"{len(g)} values" + (f"; first off: {first}" if first else "")
    a.checks.append(Check(a.page, panel, label, printed, f"{len(w)} values at {prec}", ok))
    return ok


def gfmt(a: Audit, panel: str, label: str, el: Tag | str | None, want: float | None,
         floor: float = 0.0) -> None:
    """A value printed with 4 decimals from 0.001 up, else as 'm.m × 10⁻ⁿ'. Below
    ``floor`` a printed and a recomputed value match whatever their digits (two
    quadratures' noise)."""
    txt = text_of(el)
    m = re.fullmatch(r"(-?\d+\.\d) × 10([⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)", txt)
    if want is None or abs(want) >= 1e-3 or m is None or (want == 0.0 and not floor):
        cell(a, panel, label, el, want)
        return
    e = int(m.group(2).translate(SUPER))
    printed = float(m.group(1))
    ok = (abs(printed) * 10.0 ** e < floor and abs(want) < floor) or rounds_to(want / 10.0 ** e, printed, 1)
    a.checks.append(Check(a.page, panel, label, txt, f"{want:.2e}", ok,
                          f"noise below {floor:g} matches as a bound" if floor else ""))
    if isinstance(el, Tag):
        a.cover(el)


def rows(table: Tag | None) -> list[Tag]:
    return table.select(":scope > tbody > tr") if table is not None else []


def tds(tr: Tag) -> list[Tag]:
    return tr.find_all("td", recursive=False)


def heads(table: Tag | None) -> list[Tag]:
    return table.select(":scope > thead > tr > th") if table is not None else []


def texts(a: Audit, panel: str, label: str, els: Sequence[Tag], want: Sequence[str]) -> bool:
    """The elements' texts, in order, against the expected strings (both normalised)."""
    for el in els:
        a.cover(el)
    return a.equal(panel, label, [text_of(el) for el in els], [norm(w) for w in want])


def same_len(a: Audit, panel: str, label: str, got: Sized, want: Sized) -> bool:
    return a.equal(panel, f"{label}: count", len(got), len(want))


def meta(a: Audit, pid: str) -> Tag | None:
    return a.one(f"section#{pid} > .panel-head .panel-meta")


def chart_table(a: Audit, cid: str) -> Tag | None:
    return a.one(f"div#{cid}-data.chart-data > table")


def chart_group(a: Audit, cid: str, group: str) -> Tag | None:
    return a.one(f'div#{cid}-data div.chart-data-group[data-group="{group}"] > table')


def spec(a: Audit, sid: str) -> dict | None:
    el = a.one(f"script#{sid}")
    return json.loads(el.get_text()) if el is not None else None


def series(sp: dict | None, name: str) -> list | None:
    s = next((s for s in (sp or {}).get("series") or [] if s.get("name") == name), None)
    return None if s is None else s.get("data")


def legend(a: Audit, panel: str, cid: str, names: Sequence[str]) -> None:
    """The chart's legend buttons: the series names the recompute expects, in order;
    the numbers and dates each name carries are verified on the button."""
    btns = a.soup.select(f'button[data-legend-for="{cid}"]')
    got = [b.get("data-series") for b in btns]
    a.equal(panel, f"{cid} legend", got, list(names))
    for b, name in zip(btns, names):
        want = [v for v, _, _ in numbers_in(name)]
        if want or DATE.search(name):
            nums(a, panel, f"{cid} legend '{name}'", b, want, ["-".join(m) for m in DATE.findall(name)])


def sgn(x: float | None, spec_: str = ".2f") -> str:
    if x is None:
        return "—"
    body = format(abs(x), spec_)
    return ("" if float(body.replace(",", "")) == 0.0 else ("+" if x > 0 else "−")) + body


def prose(text: str) -> str:
    """Producer prose as the page prints it: a hyphen before a number is a minus."""
    s = re.sub(r"\$-(?=\d)", "−$", str(text))
    return re.sub(r"(?<![\w/.\-–])-(\$?)(?=\.?\d)", lambda m: "−" + m.group(1), s)


def range_nums(r: tuple[int, float, float] | None) -> list[float]:
    """A tile's range over the stored sessions as printed: 'n sessions: lo–hi'."""
    return [] if r is None else [float(r[0]), r[1], r[2]]


# ── the session in view ───────────────────────────────────────────────────────
class Ctx:
    """The newest stored session, its recompute, and every other stored session."""

    def __init__(self, store: Store):
        self.store = store
        self.names = M.surface_sessions(store)
        self.s = M.load_session(store, self.names[-1])
        self.all = [M.load_session(store, n) for n in self.names]
        self.desk = [x for x in self.all if x.sj is not None]       # the page's desk inputs
        s = self.s
        self.session = s.name
        self.mkt = s.mkt
        self.exps = M.expiries(s)
        self.used = M.buckets_used(self.mkt["k"])
        self.listed = M.listed_grid(s)
        # the comparison with earlier sessions (and the charts that draw them) needs this
        # session's SVI-cal slices and at least one earlier session with a surface.json
        self.earlier = sorted((x for x in self.desk if x.name < s.name), key=lambda x: x.name, reverse=True)
        self.compare = bool(self.listed) and bool(self.earlier)
        self.cm = M.cm_grid(s)
        self.by_cm = {r["label"]: r for r in self.cm}
        self.sj, self.sc, self.vx = s.sj or {}, s.sc, s.vx or {}
        self.score_models = self._score_order()
        self.hkey = next((m for m in s.models if m.lower().startswith("heston")), None)
        self.overlay = list(dict.fromkeys(m for m in [self.hkey, *self.score_models] if m and m in s.models))
        self.vix_db = store.vix_close()
        self.fam = store.vix_family()

    def _score_order(self) -> list[str]:
        if not self.sc:
            return list(self.s.models)
        order = [str(m["model"]) for m in self.sc["models"]]
        rank = [r.get("model") for r in self.sc.get("ranking_by_oos_median_abs_hs") or []]
        pos = {m: i for i, m in enumerate(rank)}
        return sorted(order, key=lambda m: pos.get(m, len(pos) + order.index(m)))

    def sc_model(self, m: str, s: M.Session | None = None) -> dict | None:
        sc = (s or self.s).sc
        return next((r for r in (sc or {}).get("models") or [] if r.get("model") == m), None)

    def official(self, ticker: str, session: str) -> float | None:
        """The Cboe close of a VIX-family index on a session, from the market DB."""
        if ticker == "VIX":
            v = self.vix_db.get(session)
        else:
            col = "^" + ticker
            v = self.fam[col].get(session) if col in self.fam else None
        return M.num(v)

    def vix_gap(self, s: M.Session) -> float | None:
        """VIX replicated from the session's chain − the DB's Cboe close of that session."""
        rep = M.num((((s.vx or {}).get("indices") or {}).get("VIX") or {}).get("replicated"))
        off = self.official("VIX", s.name)
        return None if rep is None or off is None else rep - off


# ── page head, staleness ──────────────────────────────────────────────────────
P_HEAD = "page head"


def page_head(a: Audit, c: Ctx) -> None:
    """Session, expiry and quote counts, models; the chain badge against the stored
    chain and every stored session's expiry count."""
    items = a.soup.select(".page-head > .page-badge:not(#chain-badge) > span.nowrap")
    if not same_len(a, P_HEAD, "badge items", items, range(4)):
        return
    a.date(P_HEAD, "session", items[0], c.session)
    a.num(P_HEAD, "expiries (distinct expiries of the residual table)", items[1], float(len(c.exps)))
    a.num(P_HEAD, "quotes (one per quote of the residual table)", items[2], float(len(c.mkt)))
    a.text(P_HEAD, "models", items[3], "models: " + ", ".join(M.label(m) for m in c.s.models))
    chain_badge(a, c)


def chain_badge(a: Audit, c: Ctx) -> None:
    """The chain badge (shown with SVI-cal slices and a recorded chain source): source
    and quote time, the chain's rows (the stored chain file where it is on disk) and
    the scored quotes, the expiries against every other stored session's."""
    inp, uni = c.sj.get("inputs") or {}, c.sj.get("universe") or {}
    parts = chain_parts(c)
    badge = a.soup.select("#chain-badge > span.nowrap")
    if not parts or not c.listed:
        a.equal(P_HEAD, "chain badge (shown with a chain source and SVI-cal slices)", len(badge), 0)
        return
    qt = inp.get("quote_time") or uni.get("quote_time")
    a.text(P_HEAD, "chain source", badge[0] if badge else None, "Chain: " + parts[0])
    for i, want in enumerate(parts[1:], start=1):
        a.text(P_HEAD, f"quote time, part {i}", badge[i] if len(badge) > i else None, want)
    rows_el = next((b for b in badge if "rows →" in text_of(b)), None)
    shown = M.num(inp.get("chain_rows")) is not None and M.num(uni.get("n_universe")) is not None
    a.equal(P_HEAD, "chain rows shown (surface.json records chain_rows and n_universe)",
            rows_el is not None, shown)
    if rows_el is not None:
        chain = c.store.root.parent / str(inp.get("chain") or "")
        if inp.get("chain") and chain.is_file():
            stored = pd.read_parquet(chain, columns=["strike", "quote_time"])
            a.num(P_HEAD, "chain rows (rows of the stored chain file)", rows_el, float(len(stored)), index=0)
            cq = stored["quote_time"].dropna().astype(str)
            if qt and len(cq):
                a.equal(P_HEAD, "quote time = the stored chain's", str(qt)[:19].replace("T", " "),
                        cq.iloc[0][:19].replace("T", " "))
        else:
            a.num(P_HEAD, "chain rows (surface.json inputs.chain_rows)", rows_el,
                  M.num(inp.get("chain_rows")))
        a.num(P_HEAD, "scored quotes (residual table)", rows_el, float(len(c.mkt)), index=1)
    exp_el = next((b for b in badge if "expiries" in text_of(b)), None)
    a.equal(P_HEAD, "expiries shown (surface.json records n_expiries)", exp_el is not None,
            M.num(uni.get("n_expiries")) is not None)
    if exp_el is not None:
        other = sorted(len(M.expiries(x)) for x in c.desk if x.name != c.session)
        rng = other[:1] if other and other[0] == other[-1] else [other[0], other[-1]] if other else []
        nums(a, P_HEAD, "expiries, and every other stored session's (residual tables)", exp_el,
             [len(c.exps), *rng])


def staleness(a: Audit, c: Ctx) -> None:
    """The stale banner is there exactly when the surface session is ≥ 2 NYSE sessions
    behind the newest session the site has data for (red from 5); no loader-error banner."""
    ref = max(c.store.spx().index[-1], regime_labels(c.store).index[-1])
    age = sessions_between(c.session, ref)
    el = a.one('div[data-stale="surface"]')
    a.equal(P_HEAD, f"stale banner shown (age {age} NYSE sessions vs {ref})", el is not None,
            age >= STALE_WARN)
    if el is not None:
        a.equal(P_HEAD, "stale banner data-age", el.get("data-age"), str(age))
        a.equal(P_HEAD, "stale banner level", el.get("class"),
                ["banner", f"banner-stale-{'error' if age >= STALE_ERROR else 'warn'}"])
        a.num(P_HEAD, "stale banner age", el.select_one(".banner-title"), float(age))
        nums(a, P_HEAD, "stale banner dates", el.select_one(".banner-body"), [], [c.session, ref])
    a.equal(P_HEAD, "loader-error banner", a.one("[data-loader-errors]") is None, True)


# ── surface-state tiles ───────────────────────────────────────────────────────
P_STATE = "#surface-state"


def kpis(a: Audit, sel: str) -> dict[str, Tag]:
    return {text_of(k.select_one(".kpi-label")): k for k in a.soup.select(f"{sel} > .kpi")}


def state_tiles(a: Audit, c: Ctx) -> None:
    """1M ATMF / RR25 / BF25 / VS − ATMF and the 1M → 1Y ATMF slope of the SVI-cal
    surface at constant maturity, each with its range over every stored session."""
    tiles = kpis(a, "#surface-state")
    a.equal(P_STATE, "tiles shown (SVI-cal slices stored)", bool(tiles), bool(c.listed))
    if not c.listed:
        return
    every = [M.state_values(M.cm_grid(x)) for x in c.desk]
    now = M.state_values(c.cm)
    m1, y1 = c.by_cm["1M"], c.by_cm["1Y"]
    span = f"{c.listed[0]['T_days']:.0f}–{c.listed[-1]['T_days']:.0f} d"
    for name, key in (("1M ATMF", "atmf"), ("1M RR25", "rr25"), ("1M BF25", "bf25"), ("1M VS − ATMF", "vs"),
                      ("ATMF slope 1M → 1Y", "slope")):
        t = tiles.get(norm(name))
        if t is None:
            a.fail(P_STATE, name, "present", "missing")
            continue
        val = t.select_one(".kpi-value")
        cell(a, P_STATE, f"{name} value", val, now[key])
        rg = M.session_range([e[key] for e in every])
        ci = t.select_one(".kpi-ci")
        if rg is None:
            a.equal(P_STATE, f"{name} range (fewer than 2 sessions)", ci is None, True)
        else:
            nums(a, P_STATE, f"{name} range over {rg[0]} stored sessions", ci, range_nums(rg))
        sub = t.select_one(".kpi-sub")
        if key == "atmf":
            if not m1["available"]:
                where = f"outside the listed {span}"
            elif m1["lower"] == m1["upper"]:
                where = f"the {m1['lower']} expiry"
            else:
                where = f"between {m1['lower']} and {m1['upper']}"
            a.text(P_STATE, "1M ATMF bracketing expiries", sub, f"SVI-cal at 30 days, {where}")
        elif key == "rr25":
            a.text(P_STATE, "1M RR25 definition", sub, "25Δ call − 25Δ put; negative: puts over calls")
        elif key == "bf25":
            a.text(P_STATE, "1M BF25 definition", sub, "mean of the 25Δ wings − ATMF")
        elif key == "vs":
            vix = (c.vx.get("indices") or {}).get("VIX") or {}
            want = [m1["vs"], 30] if m1["vs"] is not None else []
            if vix:
                rep, off = M.num(vix.get("replicated")), c.official("VIX", c.session)
                want += [rep, off] if rep is not None else [off]
                if rep is None:
                    a.text(P_STATE, "VIX not replicated (vix_replication.json)", sub,
                           "VIX not replicated", contains=True)
            nums(a, P_STATE, "1M VS vol, VIX replicated, Cboe close (DB)", sub, want)
        elif now["slope"] is not None:
            nums(a, P_STATE, "1Y and 1M ATMF", sub, [y1["atmf"], m1["atmf"]])
        else:
            a.text(P_STATE, "1Y beyond the listed expiries", sub, f"1Y outside the listed {span}")


# ── score tiles, VIX tile, arbitrage tile ─────────────────────────────────────
P_SCORES = "#surface-scores"


def oos_stats(c: Ctx, m: str, s: M.Session | None = None) -> dict:
    """A model's out-of-sample metrics from the residual table: median |z|, inside %,
    within 2 half-spreads %, RMSE and mean of model − mid (vol pts), count."""
    q = M.oos_rows(s or c.s, m)
    z = q["error_hs"].abs()
    e = (q["model_iv"] - q["mid_iv"]) * 100.0
    return {"n": float(len(q)), "median_abs_hs": float(z.median()) if len(q) else None,
            "inside_spread_pct": float(q["inside_spread"].astype(bool).mean() * 100) if len(q) else None,
            "within_2hs_pct": float((z <= WITHIN_HS).mean() * 100) if len(q) else None,
            "rmse_vp": float(np.sqrt((e ** 2).mean())) if len(q) else None,
            "mean_error_vp": float(e.mean()) if len(q) else None}


def score_tiles(a: Audit, c: Ctx) -> None:
    """Each model's out-of-sample median |z| tile: the scorecard's figures, the same
    figures recomputed from the residual table, and the range over every session."""
    tiles = kpis(a, "#surface-scores")
    for m in c.score_models:
        name = f"{M.label(m)} · median |z|"
        t = tiles.get(norm(name))
        if t is None:
            a.fail(P_SCORES, name, "present", "missing")
            continue
        o = ((c.sc_model(m) or {}).get("metrics") or {}).get("out_of_sample") or {}
        r = oos_stats(c, m)
        val, sub = t.select_one(".kpi-value"), t.select_one(".kpi-sub")
        a.num(P_SCORES, f"{name} (scorecard.json)", val, M.num(o.get("median_abs_hs")))
        a.num(P_SCORES, f"{name} (residual table, error_hs)", val, r["median_abs_hs"])
        nums(a, P_SCORES, f"{M.label(m)} inside %, RMSE, n (scorecard.json)", sub,
             [M.num(o.get(k)) for k in ("inside_spread_pct", "rmse_vp", "n")])
        nums(a, P_SCORES, f"{M.label(m)} inside %, RMSE, n (residual table)", sub,
             [r[k] for k in ("inside_spread_pct", "rmse_vp", "n")])
        vals = [M.num((((c.sc_model(m, x) or {}).get("metrics") or {}).get("out_of_sample") or {})
                      .get("median_abs_hs")) for x in c.all]
        rg = M.session_range(vals)
        ci = t.select_one(".kpi-ci")
        if rg is None:
            a.equal(P_SCORES, f"{name} range (fewer than 2 sessions)", ci is None, True)
        else:
            nums(a, P_SCORES, f"{name} range over {rg[0]} sessions' scorecards", ci, range_nums(rg))
    vix = (c.vx.get("indices") or {}).get("VIX")
    t = tiles.get("VIX from this chain")
    a.equal(P_SCORES, "VIX tile shown", t is not None, bool(vix))
    if vix and t is not None:
        rep, off = M.num(vix.get("replicated")), c.official("VIX", c.session)
        cell(a, P_SCORES, "VIX replicated (vix_replication.json)", t.select_one(".kpi-value"), rep)
        sub = t.select_one(".kpi-sub")
        if rep is None:
            # the reason, re-derived: the Cboe-rules pair's own days fail to bracket the maturity
            cr = vix.get("cboe_rules") or {}
            days = M.num(cr.get("constant_maturity_days"))
            near, nxt = (M.num((cr.get(k) or {}).get("days")) for k in ("near", "next"))
            brackets = (near <= days <= nxt) if (days is not None and near is not None
                                                 and nxt is not None) else None
            why = (f"no stored expiry pair brackets {days:.0f} days" if brackets is False
                   else "see the VIX family table")
            a.text(P_SCORES, "VIX not replicated, and why (vix_replication.json terms)", sub,
                   f"not replicated: {why} · Cboe close", contains=True)
            nums(a, P_SCORES, "maturity, Cboe close (DB)", sub, [*([days] if brackets is False else []), off])
        else:
            vix_tile_sub(a, c, sub, rep, off)
        rg = M.session_range([c.vix_gap(x) for x in c.all if x.vx])
        ci = t.select_one(".kpi-ci")
        if rg is None:
            a.equal(P_SCORES, "VIX gap range (fewer than 2 sessions)", ci is None, True)
        else:
            nums(a, P_SCORES, f"VIX gap range over {rg[0]} sessions (replicated − DB close)", ci,
                 range_nums(rg))


def vix_tile_sub(a: Audit, c: Ctx, sub: Tag | None, rep: float, off: float | None) -> None:
    """A replicated VIX tile's sub-line: the Cboe close, the gap and where the
    replicated level sits in the day's VIX range."""
    # the replicated level against the official VIX's range that day (the record's
    # vix_daily low and high; the page prints the range the replication stored)
    hl = c.store.sql("SELECT low, high FROM vix_daily WHERE date = ?", (c.session,))
    lo = M.num(hl["low"].iloc[0]) if len(hl) else None
    hi = M.num(hl["high"].iloc[0]) if len(hl) else None
    ranged = lo is not None and hi is not None
    nums(a, P_SCORES, "Cboe close (DB), gap = replicated − close, the day's VIX range", sub,
         [off, None if off is None else rep - off, *([lo, hi] if ranged else [])])
    if lo is not None and hi is not None:
        where = "inside" if lo <= rep <= hi else "outside"
        a.text(P_SCORES, "replicated level against the day's VIX range", sub,
               f"{where} the day\u2019s VIX range", contains=True)
    else:
        a.equal(P_SCORES, "no day-range clause without a recorded range",
                "VIX range" in text_of(sub), False)


P_ARBT = "#arb-tile"


def arb_tile(a: Audit, c: Ctx) -> None:
    """Quotes in a constraint row violated at bid/ask and at mids (surface.json, and the
    residual table's per-quote flags), the chain, how many of the largest rows a single
    bad quote explains, and the largest violation without one."""
    arb = c.sj.get("arbitrage") or {}
    t = a.one("#arb-tile")
    a.equal(P_ARBT, "tile shown", t is not None, bool(arb))
    if not arb or t is None:
        return
    mk = c.mkt
    val, sub = t.select_one(".kpi-value"), t.select_one(".kpi-sub")
    a.num(P_ARBT, "quotes in a row violated at bid/ask (surface.json)", val,
          M.num(arb.get("n_quotes_in_executable_violation")))
    a.num(P_ARBT, "quotes in a row violated at bid/ask (residual arb_exec)", val,
          float(mk["arb_exec"].astype(bool).sum()))
    a.num(P_ARBT, "quotes in the screen (surface.json)", sub, M.num(arb.get("n_quotes")), index=0)
    a.num(P_ARBT, "quotes at mids (surface.json)", sub, M.num(arb.get("n_quotes_in_mid_violation")), index=1)
    want: list[float | None] = [float(len(mk)), float(mk["arb_mid"].astype(bool).sum())]
    chain = chain_text(c)
    if chain:
        a.text(P_ARBT, "chain", sub, chain, contains=True)
        want += [v for v, _, _ in numbers_in(chain)]
    lv = M.violations(arb)
    if lv:
        want += [sum(1 for v in lv if v.bad), len(lv)]
    top = next((v for v in lv if not v.bad), None)
    short = violation_short(top) if top is not None else ""
    if top is not None:
        a.text(P_ARBT, "largest clean violation: constraint and legs", sub,
               f"{top.type.replace('_', ' ')}, {short}", contains=True)
        usd_ = None if top.shortfall is None else top.shortfall * M.SPX_MULTIPLIER
        want += [top.shortfall, usd_, *(v for v, _, _ in numbers_in(short))]
    nums(a, P_ARBT, "quotes, at mids (residual arb_mid), chain, bad-quote rows, largest clean row", sub, want,
         [m.group(0) for m in DATE.finditer(f"{chain} {short}")])


def chain_parts(c: Ctx) -> list[str]:
    """[name, time clause] of the session's chain (data.chain_words); [] when no
    chain source is recorded."""
    w = chain_words(c.sj.get("inputs") or {}, c.sj.get("universe") or {})
    return [w[0], *w[2]] if w else []


def chain_text(c: Ctx) -> str:
    return ", ".join(chain_parts(c))


def violation_short(v: M.Violation) -> str:
    """A violation row in short: '2027-03-31 P 6260/6265' (expiries, rights, strikes)."""
    legs = v.legs
    exps = "/".join(dict.fromkeys(lg[0] for lg in legs))
    rights = "/".join(dict.fromkeys(lg[1] for lg in legs))
    strikes = "/".join(dict.fromkeys(f"{lg[2]:.0f}" for lg in legs))
    return f"{exps} {rights} {strikes}"


# ── the delta grid ────────────────────────────────────────────────────────────
P_GRID = "#delta-grid"
GRID_KEYS = ("rr25", "bf25", "rr10", "bf10")


def delta_grid(a: Audit, c: Ctx) -> None:
    """Listed expiries and constant maturities in desk coordinates, cell by cell (the
    panel shows with SVI-cal slices)."""
    a.equal(P_GRID, "panel shown", a.one("section#delta-grid") is not None, bool(c.listed))
    if not c.listed:
        return
    nums(a, P_GRID, "panel meta", meta(a, "delta-grid"), [], [c.session])
    tabs = a.soup.select('#delta-grid [role="tab"]')
    nums(a, P_GRID, "tab counts", tabs[0] if tabs else None, [len(c.listed)])
    nums(a, P_GRID, "tab counts (CM)", tabs[1] if len(tabs) > 1 else None, [len(c.cm)])
    # the conventions: ATMF at ln(K/F) = 0, BF = (call + put)/2 − ATMF; 1M and 1Y in days
    nums(a, P_GRID, "listed caption", a.one("#delta-grid-listed > caption"), [0, 2], [c.session])
    nums(a, P_GRID, "CM caption", a.one("#delta-grid-cm > caption"),
         [M.CM_TENORS[0][0], M.CM_TENORS[-1][0], 0, 2])
    hs = c.mkt.groupby("expiry")["half_spread_vp"].median()
    nq = c.mkt.groupby("expiry").size()
    fwd = c.mkt.groupby("expiry")["forward"].median()
    trs = rows(a.one("#delta-grid-listed"))
    if same_len(a, P_GRID, "listed rows", trs, c.listed):
        for tr, r in zip(trs, c.listed):
            e = r["expiry"]
            a.equal(P_GRID, f"{e} row data-exp", tr.get("data-exp"), e)
            a.date(P_GRID, f"{e} row head", tr.find("th"), e)
            td = tds(tr)
            if not same_len(a, P_GRID, f"{e} cells", td, range(16)):
                continue
            cell(a, P_GRID, f"{e} T (d) = slice T × 365", td[0], r["T_days"])
            cell(a, P_GRID, f"{e} F (residual forward)", td[1], M.num(fwd.get(e)))
            for i, (key, _) in enumerate(M.PILLARS):
                cell(a, P_GRID, f"{e} {key} vol", td[2 + i], r["vol"][key])
                a.num(P_GRID, f"{e} {key} ln(K/F) (cell title)", str(td[2 + i].get("title") or ""),
                      r["k"][key])
                a.equal(P_GRID, f"{e} {key} extrapolated flag", "xp" in (td[2 + i].get("class") or []),
                        r["xp"][key])
            for i, key in enumerate(GRID_KEYS):
                cell(a, P_GRID, f"{e} {key}", td[7 + i], r[key])
            cell(a, P_GRID, f"{e} ATM skew (vol pts per 0.1 k)", td[11], r["skew_vp"])
            cell(a, P_GRID, f"{e} VS vol (strip)", td[12], r["vs"])
            cell(a, P_GRID, f"{e} VS − ATMF", td[13], r["vs_minus_atmf"])
            cell(a, P_GRID, f"{e} median half-spread (residual table)", td[14], M.num(hs.get(e)))
            cell(a, P_GRID, f"{e} quotes (residual table)", td[15], float(nq.get(e, 0)))
    trs = rows(a.one("#delta-grid-cm"))
    if same_len(a, P_GRID, "CM rows", trs, c.cm):
        for tr, r in zip(trs, c.cm):
            lab = r["label"]
            th = tr.find("th")
            subs = th.select("span.dg-sub") if th else []
            a.equal(P_GRID, f"{lab} available (row not muted)", "muted" not in (tr.get("class") or []),
                    r["available"])
            if not r["available"]:
                a.text(P_GRID, f"{lab} bracket", subs[0] if subs else None, "outside the listed expiries")
            else:
                a.text(P_GRID, f"{lab} bracketing expiries", subs[0] if subs else None,
                       f"= {r['lower']}" if r["lower"] == r["upper"] else f"{r['lower']} → {r['upper']}")
                n_lo, n_hi = r["strikes"]
                if n_lo is not None and n_hi is not None:
                    nums(a, P_GRID, f"{lab} VS strikes of the two strips", subs[1] if len(subs) > 1 else None,
                         [n_lo] if r["lower"] == r["upper"] else [n_lo, n_hi])
            td = tds(tr)
            if not same_len(a, P_GRID, f"{lab} cells", td, range(16)):
                continue
            cell(a, P_GRID, f"{lab} T (d)", td[0], float(r["days"]))
            cell(a, P_GRID, f"{lab} F", td[1], None)
            for i, (key, _) in enumerate(M.PILLARS):
                cell(a, P_GRID, f"{lab} {key} vol", td[2 + i], r["vol"][key])
            for i, key in enumerate(GRID_KEYS):
                cell(a, P_GRID, f"{lab} {key}", td[7 + i], r[key])
            cell(a, P_GRID, f"{lab} ATM skew", td[11], r["skew_vp"])
            cell(a, P_GRID, f"{lab} VS vol", td[12], r["vs"])
            cell(a, P_GRID, f"{lab} VS − ATMF", td[13], r["vs_minus_atmf"])
            cell(a, P_GRID, f"{lab} half-spread", td[14], None)
            cell(a, P_GRID, f"{lab} quotes", td[15], None)
    both = [*c.listed, *c.cm]
    vols = [r["vol"][k] for r in both for k, _ in M.PILLARS if r["vol"][k] is not None]
    rr = [abs(r[k]) for r in both for k in ("rr25", "rr10") if r.get(k) is not None] or [1.0]
    bf = [abs(r[k]) for r in both for k in ("bf25", "bf10") if r.get(k) is not None] or [1.0]
    nums(a, P_GRID, "colour ramp: vol lo/hi, ±max |RR|, max |BF|", a.one("#delta-grid .panel-body > .ramp"),
         [min(vols), max(vols), -max(rr), max(rr), max(bf), 0])
    n_xp = sum(1 for r in c.listed for k, _ in M.PILLARS if r["xp"][k])
    any_xp = n_xp > 0 or any(r["xp"][k] for r in c.cm for k, _ in M.PILLARS)
    units = a.one("#delta-grid-units")
    nums(a, P_GRID, "units note (skew unit; extrapolated pillars of all listed)", units,
         [M.SKEW_UNIT_K] + ([n_xp, 5 * len(c.listed)] if any_xp else []))
    a.equal(P_GRID, "units note says no wing is extrapolated", "no wing is extrapolated" in text_of(units),
            not any_xp)


# ── against earlier sessions, RR25 by expiry ──────────────────────────────────
P_CHG = "#surface-change"


def pair_cell(a: Audit, panel: str, label: str, el: Tag | None, old: float | None, new: float | None) -> None:
    """A cell printed 'old → new'."""
    parts = text_of(el).split("→")
    if len(parts) != 2:
        a.fail(panel, label, "'old → new'", text_of(el) or "missing")
        return
    cell(a, panel, f"{label} (earlier)", parts[0], old)
    cell(a, panel, f"{label} (now)", parts[1], new)
    a.cover(el)


def surface_change(a: Audit, c: Ctx) -> None:
    """The constant-maturity grid against each earlier stored session: ages in NYSE
    sessions, both values and the unrounded change per tenor."""
    a.equal(P_CHG, "panel shown", a.one("section#surface-change") is not None, c.compare)
    if not c.compare:
        return
    earlier = c.earlier
    nums(a, P_CHG, "panel meta (earlier sessions)", meta(a, "surface-change"), [len(earlier)])
    tabs = a.soup.select('#surface-change [role="tab"]')
    a.equal(P_CHG, "tabs: earlier sessions, newest first", [t.get("data-value") for t in tabs],
            [x.name for x in earlier])
    a.equal(P_CHG, "default tab: the most recent earlier session",
            [t.get("data-value") for t in tabs if t.get("aria-selected") == "true"], [earlier[0].name])
    for t, x in zip(tabs, earlier):
        nums(a, P_CHG, f"tab {x.name}", t, [], [x.name])
    for x in earlier:
        old = M.cm_grid(x)
        tbl = a.one(f"#vs-table-{x.name}")
        gap = sessions_between(x.name, c.session)
        nums(a, P_CHG, f"vs {x.name}: NYSE sessions earlier", a.one(f"#vs-table-{x.name} > caption"), [gap],
             [x.name, x.name, c.session, c.session, x.name])
        trs = rows(tbl)
        if not same_len(a, P_CHG, f"vs {x.name} rows", trs, c.cm):
            continue
        by_old = {r["days"]: r for r in old}
        for tr, r in zip(trs, c.cm):
            o = by_old.get(r["days"]) or {}
            a.text(P_CHG, f"vs {x.name} tenor", tr.find("th"), r["label"])
            a.equal(P_CHG, f"vs {x.name} {r['label']} available", "muted" not in (tr.get("class") or []),
                    bool(r["available"] and o.get("available")))
            td = tds(tr)
            if not same_len(a, P_CHG, f"vs {x.name} {r['label']} cells", td, range(6)):
                continue
            for i, key in enumerate(("atmf", "rr25", "bf25")):
                nv, ov = r.get(key), o.get(key)
                pair_cell(a, P_CHG, f"vs {x.name} {r['label']} {key}", td[2 * i], ov, nv)
                cell(a, P_CHG, f"vs {x.name} {r['label']} Δ {key} (unrounded new − old)", td[2 * i + 1],
                     None if nv is None or ov is None else nv - ov)
    rr_term(a, c)
    nums(a, P_CHG, "RR25 note", a.one("#rr-term-note"), [], [c.session])
    nums(a, P_CHG, "RR25 chart description", a.one("#chart-rr-term-desc"), [], [c.session])


def rr_term(a: Audit, c: Ctx) -> None:
    """RR25 of every listed expiry of every stored session: the chart's table and series."""
    want = [(x.name, r) for x in sorted(c.desk, key=lambda x: x.name) for r in M.listed_grid(x)]
    trs = rows(chart_table(a, "chart-rr-term"))
    if same_len(a, P_CHG, "RR25 table rows", trs, want):
        for tr, (sname, r) in zip(trs, want):
            td = tds(tr)
            lab = f"RR25 {sname} {r['expiry']}"
            a.date(P_CHG, f"{lab} session", td[0], sname)
            a.date(P_CHG, f"{lab} expiry", td[1], r["expiry"])
            cell(a, P_CHG, f"{lab} T (d)", td[2], r["T_days"])
            cell(a, P_CHG, lab, td[3], r["rr25"])
    sp = spec(a, "chart-rr-term-spec")
    name = f"SVI-cal RR25 {c.session}"
    approx(a, P_CHG, f"chart-rr-term series {name}", series(sp, name),
           [[r["T_days"], r["rr25"]] for r in c.listed])
    names = (["Every stored session"] if len(c.desk) > 1 else []) + [
        f"SVI-cal RR25 {x.name}" for x in sorted(c.desk, key=lambda x: x.name) if x.name != c.session]
    legend(a, P_CHG, "chart-rr-term", [*names, f"SVI-cal RR25 {c.session}"])
    for x in c.desk:
        if x.name != c.session:
            name = f"SVI-cal RR25 {x.name}"
            approx(a, P_CHG, f"chart-rr-term series {name}", series(sp, name),
                   [[r["T_days"], r["rr25"]] for r in M.listed_grid(x)])


# ── ATM term structure, forward vols ──────────────────────────────────────────
P_ATM = "#atm-term"


def essvi_atm(sl: dict) -> tuple[float, float]:
    """eSSVI ATM vol (%) and ATM skew ∂σ/∂k (decimal) of one slice: w(0) = θ, w′(0) = ρψ."""
    atm, T = float(sl["atm_vol"]), float(sl["T"])
    theta = atm ** 2 * T
    return 100.0 * atm, float(sl["rho"]) * float(sl["psi"]) / (2.0 * math.sqrt(theta * T))


def atm_term(a: Audit, c: Ctx) -> None:
    """The ATM term structure table (quotes, eSSVI from its own slice parameters, parity
    r and q, quote counts) and the SVI-cal ATM of each earlier session."""
    ts = sorted(c.sj.get("term_structure") or [], key=lambda t: float(t["T_days"]))
    a.equal(P_ATM, "panel shown", a.one("section#atm-term") is not None, bool(ts))
    if not ts:
        return
    ess = {str(s_["expiry"])[:10]: s_ for s_ in M.slices(c.sj, "essvi")}
    nq = c.mkt.groupby("expiry").size()
    trs = rows(chart_table(a, "chart-atm-ts"))
    if same_len(a, P_ATM, "ATM table rows", trs, ts):
        for tr, t in zip(trs, ts):
            e = str(t.get("expiry"))[:10]
            td = tds(tr)
            a.date(P_ATM, f"{e} expiry", td[0], e)
            cell(a, P_ATM, f"{e} days (residual T_days)", td[1], M.num(c.exps.get(e)))
            cell(a, P_ATM, f"{e} ATM mid (surface.json)", td[2], M.num(t.get("atm_vol_mid_quotes")))
            sl = ess.get(e)
            atm, skew = essvi_atm(sl) if sl else (None, None)
            cell(a, P_ATM, f"{e} eSSVI ATM (eSSVI slice θ)", td[3], atm)
            cell(a, P_ATM, f"{e} eSSVI ATM skew (ρψ / 2√(θT))", td[4], skew)
            cell(a, P_ATM, f"{e} parity r (surface.json)", td[5], M.num(t.get("r")), scale=100.0)
            cell(a, P_ATM, f"{e} parity q (surface.json)", td[6], M.num(t.get("q")), scale=100.0)
            cell(a, P_ATM, f"{e} quotes (residual table)", td[7], float(nq.get(e, 0)))
    earlier = c.earlier if c.compare else []
    for x in earlier:
        lst = M.listed_grid(x)
        tbl = chart_group(a, "chart-atm-ts", x.name)
        nums(a, P_ATM, f"{x.name} group caption", tbl.find("caption") if tbl else None, [], [x.name])
        trs = rows(tbl)
        if same_len(a, P_ATM, f"{x.name} SVI-cal ATM rows", trs, lst):
            for tr, r in zip(trs, lst):
                td = tds(tr)
                a.date(P_ATM, f"{x.name} {r['expiry']} expiry", td[0], r["expiry"])
                cell(a, P_ATM, f"{x.name} {r['expiry']} days", td[1], r["T_days"])
                cell(a, P_ATM, f"{x.name} {r['expiry']} SVI-cal ATM", td[2], r["atmf"])
    sp = spec(a, "chart-atm-ts-spec")
    approx(a, P_ATM, "chart-atm-ts series ATM mid (quotes)", series(sp, "ATM mid (quotes)"),
           [[float(t["T_days"]), M.num(t.get("atm_vol_mid_quotes"))] for t in ts
            if M.num(t.get("atm_vol_mid_quotes")) is not None])
    approx(a, P_ATM, "chart-atm-ts series eSSVI ATM", series(sp, "eSSVI ATM"),
           [[float(t["T_days"]), essvi_atm(ess[str(t["expiry"])[:10]])[0]] for t in ts
            if str(t["expiry"])[:10] in ess])
    names = ["ATM mid (quotes)", "eSSVI ATM"]
    # the SVI-cal lines and the forward vols come with the comparison
    fw = M.forward_vols(c.cm) if c.compare else []
    if c.compare:
        for x in [c.s, *earlier]:
            name = f"SVI-cal ATM {x.name}"
            approx(a, P_ATM, f"chart-atm-ts series {name}", series(sp, name),
                   [[r["T_days"], r["atmf"]] for r in M.listed_grid(x)])
            names.append(name)
        steps: list[list[float | None]] = []
        for f in fw:
            steps += ([[f["t1"], None]] if f["fwd_vol"] is None
                      else [[f["t1"], f["fwd_vol"]], [f["t2"], f["fwd_vol"]]])
        name = "SVI-cal forward ATM vol (1M → 1Y)"
        approx(a, P_ATM, "chart-atm-ts series forward ATM vol steps", series(sp, name), steps)
        names.append(name)
        flags = [f for f in fw if f["fwd_vol"] is None]
        if flags:
            name = "Calendar flag: negative forward variance"
            approx(a, P_ATM, "chart-atm-ts calendar flags (mid-period, the shorter tenor's ATMF)",
                   series(sp, name), [[(f["t1"] + f["t2"]) / 2, f["atm1"]] for f in flags])
            names.append(name)
    legend(a, P_ATM, "chart-atm-ts", names)
    note = a.one("#forward-vols")
    a.equal(P_ATM, "forward vols shown", note is not None, bool(fw))
    if note is not None:
        spans = note.select("span.nowrap")
        if same_len(a, P_ATM, "forward vol pairs", spans, fw):
            for sp_, f in zip(spans, fw):
                lab = f"forward ATM vol {f['from']} → {f['to']}"
                a.text(P_ATM, f"{lab} tenors", sp_, f"{f['from']} → {f['to']}", contains=True)
                if f["fwd_vol"] is None:
                    a.text(P_ATM, f"{lab} calendar flag", sp_, "⚑ calendar", contains=True)
                else:
                    cell(a, P_ATM, f"{lab} √((w₂ − w₁)/(T₂ − T₁))", sp_, f["fwd_vol"])
        nums(a, P_ATM, "forward vols note", note, [f["fwd_vol"] for f in fw if f["fwd_vol"] is not None]
             + [M.FWD_MIN_DAYS])


# ── the realised-vol cone ─────────────────────────────────────────────────────
P_CONE, P_RANK = "#vol-cone", "#vol-cone-rank"


def vol_cone(a: Audit, c: Ctx) -> None:
    """The cone's percentiles, window counts and current RV from SPX closes in the DB;
    each listed expiry's ATMF ranked against RV over as many sessions. The panels show
    when some listed expiry lies within the cone's 252 sessions."""
    pts = [(r, *M.cone_h(r["T_days"])) for r in c.listed]
    placed = [(r, hs, h) for r, hs, h in pts if hs <= M.CONE_H[-1]]
    a.equal(P_CONE, "panel shown", a.one("section#vol-cone") is not None, bool(placed))
    if not placed:
        return
    samples = M.cone_samples(c.store, c.session)
    nums(a, P_CONE, "panel meta", meta(a, "vol-cone"), [], [c.session])
    tabs = a.soup.select('#vol-cone [role="tab"]')
    a.equal(P_CONE, "sample tabs", [t.get("data-value") for t in tabs], [s_.key for s_ in samples])
    for t, s_ in zip(tabs, samples):
        a.text(P_CONE, f"tab {s_.key}", t, s_.label)
    cone = [(h, s_, M.rv_windows(s_.r, h)) for h in M.CONE_H for s_ in samples]
    trs = rows(chart_table(a, "chart-vol-cone"))
    if same_len(a, P_CONE, "cone rows", trs, cone):
        for tr, (h, s_, rv) in zip(trs, cone):
            td = tds(tr)
            lab = f"h={h} {s_.key}"
            cell(a, P_CONE, f"{lab} sessions", td[0], float(h))
            cell(a, P_CONE, f"{lab} ≈ days (h·365/252)", td[1], h * 365.0 / 252.0)
            a.text(P_CONE, f"{lab} sample", td[2], s_.label)
            cell(a, P_CONE, f"{lab} windows", td[3], float(len(rv)))
            for i, pc in enumerate(M.CONE_PCTS):
                cell(a, P_CONE, f"{lab} {pc}th pct", td[4 + i],
                     float(np.percentile(rv, pc)) if len(rv) else None)
            cell(a, P_CONE, f"{lab} RV ending the session", td[9], float(rv[-1]) if len(rv) else None)
    tbl = chart_table(a, "chart-vol-cone")
    statics(a, P_CONE, "cone table caption", tbl.find("caption") if tbl else None, [100, 252], [c.session])
    for th in heads(tbl):
        statics(a, P_CONE, f"cone table head '{text_of(th)}'", th, M.CONE_PCTS,
                [c.session] if DATE.search(text_of(th)) else [])
    sp = spec(a, "chart-vol-cone-spec")
    s0 = samples[0]
    approx(a, P_CONE, f"chart-vol-cone series SPX RV, windows ending {c.session}",
           series(sp, f"SPX RV, windows ending {c.session}"),
           [[h, float(M.rv_windows(s0.r, h)[-1])] for h in M.CONE_H], sig=CONE_SIG)
    for s_ in samples:
        for pc in M.CONE_PCTS:
            approx(a, P_CONE, f"chart-vol-cone series {pc}th pct · {s_.label.lower()}",
                   series(sp, f"{pc}th pct RV · {s_.label.lower()}"),
                   [[h, float(np.percentile(M.rv_windows(s_.r, h), pc))] for h in M.CONE_H], sig=CONE_SIG)
    name = f"SVI-cal ATMF {c.session}"
    approx(a, P_CONE, f"chart-vol-cone series {name}", series(sp, name),
           [[hs, r["atmf"]] for r, hs, _ in placed], sig=CONE_SIG)
    legend(a, P_CONE, "chart-vol-cone", [f"SPX RV, windows ending {c.session}", f"SVI-cal ATMF {c.session}"])
    statics(a, P_CONE, "chart description", a.one("#chart-vol-cone-desc"),
            [*M.CONE_PCTS, M.CONE_H[0], M.CONE_H[-1], 252, 365], [c.session])
    note = a.one("#vol-cone-note")
    statics(a, P_CONE, "cone note", note,
            [*M.CONE_PCTS, *M.CONE_H, 100, 252, 365, M.CONE_YEARS, M.CONE_SINCE, 2020],
            [d for s_ in samples for d in (s_.start, c.session)])
    # the rank panel
    nums(a, P_RANK, "panel meta (listed expiries placed)", meta(a, "vol-cone-rank"), [len(placed)])
    trs = rows(a.one("#vol-cone-rank-tbl"))
    if same_len(a, P_RANK, "rank rows", trs, placed):
        for tr, (r, _, h) in zip(trs, placed):
            td = tds(tr)
            e = r["expiry"]
            a.date(P_RANK, f"{e} expiry", td[0], e)
            cell(a, P_RANK, f"{e} days", td[1], r["T_days"])
            cell(a, P_RANK, f"{e} RV horizon (T·252/365 sessions, rounded, at least 2)", td[2], float(h))
            cell(a, P_RANK, f"{e} ATMF", td[3], r["atmf"])
            for i, s_ in enumerate(samples):
                cell(a, P_RANK, f"{e} share of {h}-session RV windows below ATMF, {s_.key}", td[4 + i],
                     M.cone_rank(s_.r, h, r["atmf"]))
    tbl = a.one("#vol-cone-rank-tbl")
    statics(a, P_RANK, "rank table caption", tbl.find("caption") if tbl else None,
            [252, 365, M.CONE_SINCE, M.CONE_YEARS], [c.session])
    texts(a, P_RANK, "rank table heads", heads(tbl), ["Expiry", "Days", "Sessions", "ATMF (%)",
                                                     *(s_.short for s_ in samples)])
    # the listed expiry nearest 30 days against the 1M tile, its ranks in both samples,
    # and the 21-session cone since 2010 (5th–95th percentile, median, the newest window)
    d1m, h21, (lo, hi) = M.CM_TENORS[0][0], M.CONE_H[1], (M.CONE_PCTS[0], M.CONE_PCTS[-1])
    one, _, h1 = min(placed, key=lambda p: abs(p[0]["T_days"] - d1m))
    m1 = c.by_cm["1M"]
    c21 = M.rv_windows(s0.r, h21)
    want = [d1m, one["T_days"], one["atmf"], *([m1["atmf"], d1m] if m1["available"] else []),
            M.cone_rank(s0.r, h1, one["atmf"]), h1, M.CONE_SINCE,
            M.cone_rank(samples[1].r, h1, one["atmf"]), M.CONE_YEARS,
            h21, M.CONE_SINCE, float(np.percentile(c21, lo)), float(np.percentile(c21, hi)), lo, hi,
            float(np.percentile(c21, 50)), h21, float(c21[-1])]
    nums(a, P_RANK, "the listed expiry nearest 30 days, its ATMF, ranks and the 21-session cone",
         a.one("#vol-cone-1m"), want, [one["expiry"], c.session])
    n_beyond = len(pts) - len(placed)
    nums(a, P_RANK, "rank note (expiries beyond 252 sessions)", a.one("#vol-cone-rank-note"),
         [0] + ([n_beyond, 252] if n_beyond else []))


# ── the risk-neutral density ──────────────────────────────────────────────────
P_RND, P_TAILS, P_RNDT = "#rnd", "#rnd-tails", "#rnd-table"


def rnd(a: Audit, c: Ctx) -> None:
    """Density, tails and checks of every SVI-cal slice, from the slice alone."""
    tr_ = M.tail_rows(c.s)
    a.equal(P_RND, "panels shown (SVI-cal slices stored)", a.one("section#rnd") is not None, bool(tr_))
    if not tr_:
        return
    picks = M.rnd_picks(tr_)
    # the density chart
    nums(a, P_RND, "panel meta", meta(a, "rnd"), [], [c.session])
    tbl = chart_table(a, "chart-rnd")
    a.equal(P_RND, "density chart shown (at least 2 picks)", a.one("#chart-rnd") is not None, len(picks) >= 2)
    if len(picks) >= 2:
        hd = heads(tbl)
        if same_len(a, P_RND, "density table heads", hd, range(2 + len(picks))):
            for th, p in zip(hd[2:], picks):
                r = p["row"]
                nums(a, P_RND, f"density head {r['expiry']}", th, [r["T_days"]], [r["expiry"]])
        trs = rows(tbl)
        if same_len(a, P_RND, "density rows", trs, M.RND_TABLE_K):
            for tr, k in zip(trs, M.RND_TABLE_K):
                td = tds(tr)
                cell(a, P_RND, f"k {k:+.2f}", td[0], k)
                cell(a, P_RND, f"k {k:+.2f} K/F", td[1], math.exp(k))
                for t, p in zip(td[2:], picks):
                    r = p["row"]
                    inside = r["k_lo"] - 1e-12 <= k <= r["k_hi"] + 1e-12
                    cell(a, P_RND, f"density {r['expiry']} at k {k:+.2f}", t,
                         float(M.density(k, r["sl"])) if inside else None)
        sp = spec(a, "chart-rnd-spec")
        for p in picks:
            r = p["row"]
            name = f"{r['expiry']} ({r['T_days']:.0f} d)"
            n = max(2, math.ceil((r["k_hi"] - r["k_lo"]) / M.RND_CHART_STEP) + 1)
            ks = np.linspace(r["k_lo"], r["k_hi"], n)
            approx(a, P_RND, f"chart-rnd series {name}", series(sp, name),
                   [[float(k), float(v)] for k, v in zip(ks, M.density(ks, r["sl"]))])
        legend(a, P_RND, "chart-rnd", [f"{p['row']['expiry']} ({p['row']['T_days']:.0f} d)" for p in picks])
        nums(a, P_RND, "tenor key (shortest and longest pick, days)", a.one("#rnd .tenor-key"),
             [picks[0]["row"]["T_days"], picks[-1]["row"]["T_days"]])
        statics(a, P_RND, "chart description", a.one("#chart-rnd-desc"), [2])
    note = a.one("#rnd-note")
    nums(a, P_RND, "note: the formula's 2, then each pick's days and tenor", note,
         [2] + [x for p in picks for x in (p["row"]["T_days"], p["tenor"])],
         [p["row"]["expiry"] for p in picks])
    nums(a, P_RND, "panel title (the picks' target tenors, days)", a.one("#rnd-t"), list(M.RND_TENORS))
    # the tails chart
    nums(a, P_TAILS, "panel meta", meta(a, "rnd-tails"), [], [c.session])
    trs = rows(chart_table(a, "chart-rnd-tails"))
    if same_len(a, P_TAILS, "tail rows", trs, tr_):
        for tr, r in zip(trs, tr_):
            td = tds(tr)
            a.date(P_TAILS, f"{r['expiry']} expiry", td[0], r["expiry"])
            cell(a, P_TAILS, f"{r['expiry']} days", td[1], r["T_days"])
            for i, key in enumerate(("lt80", "lt90", "gt110")):
                tail_cell(a, P_TAILS, f"{r['expiry']} P {key}", td[2 + i], r["prob"][key])
    tbl = chart_table(a, "chart-rnd-tails")
    for th in heads(tbl):
        statics(a, P_TAILS, f"tails table head '{text_of(th)}'", th, [x for _, x, _ in M.THRESHOLDS])
    sp = spec(a, "chart-rnd-tails-spec")
    names = [f"SVI-cal P(S_T {'<' if side == 'below' else '>'} {kf:.2f} F)" for _, kf, side in M.THRESHOLDS]
    for (key, _, _), name in zip(M.THRESHOLDS, names, strict=True):
        approx(a, P_TAILS, f"chart-rnd-tails series {name}", series(sp, name),
               [[r["T_days"], None if r["prob"][key] is None else 100 * r["prob"][key]] for r in tr_])
    legend(a, P_TAILS, "chart-rnd-tails", names)
    statics(a, P_TAILS, "chart description", a.one("#chart-rnd-tails-desc"), [x for _, x, _ in M.THRESHOLDS])
    n_gt = sum(1 for r in tr_ if r["prob"]["gt110"] is None)
    nums(a, P_TAILS, "note (expiries quoting no strike as high as 1.10 F)", a.one("#rnd-tails-note"),
         [n_gt, len(tr_), 1.10] if n_gt else [])
    # the table
    nums(a, P_RNDT, "panel meta", meta(a, "rnd-table"), [len(tr_)])
    pick_of = {p["row"]["expiry"]: p["tenor"] for p in picks}
    trs = rows(a.one("#rnd-table-tbl"))
    if same_len(a, P_RNDT, "rows", trs, tr_):
        for tr, r in zip(trs, tr_):
            e = r["expiry"]
            td = tds(tr)
            first = td[0]
            want = [pick_of[e]] if e in pick_of else []
            nums(a, P_RNDT, f"{e} expiry (and its tenor when a density pick)", first, want, [e])
            a.equal(P_RNDT, f"{e} highlighted as a pick", "em" in (tr.get("class") or []), e in pick_of)
            a.equal(P_RNDT, f"{e} 'not certified' chip", first.select_one(".chip-warn") is not None,
                    not r["certified"])
            cells = td[1:]
            if not same_len(a, P_RNDT, f"{e} cells", cells, range(10)):
                continue
            cell(a, P_RNDT, f"{e} days", cells[0], r["T_days"])
            if r["k_lo"] is None:
                a.text(P_RNDT, f"{e} quoted K/F", cells[1], "—")
            else:
                nums(a, P_RNDT, f"{e} quoted K/F range", cells[1], [math.exp(r["k_lo"]), math.exp(r["k_hi"])])
            for i, key in enumerate(("lt80", "lt90", "out5", "gt110")):
                tail_cell(a, P_RNDT, f"{e} P {key}", cells[2 + i], r["prob"][key], marked=True)
            cell(a, P_RNDT, f"{e} mass below quoted", cells[6], r["below"], scale=100.0)
            cell(a, P_RNDT, f"{e} mass above quoted", cells[7], r["above"], scale=100.0)
            gfmt(a, P_RNDT, f"{e} distribution check max(|∫p − 1|, |E[S_T/F] − 1|)", cells[8],
                 max(abs(r["mass"] - 1.0), abs(r["mean"] - 1.0)), floor=QUAD_NOISE)
            gfmt(a, P_RNDT, f"{e} g_min (the slice's certificate)", cells[9], r["g_min"])
    tbl = a.one("#rnd-table-tbl")
    statics(a, P_RNDT, "table caption", tbl.find("caption") if tbl else None, [2, 1, 0], [c.session])
    for th in heads(tbl):
        statics(a, P_RNDT, f"head '{text_of(th)}'", th, [x for _, x, _ in M.THRESHOLDS] + list(M.BAND))
    rnd_table_note(a, c, tr_)
    certified_slices(a, c, tr_)


def rnd_table_note(a: Audit, c: Ctx, tr_: list[dict]) -> None:
    """The table's note: the thresholds beyond the quotes; max |∫p − 1| and max
    |E[S_T/F] − 1| over every slice (bounds: two quadratures agree only to their noise);
    the constants of '|∫p − 1|', '|E − 1|' and 'g_min ≥ 0'; and, with a repair, the
    quotes it moved outside the spread (the residual table's arb_flag)."""
    note = a.one("#rnd-table-note")
    txt = text_of(note)
    bound = re.compile(r"≤ (\S+(?: × 10[⁻⁰¹²³⁴⁵⁶⁷⁸⁹]+)?)")
    found = bound.findall(txt)
    if a.equal(P_RNDT, "note: two error bounds", len(found), 2):
        for b, key in zip(found, ("mass", "mean"), strict=True):
            gfmt(a, P_RNDT, f"note: max |{key} − 1| over every slice", b,
                 max(abs(r[key] - 1.0) for r in tr_), floor=QUAD_NOISE)
    n_beyond = sum(1 for r in tr_ for v in r["prob"].values() if v is None)
    moved = ((c.sj.get("arbitrage") or {}).get("repair") or {}).get("n_perturbed_outside_spread")
    n_out = float(c.mkt["arb_flag"].astype(bool).sum())
    text_nums(a, P_RNDT, "note: thresholds beyond quotes, constants, quotes moved outside the spread",
              bound.sub("≤ ", txt), [n_beyond, 1, 1, 0, *([n_out] if moved else [])])
    a.cover(note)


def tail_cell(a: Audit, panel: str, label: str, el: Tag | None, p: float | None,
              marked: bool = False) -> None:
    """A probability in %, or 'beyond quoted strikes' (``marked``: the cell carries the
    'beyond' class) where the threshold lies outside the quoted range."""
    if p is not None:
        a.num(panel, label, el, p, scale=100.0)
        return
    a.text(panel, label, el, "beyond quoted strikes")
    if marked:
        a.equal(panel, f"{label} marked beyond", "beyond" in ((el.get("class") or []) if el else []), True)


def certified_slices(a: Audit, c: Ctx, tr_: list[dict]) -> None:
    """A slice the certificate passes has g ≥ 0 (a non-negative density) on k ∈ [−3, 2]."""
    ks = np.linspace(-3.0, 2.0, 50001)
    bad = [r["expiry"] for r in tr_ if r["certified"] and float(np.min(M.durrleman_g(ks, r["sl"]))) < -1e-12]
    a.equal(P_RNDT, "certified slices: Durrleman g ≥ 0 on k ∈ [−3, 2] (5e4 points)", bad, [])


# ── the implied-vol heatmap, residual heatmaps ────────────────────────────────
P_IV, P_RES = "#iv-heat", "#residuals"


def heat_rows(a: Audit, panel: str, tbl: Tag | None, c: Ctx, q: pd.DataFrame, what: str,
              value: Callable[[pd.DataFrame], float],
              detail: Callable[[pd.DataFrame, float], list[float]]) -> list[float]:
    """A bucket heatmap: column heads, one row per expiry, each cell the value of its
    quotes (or empty), with its detail; returns the cell values."""
    hd = heads(tbl)
    texts(a, panel, f"{what} column heads", hd[1:], [M.bucket_label(b, True) for b in c.used])
    trs = rows(tbl)
    out: list[float] = []
    qb = q.assign(kb=M.bucket(q["k"]))
    if not same_len(a, panel, f"{what} rows", trs, c.exps):
        return out
    for tr, (e, T) in zip(trs, c.exps.items()):
        a.equal(panel, f"{what} {e} data-exp", tr.get("data-exp"), e)
        days_nums(a, panel, f"{what} {e} row head (median T_days)", tr.find("th"), [T], [e])
        td = tds(tr)
        if not same_len(a, panel, f"{what} {e} cells", td, c.used):
            continue
        for t, b in zip(td, c.used):
            g = qb[(qb["expiry"] == e) & (qb["kb"] == b)]
            lab = f"{what} {e} {M.bucket_label(b, True)}"
            if g.empty:
                a.equal(panel, f"{lab} empty", "empty" in (t.get("class") or []), True)
                a.text(panel, f"{lab} empty-cell detail", t.select_one(".sr-only"),
                       f"{e} · ln(K/F) {M.bucket_label(b, False)}: no quotes")
                a.equal(panel, f"{lab} data-detail", t.get("data-detail"),
                        f"{e} · ln(K/F) {M.bucket_label(b, False)}: no quotes")
                continue
            v = value(g)
            out.append(v)
            a.num(panel, lab, t, v)
            text_nums(a, panel, f"{lab} detail", str(t.get("data-detail") or ""), detail(g, v))
    return out


def iv_heat(a: Audit, c: Ctx) -> None:
    """Median mid IV per expiry and ln(K/F) bucket of every quote, and the 3D view's points."""
    nums(a, P_IV, "panel meta (quotes)", meta(a, "iv-heat"), [len(c.mkt)])
    vals = heat_rows(a, P_IV, a.one("#iv-heatmap"), c, c.mkt, "mid IV",
                     lambda g: float(g["mid_iv"].median() * 100),
                     lambda g, v: [v, len(g), float(g["half_spread_vp"].median())])
    if vals:
        nums(a, P_IV, "colour ramp lo/hi (cells shown)", a.one("#iv-heat .ramp"), [min(vals), max(vals)])
    note = a.one("#iv-heat p.note")
    flt = (c.sj.get("universe") or {}).get("universe_filters") or {}
    unote = prose(flt.get("note") or "the calibrator’s universe")
    a.text(P_IV, "universe filter note (surface.json)", note, unote, contains=True)
    want = [v for v, _, _ in numbers_in(unote)] + [float(c.mkt["half_spread_vp"].median()), M.K_EDGES[-1],
                                                    len(c.mkt)]
    nums(a, P_IV, "note: filter note, median half-spread, ±0.30 wings, quotes in the cells", note, want)
    if M.num(flt.get("min_T_days")) is not None:
        a.equal(P_IV, "every quote at least min_T_days out (universe filter)",
                bool((c.mkt["T_days"] >= float(flt["min_T_days"]) - 1e-9).all()), True)
    surface_3d(a, c, vals)


def surface_3d(a: Audit, c: Ctx, vals: list[float]) -> None:
    """The 3D view: one point per expiry and bucket at the bucket's midpoint (an open
    wing at its quotes' median k), the expiry's T_days and the cell's median mid IV;
    shown from 9 filled cells."""
    sp = spec(a, "surface3d-spec")
    qb = c.mkt.assign(kb=M.bucket(c.mkt["k"]))
    x3 = {b: ((M.K_EDGES[b - 1] + M.K_EDGES[b]) / 2 if 0 < b < M.N_BUCKETS - 1 else
              float(qb.loc[qb["kb"] == b, "k"].median())) for b in c.used}
    pts = []
    for e, T in c.exps.items():
        for b in c.used:
            g = qb[(qb["expiry"] == e) & (qb["kb"] == b)]
            pts.append([x3[b], float(T), float(g["mid_iv"].median() * 100) if len(g) else None])
    a.equal(P_IV, "3D view shown (at least 9 filled cells)", sp is not None,
            sum(1 for p in pts if p[2] is not None) >= 9)
    if sp is None:
        return
    got = [p if isinstance(p, list) and len(p) == 3 else [None] * 3 for p in sp.get("points") or []]
    if same_len(a, P_IV, "3D view points", got, pts):
        for i, (what, dec) in enumerate((("bucket x (4 decimals)", 4), ("T_days", None),
                                         ("median mid IV % (3 decimals)", 3))):
            approx(a, P_IV, f"3D view points: {what}", [p[i] for p in got], [p[i] for p in pts],
                   sig=FULL_SIG if dec is None else SPEC_SIG, decimals=dec)
    approx(a, P_IV, "3D view z range = the heatmap's", [sp.get("zmin"), sp.get("zmax")],
           [min(vals), max(vals)], sig=FULL_SIG)


def residual_heat(a: Audit, c: Ctx) -> None:
    """Each model's median (model − mid)/half-spread per bucket of its out-of-sample quotes."""
    tabs = a.soup.select('#residuals [role="tab"]')
    a.equal(P_RES, "model tabs", [t.get("data-value") for t in tabs], c.s.models)
    for m in c.s.models:
        q = M.oos_rows(c.s, m)
        cap = a.one(f"#resid-heat-{m} > caption")
        nums(a, P_RES, f"{M.label(m)} caption (out-of-sample quotes; |value| ≤ 1)", cap, [len(q), 1])
        heat_rows(a, P_RES, a.one(f"#resid-heat-{m}"), c, q, M.label(m),
                  lambda g: float(g["error_hs"].median()),
                  lambda g, v: [v, len(g), float(g["inside_spread"].astype(bool).mean() * 100)])
    for r in a.soup.select("#residuals .ramp"):
        statics(a, P_RES, "colour ramp ±5 half-spreads, 0 at mid", r, [-RESID_RAMP, RESID_RAMP, 0])


def residual_table(a: Audit, c: Ctx) -> None:
    """The per-quote table's own arithmetic: half-spread, error and z from the bid,
    mid, ask and model IVs, inside_spread from bid ≤ model ≤ ask."""
    for x in c.all:
        r = x.res
        hs = M.half_spread_vp(r)
        ev = (r["model_iv"] - r["mid_iv"]) * 100.0
        ok = r["model_iv"].notna()
        a.equal("residuals.parquet", f"{x.name}: half_spread_vp = (ask IV − bid IV)/2 (1e-6 vol pts)",
                bool(((hs - r["half_spread_vp"]).abs() <= 1e-6).all()), True)
        a.equal("residuals.parquet", f"{x.name}: error_vp = model − mid (1e-6 vol pts)",
                bool(((ev - r["error_vp"]).abs()[ok] <= 1e-6).all()), True)
        # relative: near the 0.01 floor a z of thousands moves by 1e-3 with the
        # stored IVs' last digit (a simulated book reached z ≈ 4,283)
        z = M.z_hs(r)
        a.equal("residuals.parquet", f"{x.name}: error_hs = error / max(half-spread, 0.01) (1e-6 relative)",
                bool(((z - r["error_hs"]).abs()[ok] <= 1e-6 * np.maximum(1.0, z.abs()[ok]) + 1e-3).all()),
                True)
        ins = (r["model_iv"] >= r["bid_iv"]) & (r["model_iv"] <= r["ask_iv"])
        a.equal("residuals.parquet", f"{x.name}: inside_spread = bid ≤ model ≤ ask",
                int((ins != r["inside_spread"].astype(bool))[ok].sum()), 0)
        sizes = r.groupby("model").size()
        a.equal("residuals.parquet", f"{x.name}: every model scored on the same quotes",
                int(sizes.min()), int(sizes.max()))


# ── smile slices ──────────────────────────────────────────────────────────────
P_SL = "#slices"


def slices_panel(a: Audit, c: Ctx) -> None:
    """Tabs, quotes and median half-spread per expiry, and each expiry's bucket table of
    bid / mid / ask and every model's median IV; the packed quotes against the table."""
    tabs = a.soup.select('#slices [role="tab"]')
    a.equal(P_SL, "expiry tabs", [t.get("data-value") for t in tabs], list(c.exps))
    default = min(c.exps.items(), key=lambda kv: abs(kv[1] - M.CM_TENORS[0][0]))[0] if c.exps else None
    a.equal(P_SL, "selected tab: the expiry nearest 30 days",
            [t.get("data-value") for t in tabs if t.get("aria-selected") == "true"], [default])
    for t, (e, T) in zip(tabs, c.exps.items()):
        days_nums(a, P_SL, f"tab {e} days", t, [T])
    hs = c.mkt.groupby("expiry")["half_spread_vp"].median()
    trs = rows(chart_table(a, "chart-slices"))
    if same_len(a, P_SL, "quotes per expiry rows", trs, c.exps):
        for tr, (e, T) in zip(trs, c.exps.items()):
            td = tds(tr)
            a.date(P_SL, f"{e} expiry", td[0], e)
            cell(a, P_SL, f"{e} days", td[1], T)
            cell(a, P_SL, f"{e} quotes", td[2], float((c.mkt["expiry"] == e).sum()))
            cell(a, P_SL, f"{e} median half-spread", td[3], float(hs[e]))
    for e, T in c.exps.items():
        tbl = chart_group(a, "chart-slices", e)
        nums(a, P_SL, f"{e} group caption", tbl.find("caption") if tbl else None, [T], [e])
        hd = heads(tbl)
        texts(a, P_SL, f"{e} group model columns", hd[5:], [M.label(m) for m in c.overlay])
        q = c.mkt[c.mkt["expiry"] == e].sort_values("k", kind="mergesort")
        kb = M.bucket(q["k"])
        model_iv = {}
        for m in c.overlay:
            qm = c.s.res[(c.s.res["model"] == m) & (c.s.res["expiry"] == e)].set_index("k")["model_iv"]
            model_iv[m] = qm[~qm.index.duplicated()].reindex(q["k"].to_numpy()).to_numpy(float) * 100.0
        bs = [b for b in c.used if (kb == b).any()]
        trs = rows(tbl)
        if not same_len(a, P_SL, f"{e} bucket rows", trs, bs):
            continue
        for tr, b in zip(trs, bs):
            sel = kb == b
            g = q[sel]
            td = tds(tr)
            lab = f"{e} {M.bucket_label(b, False)}"
            a.text(P_SL, f"{lab} bucket", td[0], M.bucket_label(b, False))
            cell(a, P_SL, f"{lab} quotes", td[1], float(len(g)))
            for i, col in enumerate(("bid_iv", "mid_iv", "ask_iv")):
                cell(a, P_SL, f"{lab} median {col}", td[2 + i], float(g[col].median() * 100))
            for i, m in enumerate(c.overlay):
                v = pd.Series(model_iv[m][sel]).median()
                cell(a, P_SL, f"{lab} median {M.label(m)} IV", td[5 + i] if 5 + i < len(td) else None,
                     M.num(v))
    slices_prose(a, c)
    packed_quotes(a, c)


def slices_prose(a: Audit, c: Ctx) -> None:
    """The smile chart's description (the anchors only when Heston was fitted to some
    quotes; the overlay models and the two shown at first) and the quote readout."""
    hrows = c.s.model_rows(c.hkey) if c.hkey else None
    anchors = hrows is not None and "in_sample" in hrows and bool(hrows["in_sample"].astype(bool).any())
    first = {c.hkey, M.HEADLINE_SURFACE} & set(c.overlay) or set(c.overlay[:2])
    shown = [M.label(m) for m in c.overlay if m in first]
    desc = a.one("#chart-slices-desc")
    a.text(P_SL, "chart description: anchors, overlay models, the ones shown", desc,
           "bid–ask bars, mids" + (" and the Heston fit's anchors (diamonds)" if anchors else "")
           + f"; model IVs at the quoted strikes for {', '.join(M.label(m) for m in c.overlay)} "
           f"({' and '.join(shown)} shown, the others behind their toggles)", contains=True)
    nums(a, P_SL, "chart description: the forward at ln(K/F) = 0, the ±1 band", desc, [0, 1])
    statics(a, P_SL, "quote readout", a.one("#chart-slices-readout"), [READOUT_PAGE])


def packed_quotes(a: Audit, c: Ctx) -> None:
    """The smile chart's packed quotes (the quote inspector's data) against the residual
    table: forward, days, strikes, rights, bid / mid / ask IV, each model's IV and flags."""
    sp = spec(a, "chart-slices-spec")
    groups = ((sp or {}).get("quotes") or {}).get("groups") or {}
    a.equal(P_SL, "packed expiries", list(groups), list(c.exps))
    bad: dict[str, list[str]] = {}
    for e in c.exps:
        if e in groups:
            for name, ok in packed_group(c, e, groups[e]).items():
                if not ok:
                    bad.setdefault(name, []).append(e)
    fields = ["forward", "days", "strikes", "rights", "mid IV", "bid IV", "ask IV",
              *(f"{M.label(m)} {w}" for m in c.overlay for w in ("IV", "fit/inside flags")),
              "arbitrage flags"]
    for f in fields:
        a.equal(P_SL, f"packed quotes: {f} = residual table, every expiry", bad.get(f, []), [])


def packed_group(c: Ctx, e: str, g: dict) -> dict[str, bool]:
    """One expiry's packed block, field by field: does it unpack to the residual table?
    IVs are packed as integers (mid in 10⁻⁵ vol pts, bid / ask off the mid in 10⁻⁸,
    each model off the mid in 10⁻⁴), so each must agree to half its unit."""
    q = c.mkt[c.mkt["expiry"] == e].sort_values("k", kind="mergesort")
    strikes = [int(g["K0"])]
    for d in g["dK"]:
        strikes.append(strikes[-1] + int(d) * int(g["step"]))
    mid = q["mid_iv"].to_numpy(float) * 100.0
    packed_mid = np.asarray(g["mid"], float) / PACK_MID
    out = {"forward": abs(float(g["F"]) - float(q["forward"].iloc[0])) <= 1e-9 * float(g["F"]),
           "days": abs(float(g["T"]) - float(q["T_days"].iloc[0])) <= 1e-9,
           "strikes": strikes == [round(k) for k in q["strike"]],
           "rights": g["r"] == "".join(str(x)[0].upper() for x in q["right"]),
           "mid IV": bool(np.allclose(packed_mid, mid, atol=half(PACK_MID), rtol=0)),
           "bid IV": bool(np.allclose(mid - np.asarray(g["db"], float) / PACK_SPREAD,
                                      q["bid_iv"].to_numpy(float) * 100, atol=half(PACK_SPREAD), rtol=0)),
           "ask IV": bool(np.allclose(mid + np.asarray(g["da"], float) / PACK_SPREAD,
                                      q["ask_iv"].to_numpy(float) * 100, atol=half(PACK_SPREAD), rtol=0))}
    keys = list(zip(q["strike"].to_numpy(float), q["right"].astype(str), strict=True))
    for m in c.overlay:
        r = c.s.res[(c.s.res["model"] == m) & (c.s.res["expiry"] == e)].drop_duplicates(["strike", "right"])
        r = r.set_index(["strike", "right"]).reindex(keys)
        want = r["model_iv"].to_numpy(float) * 100.0
        got = np.array([np.nan if x is None else x for x in g["m"].get(m, [])], float) / PACK_MODEL
        out[f"{M.label(m)} IV"] = len(got) == len(want) and bool(
            np.allclose(mid + got, want, atol=half(PACK_MODEL), rtol=0, equal_nan=True))
        st = (r["in_sample"].fillna(False).astype(bool).to_numpy() * 1
              + r["inside_spread"].fillna(False).astype(bool).to_numpy() * 2)
        out[f"{M.label(m)} fit/inside flags"] = g["s"].get(m) == "".join(str(int(x)) for x in st)
    dq = np.abs(q["repair_dprice"].to_numpy(float))
    fl = (q["arb_exec"].astype(bool).to_numpy() * 1 + (np.isfinite(dq) & (dq > M.REPAIR_TOL)) * 2
          + q["arb_flag"].astype(bool).to_numpy() * 4)
    out["arbitrage flags"] = g["f"] == "".join(str(int(x)) for x in fl)
    return out


def half(scale: float) -> float:
    """Half the unit of a value packed as an integer count of 1/scale (and float slack)."""
    return 0.5 / scale * (1.0 + 1e-6) + 1e-12


# ── the scorecard, by-expiry chart ────────────────────────────────────────────
P_SC, P_BX = "#scorecard", "#by-expiry"


def scorecard(a: Audit, c: Ctx) -> None:
    """The scorecard table against scorecard.json and the residual table; the notes and
    model descriptions as written."""
    if not c.sc:
        return
    n_oos = {oos_stats(c, m)["n"] for m in c.score_models}
    title = text_of(a.one("#scorecard-t"))
    a.equal(P_SC, "title says 'same quotes' only when every model is scored on as many quotes",
            "same quotes, same metric" in title, len(n_oos) == 1)
    a.text(P_SC, "panel meta", meta(a, "scorecard"), "source: scorecard.json")
    for th in heads(a.one("#scorecard-table")):
        statics(a, P_SC, f"head '{text_of(th)}'", th, [WITHIN_HS])
    trs = rows(a.one("#scorecard-table"))
    if same_len(a, P_SC, "rows", trs, c.score_models):
        for tr, m in zip(trs, c.score_models):
            row = c.sc_model(m) or {}
            o = (row.get("metrics") or {}).get("out_of_sample") or {}
            r = oos_stats(c, m)
            td = tds(tr)
            lab = M.label(m)
            a.text(P_SC, f"{lab} model", td[0], lab)
            a.equal(P_SC, f"{lab} highlighted (Heston)", "em" in (tr.get("class") or []),
                    m in ("heston_v2", "heston"))
            cell(a, P_SC, f"{lab} params (scorecard.json)", td[1], M.num(row.get("n_params")))
            cell(a, P_SC, f"{lab} fitted to (residual in_sample)", td[2],
                 float(c.s.model_rows(m)["in_sample"].astype(bool).sum()))
            a.num(P_SC, f"{lab} fitted to (scorecard.json)", td[2], M.num(row.get("n_in_sample")))
            for i, key in enumerate(("n", "median_abs_hs", "inside_spread_pct", "within_2hs_pct", "rmse_vp",
                                     "mean_error_vp")):
                a.num(P_SC, f"{lab} {key} (scorecard.json)", td[3 + i], M.num(o.get(key)))
                a.num(P_SC, f"{lab} {key} (residual table)", td[3 + i], r[key])
            dg = row.get("dense_grid") or {}
            if dg:
                nums(a, P_SC, f"{lab} dense-grid butterfly / calendar (scorecard.json)", td[9],
                     [M.num(dg.get("butterfly_violations")), M.num(dg.get("calendar_violations"))])
                sdg = (((c.sj.get("surfaces") or {}).get(m) or {}).get("dense_grid") or {})
                if sdg:
                    a.equal(P_SC, f"{lab} dense-grid counts: scorecard.json = surface.json",
                            [dg.get("butterfly_violations"), dg.get("calendar_violations")],
                            [sdg.get("butterfly_violations"), sdg.get("calendar_violations")])
            else:
                a.text(P_SC, f"{lab} dense grid", td[9], "—")
    notes = list(c.sc.get("notes") or [])
    ps = a.soup.select("#scorecard .panel-body > p.note")
    for i, n_ in enumerate(notes):
        a.text(P_SC, f"note {i + 1} (scorecard.json)", ps[i] if i < len(ps) else None, prose(n_))
    if notes:
        scorecard_note_counts(a, c, notes[0])
    dds = a.soup.select("#scorecard details.method dd")
    for i, m in enumerate(c.score_models):
        a.text(P_SC, f"{M.label(m)} description (scorecard.json)", dds[i] if i < len(dds) else None,
               prose((c.sc_model(m) or {}).get("description") or "—"))


def scorecard_note_counts(a: Audit, c: Ctx, note: str) -> None:
    """The producer's first note states the universe, each model's out-of-sample count and
    the common out-of-sample count: each against the residual table."""
    m = re.search(r"cleaned quotes, ([\d,]+)\)", note)
    if m:
        a.equal(P_SC, "note: universe (residual quotes)", int(m.group(1).replace(",", "")), len(c.mkt))
    for model, n in re.findall(r"\b([a-z_0-9]+) ([\d,]+)(?=[,)])", note):
        if model in c.s.models:
            a.equal(P_SC, f"note: {model} out-of-sample count (residual table)", int(n.replace(",", "")),
                    int(oos_stats(c, model)["n"]))
    m = re.search(r"same ([\d,]+) quotes", note)
    if m:
        insample = c.s.res.pivot_table(index=["expiry", "strike", "right"], columns="model",
                                       values="in_sample", aggfunc="first").fillna(False).astype(bool)
        a.equal(P_SC, "note: common out-of-sample count (quotes no model was fitted to)",
                int(m.group(1).replace(",", "")), int((~insample.any(axis=1)).sum()))


def by_expiry(a: Audit, c: Ctx) -> None:
    """Median |z| per expiry of each model's out-of-sample quotes."""
    tbl = chart_table(a, "chart-by-expiry")
    hd = heads(tbl)
    if same_len(a, P_BX, "heads", hd[1:], c.exps):
        for th, (e, T) in zip(hd[1:], c.exps.items()):
            days_nums(a, P_BX, f"head {e}", th, [T], [e])
    trs = rows(tbl)
    sp = spec(a, "chart-by-expiry-spec")
    if sp is not None:
        a.equal(P_BX, "chart expiries", sp.get("exps"), list(c.exps))
        cats = sp.get("xs") or []
        if same_len(a, P_BX, "chart categories", cats, c.exps):
            for cat, (e, T) in zip(cats, c.exps.items()):
                days_nums(a, P_BX, f"chart category {e}", str(cat), [T], [e])
    if not same_len(a, P_BX, "rows", trs, c.score_models):
        return
    for tr, m in zip(trs, c.score_models):
        q = M.oos_rows(c.s, m)
        z = q["error_hs"].abs()
        med, cnt = z.groupby(q["expiry"]).median(), z.groupby(q["expiry"]).count()
        vals = [float(med[e]) if cnt.get(e, 0) else None for e in c.exps]
        td = tds(tr)
        a.text(P_BX, f"{M.label(m)} row", td[0], M.label(m))
        for t, e, v in zip(td[1:], c.exps, vals):
            cell(a, P_BX, f"{M.label(m)} {e} median |z|", t, v)
        approx(a, P_BX, f"chart-by-expiry series {M.label(m)}", series(sp, M.label(m)), vals)
    legend(a, P_BX, "chart-by-expiry", [M.label(m) for m in c.score_models])
    statics(a, P_BX, "chart description", a.one("#chart-by-expiry-desc"), [1])


# ── static arbitrage ──────────────────────────────────────────────────────────
P_ARB = "#arbitrage"


def arbitrage(a: Audit, c: Ctx) -> None:
    """The constraint-row counts, the repair, the largest mid violations with the
    bad quote each one holds, and the surfaces' dense-grid counts."""
    arb = c.sj.get("arbitrage") or {}
    if not arb:
        a.text(P_ARB, "no screen", a.one("#arbitrage .panel-body > p.note"),
               "No arbitrage screen in surface.json for this session.")
        return
    mk = c.mkt
    cs = str(arb.get("constraint_set") or "CRW 2020").split(":")[0]
    tbl = a.one("#arb-table")
    nums(a, P_ARB, "caption: constraint set, quotes (residual table)", tbl.find("caption") if tbl else None,
         [v for v, _, _ in numbers_in(cs)] + [len(mk)])
    a.text(P_ARB, "caption names the constraint set", tbl.find("caption") if tbl else None, f"({cs})",
           contains=True)
    a.equal(P_ARB, "screen's quote count = residual quotes", M.num(arb.get("n_quotes")), float(len(mk)))
    keys = list((arb.get("n_rows") or {}).keys()) or list((arb.get("mid_violations") or {}).keys())
    trs = rows(tbl)
    if same_len(a, P_ARB, "constraint rows", trs, keys):
        for tr, k in zip(trs, keys):
            td = tds(tr)
            a.text(P_ARB, f"{k} label", td[0], M.ARB_LABELS.get(k, k.replace("_", " ")))
            for i, blk in enumerate(("n_rows", "mid_violations", "executable_violations")):
                cell(a, P_ARB, f"{k} {blk}", td[1 + i], M.num((arb.get(blk) or {}).get(k)))
            a.equal(P_ARB, f"{k} bid/ask cell marked when violated", "neg" in (td[3].get("class") or []),
                    bool((arb.get("executable_violations") or {}).get(k)))
    rp = arb.get("repair") or {}
    note = next((p for p in a.soup.select("#arbitrage p.note") if "l1 repair" in text_of(p)), None)
    a.equal(P_ARB, "repair note shown", note is not None, bool(rp))
    if rp:
        dq = np.abs(mk["repair_dprice"].to_numpy(float))
        n_moved = float((np.isfinite(dq) & (dq > M.REPAIR_TOL)).sum())
        n_out = float(mk["arb_flag"].astype(bool).sum())
        nums(a, P_ARB, "repair: quotes moved, moved outside the spread (residual table), largest move",
             note, [n_moved, n_out, M.num(rp.get("max_abs_dprice_half_spreads"))])
        a.equal(P_ARB, "repair counts: surface.json = residual table",
                [M.num(rp.get("n_perturbed")), M.num(rp.get("n_perturbed_outside_spread"))], [n_moved, n_out])
    lv = M.violations(arb)
    h3 = a.one("#arbitrage h3.sub-head")
    a.equal(P_ARB, "largest violations shown", h3 is not None, bool(lv))
    if lv:
        mids = sum(int(v) for v in (arb.get("mid_violations") or {}).values() if M.num(v) is not None)
        more = len(lv) - LV_SHOWN
        nums(a, P_ARB, "largest violations heading", h3,
             [len(lv), mids] + ([LV_SHOWN, more] if more > 0 else []))
        cap = a.one("#largest-violations > caption")
        nums(a, P_ARB, "caption ($ per SPX contract)", cap, [M.SPX_MULTIPLIER])
        got = rows(a.one("#largest-violations")) + rows(a.one("#largest-violations-rest"))
        if same_len(a, P_ARB, "violation rows", got, lv):
            for i, (tr, v) in enumerate(zip(got, lv)):
                violation_row(a, i + 1, tr, v)
        if more > 0:
            nums(a, P_ARB, "rest summary", a.one("#arbitrage details.method > summary"), [more, LV_SHOWN + 1,
                                                                                         len(lv)])
            nums(a, P_ARB, "rest caption", a.one("#largest-violations-rest > caption"),
                 [LV_SHOWN + 1, len(lv)])
    dense = next((p for p in a.soup.select("#arbitrage p.note") if "Dense-grid checks" in text_of(p)), None)
    surf = c.sj.get("surfaces") or {}
    a.equal(P_ARB, "dense-grid note shown (surfaces in surface.json)", dense is not None, bool(surf))
    if surf:
        grids = [((v or {}).get("dense_grid") or {}) for v in surf.values()]
        want = [M.num(g.get(f"{kind}_violations")) for g in grids for kind in ("butterfly", "calendar")]
        nums(a, P_ARB, "dense-grid butterfly / calendar violations (surface.json)", dense,
             [x for x in want if x is not None])
        a.equal(P_ARB, "dense-grid surfaces, in order",
                re.findall(r"(?:: |; )(.+?) \S+ butterfly", text_of(dense)), [M.label(k) for k in surf])


def violation_row(a: Audit, rank: int, tr: Tag, v: M.Violation) -> None:
    """One largest-violation row: constraint, legs by expiry ('+0.50 P6260'), shortfall,
    and either its dollars per contract and whether it survives at bid/ask, or the
    bad quote that explains it."""
    td = tds(tr)
    lab = f"violation {rank}"
    if not same_len(a, P_ARB, f"{lab} cells", td, range(5)):
        return
    a.text(P_ARB, f"{lab} constraint", td[0], v.type.replace("_", " "))
    want = []
    for e in dict.fromkeys(lg[0] for lg in v.legs):
        want.append(e)
        for _, rt, k, w in (lg for lg in v.legs if lg[0] == e):
            want.append(f"{sgn(w, '.0f' if abs(abs(w) - 1.0) < 1e-12 else '.2f')} {rt}{k:.0f}")
    texts(a, P_ARB, f"{lab} legs (expiry, weight and strike)", td[1].select("span.nowrap"), want)
    cell(a, P_ARB, f"{lab} shortfall (idx pts)", td[2], v.shortfall)
    if v.bad:
        a.text(P_ARB, f"{lab} $ (none: bad quote)", td[3], "—")
        e, rt, k = v.bad
        a.text(P_ARB, f"{lab} bad quote", td[4], f"bad quote: {e} {rt}{k:.0f}")
    else:
        cell(a, P_ARB, f"{lab} $ per contract (shortfall × 100)", td[3],
             None if v.shortfall is None else v.shortfall * M.SPX_MULTIPLIER)
        a.text(P_ARB, f"{lab} at bid/ask", td[4], "executable" if v.executable else "mids only")


# ── the VIX family, the variance-swap strip ───────────────────────────────────
P_VIX, P_VS = "#vix-replication", "#varswap"


def vix_rows(c: Ctx) -> list[tuple[str, dict]]:
    """The VIX-family indices of vix_replication.json, in the family's order."""
    ixs = c.vx.get("indices") or {}
    return [(t, ixs[t]) for t, _ in M.VIX_FAMILY if isinstance(ixs.get(t), dict) and ixs[t]]


def artifact_close(ix: dict) -> float | None:
    """The Cboe close vix_replication.json records for an index ({close: …} or a number)."""
    off = ix.get("official")
    return M.num(off.get("close")) if isinstance(off, dict) else M.num(off)


@dataclass
class GapSplit:
    """An index's gap to the Cboe close split as the page states it."""
    gap: float | None                  # replicated − the artifact's close (recomputed)
    short: float                       # missing-strike shortfall = −truncation_pts
    unexp: float                       # unexplained = −residual_pts
    far: float | None                  # stored quotes beyond the fitted range − the wing
    inside: float | None               # market − surface inside the fitted range
    fit: tuple[float, float] | None    # the fitted K/F range


def vix_split(ix: dict) -> GapSplit | None:
    """The split, shown for a replicated index whose gap_attribution is available and
    whose two parts add up to the stored gap (within 1e-6 index points)."""
    rep = M.num(ix.get("replicated"))
    raw = ix.get("gap_attribution")
    ga: dict = raw if isinstance(raw, dict) else {}
    tr, rs = M.num(ga.get("truncation_pts")), M.num(ga.get("residual_pts"))
    stored = M.num(ix.get("gap_vs_official"))
    if rep is None or not ga.get("available") or tr is None or rs is None or stored is None:
        return None
    if abs(-tr - rs - stored) > GAP_SPLIT_TOL:
        return None
    off = artifact_close(ix)
    fr = ga.get("fit_range_K_over_F")
    lo, hi = (M.num(fr[0]), M.num(fr[1])) if isinstance(fr, list) and len(fr) == 2 else (None, None)
    return GapSplit(None if off is None else rep - off, -tr, -rs,
                    M.num(ga.get("far_wing_market_minus_model_pts")), M.num(ga.get("fit_inside_range_pts")),
                    (lo, hi) if lo is not None and hi is not None else None)


def vix_terms(ix: dict) -> dict:
    """The terms of an index's headline method (Cboe rules without one)."""
    method = ix.get("headline_method")
    tm = ix.get(method) if method else ix.get("cboe_rules")
    return tm if isinstance(tm, dict) else {}


def vix_replication(a: Audit, c: Ctx) -> None:
    """Replicated VIX-family levels against the DB's Cboe closes, each index's terms and
    near weight, the gap split, ATM / VIX and the β estimator on the same chain."""
    vr = vix_rows(c)
    if not vr:
        return
    trs = rows(a.one("#vix-rep-table"))
    if same_len(a, P_VIX, "index rows", trs, vr):
        for tr, (t, ix) in zip(trs, vr):
            td = tds(tr)
            rep, off = M.num(ix.get("replicated")), c.official(t, c.session)
            a.text(P_VIX, f"{t} index", td[0], t)
            cell(a, P_VIX, f"{t} replicated (vix_replication.json)", td[1], rep)
            cell(a, P_VIX, f"{t} Cboe close (DB)", td[2], off)
            a.num(P_VIX, f"{t} Cboe close (vix_replication.json)", td[2], artifact_close(ix))
            cell(a, P_VIX, f"{t} gap = replicated − close", td[3],
                 None if rep is None or off is None else rep - off)
    splits = {t: vix_split(ix) for t, ix in vr}
    trs = rows(a.one("#vix-terms-table"))
    if same_len(a, P_VIX, "terms rows", trs, vr):
        for tr, (t, ix) in zip(trs, vr):
            td = tds(tr)
            tm = vix_terms(ix)
            near, nxt = tm.get("near") or {}, tm.get("next") or {}
            if near and nxt:
                nums(a, P_VIX, f"{t} near / next term days", td[1],
                     [M.num(near.get("days")), M.num(nxt.get("days"))])
            else:
                a.text(P_VIX, f"{t} near / next", td[1], "—")
            n1, n2 = M.num(near.get("minutes")), M.num(nxt.get("minutes"))
            cm = M.num(tm.get("constant_maturity_days"))
            weighted = M.num(tm.get("near_weight")) is not None
            if weighted and n1 is not None and n2 is not None and cm is not None and n2 != n1:
                a.num(P_VIX, f"{t} near weight (Cboe: (N₂ − N_T)/(N₂ − N₁) in minutes)", td[2],
                      (n2 - cm * 1440.0) / (n2 - n1))
            else:
                cell(a, P_VIX, f"{t} near weight", td[2], M.num(tm.get("near_weight")))
            sp = splits[t]
            cell(a, P_VIX, f"{t} missing-strike shortfall", td[3], sp.short if sp else None)
            cell(a, P_VIX, f"{t} unexplained", td[4], sp.unexp if sp else None)
    vix_terms_note(a, vr, splits)
    beta = c.vx.get("beta") or {}
    note = a.one("#atm-vix-note")
    a.equal(P_VIX, "ATM / VIX note shown (vix_replication.json beta)", note is not None,
            "atm_over_vix_official" in beta)
    if note is not None:
        off = c.official("VIX", c.session)
        be = M.beta_by_session(c.store).get(c.session)
        ratio = M.num(beta.get("atm_vol_30d_quotes"))
        want = [None if ratio is None or not off else ratio / off] + ([30, 76, be["ratio"]] if be else [])
        nums(a, P_VIX, "ATM / VIX: the VIX terms' ATM mid over the DB close; β estimator (30-day, Black-76)",
             note, want)
        a.num(P_VIX, "ATM / VIX (vix_replication.json)", note, M.num(beta.get("atm_over_vix_official")))
    deviations(a, c, splits.get("VIX"))


def vix_terms_note(a: Audit, vr: list[tuple[str, dict]], splits: dict[str, GapSplit | None]) -> None:
    """'[0, 1]', each index that is not replicated and why, and with a split the 16:15 ET
    print and the far-wing differences, positive ones first."""
    note = a.one("#vix-terms-note")
    want: list[float | None] = [0, 1]
    for t, ix in vr:
        if M.num(ix.get("replicated")) is None:
            tm = vix_terms(ix)
            why = ("its nearest candidate pair would extrapolate" if tm.get("extrapolated")
                   else str(tm.get("reason") or ""))
            a.text(P_VIX, f"{t} not replicated", note, f"{t} is not replicated" + (f": {why}" if why else ""),
                   contains=True)
            want += [v for v, _, _ in numbers_in(why)]
    shown = [sp for sp in splits.values() if sp]
    far = [sp.far for sp in shown if sp.far is not None]
    if shown:
        want += [16, 15, *(f for f in far if f > 0), *(f for f in far if f < 0)]
    nums(a, P_VIX, "terms note (not replicated, far-wing market − model per index)", note, want)


def deviations(a: Audit, c: Ctx, vix: GapSplit | None) -> None:
    """'Where the replication departs from Cboe': the producer's deviations and notes as
    written, and the gap-attribution sentence rebuilt from the recompute."""
    lis = a.soup.select("#vix-replication details.method ul.prose > li")
    devs = [prose(x) for x in c.vx.get("deviations") or []]
    notes = [prose(x) for x in c.vx.get("gap_notes") or [] if not str(x).startswith("Gap attribution (")]
    if not same_len(a, P_VIX, "departure items", lis, devs + notes + ([""] if vix else [])):
        return
    for i, (li, want) in enumerate(zip(lis, devs + notes)):
        a.text(P_VIX, f"departure {i + 1} (vix_replication.json)", li, want)
    if vix:
        fit = list(vix.fit or ())
        sentence: list[float | None] = [vix.gap, vix.short, vix.unexp]
        if vix.far is not None and abs(vix.far) >= FAR_WING_IN_LINE:
            sentence += [*fit, abs(vix.far)]
        if vix.inside is not None:
            sentence += [*fit, vix.inside]
        sentence += [vix.unexp]
        if vix.gap is not None and abs(vix.gap) > 1e-12:
            sentence += [100.0 * vix.unexp / vix.gap]
        nums(a, P_VIX, "gap attribution sentence (rebuilt from the split)", lis[-1], [*sentence, 16, 15])
    near_next_notes(a, c, lis[len(devs):len(devs) + len(notes)])


def near_next_notes(a: Audit, c: Ctx, lis: list[Tag]) -> None:
    """The producer's 'near-term' / 'next-term' notes against the VIX's headline terms."""
    ix = (c.vx.get("indices") or {}).get("VIX") or {}
    tm = ix.get(ix.get("headline_method") or "cboe_rules") or {}
    for li in lis:
        m = re.match(r"(near|next)-term (\d{4}-\d\d-\d\d): (\d+) puts / (\d+) calls, strip K/F", text_of(li))
        if not m:
            continue
        t = tm.get(m.group(1)) or {}
        nums(a, P_VIX, f"{m.group(1)}-term strikes and K/F range (the VIX's terms)", li,
             [M.num(t.get("n_puts")), M.num(t.get("n_calls")), M.num(t.get("lowest_K_over_F")),
              M.num(t.get("highest_K_over_F"))], [str(t.get("expiry"))[:10]])


def varswap(a: Audit, c: Ctx) -> None:
    """The variance-swap strip: VS and ATM vol per expiry, their difference and ratio
    recomputed from the two, the strike counts; the Cboe closes as markers."""
    strip = sorted(c.vx.get("strip") or [], key=lambda r: M.num(r.get("days")) or 0)
    if not strip:
        return
    trs = rows(chart_table(a, "chart-strip"))
    if same_len(a, P_VS, "strip rows", trs, strip):
        for tr, r in zip(trs, strip):
            td = tds(tr)
            e = str(r.get("expiry"))[:10]
            vs, atm = M.num(r.get("var_swap_vol")), M.num(r.get("atm_vol_quotes"))
            a.date(P_VS, f"{e} expiry", td[0], e)
            cell(a, P_VS, f"{e} days", td[1], M.num(r.get("days")))
            cell(a, P_VS, f"{e} var-swap vol", td[2], vs)
            cell(a, P_VS, f"{e} ATM vol (quotes)", td[3], atm)
            cell(a, P_VS, f"{e} VS − ATM", td[4], None if vs is None or atm is None else vs - atm)
            cell(a, P_VS, f"{e} ATM / VS", td[5], None if vs is None or atm is None else atm / vs)
            cell(a, P_VS, f"{e} strikes (puts + calls)", td[6],
                 float((r.get("n_puts") or 0) + (r.get("n_calls") or 0)))
    marks = []
    for t, days in M.VIX_FAMILY:
        ix = (c.vx.get("indices") or {}).get(t) or {}
        off = c.official(t, c.session) if ix.get("official") is not None else None
        if off is None:
            continue
        rec = ix.get(ix.get("headline_method") or "") or ix.get("parity_forward") or {}
        d = M.num(rec.get("constant_maturity_days")) if isinstance(rec, dict) else None
        marks.append((t, d if d is not None else float(days), off))
    sp = spec(a, "chart-strip-spec")
    approx(a, P_VS, "chart-strip series VS vol", series(sp, "Variance-swap vol (Cboe formula)"),
           [[M.num(r.get("days")), M.num(r.get("var_swap_vol"))] for r in strip])
    approx(a, P_VS, "chart-strip series ATM vol", series(sp, "ATM vol (quotes)"),
           [[M.num(r.get("days")), M.num(r.get("atm_vol_quotes"))] for r in strip])
    names = ["Variance-swap vol (Cboe formula)", "ATM vol (quotes)"]
    if marks:
        name = "Cboe close (16:15 ET): VIX9D, VIX, VIX3M, VIX6M"
        approx(a, P_VS, "chart-strip Cboe close markers (DB)", series(sp, name),
               [[d, v] for _, d, v in marks])
        names.append(name)
    legend(a, P_VS, "chart-strip", names)
    statics(a, P_VS, "chart description", a.one("#chart-strip-desc"), [d for _, d in M.VIX_FAMILY])
    note = a.one("#vix-markers")
    a.equal(P_VS, "Cboe markers note shown", note is not None, bool(marks))
    if note is not None:
        ct = c.vx.get("calculation_time_ny")
        timed = f"; the replication is timed {str(ct).replace('T', ' ')} ET" if ct else ""
        a.text(P_VS, "Cboe close time and the replication's", note, f"(16:15 ET{timed})", contains=True)
        want = [16, 15, *(v for v, _, _ in numbers_in(timed)), *(x for _, d, v in marks for x in (v, d))]
        nums(a, P_VS, "Cboe closes (DB) at their constant maturities, calculation time", note, want,
             [m.group(0) for m in DATE.finditer(timed)])
    statics(a, P_VS, "skew-premium note", next((p for p in a.soup.select("#varswap p.note")
                                                if "skew premium" in text_of(p)), None), [30])


# ── every stored session ──────────────────────────────────────────────────────
P_HIST = "#surface-history"


def history(a: Audit, c: Ctx) -> None:
    """One row per session of index.json, newest first, each column recomputed from that
    session's own artifacts and the DB."""
    idx = []
    if (c.store.root / "surface" / "index.json").exists():
        idx = sorted(c.store.json("surface/index.json").get("sessions") or [], key=lambda r: r["session"])
    a.equal(P_HIST, "panel shown (sessions in index.json)", a.one("section#surface-history") is not None,
            bool(idx))
    if not idx:
        return
    by = {x.name: x for x in c.all}
    nums(a, P_HIST, "panel meta", meta(a, "surface-history"), [len(idx)])
    beta = M.beta_by_session(c.store)
    tbl = a.one("#surface-history-table")
    want_hd = (["Session", "Scored quotes", "Expiries"]
               + [f"{M.label(m)} |z|, out of sample" for m in c.s.models]
               + ["Repair moves outside spread", "VIX rep. − close", "VIX-term ATM / VIX"]
               + (["β estimator ATM 30d / VIX"] if beta else []))
    texts(a, P_HIST, "columns", heads(tbl), want_hd)
    trs = rows(tbl)
    if not same_len(a, P_HIST, "rows", trs, idx):
        return
    for tr, r in zip(trs, list(reversed(idx))):
        name = r["session"]
        x = by.get(name)
        td = tds(tr)
        a.date(P_HIST, f"{name} session", td[0], name)
        a.equal(P_HIST, f"{name} highlighted (the page's session)", "em" in (tr.get("class") or []),
                name == c.session)
        cell(a, P_HIST, f"{name} scored quotes (its residual table)", td[1],
             float(len(x.mkt)) if x is not None else M.num(r.get("n_quotes")))
        cell(a, P_HIST, f"{name} expiries (its residual table)", td[2],
             float(len(M.expiries(x))) if x is not None and x.sj is not None else None)
        for i, m in enumerate(c.s.models):
            o = (((c.sc_model(m, x) or {}).get("metrics") or {}).get("out_of_sample") or {}) if x else {}
            v = M.num(o.get("median_abs_hs"))
            cell(a, P_HIST, f"{name} {M.label(m)} OOS median |z| (its scorecard.json)", td[3 + i], v)
            if x is not None and x.sc and v is not None:
                a.num(P_HIST, f"{name} {M.label(m)} OOS median |z| (its residual table)", td[3 + i],
                      oos_stats(c, m, x)["median_abs_hs"])
        j = 3 + len(c.s.models)
        n_out = float(x.mkt["arb_flag"].astype(bool).sum()) if x is not None and "arb_flag" in x.res else None
        cell(a, P_HIST, f"{name} repair moves outside spread (its residual arb_flag)", td[j],
             n_out if n_out is not None else M.num(r.get("arbitrage_quotes_outside_spread")))
        a.num(P_HIST, f"{name} repair moves outside spread (index.json)", td[j],
              M.num(r.get("arbitrage_quotes_outside_spread")))
        gap = c.vix_gap(x) if x is not None and x.vx else None
        cell(a, P_HIST, f"{name} VIX replicated − DB close", td[j + 1], gap)
        b = ((x.vx or {}).get("beta") or {}) if x is not None else {}
        q, off = M.num(b.get("atm_vol_30d_quotes")), c.official("VIX", name)
        cell(a, P_HIST, f"{name} VIX-term ATM / DB VIX close", td[j + 2],
             None if q is None or not off else q / off)
        cell(a, P_HIST, f"{name} VIX-term ATM / VIX (index.json)", td[j + 2],
             M.num(r.get("atm_over_vix_official")))
        if beta:
            be = beta.get(name)
            cell(a, P_HIST, f"{name} β estimator on the DB chain", td[j + 3], be["ratio"] if be else None)
            if be:
                a.equal(P_HIST, f"{name} β chain's implied spot within 0.5% of the close",
                        abs(be["spot_gap"]) <= M.BETA_SPOT_TOL, True)
    nums(a, P_HIST, "note (sessions)", a.one("#surface-history p.note"), [len(idx)])


# ── the page ──────────────────────────────────────────────────────────────────
PANELS: tuple[Callable[[Audit, Ctx], None], ...] = (
    page_head, staleness, state_tiles, score_tiles, arb_tile, delta_grid, surface_change, atm_term,
    vol_cone, rnd, iv_heat, residual_heat, residual_table, slices_panel, scorecard, by_expiry, arbitrage,
    vix_replication, varswap, history)


def run(a: Audit, store: Store) -> None:
    if not M.surface_sessions(store):
        a.equal(P_HEAD, "absent-state panel (no stored surface session)",
                a.one('div.missing[data-artifact-state="absent"]') is not None, True)
        return
    c = Ctx(store)
    for fn in PANELS:
        try:
            fn(a, c)
        except Exception as exc:  # noqa: BLE001  (a crashing panel is recorded as a failed check; the others run)
            a.fail(fn.__name__, "check raised", "no exception", f"{type(exc).__name__}: {exc}")
