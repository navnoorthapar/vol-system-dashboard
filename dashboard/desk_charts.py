"""
desk_charts.py — chart specs and the data tables behind them (pure functions).

A spec is a small JSON document the page embeds next to the chart; desk.js
turns it into an Apache ECharts option. Every spec carries:
  * ``title``     the accessible name of the chart (role="img" + aria-label);
  * ``describe``  one sentence of description (aria-describedby target text);
  * ``table``     the server-rendered table that stands behind the chart for
                  screen readers, for print and when JavaScript or ECharts is
                  unavailable (pre-formatted strings, numbers formatted once);
  * ``csv``       the download file name (the CSV is built from the same data).
Floats are rounded to 6 significant figures (desk_format.round_sig).

Series fields: name, type ('line' | 'bar' | 'scatter' | 'errorbar'), data
(aligned with ``xs`` or [x, y] / [x, lo, hi] pairs), color (a token name such
as 'headline', 'cat0'..'cat5', 'pos', 'neg', 'muted', or '#rrggbb'), width,
dash (True for dashed, 'dotted'), symbol, size, hollow, signed (bars coloured by sign), stack, invisible,
y2, pointLabels (a short label per point, on a line or scatter series; labelPos places a
scatter's), rotate (a scatter symbol's rotation in degrees: 180 turns a triangle down),
group (for tab-filtered charts), legend (show a toggle button), legendKey
(the button's name when one button stands for several series, e.g. a model's
curve on every expiry), hidden (toggled off at first), xFrom (the name of a
series whose x values this one's y-only data shares).

Compact specs (a long daily series is most of a page's weight): x values that
are ISO dates (64 or more) are stored as ``xsDates`` = {start, steps} (day
gaps); ``xsFrom`` names another chart on the page whose x values a chart shares
(a linked chart); a series with ``derive`` = {a, b?, k?, from?, digits?} holds
no data: it is k·a − b of the named series of this spec (or of chart ``from``),
rounded to ``digits``. desk.js (resolveSpec) and ``resolve`` below expand them
the same way, before anything draws, exports or tests the spec.

Quote-level smiles (the quote inspector): ``quotes`` = {groups: {expiry: the
packed block of desk_quotes.encode}, source, strip, models}; a series with
``quote`` = 'ba' | 'mid' | 'model' | 'fit' | 'z' | 'zfit' (and ``model``) holds
no data: it is that field of its group's quotes (desk_quotes.series_data). The
block is kept at full precision (round_sig would move the forward by up to
5·10⁻⁶ of itself, and the forward delta with it); ``strip`` series ('z',
'zfit') draw in the residual strip under the smile.
"""
from __future__ import annotations

import copy
import datetime as dt
import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import desk_quotes
from desk_format import round_sig

CATEGORICAL = ("cat0", "cat1", "cat2", "cat3", "cat4", "cat5")

# ── one meaning per colour, one name per model (P4.5) ─────────────────────────
# Every spec builder takes a model's colour, dash and label from MODEL_ENCODING and
# a role's colour from ROLE_COLOURS. A series that stands for a model carries
# role 'model' and its key (``model``); on a page that draws models no other
# series takes a model colour. Market observables and Cboe indices are
# 'headline', bid–ask 'muted', fit anchors 'cat3'; series indexed by maturity
# take TENOR_RAMP (a lightness ramp, short to long) with a gradient key. No
# series uses 'info' (it is --pos, the gain colour, and links). The tests read
# these tables (test_every_series_follows_the_encoding_registry).
MODEL_ENCODING: dict[str, dict[str, Any]] = {
    "heston_v2": {"color": "cat1", "dash": None, "label": "Heston"},
    "heston": {"color": "cat1", "dash": None, "label": "Heston"},
    "svi_cal": {"color": "cat0", "dash": None, "label": "SVI-cal"},
    "svi": {"color": "cat2", "dash": None, "label": "SVI"},
    "essvi": {"color": "cat4", "dash": None, "label": "eSSVI"},
    "rough_heston": {"color": "cat5", "dash": None, "label": "Rough Heston"},
}
OTHER_MODEL_LABELS = {"ah": "Andreasen–Huge", "bates": "Bates", "quintic": "Quintic OU",
                      "lifted_heston": "Lifted Heston"}
UNKNOWN_MODEL_COLOUR = "cat5"
MODEL_COLOUR_TOKENS = frozenset({e["color"] for e in MODEL_ENCODING.values()} | {UNKNOWN_MODEL_COLOUR})
ROLE_COLOURS = {"market": "headline", "cboe": "headline", "bidask": "muted", "anchor": "cat3",
                "other": "muted"}
TENOR_RAMP = ("ramp0", "ramp1", "ramp2", "ramp3", "ramp4", "ramp5")


def model_colour(key: str) -> str:
    return MODEL_ENCODING.get(key, {}).get("color", UNKNOWN_MODEL_COLOUR)


def model_label(key: str) -> str:
    if key in MODEL_ENCODING:
        return str(MODEL_ENCODING[key]["label"])
    return OTHER_MODEL_LABELS.get(key, key.replace("_", " ").title())


def tenor_colours(n: int) -> list[str]:
    """``n`` maturities, shortest first, spread over TENOR_RAMP (light → dark)."""
    if n <= 1:
        return list(TENOR_RAMP[:n])
    return [TENOR_RAMP[round(i * (len(TENOR_RAMP) - 1) / (n - 1))] for i in range(n)]
COMPACT_DATES_MIN = 64
_ISO = re.compile(r"\d{4}-\d{2}-\d{2}")


def compact_dates(xs: Sequence) -> dict | None:
    """{start, steps}: ISO dates as their first date and the day gaps after it
    (None unless every x is an ISO date and there are COMPACT_DATES_MIN of them)."""
    if len(xs) < COMPACT_DATES_MIN or not all(isinstance(x, str) and _ISO.fullmatch(x) for x in xs):
        return None
    ds = [dt.date.fromisoformat(x) for x in xs]
    return {"start": xs[0], "steps": [(b - a).days for a, b in itertools.pairwise(ds)]}


def expand_dates(c: Mapping) -> list[str]:
    d = dt.date.fromisoformat(c["start"])
    out = [c["start"]]
    for step in c["steps"]:
        d += dt.timedelta(days=int(step))
        out.append(d.isoformat())
    return out


def resolve(spec: dict, others: Mapping[str, dict] | None = None) -> dict:
    """The spec as desk.js draws it: compact x values expanded, shared x values and
    derived series filled in (``others``: the page's other specs by chart id)."""
    out = copy.deepcopy(spec)
    others = others or {}
    if "xsDates" in out:
        out["xs"] = expand_dates(out.pop("xsDates"))
    if "xsFrom" in out:
        out["xs"] = resolve(others[out["xsFrom"]], others)["xs"]
    for s in out.get("series") or []:
        d = s.get("derive")
        if not d:
            continue
        src = resolve(others[d["from"]], others) if d.get("from") else out
        by = {x["name"]: x.get("data") for x in src.get("series") or []}
        a, b, k, n = by[d["a"]], (by[d["b"]] if d.get("b") else None), d.get("k", 1), d.get("digits", 2)
        s["data"] = [None if (v is None or (b is not None and b[i] is None))
                     else round(k * v - (b[i] if b is not None else 0), n) for i, v in enumerate(a)]
    groups = (out.get("quotes") or {}).get("groups") or {}
    if groups:
        dec = {g: desk_quotes.decode(b) for g, b in groups.items()}
        for s in out.get("series") or []:
            if s.get("quote"):
                s["data"] = desk_quotes.series_data(dec[str(s["group"])], s["quote"], s.get("model"))
    return out


def xy(chart_id: str, title: str, *, x: dict, y: dict, series: list[dict],
       xs: Sequence | None = None, y2: dict | None = None, tooltip: str = "axis",
       zoom: bool | str = False, marks: dict | None = None, csv: str | None = None,
       horizontal: bool = False, grid: dict | None = None, describe: str = "",
       table: dict | None = None, zoom_presets: Sequence[tuple] = (), link: str | None = None,
       xs_from: str | None = None, quotes: dict | None = None) -> dict:
    """A cartesian chart spec (line / bar / scatter / error bars).

    zoom: True for a slider and wheel/drag zoom, 'inside' for wheel/drag only;
    link: charts with the same key zoom and point together (echarts.connect),
    so they need the same x values."""
    spec: dict[str, Any] = {"id": chart_id, "kind": "xy", "title": title, "x": x, "y": y,
                            "series": series, "tooltip": tooltip, "describe": describe}
    if xs_from:
        spec["xsFrom"] = xs_from                  # the x values of that chart (a linked chart)
    elif xs is not None:
        cd = compact_dates(list(xs))
        if cd:
            spec["xsDates"] = cd
        else:
            spec["xs"] = list(xs)
    if y2:
        spec["y2"] = y2
    if zoom:
        spec["zoom"] = zoom
    if link:
        spec["link"] = link
    if zoom_presets:
        spec["zoom_presets"] = [{"label": lab, "years": yrs} for lab, yrs in zoom_presets]
    if marks:
        spec["marks"] = marks
    if horizontal:
        spec["horizontal"] = True
    if grid:
        spec["grid"] = grid
    spec["csv"] = csv or f"{chart_id}.csv"
    spec["table"] = table or {"caption": title, "columns": [], "rows": []}
    items: dict[str, dict] = {}
    for s in series:
        if not s.get("legend") or s.get("invisible"):
            continue
        key = s.get("legendKey") or s["name"]
        if key in items:                  # one button per key, from its first series
            continue
        items[key] = {"name": key, "color": s.get("color", "cat0"),
                      "dash": bool(s.get("dash")) and s.get("dash") != "dotted",
                      "dotted": s.get("dash") == "dotted",
                      "dot": s.get("type") in ("scatter", "errorbar"),
                      "hollow": s.get("type") == "scatter" and bool(s.get("hollow")),
                      "diamond": s.get("type") == "scatter" and s.get("symbol") == "diamond",
                      "hidden": bool(s.get("hidden"))}
        if s.get("type") == "scatter" and s.get("symbol") == "triangle":     # only where it is one
            items[key].update(tri=True, down=s.get("rotate") == 180)
        # one series of one tab group (an earlier session's line): its button shows with the group
        if s.get("group") is not None and not s.get("legendKey"):
            items[key]["group"] = s["group"]
    spec["legend_items"] = list(items.values())
    out = round_sig(spec)
    if quotes:
        out["quotes"] = quotes                    # full precision (see the module docstring)
    return out


def table(caption: str, columns: Iterable, rows: Iterable[Iterable[Any]]) -> dict:
    """A data table: columns are labels (a trailing '#' marks a numeric column)."""
    cols = []
    for c in columns:
        num = isinstance(c, str) and c.endswith("#")
        cols.append({"label": c[:-1] if num else c, "num": num})
    return {"caption": caption, "columns": cols, "rows": [[str(v) for v in r] for r in rows]}


def color_cycle(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]


def dot_plot(points: Sequence[dict], lo_box: float, hi_box: float, spec: str = ".3f", width: float = 240.0,
             row_h: float = 11.0, label_w: float = 40.0) -> dict | None:
    """A server-rendered forest plot: one row per point (oldest first), the value
    as a dot with a ±2·SE whisker, the box bounds as dashed lines.

    ``points``: {session, value, se (None: no whisker), on_bound}. The x range is
    the points' value ± 2·SE, padded; a box bound within half that span of it is
    drawn (the range extends to it), a farther one is left off the scale.
    """
    pts = [p for p in points if p.get("value") is not None]
    if not pts:
        return None
    ends = []
    for p in pts:
        v, se = float(p["value"]), p.get("se")
        ends += [v - 2 * float(se), v + 2 * float(se)] if se is not None else [v]
    dlo, dhi = min(ends), max(ends)
    span = (dhi - dlo) or (abs(dhi) * 0.1) or 1.0
    shown = [b for b in (lo_box, hi_box) if (b <= dlo and dlo - b <= 0.5 * span) or
             (b >= dhi and b - dhi <= 0.5 * span) or dlo < b < dhi]
    xlo, xhi = min([dlo, *shown]), max([dhi, *shown])
    if xhi - xlo <= 0:                            # one value and no whisker: a range around it
        xlo, xhi = xlo - span / 2, xhi + span / 2
    pad = 0.06 * (xhi - xlo)
    xlo, xhi = xlo - pad, xhi + pad
    plot_w = width - label_w - 6.0

    def x(u: float) -> float:
        return round(label_w + (u - xlo) / (xhi - xlo) * plot_w, 2)

    def f(u: float) -> str:                       # a true minus sign
        return format(u, spec).replace("-", "\u2212")
    rows = []
    for i, p in enumerate(pts):
        v, se = float(p["value"]), p.get("se")
        rows.append({"session": p["session"], "label": str(p["session"])[5:], "value": v, "se": se,
                     "on_bound": bool(p.get("on_bound")), "x": x(v), "y": round(6 + i * row_h, 2),
                     "x0": x(v - 2 * float(se)) if se is not None else None,
                     "x1": x(v + 2 * float(se)) if se is not None else None,
                     "text": f(v), "se_text": f(2 * float(se)) if se is not None else None,
                     "newest": i == len(pts) - 1})
    h = round(6 + len(pts) * row_h + 12, 2)
    return {"w": width, "h": h, "rows": rows, "axis_y": round(6 + len(pts) * row_h - 4, 2),
            "bounds": [{"x": x(b), "value": b, "label": f"{b:g}".replace("-", "\u2212")} for b in shown],
            # the data range's ends, unless a bound's label sits there
            "ticks": [{"x": tx, "label": f(u), "anchor": anc}
                      for u, anc in ((xlo + pad, "start"), (xhi - pad, "end"))
                      for tx in (x(u),) if all(abs(tx - x(b)) > 36 for b in shown)],
            "x_min": xlo, "x_max": xhi, "left": label_w, "right": width - 6.0}
