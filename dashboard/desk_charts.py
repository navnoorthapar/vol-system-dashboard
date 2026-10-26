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
"""
from __future__ import annotations

import copy
import datetime as dt
import itertools
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from desk_format import round_sig

CATEGORICAL = ("cat0", "cat1", "cat2", "cat3", "cat4", "cat5")
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
    return out


def xy(chart_id: str, title: str, *, x: dict, y: dict, series: list[dict],
       xs: Sequence | None = None, y2: dict | None = None, tooltip: str = "axis",
       zoom: bool | str = False, marks: dict | None = None, csv: str | None = None,
       horizontal: bool = False, grid: dict | None = None, describe: str = "",
       table: dict | None = None, zoom_presets: Sequence[tuple] = (), link: str | None = None,
       xs_from: str | None = None) -> dict:
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
    return round_sig(spec)


def table(caption: str, columns: Iterable, rows: Iterable[Iterable[Any]]) -> dict:
    """A data table: columns are labels (a trailing '#' marks a numeric column)."""
    cols = []
    for c in columns:
        num = isinstance(c, str) and c.endswith("#")
        cols.append({"label": c[:-1] if num else c, "num": num})
    return {"caption": caption, "columns": cols, "rows": [[str(v) for v in r] for r in rows]}


def color_cycle(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]
