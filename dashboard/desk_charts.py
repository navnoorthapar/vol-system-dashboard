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
y2, group (for tab-filtered charts), legend (show a toggle button), legendKey
(the button's name when one button stands for several series, e.g. a model's
curve on every expiry), hidden (toggled off at first), xFrom (the name of a
series whose x values this one's y-only data shares).
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from desk_format import round_sig

CATEGORICAL = ("cat0", "cat1", "cat2", "cat3", "cat4", "cat5")


def xy(chart_id: str, title: str, *, x: dict, y: dict, series: list[dict],
       xs: Sequence | None = None, y2: dict | None = None, tooltip: str = "axis",
       zoom: bool = False, marks: dict | None = None, csv: str | None = None,
       horizontal: bool = False, grid: dict | None = None, describe: str = "",
       table: dict | None = None, zoom_presets: Sequence[tuple] = ()) -> dict:
    """A cartesian chart spec (line / bar / scatter / error bars)."""
    spec: dict[str, Any] = {"id": chart_id, "kind": "xy", "title": title, "x": x, "y": y,
                            "series": series, "tooltip": tooltip, "describe": describe}
    if xs is not None:
        spec["xs"] = list(xs)
    if y2:
        spec["y2"] = y2
    if zoom:
        spec["zoom"] = True
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
