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
dash, symbol, size, hollow, signed (bars coloured by sign), stack, invisible,
y2, group (for tab-filtered charts), legend (show a toggle button), hidden.
"""
from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from desk_format import _fmt, round_sig

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
    spec["legend_items"] = [
        {"name": s["name"], "color": s.get("color", "cat0"), "dash": bool(s.get("dash")),
         "dot": s.get("type") in ("scatter", "errorbar"), "hidden": bool(s.get("hidden"))}
        for s in series if s.get("legend") and not s.get("invisible")
    ]
    return round_sig(spec)


def table(caption: str, columns: Iterable, rows: Iterable[Iterable[Any]]) -> dict:
    """A data table: columns are labels (a trailing '#' marks a numeric column)."""
    cols = []
    for c in columns:
        num = isinstance(c, str) and c.endswith("#")
        cols.append({"label": c[:-1] if num else c, "num": num})
    return {"caption": caption, "columns": cols, "rows": [[str(v) for v in r] for r in rows]}


def summary_table(caption: str, series: list[dict], fmt: str = ".2f", unit: str = "",
                  xs: Sequence | None = None) -> dict:
    """First / last / lowest / highest of each aligned series (long series)."""
    rows = []
    for s in series:
        if s.get("invisible"):
            continue
        ys = [v for v in s["data"] if v is not None]
        if not ys:
            continue
        rows.append([s["name"], f"{_fmt(ys[0], fmt)}{unit}", f"{_fmt(ys[-1], fmt)}{unit}",
                     f"{_fmt(min(ys), fmt)}{unit}", f"{_fmt(max(ys), fmt)}{unit}"])
    head = (f"First ({xs[0]})#" if xs else "First#", f"Last ({xs[-1]})#" if xs else "Last#")
    return table(caption, ["Series", head[0], head[1], "Lowest#", "Highest#"], rows)


def color_cycle(i: int) -> str:
    return CATEGORICAL[i % len(CATEGORICAL)]
