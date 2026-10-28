"""
desk_monitor.py — Model Risk as an exceptions view (pure functions; no Flask, no I/O).

The Model Risk page lists its checks worst first (BREACH, WATCH, OK, INFO, then
not applicable; within a status, in inventory order), with a summary strip of
counts. Each check with numeric limits (g, a) — OK ≤ g, watch ≤ a, breach
above, app.MONITOR_LIMITS — gets a bullet: three bands on a linear scale from 0
to max(1.5·a, value) and a tick at the value. A check that can be recomputed on
every stored session (the gated fits, the surface reports, the β chains) gets
its history: one dot per session, coloured by TODAY's limits applied
retroactively, its range and where today ranks. The sessions are the ones the
store holds, at irregular spacing; the dots are not a control chart.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

SEVERITY = ("breach", "watch", "ok", "info", "na")
MINUS = "\u2212"
STATUS_WORDS = {"breach": "breach", "watch": "watch", "ok": "OK", "info": "info", "na": "N/A"}
BULLET_W, BULLET_H = 120, 10


def _num(x: Any) -> float | None:
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def rag(value: float | None, limits: Sequence[float]) -> str:
    """'ok' / 'watch' / 'breach' of a value under (g, a): OK ≤ g, watch ≤ a, breach above."""
    v = _num(value)
    if v is None:
        return "na"
    g, a = limits
    return "ok" if v <= g else ("watch" if v <= a else "breach")


def slug(text: str) -> str:
    """'Out-of-sample SPX IV RMSE (vol pts)' -> 'out-of-sample-spx-iv-rmse-vol-pts' (ASCII, for ids)."""
    t = (text or "").lower()
    t = t.replace("β", "beta").replace("−", "-").replace("|", "")
    t = re.sub(r"[^a-z0-9]+", "-", t).strip("-")
    return t or "check"


def order_checks(rows: Sequence[Mapping], model_order: Sequence[str]) -> list[dict]:
    """The checks worst first: by status (SEVERITY), then by the model's place in the
    inventory; checks of one model and status keep their order."""
    sev = {s: i for i, s in enumerate(SEVERITY)}
    mod = {m: i for i, m in enumerate(model_order)}
    keyed = [(sev.get(str(r.get("status")), len(sev)), mod.get(str(r.get("model")), len(mod)), i, dict(r))
             for i, r in enumerate(rows)]
    return [r for *_, r in sorted(keyed, key=lambda x: x[:3])]


def tally(rows: Iterable[Mapping]) -> dict[str, int]:
    out = dict.fromkeys(SEVERITY, 0)
    for r in rows:
        s = str(r.get("status"))
        out[s] = out.get(s, 0) + 1
    return out


def _g(x: float) -> str:
    return f"{x:g}"


def bullet(value: Any, limits: Sequence[float], digits: str = ".2f", width: float = BULLET_W,
           height: float = BULLET_H) -> dict | None:
    """A bullet on [0, max(1.5·a, value)]: OK band [0, g], watch (g, a], breach (a, top],
    the tick at the value (x proportional to it). None for a missing or negative value."""
    v = _num(value)
    if v is None or v < 0:
        return None
    g, a = (float(x) for x in limits)
    top = max(1.5 * a, v, 1e-12)

    def x(u: float) -> float:
        return round(u / top * width, 2)
    return {"w": width, "h": height, "top": top, "value": v,
            "bands": [{"cls": "ok", "x": 0.0, "w": x(g)},
                      {"cls": "watch", "x": x(g), "w": round(x(a) - x(g), 2)},
                      {"cls": "breach", "x": x(a), "w": round(width - x(a), 2)}],
            "tick": x(v),
            "label": f"value {format(v, digits)}; OK ≤ {_g(g)}, watch ≤ {_g(a)}, breach above {_g(a)}"}


def history_view(points: Sequence[tuple[str, Any]], limits: Sequence[float], digits: str = ".2f",
                 lower_is_better: bool = True, two_sided: bool = False) -> dict | None:
    """One dot per stored session (oldest first), each coloured by today's limits;
    the range, and where the newest (today's) value ranks: 'worst of 5'.

    ``two_sided``: a signed measurement whose limits bound its size (a gap that
    may fall either side of zero). Each dot keeps its sign, in its text, its
    height and the range ('−0.78 to +0.09'); its status and today's rank use
    |value|.

    The last point is today's. When today has no value (a check the newest
    session cannot compute, e.g. a VIX the chain does not replicate) the dots of
    the earlier sessions remain, ``last`` is None and nothing is ranked: an
    earlier session's value is never shown as today's."""
    pts: list[tuple[str, float]] = []
    for s, raw in points:
        v = _num(raw)
        if v is not None:
            pts.append((str(s), v))
    today_missing = bool(points) and _num(points[-1][1]) is None
    if len(pts) < 2:
        return None
    vals = [v for _, v in pts]
    size = abs if two_sided else float

    def text(v: float) -> str:
        return signed(v, digits) if two_sided else _plain(v, digits)
    lo_, hi_ = min(vals), max(vals)
    rng = f"{text(lo_)} to {text(hi_)}" if two_sided else f"{text(lo_)}–{text(hi_)}"
    dots = [{"session": s, "value": v, "status": rag(size(v), limits), "text": text(v)} for s, v in pts]
    if today_missing:
        return {"dots": dots, "lo": lo_, "hi": hi_, "n": len(vals), "rank": None, "today_missing": True,
                "range": rng, "last": None, "first_session": pts[0][0], "last_session": pts[-1][0]}
    today = size(vals[-1])
    sign = 1.0 if lower_is_better else -1.0
    better = sum(1 for x in vals if sign * (size(x) - today) < -1e-12)
    worse = sum(1 for x in vals if sign * (size(x) - today) > 1e-12)
    n = len(vals)
    tied = "tied " if n - better - worse > 1 else ""
    word = (f"{tied}worst" if worse == 0 else f"{tied}best" if better == 0
            else f"{tied}{_ordinal(better + 1)}")
    return {"dots": dots, "lo": lo_, "hi": hi_, "n": n, "rank": f"{word} of {n}", "range": rng,
            "last": vals[-1], "today_missing": False, "first_session": pts[0][0], "last_session": pts[-1][0]}


def _plain(v: float, digits: str) -> str:
    return format(v, digits).replace("-", MINUS)


def signed(v: float, digits: str) -> str:
    """'+0.09', '−0.78', '0.00' (a value that rounds to zero carries no sign)."""
    body = format(abs(v), digits)
    return body if float(body) == 0.0 else ("+" if v > 0 else MINUS) + body


def _ordinal(n: int) -> str:
    suf = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"
