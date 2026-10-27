"""
desk_format.py — number formatting, colour scales and small SVG helpers for the
dashboard (pure functions; no I/O).

Conventions every page follows:
  * numbers print with a true minus sign U+2212, never a hyphen;
  * signed quantities (changes, P&L, residuals) carry an explicit "+";
  * P&L and changes are coloured blue (gain) / orange (loss) AND carry the sign,
    so colour is never the only cue (red/green is reserved for status);
  * a missing value prints "—", never 0, "nan" or "None";
  * floats sent to the browser are rounded to 6 significant figures so a
    rebuild of unchanged data produces byte-identical pages.

Colour scales are sampled once from their published sources and hard-coded:
  * CIVIDIS_11 — matplotlib's cividis (Nuñez, Anderton & Renslow 2018), the
    sequential map for levels (implied vol);
  * BERLIN_11 — Fabio Crameri's Scientific Colour Map "berlin" (MIT; read from
    cmcrameri 1.9's berlin.txt, 256 entries, 11 evenly spaced samples), the
    diverging map with a dark midpoint for signed residuals on a dark UI.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from typing import Any

from jinja2.exceptions import UndefinedError
from markupsafe import Markup, escape

MINUS = "−"   # '−': numbers are printed with a true minus sign, not a hyphen

CIVIDIS_11 = ("#00224E", "#083370", "#35456C", "#4F576C", "#666970", "#7D7C78",
              "#948E77", "#AEA371", "#C8B866", "#E5CF52", "#FEE838")
BERLIN_11 = ("#9EB0FF", "#60A5DF", "#3280A6", "#20526A", "#112732", "#190C09",
             "#371000", "#5F1F0A", "#964A36", "#CA7B71", "#FFADAD")
# The diverging ramp of every signed heatmap that is not P&L (residual z, RR / BF,
# vanna, delta), dark at zero: violet below, teal above. Berlin's blue end
# (#9EB0FF) sat next to --pos and its red end next to --breach; these ends are
# neither a P&L hue (blue gain, orange loss) nor breach red. P&L heatmaps alone
# use the P&L ramp (app.PNL_DIV).
DIVERGE_11 = ("#d4a0ff", "#a687f0", "#7f63c8", "#574393", "#33295a", "#1c2127",
              "#1d3d3a", "#22605a", "#2b867c", "#45ad9f", "#7fd3c4")


def _num(x: Any) -> float | None:
    """float(x) if x is a finite number, else None."""
    try:
        f = float(x)
    except (TypeError, ValueError, UndefinedError):   # None, a string, a Jinja Undefined
        return None
    return f if math.isfinite(f) else None


def ordinal(n: Any) -> str:
    """1 -> '1st', 2 -> '2nd', 11 -> '11th', 75 -> '75th' (rounded to an integer)."""
    if _num(n) is None:
        return "—"
    k = round(float(n))
    suffix = "th" if 10 <= k % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(k % 10, "th")
    return f"{k}{suffix}"


def _fmt(value: Any, spec: str = "", default: str = "—") -> str:
    """Jinja filter: format a number, or render `default` when it is missing."""
    if value is None:
        return default
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if not math.isfinite(float(value)):
            return default
        return format(value, spec).replace("-", MINUS)   # typographic minus sign
    return str(value)


def signed(value: Any, spec: str = ".2f", unit: str = "", default: str = "—") -> str:
    """'+1.23', '−1.23' or '0.00' (a value that rounds to zero carries no sign)."""
    x = _num(value)
    if x is None:
        return default
    body = format(abs(x), spec)
    zero = _num(body.replace(",", "").rstrip("%")) == 0.0
    sign = "" if zero else ("+" if x > 0 else MINUS)
    return f"{sign}{body}{unit}"


def tone(value: Any, eps: float = 0.0) -> str:
    """CSS class for a signed quantity: 'pos', 'neg' or 'flat'."""
    x = _num(value)
    if x is None or abs(x) <= eps:
        return "flat"
    return "pos" if x > 0 else "neg"


def pnl(value: Any, spec: str = ".2f", unit: str = "", glyph: bool = False) -> Markup:
    """A signed number in the P&L colour convention: <span class="pos">+1.23</span>.

    The sign is always in the text; `glyph` adds ▲ / ▼ for change columns.
    """
    x = _num(value)
    if x is None:
        return Markup("—")
    text = signed(x, spec, unit)
    cls = tone(x) if text[0] in "+" + MINUS else "flat"
    arrow = ""
    if glyph and cls != "flat":
        arrow = ("▲" if cls == "pos" else "▼") + " "
    # cls is one of three fixed class names; both text parts are escaped
    return Markup(f'<span class="{cls}">{escape(arrow)}{escape(text)}</span>')  # noqa: S704


def _usd_k(value: Any, signed: bool = True) -> str:
    """Prose dollars in thousands: 693938 -> '+$694K', -599995 -> '−$600K'."""
    x = _num(value)
    if x is None:
        return "—"
    body = f"${abs(x) / 1e6:,.1f}M" if abs(x) >= 999_500 else f"${abs(x) / 1000:,.0f}K"
    if body == "$0K":                                   # rounds to zero: no sign
        return body
    sign = MINUS if x < 0 else ("+" if signed else "")
    return f"{sign}{body}"


def _usd(value: Any, signed: bool = False) -> str:
    """Whole dollars with a true minus sign: -1039.2 -> '−$1,039'."""
    x = _num(value)
    if x is None:
        return "—"
    body = f"${abs(x):,.0f}"
    if body == "$0":
        return body
    sign = MINUS if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}{body}"


def _pct_prose(value: Any, digits: int = 1, signed: bool = True) -> str:
    """Prose percent of a decimal: -0.2612 -> '−26.1%'."""
    x = _num(value)
    if x is None:
        return "—"
    body = f"{abs(x) * 100:.{digits}f}"
    if float(body) == 0.0:
        return f"{body}%"
    sign = MINUS if x < 0 else ("+" if signed and x > 0 else "")
    return f"{sign}{body}%"


def plural(n: Any, word: str, many: str | None = None) -> str:
    """'1 session', '2 sessions' ('1 sessions old' was printed on the Greeks page)."""
    k = _num(n)
    if k is None:
        return f"— {many or word + 's'}"
    k_int = int(k)
    return f"{k_int:,} {word if k_int == 1 else (many or word + 's')}"


_HYPHEN_NUMBER = re.compile(r"(?<![\w/.\-–])-(\$?)(?=\.?\d)")


def prose(text: Any) -> str:
    """Producer-written prose (notes, headlines) in the page's number style:
    '-0.15' and '[-0.68, 0.50]' print a true minus; '$-158' prints '−$158'."""
    if text is None:
        return "—"
    s = str(text)
    s = re.sub(r"\$-(?=\d)", MINUS + "$", s)
    return _HYPHEN_NUMBER.sub(lambda m: MINUS + m.group(1), s)


# ── desk units: cash Greeks of an SPX straddle ───────────────────────────────
SPX_MULTIPLIER = 100.0     # index points → dollars for one SPX option contract
VOL_PT = 0.01              # one vol point (1.00 vol = 100 vol pts)
SPOT_PCT = 0.01            # a 1% spot move


def cash_greeks(S: float, F: float, K: float, T: float, sigma: float, D: float,
                multiplier: float = SPX_MULTIPLIER) -> dict:
    """Black-76 Greeks of one straddle (call + put at K) and their desk units.

    S spot, F the forward to T (F = S·e^{(r−q)T}, so ∂F/∂S = F/S), K strike,
    T years, sigma the straddle's vol (decimal), D the discount factor. With
    d₁ = (ln(F/K) + ½σ²T)/(σ√T), d₂ = d₁ − σ√T and φ the normal density:
      vega  ∂V/∂σ    = 2·D·F·φ(d₁)·√T
      gamma ∂²V/∂S²  = (F/S)²·2·D·φ(d₁)/(F·σ·√T)
      vanna ∂²V/∂S∂σ = −(F/S)·2·D·φ(d₁)·d₂/σ
      volga ∂²V/∂σ²  = vega·d₁·d₂/σ
    per unit of the index and per 1.00 of vol. The desk units are dollars for
    one SPX contract (× ``multiplier``):
      $vega  per vol pt               = vega·0.01·m
      $gamma per 1% spot move         = ½·gamma·(0.01·S)²·m   (the P&L of the move)
      $vanna per 1% spot per vol pt   = vanna·(0.01·S)·0.01·m
      $volga per vol pt²              = volga·0.01²·m          (P&L ≈ ½·$volga·Δσ²)
    """
    sq = math.sqrt(T)
    d1 = (math.log(F / K) + 0.5 * sigma * sigma * T) / (sigma * sq)
    d2 = d1 - sigma * sq
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    vega = 2.0 * D * F * pdf * sq
    gamma = (F / S) ** 2 * 2.0 * D * pdf / (F * sigma * sq)
    vanna = -(F / S) * 2.0 * D * pdf * d2 / sigma
    volga = vega * d1 * d2 / sigma
    return {"vega": vega, "gamma": gamma, "vanna": vanna, "volga": volga, "d1": d1, "d2": d2,
            "vega_usd": vega * VOL_PT * multiplier,
            "gamma_usd": 0.5 * gamma * (SPOT_PCT * S) ** 2 * multiplier,
            "vanna_usd": vanna * (SPOT_PCT * S) * VOL_PT * multiplier,
            "volga_usd": volga * VOL_PT ** 2 * multiplier}


def usd_desk(value: Any, signed: bool = True) -> str:
    """Dollars at desk precision: whole dollars from $100, cents below
    (1750.67 -> '+$1,751', 8.7534 -> '+$8.75', -0.0539 -> '−$0.05')."""
    x = _num(value)
    if x is None:
        return "—"
    body = f"${abs(x):,.0f}" if abs(x) >= 99.995 else f"${abs(x):.2f}"
    if float(body[1:].replace(",", "")) == 0.0:
        return body
    sign = MINUS if x < 0 else ("+" if signed else "")
    return f"{sign}{body}"


# ── contrast (WCAG 2.x) ───────────────────────────────────────────────────────
def _relative_luminance(hex_colour: str) -> float:
    """WCAG 2.x relative luminance of '#rrggbb'."""
    h = hex_colour.lstrip("#")

    def lin(c8: int) -> float:
        c = c8 / 255.0
        return c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    r, g, b = (int(h[i:i + 2], 16) for i in (0, 2, 4))
    return 0.2126 * lin(r) + 0.7152 * lin(g) + 0.0722 * lin(b)


def contrast_ratio(fg: str, bg: str) -> float:
    """WCAG 2.x contrast ratio between two opaque '#rrggbb' colours (1..21)."""
    hi, lo = sorted((_relative_luminance(fg), _relative_luminance(bg)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def _text_colour_on(bg: str) -> str:
    """Black or white, whichever contrasts more with `bg` (always >= 4.58:1)."""
    return "#000000" if contrast_ratio("#000000", bg) >= contrast_ratio("#ffffff", bg) else "#ffffff"


# ── colour scales ─────────────────────────────────────────────────────────────
def _hex_to_rgb(h: str) -> tuple[int, int, int]:
    h = h.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def ramp(stops: Iterable[str], t: float) -> str:
    """Colour at t ∈ [0, 1] on a piecewise-linear (sRGB) ramp through `stops`."""
    stops = list(stops)
    t = min(max(float(t), 0.0), 1.0)
    pos = t * (len(stops) - 1)
    i = min(int(pos), len(stops) - 2)
    f = pos - i
    a, b = _hex_to_rgb(stops[i]), _hex_to_rgb(stops[i + 1])
    return "#" + "".join(f"{round(x + (y - x) * f):02x}" for x, y in zip(a, b))


def seq_colour(value: Any, lo: float, hi: float) -> str | None:
    """Cividis colour of `value` on [lo, hi] (None when the value is missing)."""
    x = _num(value)
    if x is None:
        return None
    return ramp(CIVIDIS_11, 0.5 if hi <= lo else (x - lo) / (hi - lo))


def div_colour(value: Any, limit: float) -> str | None:
    """DIVERGE_11 colour of a signed value, symmetric on [−limit, +limit], dark at 0."""
    x = _num(value)
    if x is None or limit <= 0:
        return None
    return ramp(DIVERGE_11, 0.5 + 0.5 * max(-1.0, min(1.0, x / limit)))


# Greeks heatmap: the vomma P&L ratio from 0 to its flag threshold, on the
# sequential scale; a flagged node also gets an outline and a ⚠ in its text.
_HEAT_NEUTRAL = (0x1c, 0x21, 0x27)
_HEAT_NEG = (0x32, 0x80, 0xa6)
_HEAT_POS = (0xc8, 0x76, 0x19)


def _vomma_color(z: float | None) -> str:
    """Map a vomma z-score (or the ratio rescaled to ±2) to a cell background."""
    if z is None or (isinstance(z, float) and math.isnan(z)):
        return "#1c2127"
    end = _HEAT_POS if z > 0 else _HEAT_NEG
    t = min(abs(z) / 2.0, 1.0)
    return "#" + "".join(f"{round(a + (b - a) * t):02x}" for a, b in zip(_HEAT_NEUTRAL, end))


def cell(value_text: str, bg: str | None, detail: str = "", flag: bool = False,
         sr: str = "") -> dict:
    """One heatmap cell: text, colours (text colour chosen for contrast) and detail."""
    if bg is None:
        return {"empty": True, "text": value_text or "—", "detail": detail, "sr": sr}
    return {"text": value_text, "bg": bg, "fg": _text_colour_on(bg), "detail": detail,
            "flag": flag, "sr": sr}


# ── JSON payloads ─────────────────────────────────────────────────────────────
def round_sig(obj: Any, sig: int = 6) -> Any:
    """Round every float in a JSON-like structure to `sig` significant figures.

    The browser never needs more, and full-precision floats made every rebuild
    of the same data produce a different page (last-bit noise in the JSON).
    """
    if isinstance(obj, dict):
        return {k: round_sig(v, sig) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_sig(v, sig) for v in obj]
    if isinstance(obj, bool) or obj is None or isinstance(obj, (int, str)):
        return obj
    x = _num(obj)
    if x is None:
        return None
    if x == 0.0:
        return 0.0
    digits = sig - math.floor(math.log10(abs(x))) - 1
    return float(round(x, digits))


# ── sparklines (inline SVG, no JS) ────────────────────────────────────────────
def sparkline(values: Iterable[Any], width: float = 100.0, height: float = 28.0,
              bound: float | None = None, zero: bool = False, pad: float = 2.0) -> dict | None:
    """Polyline points for an inline-SVG sparkline on a [0, width] × [0, height] box.

    `bound` draws a dashed reference line (e.g. a parameter's box bound) and is
    included in the vertical range so the line is always visible; `zero` adds a
    zero line when the series crosses it. Returns None with fewer than 2 points.
    """
    ys = [_num(v) for v in values]
    pts = [(i, y) for i, y in enumerate(ys) if y is not None]
    if len(pts) < 2:
        return None
    lo = min(y for _, y in pts)
    hi = max(y for _, y in pts)
    if bound is not None:
        lo, hi = min(lo, bound), max(hi, bound)
    if zero:
        lo, hi = min(lo, 0.0), max(hi, 0.0)
    if hi - lo < 1e-12:
        lo, hi = lo - 1.0, hi + 1.0
    n = len(ys) - 1

    def sx(i: float) -> float:
        return round(pad + (width - 2 * pad) * (i / n if n else 0.5), 2)

    def sy(y: float) -> float:
        return round(height - pad - (height - 2 * pad) * (y - lo) / (hi - lo), 2)

    out = {"points": " ".join(f"{sx(i)},{sy(y)}" for i, y in pts),
           "last": {"x": sx(pts[-1][0]), "y": sy(pts[-1][1])},
           "width": width, "height": height, "lo": lo, "hi": hi}
    if bound is not None:
        out["bound_y"] = sy(bound)
    if zero and lo <= 0 <= hi:
        out["zero_y"] = sy(0.0)
    return out
