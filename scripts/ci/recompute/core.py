"""
core.py: the recompute audit's checks, number parsing and coverage count.

Every check compares one number (or date, or count) the frozen page prints with
the value this audit recomputes from data_store by its own code. A printed
number matches when the recomputed value rounds to it: |printed − recomputed|
is at most half a unit in the printed number's last digit (plus 1e-9 of slack
for binary floating point), so a value the page rounds correctly always passes
and a value one display unit away always fails.

Nothing here imports dashboard/ or joint_vol_calibration/ (a test reads every
module of this package and fails on such an import): an audit that called the
code it audits would only check that the code agrees with itself.
"""
from __future__ import annotations

import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from bs4 import BeautifulSoup, NavigableString, Tag

MINUS = "\u2212"
_SPACES = {"\u2009": " ", "\u202f": " ", "\u00a0": " ", "\u200b": ""}
# a printed number: optional sign, digits with thousands commas, optional decimals
NUM = re.compile(r"(?<![\w.])([+\-−]?)(\d{1,3}(?:,\d{3})+|\d+)(\.\d+)?(?=(?:st|nd|rd|th)\b|\W|$)")
DATE = re.compile(r"\b(\d{4})-(\d{2})-(\d{2})\b")


def norm(text: str) -> str:
    """Page text with a hyphen for U+2212, plain spaces, and runs of space collapsed."""
    t = text.replace(MINUS, "-")
    for a, b in _SPACES.items():
        t = t.replace(a, b)
    return re.sub(r"\s+", " ", t).strip()


def text_of(el: Tag | NavigableString | str | None) -> str:
    if el is None:
        return ""
    if isinstance(el, str):
        return norm(el)
    return norm(el.get_text(" "))


def parse_num(text: str) -> tuple[float, int]:
    """The first number in ``text`` and its count of decimals: '−1,573.20 pts' -> (-1573.2, 2).

    Raises ValueError when there is none."""
    t = norm(text)
    m = NUM.search(t)
    if not m:
        raise ValueError(f"no number in {text!r}")
    sign, whole, frac = m.group(1), m.group(2), m.group(3) or ""
    v = float(whole.replace(",", "") + frac)
    return (-v if sign == "-" else v), max(len(frac) - 1, 0)


def numbers_in(text: str) -> list[tuple[float, int, str]]:
    """Every number in ``text``: (value, decimals, matched text), dates excluded."""
    t = DATE.sub(" ", norm(text))
    out = []
    for m in NUM.finditer(t):
        sign, whole, frac = m.group(1), m.group(2), m.group(3) or ""
        v = float(whole.replace(",", "") + frac)
        out.append((-v if sign == "-" else v, max(len(frac) - 1, 0), m.group(0)))
    return out


def rounds_to(expected: float | None, printed: float, decimals: int, scale: float = 1.0) -> bool:
    """True when ``expected``·scale, rounded to ``decimals``, is the printed value."""
    if expected is None or not math.isfinite(expected):
        return False
    half = 0.5 * 10.0 ** (-decimals)
    return abs(expected * scale - printed) <= half + 1e-9 * max(1.0, abs(printed))


@dataclass
class Check:
    page: str
    panel: str
    label: str
    printed: str
    expected: str
    ok: bool
    note: str = ""

    def __post_init__(self) -> None:
        # a comparison of recomputed values can yield numpy scalars (np.bool_),
        # which the --json report cannot serialise
        self.ok = bool(self.ok)
        self.printed, self.expected, self.note = str(self.printed), str(self.expected), str(self.note)


@dataclass
class Audit:
    """One page's checks. Each check marks the element it verified, so the coverage
    count can say which printed numbers no check looked at."""
    page: str
    soup: BeautifulSoup
    checks: list[Check] = field(default_factory=list)
    covered: set[int] = field(default_factory=set)

    # ── locating ──────────────────────────────────────────────────────────────
    def one(self, selector: str, root: Tag | None = None) -> Tag | None:
        return (root or self.soup).select_one(selector)

    def need(self, selector: str, panel: str, root: Tag | None = None) -> Tag | None:
        """The element, or a failed check when the page lacks it (a check that cannot
        find its number must fail, never pass by being skipped)."""
        el = self.one(selector, root)
        if el is None:
            self.fail(panel, f"element {selector}", "present", "missing")
        return el

    def cover(self, el: Tag | None) -> None:
        if el is not None:
            self.covered.add(id(el))

    # ── recording ─────────────────────────────────────────────────────────────
    def fail(self, panel: str, label: str, expected: str, printed: str, note: str = "") -> None:
        self.checks.append(Check(self.page, panel, label, printed, expected, False, note))

    def num(self, panel: str, label: str, el: Tag | str | None, expected: float | None,
            scale: float = 1.0, index: int = 0, note: str = "", abs_value: bool = False) -> bool:
        """The ``index``-th number in ``el`` against ``expected`` (×``scale``)."""
        txt = text_of(el)
        nums = numbers_in(txt)
        if index >= len(nums):
            self.fail(panel, label, _show(expected, scale), txt or "missing", note or "no number printed")
            return False
        v, dec, raw = nums[index]
        if abs_value:
            v = abs(v)
        ok = rounds_to(expected, v, dec, scale)
        self.checks.append(Check(self.page, panel, label, raw, _show(expected, scale, dec), ok, note))
        if isinstance(el, Tag):
            self.cover(el)
        return ok

    def nums(self, panel: str, label: str, el: Tag | None, expected: Iterable[float | None],
             scale: float = 1.0, note: str = "") -> bool:
        """Every number in ``el``, in order, against ``expected`` (dates excluded)."""
        exp = list(expected)
        txt = text_of(el)
        got = numbers_in(txt)
        if len(got) != len(exp):
            self.fail(panel, label, f"{len(exp)} numbers", f"{len(got)} numbers: {txt[:160]}", note)
            return False
        ok_all = True
        for i, ((v, dec, raw), e) in enumerate(zip(got, exp, strict=True)):
            ok = rounds_to(e, v, dec, scale)
            ok_all &= ok
            self.checks.append(Check(self.page, panel, f"{label} [{i}]", raw, _show(e, scale, dec), ok, note))
        self.cover(el)
        return ok_all

    def text(self, panel: str, label: str, el: Tag | str | None, expected: str, contains: bool = False,
             note: str = "") -> bool:
        """The element's text (normalised) equals, or contains, ``expected``."""
        txt = text_of(el)
        want = norm(expected)
        ok = (want in txt) if contains else (txt == want)
        self.checks.append(Check(self.page, panel, label, txt[:160] or "missing", want, ok, note))
        if isinstance(el, Tag):
            self.cover(el)
        return ok

    def date(self, panel: str, label: str, el: Tag | str | None, expected: str | None,
             index: int = 0) -> bool:
        txt = text_of(el)
        found = [m.group(0) for m in DATE.finditer(txt)]
        got = found[index] if index < len(found) else "missing"
        ok = expected is not None and got == expected
        self.checks.append(Check(self.page, panel, label, got, str(expected), ok))
        if isinstance(el, Tag):
            self.cover(el)
        return ok

    def equal(self, panel: str, label: str, printed: Any, expected: Any, note: str = "") -> bool:
        ok = printed == expected
        self.checks.append(Check(self.page, panel, label, str(printed), str(expected), ok, note))
        return ok

    # ── coverage ──────────────────────────────────────────────────────────────
    def coverage(self, root_selector: str = "main") -> dict[str, Any]:
        """Numeric tokens inside ``root_selector`` (scripts, styles and the chart
        specs excluded), and how many sit inside an element some check verified."""
        root = self.soup.select_one(root_selector)
        total, covered = 0, 0
        missed: dict[str, int] = {}
        if root is None:
            return {"total": 0, "covered": 0, "missed": {}}
        for node in root.find_all(string=True):
            parent = node.parent
            if parent is None or parent.name in ("script", "style", "template"):
                continue
            n = len(numbers_in(str(node))) + len(DATE.findall(str(node)))
            if not n:
                continue
            total += n
            hit = any(id(a) in self.covered for a in [parent, *parent.parents])
            if hit:
                covered += n
            else:
                panel = _panel_of(parent)
                missed[panel] = missed.get(panel, 0) + n
        return {"total": total, "covered": covered, "missed": missed}


def _panel_of(el: Tag) -> str:
    for a in [el, *el.parents]:
        if isinstance(a, Tag) and a.name in ("section", "details", "header", "nav", "footer") and a.get("id"):
            return f"#{a['id']}"
    for a in [el, *el.parents]:
        if isinstance(a, Tag) and a.get("id"):
            return f"#{a['id']}"
    return "(no id)"


def _show(x: float | None, scale: float = 1.0, decimals: int | None = None) -> str:
    if x is None or not math.isfinite(x):
        return "—"
    v = x * scale
    return f"{v:.{decimals}f}" if decimals is not None else f"{v:.10g}"
