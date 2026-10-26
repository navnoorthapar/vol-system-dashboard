"""
site_links.py — the site's internal links, checked (pure functions; no Flask, no I/O).

``link_report`` takes every page's HTML (by its file name) and returns:
  * ``broken``: each internal link with a fragment ('page.html#id', '/route#id'
    or '#id') whose id does not exist on the target page;
  * ``cross``: per page, the distinct other-page fragments it links to from
    inside <main> (its cross-page evidence links). The 'Monitored' chips that
    point back at Model Risk are not counted: they are monitoring, not evidence.
"""
from __future__ import annotations

import html as _html
import re
from collections.abc import Mapping

_HREF = re.compile(r"<a\b([^>]*?)\bhref=\"([^\"]*)\"")
_ID = re.compile(r"\bid=\"([^\"]+)\"")
_FRAG = re.compile(r"^(?P<page>/[\w-]*|[\w-]+\.html)?#(?P<frag>[^&\s]+)$")


def _main(html: str) -> str:
    a, b = html.find("<main"), html.find("</main>")
    return html[a:b] if a >= 0 and b > a else ""


def link_report(pages: Mapping[str, str], routes: Mapping[str, str]) -> dict:
    """``pages``: {file name: html}; ``routes``: {'/route': file name} (root-relative links)."""
    ids = {f: set(_ID.findall(h)) for f, h in pages.items()}
    broken: list[tuple[str, str]] = []
    cross: dict[str, set[str]] = {}
    for f, h in pages.items():
        main = _main(h)
        for _attrs, raw in _HREF.findall(h):
            href = _html.unescape(raw)
            m = _FRAG.match(href)
            if not m:
                continue
            target = m.group("page")
            page = routes.get(target, target) if target else f
            if page not in ids:
                continue                  # not a page of this site
            if m.group("frag") not in ids[page]:
                broken.append((f, href))
        cross[f] = set()
        for attrs, raw in _HREF.findall(main):
            href = _html.unescape(raw)
            m = _FRAG.match(href)
            if not m or not m.group("page") or "mon-link" in attrs:
                continue
            page = routes.get(m.group("page"), m.group("page"))
            if page in ids and page != f:
                cross[f].add(f"{page}#{m.group('frag')}")
    return {"broken": broken, "cross": cross}
