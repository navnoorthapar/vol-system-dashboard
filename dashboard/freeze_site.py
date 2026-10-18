"""
freeze_site.py — Render the Flask dashboard to static HTML for GitHub Pages.

The dashboard (app.py) is a read-only view of data_store/, so it renders
deterministically. This script drives Flask's test client over the routes in
app.ROUTES, rewrites the absolute nav links to relative .html (so the pages work
both at navnoorbawa.me and the github.io project URL), and writes the CNAME,
.nojekyll, robots.txt, sitemap.xml and 404 page GitHub Pages needs.

It FAILS CLOSED: if any page returns a non-200 status, contains a traceback, or
reports a loader error (app.py sets the X-Dashboard-Loader-Errors header and
renders a red error banner when an input fails to load), nothing is written and
the script exits 1. scripts/cloud_refresh.py then stops before the deploy step,
so the previous good build stays live instead of a page with missing data.

The vendored assets under dashboard/static (Chart.js) are copied to
<out>/static. The build also fails if a page references a static file that
does not exist, or if the vendored Chart.js no longer matches the SRI hash the
pages carry (a browser would refuse to run it).

Stale inputs (age in NYSE sessions past app.STALE_WARN_SESSIONS) do not block
the build — the pages carry their own banners — but are echoed as GitHub
Actions warning annotations.

Usage:
    python dashboard/freeze_site.py [OUTPUT_DIR]
    (OUTPUT_DIR defaults to <repo>/.site-build)
"""
from __future__ import annotations

import base64
import hashlib
import importlib.util
import os
import re
import shutil
import sys
from typing import Any

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC = os.path.join(ROOT, "dashboard", "static")

_404_HTML = (
    "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
    "<meta name='viewport' content='width=device-width,initial-scale=1'>"
    "<title>404 · SPX/VIX Vol System</title>"
    "<link rel='icon' href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><path d='M5 10 Q16 28 27 10' fill='none' stroke='%2300ff88' stroke-width='3' stroke-linecap='round'/></svg>\">"
    "<style>html,body{height:100%;margin:0}body{background:#0a0a0a;color:#e8e8e8;"
    "font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;"
    "align-items:center;justify-content:center;text-align:center}"
    "h1{font-size:72px;margin:0;color:#00ff88}p{color:#8a8a8a}"
    "a{color:#00ff88;text-decoration:none;font-weight:700}</style></head>"
    "<body><main><h1>404</h1>"
    "<p>That page isn't part of the vol system.</p>"
    "<p><a href='/'>← Back to the dashboard</a></p></main></body></html>"
)


def load_app(root: str = ROOT) -> Any:
    """Import dashboard/app.py as a module (it is not inside a package)."""
    if root not in sys.path:
        sys.path.insert(0, root)
    spec = importlib.util.spec_from_file_location(
        "dash_app", os.path.join(root, "dashboard", "app.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def render_pages(mod: Any) -> tuple[dict, list[str], list[str]]:
    """Render every route. Returns (pages, problems, warnings).

    pages[route] = {"file", "html", "data_date"}; any entry in `problems`
    means the site must not be published.
    """
    client = mod.app.test_client()
    rewrites = [(f'href="{route}"', f'href="{fname}"') for route, fname in mod.ROUTES.items()]
    pages, problems, warnings = {}, [], []
    for route, fname in mod.ROUTES.items():
        resp = client.get(route)
        html = resp.get_data(as_text=True)
        for a, b in rewrites:
            html = html.replace(a, b)
        n_err = resp.headers.get("X-Dashboard-Loader-Errors")
        broken = "Traceback" in html or "werkzeug.exceptions" in html
        page_problems = []
        if resp.status_code != 200:
            page_problems.append(f"HTTP {resp.status_code}")
        if broken:
            page_problems.append("traceback in page")
        if n_err is None:
            page_problems.append("no X-Dashboard-Loader-Errors header (cannot verify the loaders)")
        elif n_err != "0":
            page_problems.append(
                f"{n_err} loader error(s): {resp.headers.get('X-Dashboard-Error-Components', '?')}")
        stale = resp.headers.get("X-Dashboard-Stale")
        if stale:
            warnings.append(f"{route}: stale inputs (NYSE sessions old): {stale}")
        ok = not page_problems
        print(f"{'OK' if ok else 'FAIL'} {route:14s} -> {fname:18s} "
              f"{len(html):7d} bytes  status={resp.status_code}"
              f"{'' if ok else '  <-- ' + '; '.join(page_problems)}")
        problems += [f"{route}: {p}" for p in page_problems]
        pages[route] = {"file": fname, "html": html,
                        "data_date": resp.headers.get("X-Dashboard-Data-Date")}
    return pages, problems, warnings


def sri_sha384(path: str) -> str:
    """Subresource-integrity value ('sha384-<base64>') of a file."""
    with open(path, "rb") as fh:
        return "sha384-" + base64.b64encode(hashlib.sha384(fh.read()).digest()).decode()


def check_static(mod: Any, pages: dict) -> list[str]:
    """Problems with the static assets the rendered pages reference."""
    problems = []
    chartjs = getattr(mod, "CHARTJS", None)
    if chartjs:
        local = os.path.join(STATIC, os.path.relpath(chartjs["src"], "static"))
        if not os.path.exists(local):
            problems.append(f"vendored Chart.js missing: {local}")
        elif sri_sha384(local) != chartjs["sri"]:
            problems.append(f"vendored Chart.js does not match its SRI hash ({chartjs['sri']})")
    for route, page in pages.items():
        for ref in sorted(set(re.findall(r'(?:src|href)="(static/[^"]+)"', page["html"]))):
            if not os.path.exists(os.path.join(STATIC, os.path.relpath(ref, "static"))):
                problems.append(f"{route}: references missing static file {ref}")
    return problems


def sitemap_xml(mod: Any, pages: dict) -> str:
    """Sitemap whose lastmod is the date of the data each page shows.

    Stamping today's date on every page every run told crawlers that pages
    which had not changed (e.g. the backtest) were new. A page whose data date
    is unknown gets no lastmod rather than a wrong one.
    """
    rows = []
    for route, page in pages.items():
        lastmod = f"<lastmod>{page['data_date']}</lastmod>" if page["data_date"] else ""
        rows.append(f"  <url><loc>{mod.canonical_url(route)}</loc>{lastmod}</url>\n")
    return ('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{''.join(rows)}</urlset>\n")


def write_site(out: str, mod: Any, pages: dict) -> None:
    os.makedirs(out, exist_ok=True)
    for page in pages.values():
        with open(os.path.join(out, page["file"]), "w") as f:
            f.write(page["html"])
    with open(os.path.join(out, "CNAME"), "w") as f:
        f.write("navnoorbawa.me\n")
    with open(os.path.join(out, ".nojekyll"), "w") as f:
        f.write("")
    with open(os.path.join(out, "robots.txt"), "w") as f:
        f.write(f"User-agent: *\nAllow: /\nSitemap: {mod.SITE_URL}sitemap.xml\n")
    with open(os.path.join(out, "sitemap.xml"), "w") as f:
        f.write(sitemap_xml(mod, pages))
    # Branded 404 — GitHub Pages serves /404.html on unknown paths
    with open(os.path.join(out, "404.html"), "w") as f:
        f.write(_404_HTML)
    # Vendored assets (Chart.js), served from the same origin as the pages
    if os.path.isdir(STATIC):
        shutil.copytree(STATIC, os.path.join(out, "static"), dirs_exist_ok=True,
                        ignore=shutil.ignore_patterns("__pycache__", ".*"))
        print(f"COPIED static/ -> {out}/static")
    # Social-preview card, so the absolute og:image URL resolves
    # (regenerate it with `python dashboard/make_og_image.py`).
    og_src = os.path.join(ROOT, "dashboard", "og-image.png")
    if os.path.exists(og_src):
        shutil.copy(og_src, os.path.join(out, "og-image.png"))
        print(f"COPIED og-image.png -> {out}")
    else:
        print("WARN  og-image.png not found — run dashboard/make_og_image.py")


def freeze(out: str, mod: Any = None) -> int:
    """Render, check and (only if every page is clean) write the site. Returns an exit code."""
    mod = mod or load_app()
    pages, problems, warnings = render_pages(mod)
    problems += check_static(mod, pages)
    for w in warnings:
        print(f"::warning title=Stale dashboard input::{w}")
    if problems:
        for p in problems:
            print(f"::error title=Dashboard not published::{p}")
        print(f"NOT WRITTEN: {len(problems)} problem(s); nothing was written to {out}")
        return 1
    write_site(out, mod, pages)
    print(f"WROTE {len(pages)} pages + static/, CNAME, .nojekyll, robots.txt, sitemap.xml, 404.html -> {out}")
    return 0


def main(argv: list[str]) -> int:
    out = argv[1] if len(argv) > 1 else os.path.join(ROOT, ".site-build")
    return freeze(out)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
