"""
freeze_site.py — Render the Flask dashboard to static HTML for GitHub Pages.

The live dashboard (app.py) reads a point-in-time snapshot from data_store/, so it
renders deterministically — perfect for static hosting. This script drives Flask's
test client over the 4 routes, rewrites the absolute nav links to relative .html
(so the pages work both at navnoorbawa.me and the github.io project URL), and writes
the CNAME + .nojekyll files GitHub Pages needs.

Usage:
    python dashboard/freeze_site.py [OUTPUT_DIR]
    (OUTPUT_DIR defaults to <repo>/.site-build)
"""
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(ROOT, ".site-build")
os.makedirs(OUT, exist_ok=True)

sys.path.insert(0, ROOT)
spec = importlib.util.spec_from_file_location(
    "dash_app", os.path.join(ROOT, "dashboard", "app.py")
)
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)
app = mod.app

ROUTES = {
    "/": "index.html",
    "/calibration": "calibration.html",
    "/greeks": "greeks.html",
    "/backtest": "backtest.html",
}
REWRITES = [
    ('href="/"', 'href="index.html"'),
    ('href="/calibration"', 'href="calibration.html"'),
    ('href="/greeks"', 'href="greeks.html"'),
    ('href="/backtest"', 'href="backtest.html"'),
]

client = app.test_client()
all_ok = True
for route, fname in ROUTES.items():
    resp = client.get(route)
    html = resp.get_data(as_text=True)
    for a, b in REWRITES:
        html = html.replace(a, b)
    with open(os.path.join(OUT, fname), "w") as f:
        f.write(html)
    broken = "Traceback" in html or "werkzeug.exceptions" in html
    ok = resp.status_code == 200 and not broken
    all_ok = all_ok and ok
    print(f"{'OK' if ok else 'FAIL'} {route:14s} -> {fname:18s} "
          f"{len(html):7d} bytes  status={resp.status_code}"
          f"{'  <-- ERROR IN PAGE' if broken else ''}")

from datetime import date as _date

with open(os.path.join(OUT, "CNAME"), "w") as f:
    f.write("navnoorbawa.me\n")
with open(os.path.join(OUT, ".nojekyll"), "w") as f:
    f.write("")

# robots.txt + sitemap.xml — let crawlers index the four pages cleanly
with open(os.path.join(OUT, "robots.txt"), "w") as f:
    f.write("User-agent: *\nAllow: /\nSitemap: https://navnoorbawa.me/sitemap.xml\n")

_today = _date.today().isoformat()
_urls = "".join(
    f"  <url><loc>https://navnoorbawa.me/{p}</loc>"
    f"<lastmod>{_today}</lastmod></url>\n"
    for p in ("", "calibration.html", "greeks.html", "backtest.html")
)
with open(os.path.join(OUT, "sitemap.xml"), "w") as f:
    f.write('<?xml version="1.0" encoding="UTF-8"?>\n'
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
            f"{_urls}</urlset>\n")

# Branded 404 — GitHub Pages serves /404.html on unknown paths
with open(os.path.join(OUT, "404.html"), "w") as f:
    f.write(
        "<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'>"
        "<meta name='viewport' content='width=device-width,initial-scale=1'>"
        "<title>404 · SPX/VIX Vol System</title>"
        "<link rel='icon' href=\"data:image/svg+xml,<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><rect width='32' height='32' rx='6' fill='%230a0a0a'/><path d='M5 10 Q16 28 27 10' fill='none' stroke='%2300ff88' stroke-width='3' stroke-linecap='round'/></svg>\">"
        "<style>html,body{height:100%;margin:0}body{background:#0a0a0a;color:#e8e8e8;"
        "font-family:system-ui,-apple-system,Segoe UI,Roboto,sans-serif;display:flex;"
        "align-items:center;justify-content:center;text-align:center}"
        "h1{font-size:72px;margin:0;color:#00ff88}p{color:#8a8a8a}"
        "a{color:#00ff88;text-decoration:none;font-weight:700}</style></head>"
        "<body><div><h1>404</h1>"
        "<p>That page isn't part of the vol system.</p>"
        "<p><a href='/'>← Back to the dashboard</a></p></div></body></html>"
    )

# Copy the social-preview card so the absolute og:image URL resolves
# (regenerate it with `python dashboard/make_og_image.py`).
import shutil
og_src = os.path.join(ROOT, "dashboard", "og-image.png")
if os.path.exists(og_src):
    shutil.copy(og_src, os.path.join(OUT, "og-image.png"))
    print(f"COPIED og-image.png -> {OUT}")
else:
    print("WARN  og-image.png not found — run dashboard/make_og_image.py")

print(f"{'WROTE' if all_ok else 'WROTE (WITH ERRORS)'} CNAME + .nojekyll -> {OUT}")
sys.exit(0 if all_ok else 1)
