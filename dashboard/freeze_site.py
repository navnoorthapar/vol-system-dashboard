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

with open(os.path.join(OUT, "CNAME"), "w") as f:
    f.write("navnoorbawa.me\n")
with open(os.path.join(OUT, ".nojekyll"), "w") as f:
    f.write("")

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
