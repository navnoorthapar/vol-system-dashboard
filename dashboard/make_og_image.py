"""
make_og_image.py — Generate the 1200×630 social-preview card (og-image.png).

The base template points og:image at https://navnoorbawa.me/og-image.png for
LinkedIn, Twitter and Substack link previews. The card is drawn in the site's
desk palette: every colour below is a token in static/desk.css, which
test_dashboard.py checks.

The card carries only facts that do not drift from commit to commit. An
earlier version showed a hand-typed test count ("631 tests") that was already
stale against the page meta (666) and CI (642 passed + 24 skipped). A count
belongs to CI and the README, not to a PNG nobody regenerates, so
test_dashboard.py also checks that the card's stats carry no test count.

Usage:
    python dashboard/make_og_image.py [OUTPUT_PATH]
    The default output is <repo>/dashboard/og-image.png. freeze_site.py copies
    it to the site root, so the absolute og:image URL resolves.
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# desk.css tokens (dark desk palette)
BG = "#111418"        # --bg
BG_DEEP = "#0d1014"   # --bg-deep
SURFACE = "#1c2127"   # --surface
LINE = "#2f343c"      # --line
LINE2 = "#404854"     # --line2
TEXT = "#e5e8eb"      # --text
TEXT2 = "#abb3bf"     # --text2
TEXT3 = "#9aa3b0"     # --text3
ACCENT = "#8abbff"    # --pos / --info
OK = "#72ca9b"        # --ok
PALETTE = (BG, BG_DEEP, SURFACE, LINE, LINE2, TEXT, TEXT2, TEXT3, ACCENT, OK)

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "dashboard", "og-image.png")


def _correction_rounds() -> str:
    """'C16–C23': the first and last round of the corrections registry."""
    with open(os.path.join(ROOT, "dashboard", "corrections.json"), encoding="utf-8") as fh:
        rounds = json.load(fh)["rounds"]
    return f"{rounds[0]['id']}–{rounds[-1]['id']}"


# The project's title (README, CITATION.cff): joint SPX/VIX calibration, not a
# joint SPX/VIX *smile* fit; the VIX-options leg is off (CLM-05).
TITLE_LINES = ("Joint SPX/VIX", "Calibration")
SUBTITLE = "The SPX smile and the VIX index term structure, one Heston fit"
# (value, label) pairs of the stat band: stable facts only (no test count, no metric that moves daily)
STATS = [("SPX + VIX", "one joint Heston fit"), ("2018–2025", "hedged-straddle backtest"),
         (_correction_rounds(), "correction rounds logged")]
CHIPS = ["Heston (joint)", "SVI surface", "Rough Heston", "Greeks", "Bootstrap & nulls"]
SANS, MONO = "DejaVu Sans", "DejaVu Sans Mono"


def build(out: str = DEFAULT_OUT) -> str:
    fig = plt.figure(figsize=(12.0, 6.3), dpi=100)        # 1200 × 630 px
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")

    # status bar, as on the site
    ax.add_patch(plt.Rectangle((0, 598), 1200, 32, color=BG_DEEP, lw=0))
    ax.plot([0, 1200], [598, 598], color=LINE, lw=1)
    ax.add_patch(plt.Circle((40, 614), 5, color=OK, lw=0))
    ax.text(54, 614, "END OF DAY   ·   SPX / VIX   ·   STATIC BUILD FROM STORED ARTIFACTS",
            color=TEXT3, fontsize=12, family=MONO, va="center")

    # smile motif: a schematic (drawn from a formula, not a session's data) of
    # quotes with bid-ask bars around a model curve, and labelled as such
    k = np.linspace(-1.0, 1.0, 200)
    x0, w, y0, h = 924, 216, 330, 190
    curve = 0.15 + 0.55 * (k - 0.25) ** 2 + 0.12 * np.maximum(-k, 0)
    ax.add_patch(plt.Rectangle((x0 - 24, y0 - 34), w + 48, h + 92, color=SURFACE, ec=LINE, lw=1))
    ax.plot(x0 + w * (k + 1) / 2, y0 + h * curve, color=ACCENT, lw=3, solid_capstyle="round")
    for kq in np.linspace(-0.9, 0.9, 13):
        cq = 0.15 + 0.55 * (kq - 0.25) ** 2 + 0.12 * max(-kq, 0)
        off = 0.035 * np.sin(7.3 * kq)
        xq, yq = x0 + w * (kq + 1) / 2, y0 + h * (cq + off)
        ax.plot([xq, xq], [yq - 9, yq + 9], color=TEXT3, lw=1.4)
        ax.add_patch(plt.Circle((xq, yq), 3.2, color=TEXT, lw=0))
    ax.text(x0 + w / 2, y0 - 20, "ln(K/F)", color=TEXT3, fontsize=12, family=MONO, ha="center")
    ax.text(x0 - 16, y0 + h + 40, "schematic: model vs bid–ask", color=TEXT2, fontsize=13, family=SANS)

    # title block
    ax.text(64, 500, TITLE_LINES[0], color=TEXT, fontsize=50, fontweight="bold", family=SANS)
    ax.text(64, 436, TITLE_LINES[1], color=TEXT, fontsize=50, fontweight="bold", family=SANS)
    ax.text(66, 388, SUBTITLE, color=TEXT2, fontsize=19, family=SANS)

    # chips
    x = 66
    for c in CHIPS:
        cw = 22 + len(c) * 9.6
        ax.add_patch(plt.Rectangle((x, 312), cw, 34, color=SURFACE, ec=LINE2, lw=1))
        ax.text(x + cw / 2, 329, c, color=TEXT, fontsize=13.5, ha="center", va="center", family=SANS)
        x += cw + 10

    # stat band
    ax.plot([64, 1136], [262, 262], color=LINE, lw=1)
    x = 66
    for val, lab in STATS:
        ax.text(x, 202, val, color=ACCENT, fontsize=30, fontweight="bold", family=MONO)
        ax.text(x + 2, 168, lab, color=TEXT3, fontsize=15, family=SANS)
        x += 360
    ax.plot([64, 1136], [128, 128], color=LINE, lw=1)

    # footer
    ax.text(66, 70, "navnoorbawa.me", color=ACCENT, fontsize=22, fontweight="bold", family=MONO)
    ax.text(66, 36, "Navnoor Bawa  ·  personal research project  ·  end-of-day data",
            color=TEXT3, fontsize=15, family=SANS)

    # no Software/creation metadata, so the PNG is byte-stable for the same inputs
    fig.savefig(out, facecolor=BG, dpi=100, metadata={"Software": None})
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    print(f"Wrote og-image.png -> {path}")
