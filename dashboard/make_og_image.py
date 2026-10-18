"""
make_og_image.py — Generate the 1200×630 social-preview card (og-image.png).

The base template references https://navnoorbawa.me/og-image.png for LinkedIn /
Twitter / Substack link previews. This script renders that card in the
dashboard's dark/green brand so the preview is rich rather than blank.

The card carries only facts that do not drift with each commit. It used to
show a hand-typed test count ('631 tests') that was already stale against the
page meta ('666') and CI (642 passed + 24 skipped); a count belongs to CI and
the README, not to a PNG nobody regenerates. test_dashboard.py checks that the
card's stats and the page meta carry no test count.

Usage:
    python dashboard/make_og_image.py [OUTPUT_PATH]
    (defaults to <repo>/dashboard/og-image.png; freeze_site.py copies it to the
     site root so the absolute og:image URL resolves.)
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BG     = "#0a0a0a"
GREEN  = "#00ff88"
WHITE  = "#e8e8e8"
MUTED  = "#8a8a8a"
PANEL  = "#141414"

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_OUT = os.path.join(ROOT, "dashboard", "og-image.png")

# (value, label) pairs of the stat band — stable facts only (no test count, no correction-round number)
STATS = [("14", "components"), ("0", "look-ahead"), ("2018–25", "backtest")]


def build(out: str = DEFAULT_OUT) -> str:
    # 1200×630 at dpi=100
    fig = plt.figure(figsize=(12.0, 6.3), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1200)
    ax.set_ylim(0, 630)
    ax.axis("off")

    # Top accent rule
    ax.add_patch(plt.Rectangle((0, 612), 1200, 18, color=GREEN, lw=0))

    # Volatility-smile motif (top-right corner), echoing the favicon
    k = np.linspace(-1, 1, 200)
    smile = k**2
    ax.plot(1010 + 150 * (k + 1) / 2, 520 + 64 * smile,
            color=GREEN, lw=4, alpha=0.8, solid_capstyle="round")

    # Title
    ax.text(70, 500, "Joint SPX / VIX", color=WHITE, fontsize=52,
            fontweight="bold", family="DejaVu Sans")
    ax.text(70, 436, "Smile Calibration System", color=GREEN, fontsize=52,
            fontweight="bold", family="DejaVu Sans")

    # One-line honest positioning
    ax.text(72, 372,
            "Calibration & risk infrastructure  +  a documented negative result",
            color=WHITE, fontsize=23, family="DejaVu Sans")

    # Tech chips
    chips = ["Heston", "SVI / SSVI", "Quintic OU", "Bates SVJ",
             "PDV", "Regime classifier"]
    x = 72
    for c in chips:
        w = 24 + len(c) * 12.5
        ax.add_patch(plt.Rectangle((x, 286), w, 40, color=PANEL, lw=1,
                                   ec="#2a2a2a", joinstyle="round"))
        ax.text(x + w / 2, 306, c, color=WHITE, fontsize=16,
                ha="center", va="center", family="DejaVu Sans")
        x += w + 16

    # Stat band
    x = 72
    for val, lab in STATS:
        ax.text(x, 196, val, color=GREEN, fontsize=40, fontweight="bold",
                family="DejaVu Sans Mono")
        ax.text(x + 4, 158, lab, color=MUTED, fontsize=17, family="DejaVu Sans")
        x += 80 + max(33 * len(val), 11 * len(lab))

    # Footer
    ax.text(72, 60, "navnoorbawa.me", color=GREEN, fontsize=24,
            fontweight="bold", family="DejaVu Sans Mono")
    ax.text(72, 28, "Navnoor Bawa  ·  honest negative result, validated end-to-end",
            color=MUTED, fontsize=16, family="DejaVu Sans")

    fig.savefig(out, facecolor=BG, dpi=100)
    plt.close(fig)
    return out


if __name__ == "__main__":
    path = build(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUT)
    print(f"Wrote og-image.png -> {path}")
