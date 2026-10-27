#!/usr/bin/env python3
"""
make_readme_figure.py — the README's figure, drawn from the committed surface report.

Four expiries of one session (default 2026-10-16): every stored quote as its
bid–ask implied-vol bar, the arbitrage-free calendar-ordered SVI surface
(svi_cal) and the joint SPX/VIX Heston fit (heston_v2) at the quoted strikes.
Each panel's title carries both models' out-of-sample median |model − mid| in
half-spreads on that expiry. Everything is read from
data_store/surface/<session>/residuals.parquet; nothing is refitted.

Writes docs/figures/smile_<session>.png and a sidecar
docs/figures/smile_<session>.json with the numbers drawn and the SHA-256 of
the residual file (test_readme.py checks both against the artifact).

    python scripts/make_readme_figure.py [--session 2026-10-16] [--out docs/figures]
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SURFACE = "svi_cal"
HESTON = "heston_v2"
TARGET_DAYS = (14, 31, 91, 364)        # nearest stored expiry to each
# validated categorical pair (dataviz reference palette, slots 1 and 2, light surface)
BLUE, ORANGE, INK, MUTED, GRID = "#2a78d6", "#eb6834", "#0b0b0b", "#8a8984", "#e6e5e0"


def pick_expiries(res: pd.DataFrame, targets=TARGET_DAYS) -> list[str]:
    t = res.groupby("expiry")["T_days"].first()
    out: list[str] = []
    for d in targets:
        e = str((t - d).abs().idxmin())
        if e not in out:
            out.append(e)
    return out


def figure_numbers(res: pd.DataFrame, session: str, expiries: list[str]) -> dict:
    """The numbers the figure shows, per expiry (out-of-sample medians in half-spreads)."""
    rows = []
    for e in expiries:
        q = res[res["expiry"] == e]
        row: dict = {"expiry": e, "T_days": round(float(q["T_days"].iloc[0]), 3),
                     "n_quotes": int((q["model"] == HESTON).sum())}
        for m in (SURFACE, HESTON):
            oos = q[(q["model"] == m) & ~q["in_sample"].astype(bool)]
            row[f"{m}_oos_median_abs_hs"] = round(float(oos["error_hs"].abs().median()), 4)
            row[f"{m}_n_oos"] = len(oos)
        rows.append(row)
    return {"session": session, "surface_model": SURFACE, "heston_model": HESTON, "expiries": rows}


def draw(res: pd.DataFrame, nums: dict, path: Path) -> None:
    plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "xtick.color": INK, "ytick.color": INK, "svg.hashsalt": "readme"})
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.4), dpi=150, constrained_layout=True)
    for ax, row in zip(axes.flat, nums["expiries"]):
        q = res[res["expiry"] == row["expiry"]]
        mkt = q[q["model"] == HESTON].sort_values("k")
        ax.vlines(mkt["k"], mkt["bid_iv"] * 100, mkt["ask_iv"] * 100, color="#52514e", linewidth=1.2,
                  label="SPX bid–ask (implied vol)")
        for m, colour, style, name in ((SURFACE, BLUE, "-", "arbitrage-free SVI surface"),
                                       (HESTON, ORANGE, "--", "joint SPX/VIX Heston fit")):
            g = q[q["model"] == m].sort_values("k")
            ax.plot(g["k"], g["model_iv"] * 100, color=colour, linestyle=style, linewidth=2.0, label=name)
        ax.set_title(f"{row['expiry']} · {row['T_days']:.0f} days\n"
                     f"median |model − mid| out of sample: SVI {row[SURFACE + '_oos_median_abs_hs']:.2f}, "
                     f"Heston {row[HESTON + '_oos_median_abs_hs']:.1f} half-spreads",
                     fontsize=8.5, color=INK, loc="left")
        ax.set_xlabel("log-moneyness ln(K/F)")
        ax.set_ylabel("implied vol (%)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
    axes.flat[0].legend(loc="upper right", frameon=False, fontsize=8)
    fig.suptitle(f"SPX options, {nums['session']}: bid–ask quotes, the arbitrage-free SVI surface and the "
                 f"joint SPX/VIX Heston fit\n(data_store/surface/{nums['session']}/residuals.parquet)",
                 fontsize=10, color=INK)
    fig.savefig(path, facecolor="white", metadata={"Software": None})
    plt.close(fig)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--session", default="2026-10-16")
    ap.add_argument("--out", default=str(ROOT / "docs" / "figures"))
    a = ap.parse_args(argv)
    src = ROOT / "data_store" / "surface" / a.session / "residuals.parquet"
    res = pd.read_parquet(src)
    nums = figure_numbers(res, a.session, pick_expiries(res))
    nums["source"] = str(src.relative_to(ROOT))
    nums["source_sha256"] = hashlib.sha256(src.read_bytes()).hexdigest()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    draw(res, nums, out / f"smile_{a.session}.png")
    (out / f"smile_{a.session}.json").write_text(json.dumps(nums, indent=1) + "\n")
    print(f"wrote {out / f'smile_{a.session}.png'} and its .json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
