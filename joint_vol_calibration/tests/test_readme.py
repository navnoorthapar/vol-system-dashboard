"""
test_readme.py — every number the README front door quotes is the committed artifact's.

The README's first screen quotes three results with their intervals, and its
calibration and backtest sections quote tables. Each figure is recomputed here
from the file it names and must appear in the README exactly as printed. The
results are pinned to the five sessions C23 measured (2026-03-24 ... 2026-10-16),
so the daily job adding a session does not change them; a regenerated artifact
that moves a number fails here until the README (and the docs that quote the
same number) say the new one.

Also: the figure is drawn from the surface report it names (its sidecar's
numbers and source hash), docs/corrections.md is the render of the corrections
registry, every relative link resolves, the badges carry no hand-typed counts,
and QuantLib is a dev-only test oracle.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import pickle
import re
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data_store"
README_RAW = (ROOT / "README.md").read_text()
FIRST_SCREEN_RAW = README_RAW.split("\n---\n")[0]
# whitespace-collapsed copies: a re-wrapped paragraph must not fail a number check
README = " ".join(README_RAW.split())
FIRST_SCREEN = " ".join(FIRST_SCREEN_RAW.split())
SESSIONS = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")
HEADLINE = "2026-10-16"
MINUS = "−"


def fmt(x: float, spec: str) -> str:
    return format(x, spec).replace("-", MINUS)


def rng(values, spec: str) -> str:
    v = list(values)
    return f"{fmt(min(v), spec)}–{fmt(max(v), spec)}"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def scorecards():
    out = {}
    for s in SESSIONS:
        sc = json.loads((DATA / "surface" / s / "scorecard.json").read_text())
        out[s] = {m["model"]: m for m in sc["models"]}
    return out


@pytest.fixture(scope="module")
def diag():
    rows = json.loads((DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]
    by = {r["session"]: r for r in rows}
    return [by[s] for s in SESSIONS]


@pytest.fixture(scope="module")
def inference():
    return json.loads((DATA / "backtest" / "inference" / "summary.json").read_text())


def test_first_screen_is_short_and_names_its_parts():
    body = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", FIRST_SCREEN_RAW)
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    words = re.findall(r"[A-Za-z0-9][\w.,'’%·−/-]*", body)
    assert len(words) <= 340, len(words)
    for part in ("**Three results**", "**Limitations.**", "**Links.**", "docs/figures/smile_2026-10-16.png",
                 "https://navnoorbawa.me", "docs/model_validation/README.md", "docs/reproduce.md",
                 "docs/corrections.md", "docs/free_resources_scorecard.md", "CITATION.cff",
                 "docs/install.md"):
        assert part in FIRST_SCREEN, part
    # research prototypes are not on the first screen
    for word in ("classifier", "XGBoost", "neural", "NN "):
        assert word not in FIRST_SCREEN, word


def test_result_1_quote_grade_surface(scorecards):
    oos = {s: scorecards[s]["svi_cal"]["metrics"]["out_of_sample"] for s in SESSIONS}
    h = oos[HEADLINE]
    assert f"**{fmt(h['median_abs_hs'], '.2f')} half-spreads**" in FIRST_SCREEN
    assert f"({fmt(h['inside_spread_pct'], '.1f')}% inside it)" in FIRST_SCREEN
    assert f"**{rng((o['median_abs_hs'] for o in oos.values()), '.2f')}** half-spreads" in FIRST_SCREEN
    dg = scorecards[HEADLINE]["svi_cal"]["dense_grid"]
    assert dg["butterfly_violations"] == 0
    assert dg["calendar_violations"] == 0          # "arbitrage-free" on its dense grid
    hes = [scorecards[s]["heston_v2"]["metrics"]["out_of_sample"]["median_abs_hs"] for s in SESSIONS]
    svi = rng((o["median_abs_hs"] for o in oos.values()), ".2f")
    claim = f"Heston's out-of-sample median |model − mid| is {rng(hes, '.1f')} half-spreads against {svi}"
    assert claim in README


def test_result_2_joint_calibration_heston_and_rough(diag):
    cur = next(r for r in diag if r["session"] == HEADLINE)
    with open(DATA / "calibrations" / f"joint_cal_{HEADLINE}.pkl", "rb") as fh:
        cal = pickle.load(fh)
    n_oos = cal["oos"]["n_spx_oos"]
    heston_oos = fmt(cur["sigma_cap2_oos"], ".2f")
    assert f"misses the other {n_oos:,} quotes by **{heston_oos} vol pts**" in FIRST_SCREEN
    assert f"Rough Heston on the same quotes: **{fmt(cur['joint_rough_oos'], '.2f')}**" in FIRST_SCREEN
    heston = [r["sigma_cap2_oos"] for r in diag]
    rough = [r["joint_rough_oos"] for r in diag]
    assert f"Heston {rng(heston, '.2f')}, rough Heston {rng(rough, '.2f')}" in FIRST_SCREEN
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in diag)
    same = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)
    verdict = f"rough is lower on {lower} of {len(diag)} sessions and returns H = ½ (Heston) on {same}"
    assert verdict in FIRST_SCREEN
    assert "sigma" in cur["on_bound"]                  # "with σ on its 2.0 cap"
    assert len(cal["anchors_by_expiry"]) == 6
    assert sum(a["n"] for a in cal["anchors_by_expiry"]) == 53


def test_result_3_backtest_interval_and_nulls(inference):
    b = inference["bootstrap"]["portfolio"]["sharpe"]
    assert f"Portfolio Sharpe **{fmt(b['point'], '.2f')}**" in FIRST_SCREEN
    assert f"95% CI **{fmt(b['ci95'][0], '.2f')} to {fmt(b['ci95'][1], '.2f')}**" in FIRST_SCREEN
    spa = inference["multiple_testing"]["spa_published"]["p_consistent"]
    assert f"(Hansen SPA p {spa:.2f})" in FIRST_SCREEN
    s3 = inference["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
    assert f"S3's entry timing has p {s3['p_value']:.3f}" in FIRST_SCREEN
    holm = f"(Holm over six series {s3['p_holm_6_sleeves']:.2f})"
    assert f"against {s3['n_paths']:,} exposure-matched circular shifts {holm}" in FIRST_SCREEN


def test_backtest_table_is_the_canonical_run(inference):
    with open(DATA / "backtest" / "full_results.pkl", "rb") as fh:
        m = pickle.load(fh)["metrics"]

    def usd(x):
        return f"{MINUS if x < 0 else '+'}${abs(x):,.0f}"

    total = m["cumulative_return"] * 1_000_000
    rows = {"Portfolio (S1C + S3 + S4)": (total, m["sharpe"], "portfolio", m["n_trades"])}
    for key, label in (("s1c", "S1C contrarian PDV"), ("s3", "S3 dispersion, index leg"),
                       ("s4", "S4 volatility risk premium"), ("s1", "S1 PDV short-VRP (reference)"),
                       ("s2", "S2 VIX term structure (reference)")):
        rows[label] = (m[f"{key}_total_pnl"], m[f"{key}_sharpe"], key, m[f"{key}_n_trades"])
    for label, (pnl, sharpe, series, trades) in rows.items():
        ci = inference["bootstrap"][series]["sharpe"]["ci95"]
        interval = f"{fmt(ci[0], '.2f')} to {fmt(ci[1], '.2f')}"
        line = f"| {label} | {usd(pnl)} | {fmt(sharpe, '.2f')} | {interval} | {trades} |"
        assert line in README_RAW, line
    nav = 1_000_000 * (1 + m["cumulative_return"])
    assert f"NAV ${nav:,.0f} ({fmt(m['cumulative_return'] * 100, '.1f')}%)" in README
    assert f"T-bills earned {m['cash_cum_return'] * 100:.1f}%" in README
    grid = pd.read_csv(DATA / "backtest" / "beta_sensitivity.csv").set_index("beta")
    assert f"+${grid.loc[1.0, 's1_pnl']:,.0f} at β = 1.00" in README
    assert f"{MINUS}${abs(grid.loc[0.78, 's1_pnl']):,.0f} at β = 0.78" in README
    assert (grid["pnl"] < 0).all()                        # "negative at every β"
    bench = json.loads((DATA / "backtest" / "inference" / "benchmarks.json").read_text())["implied_beta"]
    assert round(bench["PUT"]["overall"], 2) == round(bench["WPUT"]["overall"], 2) == 0.80


def test_calibration_tables_are_the_pickles_and_the_diagnostics(diag):
    for s in SESSIONS:
        with open(DATA / "calibrations" / f"joint_cal_{s}.pkl", "rb") as fh:
            cal = pickle.load(fh)
        p, ll = cal["params"], cal["leg_losses"]
        cells = [f"{p['kappa']:.3f}", f"{p['theta']:.4f}", f"{p['sigma']:.3f}", fmt(p["rho"], ".3f"),
                 f"{p['v0']:.4f}", f"{ll['spx_iv_rmse']:.3f} / {ll['spx_oos_iv_rmse']:.3f}",
                 f"{ll['vix_index_rmse']:.3f}"]
        line = next(x for x in README_RAW.splitlines() if x.startswith((f"| {s} |", f"| **{s}** |")))
        for c in cells:
            assert c in line, (s, c, line)
        r = next(x for x in diag if x["session"] == s)
        rough_line = next(x for x in README_RAW.splitlines()
                          if x.startswith(f"| {s} | {fmt(r['sigma_cap2_oos'], '.2f')} |"))
        cells = (fmt(r["sigma_cap2_oos"], ".2f"), fmt(r["joint_rough_oos"], ".2f"),
                 f"{r['joint_rough_H']:.2f}")
        assert rough_line == f"| {s} | {' | '.join(cells)} |"
    cur = next(r for r in diag if r["session"] == HEADLINE)
    assert (f"gives σ {cur['sigma_free']:.2f}, κ {cur['kappa_free']:.1f}, a loss "
            f"{abs(cur['sigma_free_loss_rel']) * 100:.1f}% lower and an out-of-sample RMSE of "
            f"{cur['sigma_free_oos']:.2f} instead of {cur['sigma_cap2_oos']:.2f}") in README
    first = diag[0]
    cost = f"{abs(first['sigma_free_loss_rel']) * 100:.1f}%"
    assert f"on {first['session']} the box costs {cost} of the loss" in README
    assert (f"model's VVIX is {rng((r['vvix_model_cap2'] for r in diag), '.0f')} against a market ^VVIX of "
            f"{rng((r['vvix_market'] for r in diag), '.0f')}") in README
    cond = rng((r["rel_jtj_condition_number"] for r in diag), ".0f")
    assert f"cond(JᵀJ) of the relative-sensitivity Jacobian is {cond}" in README
    kse = rng((r["relative_standard_errors"]["kappa"] * 100 for r in diag), ".0f")
    assert f"standard error is {kse}% of κ" in README
    worst = max(abs(r["trf_loss_rel_diff"]) for r in diag)
    assert all(r["trf_loss_rel_diff"] <= 0 for r in diag)            # "slightly below them"
    assert f"relative loss of at most {worst:.1e}".replace("e-0", "e-") in README
    trf, de = rng((r["trf_seconds"] for r in diag), ".1f"), rng((r["de_serial_seconds"] for r in diag), ".1f")
    assert f"in {trf} s, against {de} s" in README
    assert all(r["trf_basins"] == 1 for r in diag)        # "one basin on every session"
    idx = {r["session"]: r for r in json.loads((DATA / "surface" / "index.json").read_text())["sessions"]}
    gaps = [abs(idx[s]["vix_replicated"] - idx[s]["vix_official"]) for s in SESSIONS]
    assert f"within {rng(gaps, '.2f')} index points of the official close" in README


def test_figure_is_drawn_from_the_surface_report():
    mrf = _script("make_readme_figure")
    side = json.loads((ROOT / "docs" / "figures" / f"smile_{HEADLINE}.json").read_text())
    src = ROOT / side["source"]
    assert side["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    res = pd.read_parquet(src)
    want = mrf.figure_numbers(res, HEADLINE, mrf.pick_expiries(res))
    assert {k: side[k] for k in want} == want
    png = (ROOT / "docs" / "figures" / f"smile_{HEADLINE}.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert f"](docs/figures/smile_{HEADLINE}.png)" in FIRST_SCREEN


def test_corrections_doc_is_the_render_of_the_registry():
    rc = _script("render_corrections")
    doc = (ROOT / "docs" / "corrections.md").read_text()
    assert rc.splice(doc, rc.render()) == doc, "run python scripts/render_corrections.py"
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    for r in reg["rounds"]:
        assert f"### {r['id']}" in doc
        for c in r["commits"]:
            assert f"/commit/{c})" in doc


def test_relative_links_resolve():
    docs = [ROOT / "README.md", ROOT / "docs" / "corrections.md",
            ROOT / "docs" / "free_resources_scorecard.md",
            ROOT / "docs" / "model_validation" / "README.md", ROOT / "docs" / "README.md",
            ROOT / "substack_article.md"]
    for doc in docs:
        text = doc.read_text()
        for target in re.findall(r"\]\(([^)\s]+)\)", text):
            if re.match(r"[a-z]+:", target) or target.startswith("#"):
                continue
            path = (doc.parent / target.split("#")[0]).resolve()
            assert path.exists(), (doc.name, target)


def test_badges_are_live_and_carry_no_hand_typed_counts():
    badges = re.findall(r"\[!\[[^\]]*\]\(([^)]+)\)\]", FIRST_SCREEN_RAW)
    assert badges
    for url in badges:
        assert not re.search(r"\d+(%25|%)?[-_ ](passed|tests|coverage)", url), url
        m = re.search(r"actions/workflows/([\w.-]+)/badge\.svg", url)
        if m:
            assert (ROOT / ".github" / "workflows" / m.group(1)).exists(), url
    assert not re.search(r"\b\d[\d,]* (tests )?passed\b", FIRST_SCREEN)


def test_quantlib_is_a_dev_only_test_oracle():
    assert "QuantLib is used only as a dev-only test oracle" in README.replace("**", "")
    runtime = (ROOT / "requirements.txt").read_text().lower()
    assert "quantlib" not in runtime
    assert "quantlib" in (ROOT / "requirements-dev.txt").read_text().lower()
    for base in ("joint_vol_calibration", "dashboard", "scripts"):
        for py in (ROOT / base).rglob("*.py"):
            if "tests" in py.parts or "bench" in py.parts:
                continue
            assert not re.search(r"^\s*(import QuantLib|from QuantLib)", py.read_text(), re.MULTILINE), py


def test_article_c23_note_carries_the_canonical_numbers(inference):
    """substack_article.md opens with a dated C23 note; the numbers C23 moved are the artifacts'."""
    art = " ".join((ROOT / "substack_article.md").read_text().split())
    note = art.split("**C23 note")[1].split("**Correction (October 2026, C22).**")[0]
    with open(DATA / "backtest" / "full_results.pkl", "rb") as fh:
        m = pickle.load(fh)["metrics"]
    nav = 1_000_000 * (1 + m["cumulative_return"])
    cum, sharpe = fmt(m["cumulative_return"] * 100, ".1f"), fmt(m["sharpe"], ".2f")
    head = f"**{cum}% (NAV ${nav:,.0f}), Sharpe {sharpe}**"
    assert head in note
    b = inference["bootstrap"]["portfolio"]["sharpe"]["ci95"]
    assert f"{fmt(b[0], '.2f')} to {fmt(b[1], '.2f')}" in note
    for key in ("s1", "s1c", "s3"):
        k = f"{abs(m[key + '_total_pnl']) / 1000:,.0f}K"
        assert k in note, key
