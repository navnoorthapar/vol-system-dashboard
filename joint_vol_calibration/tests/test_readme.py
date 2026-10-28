"""
test_readme.py — every number the README front door quotes is the committed artifact's.

The README's first screen quotes three results with their intervals, and its
calibration and backtest sections quote tables. Each figure is recomputed here
from the file it names and must appear in the README exactly as printed.

The results lead with the range over the sessions the README names (VALIDATOR-06,
CODE-04, CIO-05: it led with one near-best session, 2026-10-16, and said 'no
ledger entry' after one was committed). The README states the span of those
sessions (2026-03-24 -> its last date); RESULT_SESSIONS is every stored session
in that span with a surface scorecard and a model_version 2 fit, so a session the
daily job adds after the span does not move a number, while a session missing
from the README's count inside it fails here. A single value names its session
and is read from that session's artifact. C23_SESSIONS are the five sessions the
C23 round measured; the claims about that round (the DE timings, the Data table's
option counts, the article's C23 note) stay pinned to them. test_readme.py is in
no gate of the daily job (daily_refresh.yml).

The Data table's row counts are counted as of the end date each row states,
so the daily job appending a session does not move them (CLM-03: the Data
table, the VX column and the limitation figures were outside the tested set).

Also: the figure is drawn from the surface report it names (its sidecar's
numbers and source hash), docs/corrections.md and the validation pack's
findings log are renders of the corrections registry, every relative link
resolves, the badges carry no hand-typed counts, and QuantLib is a dev-only
test oracle.
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
C23_SESSIONS = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")
C23_HEADLINE = "2026-10-16"
FIGURE = "2026-10-16"                       # the session docs/figures/smile_<session>.png draws
MINUS = "−"
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}


def _result_sessions(text: str) -> tuple[str, ...]:
    """Every stored session in the span ``text`` states ('... an accepted fit,
    <first> → <last>'), with a surface scorecard and a model_version 2 fit."""
    m = re.search(r"accepted fit, (\d{4}-\d{2}-\d{2}) → (\d{4}-\d{2}-\d{2})", text)
    assert m, "the text does not state the span of its sessions"
    out = []
    for d in sorted((DATA / "surface").iterdir()):
        s = d.name
        pkl = DATA / "calibrations" / f"joint_cal_{s}.pkl"
        if not (m.group(1) <= s <= m.group(2) and (d / "scorecard.json").exists() and pkl.exists()):
            continue
        with open(pkl, "rb") as fh:
            if pickle.load(fh).get("model_version") == 2:
                out.append(s)
    assert out[0] == m.group(1)
    assert out[-1] == m.group(2)
    return tuple(out)


RESULT_SESSIONS = _result_sessions(FIRST_SCREEN)
LAST = RESULT_SESSIONS[-1]
PACK_RAW = (ROOT / "docs" / "model_validation" / "README.md").read_text()
PACK = " ".join(PACK_RAW.split())


def _cal(s: str) -> dict:
    with open(DATA / "calibrations" / f"joint_cal_{s}.pkl", "rb") as fh:
        return pickle.load(fh)


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
    for s in RESULT_SESSIONS:
        sc = json.loads((DATA / "surface" / s / "scorecard.json").read_text())
        out[s] = {m["model"]: m for m in sc["models"]}
    return out


@pytest.fixture(scope="module")
def diag():
    rows = json.loads((DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]
    by = {r["session"]: r for r in rows}
    return [by[s] for s in RESULT_SESSIONS]


@pytest.fixture(scope="module")
def inference():
    return json.loads((DATA / "backtest" / "inference" / "summary.json").read_text())


def test_first_screen_is_short_and_names_its_parts():
    body = re.sub(r"\[!\[.*?\]\(.*?\)\]\(.*?\)", "", FIRST_SCREEN_RAW)
    body = re.sub(r"!\[.*?\]\(.*?\)", "", body)
    body = re.sub(r"\]\([^)]*\)", "]", body)
    words = re.findall(r"[A-Za-z0-9][\w.,'’%·−/-]*", body)
    assert len(words) <= 340, len(words)
    for part in ("**What it is, and is not.**", "A model-validation record", "It is not an alpha claim",
                 "the backtest result is negative and no signal is deployable", "**Three results**",
                 "**Limitations.**", "**Links.**",
                 f"docs/figures/smile_{FIGURE}.png",
                 "https://navnoorbawa.me", "docs/model_validation/README.md", "docs/reproduce.md",
                 "docs/corrections.md", "docs/free_resources_scorecard.md", "CITATION.cff",
                 "docs/install.md"):
        assert part in FIRST_SCREEN, part
    # research prototypes are not on the first screen
    for word in ("classifier", "XGBoost", "neural", "NN "):
        assert word not in FIRST_SCREEN, word


def test_first_screen_limitations_are_the_records():
    first = min(json.loads(p.read_text())["session"] for p in (DATA / "options" / "cboe").glob("spx_*.json"))
    assert f"Cboe's timestamped books are stored from {first}" in FIRST_SCREEN
    assert f"Cboe's timestamped books are stored from {first}" in README.split("## Known limitations")[1]
    assert "VIX options are a research leg, not in the fit" in FIRST_SCREEN
    for s in RESULT_SESSIONS:                           # ... because every committed fit has w3 = 0
        assert _cal(s)["weights"]["w3"] == 0.0, s
    # the forward ledger: its first registered session, never 'no entry' once one is committed
    ledgers = sorted(p.stem.removeprefix("ledger_") for p in (DATA / "forward").glob("ledger_*.json"))
    sources = " ".join((ROOT / "DATA_SOURCES.md").read_text().split())
    if ledgers:
        assert f"the forward ledger at executable prices starts on {ledgers[0]}" in FIRST_SCREEN
        assert f"Its first entry is {ledgers[0]}" in README
        assert f"first entry {ledgers[0]}" in sources
        for text in (README, sources):
            assert "no entry yet" not in text


def test_results_lead_with_the_range_over_the_named_sessions():
    """The results' preamble counts the sessions it ranges over, and names the
    live site for the newest one."""
    n = len(RESULT_SESSIONS)
    preamble = f"ranges over the {WORDS.get(n, n)} sessions with a stored chain and an accepted fit"
    assert preamble in FIRST_SCREEN
    assert f"{LAST}, not confidence intervals" in FIRST_SCREEN
    live = "[live site](https://navnoorbawa.me) shows the newest session"
    assert live in FIRST_SCREEN_RAW.replace("\n", " ")


def test_result_1_quote_grade_surface(scorecards):
    oos = {s: scorecards[s]["svi_cal"]["metrics"]["out_of_sample"] for s in RESULT_SESSIONS}
    h = oos[LAST]
    assert f"**{rng((o['median_abs_hs'] for o in oos.values()), '.2f')} half-spreads**" in FIRST_SCREEN
    assert f"{rng((o['inside_spread_pct'] for o in oos.values()), '.1f')}% inside the bid–ask" in FIRST_SCREEN
    assert (f"({LAST}: {fmt(h['median_abs_hs'], '.2f')}, {fmt(h['inside_spread_pct'], '.1f')}% inside)"
            in FIRST_SCREEN)
    for s in RESULT_SESSIONS:
        dg = scorecards[s]["svi_cal"]["dense_grid"]
        assert dg["butterfly_violations"] == 0, s
        assert dg["calendar_violations"] == 0, s   # "arbitrage-free" on its dense grid
    # CLM-15: compared on the quotes no model was fitted to, not "the same quotes" each model held out
    common = {s: {m: scorecards[s][m]["metrics"]["common_out_of_sample"] for m in ("heston_v2", "svi_cal")}
              for s in RESULT_SESSIONS}
    n = rng((common[s]["svi_cal"]["n"] for s in RESULT_SESSIONS), ",d")
    hes = rng((common[s]["heston_v2"]["median_abs_hs"] for s in RESULT_SESSIONS), ".1f")
    svi = rng((common[s]["svi_cal"]["median_abs_hs"] for s in RESULT_SESSIONS), ".2f")
    assert all(common[s]["svi_cal"]["n"] == common[s]["heston_v2"]["n"] for s in RESULT_SESSIONS)
    claim = (f"On the {n} quotes per session that no model was fitted to (`data_store/surface/<session>/"
             f"scorecard.json`, `metrics.common_out_of_sample`), Heston's median |model − mid| is {hes} "
             f"half-spreads against {svi} for the certified arbitrage-free SVI surface")
    assert claim in README
    for s in RESULT_SESSIONS:                           # "certified": the label follows the certificate
        surf = json.loads((DATA / "surface" / s / "surface.json").read_text())
        assert surf["surfaces"]["svi_cal"]["arbitrage_free_certified"] is True, s


def test_result_2_joint_calibration_heston_and_rough(diag):
    from joint_vol_calibration.config import HESTON_BOUNDS

    cals = {s: _cal(s) for s in RESULT_SESSIONS}
    anchors = [sum(a["n"] for a in c["anchors_by_expiry"]) for c in cals.values()]
    assert f"fitted to {rng(anchors, 'd')} SPX anchors and the four Cboe VIX indices" in FIRST_SCREEN
    heston = [r["sigma_cap2_oos"] for r in diag]
    assert f"misses held-out quotes by **{rng(heston, '.2f')} vol pts** RMSE" in FIRST_SCREEN
    # the single value names its session, and is that session's fit
    last = cals[LAST]
    on_bound = "no parameter on a bound" if not last.get("pinned_params") else "on a bound"
    assert (f"({LAST}: {fmt(last['leg_losses']['spx_oos_iv_rmse'], '.2f')} on {last['oos']['n_spx_oos']:,} "
            f"quotes, {on_bound})") in FIRST_SCREEN
    cap = HESTON_BOUNDS["sigma"][1]
    capped = [s for s, c in cals.items() if "sigma" in (c.get("pinned_params") or [])]
    assert all(cals[s]["params"]["sigma"] == pytest.approx(cap) for s in capped)
    assert (f"σ sits on its {cap:.1f} cap in {len(capped)} of the {len(cals)} fits ({', '.join(capped)})"
            in FIRST_SCREEN)
    rough = [r["joint_rough_oos"] for r in diag]
    assert f"Rough Heston: **{rng(rough, '.2f')}**" in FIRST_SCREEN
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in diag)
    same = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)
    verdict = f"lower on {lower} of {len(diag)} sessions, H = ½ (Heston) on {same}"
    assert verdict in FIRST_SCREEN


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


def test_backtest_prose_is_the_canonical_run_and_its_inference(inference):
    from joint_vol_calibration.backtest.backtest_engine import ATM_TO_VIX_RATIO

    eq = pd.read_parquet(DATA / "backtest" / "full_results.parquet")
    start, end = (str(pd.Timestamp(x).date()) for x in (eq.index[0], eq.index[-1]))
    assert f"2018-01-02 → 2025-03-24 ({len(eq):,} sessions)" in README
    assert (start, end) == ("2018-01-02", "2025-03-24")
    assert f"Backtest, {start} → {end}, straddles marked at β·VIX" in FIRST_SCREEN
    assert f"β = {ATM_TO_VIX_RATIO:.2f} estimated from" in README
    # STATS-3: the DSR at the ledger's counts, and the lenient bound with its basis
    t = inference["trials"]
    s3 = t["dsr_range"]["s3"]
    led = s3["at_ledger_counts"]
    n = t["n_choices"]
    counts = [n[k] for k in t["n_choices_ledger_based"]]
    assert (f"S3's deflated Sharpe ratio is {led['min']:.3f}–{led['max']:.3f} at the ledger's trial counts "
            f"(N {min(counts):.0f}–{max(counts):.0f}); it reaches {s3['max']:.2f} only if the search is "
            f"counted as {n['n_eigen_participation']:.1f} independent trials") in README
    assert s3["argmax"].endswith("n_eigen_participation")


def test_data_table_counts_as_of_their_end_dates():
    """Each row's count is recomputed for the coverage it states, so the daily
    job appending sessions after it does not move the number."""
    import sqlite3

    def row(name: str) -> str:
        return next(x for x in README_RAW.splitlines() if x.startswith(f"| {name} |"))

    def span(line: str) -> tuple[str, str]:
        m = re.search(r"(\d{4}-\d{2}-\d{2}) → (\d{4}-\d{2}-\d{2})", line)
        assert m is not None, line
        return m.group(1), m.group(2)

    def market(table: str) -> pd.DataFrame:
        df = pd.concat([pd.read_parquet(f) for f in sorted((DATA / "market" / table).glob("*.parquet"))])
        df["date"] = pd.to_datetime(df["date"])
        return df

    for name, table in (("SPX OHLCV", "spx_ohlcv"), ("VIX daily", "vix_daily"),
                        ("VIX term structure", "vix_term_structure"), ("T-bill rate", "tbill_rates"),
                        ("Treasury curve and SOFR", "rates_daily")):
        line = row(name)
        a, b = span(line)
        df = market(table)
        df = df[df["date"] <= b]
        assert str(df["date"].min().date()) == a, (name, a)
        assert f"| {len(df):,}" in line, (name, len(df), line)
        if table == "vix_term_structure":
            assert f"{len(df):,} ({int((df['source'] == 'yfinance').sum())} Yahoo rows)" in line
        if table == "rates_daily":
            assert f"{len(df):,} ({df['series'].nunique()} series)" in line
    opts = {s: pd.read_parquet(DATA / "options" / f"spx_{s}.parquet") for s in C23_SESSIONS}
    assert f"| {sum(len(o) for o in opts.values()):,} |" in row("SPX options")
    vix_sessions = C23_SESSIONS[:4]
    n_vix = sum(len(pd.read_parquet(DATA / "options" / f"vix_{s}.parquet")) for s in vix_sessions)
    assert f"| {n_vix:,} |" in row("VIX options")
    caps = sorted((DATA / "options" / "cboe").glob("*.json"),
                  key=lambda q: json.loads(q.read_text())["fetched_at"])
    first = {}
    for q in caps:
        rec = json.loads(q.read_text())
        first.setdefault(rec["underlying"], rec)
    spx, vix = first["SPX"], first["VIX"]
    assert (f"from {spx['session']} | {spx['rows']:,} SPX + {vix['rows']:,} VIX lines "
            "(first capture)") in row("SPX and VIX option books (C23)")
    with sqlite3.connect(f"file:{DATA / 'vol_system.db'}?mode=ro", uri=True) as con:
        a, b = span(row("VX futures"))
        cnt, lo, k, e0, e1 = con.execute(
            "SELECT COUNT(*), MIN(date), COUNT(DISTINCT expiry_month), MIN(expiry_month), MAX(expiry_month) "
            "FROM vix_futures_daily WHERE date <= ?", (b,)).fetchone()
        assert lo == a
        assert f"({k} monthly contracts expiring {e0} → {e1}) | {a} → {b} | {cnt:,} |" in row("VX futures")
        a, b = span(row("VX final settlements"))
        cnt, lo = con.execute("SELECT COUNT(*), MIN(settlement_date) FROM vix_futures_settlements "
                              "WHERE settlement_date <= ?", (b,)).fetchone()
        assert lo == a
        assert f"| {cnt:,} |" in row("VX final settlements")
    man = json.loads((DATA / "benchmarks" / "manifest.json").read_text())["files"]
    a, b = span(row("Cboe strategy indices"))
    assert {f["first_date"] for f in man.values()} == {a}
    assert {f["last_date"] for f in man.values()} == {b}
    assert all(name in row("Cboe strategy indices") for name in man)


def test_known_limitations_figures():
    from joint_vol_calibration.backtest import run_backtest as rb
    from joint_vol_calibration.backtest.backtest_engine import ATM_TO_VIX_RATIO

    lim = README.split("## Known limitations")[1].split("## References")[0]
    assert f"at one β ({ATM_TO_VIX_RATIO:.2f}) estimated from 2026 chains" in lim
    beta = rb.estimate_beta(include_live=False).set_index("session")["ratio"]
    assert f"The first later chain ({C23_HEADLINE}) gives {beta[C23_HEADLINE]:.2f}" in lim
    bench = json.loads((DATA / "backtest" / "inference" / "benchmarks.json").read_text())
    bench = bench["implied_beta"]["PUT"]
    full = [v for y, v in bench["by_year"].items() if y not in bench["partial_years"]]
    assert f"the Cboe PUT index implies {rng(full, '.2f')} by full year" in lim
    with open(DATA / "backtest" / "full_results.pkl", "rb") as fh:
        m = pickle.load(fh)["metrics"]
    assert (f"the portfolio loses {abs(m['cumulative_return']) * 100:.1f}% while T-bills earn "
            f"{m['cash_cum_return'] * 100:.1f}%") in lim
    vx = [v for s in RESULT_SESSIONS if (v := _cal(s)["leg_losses"].get("vx_futures_rmse_oos")) is not None]
    assert f"misses real VX futures by {rng(vx, '.1f')} pts out of sample" in lim
    # σ on its cap and the boxed fits' VVIX, over the sessions the results name
    from joint_vol_calibration.config import HESTON_BOUNDS

    rows = {r["session"]: r for r in json.loads(
        (DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]}
    diag = [rows[s] for s in RESULT_SESSIONS]
    n, w = len(diag), WORDS.get(len(diag), len(diag))
    capped = sum("sigma" in (_cal(s).get("pinned_params") or []) for s in RESULT_SESSIONS)
    assert f"σ sits on its {HESTON_BOUNDS['sigma'][1]:.1f} cap in {capped} of the {n} fits to {LAST}" in lim
    above = all(r["vvix_model_cap2"] > r["vvix_market"] for r in diag)
    assert above                                       # "on all six"
    model = rng((r["vvix_model_cap2"] for r in diag), ".0f")
    market = rng((r["vvix_market"] for r in diag), ".0f")
    assert f"the boxed fits price VVIX at {model} against a market ^VVIX of {market} on all {w}" in lim
    q = sorted((DATA / "calibrations").glob("quintic_cal_*.pkl"))
    with open(q[-1], "rb") as fh:
        assert not pickle.load(fh)["success"]            # "The Quintic OU fit ... did not converge"
    # quote quality: every chain the results name is a Yahoo download, each with executable static arbitrage
    idx = {r["session"]: r for r in json.loads((DATA / "surface" / "index.json").read_text())["sessions"]}
    for s in RESULT_SESSIONS:
        surf = json.loads((DATA / "surface" / s / "surface.json").read_text())
        assert surf["inputs"]["source"] == "yfinance", s
        assert idx[s]["executable_arbitrage"] is True, s
    assert (f"The {w} chains the surfaces are built from are Yahoo end-of-day downloads without a quote "
            "time; static arbitrage that is executable at the quoted bid and ask is present on "
            f"all {w}") in lim


def test_regeneration_command_reproduces_the_committed_fits():
    """CLM-07: the README's command for joint_cal_<session>.pkl was a new TRF
    fit on the Treasury curve, not the committed DE fit on ^IRX. It is now the
    refit make reproduce runs: the recorded inputs and the optimiser that made it
    (DE before C23, multi-start TRF from 2026-10-26)."""
    line = next(x for x in README_RAW.splitlines() if x.startswith("| `data_store/calibrations/joint_cal_"))
    assert "**recorded_inputs(ref)" in line
    choose = "'de' if ref.get('optimizer', 'de') == 'de' else 'trf'"
    assert f"c.calibrate(optimizer={choose})" in line
    refit = (ROOT / "scripts" / "ci" / "reproduce.py").read_text()
    assert "**recorded_inputs(ref)" in refit
    assert f"opt = {choose}" in refit
    for s in C23_SESSIONS:
        assert _cal(s).get("optimizer", "de") == "de", s      # every fit C23 measured is DE
    assert {_cal(s).get("optimizer", "de") for s in RESULT_SESSIONS} <= {"de", "trf_multistart"}


def test_calibration_tables_are_the_pickles_and_the_diagnostics(diag):
    params = [_cal(s)["params"] for s in RESULT_SESSIONS]
    n_feller = sum(2 * q["kappa"] * q["theta"] < q["sigma"] ** 2 for q in params)
    assert n_feller == len(RESULT_SESSIONS)
    assert f"Feller is violated in all {WORDS[n_feller]} and imposed in none" in README
    assert f"The table stops at {LAST}" in README
    for s in RESULT_SESSIONS:
        cal = _cal(s)
        p, ll = cal["params"], cal["leg_losses"]
        cells = [f"{p['kappa']:.3f}", f"{p['theta']:.4f}", f"{p['sigma']:.3f}", fmt(p["rho"], ".3f"),
                 f"{p['v0']:.4f}", f"{ll['spx_iv_rmse']:.3f} / {ll['spx_oos_iv_rmse']:.3f}",
                 f"{ll['vix_index_rmse']:.3f}"]
        line = next(x for x in README_RAW.splitlines() if x.startswith((f"| {s} |", f"| **{s}** |")))
        for c in cells:
            assert c in line, (s, c, line)
        vx = ll.get("vx_futures_rmse_oos")                # the 'VX futures, OOS' column
        last_cell = line.rstrip(" |").split("|")[-1].strip().strip("*")
        assert last_cell == (f"{vx:.2f}" if vx is not None else "n/a†"), (s, last_cell)
        on_cap = "sigma" in next(r for r in diag if r["session"] == s)["on_bound"]
        starred = re.search(re.escape(f"{p['sigma']:.3f}") + r"\\?\*", line) is not None
        assert starred is on_cap, (s, "σ on the box ceiling is starred")
        r = next(x for x in diag if x["session"] == s)
        rough_line = next(x for x in README_RAW.splitlines()
                          if x.startswith(f"| {s} | {fmt(r['sigma_cap2_oos'], '.2f')} |"))
        cells = (fmt(r["sigma_cap2_oos"], ".2f"), fmt(r["joint_rough_oos"], ".2f"),
                 f"{r['joint_rough_H']:.2f}")
        assert rough_line == f"| {s} | {' | '.join(cells)} |"
    cur = next(r for r in diag if r["session"] == C23_HEADLINE)
    assert (f"gives σ {cur['sigma_free']:.2f}, κ {cur['kappa_free']:.1f}, a loss "
            f"{abs(cur['sigma_free_loss_rel']) * 100:.1f}% lower and an out-of-sample RMSE of "
            f"{cur['sigma_free_oos']:.2f} instead of {cur['sigma_cap2_oos']:.2f}") in README
    first = diag[0]
    cost = f"{abs(first['sigma_free_loss_rel']) * 100:.1f}%"
    assert f"on {first['session']} the box costs {cost} of the loss" in README
    assert (f"model's VVIX is {rng((r['vvix_model_cap2'] for r in diag), '.0f')} against a market ^VVIX of "
            f"{rng((r['vvix_market'] for r in diag), '.0f')}") in README
    cond = rng((r["rel_jtj_condition_number"] for r in diag), ".0f")
    assert (f"cond(JᵀJ) of the relative-sensitivity Jacobian is {cond} over the "
            f"{WORDS[len(diag)]} sessions") in README
    kse = rng((r["relative_standard_errors"]["kappa"] * 100 for r in diag), ".0f")
    assert f"standard error is {kse}% of κ" in README
    c23 = [r for r in diag if r["session"] in C23_SESSIONS]   # the DE optima C23 timed TRF against
    worst = max(abs(r["trf_loss_rel_diff"]) for r in c23)
    assert all(r["trf_loss_rel_diff"] <= 0 for r in c23)             # "slightly below them"
    assert f"reaches the {WORDS[len(c23)]} stored differential-evolution optima" in README
    assert f"relative loss of at most {worst:.1e}".replace("e-0", "e-") in README
    trf, de = rng((r["trf_seconds"] for r in c23), ".1f"), rng((r["de_serial_seconds"] for r in c23), ".1f")
    assert f"in {trf} s, against {de} s" in README
    assert all(r["trf_basins"] == 1 for r in diag)        # "one basin on every session"
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in diag)
    assert f"error on {lower} of {len(diag)} sessions and returns H = ½ (Heston) on the other two" in README
    idx = {r["session"]: r for r in json.loads((DATA / "surface" / "index.json").read_text())["sessions"]}
    gaps = [abs(idx[s]["vix_replicated"] - idx[s]["vix_official"]) for s in RESULT_SESSIONS]
    assert f"within {rng(gaps, '.2f')} index points of the official close" in README


def test_figure_is_drawn_from_the_surface_report():
    mrf = _script("make_readme_figure")
    side = json.loads((ROOT / "docs" / "figures" / f"smile_{FIGURE}.json").read_text())
    src = ROOT / side["source"]
    assert side["source_sha256"] == hashlib.sha256(src.read_bytes()).hexdigest()
    res = pd.read_parquet(src)
    want = mrf.figure_numbers(res, FIGURE, mrf.pick_expiries(res))
    assert {k: side[k] for k in want} == want
    png = (ROOT / "docs" / "figures" / f"smile_{FIGURE}.png").read_bytes()
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    assert f"![SPX smiles on {FIGURE}:" in FIRST_SCREEN       # the caption names its session
    assert f"](docs/figures/smile_{FIGURE}.png)" in FIRST_SCREEN


def test_corrections_doc_is_the_render_of_the_registry():
    rc = _script("render_corrections")
    doc = (ROOT / "docs" / "corrections.md").read_text()
    assert rc.splice(doc, rc.render()) == doc, "run python scripts/render_corrections.py"
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    for r in reg["rounds"]:
        assert f"### {r['id']}" in doc
        for c in r["commits"]:
            assert f"/commit/{c})" in doc


def test_validation_pack_findings_log_is_the_render_of_the_registry():
    """CLM-14: the pack's findings log named docs/corrections.md as its source
    yet disagreed with it (C17's severity, C20's dates, four C23 findings and
    six commits missing). It is now rendered from the same registry."""
    rc = _script("render_corrections")
    pack = (ROOT / "docs" / "model_validation" / "README.md").read_text()
    assert rc.LOG_BEGIN in pack, "the pack's findings log is not the generated block"
    rendered = rc.splice(pack, rc.render_findings_log(), rc.LOG_BEGIN)
    assert rendered == pack, "run python scripts/render_corrections.py"
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    log = pack[pack.index(rc.LOG_BEGIN):]
    doc = (ROOT / "docs" / "corrections.md").read_text()
    for r in reg["rounds"]:
        ids = [f["id"] for f in r["findings"] if f.get("id")] or [r["id"]]
        for i in ids:
            assert f"| {i} |" in log, i
            if i != r["id"]:
                assert f"[{i}]" in doc, i                  # the same ID in the postmortem
        for c in r["commits"]:
            assert f"`{c[:7]}`" in log, (r["id"], c)       # every commit of the round is in the log
        for f in r["findings"]:
            assert set(f.get("commits") or []) <= set(r["commits"]), (r["id"], f.get("id"))
    for line in log.splitlines():
        for sha in re.findall(r"`([0-9a-f]{7})`", line):
            assert f"/commit/{sha}" in doc, sha            # every commit in the log is in corrections.md


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
    # the notes are block quotes: drop the '>' that starts each line before joining
    art = " ".join(re.sub(r"(?m)^>[ ]?", "", (ROOT / "substack_article.md").read_text()).split())
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
    # CLM-11: the note says every number in it is checked by the tests; these are the rest
    spa = inference["multiple_testing"]["spa_published"]["p_consistent"]
    assert f"Hansen SPA p {spa:.2f}" in note
    s3 = inference["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
    assert (f"Against {s3['n_paths']:,} exposure-matched circular shifts, S3's entry timing has p "
            f"{s3['p_value']:.3f} ({s3['p_holm_6_sleeves']:.2f} after a Holm correction over the six "
            "series)") in note
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    c22 = next(pt for pt in reg["headline_series"]["points"] if pt["round"] == "C22")
    assert f"against {fmt(c22['cum_ret_pct'], '.1f')}% and −0.88 in the C22 note below" in note
    assert "Sharpe −1.80 → −0.88" in next(r for r in reg["rounds"] if r["id"] == "C22")["effect"]
    fam = json.loads((DATA / "market" / "corrections" / "c23_vix_index_family.json").read_text())
    vvix = fam["tables"]["vix_term_structure"]["by_ticker"]["^VVIX"]["added_from_cboe"]
    assert f"{vvix} VVIX sessions missing" in note
    rows = {r["session"]: r for r in json.loads(
        (DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]}
    diag = [rows[s] for s in C23_SESSIONS]
    sc = {s: {m["model"]: m for m in json.loads((DATA / "surface" / s / "scorecard.json").read_text())
              ["models"]} for s in C23_SESSIONS}
    hes = rng((sc[s]["heston_v2"]["metrics"]["out_of_sample"]["median_abs_hs"] for s in C23_SESSIONS), ".1f")
    svi = rng((sc[s]["svi_cal"]["metrics"]["out_of_sample"]["median_abs_hs"] for s in C23_SESSIONS), ".2f")
    assert f"misses by a median {hes} half-spreads" in note
    assert f"arbitrage-free SVI surface on the same sessions misses by {svi}" in note
    cur = rows[C23_HEADLINE]
    assert (f"cuts its out-of-sample error from {cur['sigma_cap2_oos']:.2f} to {cur['sigma_free_oos']:.2f} "
            f"vol pts, at σ = {cur['sigma_free']:.1f}") in note
    lower = sum(r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01 for r in diag)
    same = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)
    assert (f"lower on {lower} of {len(diag)} sessions and identical to Heston on the other {same}") in note
    art_full = " ".join((ROOT / "substack_article.md").read_text().split())
    worst = max(abs(r["trf_loss_rel_diff"]) for r in diag)
    assert f"whose loss it matches to within {worst:.1e} relative".replace("e-0", "e-") in art_full


def test_citation_is_this_version():
    """CLM-12: CITATION.cff cited 0.22.0 'released' 2026-10-19 while its
    abstract described C23 work committed after that date. The cited version is
    the package's, released no earlier than the last correction round it
    describes, and the title is the README's."""
    cff = (ROOT / "CITATION.cff").read_text()
    version = re.search(r"^version: (\S+)$", cff, re.MULTILINE).group(1)
    released = re.search(r"^date-released: (\d{4}-\d{2}-\d{2})$", cff, re.MULTILINE).group(1)
    toml = (ROOT / "pyproject.toml").read_text()
    pyproject = re.search(r'^version = "([^"]+)"$', toml, re.MULTILINE).group(1)
    assert version == pyproject
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    last = max(r.get("date_end") or r["date"] for r in reg["rounds"])
    assert released >= last, (released, last)
    title = re.search(r'^title: "([^"]+)"$', cff, re.MULTILINE).group(1)
    assert README_RAW.splitlines()[0] == f"# {title}"
    assert "no release tag" in " ".join(cff.split())       # nothing claims a tag or DOI that does not exist
    assert "doi" not in cff.lower()


def _pack_row(model: str, outcome: str) -> str:
    """The validation pack's §3 outcome row of ``model`` whose outcome cell starts with ``outcome``."""
    rows = [x for x in PACK_RAW.splitlines() if x.startswith(f"| {model} | {outcome}")]
    assert len(rows) == 1, (model, outcome, rows)
    return rows[0]


def test_validation_pack_counts_and_outcomes_are_the_artifacts():
    """VALIDATOR-10: the validation pack was frozen at five sessions ('σ on its
    2.0 box ceiling in 2 of 5', 'rough lower on 3 of 5', '32–80% inside', 'cond
    150–293', 'no entry committed yet', '5 of the 13 checks breach … 7.46 vol pts')
    and disagreed with the site. It now states the span of its sessions; every
    count, range and §3 outcome row it touches is recomputed here from data_store
    over the sessions in that span, and it copies no monitoring status."""
    from joint_vol_calibration.config import HESTON_BOUNDS

    sessions = _result_sessions(PACK)
    n, w, last = len(sessions), WORDS.get(len(sessions), len(sessions)), sessions[-1]
    cals = {s: _cal(s) for s in sessions}
    rows = {r["session"]: r for r in json.loads(
        (DATA / "calibrations" / "diagnostics" / "summary.json").read_text())["rows"]}
    diag = [rows[s] for s in sessions]
    sc = {s: {m["model"]: m for m in json.loads((DATA / "surface" / s / "scorecard.json").read_text())
              ["models"]} for s in sessions}
    assert f"the {w} sessions with a stored chain and an accepted fit, {sessions[0]} → {last}" in PACK
    # §1 limitations
    capped = [s for s, c in cals.items() if "sigma" in (c.get("pinned_params") or [])]
    cap = HESTON_BOUNDS["sigma"][1]
    params = [c["params"] for c in cals.values()]
    feller = sum(2 * q["kappa"] * q["theta"] < q["sigma"] ** 2 for q in params)
    assert (f"σ on its {cap:.1f} box ceiling in {len(capped)} of the {n} fits to {last} "
            f"({', '.join(capped)}); Feller violated in all {feller}") in PACK
    assert feller == n
    half = sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)
    assert f"½ on {half} of the {n} sessions to {last}" in PACK
    ledgers = sorted(p.stem.removeprefix("ledger_") for p in (DATA / "forward").glob("ledger_*.json"))
    if ledgers:
        assert f"first entry {ledgers[0]}, not yet evidence" in PACK
        assert "no entry" not in PACK
        assert f"first entry {ledgers[0]} (`data_store/forward/`)" in _pack_row("M18", "Forward test")
    # §2: identifiability over the sessions, the common held-out counts, the replication's departures
    cond = rng((r["rel_jtj_condition_number"] for r in diag), ".0f")
    assert f"cond(JᵀJ) of the relative-sensitivity Jacobian is {cond} over the {w} sessions to {last}" in PACK
    se = {k: rng((r["relative_standard_errors"][k] * 100 for r in diag), ".0f")
          for k in ("kappa", "theta", "rho", "v0")}
    assert f"κ {se['kappa']}%, θ {se['theta']}%, ρ {se['rho']}%, v₀ {se['v0']}%" in PACK
    assert f"Gauss–Newton errors of {se['kappa']}% on κ" in PACK
    common = " / ".join(f"{sc[s]['svi_cal']['metrics']['common_out_of_sample']['n']:,}" for s in sessions)
    assert f"{common} quotes, {sessions[0]} → {last}" in PACK
    vr = json.loads((DATA / "surface" / last / "vix_replication.json").read_text())
    assert (f"on {last}, {len(vr.get('deviations') or [])} deviations from Cboe's rules and "
            f"{len(vr.get('gap_notes') or [])} notes on the gap") in PACK
    assert "(Cboe rules)" not in PACK
    assert "with Cboe's rules" not in PACK
    # §3 outcome rows
    oos = " / ".join(fmt(c["leg_losses"]["spx_oos_iv_rmse"], ".2f") for c in cals.values())
    dates = " / ".join([sessions[0], *(s[5:] for s in sessions[1:])])
    assert f"| {oos} vp for {dates} (" in _pack_row("M4", "Out-of-sample SPX IV RMSE")
    assert f", {w} v2 sessions |" in _pack_row("M4", "Out-of-sample SPX IV RMSE")
    hes = [sc[s]["heston_v2"]["metrics"]["out_of_sample"] for s in sessions]
    assert (f"{rng((h['median_abs_hs'] for h in hes), '.1f')} half-spreads; "
            f"{rng((h['inside_spread_pct'] for h in hes), '.1f')}% inside, {w} sessions"
            ) in _pack_row("M4", "Error in")
    svi = {s: sc[s]["svi_cal"]["metrics"]["out_of_sample"] for s in sessions}
    row = _pack_row("M14", "The certified")
    assert (f"median {rng((o['median_abs_hs'] for o in svi.values()), '.2f')} half-spreads; "
            f"{rng((o['inside_spread_pct'] for o in svi.values()), '.0f')}% inside the bid/ask, {w} sessions "
            f"({last}: {fmt(svi[last]['median_abs_hs'], '.2f')}, "
            f"{fmt(svi[last]['inside_spread_pct'], '.1f')}%)") in row
    arb = [json.loads((DATA / "surface" / s / "surface.json").read_text())["arbitrage"] for s in sessions]
    assert all(a["executable_arbitrage_certificate"]["executable_arbitrage"] for a in arb)
    n_ex = rng((a["n_quotes_in_executable_violation"] for a in arb), ",d")
    assert (f"executable at the bid/ask on all {n} sessions; {n_ex} quotes per session"
            in _pack_row("M14", "Static arbitrage"))
    idx = {r["session"]: r for r in json.loads((DATA / "surface" / "index.json").read_text())["sessions"]}
    rep_ = " / ".join(fmt(idx[s]["vix_replicated"], ".2f") for s in sessions)
    off = " / ".join(fmt(idx[s]["vix_official"], ".2f") for s in sessions)
    assert f"| {rep_} against {off} (" in _pack_row("M14", "VIX replicated")
    lower = [r for r in diag if r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01]
    pairs = ", ".join(f"{fmt(r['joint_rough_oos'], '.2f')} vs {fmt(r['sigma_cap2_oos'], '.2f')}"
                      for r in lower)
    assert (f"rough lower on {len(lower)} of {n} sessions ({pairs} vp); H = ½, identical, on {half}"
            in _pack_row("M4, M15", "Rough Heston"))
    # §4 copies no monitoring status the daily job changes
    mon = PACK.split("## 4. Ongoing monitoring")[1].split("## 5.")[0]
    assert not re.search(r"\b\d+ of the \d+ checks breach", mon)


def test_ui_plan_marks_its_withdrawn_acceptance_text_as_history():
    """HANDOFF D: docs/roadmap/ui_plan.json still read 'the tile shows 95.0' as
    P2.5's acceptance after R3-T4 withdrew that number (a single bad Yahoo quote,
    not executable arbitrage). The plan is history: a superseded acceptance line
    is marked, its item says why in a history_note, and the withdrawal it cites
    is in the corrections register."""
    plan = json.loads((ROOT / "docs" / "roadmap" / "ui_plan.json").read_text())
    items = {i["id"]: i for pk in plan["packages"] for i in pk["items"]}
    p25 = items["P2.5"]
    line = next(x for x in p25["acceptance"].splitlines() if "95.0" in x)
    assert "[Superseded, see history_note]" in line
    assert "3da6928" in p25["history_note"]
    for item in items.values():
        if "[Superseded" in item.get("acceptance", ""):
            assert item.get("history_note"), item["id"]
    reg = json.loads((ROOT / "dashboard" / "corrections.json").read_text())
    shas = [c for r in reg["rounds"] for c in r["commits"]]
    assert any(c.startswith("3da6928") for c in shas)
