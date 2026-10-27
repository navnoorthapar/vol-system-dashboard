# Handoff: keep going until the system is fund-grade

This is a living brief. Any Claude session with access to this repository (any account, cloud
only) continues from here. Read it first, then `docs/roadmap/README.md`.

**Read next:** `docs/roadmap/FUND_GRADE_PLAN.md`, the strategy: an honest assessment of what a
fund reviewer holds against the project today and the tiered plan (integrity, desk research
substance, the joint SPX/VIX problem, engineering, desk UX) that closes the gap. This file is
the operational brief; that one says what to build and why.

**The mandate.** Keep improving this system until a top volatility fund (US$1B+ AUM) would
trust it as a desk tool and as evidence of its author's judgement. That bar is concrete; see
"The fund-grade bar". Work in loops: critique, fix, verify, ship, redeploy, update this file.
Repeat. Reaching the bar starts the research roadmap below; it does not end the work.

**Hard constraints.**
- Free data and free infrastructure only (GitHub Actions, GitHub Pages, public Cboe/FRED/Treasury/Yahoo data).
- Honesty over flattery. The backtest result is negative (portfolio Sharpe −0.99, 95% CI [−1.81, −0.30]) and stays published as such.
- Never invent a number, a result or a capability. Every claim on the site traces to a committed artifact and a test.

## Session continuity (do this every session)

- **Start:**
  - Read this file.
  - Run `git log --oneline origin/main | head -60`.
  - Skip every item whose ID is already in a commit subject.
  - Check the latest `daily_refresh.yml` run and the open PRs.
- **During:** commit and push each verified item as soon as it passes. Never leave work uncommitted.
- **Before the session or its context runs out:**
  - Update the "Status log" at the end of this file: what you finished (commit ids), what is half done, what you learned.
  - Commit it and ship it.
  - The next session starts from that log.

## Where things stand (2026-10-27, end of the first build)

- **main** has:
  - C23 (PR #4: arbitrage-free surface, identifiable joint fit, inference, hardened bot);
  - UI round 2 (PR #5);
  - round 3 shipped in batches (PRs #11, #12 and the ones after them).
- **Live site:** https://navnoorbawa.me, from the gh-pages branch. The daily bot redeploys it.
- **Daily bot** (`.github/workflows/daily_refresh.yml`):
  - close slot 22:07 UTC Mon–Fri; book slots 02:30 and 04:30 UTC Tue–Sat;
  - also runs on workflow_dispatch.
  - On 2026-10-27 its first run on the new pipeline accepted the 2026-10-26 fit and deployed it.
  - The forward ledger got its first entry (`data_store/forward/ledger_2026-10-26.json`).
- **Open PRs:**
  - **#10:** the round-3 working branch. Close it once its commits are on main; check with `git log origin/main..origin/<branch>`.
  - **Dependabot #6–#9:** unreviewed. Merge one only if CI is green and the bump changes no numerics that tests pin.
- **Critic scores before round 3** (main at bbde2e1):
  - trust lens: Overview 7, Surface 7, Calibration 8, Risk 8, Research 8, Corrections 8, Model Risk 7;
  - design lens: Overview 7, Surface 7, Calibration 7, Risk 8, Research 7, Corrections 7, Model Risk 8.
  - Round 3 fixed the high findings (T4+D1, T1) and most mediums. Re-score before trusting these numbers.

## Not finished because of the time limit (do these first, in this order)

Every commit subject carries its item ID, for example `(R3-D4)` or `A1-F16:`. Skip what is already on main.

### A. Round-3 critic findings (medium)
All five (R3-D3, R3-D2, T3, T2, R3-D6) shipped on 2026-10-27; see the status log. Nothing open here.

### B. Round-2 backlog still open
- **C3 (F20):** In the Surface tail-probability table, ∫p and E[S_T/F] are 1.0000 in every row.
  - Replace both with one 'distribution check' column: max(|∫p−1|, |E[S_T/F]−1|) in scientific notation.
- **C1 (F11):** The Overview prints 'rv₂₀ 9.75%' next to 'RV21 9.67%'.
  - Add an rv20 glossary entry (dashboard/desk_glossary.py, taken from the code that computes it).
  - Label each number '20-session' or '21-session'.
- **C2 (F17):**
  - The quote-grade surface row of the freshness table shows '—' for its age. Compute it in NYSE sessions like the other rows.
  - The status strip's SURFACE date shows its age.
- **C4 (F19):** At 390 and 320 px, captions inside horizontal scrollers carry the scroll shadow, with ~50 px before the header. Move the caption outside the shadow and keep the gap at 12 px or less.
- **D2 (page weight):** surface.html is ~704 KB and research.html ~407 KB, over its old 400 KB budget.
  - Cut inline chart JSON without changing any displayed value: round to display precision plus one digit, dedupe, drop unused fields.
  - Targets: surface ≤ 450 KB, research ≤ 400 KB. Update the page-budget test.
- **A3 (no-JS):** With JS off, surface.html is wider than the viewport.
  - Every page at 320, 390 and 1440 px with JS off must have scrollWidth ≤ innerWidth.
  - Add that check to scripts/ci/a11y/a11y.mjs.
- **A2 (tab stops):** Calibration has ~105 tab stops; the target is 60 or fewer.
  - Use roving tabindex per legend group, tab list and small-multiple group, like the cone tabs.
  - The test must not depend on the expiry count.
- **C5 (F10):** At 1440 the Overview Regime panel has a 124 px gap. The layout choice is yours; the interact gate fails any blank over 96 px.
- **D1 (performance budget):** Add scripts/ci/perf_budget.mjs and scripts/ci/perf_budget.json.
  - Per page at 1440 and 390, measure HTML bytes, DOM nodes, time to first chart (also at 4× CPU), long tasks and CLS.
  - Set budgets at the measured value plus 10%.
  - Add it as a step in the accessibility job of tests.yml.
- **D3 (expiry switch):** Click-to-chart is ~41 ms, and 151–198 ms at 4× CPU. Profile it; the target is under 100 ms at 4×.

### C. Round-3 critic findings still open (low)
- **T11:** Model Risk V1 history drops the sign, because app.py stores abs(gap). Store the signed gap and apply abs only in the limit check.
- **T10:** calibration.html calls the VIX-index leg 'the model's expected average vol'. It is 100·√(expected average variance).
- **R3-D7:** The delta grid and the #heat heatmap share the cividis ramp on different domains. Use one shared, clamped domain.
- **R3-D12:** The cone table's SESSIONS column prints 9.0, 9.7 … while the caption says 'rounded'. Print the integer window used.
- **R3-D11:**
  - At 320 px KPI ranges break mid-value. Add white-space:nowrap on .kpi-range.
  - At 390 px an odd last KPI tile spans the row.
- **R3-D8:** Reader-facing text uses code identifiers and ASCII math ('svi.butterfly_certificate', '>=', '1e-4', 'w_(i+1) - w_i'). Rewrite it in the house style.
- **R3-D10:** At 1440 the ticker wraps to two rows. Fit it on one.
- **R3-D9:** In the Research β table, the uppercase transform turns β into 'Β'. Wrap it in `<span class="sym">`.

### D. Loose ends the round-3 engineers recorded
- **Surface 'Model descriptions'** still say the Heston fit was 'DE + L-BFGS-B', but the 2026-10-26 fit used multi-start TRF.
  - The text is hard-coded in joint_vol_calibration/surface/scorecard.py:257, so it is stored in each scorecard.json.
  - Build it from the fit's `methods.optimizer`, with the same logic as `_optimizer_settings` in app.py.
- **The static-arbitrage totals** (198 quotes, 6.8%) still include outlier rows, because surface.json stores leg detail only for the 25 largest violations.
  - Store the outlier flag for every violating row in the surface report, so the share can exclude them.
- **Two CSV buttons gave no download** within 3 s in a test script: Surface's static-arbitrage constraint-rows table, and Research's 21-day realised-vol forecast table. Investigate both.
- **At 1440 px** the s4 table's 'At bid/ask' column runs past its panel, and the residuals heatmap beside it stretches to ~1,349 px while the disclosure is closed.
- **calibration.html (~line 345)** still prints a raw 'E[VIX_T]' in a note. Use the |subs filter.
- **docs/roadmap/ui_plan.json** holds stale P2.5 acceptance text ('tile shows 95.0'). It is history; update or annotate it.
- **`make reproduce`** (rebuilds the backtest, Greeks, C7 and the v2 calibrations, then diffs against what is committed) was not re-run after round 3. Run it.
- **docs/model_validation/README.md line ~171** still says M14 replicates the VIX '(Cboe rules)'; the site now says 'the Cboe formula, with its listed departures' (T3). Align the doc, and note the departures count is 12 on 2026-10-26 (6 deviations + 6 gap notes), not 8.
- **The Calibration page's census sentence** ('k of k fits pass') was not reworded with the Overview's (T2). Reword it the same way.
- **B1/B2 (cone and frontier labels)** were fixed in the chart specs with spec-level tests; no pixel-level browser measurement of label rectangles at 390–1920 px was added. Add one to the interact gate.

### E. Owner decisions (ask the owner; do not decide alone)
- The licence of `data_store/benchmarks/*.csv` (Cboe strategy-index data).
- Git LFS or a data branch for the raw captures (~0.4–0.6 GB a year in git).
- Tagging v0.23.0 and minting a Zenodo DOI.

## The fund-grade bar (the stop condition for the UI loop)

All of these must hold at once:
1. **Critic scores:** four independent critics give every page a median ≥ 9/10, with no high or medium finding open. Each critic is a separate agent that has not seen the others' output:
   - a senior vol quant who recomputes headline numbers from data_store;
   - a head of desk-tools design who measures in px and contrast ratios;
   - an SR 11-7 model validator who traces every claim to an artifact;
   - the CIO of a US$1B+ volatility fund who asks: would I trust this tool, and would I hire its author? This lens judges insight, rigour, honesty and signal-to-noise.
2. **Every gate green on main:**
   - the full suite;
   - `simulate_next_session.py --full`;
   - the a11y + interact gate;
   - the performance budgets (once D1 exists);
   - `make reproduce` matching the committed artifacts.
3. **The bot deploys 10 consecutive NYSE sessions** without a manual fix.
4. **A recompute audit:** a test (or a script run in CI) recomputes every number the Overview and Surface pages print from data_store by an independent path, and matches.

## After the bar: research roadmap (candidates, not commitments)

Before building any of these, have the critics confirm it adds value. Each one goes through the same rules and gates, with honest results.
- **Local vol:**
  - Dupire local vol from the certified SVI-cal surface.
  - Reprice the surface with it (PDE or MC) and show the repricing error in half-spreads.
- **Smile dynamics:**
  - Estimate sticky-strike versus sticky-delta behaviour (the skew-stickiness ratio) from consecutive sessions as the bot accrues them.
  - Show the estimate with its uncertainty.
- **The joint SPX/VIX smile problem:**
  - Fit or score the VIX option smile beside the SPX smile with the existing Heston, rough Heston and Quintic OU fits.
  - Report the joint-fit error honestly.
- **Risk:**
  - A vega ladder bucketed by tenor and delta.
  - Realised hedge slippage against the model's Greeks.
- **Forward track record:** show the forward (pre-registered, no look-ahead) ledger as a live out-of-sample record as entries accrue. Any new signal goes through the trials ledger and DSR before it appears.
- **Operations:**
  - The bot opens a GitHub issue after two consecutive failed runs.
  - A weekly `make reproduce` in CI.
  - Model cards for every model in `dashboard/model_inventory.json`.
- **Desk UX:**
  - Permalinks for every control state (the cone already writes the URL hash).
  - PNG export per chart.
  - A dark theme (desk.css has none).
  - These must keep the CSP, axe and overflow gates green.
- **Items skipped in round 2 as low value** (B14, Q4-rerun, Q8-NAV, Q15, Q19-bands, D10-axes): reconsider only if a critic asks for them with evidence.

## Rules (each was broken once; CI enforces most)

- Every behaviour change gets a test that fails on the old code and passes on the new.
- **No test may pin a value the daily bot replaces:** dates, fitted numbers, session counts, row counts. Derive expected values from the data or the module.
  - `scripts/ci/simulate_next_session.py --full` fails any test that does.
  - The presentation gate failed on the real 2026-10-26 fit for this reason: a 2.30497 value pre-rounded to 2.305 printed 2.31, and a test wrote '−' as '-'.
- Browser tests never call `page.wait_for_function` with a string, because the CSP blocks eval. Use `_wait_for_chart` or evaluate a function.
- The CSP stays `script-src 'self'`: no inline scripts or handlers.
- axe must report 0 violations, with no horizontal overflow at 320–1920 px, including the 0.06em letter-spacing stress at 320.
- Every number on a page comes from the data, never from a template or test literal.
- The ruff (1295) and mypy (368) baselines may only shrink: `make lint typecheck`.
- desk.js stays within its gzip budget (31 KB; raise it by at most 1 KB, with a reason).
- Git:
  - No WIP or unverified commits.
  - Never force-push or rewrite history.
  - Never push to main directly.
  - Never commit anything under `data_store/`: the bot owns it. If a test or a freeze changes `data_store/vol_system.db`, restore it with `git checkout`.
- Commit messages say what changed, why, and which test proves it. Put the item ID in the subject.

## Verify

```
python -m pytest -q -p no:cacheprovider joint_vol_calibration/tests/test_dashboard.py   # presentation gate, ~5 min
make lint typecheck
python dashboard/freeze_site.py /tmp/site                                              # static site
cd scripts/ci/a11y && npm ci --no-audit --no-fund && \
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node a11y.mjs /tmp/site && \
  PLAYWRIGHT_BROWSERS_PATH=/opt/pw-browsers node interact.mjs /tmp/site               # must exit 0
make test                                                                              # full suite, ~12 min (CI runs it too)
python scripts/ci/simulate_next_session.py --full                                      # ~20 min (CI runs it too)
```

## Ship

1. Push to the session's branch.
2. Open a PR to main.
3. When CI is green on the exact head, merge it with a merge commit, passing `expectedHeadSha`. The owner has given permission for this.
4. tests.yml cancels a running CI when the same branch gets a new push. If commits arrive faster than CI (~21 min), push a snapshot of the branch to a separate ship branch, open the PR from that, and merge it when green.
5. After a merge, run `daily_refresh.yml` (workflow_dispatch, steps `market,capture,calibrate,freeze`) to redeploy the site. Otherwise the next scheduled slot does it.
6. If a bot run fails its presentation gate on new data:
   - find the test or bug that pins a value the bot replaced;
   - fix it on a branch and ship it;
   - re-dispatch. The data it captured is committed before the gates, so nothing is lost.

## Status log (newest first; every session appends one entry)

- **2026-10-27, 09:05 UTC, first build session, final entry.**
  - **Shipped to main today** (PRs #11, #12, #13 and the final batch): the deploy unblock on the
    bot's 2026-10-26 fit (fit tiles round once); A1/F16; B3/F13; R3-T4+D1; R3-T1; R3-T5; R3-T8;
    R3-T7; R3-T9; R3-D5; R3-D4; R3-D3; R3-D2; T3; T2; R3-D6; B4/F14; B1+B2/F12+F15; this file and
    FUND_GRADE_PLAN.md. R3-T6 was checked and found no longer to apply (the 2026-10-26 headline
    fit is a TRF fit, matching M4's text); nothing changed for it.
  - **Site:** navnoorbawa.me redeployed three times today from main; the bot's first run on the
    new pipeline accepted the 2026-10-26 fit. The forward ledger has its first entry.
  - **Still open:** everything in sections B (C3, C1, C2, C4, D2, A3, A2, C5, D1, D3), C (all
    low findings), D (loose ends) and E (owner decisions); then the fund-grade bar and
    FUND_GRADE_PLAN.md tiers 0–4.
  - **Next step for the next session:** FUND_GRADE_PLAN.md tier 0 (the recompute audit first),
    interleaved with section B above. Re-run the four-critic review after tier 0.

