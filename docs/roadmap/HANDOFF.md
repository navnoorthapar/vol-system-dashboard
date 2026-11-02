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
- **C3 (F20):** In the Surface tail-probability table, ∫p and E[S_T/F] are 1.0000 in every row. **Done 2026-10-29 (PR #15).**
  - Replace both with one 'distribution check' column: max(|∫p−1|, |E[S_T/F]−1|) in scientific notation.
- **C1 (F11):** The Overview prints 'rv₂₀ 9.75%' next to 'RV21 9.67%'. **Done 2026-10-29 (PR #15).**
  - Add an rv20 glossary entry (dashboard/desk_glossary.py, taken from the code that computes it).
  - Label each number '20-session' or '21-session'.
- **C2 (F17):** **Done 2026-10-29 (PR #15).**
  - The quote-grade surface row of the freshness table shows '—' for its age. Compute it in NYSE sessions like the other rows.
  - The status strip's SURFACE date shows its age.
- **C4 (F19):** At 390 and 320 px, captions inside horizontal scrollers carry the scroll shadow, with ~50 px before the header. Move the caption outside the shadow and keep the gap at 12 px or less.
- **D2 (page weight):** surface.html is ~704 KB and research.html ~407 KB, over its old 400 KB budget.
  - Cut inline chart JSON without changing any displayed value: round to display precision plus one digit, dedupe, drop unused fields.
  - Targets: surface ≤ 450 KB, research ≤ 400 KB. Update the page-budget test.
  - **Measured on w3/surface (C0–C7, 2026-10-31), not yet on main.** A full Cboe book (45 expiries) had made surface.html 1.67 MB.
    - surface.html, rebuilt Cboe store 2026-10-26 (45 expiries): 1,670,137 → 441,347 B raw, 336,416 → 78,066 B gzipped (gzip level 9).
    - surface.html, committed store (17 expiries): 777,649 → 290,854 B raw, 158,573 → 57,954 B gzipped (level 9).
    - **Correction (w3/surface fixes, 2026-10-31):** those two figures held only for the 6 earlier sessions stored then. #surface-change built a table per earlier stored session and the ATM / RR25 term charts drew them all, about 11 kB a session at 45 expiries. On w1/integrate (8 earlier sessions) the 45-expiry page (2026-10-28) was 469,272 B, over its 450,000 B budget, and the 17-expiry page 309,677 B, over the 300,000 target. Now at most four sessions are compared (the newest earlier one and the newest at least 5, 21 and 63 NYSE sessions back), so only the "Every session" table still grows (about 300 B a stored session; see E): with the strip fixes, 421,799 B raw, 78,123 / 75,912 B gzipped at level 6 / 9 (45 expiries, 2026-10-28, 3 of 8 earlier sessions compared) and 267,068 B, 59,455 / 58,493 B (17 expiries, 2026-10-29). The session data file is 524,621 B, 208,955 B at level 6 (45 expiries).
    - research.html 409,472 → 352,817 B; calibration.html 345,279 → 252,687 B (17-expiry store).
    - How: the smile's every-expiry quotes, the heatmap cells' readout numbers and the 3D points moved to one hashed session data file (`data/surface-<session>-<sha256[:16]>.json`, 527,509 B / 209,006 B gzipped at 45 expiries), fetched once after the smile's first draw; compact spec JSON without the fields the page renders as HTML; colour classes; one bucket table. No displayed number changed.
    - Budgets: pytest on synthetic 17/45/80-expiry sessions (≤ 7,000 B an expiry, ≤ 300,000 B at 17; research ≤ 400,000 B) and on synthetic stores with N and N + 1 earlier sessions (no growth past the four compared); CI's a11y job runs `scripts/ci/page_budget.py`: 450,000 B at 45 expiries, + 7,000 B an expiry past and − 5,500 B an expiry short (296,000 B at 17), n read from the page; it prints the gzipped size at level 6, ungated. The committed-fixture freeze test holds surface.html to the same formula (it was a flat 600,000 B). The bot's deploy is not gated on it.
- **A3 (no-JS):** With JS off, surface.html is wider than the viewport. **Done 2026-10-29 (PR #15).**
  - Every page at 320, 390 and 1440 px with JS off must have scrollWidth ≤ innerWidth.
  - Add that check to scripts/ci/a11y/a11y.mjs.
  - **w3/surface (2026-10-31):** no page overflows with JS off at 320/390/1440 px on the 17- and 45-expiry builds. Without JS the smile's per-expiry bucket rows are one table, a row group per expiry, in a labelled, focusable region bounded to 60% of the screen; a11y.mjs checks it (one row group per tab).
- **A2 (tab stops):** Calibration has ~105 tab stops; the target is 60 or fewer.
  - Use roving tabindex per legend group, tab list and small-multiple group, like the cone tabs.
  - The test must not depend on the expiry count.
- **C5 (F10):** At 1440 the Overview Regime panel has a 124 px gap. The layout choice is yours; the interact gate fails any blank over 96 px.
- **D1 (performance budget):** Add scripts/ci/perf_budget.mjs and scripts/ci/perf_budget.json.
  - Per page at 1440 and 390, measure HTML bytes, DOM nodes, time to first chart (also at 4× CPU), long tasks and CLS.
  - Set budgets at the measured value plus 10%.
  - Add it as a step in the accessibility job of tests.yml.
- **D3 (expiry switch):** Click-to-chart is ~41 ms, and 151–198 ms at 4× CPU. Profile it; the target is under 100 ms at 4×.
  - **Measured on w3/surface (2026-10-31), 45 expiries.** A switch now writes 6 DOM nodes in its click task (about 160 before), draws at most 16 series (10 here; 17 before, the models toggled off among them), 0 legend keys and 0 chart resizes, and decodes an expiry only when shown (all 45 were decoded at load).
    - Switch / click-to-chart: 16 / 41 ms at 1× CPU, 48 / 190 ms at 4× (load average 1.0–1.7 on 4 shared CPUs; the gate's limits are 50 / 100 and 150 / 300 ms). At 17 expiries: 16 / 45 and 48 / 196 ms.
    - The < 100 ms at 4× target is not met: ECharts `setOption` alone is 70–300 ms at 4×. Restating D3 is an owner decision; the 4× timings still need three consecutive CI runs.
    - **Expiry strip (w3/surface fixes, 2026-10-31):** the one-line strip showed 5–7 of 17 tabs (5–11 of 45) with no sign of the rest, and a vertical wheel did not scroll it. It now shows two full lines and scrolls a line at a time (the wheel scrolls it natively); the status line starts "Expiry k of N"; the first tab of each year past the session's carries it ("Jan 15 ’27 · 79d"); the days' "d" is desk.css's, read "days". Tabs in view at 1024 / 1280 / 1440 / 1920 px: 17 / 10 / 11 / 13 of 17 and 22 / 10 / 12 / 14 of 45. interact.mjs checks the two full lines, the wheel, the first, middle and last expiry in view with their status line (kept two lines tall: at 1440 px it ran to one or two lines by expiry), and that one expiry is decoded after load and one more per switch.
    - After the fixes (2026-10-31, load average 1.0–2.2 on 4 shared CPUs): switch / click-to-chart 16 / 41 ms at 1×, 40 / 151 ms at 4× (45 expiries); 16 / 39 and 32 / 133 ms (17). Surface tab stops 226 at 1440 px and 221 at 390 px (45 expiries), 228 / 222 (17).

### C. Round-3 critic findings still open (low)
- **T11:** Model Risk V1 history drops the sign, because app.py stores abs(gap). Store the signed gap and apply abs only in the limit check. **Done 2026-10-29 (PR #15).**
- **T10:** calibration.html calls the VIX-index leg 'the model's expected average vol'. It is 100·√(expected average variance). **Done 2026-10-29 (PR #15).**
- **R3-D7:** The delta grid and the #heat heatmap share the cividis ramp on different domains. Use one shared, clamped domain.
- **R3-D12:** The cone table's SESSIONS column prints 9.0, 9.7 … while the caption says 'rounded'. Print the integer window used. **Done 2026-10-29 (PR #15).**
- **R3-D11:** **Done 2026-10-29 (PR #15).**
  - At 320 px KPI ranges break mid-value. Add white-space:nowrap on .kpi-range.
  - At 390 px an odd last KPI tile spans the row.
- **R3-D8:** Reader-facing text uses code identifiers and ASCII math ('svi.butterfly_certificate', '>=', '1e-4', 'w_(i+1) - w_i'). Rewrite it in the house style.
- **R3-D10:** At 1440 the ticker wraps to two rows. Fit it on one.
- **R3-D9:** In the Research β table, the uppercase transform turns β into 'Β'. Wrap it in `<span class="sym">`. **Done 2026-10-29 (PR #15).**

### D. Loose ends the round-3 engineers recorded
- **Surface 'Model descriptions'** still say the Heston fit was 'DE + L-BFGS-B', but the 2026-10-26 fit used multi-start TRF.
  - The text is hard-coded in joint_vol_calibration/surface/scorecard.py:257, so it is stored in each scorecard.json.
  - Build it from the fit's `methods.optimizer`, with the same logic as `_optimizer_settings` in app.py.
- **The static-arbitrage totals** (198 quotes, 6.8%) still include outlier rows, because surface.json stores leg detail only for the 25 largest violations.
  - Store the outlier flag for every violating row in the surface report, so the share can exclude them.
- **Two CSV buttons gave no download** within 3 s in a test script: Surface's static-arbitrage constraint-rows table, and Research's 21-day realised-vol forecast table. Investigate both.
- **At 1440 px** the s4 table's 'At bid/ask' column runs past its panel, and the residuals heatmap beside it stretches to ~1,349 px while the disclosure is closed.
- **calibration.html (~line 345)** still prints a raw 'E[VIX_T]' in a note. Use the |subs filter. **Done 2026-10-29 (PR #15).**
- **docs/roadmap/ui_plan.json** holds stale P2.5 acceptance text ('tile shows 95.0'). It is history; update or annotate it.
- **`make reproduce`** (rebuilds the backtest, Greeks, C7 and the v2 calibrations, then diffs against what is committed) was not re-run after round 3. Run it.
- **docs/model_validation/README.md line ~171** still says M14 replicates the VIX '(Cboe rules)'; the site now says 'the Cboe formula, with its listed departures' (T3). Align the doc, and note the departures count is 12 on 2026-10-26 (6 deviations + 6 gap notes), not 8.
- **The Calibration page's census sentence** ('k of k fits pass') was not reworded with the Overview's (T2). Reword it the same way. **Done 2026-10-29 (PR #15).**
- **B1/B2 (cone and frontier labels)** were fixed in the chart specs with spec-level tests; no pixel-level browser measurement of label rectangles at 390–1920 px was added. Add one to the interact gate.

### E. Owner decisions (ask the owner; do not decide alone)
- The licence of `data_store/benchmarks/*.csv` (Cboe strategy-index data).
- Git LFS or a data branch for the raw captures (~0.4–0.6 GB a year in git).
- Tagging v0.23.0 and minting a Zenodo DOI.
- **Surface page growth (D2, from the w3/surface fixes).** The page compares at most four earlier sessions (the newest, and the newest at least 5, 21 and 63 NYSE sessions back). Six horizons (adding about half a year and a year) would cost about 11 kB each at 45 expiries and need about 456 kB: raise the 450 kB budget, or keep four. The "Every session the surface report covers" table still grows about 300 B a stored session; at 45 expiries the page has about 28 kB of headroom, so with a fourth 45-expiry session compared and a default expiry of ~380 quotes CI's "Surface page weight" step can fail after roughly 25–90 more daily sessions. Cap that table at recent sessions (the full record stays in `data_store/surface/index.json`), or let the budget count its rows.
- **The 80 kB gzip figure (D2).** `page_budget.py` prints the gzipped size at level 6 and does not gate on it (the server picks the level; the 45-expiry page is about 78 kB at level 6, 76 kB at level 9). Gate it, at which level, or drop the figure.
- **AM / PM tags (D3).** The tabs and status line show AM / PM only from the residual table's OCC root, which no stored session carries; near-dated third Fridays are PM (SPXW) lines and from April 2027 AM. Have the producer write `root`, or tag from the time of day in T.

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
- The ruff (1293) and mypy (368) baselines may only shrink: `make lint typecheck`.
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

- **2026-11-01, fund-grade session 1, continued: wave 1 shipping; 0.8 and the tier-0 checks building.**
  - **Shipped to main since the last entry:** PR #18. It unblocked the 2026-10-29 deploy:
    - no Surface β from a chain the backtest excludes;
    - a true minus on Research;
    - the simulator's Friday book state.

    Run 145 deployed 2026-10-29. The live pages pass the recompute audit, 12,145 of 12,145.
  - **Wave 1, in one PR from `w1/integrate`:**
    - w1/ops, w1/surface-source, w1/truth-in-text and w1/vx-complex;
    - w2/vixrates (0.5) and w2/readme (the results snapshot);
    - w3/surface with its review fixes;
    - integration fixes, each with a test that fails on the old code:
      - Calibration smiles without quotes say so and keep their height;
      - the T-origin test rebuilds fits on the committed record, not the stale cache;
      - a fit with no recorded VX score may have a leg once its session's settlements are stored;
      - the Surface page budget grows with the default expiry's quotes and the stored sessions;
      - the Surface note's false "keeps one size" is dropped;
      - freeze refuses a heatmap entry without its detail.
  - **Verified on the integration branch:**
    - 17-expiry build: audit 7,689 of 7,689; 45-expiry build: audit 16,004 of 16,004;
    - a11y 0 and interact 0 on both builds;
    - page budget ok;
    - `test_dashboard.py` 353 passed, everything else 2,494 passed;
    - the next-session simulation (a Monday book).
  - **In progress:**
    - w2/closebook (0.8): its review found that the deploy gate fails when a session has both a closing and an overnight book, plus two majors; a fix round is running.
    - w2/gates (0.2's enumerating badge test, 0.4's forced-failure check): not started.
  - **After that:** the four-critic review, with its medians in FUND_GRADE_PLAN §6; extend the recompute audit to Calibration, Research, Risk and Model Risk; then tier 1.
  - **Learned:**
    - Test against the committed record, never the tracked SQLite cache. The bot never commits the cache, so a test on it silently tests an older store.
    - A page budget must allow for what grows with the data (here the default expiry's quotes and the stored sessions), or it fails with no markup change.
    - The next-session simulation runs whichever weekday follows the newest session; a Friday and a Monday exercise different book states.

- **2026-10-31, w3/surface review fixes (on w1/integrate, not pushed).**
  - Commits 4a8d085 (at most four compared sessions), fd250ee (ruff gate), a0366c5 (budget sloped below 45; gzip printed at level 6), f779b51 (heatmap readout hang; freeze checks heat entries), da37873 (audit: schema and session per reader, typed readout numbers), e3b72ec (two-line expiry strip, "Expiry k of N", year marks, "days"), 6fe8e3e (heat grids: hidden tab, focus, scroll padding).
  - Acceptance on the 17- and 45-expiry builds: axe 0, interact 0 failures, audit 7,685 / 15,997 checks with 0 mismatches, page budget ok (267,068 / 421,799 B).
  - Open: the owner decisions added to E (page growth from the "Every session" table, the gzip figure, AM / PM tags) and D3's 4× target.
- **2026-10-29, fund-grade session 1 (tier 0): batch 1 shipped; wave 1 integrated; wave 2 and the Surface rework in progress.**
  - **Shipped to main, bot green, live site on 2026-10-28 and audit-clean:**
    - **PR #15 (fund-grade batch 1).**
      - Tier 0.1: the independent recompute audit. It covers 100% of the numbers on Overview and Surface, about 11,900 checks, and gates tests.yml and the bot's deploy.
      - The site renders from the committed record (`app.record_db`), and chain provenance is stated truthfully.
      - `commit_and_push.sh` no longer exits 0 after a failed re-derive.
      - The 2026-10-27/28 robustness fixes:
        - the σ panel shows the newest diagnosed fit while the headline's diagnostics are pending;
        - an unreplicated VIX says why;
        - an expiry with no fit-set quote is left out of the surface (with its reason) instead of failing the session;
        - half-spreads are packed at 10⁻⁸ vol pts.
      - Items T11, T10, R3-D9, R3-D12, R3-D11, C1, C2, C3 and A3, and the census wording.
    - **PR #16 (owner):** the rough-Heston tail mask that ended the diagnostics timeouts. This branch's own fix for the same bug was dropped in favour of it.
    - **PR #17:** the pages named H = ½ (Heston) a "0.02 floor"; they now name the side of the bound. Two deploy-gate tests compared rounded k and whole-page digits.
    - **Checks:** run 139 succeeded and deployed. The recompute audit against the live pages passes 11,933 of 11,933.
  - **Integrated on `w1/integrate`, not yet shipped:**
    - w1/ops: tier 0.3, 0.4, 0.6, 0.9, 0.10, the 0.11 growth gate, and the private Cboe store.
    - w1/surface-source: tier 0.7 and the producer half of 0.2.
    - w1/truth-in-text: inventory prose, captions, README, validation pack, Model Risk coverage, corrections round C24.
  - **What blocks shipping it:**
    - A full Cboe book has about 45 expiries; at that size the Surface page fails CI's interaction gate, and surface.html grows to 1.67 MB. The rework (plan in the session's `surface_plan.md`) is being built on `w3/surface`.
    - The README's results must move to a committed snapshot, or the backfill's rebuilds turn tests.yml red.
  - **In progress:**
    - w1/vx-complex: the VX futures refresh and its labels.
    - Wave 2: tier 0.5 (per-term rates and capture time in the VIX replication), 0.8 (a same-instant VIX reference), 0.2's enumerating badge test and 0.4's forced-failure check.
  - **Owner decisions raised (section E):**
    - Fit on the Yahoo chain and score on the Cboe book (as built), or fit on the Cboe book too.
    - Reproducing Cboe-sourced sessions will depend on the private store once raw books move there.
    - A failed private push loses that run's raw books (no public artifact as a fallback).
    - The PR growth gate has no override for files added under data_store/, and its size limit is 2 MB.
    - Page budgets: corrections.html at 150 KB; Model Risk budgeted per history dot.
    - Whether the research classifier, and the VX futures table the bot does not refresh, count as "by design" breaches.
    - D3's 100 ms at 4× CPU is out of reach while ECharts draws the chart (its `setOption` alone takes 70–300 ms at 4×); restate the target.
  - **Next step:**
    - Merge w3/surface and wave 2 into `w1/integrate`, verify it (tests, audit, gates, simulation), and ship it.
    - Extend the recompute audit to Calibration, Research, Risk and Model Risk.
    - Run the four-critic review and record the medians in FUND_GRADE_PLAN.md §6.
    - Then tier 1.

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

