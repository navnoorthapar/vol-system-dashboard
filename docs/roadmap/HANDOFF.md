# Handoff: finishing the UI after round 3

Written 2026-10-27, when the session that built C23 and UI rounds 1–3 ended. A new
session (any account with access to this repository) continues from here. Read this
file first, then `docs/roadmap/README.md`.

## Where things stand

- **main** has:
  - C23 (PR #4), UI round 2 (PR #5) and round 3 batches 1–3 (PRs #11, #12 and the ones after them).
  - The live site, https://navnoorbawa.me, deployed from main by the daily bot (gh-pages).
- **The daily bot** (`.github/workflows/daily_refresh.yml`) runs on GitHub Actions by itself:
  - close slot 22:07 UTC Mon–Fri;
  - book slots 02:30 and 04:30 UTC Tue–Sat.
  - It needs no session. On 2026-10-27 it accepted the 2026-10-26 fit and redeployed the site.
- **Open PRs:**
  - **#10:** the round-3 working branch, superseded by the batch PRs. Close it once its commits are on main; check with `git log origin/main..origin/<branch>`.
  - **Dependabot #6–#9:** unreviewed. Merge one only if CI is green and the bump does not change numerics that tests pin.
- **Round-3 critic scores** (main at bbde2e1, before the round-3 fixes):
  - trust lens: Overview 7, Surface 7, Calibration 8, Risk 8, Research 8, Corrections 8, Model Risk 7;
  - design lens: Overview 7, Surface 7, Calibration 7, Risk 8, Research 7, Corrections 7, Model Risk 8.
  - Goal: every page ≥ 9 from three independent critics, with no high or medium findings left.

## What is left, in priority order

Every item's commit subject carries its ID, for example `(R3-D4)` or `A1-F16:`. Run
`git log --oneline origin/main | grep -E 'R3-|F1[0-9]|F20|A[123]-|B[1-4]|C[1-5]|D[123]-'`
and skip any item already there.

### Round-3 critic findings (medium)
- **R3-D3:** Research #monthly heatmap. The Year column uses a ±8.1% colour scale, but the only legend shows ±2.7%. Give it its own legend or a shared scale.
- **R3-D2:** Calibration and Surface monitoring chips show the model ID, not the check. One panel reads 'MONITORED · M4 · BREACH' beside 'MONITORED · M4 · WATCH'.
  - Add a short label to each Model Risk check dict in app.py: 'short-end RMSE', 'OOS RMSE', 'VIX RMSE', 'box bound', 'fit age', 'Feller'.
  - The chip macro then prints 'M4 short-end RMSE · BREACH'.
- **T3:** The Overview (app.py, findings 'Works') and Model Risk V1 say the VIX was replicated 'by the Cboe rules'. Surface lists 8 departures.
  - Change the text to 'by the Cboe formula, with N listed departures (Surface)', where N = len(deviations + gap_notes) from vix_replication.json.
  - Use the same wording in the V1 note.
- **T2:** The Overview line '5 of 5 current-schema fits pass' is true by construction, because the job stores only gate-passing fits.
  - Reword it to '5 current-schema fits stored (the job stores only gate-passing fits; rejections are counted in the run log)', or cite the gate's regression tests.
  - The count comes from the data.
- **R3-D6:** On every smile chart (Surface #smile and the Calibration small multiples), the upper axis's '10' tick label sits on the residual panel's '+10' label.
  - The cause is desk.js quoteOption using gap = 10 px.
  - Set gap = 20, or showMaxLabel:false on the residual y axis.

### Backlog from the round-2 critique
- **B4 (F14):** Replace the yy-mm-dd format ('26-10-29 · 13d') with ISO 'YYYY-MM-DD' everywhere: the Surface heatmap rows, the |z|-by-expiry axis, the OOS-error bars, the VX-futures axis and the constant-maturity brackets.
  - Keep the 320 px overflow gate and the 0.06em stress green.
  - The test asserts that no page text or chart spec contains yy-mm-dd.
- **B1/B2 (F12, F15):**
  - Volatility cone end labels under ~12 px apart must not touch: dodge them in the spec (dashboard/desk_cone.py).
  - The Calibration σ-profile frontier labels only the headline session's caps; other sessions show theirs in the tooltip.
- **C3 (F20):** The Surface tail-probability table's ∫p and E[S_T/F] columns print 1.0000 in every row.
  - Replace both with one 'distribution check' column: max(|∫p−1|, |E[S_T/F]−1|) in scientific notation.
- **C1 (F11):** The Overview prints 'rv₂₀ 9.75%' next to 'RV21 9.67%'.
  - Add an rv20 glossary entry (dashboard/desk_glossary.py, taken from the code that computes it).
  - Label each number '20-session' or '21-session'.
- **C2 (F17):**
  - The quote-grade surface row of the freshness table shows '—' for its age. Compute it in NYSE sessions like the other rows.
  - The status strip's SURFACE date shows its age.
- **C4 (F19):** At 390 and 320 px, table captions inside horizontal scrollers carry the scroll shadow, with about 50 px before the header. The caption sits outside the shadow, and the gap is at most 12 px.
- **D2 (page weight):** surface.html is ~704 KB and research.html ~407 KB, which is over its old 400 KB budget.
  - Cut inline chart JSON without changing a displayed value: round to display precision plus one digit, dedupe, drop unused fields.
  - Targets: surface ≤ 450 KB, research ≤ 400 KB. Update the page-budget test.
- **A3 (no-JS):** With JS disabled, surface.html is wider than the viewport.
  - Fix it so that with JS off at 320, 390 and 1440 px every page's scrollWidth ≤ innerWidth.
  - Add that check to scripts/ci/a11y/a11y.mjs.
- **A2 (tab stops):** Calibration has ~105 tab stops; the target is 60 or fewer.
  - Use roving tabindex per legend group, tab list and small-multiple group, like the cone tabs.
  - The test must not depend on the data's expiry count.
- **C5 (F10):** At 1440 the Overview Regime panel has a 124 px gap.
  - The layout choice is yours. The interact gate fails any blank over 96 px.
- **D1 (performance budget):** Add scripts/ci/perf_budget.mjs and scripts/ci/perf_budget.json.
  - Per page at 1440 and 390, measure HTML bytes, DOM nodes, time to first chart (also at 4× CPU), long tasks and CLS.
  - Set budgets at the measured value plus 10%.
  - Add it as a step in the accessibility job of tests.yml.

### Round-3 critic findings (low)
- **T11:** Model Risk V1 history drops the sign, because app.py stores abs(gap). Store the signed gap and apply abs only in the limit comparison.
- **T10:** calibration.html calls the VIX-index leg 'the model's expected average vol'. It is 100·√(expected average variance).
- **R3-D7:** The delta grid and the #heat heatmap use the same cividis ramp on different domains. Use one shared, clamped domain.
- **R3-D12:** The cone table's SESSIONS column prints 9.0, 9.7 … while its caption says 'rounded'. Print the integer window actually used.
- **R3-D11:**
  - At 320 px KPI ranges break mid-value. Add white-space:nowrap on .kpi-range.
  - At 390 px, an odd last KPI tile spans the row.
- **R3-D8:** Reader-facing text uses code identifiers and ASCII math ('svi.butterfly_certificate', '>=', '1e-4', 'E[VIX_T]'). Use the house style.
- **R3-D10:** At 1440 the ticker wraps to two rows. Fit it on one.
- **R3-D9:** In the Research β table, the uppercase transform turns β into 'Β'. Wrap it in `<span class="sym">`.

### After the list: the critique loop (docs/roadmap/README.md section B)
1. Freeze main.
2. Have three independent critics score every page from 1 to 10 with evidence:
   - a senior vol quant who recomputes numbers from data_store;
   - a head of desk-tools design;
   - an SR 11-7 model validator.
3. Fix the high and medium findings.
4. Repeat until every page's median is ≥ 9 and no high or medium finding is left.

### Owner decisions (do not decide these alone)
- The licence of `data_store/benchmarks/*.csv`.
- Git LFS or a data branch for the raw captures (~0.4–0.6 GB a year).
- Tagging v0.23.0 and minting a Zenodo DOI.

## Rules (each one has been broken before; CI enforces most)

- Every behaviour change gets a test that fails on the old code and passes on the new.
- **No test may pin a value the daily bot replaces:** dates, fitted numbers, session counts, row counts. Derive the expected values from the data or the module.
  - `scripts/ci/simulate_next_session.py --full` (in CI) fails any test that does.
  - The presentation gate failed on the real 2026-10-26 fit for this reason: a 2.30497 value pre-rounded to 2.305 printed 2.31, and a test wrote '−' as '-'.
- Browser tests never call `page.wait_for_function` with a string, because the CSP blocks eval. Use `_wait_for_chart` or evaluate a function.
- The CSP stays `script-src 'self'`: no inline scripts or handlers.
- axe must report 0 violations, with no horizontal overflow at 320–1920 px, including the 0.06em letter-spacing stress at 320.
- Every number on a page comes from the data, never from a template or test literal.
- The ruff (1295) and mypy (368) baselines may only shrink: `make lint typecheck`.
- desk.js stays within its gzip budget (31 KB; raise it by at most 1 KB, with a reason).
- Git:
  - No WIP or unverified commits; the owner rejected those.
  - Never force-push or rewrite history.
  - Never push to main directly.
  - Never commit anything under `data_store/`: the bot owns it. If a test or a freeze changes `data_store/vol_system.db`, restore it with `git checkout`.
- Commit messages say what changed, why, and which test proves it.

## Verify (in a cloud session)

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
