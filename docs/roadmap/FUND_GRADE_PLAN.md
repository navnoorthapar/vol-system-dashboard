# From research prototype to fund-grade: an honest assessment and the plan

Written 2026-10-27 at the end of the first build. `HANDOFF.md` is the operational brief (open
items, rules, how to ship). This file is the strategy: what the project is, what a serious
reviewer at a US$1B+ volatility fund would hold against it today, and the ordered plan that
closes the gap. Future sessions: read HANDOFF.md first, then this, then pick the highest tier
with open work. Update the "Decisions and progress" section at the end as things land.

## 1. What this project is (the claim a reviewer should take away)

A joint SPX/VIX volatility research and model-validation system that treats its own results
as hypotheses to be falsified. It fits Heston, rough Heston and Quintic OU jointly to the SPX
smile and the VIX complex; builds an arbitrage-free SVI surface scored in half-spreads on
held-out quotes; replicates the VIX from the chain by the Cboe formula and attributes the gap;
backtests a short-vol book marked to model and then subjects the result to a full inference
stack (stationary bootstrap, PSR/DSR against a trials ledger, Hansen SPA, StepM, CSCV PBO,
QLIKE/MCS). The headline result is negative (portfolio Sharpe −0.99, 95% CI [−1.81, −0.30]) and
is published as such, with the reasons. A daily bot captures Cboe books, refits, re-scores,
gates itself and redeploys a 7-page desk site, all on free infrastructure.

The strongest honest claim: **this is how a model-validation function would want a desk's
vol research to be built**: every number traceable, every claim gated, every negative kept.
That is also what a fund reviewer is really assessing: the author's judgement, not the P&L.

## 2. How a fund reviewer reads it, and what breaks the read today

A reviewer gives it ten minutes: the Overview, one deep page (Surface), the Research
conclusions, then the code. What they will hold against it, stated plainly:

1. **Thin history.** Six quote-grade sessions (2026-03-24 → 2026-10-26, with a four-month gap)
   and Cboe timestamped books only since 2026-10-20. Nothing about surface *dynamics* can be
   claimed yet; the bot must run unbroken for months. The 10-session deployment streak in
   HANDOFF's bar is the first milestone, not the last.
2. **Quote provenance.** The five earlier chains are Yahoo end-of-day snapshots with no quote
   time; executable static arbitrage is present in all five (160–629 quotes per session), and a
   single bad print (the P6265 case) reached a headline tile until round 3. A desk will not
   trust a surface whose inputs it cannot timestamp. Cboe books fix this going forward; the
   history must be labelled by source everywhere it is shown.
3. **The headline model is the wrong tool and the site half-admits it.** Heston needs σ at
   its box cap and a model VVIX far above the market to fit SPX and the VIX indices together;
   SVI-cal beats it by 4–15× in half-spreads; VX futures miss by 1.9–3.2 pts. The honest
   framing exists (Calibration, Model Risk) but the *narrative* still leads with Heston.
   Lead with the surface and the replication; present Heston as the joint-model benchmark it is.
4. **The joint SPX/VIX problem is posed but not solved.** VIX options are not in any fit (too
   few sessions of VIX option quotes); the Quintic OU fit did not converge. This is the one
   research question in the repo that a vol fund actually cares about, and it is open.
5. **No independent validation.** One developer built and validated everything
   (docs/model_validation §5). An SR 11-7 reviewer will ask for an independent recompute.
6. **Engineering debt visible from the outside.** dashboard/app.py is ~7,000 lines; the ruff
   baseline carries 1,295 findings and mypy 368; raw captures grow git by 2–3 MB per accepted
   session; the site's Surface page is ~700 KB. None of this is disqualifying, all of it is
   noticed.
7. **The backtest is a proxy study.** All P&L is mark-to-model at one β (0.82) held constant
   over 2018–2025; the first later chain gives 0.70. The site says so, but the Research page
   still spends most of its length on the sleeves rather than on what the proxy can and cannot
   support.

None of these are fatal. All of them are addressable with free data and the existing bot.

## 3. The plan, in tiers

Each item: why a fund cares, how, and the acceptance criterion that proves it. Effort is in
sessions of agent work. Do tiers in order; within a tier, in order. Nothing ships without the
rules in HANDOFF.md (a test that fails first, no pinned bot values, every gate green).

### Tier 0: integrity (make every number provably right)

- **0.1 Recompute audit in CI.** A script (`scripts/ci/recompute_audit.py`) recomputes every
  number printed on the Overview and Surface from `data_store` by an independent code path
  (not the dashboard's functions) and diffs against the frozen HTML. Why: this is what an
  independent validator would do first. Acceptance: the script runs in `tests.yml`, prints a
  table of every tile and its recomputed value, and fails on any mismatch beyond display
  rounding. Effort: 1 session.
- **0.2 Source labelling of every quote-grade artifact.** Each surface session carries its
  source (Yahoo snapshot / Cboe timestamped book), quote time, and two-sided share; every page
  that shows a session-level number shows the badge. Why: weakness 2. Acceptance: no page
  shows a number from a Yahoo-sourced session without the badge; a test enumerates them.
  Effort: 0.5 session.
- **0.3 Weekly `make reproduce` in CI, on a schedule, with a visible result.** It exists as a
  skipped job on PRs; make the Monday run mandatory and surface "last reproduced: <date>, 14/14
  artifacts bit-for-bit" on the Model Risk page from a committed JSON. Acceptance: the tile
  reads from `data_store/reproduce_status.json` written by the job; a failed reproduce opens a
  GitHub issue. Effort: 0.5 session.
- **0.4 Bot resilience and streak.** The bot opens a GitHub issue after two consecutive
  failed runs and the site shows "N consecutive sessions deployed". Acceptance: the streak
  counter is derived from gh-pages history, and the issue path is tested with a forced failure
  in `simulate_next_session.py`. Effort: 0.5 session.
- **0.5 Close the VIX replication departures that can be closed.** Of the listed departures,
  two are fixable with free data: use the Treasury CMT-style interpolation at each term's
  exact maturity (not one 91-day rate for both), and record the Cboe book's capture time as
  the calculation time. Keep listing the rest (no quote time on Yahoo history, PM-settled
  stand-ins) honestly. Acceptance: `vix_replication.json` lists fewer departures; the gap
  attribution test still passes on every session; the gap on Cboe-book sessions does not grow.
  Effort: 1 session.

### Tier 1: what a vol desk expects every morning (the research substance)

- **1.1 Surface change attribution, session over session.** Decompose today's SVI-cal
  surface change against the prior session into level, slope (skew), curvature and term
  components per expiry and at constant maturity (30/60/90/180 d), with the residual. Show it
  first on the Overview: "what moved since yesterday". Why: this is the single most-used
  object on a vol desk and the site has no "change" view at all. Acceptance: a pure function
  in `dashboard/desk_change.py` with tests on synthetic surfaces (a pure level shift
  attributes 100% to level, etc.); the Overview's first panel is the change table once ≥ 2
  consecutive Cboe-book sessions exist. Effort: 1.5 sessions.
- **1.2 Skew dynamics: stickiness and the term structure of skew.** As consecutive sessions
  accrue, estimate the skew-stickiness ratio (regression of ATM-vol change on spot return,
  per tenor) with a stationary-bootstrap CI, and show the term structure of 25Δ skew with its
  power-law fit over time. Gate it on n ≥ 20 sessions; before that, show the panel with
  "collecting: 7 of 20 sessions". Why: sticky-strike vs sticky-delta is how desks hedge.
  Acceptance: estimator + CI tested on simulated paths with known stickiness; the panel never
  prints an estimate below the sample gate. Effort: 1.5 sessions.
- **1.3 Variance-swap curve and VIX basis, daily.** From the certified surface, the model-free
  variance-swap strike at 30/60/90/180/360 d (log-contract replication with the extrapolated
  wings flagged), the VIX futures basis to the replicated VIX term structure, and the implied
  roll-down. Why: the VIX complex is the desk's second screen; parts exist (VS strip, VX leg)
  but not as one curve with a daily record. Acceptance: a `data_store/surface/<session>/vs_curve.json`
  written by the bot, a page panel, and the VS strikes agree with the existing strip to 1e-6
  where they overlap. Effort: 1 session.
- **1.4 VIX options smile beside the SPX smile.** Capture VIX option quotes from the Cboe book
  daily (the book capture exists; store the VIX chain too), fit SVI to the VIX smile per expiry
  with the same arbitrage certificates, and show it next to the SPX smile with its ATMF and
  skew. No joint model yet; just the market object, scored and certified. Why: weakness 4
  starts here; a fund expects the VIX smile to be a first-class object. Acceptance: a
  `vix_surface.json` per session with the same scorecard schema; the Model Risk inventory gets
  the entry. Effort: 2 sessions.
- **1.5 Forward ledger as the live out-of-sample record.** The pre-registered forward ledger
  (`data_store/forward/`) got its first entry on 2026-10-26. Show it as a live track record:
  entries, mark-to-market at executable prices, running Sharpe with CI, and the pre-registered
  hypothesis each entry tests, with no edits allowed after the fact (a hash chain over entries,
  tested). Why: a negative backtest plus a clean, growing, pre-registered forward record is
  the most credible thing a self-taught researcher can show a fund. Acceptance: the ledger
  page renders from the files; the hash chain test fails if any past entry changes. Effort:
  1 session.
- **1.6 Realised-vs-implied, properly.** The implied/realised panel exists (RV21, VIX−RV21,
  β·VIX premium). Add the variance-risk-premium term structure (model-free VS strike minus
  subsequent realised variance at matched horizons, where history allows), with the caveat
  block. Acceptance: computed from `data_store` only; matched horizons tested against a
  hand calculation. Effort: 0.5 session.

### Tier 2: the joint SPX/VIX problem, done to a standard

This is the academic headline (Guyon's "joint calibration" problem). The repo already has the
pieces; the work is to pose it exactly and report it honestly.

- **2.1 One joint scorecard.** For every model in the inventory that claims the VIX
  (Heston, rough Heston, Quintic OU), one table per session: SPX smile error (half-spreads,
  held-out), VIX index error (index pts), VX futures error at settlement (pts), VIX smile
  error (half-spreads, once 1.4 exists), fit time, parameters at bounds. Acceptance: one
  `joint_scorecard.json` per session, one page table, one test that the numbers equal the
  per-model artifacts. Effort: 1 session.
- **2.2 Quintic OU to convergence, or a documented failure.** The single Quintic OU fit did
  not converge. Either make it converge (multi-start, analytic gradients where possible, a
  tighter box with the reason) on ≥ 3 sessions, or document precisely why it cannot on this
  data with the loss landscape shown. Both are publishable. Acceptance: `converged: true` on
  ≥ 3 sessions with the joint scorecard filled, or a Corrections entry with the evidence.
  Effort: 2–3 sessions.
- **2.3 Honest model ranking on the Overview.** Replace any Heston-first narrative with the
  joint scorecard's ranking, in words generated from the data ("SVI-cal prices the SPX smile
  within 0.55 half-spreads; no joint model gets within 4; rough Heston beats Heston on 3 of 6
  sessions"). Acceptance: the sentence is a tested template over `joint_scorecard.json`.
  Effort: 0.5 session.

### Tier 3: engineering to a validator's standard

- **3.1 Split `dashboard/app.py`.** One module per page (`desk_overview.py`, `desk_surface_page.py`,
  …) plus `desk_common.py`; `app.py` becomes routing only. No behaviour change: the frozen HTML
  before and after is byte-identical (that is the test). Effort: 1–2 sessions.
- **3.2 Lint and type debt to zero on the core.** Drive the ruff baseline from 1,295 to 0 and
  mypy to 0 for `joint_vol_calibration/` first, then the dashboard; the gates then forbid any
  finding rather than any growth. Do it in mechanical, reviewable commits (one rule family per
  commit). Effort: 2 sessions.
- **3.3 Model cards.** One `docs/model_validation/cards/M<n>.md` per inventory entry: purpose,
  inputs, assumptions, known failure modes, tests that cover it, last validated date, owner.
  Generated skeletons from `model_inventory.json`, filled by hand, checked by a test that every
  inventory id has a card. Effort: 1 session.
- **3.4 Independent review packet.** A single `docs/model_validation/REVIEW_PACKET.md` that an
  outside validator can follow in two hours: how to reproduce, what to recompute, where the
  gates are, the open findings. Then actually get one outside review (a peer, a forum, an
  academic) and log it in §5 of the validation doc. Effort: 0.5 session plus the owner's time.
- **3.5 Raw captures out of the git history path.** Decide (owner decision) between Git LFS
  and a data branch; implement the chosen one so the bot keeps working and `make reproduce`
  still finds every capture. Effort: 1 session after the decision.
- **3.6 Performance budgets in CI** (HANDOFF D1) and the Surface page under 450 KB (D2).

### Tier 4: desk UX

- **4.1 The first screen is "what changed".** After 1.1 exists, the Overview opens with the
  change table, the VIX complex, and the gate status; everything else below the fold.
- **4.2 Permalinks for every control state** (the cone already writes the URL hash); PNG export
  per chart; a dark theme. Each keeps the CSP, axe and overflow gates green.
- **4.3 Keyboard and no-JS parity** (HANDOFF A2, A3).

## 4. What not to do

- Do not add a signal to escape the negative result. Any new strategy goes through the trials
  ledger, is pre-registered in the forward ledger, and is reported with DSR against the full
  trial count. A fund will respect a clean negative far more than a dressed-up positive.
- Do not add a model without a scorecard entry, a Model Risk inventory entry and a card.
- Do not widen the site. Every new panel must displace or fold something; the 10-second read
  gets shorter, not longer.
- Do not let a gate go yellow to ship. The bot's value is that it has never deployed a page it
  could not verify.

## 5. Measuring progress

- The fund-grade bar is defined in HANDOFF.md (four-critic median ≥ 9 on every page, every
  gate green including weekly reproduce, 10 consecutive bot deployments, recompute audit).
- Re-run the four-critic review after each tier, not after each item; record the medians here.
- Keep a one-line entry per shipped item in "Decisions and progress" below with its commit.

## 6. Decisions and progress (append; newest first)

- **2026-10-27.** Plan written at the end of the first build. Tiers 0–4 all open. Critic medians
  before round 3: 7–8 on every page (trust and design lenses). Owner decisions still open:
  benchmark data licence, LFS vs data branch, v0.23.0 tag and Zenodo DOI.
