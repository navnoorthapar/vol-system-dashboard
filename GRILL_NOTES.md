# GRILL_NOTES.md — Interview Prep Session Log

Maintained by the mock-interviewer sessions. Read at session start; re-test weak
points before hunting new ones.

---

## Session 2026-06-11/12 — REDTEAM + C17 fixes

### What happened
User invoked REDTEAM before answering any questions this session. Ten weaknesses
were identified and ranked; the five code-fixable ones were fixed (C17), all
artifacts regenerated, dashboard + live site updated. **The headline outcome
inverted: the C16 "+19.1%" portfolio became −9.1% under the fully corrected
pipeline.** The user has NOT yet been re-quizzed on any of this.

### C17 findings (all confirmed against code/data, now fixed or disclosed)
1. **Classifier loses to persistence baseline** — "predict yesterday's regime"
   scores 90.0% on 2020+ vs the honest 63.4% XGBoost. Labels are same-day
   observables so y(t−1) is a legitimate causal predictor. ML demoted from the
   trading loop (backtest now uses lagged rule labels). THE key new fact.
2. **The C16 "+19.1% / S1C +$640K" was a label-noise artifact** — under accurate
   rule labels: portfolio −9.07% cum, Sharpe −0.99, maxDD −19.1%; S1C −$97K
   (67 trades, 48% win); pseudo-OOS 2013-17 collapsed +$785K → −$333K.
   S1C has NO validated edge, full stop.
3. **Calibration stack never touched backtest P&L** — monthly Heston recal was
   dead code (0 calibrations ever, output never consumed); removed. Backtest is
   BS on VIX-TS ATM vol. Heston/Quintic/NN serve C6 Greeks + C7 hedge sim only.
4. **Margin was documented but unenforced** — now capped: nav/(0.20·K·100).
5. **S1C had no R2 exit** — open short straddle could ride a VVIX spike 21 days.
   Now uses _run_statemachine_r2exit.
6. **Mark-to-model P&L** — zero historical option prices in DB; disclosed.
7. **Pseudo-OOS confound** — 2013-17 was the golden short-vol era; disclosed
   (and now moot since the pseudo-OOS is negative anyway).
8. **ICJ correlation claim was unverified** — deleted from S3 docstring.
9. **Sharpe used flat 5% rf while pricing used per-date ^IRX** — now consistent
   (per-date T-bill path everywhere).
10. **Quintic VIX-futures RMSE 0.0011 is fit-by-construction** (ξ₀ bootstrapped
   from the same curve; no tradable futures in DB) — caveat now on site.

### Current canonical numbers (memorize; old ones are dead)
- Portfolio S1C+S3+S4: **−9.07% cum, Sharpe −0.99 (vs T-bill path), maxDD −19.12%**, NAV $909,329
- S1C **−$97K** (67 trades, 48% win) | S3 **+$25K** (17 trades, 88% win) | S4 **−$199K** (29 trades)
- Refs: S1 −$1.53M, S2 −$108K
- Crisis: COVID +3.9%, FedHikes +2.3%, Tariffs +3.1%; ex-COVID P&L −$122K
- Pseudo-OOS 2013-17: portfolio −19.5%; S1C −$333K; S3 +$20K; S4 −$250K
- S3 sensitivity {0.5,1,2,4}%/z → {+$5K, +$12K, +$25K, +$52K}, win 15-17/17
- Classifier: 63.4% vs **90.0% persistence baseline** → research-only
- **577 tests**. Commit 8c3ba9b. Live on navnoorbawa.me.

### The defensible identity (the only story that survives)
"Calibration + risk infrastructure, plus a fully documented negative result.
Every methodology correction made the backtest worse; the apparent edge was a
stack of artifacts (look-ahead → circular feature → label noise). S3's timing
is the one robust scrap. The audit trail IS the deliverable."
Any drift toward "profitable system" is indefensible.

### Gaps — user must be re-quizzed on (priority order)
1. **The C17 story itself** — user has not yet explained back: persistence
   baseline logic, why the +$640K vanished, what "label-noise artifact" means
   mechanically (S1C profits keyed off MISclassified R0/R1 days).
2. Why persistence (90%) doesn't itself constitute a tradeable signal (it's
   label autocorrelation, not prediction skill; the rule IS the label).
3. Carr-Madan derivation (never tested)
4. Feller condition math (asked at session start 2026-06-11 — NEVER ANSWERED;
   user invoked REDTEAM instead)
5. Characteristic function derivation (never tested)
6. Greeks/vomma — negative near-ATM vomma, 3 unstable nodes (never tested)
7. Delta-hedge P&L decomposition, hedge efficiency 0.016 vs 0.303 (never tested)
8. NN pricer (never tested) — and the 8.1× vs 100× target miss
9. Quintic OU — and why its futures fit is by-construction

### Previously verified strengths (from 2026-06-11 session)
- Can recite Heston-failure logic (κ,θ,v0 consumed by SPX smile → VIX locked)
- Can recite the five C16 weaknesses from memory — **NOW STALE: the five-item
  list has become a ten-item audit and the S1C/pseudo-OOS numbers flipped sign.
  User's memorized script is OUT OF DATE and will misfire in an interview.**

### Verdict as of this session
Would not advance yet — not because the project is weak (the C17 audit is
genuinely strong work) but because the user hasn't yet internalized the new
narrative and has answered zero technical-math questions across two sessions.
The single most dangerous failure mode: user recites the stale "+19.1% /
S1C +$640K / pseudo-OOS +$785K" script that no longer matches their own site.

### Next session plan
1. Re-test: Feller condition (still owed an answer)
2. Make user explain C17 end-to-end: baseline → demotion → P&L inversion
3. Drill: "Why publish a negative result?" — make them own it without hedging
4. Then: Carr-Madan / CF derivation block
