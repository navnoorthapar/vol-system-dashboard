"""
dashboard/desk_inventory.py: the model inventory's prose, computed from the artifacts.

model_inventory.json typed facts the daily job changes ('σ sits on its box
ceiling in 2 of 5 fits', 'on its 2.0 cap in the headline fit', 'No entry
committed yet'), and the Model Risk page printed them beside checks that said
otherwise (VALIDATOR-04, CODE-03, CIO-04, QUANT-05, CITSEC-13). Each fact is
now a placeholder filled by one function here. These tests drive each function
both ways on synthetic inputs; test_dashboard.py checks the rendered text
against the shipped artifacts and holds the JSON's literal prose to STALE_PROSE.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_inventory as inv  # noqa: E402


@pytest.mark.parametrize("text", [
    "σ sits on its box ceiling in 2 of 5 fits",
    "Feller is violated in all 5",
    "ranks 6th of 7 on QLIKE",
    "rough is lower on two of the sessions",
    "the first later chain (2026-10-16) gives 0.70",
    "the vomma shock is sized from the fitted σ, on its 2.0 cap in the headline fit",
    "H sits on its 0.02 floor",
    "Not converged (ε on its −0.99 bound)",
    "No entry committed yet; marks are the captured book.",
    "12.5% of quotes sit inside the spread",
])
def test_stale_prose_flags_every_fact_the_data_can_change(text):
    assert inv.stale_prose(text), text


@pytest.mark.parametrize("text", [
    "Quadrature target 1e-10 of DF·F; 5e-9 of F at the slowest-decaying corners of the parameter box.",
    "Instantaneous shocks (no theta); parallel vol shifts; one reference book, not a position book.",
    "β is held constant over 2018–2025, and {beta_later}.",
    "One-factor diffusion: misses the short end; {sigma_box}; {sigma_box_cost}.",
    "Thresholds are fixed rules, not estimated; VVIX gaps are carried one session.",
    "VVIX gaps are carried one session; 5.9% of labels flip within measurement noise (C22).",
])
def test_stale_prose_passes_design_constants_and_placeholders(text):
    assert inv.stale_prose(text) == []


def test_fill_names_every_placeholder_and_refuses_an_unknown_one():
    assert inv.fill("a {x} b {y}", {"x": "1", "y": "2"}) == "a 1 b 2"
    assert inv.placeholders("a {x} b {y_2}") == ["x", "y_2"]
    with pytest.raises(KeyError):
        inv.fill("a {typo}", {"x": "1"})


def test_fill_models_drops_the_pending_note_only_once_the_output_is_written():
    models = [{"id": "M18", "outputs": "data_store/forward/", "outputs_pending": "no entry yet",
               "limitations": "{ledger_record}; marks are the captured book."}]
    vals = {"ledger_record": "1 session registered"}
    kept = inv.fill_models(models, vals, written=lambda o: False)
    assert kept[0]["outputs_pending"] == "no entry yet"
    assert kept[0]["limitations"] == "1 session registered; marks are the captured book."
    dropped = inv.fill_models(models, vals, written=lambda o: True)
    assert "outputs_pending" not in dropped[0]
    assert models[0]["limitations"].startswith("{")              # the inventory itself is not edited


def test_sigma_box_counts_the_fits_on_each_side_of_the_box():
    fits = [("2026-03-24", "ceiling"), ("2026-03-27", None), ("2026-10-16", "ceiling"), ("2026-10-26", None)]
    assert inv.sigma_box_text(fits) == ("σ sits on its box ceiling in 2 of 4 gate-passing fits "
                                        "(2026-03-24, 2026-10-16)")
    inside = inv.sigma_box_text([(s, None) for s, _ in fits])
    assert inside == "σ is inside its box in all 4 gate-passing fits"
    assert inv.sigma_box_text([("2026-10-16", "ceiling")]) == ("σ sits on its box ceiling in the only "
                                                               "gate-passing fit (2026-10-16)")
    assert "box floor in 1 of 2" in inv.sigma_box_text([("a", "floor"), ("b", None)])
    assert inv.sigma_box_text([]).startswith("no gate-passing fit")


def _row(session, loss, s1=6.9, k1=27.4, oos0=2.96, oos1=1.19):
    return {"session": session, "sigma_free": s1, "kappa_free": k1, "sigma_cap2_oos": oos0,
            "sigma_free_oos": oos1, "sigma_free_loss_rel": loss}


def test_sigma_box_cost_names_the_session_where_the_box_costs_most():
    rows = [_row("2026-03-24", -0.021, 2.21, 7.0, 1.98, 1.82), _row("2026-10-16", -0.277)]
    boxed = {"2026-10-16": {"sigma": 2.0, "kappa": 4.99}}
    txt = inv.sigma_box_cost_text(rows, boxed, 10.0, 60.0)
    assert txt.startswith("refitted with σ ≤ 10 and κ ≤ 60, the 2026-10-16 fit moves from σ 2.00, κ 5.0 to "
                          "σ 6.90, κ 27.4")
    assert "from 2.96 to 1.19 vol pts (loss −27.7%)" in txt
    assert "over 2 diagnosed sessions" in txt
    assert txt.endswith("κ follows the σ cap along a κ–σ ridge")
    # no ridge claim when κ does not move with the cap
    assert "ridge" not in inv.sigma_box_cost_text([_row("x", -0.1, 2.5, 5.2)],
                                                  {"x": {"sigma": 2.0, "kappa": 5.0}},
                                                  10.0, 60.0)
    # a box that costs nothing anywhere says so
    flat = inv.sigma_box_cost_text([_row("a", 0.0), _row("b", -1e-13)], {}, 10.0, 60.0)
    assert flat == "refitted with σ ≤ 10 and κ ≤ 60, no diagnosed fit's loss falls (2 diagnosed sessions)"
    assert inv.sigma_box_cost_text([], {}, 10.0, 60.0).startswith("no σ profile is stored")


def test_vvix_counts_the_sessions_priced_above_the_market():
    rows = [{"vvix_model_cap2": 190.7, "vvix_market": 124.1},
            {"vvix_model_cap2": 100.0, "vvix_market": 110.0},
            {"vvix_model_cap2": None, "vvix_market": 90.0}]
    assert inv.vvix_boxed_text(rows) == ("the boxed fits price VVIX above the market on 1 of 2 diagnosed "
                                         "sessions (model 100–191 against ^VVIX 110–124)")
    assert "on all 1" not in inv.vvix_boxed_text(rows[:1])
    assert inv.vvix_boxed_text(rows[:1]).startswith("the boxed fits price VVIX above the market on the only")
    assert "none of the 2" in inv.vvix_boxed_text([rows[1], rows[1]])
    assert inv.vvix_boxed_text([]) == "the boxed fits' model VVIX is not stored"


def test_vix_options_leg_weight():
    assert inv.vix_options_leg_text([0.0, 0.0, 0.0]) == ("the VIX-options leg has weight 0 in all 3 "
                                                        "gate-passing fits")
    # a fit that records no weight is not counted as weight 0
    assert inv.vix_options_leg_text([0.0, 0.0, None]) == ("the VIX-options leg has weight 0 in all 2 "
                                                         "gate-passing fits that record it")
    assert inv.vix_options_leg_text([0.0, 0.5]) == ("the VIX-options leg is weighted in 1 of 2 "
                                                   "gate-passing fits")
    assert inv.vix_options_leg_text([None]).startswith("no gate-passing fit records")


def test_greeks_sigma_names_the_cap_only_when_sigma_is_on_it():
    box = (0.001, 2.0)
    on = inv.greeks_sigma_text("2026-10-16", 2.0, box, "upper")
    assert on == ("2.00 in the 2026-10-16 fit the surface is built from, on its σ ≤ 2 cap, "
                  "so the box sizes the shock")
    off = inv.greeks_sigma_text("2026-10-26", 1.833, box, None)
    assert off == "1.83 in the 2026-10-26 fit the surface is built from, inside its box (0.001 ≤ σ ≤ 2)"
    assert "cap" not in off
    assert "on its σ ≥ 0.001 floor" in inv.greeks_sigma_text("x", 0.001, box, "lower")
    assert inv.greeks_sigma_text(None, None, box, None) == "the fit the surface is built from records no σ"


def test_beta_later_lists_every_chain_after_the_estimation_set():
    assert inv.beta_later_text([]) == "the market DB holds no chain after the estimation set"
    assert inv.beta_later_text([("2026-10-16", 0.7016)]) == (
        "the market DB's one chain after the estimation set gives 0.70 (2026-10-16)")
    assert inv.beta_later_text([("2026-10-16", 0.7016), ("2026-10-26", 0.7602), ("2026-10-27", 0.731)]) == (
        "the market DB's 3 chains after the estimation set give 0.70 (2026-10-16), 0.76 (2026-10-26) "
        "and 0.73 (2026-10-27)")


def test_pdv_rank_gives_both_losses_or_says_none_is_stored():
    rank = {"start": "2018-01-02", "end": "2025-03-24",
            "losses": [{"label": "21-day MSE", "rank": 2, "n": 7}, {"label": "QLIKE", "rank": 6, "n": 7}]}
    assert inv.pdv_rank_text(rank) == ("2nd of 7 on 21-day MSE and 6th of 7 on QLIKE among the forecasts "
                                       "compared (2018-01-02 → 2025-03-24)")
    assert inv.pdv_rank_text(None) == "its rank among the compared forecasts is not stored"


def test_classifier_against_persistence_both_ways():
    assert inv.classifier_text(74.9, 87.0) == ("Loses to the no-skill persistence baseline (74.9% against "
                                               "87.0% accuracy out of sample)")
    assert inv.classifier_text(90.0, 87.0).startswith("Beats the no-skill persistence baseline (90.0%")
    assert inv.classifier_text(None, 87.0) == "Its out-of-sample evaluation is not stored"


def test_quintic_state_from_the_fit_record():
    msgs = ["Maximum number of iterations has been exceeded.", "STOP: TOTAL NO. OF ITERATIONS REACHED LIMIT"]
    assert inv.quintic_text(True, False, msgs, [("epsilon", -0.99)]) == ("Not converged (iteration limit; "
                                                                        "ε on its −0.99 bound)")
    assert inv.quintic_text(True, True, msgs, []) == "Converged"
    assert inv.quintic_text(True, False, ["Desired error not necessarily achieved"], []) == (
        "Not converged (Desired error not necessarily achieved)")
    assert inv.quintic_text(False, False, [], []) == "No fit is stored"


def test_rough_h_floor_and_heston_sessions():
    rows = [{"session": "2026-03-24", "joint_rough_H": 0.28},
            {"session": "2026-05-29", "joint_rough_H": 0.4999999999},
            {"session": "2026-06-18", "joint_rough_H": 0.5},
            {"session": "2026-10-16", "joint_rough_H": 0.020000000004}]
    assert inv.rough_h_text(rows, 0.02, 0.5) == ("on its 0.02 floor on 2026-10-16; ½, which is Heston, "
                                                 "on 2 of 4 diagnosed sessions: 2026-05-29 and 2026-06-18")
    assert inv.rough_h_text(rows[:1], 0.02, 0.5) == ("never on its 0.02 floor; never ½ (Heston) "
                                                     "over 1 diagnosed session")
    assert inv.rough_h_text([], 0.02, 0.5) == "no diagnosed session stores a rough-Heston fit"


def test_ledger_record_says_what_is_registered():
    assert inv.ledger_text({"state": "absent"}) == "No entry committed yet"
    ok = {"state": "ok", "n": 2, "first": "2026-10-26", "last": "2026-10-27"}
    assert inv.ledger_text(ok) == "2 sessions registered (2026-10-26 → 2026-10-27), its hash chain intact"
    one = {"state": "ok", "n": 1, "first": "2026-10-26", "last": "2026-10-26"}
    assert inv.ledger_text(one) == "1 session registered (2026-10-26), its hash chain intact"
