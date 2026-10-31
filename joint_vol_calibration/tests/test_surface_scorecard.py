"""
Model scorecard (surface/scorecard.py and quotes.py): the scoring universe is
the calibrator's own, Heston at the stored model_version-2 fit reproduces the
pickle's in-sample and out-of-sample RMSE through the generic iv(k, T)
interface, and new models plug in through the registry.
"""
import numpy as np
import pytest

from joint_vol_calibration.calibration.joint_calibrator import JointCalibrator, recorded_inputs
from joint_vol_calibration.surface import quotes as sqm
from joint_vol_calibration.surface import scorecard as sc

SESSION = "2026-10-16"


@pytest.fixture(scope="module")
def sq():
    return sqm.load_session_quotes(SESSION)


@pytest.fixture(scope="module")
def cal():
    # the committed fit's own inputs (^IRX reference): the surface layer cleans
    # each session with the reference its fit recorded (session_rate_reference)
    return JointCalibrator(SESSION, **recorded_inputs(sc.load_v2_calibration(SESSION)))


def _ctx(sq):
    flag = np.zeros(len(sq.quotes), dtype=bool)
    return sc.ScoreContext(sq, flag, sc.surface_fit_mask(sq.quotes, flag))


def test_universe_is_the_calibrators_oos_universe(sq, cal):
    full = cal.spx_surface_full
    q = sq.quotes
    assert len(q) == len(full) == 2913
    np.testing.assert_array_equal(q["strike"].to_numpy(), full["strike"].to_numpy())
    np.testing.assert_array_equal(q["expiry"].to_numpy(), full["expiry"].astype(str).to_numpy())
    np.testing.assert_allclose(q["mid_iv"].to_numpy(), full["implied_vol"].to_numpy(), rtol=0, atol=0)
    np.testing.assert_allclose(q["forward"].to_numpy(), full["forward"].to_numpy(), rtol=0, atol=0)
    assert (q["bid_iv"] <= q["mid_iv"]).all()
    assert (q["mid_iv"] <= q["ask_iv"]).all()


def test_anchor_mask_is_the_calibrators_anchor_set(sq, cal):
    mask = sc.heston_anchor_mask(sq.quotes, SESSION)
    got = set(map(tuple, sq.quotes.loc[mask, ["expiry", "strike", "right"]].astype(str).to_numpy()))
    want = set(map(tuple, cal.spx_surface[["expiry", "strike", "right"]].astype(str).to_numpy()))
    assert got == want
    assert len(got) == 53


def test_heston_v2_reproduces_the_stored_rmses(sq):
    """Generic iv(k, T) on F = 1, D = 1 vs the calibrator's own pricing on
    (F_T, D_T): the pickle's OOS (2,860 quotes) and in-sample RMSEs."""
    card, res = sc.score_models(_ctx(sq), ["heston_v2"])
    row = card["models"][0]
    stored = sc.load_v2_calibration(SESSION)["leg_losses"]
    m = row["metrics"]
    assert row["n_in_sample"] == 53
    assert m["out_of_sample"]["n"] == 2860
    assert m["out_of_sample"]["rmse_vp"] == pytest.approx(stored["spx_oos_iv_rmse"], rel=1e-9)
    assert m["in_sample"]["rmse_vp"] == pytest.approx(stored["spx_iv_rmse_exact"], rel=1e-9)
    assert row["dense_grid"]["butterfly_violations"] == 0
    assert row["dense_grid"]["calendar_violations"] == 0
    assert row["params"]["pinned_params"] == ["sigma"]
    assert set(res["model"]) == {"heston_v2"}
    assert len(res) == len(sq.quotes)


def test_heston_iv_function_has_the_black_scholes_limit():
    iv = sc.heston_iv_function({"kappa": 2.0, "theta": 0.04, "sigma": 1e-4, "rho": 0.0, "v0": 0.04})
    np.testing.assert_allclose(iv(np.linspace(-0.3, 0.2, 11), 0.5), 0.2, atol=2e-6)


def test_iv_from_normalised_calls_inverts_a_black_scholes_pricer():
    """The integration hook for further models: a forward-normalised call
    pricer at 25% vol comes back as a flat 25% smile, puts by parity."""
    from scipy.special import ndtr

    def bs(K, T, vol=0.25):
        sd = vol * np.sqrt(T)
        d1 = -np.log(K) / sd + sd / 2
        return ndtr(d1) - K * ndtr(d1 - sd)
    iv = sc.iv_from_normalised_calls(bs)
    np.testing.assert_allclose(iv(np.linspace(-0.4, 0.3, 15), 0.5), 0.25, rtol=0, atol=1e-9)
    np.testing.assert_allclose(iv(np.array([-0.05, 0.0, 0.05]), 7 / 365), 0.25, rtol=0, atol=1e-9)


def test_registered_model_is_scored_on_the_same_quotes(sq):
    def flat(ctx):
        return sc.ModelSpec("flat20", lambda k, T: np.full(len(k), 0.20),
                            np.zeros(len(ctx.quotes), dtype=bool), 1, "flat 20% (test)")
    sc.register_model("flat20", flat)
    try:
        card, _res = sc.score_models(_ctx(sq), ["flat20"])
    finally:
        sc.MODEL_REGISTRY.pop("flat20")
    q = sq.quotes
    inside = ((q["bid_iv"] <= 0.2) & (q["ask_iv"] >= 0.2)).mean() * 100
    m = card["models"][0]["metrics"]["all"]
    assert m["inside_spread_pct"] == pytest.approx(inside)
    assert m["rmse_vp"] == pytest.approx(np.sqrt(np.mean((20 - 100 * q["mid_iv"]) ** 2)))
    assert "flat20" not in sc.MODEL_REGISTRY


def test_legacy_or_missing_calibration_is_not_scored():
    assert sc.load_v2_calibration("2026-10-12") is None          # pre-C22 pickle, no chain
    assert sc.load_v2_calibration("2026-01-02") is None          # no pickle


def test_surface_fit_mask_alternates_strikes_and_drops_flagged(sq):
    q = sq.quotes
    flag = np.zeros(len(q), dtype=bool)
    flag[0] = True
    m = sc.surface_fit_mask(q, flag)
    assert not m[0]
    first = q[q["expiry"] == q["expiry"].iloc[0]].sort_values("strike")
    pos = np.arange(len(first))
    assert m[first.index.to_numpy()][1:].tolist() == list((pos % 2 == 0)[1:])
    assert 0.45 < m.mean() < 0.55


@pytest.mark.parametrize(("book_state", "chain", "noted"), [
    ("closing", {"quote_time": "2026-10-30T08:51:33Z"}, True),    # a pre-QUANT-01 fit: its download clock
    ("closing", {"fetched_at": "2026-10-30T08:51:33Z"}, True),    # Fri 04:51 ET: the overnight book
    ("closing", {"fetched_at": "2026-10-29T22:00:00Z"}, False),   # Thu 18:00 ET: still the closing book
    ("gth", {"fetched_at": "2026-10-30T08:51:33Z"}, False),       # scored on the overnight book itself
    ("closing", {}, False),                                       # no download time recorded
])
def test_a_fit_made_overnight_and_scored_on_the_closing_book_says_so(book_state, chain, noted):
    """FUND_GRADE_PLAN 0.8: a session's surface is now built from its closing book
    when one is stored, while the bot still fits Heston to the Yahoo chain it
    downloads overnight. The Heston row then says the fit's chain was downloaded
    after the closing book had given way to the overnight one, so part of its error
    on these quotes is the market's move between the two."""
    from types import SimpleNamespace

    ctx = SimpleNamespace(sq=SimpleNamespace(session="2026-10-29", inputs={"book_state": book_state},
                                             diagnostics={"source": "cboe"}))
    note = sc._fit_time_note(chain, ctx)
    assert bool(note) is noted
    if noted:
        assert note.startswith("; that chain was downloaded at 2026-10-30 08:51 UTC, after the closing "
                               "book scored here had given way to the overnight one")
    # the Heston row's note on a fit made on another book carries it
    row = sc._fit_book_note({"chain_diagnostics": {"source": "yfinance", **chain}}, ctx)
    assert row.startswith(". The fit was made on the session's Yahoo chain")
    assert row.endswith(note)
