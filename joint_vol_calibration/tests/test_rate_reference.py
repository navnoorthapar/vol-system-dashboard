"""
test_rate_reference.py — the Treasury zero curve wired into the consumers (C23).

Before C23 every consumer discounted with one number: the calibrator and the
surface layer with Yahoo's ^IRX 13-week bill for every expiry, the C7 hedge
study with a hard-coded 1.5%, the Greeks surface with one r and q (medians
over the anchor expiries) for every maturity. Now:

  * option_chain.clean_option_chain takes r_ref as a number or a function r(T),
    evaluated at each expiry (parity plausibility check, fixed-D fallback);
  * JointCalibrator's default reference is the session's Treasury zero curve
    (^IRX when the curve is missing or stale, recorded); every fit committed
    before C23 recorded ^IRX and is rebuilt with recorded_inputs();
  * the surface layer cleans a session's chain with that session's recorded
    reference (session_rate_reference), so its universe stays the fit's;
  * delta_hedger's C7 study uses the 1-year zero rate at entry;
  * the Greeks surface uses per-maturity carry: the fit's parity forwards and
    the Treasury curve's discount (risk_monitor.calibration_carry).
"""
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.calibration import joint_calibrator as jc
from joint_vol_calibration.config import DATA_DIR
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import option_chain as oc
from joint_vol_calibration.data import rates
from joint_vol_calibration.tests.test_joint_calibrator import _heston_chain

_CAL = DATA_DIR / "calibrations"
_V2 = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")


def _committed(session):
    with open(_CAL / f"joint_cal_{session}.pkl", "rb") as f:
        return pickle.load(f)


# ── option_chain: r_ref per expiry ───────────────────────────────────────────

def test_a_term_reference_is_evaluated_at_each_expiry():
    """A chain priced at r = 4%: with one reference rate every expiry keeps
    its parity regression; with r(T) = 6% beyond 0.2y the long expiries fail
    the 150 bp plausibility check against THEIR rate and fall back to the
    fixed discount at that rate, while the short ones are untouched."""
    session, S = "2026-09-23", 6600.0
    exps = ["2026-10-16", "2026-11-20", "2027-01-15", "2027-03-19", "2027-06-17"]
    raw = _heston_chain(session=session, S=S, r=0.04, q=0.013, expiries=exps)
    flat = oc.clean_option_chain(raw, session, S, r_ref=0.04)
    term = oc.clean_option_chain(raw, session, S, r_ref=lambda T: 0.04 if T < 0.2 else 0.06)
    assert flat.ok
    assert term.ok
    assert (flat.carry["method"] == "parity_regression").all()
    long_ = term.carry["T"] >= 0.2
    assert long_.any()
    assert (~long_).any()
    assert (term.carry.loc[long_, "method"] == "parity_fixed_D").all()
    np.testing.assert_allclose(term.carry.loc[long_, "discount"],
                               np.exp(-0.06 * term.carry.loc[long_, "T"]), rtol=1e-14)
    pd.testing.assert_frame_equal(term.carry.loc[~long_].reset_index(drop=True),
                                  flat.carry.loc[~long_].reset_index(drop=True))
    assert term.diagnostics["r_ref"] == "r(T) per expiry"
    assert flat.diagnostics["r_ref"] == 0.04


# ── JointCalibrator: the reference rate ──────────────────────────────────────

def test_default_reference_is_the_sessions_treasury_curve():
    cal = jc.JointCalibrator(as_of_date="2026-10-16")
    curve = rates.get_zero_curve("2026-10-16")
    assert cal.rate_ref.kind == "treasury_curve"
    assert cal.r_reference == pytest.approx(curve.zero_rate(91 / 365), abs=1e-15)
    assert cal.r_reference_source.startswith("Treasury zero curve 2026-10-16")
    assert cal.rate_ref.rate(1.0) == pytest.approx(curve.zero_rate(1.0), abs=1e-15)
    rec = cal.rate_ref.record()
    assert rec["kind"] == "treasury_curve"
    assert rec["curve_date"] == "2026-10-16"


def test_the_curve_moves_the_fixed_discount_expiries_only():
    """2026-10-16: the expiries whose parity slope is implausible are
    discounted at r(T) under the curve and at ^IRX under the committed
    reference; every regression expiry keeps its own D_T either way."""
    ref = _committed("2026-10-16")
    irx = jc.JointCalibrator(as_of_date="2026-10-16", **jc.recorded_inputs(ref))
    cur = jc.JointCalibrator(as_of_date="2026-10-16")
    a = irx.spx_carry.set_index("expiry")
    b = cur.spx_carry.set_index("expiry")
    both = a.index.intersection(b.index)
    fixed = [e for e in both if a.loc[e, "method"] == "parity_fixed_D"
             and b.loc[e, "method"] == "parity_fixed_D"]
    assert fixed
    for e in fixed:
        T = float(b.loc[e, "T"])
        assert b.loc[e, "discount"] == pytest.approx(math.exp(-cur.rate_ref.rate(T) * T), rel=1e-14)
        assert a.loc[e, "discount"] == pytest.approx(math.exp(-irx.r_reference * T), rel=1e-14)
    regr = [e for e in both if a.loc[e, "method"] == b.loc[e, "method"] == "parity_regression"]
    assert regr
    for e in regr:
        assert a.loc[e, "discount"] == b.loc[e, "discount"]
    assert irx.r_reference == ref["r_reference"]
    assert irx.r_reference_source == ref["r_reference_source"]


def test_a_missing_curve_falls_back_to_irx_and_says_so(monkeypatch):
    def stale(as_of, **kw):
        raise db.StaleDataError("newest Treasury quote is 9 days old")
    monkeypatch.setattr(rates, "get_zero_curve", stale)
    ref = jc.reference_rate("2026-10-16")
    assert ref.kind == "irx"
    assert ref.r_91d == pytest.approx(db.get_tbill_rate("2026-10-16"))
    assert "9 days old" in ref.record()["fallback"]
    assert jc.recorded_rate_reference({"rate_reference": ref.record()}) == "irx"
    with pytest.raises(ValueError, match="rate reference must be one of"):
        jc.reference_rate("2026-10-16", "sofr")


@pytest.mark.parametrize("session", _V2)
def test_committed_fits_recorded_irx(session):
    """Every committed fit predates the curve: its record says ^IRX, and the
    recorded inputs rebuild it on ^IRX."""
    ref = _committed(session)
    assert "rate_reference" not in ref
    assert ref["r_reference_source"].startswith("^IRX")
    assert jc.recorded_inputs(ref)["rate_reference"] == "irx"
    assert jc.session_rate_reference(session).kind == "irx"


def test_recorded_inputs_of_curve_and_argument_fits():
    curve = {"vix_index": [{"ticker": "^VIX", "market": 20.0}],
             "rate_reference": {"kind": "treasury_curve", "r_91d": 0.04, "label": "x"}}
    arg = {"vix_index": [{"ticker": "^VIX", "market": 20.0}],
           "rate_reference": {"kind": "argument", "r_91d": 0.031, "label": "argument"}}
    assert jc.recorded_inputs(curve) == {"vix_targets": {"^VIX": 20.0},
                                         "rate_reference": "treasury_curve"}
    assert jc.recorded_inputs(arg) == {"vix_targets": {"^VIX": 20.0}, "r": 0.031}


def test_session_rate_reference_reads_the_sessions_own_fit(tmp_path):
    v2 = {"model_version": 2, "spot_date": "2026-10-16",
          "rate_reference": {"kind": "treasury_curve", "r_91d": 0.04, "label": "x"}}
    with open(tmp_path / "joint_cal_2026-10-16.pkl", "wb") as f:
        pickle.dump(v2, f)
    assert jc.session_rate_reference("2026-10-16", tmp_path).kind == "treasury_curve"
    with open(tmp_path / "joint_cal_2026-10-16.pkl", "wb") as f:
        pickle.dump({**v2, "rate_reference": None}, f)
    assert jc.session_rate_reference("2026-10-16", tmp_path).kind == "irx"
    # no stored fit: the calibrator's default
    assert jc.session_rate_reference("2026-10-15", tmp_path).kind == jc.DEFAULT_RATE_REFERENCE


# ── surface layer: the session's own reference ───────────────────────────────

def test_surface_quotes_clean_with_the_sessions_recorded_reference():
    from joint_vol_calibration.surface import quotes as sqm
    sq = sqm.load_session_quotes("2026-10-16")
    assert sq.r_ref == db.get_tbill_rate("2026-10-16")
    assert sq.r_ref_source == sqm.IRX_SOURCE
    ref = jc.reference_rate("2026-10-16")
    cur = sqm.load_session_quotes("2026-10-16", r_ref=ref.r_91d)
    assert cur.r_ref_source == "argument"


# ── C7 delta-hedge study: the 1-year zero rate at entry ──────────────────────

def test_c7_uses_the_one_year_zero_rate_at_entry():
    from joint_vol_calibration.backtest import delta_hedger as dh
    r = dh.entry_rate()
    assert r == pytest.approx(rates.zero_rate("2020-01-02", 1.0), abs=1e-15)
    assert abs(r - 0.015) > 1e-4                      # not the old hard-coded 1.5%
    committed = pd.read_parquet(dh.OUTPUT_PARQUET)
    assert committed.attrs["r"] == pytest.approx(r, abs=1e-15)
    assert committed.attrs["r_source"].startswith("Treasury zero curve 2020-01-02")


# ── Greeks surface: per-maturity carry ───────────────────────────────────────

def test_greeks_carry_reprices_the_parity_forwards_on_the_treasury_curve():
    """The headline fit's carry: S e^{(r(T)-q(T))T} is its parity forward at
    every listed expiry and r(T) the Treasury zero rate. The flat medians
    used before C23 miss the 1-year forward by 0.5%."""
    from joint_vol_calibration.greeks.risk_monitor import RiskMonitor, calibration_carry
    cal = _committed("2026-10-16")
    r_fn, q_fn, rec = calibration_carry(cal)
    S = float(cal["S"])
    curve = rates.get_zero_curve("2026-10-16")
    for row in cal["carry_by_expiry"]:
        T = float(row["T"])
        assert S * math.exp((r_fn(T) - q_fn(T)) * T) == pytest.approx(row["forward"], rel=1e-12)
        assert r_fn(T) == pytest.approx(curve.zero_rate(T), abs=1e-15)
    assert rec["discount"].startswith("Treasury zero curve 2026-10-16")
    F1 = S * math.exp((float(cal["r"]) - float(cal["q"])) * 1.0)
    F1_parity = S * math.exp((r_fn(1.0) - q_fn(1.0)) * 1.0)
    assert abs(F1 / F1_parity - 1) > 0.004            # the flat-carry error it replaces
    m = RiskMonitor.from_calibration(_CAL / "joint_cal_2026-10-16.pkl")
    assert callable(m.r)
    assert m.carry == rec
    # without a curve: the parity discounts, same forwards
    r2, q2, rec2 = calibration_carry(cal, curve=False)
    for row in cal["carry_by_expiry"]:
        T = float(row["T"])
        assert S * math.exp((r2(T) - q2(T)) * T) == pytest.approx(row["forward"], rel=1e-12)
        assert math.exp(-r2(T) * T) == pytest.approx(row["discount"], rel=1e-12)
    assert rec2["discount"].startswith("parity discounts")


def test_published_greeks_surface_carries_the_per_maturity_carry():
    from joint_vol_calibration.greeks.risk_monitor import calibration_carry, load_greeks_surface
    df = load_greeks_surface(DATA_DIR / "greeks" / "greeks_surface.parquet")
    prov = df.attrs["provenance"]
    cal = _committed(prov["as_of_date"])
    r_fn, q_fn, _ = calibration_carry(cal)
    for row in prov["carry"]["by_maturity"]:
        T = row["T_days"] / 365.0
        assert row["r"] == pytest.approx(r_fn(T), abs=1e-15)
        assert row["q"] == pytest.approx(q_fn(T), abs=1e-15)
    for _, g in df.groupby("T_days"):
        T = float(g["T"].iloc[0])
        np.testing.assert_allclose(g["r"], r_fn(T), rtol=0, atol=1e-15)
        np.testing.assert_allclose(g["q"], q_fn(T), rtol=0, atol=1e-15)
