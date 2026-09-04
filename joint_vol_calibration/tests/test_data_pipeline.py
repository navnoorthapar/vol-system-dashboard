"""
Data-pipeline provenance tests (2026-09-04 audit).

A month of unattended daily runs showed two-thirds of August's joint
calibrations were under-identified (one VIX tenor). The cause was not a flaky
feed: yfinance treats ``end`` as EXCLUSIVE while the downloader documented it
as inclusive, so any run that fired before midnight UTC dropped the session
that had just closed — and the VIX9D/3M/6M indices, which Yahoo exposes only
as a current-day row, were never captured. These tests pin the three fixes:

  1. every downloader shifts the inclusive end date by one day for yfinance;
  2. the calibrator takes the VIX term-structure row ON the SPX spot session
     (a missing row leaves the leg empty for the gate — it never borrows the
     last row the feed happened to deliver);
  3. the cloud refresh names each calibration by the closed session it
     describes, not by the run date.
"""
import importlib.util
import pathlib

import pandas as pd
import pytest

from joint_vol_calibration.data import yf_downloader as dl
from joint_vol_calibration.calibration import joint_calibrator as jc


# ── 1. inclusive end dates ────────────────────────────────────────────────────

class _Recorded(Exception):
    """Raised by the fakes once the call arguments have been captured."""


def test_yf_end_shifts_inclusive_end_by_one_day():
    assert dl._yf_end("2026-09-04") == "2026-09-05"
    assert dl._yf_end("2026-09-30") == "2026-10-01"   # month boundary
    assert dl._yf_end("2026-12-31") == "2027-01-01"   # year boundary


def test_spx_download_requests_the_session_after_end(monkeypatch):
    seen = {}

    class _FakeTicker:
        def __init__(self, symbol):
            seen["symbol"] = symbol

        def history(self, **kw):
            seen.update(kw)
            raise _Recorded

    monkeypatch.setattr(dl.yf, "Ticker", _FakeTicker)
    try:
        dl.download_spx_ohlcv(start="2026-08-01", end="2026-09-04")
    except _Recorded:
        pass   # some downloaders swallow provider errors and return empty
    assert seen["symbol"] == "^GSPC"
    assert seen["start"] == "2026-08-01"
    assert seen["end"] == "2026-09-05", "the just-closed session must be inside yfinance's exclusive window"


@pytest.mark.parametrize("fn", [
    dl.download_vix_index,
    dl.download_tbill_rate,
    dl.download_vix_term_structure,
])
def test_bulk_downloads_request_the_session_after_end(monkeypatch, fn):
    seen = {}

    def _fake_download(*args, **kw):
        seen.update(kw)
        raise _Recorded

    monkeypatch.setattr(dl.yf, "download", _fake_download)
    try:
        fn(start="2026-08-01", end="2026-09-04")
    except _Recorded:
        pass   # some downloaders swallow provider errors and return empty
    assert "end" in seen, "yfinance was never called"
    assert seen["start"] == "2026-08-01"
    assert seen["end"] == "2026-09-05"


# ── 2. VIX term-structure row pinned to the spot session ─────────────────────

def _wide_fixture() -> pd.DataFrame:
    """Two sessions: 09-03 complete, 09-04 with only ^VIX (sub-tenors missing)."""
    return pd.DataFrame({
        "date":   ["2026-09-03", "2026-09-04"],
        "^VIX9D": [13.1, None],
        "^VIX":   [14.3, 14.9],
        "^VIX3M": [16.0, None],
        "^VIX6M": [17.2, None],
        "^VVIX":  [84.0, 86.0],
    })


def _calibrator(spot_date, monkeypatch):
    cal = jc.JointCalibrator.__new__(jc.JointCalibrator)
    cal.as_of_date = "2026-09-04"
    if spot_date is not None:
        cal.spot_date = spot_date
    monkeypatch.setattr(jc.db, "get_vix_term_structure_wide",
                        lambda as_of_date, start_date=None: _wide_fixture())
    return cal


def test_term_structure_uses_row_on_spot_session(monkeypatch):
    ts = _calibrator("2026-09-03", monkeypatch)._prepare_vix_term_structure()
    assert len(ts) == 4
    assert ts["market_price"].tolist() == [13.1, 14.3, 16.0, 17.2]


def test_term_structure_does_not_borrow_a_complete_earlier_row(monkeypatch):
    # Spot is 09-04, whose sub-tenors are missing. The leg must reflect THAT
    # session (1 tenor -> rejected by the >=3 gate), not quietly use 09-03.
    ts = _calibrator("2026-09-04", monkeypatch)._prepare_vix_term_structure()
    assert len(ts) == 1
    assert ts["market_price"].tolist() == [14.9]
    ok, reason = jc.is_acceptable_calibration(
        {"kappa": 2.0, "theta": 0.05, "sigma": 0.4, "rho": -0.7, "v0": 0.02},
        spx_iv_rmse=0.5, n_vix_tenors=len(ts))
    assert not ok and "under-identified" in reason


def test_term_structure_missing_spot_row_is_empty(monkeypatch):
    ts = _calibrator("2026-09-08", monkeypatch)._prepare_vix_term_structure()
    assert ts.empty


def test_term_structure_legacy_fallback_without_spot_date(monkeypatch):
    # Calibrators built without _load_market_data keep the old last-row behaviour.
    ts = _calibrator(None, monkeypatch)._prepare_vix_term_structure()
    assert ts["market_price"].tolist() == [14.9]


# ── 3. calibrations named by closed session ──────────────────────────────────

def _cloud_refresh():
    path = pathlib.Path(__file__).resolve().parents[2] / "scripts" / "cloud_refresh.py"
    spec = importlib.util.spec_from_file_location("cloud_refresh_under_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_calibration_is_named_by_closed_session_not_run_date():
    cr = _cloud_refresh()
    spx = pd.DataFrame({"date": ["2026-09-03", "2026-09-04"], "close": [7747.71, 7718.60]})
    assert cr.latest_session_date(spx) == "2026-09-04"
    assert cr.latest_session_date(pd.DataFrame(columns=["date", "close"])) is None
    assert cr.latest_session_date(None) is None
    assert cr.calibration_path("2026-09-04").name == "joint_cal_2026-09-04.pkl"
    assert cr.calibration_path("2026-09-04").parent.name == "calibrations"
