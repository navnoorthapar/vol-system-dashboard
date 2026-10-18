"""
C22 Stage B: the horizon-21 PDV forecast is what S1/S1C trade against.

pdv_iv_spread = σ̂(t) − VIX(t)/100 must use σ̂ = the walk-forward PDVLinear
forecast of the RMS vol over the next 21 sessions (variance-consistent and
horizon-matched with the 30-day VIX), dated at its forecast origin (≤ t).
Pre-C22 the backtest used a 1-day |r| (E|r|) model refit yearly, whose mean
was ~0.7× realised vol, so "VIX > σ̂ + 2 vp" held on ~95% of 2018-2025
sessions and the S1C long trigger was nearly always on.

Real-data numbers come from the captured SPX/VIX fixture (2010-2025).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.backtest import backtest_engine as bte
from joint_vol_calibration.models.pdv import forward_target, walk_forward_vol_forecast
from joint_vol_calibration.signals.regime_classifier import (
    PDV_SOURCE_INJECTED,
    PDV_SOURCE_RV_PROXY,
    PDV_SOURCE_WALK_FORWARD,
    PDV_WALK_FORWARD,
    build_features,
    walk_forward_pdv_vol,
)
from joint_vol_calibration.signals.signal_engine import SignalEngine

_FIXTURE = Path(__file__).parent / "fixtures" / "pdv_spx_vix_daily_2010_2025.csv.gz"
BT_START, BT_END = "2018-01-01", "2025-03-24"


def _frames(raw: pd.DataFrame):
    r = raw["log_return"].fillna(0.0)
    spx = pd.DataFrame({"date": raw.index, "close": 2000.0 * np.exp(r.cumsum()).values,
                        "log_return": r.values})
    vix = pd.DataFrame({"date": raw.index, "^VIX": raw["vix_close"].values})
    return spx, vix


@pytest.fixture(scope="module")
def raw():
    df = pd.read_csv(_FIXTURE, index_col=0, parse_dates=True)
    return df[df.index <= BT_END]


@pytest.fixture(scope="module")
def frames(raw):
    return _frames(raw)


@pytest.fixture(scope="module")
def feats21(frames):
    spx, vix = frames
    return build_features(spx, vix, pdv_vol_series=PDV_WALK_FORWARD)


def _window(x):
    return x[(x.index >= BT_START) & (x.index <= BT_END)]


# ── The forecast itself ───────────────────────────────────────────────────────

class TestHelper:
    def test_helper_is_the_pdv_walk_forward_forecast(self, frames, raw):
        spx, _ = frames
        got = walk_forward_pdv_vol(spx)
        # closes rebuild every return but the first row's
        want = walk_forward_vol_forecast(raw["log_return"].iloc[1:].dropna(), horizon=21)
        pd.testing.assert_series_equal(got, want, check_names=False, rtol=1e-12)

    def test_returns_rebuilt_from_closes_not_dropped(self, frames):
        """A NaN in the stored log_return (the DB has one on 2025-05-08) must
        not merge two sessions into one return."""
        spx, _ = frames
        holed = spx.copy()
        holed.loc[holed.index[-30], "log_return"] = np.nan
        a, b = walk_forward_pdv_vol(spx), walk_forward_pdv_vol(holed)
        pd.testing.assert_series_equal(a, b)
        assert spx["date"].iloc[-30] in set(b.index)

    def test_bad_sentinel_rejected(self, frames):
        spx, vix = frames
        with pytest.raises(ValueError):
            build_features(spx, vix, pdv_vol_series="pdv21")


class TestRealDataScale:
    def test_mean_forecast_matches_forward_rv21(self, feats21, frames):
        """2018-01..2025-03: mean σ̂ / mean forward RV21 = 0.92 (history from 2010)."""
        spx, vix = frames
        sigma = _window(feats21["pdv_iv_spread"] + feats21["vix"])
        r = spx.set_index("date")["log_return"]
        y = forward_target(r, 21, "rv")
        both = pd.concat([sigma.rename("f"), y.rename("y")], axis=1, join="inner").dropna()
        assert len(both) > 1700
        assert 0.85 <= both["f"].mean() / both["y"].mean() <= 1.15

    def test_long_trigger_share_is_no_longer_near_permanent(self, feats21, frames):
        """Share of sessions with VIX > σ̂ + 2 vp (the S1C long / S1 short
        trigger): 1-day E|r| model ~95%, PDV-21 ~79%, ex-post truth
        (VIX > forward RV21 + 2 vp) ~71%."""
        spx, vix = frames
        new = _window(feats21["pdv_iv_spread"])
        share_new = float((new < -0.02).mean())

        legacy = bte.BacktestEngine(start_date=BT_START, end_date=BT_END, pdv_horizon=1)
        old_vol = legacy._build_walkforward_pdv_vol(spx)
        old = _window(build_features(spx, vix, pdv_vol_series=old_vol)["pdv_iv_spread"])
        share_old = float((old < -0.02).mean())

        r = spx.set_index("date")["log_return"]
        truth = _window(forward_target(r, 21, "rv") - feats21["vix"]).dropna()
        share_truth = float((truth < -0.02).mean())

        assert share_old > 0.90                       # the pre-C22 artefact
        assert share_new < 0.85
        assert share_old - share_new > 0.10
        assert abs(share_new - share_truth) < 0.12

    def test_whole_window_uses_the_forecast_not_the_proxy(self, feats21):
        assert feats21.attrs["pdv_iv_spread_source"] == PDV_SOURCE_WALK_FORWARD
        sigma = feats21["pdv_iv_spread"] + feats21["vix"]
        spx_vol = walk_forward_pdv_vol(_frames(pd.read_csv(_FIXTURE, index_col=0,
                                                           parse_dates=True))[0])
        w = _window(sigma)
        np.testing.assert_allclose(w.values, spx_vol.reindex(w.index).values, rtol=1e-9)


# ── No look-ahead ─────────────────────────────────────────────────────────────

class TestNoLookAhead:
    def test_spread_on_t_ignores_returns_after_t(self, frames):
        """Poison every return after T: σ̂ and the spread dated ≤ T are unchanged."""
        spx, vix = frames
        cut = pd.Timestamp("2020-02-20")
        poisoned = spx.copy()
        after = poisoned["date"] > cut
        rng = np.random.default_rng(0)
        poisoned.loc[after, "log_return"] = rng.normal(0, 0.08, int(after.sum()))
        poisoned["close"] = 2000.0 * np.exp(poisoned["log_return"].cumsum())
        a = build_features(spx, vix, pdv_vol_series=PDV_WALK_FORWARD)["pdv_iv_spread"]
        b = build_features(poisoned, vix, pdv_vol_series=PDV_WALK_FORWARD)["pdv_iv_spread"]
        np.testing.assert_allclose(a[a.index <= cut].values, b[b.index <= cut].values,
                                   rtol=1e-10, atol=1e-12)
        assert not np.allclose(a[a.index > cut].values[25:60], b[b.index > cut].values[25:60])

    def test_signal_engine_default_is_pdv21_lagged_one_session(self, frames, feats21):
        spx, vix = frames
        sub_spx = spx[spx["date"] >= "2014-01-01"]
        sub_vix = vix[vix["date"] >= "2014-01-01"]
        out = SignalEngine().generate(sub_spx, sub_vix, start_date="2019-01-01",
                                      end_date="2019-12-31")
        assert out.attrs["pdv_iv_spread_source"] == PDV_SOURCE_WALK_FORWARD
        raw = build_features(sub_spx, sub_vix, pdv_vol_series=PDV_WALK_FORWARD)
        lagged = raw["pdv_iv_spread"].shift(1).reindex(out.index)
        np.testing.assert_allclose(out["pdv_iv_spread"].values, lagged.values, rtol=1e-12)

    def test_signal_engine_none_still_means_rv_proxy(self, frames):
        spx, vix = frames
        out = SignalEngine().generate(spx, vix, start_date="2019-01-01",
                                      end_date="2019-06-30", precomputed_pdv_vol=None)
        assert out.attrs["pdv_iv_spread_source"] == PDV_SOURCE_RV_PROXY


# ── Backtest engine ───────────────────────────────────────────────────────────

class TestBacktestEngine:
    def test_canonical_horizon_is_21(self):
        assert bte.PDV_SPREAD_HORIZON == 21
        assert bte.BacktestEngine().pdv_horizon == 21
        assert not any("PDV spread units" in f for f in bte.BacktestEngine.HONEST_FAILURES)

    def test_engine_spread_is_the_daily_purged_forecast(self, frames):
        spx, _ = frames
        eng = bte.BacktestEngine(start_date=BT_START, end_date=BT_END)
        got = eng._build_walkforward_pdv_vol(spx)
        want = walk_forward_pdv_vol(spx)
        want = want[want.index <= BT_END]
        pd.testing.assert_series_equal(got, want)

    def test_injected_series_is_used_and_truncated(self, frames):
        spx, _ = frames
        s = walk_forward_pdv_vol(spx)
        eng = bte.BacktestEngine(start_date="2019-01-01", end_date="2019-12-31",
                                 pdv_vol_series=s * 1.1)
        got = eng._build_walkforward_pdv_vol(spx.iloc[:10])     # spx ignored when injected
        assert got.index.max() <= pd.Timestamp("2019-12-31")
        np.testing.assert_allclose(got.values, (s * 1.1).reindex(got.index).values)

    def test_signals_df_carries_the_pdv21_spread(self, frames):
        spx, vix = frames
        vix = vix.assign(**{"^VIX3M": vix["^VIX"] * 1.05, "^VIX6M": vix["^VIX"] * 1.1,
                            "^VIX9D": vix["^VIX"] * 0.95, "^VVIX": 90.0})
        eng = bte.BacktestEngine(start_date="2019-01-01", end_date="2019-12-31")
        eng.run(spx[spx["date"] >= "2015-01-01"], vix[vix["date"] >= "2015-01-01"])
        assert eng.signals_df.attrs["pdv_iv_spread_source"] == PDV_SOURCE_INJECTED
        sig = eng.signals_df["pdv_iv_spread"]
        sigma = walk_forward_pdv_vol(spx[spx["date"] >= "2015-01-01"])
        vixs = vix.set_index("date")["^VIX"] / 100.0
        want = (sigma - vixs).shift(1).reindex(sig.index)
        np.testing.assert_allclose(sig.values, want.values, rtol=1e-10)


# ── c22_bridge applies both upstream rows ─────────────────────────────────────

class TestBridgeSteps:
    def test_last_two_rows_are_always_applied(self):
        from joint_vol_calibration.backtest import c22_bridge
        steps = {s["key"]: s for s in c22_bridge.build_steps()}
        assert steps["r2_gate"]["set"] == {"gate": bte._DEFAULT_GATE}
        assert steps["pdv_h21"]["set"] == {"pdv_h": 21}
        assert list(steps)[-1] == "pdv_h21"

    def test_regressed_upstream_raises_instead_of_no_op(self, monkeypatch):
        from joint_vol_calibration.backtest import c22_bridge
        monkeypatch.setattr(c22_bridge, "probe_pdv_forward_window", lambda: False)
        with pytest.raises(RuntimeError, match="pdv_h21"):
            c22_bridge.build_steps()
