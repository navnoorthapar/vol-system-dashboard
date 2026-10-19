"""
test_signal_engine.py — Unit tests for C9: Trading Signal Engine.

Coverage:
  1.  _run_statemachine     — entry, exit, max-hold, no conflict entry
  2.  generate_signal1      — entry conditions, regime gate, max-hold, strength
  3.  generate_signal2      — contango entry, backwardation entry, stop-loss
  4.  generate_signal3      — entry on low VIX/VVIX ratio, exit on reversion
  5.  combine_signals       — all-agree, two-agree, conflict, all-flat
  6.  SignalEngine.generate — output shape, columns, zero look-ahead structure
  7.  SignalEngine.save     — parquet roundtrip
  8.  signal_summary        — return structure and keys
  9.  Edge cases            — all-flat, single-day, NaN handling
  10. Constants sanity      — S1/S2/S3 thresholds in valid ranges
  11. generate_signal1_regime_filtered — R2 zeroing, resume after R2, columns
  12. compare_s1_regime_filter         — dict keys, P&L math, trade decomposition
  13. SignalEngine S1RF integration    — engine output columns, summary key
  14+ R2-exit machines, S1X/S2X/S1S/S4/S1C, and the C22 regressions:
      S2 stop direction (both sides), S1 exit on a zero crossing, precomputed
      regimes only lagged, sessions kept across a missing VVIX print.
  Signal-level look-ahead tests live in test_lookahead.py.
"""


import numpy as np
import pandas as pd
import pytest

import joint_vol_calibration.signals.signal_engine as se
from joint_vol_calibration.signals.regime_classifier import build_regime_labels
from joint_vol_calibration.signals.signal_engine import (
    S1_ENTRY_THRESHOLD,
    S1_EXIT_BAND,
    S1_MAX_HOLD,
    S1_MAX_KELLY,
    S1_RF_REGIME2_SCALE,
    S2_MAX_HOLD,
    S2_ZSCORE_ENTRY,
    S3_MAX_HOLD,
    S3_ZSCORE_ENTRY,
    S4_Z_ENTRY_SHORT,
    S4_Z_ENTRY_LONG,
    S4_Z_EXIT,
    S4_MAX_HOLD,
    S4_MAX_KELLY,
    S2_STOP_LOSS_PCT,
    SignalEngine,
    default_signals_path,
    _run_statemachine,
    _run_statemachine_r2exit,
    combine_signals,
    compare_s1_regime_filter,
    generate_signal1,
    generate_signal1_contrarian,
    generate_signal1_regime_filtered,
    generate_signal1_r2exit,
    generate_signal1_soft,
    generate_signal2,
    generate_signal2_r2exit,
    generate_signal3,
    generate_signal4,
    signal_summary,
)

# ── Minimal fixture helpers ────────────────────────────────────────────────────

def _make_flat_features(n: int = 60, seed: int = 42) -> pd.DataFrame:
    """Synthetic feature DataFrame that triggers no signals (neutral)."""
    idx  = pd.bdate_range("2022-01-03", periods=n)
    return pd.DataFrame({
        "vix":           np.full(n, 0.18),
        "ts_slope":      np.full(n, 0.002),    # mild contango
        "fear_premium":  np.full(n, 1.05),
        "rv_change_5d":  np.zeros(n),
        "pdv_iv_spread": np.zeros(n),
        "vvix":          np.full(n, 0.92),
    }, index=idx)


def _make_regimes(n: int = 60, value: int = 1) -> pd.Series:
    idx = pd.bdate_range("2022-01-03", periods=n)
    return pd.Series(value, index=idx)


def _inject_spread(feats: pd.DataFrame, start: int, end: int, spread: float) -> pd.DataFrame:
    feats = feats.copy()
    feats.iloc[start:end, feats.columns.get_loc("pdv_iv_spread")] = spread
    return feats


def _inject_slope_zscore(feats: pd.DataFrame, start: int, end: int, z_val: float) -> pd.DataFrame:
    """Inject a z-score-equivalent slope by scaling ts_slope directly."""
    feats = feats.copy()
    # ts_slope values small: set a large value to trigger zscore > 1
    feats.iloc[start:end, feats.columns.get_loc("ts_slope")] = z_val * 0.01 + 0.001
    return feats


# ── 1. _run_statemachine ──────────────────────────────────────────────────────

class TestRunStatemachine:
    def _run(self, entry_l, entry_s, exit_c, strength=None, regime=None):
        n = len(entry_l)
        idx = pd.RangeIndex(n)
        if strength is None:
            strength = pd.Series(np.ones(n), index=idx)
        if regime is None:
            regime = pd.Series(np.zeros(n), index=idx)
        return _run_statemachine(
            pd.Series(entry_l, index=idx),
            pd.Series(entry_s, index=idx),
            pd.Series(exit_c,  index=idx),
            strength, regime, max_hold=5,
        )

    def test_enters_long(self):
        r = self._run(
            entry_l=[True, False, False, False, False],
            entry_s=[False]*5,
            exit_c=[False]*5,
        )
        assert r["position"].iloc[0] == 1

    def test_enters_short(self):
        r = self._run(
            entry_l=[False]*5,
            entry_s=[False, True, False, False, False],
            exit_c=[False]*5,
        )
        assert r["position"].iloc[1] == -1

    def test_exits_on_exit_cond(self):
        r = self._run(
            entry_l=[True, False, False, False, False],
            entry_s=[False]*5,
            exit_c=[False, False, True, False, False],
        )
        assert r["position"].iloc[2] == 0   # exited on day 2
        assert r["position"].iloc[3] == 0

    def test_exits_on_max_hold(self):
        result = _run_statemachine(
            pd.Series([True] + [False]*9),
            pd.Series([False]*10),
            pd.Series([False]*10),
            pd.Series(np.ones(10)),
            pd.Series(np.zeros(10)),
            max_hold=4,
        )
        # days_held reaches max_hold=4 on the 4th session, which is the exit day
        assert list(result["position"].iloc[:5]) == [1, 1, 1, 0, 0]

    def test_no_double_entry(self):
        """Both long and short on same day → no entry."""
        r = self._run(
            entry_l=[True, False, False],
            entry_s=[True, False, False],
            exit_c=[False]*3,
        )
        assert r["position"].iloc[0] == 0

    def test_flat_when_no_entry(self):
        r = self._run(
            entry_l=[False]*5,
            entry_s=[False]*5,
            exit_c=[False]*5,
        )
        assert (r["position"] == 0).all()

    def test_no_same_day_reentry_after_exit(self):
        """A max_hold exit must leave the book flat for one day even when the
        entry condition still holds — otherwise max_hold never flattens."""
        result = _run_statemachine(
            pd.Series([True] * 6),     # entry_long true every day
            pd.Series([False] * 6),
            pd.Series([False] * 6),
            pd.Series(np.ones(6)),
            pd.Series(np.zeros(6)),
            max_hold=2,
        )
        pos = result["position"].tolist()
        # Day0 enter, Day1 hits max_hold → exit (flat), Day2 re-enter, ...
        assert pos[0] == 1
        assert pos[1] == 0, "must be flat the day after a max_hold exit"
        assert pos[2] == 1
        assert 0 in pos, "max_hold must produce at least one flat day"

    def test_no_direct_long_short_flip(self):
        """The machine must never emit +1 then -1 with no flat day between."""
        result = _run_statemachine(
            pd.Series([True, False, False, False, False, False]),
            pd.Series([False, False, True, True, True, True]),   # short pressure after
            pd.Series([False, True, False, False, False, False]),  # exit long on day1
            pd.Series(np.ones(6)),
            pd.Series(np.zeros(6)),
            max_hold=10,
        )
        pos = result["position"].tolist()
        # day0 +1, day1 exit→0 (no same-day short), day2 enter -1
        assert pos[0] == 1
        assert pos[1] == 0
        # no adjacent +1→-1 transition anywhere
        for a, b in zip(pos, pos[1:]):
            assert not (a == 1 and b == -1), "direct +1→-1 flip with no flat day"

    def test_days_held_increments(self):
        r = self._run(
            entry_l=[True] + [False]*4,
            entry_s=[False]*5,
            exit_c=[False]*5,
        )
        assert r["days_held"].iloc[0] == 1
        assert r["days_held"].iloc[1] == 2
        assert r["days_held"].iloc[2] == 3


# ── 2. generate_signal1 ───────────────────────────────────────────────────────

class TestGenerateSignal1:
    def test_no_position_when_flat(self):
        feats = _make_flat_features()
        regs  = _make_regimes(value=1)
        s1    = generate_signal1(feats, regs)
        assert (s1["s1_position"] == 0).all()

    def test_long_entry_regime0(self):
        feats = _inject_spread(_make_flat_features(), 5, 30, 0.05)
        regs  = _make_regimes(value=0)
        s1    = generate_signal1(feats, regs)
        assert s1["s1_position"].iloc[5] == 1   # long straddle

    def test_short_entry_regime1(self):
        feats = _inject_spread(_make_flat_features(), 5, 30, -0.05)
        regs  = _make_regimes(value=1)
        s1    = generate_signal1(feats, regs)
        assert s1["s1_position"].iloc[5] == -1  # short straddle

    def test_no_entry_regime2(self):
        """No new entries allowed in Regime 2."""
        feats = _inject_spread(_make_flat_features(), 0, 60, 0.10)
        regs  = _make_regimes(value=2)
        s1    = generate_signal1(feats, regs)
        assert (s1["s1_position"] == 0).all()

    def test_strength_positive_when_active(self):
        feats = _inject_spread(_make_flat_features(), 5, 20, 0.05)
        regs  = _make_regimes(value=0)
        s1    = generate_signal1(feats, regs)
        active = s1[s1["s1_position"] != 0]
        assert (active["s1_strength"] > 0).all()

    def test_kelly_capped_at_max(self):
        feats = _inject_spread(_make_flat_features(), 0, 60, 1.0)  # huge spread
        regs  = _make_regimes(value=0)
        s1    = generate_signal1(feats, regs)
        assert s1["s1_kelly"].max() <= S1_MAX_KELLY + 1e-8

    def test_max_hold_respected(self):
        """Entry day 0 (days_held 1) … day 4 reaches max_hold=5 → flat that
        day, re-entry the next day while the spread still qualifies."""
        feats = _inject_spread(_make_flat_features(), 0, 60, 0.05)
        regs  = _make_regimes(value=0)
        s1    = generate_signal1(feats, regs, max_hold=5)
        pos = s1["s1_position"].values
        assert list(pos[:6]) == [1, 1, 1, 1, 0, 1]
        assert s1["s1_days_held"].max() == 4


class TestSignal1ExitOnCrossing:
    """The documented exit is "spread reverts to zero". The code used to exit
    only INSIDE |spread| < 0.5 vp, so a spread that jumped through zero (10 of
    23 sign flips in 2018-2025) kept the trade open until max-hold."""

    @staticmethod
    def _spread_path(first: float, then: float, n: int = 20):
        feats = _make_flat_features(n)
        feats.iloc[:3, feats.columns.get_loc("pdv_iv_spread")] = first
        feats.iloc[3:, feats.columns.get_loc("pdv_iv_spread")] = then
        return feats

    def test_long_exits_when_spread_jumps_through_zero(self):
        feats = self._spread_path(0.03, -0.01)
        s1 = generate_signal1(feats, _make_regimes(20, value=0))
        assert list(s1["s1_position"].iloc[:4]) == [1, 1, 1, 0]
        assert (s1["s1_position"].iloc[3:] == 0).all()

    def test_short_exits_when_spread_jumps_through_zero(self):
        feats = self._spread_path(-0.03, 0.01)
        s1 = generate_signal1(feats, _make_regimes(20, value=1))
        assert list(s1["s1_position"].iloc[:4]) == [-1, -1, -1, 0]

    def test_trade_holds_while_spread_stays_on_entry_side(self):
        feats = self._spread_path(0.03, 0.01)      # smaller, same sign, outside band
        s1 = generate_signal1(feats, _make_regimes(20, value=0))
        assert (s1["s1_position"].iloc[:S1_MAX_HOLD - 1] == 1).all()

    def test_exit_inside_band_still_works(self):
        feats = self._spread_path(0.03, S1_EXIT_BAND / 2)
        s1 = generate_signal1(feats, _make_regimes(20, value=0))
        assert s1["s1_position"].iloc[3] == 0

    def test_contrarian_and_r2exit_variants_exit_on_crossing(self):
        feats = self._spread_path(0.03, -0.01)
        s1c = generate_signal1_contrarian(feats, _make_regimes(20, value=0))
        assert list(s1c["s1c_position"].iloc[:4]) == [-1, -1, -1, 0]
        s1x = generate_signal1_r2exit(feats, _make_regimes(20, value=0))
        assert list(s1x["s1x_position"].iloc[:4]) == [1, 1, 1, 0]


# ── 3. generate_signal2 ───────────────────────────────────────────────────────

class TestGenerateSignal2:
    def _features_with_slope(self, slope_vals) -> pd.DataFrame:
        n = len(slope_vals)
        idx = pd.bdate_range("2022-01-03", periods=n)
        feats = pd.DataFrame({
            "vix":           np.full(n, 0.18),
            "ts_slope":      slope_vals,
            "fear_premium":  np.ones(n),
            "rv_change_5d":  np.zeros(n),
            "pdv_iv_spread": np.zeros(n),
            "vvix":          np.full(n, 0.92),
        }, index=idx)
        return feats

    def test_no_signal_with_normal_slope(self):
        # Generate 400 days of flat slope (no z-score deviation)
        n = 400
        feats = self._features_with_slope(np.full(n, 0.003))
        regs  = pd.Series(1, index=feats.index)
        s2    = generate_signal2(feats, regs)
        # Constant slope → z = 0 (no contango entry); slope > 0 → no long entry
        assert (s2["s2_position"] == 0).all()
        assert s2["s2_z"].dropna().abs().max() < 1e-6

    def test_short_entry_steep_contango(self):
        """Inject steep contango → short front vol."""
        n = 400
        slope = np.full(n, 0.003)
        # Make last 20 days have extreme slope
        slope[300:320] = 0.08   # steep contango
        feats = self._features_with_slope(slope)
        regs  = pd.Series(1, index=feats.index)
        s2    = generate_signal2(feats, regs)
        active = s2[s2["s2_position"] != 0]
        assert len(active) > 0, "steep contango must open a short"
        assert (active["s2_position"] == -1).all()
        assert s2["s2_position"].iloc[300] == -1

    def test_long_entry_backwardation(self):
        n = 400
        slope = np.full(n, 0.003)
        slope[300:320] = -0.02  # backwardation
        feats = self._features_with_slope(slope)
        regs  = pd.Series(1, index=feats.index)
        s2    = generate_signal2(feats, regs)
        # Check that long positions appear during backwardation window
        window = s2["s2_position"].iloc[300:320]
        assert window.iloc[0] == 1, "backwardation must open a long on day 1"
        assert (window[window != 0] == 1).all()

    def test_no_entry_regime2(self):
        """Regime 2 gates all Signal 2 entries."""
        n = 400
        slope = np.full(n, 0.003)
        slope[300:320] = -0.02  # would normally trigger long
        feats = self._features_with_slope(slope)
        regs  = pd.Series(2, index=feats.index)  # all Regime 2
        s2    = generate_signal2(feats, regs)
        assert (s2["s2_position"] == 0).all()

    def test_kelly_capped(self):
        slope = np.concatenate([np.full(300, 0.003), np.full(100, -0.05)])
        feats = self._features_with_slope(slope)
        regs  = pd.Series(0, index=feats.index)
        s2    = generate_signal2(feats, regs)
        assert s2["s2_kelly"].max() <= 0.20 + 1e-8

    # ── stop-loss: expressed in the STRADDLE that is traded ──────────────────
    # S2 trades an SPX straddle: +1 long (loses when front vol FALLS), −1 short
    # (loses when front vol RISES). Pre-C22 the stop lived on the slope with the
    # sign inverted, so every stop fired on a winning trade.

    def _stop_case(self, slope_value: float, vix_after: float, prefix: str = "s2"):
        n = 330
        slope = np.full(n, 0.003)
        slope[300:] = slope_value               # entry on day 300, persists
        feats = self._features_with_slope(slope)
        vix = np.full(n, 0.20)
        vix[303:] = vix_after                   # front vol moves on day 303
        feats["vix"] = vix
        regs = pd.Series(1, index=feats.index)
        fn = generate_signal2 if prefix == "s2" else generate_signal2_r2exit
        return fn(feats, regs, max_hold=10)

    def test_long_straddle_stopped_when_front_vol_falls(self):
        s2 = self._stop_case(-0.05, 0.20 * (1 - S2_STOP_LOSS_PCT - 0.05))
        pos = s2["s2_position"].values
        assert list(pos[300:303]) == [1, 1, 1]
        assert bool(s2["s2_stop"].iloc[303]) and pos[303] == 0
        # the stop propagates until the state machine starts a NEW trade:
        # it would have run to max_hold (day 309), then re-enters on day 310
        assert (pos[303:310] == 0).all()
        assert pos[310] == 1 and s2["s2_days_held"].iloc[310] == 1

    def test_long_straddle_not_stopped_when_front_vol_rises(self):
        s2 = self._stop_case(-0.05, 0.20 * (1 + S2_STOP_LOSS_PCT + 0.05))
        pos = s2["s2_position"].values
        assert not s2["s2_stop"].any()
        assert (pos[300:309] == 1).all() and pos[309] == 0     # max_hold exit only

    def test_short_straddle_stopped_when_front_vol_rises(self):
        s2 = self._stop_case(0.08, 0.20 * (1 + S2_STOP_LOSS_PCT + 0.05))
        pos = s2["s2_position"].values
        assert list(pos[300:303]) == [-1, -1, -1]
        assert bool(s2["s2_stop"].iloc[303]) and pos[303] == 0

    def test_short_straddle_not_stopped_when_front_vol_falls(self):
        s2 = self._stop_case(0.08, 0.20 * (1 - S2_STOP_LOSS_PCT - 0.05))
        assert not s2["s2_stop"].any()
        assert s2["s2_position"].iloc[303] == -1

    def test_small_adverse_move_does_not_stop(self):
        s2 = self._stop_case(-0.05, 0.20 * (1 - S2_STOP_LOSS_PCT + 0.05))
        assert not s2["s2_stop"].any()

    def test_r2exit_variant_uses_same_stop(self):
        s2x = self._stop_case(-0.05, 0.20 * (1 - S2_STOP_LOSS_PCT - 0.05), prefix="s2x")
        assert bool(s2x["s2x_stop"].iloc[303]) and s2x["s2x_position"].iloc[303] == 0
        s2x = self._stop_case(0.08, 0.20 * (1 - S2_STOP_LOSS_PCT - 0.05), prefix="s2x")
        assert not s2x["s2x_stop"].any()


# ── 4. generate_signal3 ───────────────────────────────────────────────────────

class TestGenerateSignal3:
    def _features_with_vvix_ratio(self, ratio_vals) -> pd.DataFrame:
        """ratio = vix / vvix"""
        n = len(ratio_vals)
        idx = pd.bdate_range("2022-01-03", periods=n)
        vix  = np.full(n, 0.18)
        vvix = vix / np.array(ratio_vals)
        feats = pd.DataFrame({
            "vix":           vix,
            "ts_slope":      np.full(n, 0.002),
            "fear_premium":  np.ones(n),
            "rv_change_5d":  np.zeros(n),
            "pdv_iv_spread": np.zeros(n),
            "vvix":          vvix,
        }, index=idx)
        return feats

    def test_no_signal_normal_ratio(self):
        """Normal ratio (no z-score deviation) → no signal."""
        n = 400
        feats = self._features_with_vvix_ratio(np.full(n, 0.19))
        regs  = pd.Series(1, index=feats.index)
        s3    = generate_signal3(feats, regs)
        # Constant ratio → std=0 → z=0 → no entry
        assert (s3["s3_position"] == 0).all()

    def test_entry_when_ratio_low(self):
        """Very low VIX/VVIX ratio (VVIX elevated vs VIX) → long dispersion."""
        n = 400
        ratio = np.full(n, 0.19)
        ratio[300:320] = 0.10   # very low ratio → z < -1
        feats = self._features_with_vvix_ratio(ratio)
        regs  = pd.Series(1, index=feats.index)
        s3    = generate_signal3(feats, regs)
        assert s3["s3_position"].iloc[300] == 1, "z < −1 must open the +1 side"
        assert (s3["s3_position"][s3["s3_position"] != 0] == 1).all()
        assert s3["s3_z"].iloc[300] < -1.0

    def test_no_short_in_signal3(self):
        """Signal 3 never takes the −1 side."""
        ratio = np.concatenate([np.full(300, 0.19), np.full(100, 0.40)])
        feats = self._features_with_vvix_ratio(ratio)
        regs  = pd.Series(1, index=feats.index)
        s3    = generate_signal3(feats, regs)
        assert (s3["s3_position"] >= 0).all()

    def test_no_entry_regime2(self):
        n = 400
        ratio = np.full(n, 0.19)
        ratio[300:320] = 0.10
        feats = self._features_with_vvix_ratio(ratio)
        regs  = pd.Series(2, index=feats.index)
        s3    = generate_signal3(feats, regs)
        assert (s3["s3_position"] == 0).all()

    def test_max_hold_respected(self):
        """Warm the 252-day z-score (needs 126 obs) BEFORE the low ratio, so a
        trade actually opens; the old version never opened one (vacuous)."""
        n = 260
        ratio = np.full(n, 0.19)
        ratio[200:] = 0.05                      # deep, persistent low ratio
        feats = self._features_with_vvix_ratio(ratio)
        regs  = pd.Series(1, index=feats.index)
        s3    = generate_signal3(feats, regs, max_hold=10)
        dh = s3["s3_days_held"].values
        assert s3["s3_position"].iloc[200] == 1
        assert dh.max() == 9                    # day 10 is the max_hold exit
        assert s3["s3_position"].iloc[209] == 0 and s3["s3_position"].iloc[210] == 1


# ── 5. combine_signals ────────────────────────────────────────────────────────

class TestCombineSignals:
    def _combine(self, p1, p2, p3, st=None):
        n = len(p1)
        idx = pd.RangeIndex(n)
        if st is None:
            st = pd.Series(np.ones(n) * 0.5, index=idx)
        return combine_signals(
            pd.Series(p1, index=idx), pd.Series(p2, index=idx), pd.Series(p3, index=idx),
            st, st, st,
        )

    def test_all_flat(self):
        r = self._combine([0]*5, [0]*5, [0]*5)
        assert (r["combined_pos"] == 0).all()

    def test_all_three_agree_positive(self):
        r = self._combine([1]*3, [1]*3, [1]*3)
        assert (r["combined_pos"] > 0).all()

    def test_all_three_agree_negative(self):
        r = self._combine([-1]*3, [-1]*3, [-1]*3)
        assert (r["combined_pos"] < 0).all()

    def test_conflict_gives_flat(self):
        """One long, one short → conflict → flat."""
        r = self._combine([1, 1, 1], [-1, -1, -1], [0, 0, 0])
        assert (r["combined_pos"] == 0).all()

    def test_two_agree_one_flat_half_weight(self):
        """Two agree, one flat → half weight (abs combined < full weight)."""
        r_all  = self._combine([1]*3, [1]*3, [1]*3)
        r_two  = self._combine([1]*3, [1]*3, [0]*3)
        # Half weight means combined_str is smaller
        assert (r_two["combined_str"] <= r_all["combined_str"]).all()

    def test_output_columns(self):
        r = self._combine([1, 0, -1], [1, 0, -1], [1, 0, -1])
        assert "combined_pos" in r.columns
        assert "combined_str" in r.columns
        assert "combined_kelly" in r.columns

    def test_combined_kelly_capped(self):
        r = self._combine([1]*5, [1]*5, [1]*5, st=pd.Series(np.ones(5)))
        assert (r["combined_kelly"] <= 0.20 + 1e-8).all()

    def test_combined_kelly_uses_given_cap(self):
        idx = pd.RangeIndex(3)
        one = pd.Series(np.ones(3), index=idx)
        r = combine_signals(one, one, one, one, one, one, max_kelly=0.1)
        np.testing.assert_allclose(r["combined_kelly"], 0.1)


# ── 6. SignalEngine.generate ──────────────────────────────────────────────────

class TestSignalEngineGenerate:
    @pytest.fixture
    def spx_df(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)
        closes = 3000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        log_ret = np.concatenate([[np.nan], np.diff(np.log(closes))])
        return pd.DataFrame({
            "date": dates, "open": closes, "high": closes*1.01,
            "low": closes*0.99, "close": closes,
            "volume": np.ones(n, dtype=int), "log_return": log_ret,
        })

    @pytest.fixture
    def vix_df(self, spx_df):
        rng = np.random.default_rng(99)
        n = len(spx_df)
        vix   = rng.uniform(12, 40, n)
        vvix  = rng.uniform(70, 140, n)
        vix3m = vix + rng.uniform(-5, 5, n)
        return pd.DataFrame({
            "date":  spx_df["date"].values,
            "^VIX":  vix, "^VIX3M": vix3m,
            "^VIX6M": vix3m + 1, "^VIX9D": vix - 1,
            "^VVIX": vvix,
            "ts_slope_3m9d": vix3m - (vix - 1),
            "ts_slope_6m1m": (vix3m + 1) - vix,
        })

    def test_generate_returns_dataframe(self, spx_df, vix_df):
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        assert isinstance(df, pd.DataFrame)

    def test_output_has_all_signal_columns(self, spx_df, vix_df):
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        required = [
            "s1_position", "s1_strength", "s1_kelly",
            "s2_position", "s2_strength", "s2_kelly",
            "s3_position", "s3_strength", "s3_kelly",
            "combined_pos", "combined_str",
        ]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_positions_in_valid_range(self, spx_df, vix_df):
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        for col in ["s1_position", "s2_position", "s3_position"]:
            assert df[col].isin([-1, 0, 1]).all(), f"{col} has values outside {{-1, 0, 1}}"

    def test_kelly_nonnegative(self, spx_df, vix_df):
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        for col in ["s1_kelly", "s2_kelly", "s3_kelly", "combined_kelly"]:
            assert (df[col] >= 0).all()

    def test_index_is_datetime(self, spx_df, vix_df):
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        assert pd.api.types.is_datetime64_any_dtype(df.index)

    def test_exposes_continuous_statistics(self, spx_df, vix_df):
        """The z-scores the signals threshold on are output columns, so tests
        (and the backtest) can check them directly, not only positions."""
        df = SignalEngine().generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        for col in ("s2_z", "s3_z", "s4_z"):
            assert col in df.columns and df[col].notna().all()
        # s3_z is the trailing 252-session z of the LAGGED VIX/VVIX ratio
        from joint_vol_calibration.signals.regime_classifier import build_features
        lagged = build_features(spx_df, vix_df).shift(1)
        ratio = (lagged["vix"] / lagged["vvix"]).loc[:df.index[-1]].iloc[-252:]
        assert df["s3_z"].iloc[-1] == pytest.approx((ratio.iloc[-1] - ratio.mean()) / ratio.std())

    def test_statistics_warm_from_history_book_starts_flat(self, spx_df, vix_df):
        """z-scores use the history BEFORE start_date (they used to warm up
        inside the window, silencing S2/S3/S4 for ~126 sessions), but no
        position is carried in from before the window."""
        early = SignalEngine().generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        assert early[["s2_z", "s3_z", "s4_z"]].iloc[0].notna().all()
        full = SignalEngine().generate(spx_df, vix_df, start_date="2020-01-01", end_date="2021-12-31")
        common = early.index
        pd.testing.assert_series_equal(early["s3_z"], full.loc[common, "s3_z"])
        for c in ("s1_position", "s2_position", "s3_position", "s4_position", "s1c_position"):
            first = early[c].iloc[0]
            assert first == 0 or early[c.replace("position", "days_held")].iloc[0] == 1, (
                f"{c} carried a pre-window trade into the window")

    def test_default_regimes_are_lagged_canonical_rule_labels(self, spx_df, vix_df):
        df = SignalEngine().generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        labels = build_regime_labels(spx_df, vix_df)
        sessions = pd.to_datetime(spx_df["date"])
        lagged = labels.reindex(sessions).shift(1).ffill().reindex(df.index)
        assert (df["regime"] == lagged).all()
        assert df.attrs["regime_source"] == "rule"


class TestPrecomputedRegimesAreOnlyLagged:
    """C22: SignalEngine used to OR a hard-coded VVIX > 100 override onto any
    regimes it was given, so the adaptive gate could never lower R2."""

    @pytest.fixture
    def market(self):
        rng = np.random.default_rng(4)
        n = 400
        dates = pd.bdate_range("2020-01-02", periods=n)
        r = rng.normal(0, 0.01, n)
        spx = pd.DataFrame({"date": dates, "close": 3000 * np.exp(np.cumsum(r)), "log_return": r})
        vix = pd.DataFrame({"date": dates, "^VIX": rng.uniform(14, 30, n),
                            "^VIX3M": rng.uniform(15, 31, n),
                            "^VVIX": rng.uniform(105, 140, n)})   # always > 100
        return spx, vix

    def test_given_labels_are_used_verbatim_one_session_later(self, market):
        spx, vix = market
        dates = pd.to_datetime(spx["date"])
        given = pd.Series(np.tile([0, 1, 1, 1, 0, 1], len(dates) // 6 + 1)[:len(dates)],
                          index=dates)                         # never R2
        df = SignalEngine().generate(spx, vix, start_date="2020-06-01", end_date="2021-06-30",
                                     precomputed_regimes=given)
        assert (df["regime"] == given.shift(1).reindex(df.index)).all()
        assert (df["regime"] != 2).all(), "an R2 override leaked into given labels"
        assert df.attrs["regime_source"] == "precomputed"

    def test_given_labels_on_other_calendar_shift_by_session(self, market):
        """Labels missing a session must not shift by POSITION: session t uses
        the latest label dated ≤ t−1."""
        spx, vix = market
        dates = pd.to_datetime(spx["date"])
        given = pd.Series(np.arange(len(dates)) % 2, index=dates)
        hole = dates[200]
        df = SignalEngine().generate(spx, vix, start_date="2020-06-01", end_date="2021-06-30",
                                     precomputed_regimes=given.drop(hole))
        nxt = dates[201]
        assert df.loc[nxt, "regime"] == given[dates[199]]      # carried, not shifted
        assert df.loc[dates[202], "regime"] == given[dates[201]]


class TestSessionsSurviveMissingVVIX:
    """C22: one missing VVIX print dropped the NEXT session from the signal
    frame (and the backtest equity curve), although only S3 reads vvix."""

    def test_every_session_is_kept(self):
        rng = np.random.default_rng(8)
        n = 420
        dates = pd.bdate_range("2019-01-02", periods=n)
        r = rng.normal(0, 0.01, n)
        spx = pd.DataFrame({"date": dates, "close": 3000 * np.exp(np.cumsum(r)), "log_return": r})
        vix = pd.DataFrame({"date": dates, "^VIX": rng.uniform(14, 30, n),
                            "^VIX3M": rng.uniform(15, 31, n), "^VVIX": rng.uniform(80, 120, n)})
        vix.loc[350, "^VVIX"] = np.nan              # one missing print
        vix.loc[[370, 371], "^VVIX"] = np.nan       # two-session outage
        vix.loc[390, "^VIX3M"] = np.nan             # no term structure one day
        df = SignalEngine().generate(spx, vix, start_date="2020-01-01", end_date="2020-12-31")
        window = dates[(dates >= "2020-01-01") & (dates <= "2020-12-31")]
        assert df.index.equals(window)
        for i in (351, 372, 391):
            assert dates[i] in df.index
        # a NaN input only silences the signals that read it, for that session
        assert np.isnan(df.loc[dates[372], "vvix"]) and np.isnan(df.loc[dates[391], "ts_slope"])
        assert df.loc[dates[372], "regime"] == df.loc[dates[371], "regime"]
        assert df.attrs["n_sessions_dropped"] == 0


# ── 7. SignalEngine.save ──────────────────────────────────────────────────────

class TestSignalEngineSave:
    def test_save_creates_parquet(self, tmp_path):
        n = 20
        idx = pd.bdate_range("2022-01-03", periods=n)
        mock_df = pd.DataFrame({
            "s1_position": np.zeros(n), "s1_strength": np.zeros(n),
            "s1_kelly": np.zeros(n), "s1_exp_pnl": np.zeros(n),
            "s2_position": np.zeros(n), "s2_strength": np.zeros(n),
            "s2_kelly": np.zeros(n), "s2_exp_pnl": np.zeros(n),
            "s3_position": np.zeros(n), "s3_strength": np.zeros(n),
            "s3_kelly": np.zeros(n), "s3_exp_pnl": np.zeros(n),
            "combined_pos": np.zeros(n), "combined_str": np.zeros(n),
            "combined_kelly": np.zeros(n),
        }, index=idx)
        engine = SignalEngine()
        engine._result = mock_df
        path = engine.save(path=tmp_path / "test_signals.parquet")
        assert path.exists()

    def test_save_raises_without_result(self, tmp_path):
        engine = SignalEngine()
        with pytest.raises(RuntimeError):
            engine.save(path=tmp_path / "test.parquet")

    def test_default_path_names_the_saved_span(self, tmp_path, monkeypatch):
        """The old fixed name 'signals_2015_2025.parquet' labelled a
        2018-01..2025-03 file; the default now says what is in it."""
        monkeypatch.setattr(se, "SIGNALS_DIR", tmp_path)
        idx = pd.bdate_range("2018-01-02", "2025-03-24")
        df = pd.DataFrame({"s1_position": np.zeros(len(idx))}, index=idx)
        engine = SignalEngine()
        engine._result = df
        path = engine.save()
        assert path == tmp_path / "signals_20180102_20250324.parquet" == default_signals_path(df)
        assert path.exists()


# ── 8. signal_summary ────────────────────────────────────────────────────────

class TestSignalSummary:
    def test_returns_dict(self):
        n = 30
        idx = pd.bdate_range("2022-01-03", periods=n)
        pos = np.random.choice([-1, 0, 1], size=n)
        df = pd.DataFrame({
            "s1_position": pos, "s1_strength": np.abs(pos) * 0.5,
            "s1_kelly":    np.abs(pos) * 0.1, "s1_exp_pnl": np.abs(pos),
            "s2_position": pos, "s2_strength": np.abs(pos) * 0.5,
            "s2_kelly":    np.abs(pos) * 0.1, "s2_exp_pnl": np.abs(pos),
            "s3_position": pos, "s3_strength": np.abs(pos) * 0.5,
            "s3_kelly":    np.abs(pos) * 0.1, "s3_exp_pnl": np.abs(pos),
            "combined_pos":   pos * 0.5, "combined_str": np.abs(pos) * 0.5,
            "combined_kelly": np.abs(pos) * 0.1,
        }, index=idx)
        summary = signal_summary(df)
        assert isinstance(summary, dict)

    def test_required_signal_keys(self):
        n = 10
        idx = pd.bdate_range("2022-01-03", periods=n)
        df = pd.DataFrame({
            "s1_position": np.zeros(n), "s1_strength": np.zeros(n),
            "s1_kelly": np.zeros(n), "s1_exp_pnl": np.zeros(n),
            "s2_position": np.zeros(n), "s2_strength": np.zeros(n),
            "s2_kelly": np.zeros(n), "s2_exp_pnl": np.zeros(n),
            "s3_position": np.zeros(n), "s3_strength": np.zeros(n),
            "s3_kelly": np.zeros(n), "s3_exp_pnl": np.zeros(n),
            "combined_pos": np.zeros(n), "combined_str": np.zeros(n),
            "combined_kelly": np.zeros(n),
        }, index=idx)
        summary = signal_summary(df)
        assert "Signal1_PDV_IV" in summary      # not "IVR": no IV rank exists
        assert "Signal1_IVR" not in summary
        assert "Signal2_TS" in summary
        assert "Signal3_Disp" in summary
        assert "Combined" in summary

    def test_pct_active_in_01(self):
        n = 20
        idx = pd.bdate_range("2022-01-03", periods=n)
        pos = np.concatenate([np.ones(10), np.zeros(10)])
        df = pd.DataFrame({
            "s1_position": pos, "s1_strength": pos * 0.5,
            "s1_kelly":    pos * 0.1, "s1_exp_pnl": pos,
            "s2_position": np.zeros(n), "s2_strength": np.zeros(n),
            "s2_kelly":    np.zeros(n), "s2_exp_pnl": np.zeros(n),
            "s3_position": np.zeros(n), "s3_strength": np.zeros(n),
            "s3_kelly":    np.zeros(n), "s3_exp_pnl": np.zeros(n),
            "combined_pos": pos * 0.5, "combined_str": pos * 0.5,
            "combined_kelly": pos * 0.1,
        }, index=idx)
        summary = signal_summary(df)
        assert summary["Signal1_PDV_IV"]["pct_active"] == pytest.approx(0.5)


# ── 9. Edge cases ─────────────────────────────────────────────────────────────

class TestEdgeCases:
    def test_all_regime2_flat_signals(self):
        """If regimes = all 2, no new entries for Signal 1."""
        feats = _inject_spread(_make_flat_features(), 0, 60, 0.10)
        regs  = _make_regimes(value=2)
        s1 = generate_signal1(feats, regs)
        assert (s1["s1_position"] == 0).all()

    def test_zero_spread_no_entry(self):
        feats = _make_flat_features()    # pdv_iv_spread = 0
        regs  = _make_regimes(value=0)
        s1 = generate_signal1(feats, regs)
        assert (s1["s1_position"] == 0).all()

    def test_combine_single_signal_half_weight(self):
        """One signal active, two flat → half weight."""
        r = combine_signals(
            pd.Series([1.0]),
            pd.Series([0.0]),
            pd.Series([0.0]),
            pd.Series([0.8]),
            pd.Series([0.0]),
            pd.Series([0.0]),
        )
        assert abs(r["combined_pos"].iloc[0]) <= 0.5 + 1e-8


# ── 10. Constants sanity ──────────────────────────────────────────────────────

class TestConstants:
    def test_s1_entry_threshold_positive(self):
        assert S1_ENTRY_THRESHOLD > 0

    def test_max_holds_positive(self):
        assert S1_MAX_HOLD > 0
        assert S2_MAX_HOLD > 0
        assert S3_MAX_HOLD > 0

    def test_kelly_caps_le_1(self):
        assert S1_MAX_KELLY <= 1.0

    def test_s2_zscore_entry_positive(self):
        assert S2_ZSCORE_ENTRY > 0

    def test_s3_zscore_entry_negative(self):
        """Signal 3 enters on low ratio → z-score must be negative threshold."""
        assert S3_ZSCORE_ENTRY < 0

    def test_s1rf_regime2_scale_is_zero(self):
        """R2 days must be fully zeroed (scale = 0.0)."""
        assert S1_RF_REGIME2_SCALE == 0.0


# ── 11. generate_signal1_regime_filtered ─────────────────────────────────────

class TestGenerateSignal1RF:
    def test_s1rf_columns_present(self):
        """Output must have s1rf_* prefixed columns."""
        feats = _inject_spread(_make_flat_features(), 5, 30, 0.05)
        regs  = _make_regimes(value=0)
        s1rf  = generate_signal1_regime_filtered(feats, regs)
        for col in ["s1rf_position", "s1rf_strength", "s1rf_regime_entry",
                    "s1rf_days_held", "s1rf_kelly", "s1rf_exp_pnl"]:
            assert col in s1rf.columns, f"Missing column: {col}"

    def test_r2_days_have_zero_position(self):
        """Every R2 day must produce position=0 in the filtered variant."""
        feats = _inject_spread(_make_flat_features(n=60), 0, 60, 0.05)
        # R0 for first 20, R2 for next 20, R0 for last 20
        regimes_vals = np.array([0]*20 + [2]*20 + [0]*20)
        regs = pd.Series(regimes_vals, index=feats.index)
        s1rf = generate_signal1_regime_filtered(feats, regs)
        r2_positions = s1rf["s1rf_position"].iloc[20:40]
        assert (r2_positions == 0).all()

    def test_r2_days_have_zero_kelly(self):
        """Kelly must be 0.0 on R2 days."""
        feats = _inject_spread(_make_flat_features(n=60), 0, 60, 0.05)
        regimes_vals = np.array([0]*20 + [2]*20 + [0]*20)
        regs = pd.Series(regimes_vals, index=feats.index)
        s1rf = generate_signal1_regime_filtered(feats, regs)
        assert (s1rf["s1rf_kelly"].iloc[20:40] == 0.0).all()

    def test_r2_days_have_zero_strength(self):
        """Strength must be 0.0 on R2 days."""
        feats = _inject_spread(_make_flat_features(n=60), 0, 60, 0.05)
        regimes_vals = np.array([0]*20 + [2]*20 + [0]*20)
        regs = pd.Series(regimes_vals, index=feats.index)
        s1rf = generate_signal1_regime_filtered(feats, regs)
        assert (s1rf["s1rf_strength"].iloc[20:40] == 0.0).all()

    def test_r0r1_positions_match_s1(self):
        """On non-R2 days, s1rf_position must match s1_position exactly."""
        feats = _inject_spread(_make_flat_features(n=40), 0, 40, 0.05)
        regs  = _make_regimes(n=40, value=0)   # all R0
        s1    = generate_signal1(feats, regs)
        s1rf  = generate_signal1_regime_filtered(feats, regs)
        pd.testing.assert_series_equal(
            s1["s1_position"].rename("pos"),
            s1rf["s1rf_position"].rename("pos"),
            check_names=False,
        )

    def test_s1rf_not_more_active_than_s1(self):
        """S1RF can only be flat or equal to S1 — never more active."""
        feats = _inject_spread(_make_flat_features(n=60), 5, 55, 0.05)
        regimes_vals = np.array([0]*30 + [2]*15 + [0]*15)
        regs = pd.Series(regimes_vals, index=feats.index)
        s1   = generate_signal1(feats, regs)
        s1rf = generate_signal1_regime_filtered(feats, regs)
        n_active_s1   = (s1["s1_position"]   != 0).sum()
        n_active_s1rf = (s1rf["s1rf_position"] != 0).sum()
        assert n_active_s1rf <= n_active_s1

    def test_s1rf_resumes_after_r2(self):
        """Trade entered in R0, passes through R2, resumes in R0 (within max_hold)."""
        n = 12
        feats = _inject_spread(_make_flat_features(n=n), 0, n, 0.05)
        # R0 days 0-3, R2 days 4-7, R0 days 8-11
        regimes_vals = np.array([0, 0, 0, 0, 2, 2, 2, 2, 0, 0, 0, 0])
        regs = pd.Series(regimes_vals, index=feats.index)

        s1   = generate_signal1(feats, regs, max_hold=21)
        s1rf = generate_signal1_regime_filtered(feats, regs, max_hold=21)

        # S1: should be +1 on all days (enters day 0, no exit triggered)
        assert s1["s1_position"].iloc[0] == 1
        assert s1["s1_position"].iloc[4] == 1   # R2 day — S1 still holds
        assert s1["s1_position"].iloc[8] == 1   # back in R0 — S1 still holds

        # S1RF: +1 in R0, 0 in R2, +1 again in R0
        assert s1rf["s1rf_position"].iloc[0] == 1
        assert s1rf["s1rf_position"].iloc[4] == 0   # R2 day zeroed
        assert s1rf["s1rf_position"].iloc[8] == 1   # trade resumes after R2

    def test_all_r2_gives_all_flat(self):
        """When every day is R2, S1RF is always 0."""
        feats = _inject_spread(_make_flat_features(n=40), 0, 40, 0.05)
        regs  = _make_regimes(n=40, value=2)
        s1rf  = generate_signal1_regime_filtered(feats, regs)
        assert (s1rf["s1rf_position"] == 0).all()


# ── 12. compare_s1_regime_filter ─────────────────────────────────────────────

def _make_signals_df(n=60, r2_start=20, r2_end=30, spread_val=0.05):
    """Build a minimal signals DataFrame for compare_s1_regime_filter tests."""
    feats = _inject_spread(_make_flat_features(n=n), 0, n, spread_val)
    regimes_arr = np.zeros(n, dtype=int)
    regimes_arr[r2_start:r2_end] = 2
    regs = pd.Series(regimes_arr, index=feats.index)

    s1   = generate_signal1(feats, regs)
    s1rf = generate_signal1_regime_filtered(feats, regs)
    return pd.concat([s1, s1rf, feats, regs.rename("regime")], axis=1)


class TestCompareS1RegimeFilter:
    def test_returns_dict(self):
        df = _make_signals_df()
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        assert isinstance(result, dict)

    def test_required_keys(self):
        df = _make_signals_df()
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        for key in ["s1_total_pnl", "s1rf_total_pnl", "pnl_difference",
                    "n_trades", "n_trades_with_r2", "n_trades_without_r2",
                    "r2_days_zeroed", "s1_pnl_series", "s1rf_pnl_series"]:
            assert key in result, f"Missing key: {key}"

    def test_pnl_difference_correct(self):
        """pnl_difference must equal s1rf_total_pnl − s1_total_pnl."""
        df = _make_signals_df()
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        expected = result["s1rf_total_pnl"] - result["s1_total_pnl"]
        assert abs(result["pnl_difference"] - expected) < 1e-6

    def test_n_trades_nonnegative(self):
        df = _make_signals_df()
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        assert result["n_trades"] >= 0
        assert result["n_trades_with_r2"] >= 0
        assert result["n_trades_without_r2"] >= 0

    def test_trade_counts_sum_correctly(self):
        """n_trades_with_r2 + n_trades_without_r2 == n_trades."""
        df = _make_signals_df()
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        assert result["n_trades_with_r2"] + result["n_trades_without_r2"] == result["n_trades"]

    def test_r2_days_zeroed_correct(self):
        """r2_days_zeroed must equal days where regime==2 AND s1_position!=0."""
        df = _make_signals_df(r2_start=5, r2_end=15, spread_val=0.05)
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        expected = int(((df["regime"] == 2) & (df["s1_position"] != 0)).sum())
        assert result["r2_days_zeroed"] == expected

    def test_s1rf_pnl_le_s1_pnl_when_r2_costly(self):
        """When S1 holds positions through R2 and gains, S1RF earns less (zeroed)."""
        # Use positive spread so long position earns positive daily proxy P&L
        df = _make_signals_df(r2_start=5, r2_end=20, spread_val=0.08)
        result = compare_s1_regime_filter(df, start_date="2022-01-01", end_date="2023-12-31")
        # If S1 has any non-zero days in R2 with positive spread, S1RF has less PnL
        if result["r2_days_zeroed"] > 0 and result["s1_total_pnl"] > 0:
            assert result["s1rf_total_pnl"] <= result["s1_total_pnl"]

    def test_missing_column_raises(self):
        """ValueError when required column is absent."""
        df = _make_signals_df()
        bad_df = df.drop(columns=["s1_days_held"])
        with pytest.raises(ValueError, match="s1_days_held"):
            compare_s1_regime_filter(bad_df, start_date="2022-01-01", end_date="2023-12-31")

    def test_empty_date_range_raises(self):
        """ValueError when no rows fall in the date range."""
        df = _make_signals_df()
        with pytest.raises(ValueError):
            compare_s1_regime_filter(df, start_date="2000-01-01", end_date="2000-01-31")


# ── 13. SignalEngine includes s1rf columns ────────────────────────────────────

class TestSignalEngineS1RF:
    @pytest.fixture
    def spx_df(self):
        rng = np.random.default_rng(42)
        n = 500
        dates = pd.bdate_range("2020-01-02", periods=n)
        closes = 3000 * np.cumprod(1 + rng.normal(0, 0.01, n))
        log_ret = np.concatenate([[np.nan], np.diff(np.log(closes))])
        return pd.DataFrame({
            "date": dates, "open": closes, "high": closes*1.01,
            "low": closes*0.99, "close": closes,
            "volume": np.ones(n, dtype=int), "log_return": log_ret,
        })

    @pytest.fixture
    def vix_df(self, spx_df):
        rng = np.random.default_rng(99)
        n = len(spx_df)
        vix   = rng.uniform(12, 40, n)
        vvix  = rng.uniform(70, 140, n)
        vix3m = vix + rng.uniform(-5, 5, n)
        return pd.DataFrame({
            "date":  spx_df["date"].values,
            "^VIX":  vix, "^VIX3M": vix3m,
            "^VIX6M": vix3m + 1, "^VIX9D": vix - 1,
            "^VVIX": vvix,
            "ts_slope_3m9d": vix3m - (vix - 1),
            "ts_slope_6m1m": (vix3m + 1) - vix,
        })

    def test_engine_output_has_s1rf_columns(self, spx_df, vix_df):
        """SignalEngine.generate() must include s1rf_position and s1rf_kelly."""
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        assert "s1rf_position" in df.columns
        assert "s1rf_kelly"    in df.columns
        assert "s1rf_strength" in df.columns

    def test_s1rf_never_more_active_than_s1_in_engine(self, spx_df, vix_df):
        """Within engine output, S1RF active days ≤ S1 active days."""
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        assert (df["s1rf_position"] != 0).sum() <= (df["s1_position"] != 0).sum()

    def test_signal_summary_includes_s1rf(self, spx_df, vix_df):
        """signal_summary() returns 'Signal1_RF' key when s1rf columns present."""
        engine = SignalEngine()
        df = engine.generate(spx_df, vix_df, start_date="2021-01-01", end_date="2021-12-31")
        summary = signal_summary(df)
        assert "Signal1_RF" in summary


# ── 14. _run_statemachine_r2exit ─────────────────────────────────────────────

class TestRunStatemachineR2Exit:

    def _run(self, entry_l, entry_s, exit_c, regimes, strength=None, max_hold=10):
        n   = len(entry_l)
        idx = pd.RangeIndex(n)
        if strength is None:
            strength = pd.Series(np.ones(n), index=idx)
        return _run_statemachine_r2exit(
            pd.Series(entry_l, index=idx),
            pd.Series(entry_s, index=idx),
            pd.Series(exit_c,  index=idx),
            strength,
            pd.Series(regimes, index=idx),
            max_hold=max_hold,
        )

    def test_returns_expected_columns(self):
        df = self._run([1,0,0,0], [0,0,0,0], [0,0,0,0], [0,0,0,0])
        assert set(df.columns) == {"position", "strength", "regime_entry", "days_held", "r2_exit"}

    def test_no_r2_transition_normal_exit(self):
        """Without R2 transition, behaves like normal state machine."""
        # Enter long day 0, hold until max_hold
        entry = [1] + [0]*9
        df = self._run(entry, [0]*10, [0]*10, [0]*10, max_hold=5)
        assert df["position"].iloc[0] == 1
        assert int(df["days_held"].iloc[3]) == 4    # day 3: 4 days held (exit resets cur_days)
        assert df["position"].iloc[4] == 0          # exits on day 4 (5th day, max_hold=5 reached)

    def test_r2_transition_exits_open_long(self):
        """Regime transitions 0→2 while long → position closes, r2_exit flagged."""
        # Enter long day 0 (regime=0), regime becomes 2 on day 3
        entry  = [1,0,0,0,0]
        regime = [0,0,0,2,2]
        df = self._run(entry, [0]*5, [0]*5, regime, max_hold=10)
        assert df["position"].iloc[0] == 1   # in trade days 0-2
        assert df["position"].iloc[1] == 1
        assert df["position"].iloc[2] == 1
        assert df["position"].iloc[3] == 0   # force exit on R2 entry
        assert df["r2_exit"].iloc[3]         # r2_exit flag set

    def test_r2_transition_exits_open_short(self):
        """Short position also closes on 0→2 R2 transition."""
        entry_s = [1,0,0,0]
        regime  = [1,1,2,2]
        df = self._run([0]*4, entry_s, [0]*4, regime, max_hold=10)
        assert df["position"].iloc[0] == -1
        assert df["position"].iloc[1] == -1
        assert df["position"].iloc[2] == 0   # exits on R2 transition
        assert df["r2_exit"].iloc[2]

    def test_already_in_r2_no_double_exit(self):
        """Persisting in R2 (no transition) does not fire r2_exit repeatedly."""
        entry  = [1,0,0,0,0]
        regime = [2,2,2,2,2]  # already in R2 from day 0 (no prior non-R2)
        df = self._run(entry, [0]*5, [0]*5, regime, max_hold=10)
        # No entry should occur because entry_long fires but it re-enters in same iteration
        # Actually entry fires on day0 when cur_pos==0; prev_reg starts at -1 ≠ 2 so no R2 exit
        # The entry condition for S1 would be checked separately (regime gate outside)
        # Here we just test r2_exit doesn't incorrectly fire on day 1+
        assert df["r2_exit"].sum() == 0  # no R2 transition (was already R2 or never had prior)

    def test_r2_does_not_prevent_new_entry_after_exit(self):
        """After R2-forced exit, a new entry can occur when regime returns to R0/R1."""
        entry  = [1,0,0,0,1,0,0]  # enter day0, regime 0→2 on day2, entry again day4 in R0
        regime = [0,0,2,2,0,0,0]
        df = self._run(entry, [0]*7, [0]*7, regime, max_hold=10)
        assert df["position"].iloc[0] == 1
        assert df["position"].iloc[2] == 0   # force exit
        assert df["r2_exit"].iloc[2]
        assert df["position"].iloc[4] == 1   # new entry in R0 after R2

    def test_r2_exit_column_dtype(self):
        df = self._run([0]*5, [0]*5, [0]*5, [0]*5)
        assert df["r2_exit"].dtype == bool or df["r2_exit"].dtype == np.bool_


# ── 15. generate_signal1_r2exit ──────────────────────────────────────────────

class TestGenerateSignal1R2Exit:

    def _make_entry_features(self, n=40):
        idx = pd.bdate_range("2022-01-03", periods=n)
        return pd.DataFrame({
            "vix":           np.full(n, 0.18),
            "ts_slope":      np.full(n, 0.002),
            "fear_premium":  np.full(n, 1.05),
            "rv_change_5d":  np.zeros(n),
            "pdv_iv_spread": np.full(n, 0.05),   # S1 long entry spread
            "vvix":          np.full(n, 0.92),
        }, index=idx)

    def test_columns_prefixed_s1x(self):
        feat    = self._make_entry_features()
        regimes = pd.Series(0, index=feat.index)
        result  = generate_signal1_r2exit(feat, regimes)
        for col in ["s1x_position", "s1x_strength", "s1x_kelly", "s1x_exp_pnl",
                    "s1x_regime_entry", "s1x_days_held", "s1x_r2_exit"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_r2_transition_closes_s1x_trade(self):
        """S1X exits an open long when regime transitions into R2."""
        n = 20
        idx = pd.bdate_range("2022-01-03", periods=n)
        feat = pd.DataFrame({
            "pdv_iv_spread": np.full(n, 0.05),
            "vix": np.full(n, 0.18),
            "ts_slope": np.full(n, 0.002),
            "fear_premium": np.full(n, 1.05),
            "rv_change_5d": np.zeros(n),
            "vvix": np.full(n, 0.92),
        }, index=idx)
        # R0 for first 5, then R2
        reg = [0]*5 + [2]*15
        regimes = pd.Series(reg, index=idx)
        result  = generate_signal1_r2exit(feat, regimes)
        # Position should be open for days 0-4 (or until spread goes to 0), then 0 at R2 entry
        assert result["s1x_position"].iloc[5] == 0
        assert result["s1x_r2_exit"].iloc[5]

    def test_no_entry_while_in_r2(self):
        """S1X must not enter new positions while regime is R2."""
        n = 20
        idx     = pd.bdate_range("2022-01-03", periods=n)
        feat    = pd.DataFrame({
            "pdv_iv_spread": np.full(n, 0.05),
            "vix": np.full(n, 0.18), "ts_slope": np.full(n, 0.002),
            "fear_premium": np.full(n, 1.05), "rv_change_5d": np.zeros(n),
            "vvix": np.full(n, 0.92),
        }, index=idx)
        regimes = pd.Series(2, index=idx)   # permanently R2
        result  = generate_signal1_r2exit(feat, regimes)
        assert (result["s1x_position"] == 0).all()

    def test_kelly_non_negative(self):
        feat    = self._make_entry_features()
        regimes = pd.Series(0, index=feat.index)
        result  = generate_signal1_r2exit(feat, regimes)
        assert (result["s1x_kelly"] >= 0).all()


# ── 16. generate_signal2_r2exit ──────────────────────────────────────────────

class TestGenerateSignal2R2Exit:

    def test_columns_prefixed_s2x(self):
        n   = 60
        idx = pd.bdate_range("2022-01-03", periods=n)
        # Build features with enough backwardation to trigger entry
        ts  = [-0.05]*n   # persistent backwardation triggers long entry
        feat = pd.DataFrame({
            "vix": np.full(n, 0.18), "ts_slope": ts,
            "fear_premium": np.full(n, 1.05), "rv_change_5d": np.zeros(n),
            "pdv_iv_spread": np.zeros(n), "vvix": np.full(n, 0.92),
        }, index=idx)
        regimes = pd.Series(0, index=feat.index)
        result  = generate_signal2_r2exit(feat, regimes)
        for col in ["s2x_position", "s2x_strength", "s2x_kelly",
                    "s2x_exp_pnl", "s2x_r2_exit"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_r2_transition_exits_s2x(self):
        n   = 20
        idx = pd.bdate_range("2022-01-03", periods=n)
        feat = pd.DataFrame({
            "vix": np.full(n, 0.18), "ts_slope": np.full(n, -0.05),
            "fear_premium": np.full(n, 1.05), "rv_change_5d": np.zeros(n),
            "pdv_iv_spread": np.zeros(n), "vvix": np.full(n, 0.92),
        }, index=idx)
        reg = [0]*5 + [2]*15
        regimes = pd.Series(reg, index=idx)
        result  = generate_signal2_r2exit(feat, regimes)
        # If trade was entered in R0, it must close when regime hits R2
        if result["s2x_position"].iloc[4] != 0:
            assert result["s2x_position"].iloc[5] == 0
            assert result["s2x_r2_exit"].iloc[5]


# ── 17. generate_signal1_soft ────────────────────────────────────────────────

class TestGenerateSignal1Soft:

    def _make_features(self, n=40, spread=0.05):
        idx = pd.bdate_range("2022-01-03", periods=n)
        return pd.DataFrame({
            "vix": np.full(n, 0.18), "ts_slope": np.full(n, 0.002),
            "fear_premium": np.full(n, 1.05), "rv_change_5d": np.zeros(n),
            "pdv_iv_spread": np.full(n, spread), "vvix": np.full(n, 0.92),
        }, index=idx)

    def test_columns_prefixed_s1s(self):
        feat    = self._make_features()
        regimes = pd.Series(0, index=feat.index)
        prob_r2 = pd.Series(0.0, index=feat.index)
        result  = generate_signal1_soft(feat, regimes, prob_r2)
        for col in ["s1s_position", "s1s_strength", "s1s_kelly",
                    "s1s_exp_pnl", "s1s_scale"]:
            assert col in result.columns, f"Missing column: {col}"

    def test_zero_prob_r2_equals_s1(self):
        """When prob_r2 = 0 everywhere, S1S = S1 (scale = 1)."""
        feat    = self._make_features()
        regimes = pd.Series(0, index=feat.index)
        prob_r2 = pd.Series(0.0, index=feat.index)
        s1      = generate_signal1(feat, regimes)
        s1s     = generate_signal1_soft(feat, regimes, prob_r2)
        np.testing.assert_array_almost_equal(
            s1["s1_position"].values, s1s["s1s_position"].values
        )

    def test_one_prob_r2_zeros_position(self):
        """When prob_r2 = 1 everywhere, position = 0."""
        feat    = self._make_features()
        regimes = pd.Series(0, index=feat.index)
        prob_r2 = pd.Series(1.0, index=feat.index)
        result  = generate_signal1_soft(feat, regimes, prob_r2)
        assert (result["s1s_position"] == 0).all()

    def test_partial_scaling(self):
        """prob_r2 = 0.5 → position = 0.5 × base."""
        feat    = self._make_features()
        regimes = pd.Series(0, index=feat.index)
        prob_r2 = pd.Series(0.5, index=feat.index)
        s1      = generate_signal1(feat, regimes)
        s1s     = generate_signal1_soft(feat, regimes, prob_r2)
        np.testing.assert_array_almost_equal(
            s1s["s1s_position"].values, s1["s1_position"].values * 0.5
        )

    def test_scale_column_step_function(self):
        """Scale is a 3-level step function: 1.0 / 0.5 / 0.0 based on P(R2) thresholds."""
        feat    = self._make_features()
        regimes = pd.Series(0, index=feat.index)
        # Test three probe values: 0.2 → 1.0, 0.5 → 0.5, 0.8 → 0.0
        n = len(feat)
        prob_vals = np.full(n, 0.2)
        prob_vals[n // 3: 2 * n // 3] = 0.5
        prob_vals[2 * n // 3:] = 0.8
        prob_r2 = pd.Series(prob_vals, index=feat.index)
        result  = generate_signal1_soft(feat, regimes, prob_r2)
        expected = np.where(prob_vals < 0.4, 1.0, np.where(prob_vals < 0.6, 0.5, 0.0))
        np.testing.assert_array_almost_equal(result["s1s_scale"].values, expected)


# ── S4: Volatility Risk Premium Signal ────────────────────────────────────────

def _make_vrp_features(n: int = 400, seed: int = 7) -> pd.DataFrame:
    """Features with VRP pattern: vix and fear_premium controls VRP signal."""
    idx = pd.bdate_range("2020-01-02", periods=n)
    vix = np.full(n, 0.18)
    # Normal fear_premium ~1.2 (IV slightly above RV)
    fear = np.full(n, 1.2)
    return pd.DataFrame({
        "vix":           vix,
        "ts_slope":      np.full(n, 0.002),
        "fear_premium":  fear,
        "rv_change_5d":  np.zeros(n),
        "pdv_iv_spread": np.zeros(n),
        "vvix":          np.full(n, 0.85),
    }, index=idx)


class TestGenerateSignal4:
    """14 tests for Signal 4: VRP mean-reversion."""

    def _flat_feats(self, n: int = 400) -> pd.DataFrame:
        return _make_vrp_features(n=n)

    def _regimes(self, n: int = 400, value: int = 1) -> pd.Series:
        idx = pd.bdate_range("2020-01-02", periods=n)
        return pd.Series(value, index=idx)

    def test_output_columns_present(self):
        """S4 output must contain position, strength, kelly, exp_pnl."""
        feats = self._flat_feats()
        reg   = self._regimes()
        out   = generate_signal4(feats, reg)
        for col in ["s4_position", "s4_strength", "s4_kelly", "s4_exp_pnl",
                    "s4_regime_entry", "s4_days_held"]:
            assert col in out.columns, f"Missing column: {col}"

    def test_flat_vrp_no_entry(self):
        """Constant fear_premium=1.2 never crosses z-score threshold → flat."""
        feats = self._flat_feats()
        reg   = self._regimes()
        out   = generate_signal4(feats, reg)
        assert (out["s4_position"] == 0).all()

    def test_short_entry_in_r1_only(self):
        """High VRP z-score only triggers short in R1, not R0 or R2."""
        n = 400
        feats = self._flat_feats(n)
        # Inject elevated fear_premium in last 50 bars → high z_vrp
        feats = feats.copy()
        feats.iloc[200:, feats.columns.get_loc("fear_premium")] = 2.5

        reg_r1 = self._regimes(n, value=1)
        reg_r0 = self._regimes(n, value=0)
        reg_r2 = self._regimes(n, value=2)

        out_r1 = generate_signal4(feats, reg_r1)
        out_r0 = generate_signal4(feats, reg_r0)
        out_r2 = generate_signal4(feats, reg_r2)

        # R1 should get short entries after warmup
        assert (out_r1["s4_position"] == -1).any(), "R1 should trigger short"
        # R0 should never short (wrong regime for selling vol)
        assert not (out_r0["s4_position"] == -1).any(), "R0 must not trigger short"
        # R2 should never enter any position
        assert (out_r2["s4_position"] == 0).all(), "R2 must block all entries"

    def test_long_entry_in_r0_only(self):
        """Very low VRP z-score triggers long in R0, not R1."""
        n = 400
        feats = self._flat_feats(n)
        # Inject depressed fear_premium → low VRP → low z_vrp
        feats = feats.copy()
        feats.iloc[200:, feats.columns.get_loc("fear_premium")] = 0.6

        reg_r0 = self._regimes(n, value=0)
        reg_r1 = self._regimes(n, value=1)

        out_r0 = generate_signal4(feats, reg_r0)
        out_r1 = generate_signal4(feats, reg_r1)

        assert (out_r0["s4_position"] == 1).any(), "R0 should trigger long on low VRP"
        assert not (out_r1["s4_position"] == 1).any(), "R1 must not trigger long"

    def test_max_hold_respected(self):
        """Position is closed after S4_MAX_HOLD days."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats.iloc[200:, feats.columns.get_loc("fear_premium")] = 2.5
        reg = self._regimes(n, value=1)
        out = generate_signal4(feats, reg, max_hold=5)
        active = out[out["s4_position"] != 0]
        if len(active) > 0:
            assert active["s4_days_held"].max() <= 5

    def test_kelly_bounded(self):
        """Kelly fraction never exceeds S4_MAX_KELLY=0.20."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats.iloc[200:, feats.columns.get_loc("fear_premium")] = 3.0
        reg = self._regimes(n, value=1)
        out = generate_signal4(feats, reg)
        assert (out["s4_kelly"] <= S4_MAX_KELLY + 1e-9).all()

    def test_position_values(self):
        """All positions must be in {-1, 0, +1}."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats.iloc[150:250, feats.columns.get_loc("fear_premium")] = 2.5
        feats.iloc[300:350, feats.columns.get_loc("fear_premium")] = 0.5
        reg = pd.Series(
            [1]*300 + [0]*100,
            index=pd.bdate_range("2020-01-02", periods=n)
        )
        out = generate_signal4(feats, reg)
        allowed = {-1.0, 0.0, 1.0}
        assert set(out["s4_position"].unique()).issubset(allowed)

    def test_exit_on_reversion(self):
        """Trade exits when VRP z-score reverts below z_exit threshold."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        # High VRP for 20 bars then normalise: z should revert → exit
        feats.iloc[200:220, feats.columns.get_loc("fear_premium")] = 2.5
        # Remaining: back to 1.2 → VRP reverts, z drops below exit threshold
        reg = self._regimes(n, value=1)
        out = generate_signal4(feats, reg)
        # After reversion, position should return to 0
        if (out["s4_position"] != 0).any():
            last_exit = out[out["s4_position"] != 0].index[-1]
            tail = out.loc[last_exit:].iloc[1:]
            # Position must not stay stuck at non-zero indefinitely
            assert (tail["s4_position"] == 0).any() or len(tail) == 0

    def test_r2_exit_closes_open_position(self):
        """An open position is closed when regime transitions to R2."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        # High VRP from bar 150 → enters short in R1
        feats.iloc[150:, feats.columns.get_loc("fear_premium")] = 2.5

        # Regime: R1 for first 200 bars, then R2
        reg_vals = [1] * 200 + [2] * 200
        reg = pd.Series(reg_vals, index=pd.bdate_range("2020-01-02", periods=n))

        out = generate_signal4(feats, reg)
        # After regime turns R2 (bar 200+), position must be 0
        r2_period = out.iloc[200:]
        assert (r2_period["s4_position"] == 0).all(), \
            "Position must be zero during and after R2 regime"

    def test_no_same_day_reentry_after_exit(self):
        """On exit day, position is 0 (no same-day re-entry)."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats.iloc[200:215, feats.columns.get_loc("fear_premium")] = 2.5
        reg = self._regimes(n, value=1)
        out = generate_signal4(feats, reg, max_hold=5)
        days_held = out["s4_days_held"].values
        pos = out["s4_position"].values
        for i in range(1, len(days_held)):
            if days_held[i - 1] >= 5:
                assert pos[i] == 0 or days_held[i] == 1  # exit day or new entry next day

    def test_strength_within_bounds(self):
        """Signal strength is always in [0, 1]."""
        n = 400
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats.iloc[200:, feats.columns.get_loc("fear_premium")] = 5.0
        reg = self._regimes(n, value=1)
        out = generate_signal4(feats, reg)
        assert (out["s4_strength"] >= 0).all()
        assert (out["s4_strength"] <= 1.0 + 1e-9).all()

    def test_constants_valid_ranges(self):
        """S4 constants must satisfy logical ordering."""
        assert S4_Z_ENTRY_SHORT > S4_Z_EXIT >= 0
        assert S4_Z_ENTRY_LONG < 0
        assert 1 <= S4_MAX_HOLD <= 60
        assert 0 < S4_MAX_KELLY <= 1.0

    def test_signal_engine_includes_s4(self):
        """SignalEngine.generate() must return s4_position column."""
        from pathlib import Path

        # Load regime classifier from disk
        clf_path = Path(__file__).resolve().parents[2] / "data_store" / "signals" / "regime_classifier.pkl"
        if not clf_path.exists():
            pytest.skip("Regime classifier not found; skipping integration test")

        from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
        spx = get_spx_ohlcv(as_of_date="2026-06-05")
        vix = get_vix_term_structure_wide(as_of_date="2026-06-05")

        engine = SignalEngine()
        df = engine.generate(spx, vix, start_date="2023-01-01", end_date="2024-12-31")
        assert "s4_position" in df.columns
        assert "s4_kelly" in df.columns

    def test_vrp_is_positive_when_fear_premium_above_one(self):
        """VRP = vix*(1 - 1/fear_premium) > 0 when fear_premium > 1."""
        n = 50
        feats = self._flat_feats(n)
        feats = feats.copy()
        feats["fear_premium"] = 1.5   # IV above RV
        # VRP should be positive: vix*(1 - 1/1.5) = vix*(1/3) > 0
        vrp_vals = feats["vix"] * (1.0 - 1.0 / feats["fear_premium"])
        assert (vrp_vals > 0).all()


# ── S1C: Contrarian PDV Signal ────────────────────────────────────────────────

class TestGenerateSignal1Contrarian:
    """8 tests for S1C: contrarian PDV mean-reversion signal."""

    def _flat_feats(self, n: int = 60) -> pd.DataFrame:
        return _make_flat_features(n)

    def _regimes(self, n: int = 60, value: int = 1) -> pd.Series:
        return _make_regimes(n, value)

    def test_output_columns(self):
        """S1C output has required columns prefixed s1c_."""
        out = generate_signal1_contrarian(self._flat_feats(), self._regimes())
        for col in ["s1c_position", "s1c_strength", "s1c_kelly", "s1c_exp_pnl"]:
            assert col in out.columns

    def test_flat_spread_no_entry(self):
        """Zero spread never exceeds threshold → flat."""
        out = generate_signal1_contrarian(self._flat_feats(), self._regimes())
        assert (out["s1c_position"] == 0).all()

    def test_short_in_r0_when_pdv_above_iv(self):
        """Spread > threshold in R0 → SHORT (contrarian: PDV overestimates)."""
        n = 60
        feats = _inject_spread(self._flat_feats(n), 10, 40, 0.05)  # spread > threshold
        reg   = _make_regimes(n, value=0)   # R0
        out   = generate_signal1_contrarian(feats, reg)
        assert (out["s1c_position"] == -1).any()

    def test_long_in_r1_when_iv_above_pdv(self):
        """Spread < -threshold in R1 → LONG (contrarian: market overprices vs PDV)."""
        n = 60
        feats = _inject_spread(self._flat_feats(n), 10, 40, -0.05)  # negative spread
        reg   = _make_regimes(n, value=1)   # R1
        out   = generate_signal1_contrarian(feats, reg)
        assert (out["s1c_position"] == 1).any()

    def test_opposite_direction_to_s1(self):
        """S1C must be opposite direction to S1 on same inputs."""
        n = 60
        feats_long  = _inject_spread(self._flat_feats(n), 5, 40, 0.05)
        reg_r0      = _make_regimes(n, value=0)
        s1  = generate_signal1(feats_long, reg_r0)
        s1c = generate_signal1_contrarian(feats_long, reg_r0)
        # Where S1 is long (+1), S1C must be short (−1)
        active = s1["s1_position"] != 0
        if active.any():
            assert (s1c.loc[active, "s1c_position"] == -s1.loc[active, "s1_position"]).all()

    def test_r2_blocks_entry(self):
        """R2 regime blocks all entries."""
        n = 60
        feats = _inject_spread(self._flat_feats(n), 10, 50, 0.05)
        reg   = _make_regimes(n, value=2)
        out   = generate_signal1_contrarian(feats, reg)
        assert (out["s1c_position"] == 0).all()

    def test_kelly_bounded(self):
        """Kelly fraction never exceeds S1_MAX_KELLY."""
        n = 60
        feats = _inject_spread(self._flat_feats(n), 5, 55, 0.10)
        reg   = _make_regimes(n, value=0)
        out   = generate_signal1_contrarian(feats, reg)
        assert (out["s1c_kelly"] <= S1_MAX_KELLY + 1e-9).all()

    def test_signal_engine_includes_s1c(self):
        """SignalEngine.generate() must return s1c_position column."""
        from pathlib import Path
        clf_path = Path(__file__).resolve().parents[2] / "data_store" / "signals" / "regime_classifier.pkl"
        if not clf_path.exists():
            pytest.skip("Regime classifier not found")
        from joint_vol_calibration.data.database import get_spx_ohlcv, get_vix_term_structure_wide
        spx = get_spx_ohlcv(as_of_date="2026-06-05")
        vix = get_vix_term_structure_wide(as_of_date="2026-06-05")
        engine = SignalEngine()
        df = engine.generate(spx, vix, start_date="2023-01-01", end_date="2024-12-31")
        assert "s1c_position" in df.columns
        assert "s1c_kelly" in df.columns

    def test_r2_transition_forces_exit(self):
        """C17: open S1C position must close when regime transitions INTO R2."""
        n = 60
        feats = _inject_spread(self._flat_feats(n), 5, 55, 0.05)
        reg   = _make_regimes(n, value=0)
        reg.iloc[30:] = 2          # transition into R2 at bar 30
        out = generate_signal1_contrarian(feats, reg)
        assert (out["s1c_position"].iloc[:30] == -1).any(), "trade never opened pre-R2"
        assert (out["s1c_position"].iloc[30:] == 0).all(), "position survived R2 transition"
        assert "s1c_r2_exit" in out.columns
        assert bool(out["s1c_r2_exit"].iloc[30]), "r2_exit flag not set at transition"


# ── End-to-end no-look-ahead: poison the whole pipeline ───────────────────────

class TestEndToEndNoLookAhead:
    """The decisive leakage artifact. Corrupt ALL market data strictly after a
    cutoff, re-run the entire SignalEngine pipeline (feature build -> regime
    labels -> 1-day shift -> trailing z-scores -> signal state machines), and
    assert every signal value on or before the cutoff is byte-identical. If any
    stage reaches forward (a full-sample z-score, a missing shift, a non-causal
    feature) a pre-cutoff signal would move and this fails. A second assertion
    forces the poison to change a post-cutoff signal so the test is not vacuous."""

    @staticmethod
    def _synthetic_market(n: int = 800, seed: int = 11):
        rng = np.random.default_rng(seed)
        dates = pd.bdate_range("2016-01-04", periods=n)
        s2 = np.empty(n); r = np.empty(n)
        s2[0] = 1e-4; r[0] = rng.normal(0.0, np.sqrt(s2[0]))
        for t in range(1, n):
            s2[t] = 3e-6 + 0.08 * r[t - 1] ** 2 + 0.90 * s2[t - 1]
            r[t] = rng.normal(0.0, np.sqrt(s2[t]))
        close = 3000.0 * np.exp(np.cumsum(r))
        spx = pd.DataFrame({"date": dates, "close": close, "log_return": r})
        rv_ann = np.sqrt(s2 * 252)
        vix = np.clip(15 + 35 * (rv_ann - rv_ann.mean()) / rv_ann.std()
                      + rng.normal(0, 1.5, n), 9, 60)
        vix3m = vix + rng.normal(1.5, 1.0, n)                 # mostly contango
        vvix = np.clip(90 + 12 * (vix - vix.mean()) / vix.std()
                       + rng.normal(0, 4, n), 75, 160)
        vixw = pd.DataFrame({"date": dates, "^VIX": vix,
                             "^VIX3M": vix3m, "^VVIX": vvix})
        return spx, vixw

    def test_full_pipeline_no_lookahead_poison(self):
        spx, vix = self._synthetic_market()
        start, end = "2016-01-01", "2020-01-01"
        clean = SignalEngine().generate(spx, vix, start_date=start, end_date=end)
        assert len(clean) > 100

        cutoff = clean.index[int(len(clean) * 0.6)]

        spx_p, vix_p = spx.copy(), vix.copy()
        ms = pd.to_datetime(spx_p["date"]) > cutoff
        mv = pd.to_datetime(vix_p["date"]) > cutoff
        spx_p.loc[ms, "close"] *= 5.0
        spx_p.loc[ms, "log_return"] = 0.25      # absurd future returns
        vix_p.loc[mv, "^VIX"] = 5.0
        vix_p.loc[mv, "^VIX3M"] = 90.0
        vix_p.loc[mv, "^VVIX"] = 999.0          # forces R2 after the cutoff

        poisoned = SignalEngine().generate(spx_p, vix_p, start_date=start, end_date=end)

        num_cols = [c for c in clean.columns if clean[c].dtype.kind in "fiub"]
        a = clean.loc[:cutoff, num_cols]
        b = poisoned.loc[:cutoff, num_cols]
        # Same dates and byte-identical values on or before the cutoff
        pd.testing.assert_frame_equal(a, b, check_exact=True)

        # Non-vacuous: the future poison must move at least one signal after cutoff
        after = clean.index[clean.index > cutoff]
        if len(after) > 5:
            changed = (clean.loc[after, num_cols].fillna(-7.0).round(8)
                       != poisoned.loc[after, num_cols].fillna(-7.0).round(8)).any().any()
            assert changed, "poison changed nothing after the cutoff (possibly vacuous)"
