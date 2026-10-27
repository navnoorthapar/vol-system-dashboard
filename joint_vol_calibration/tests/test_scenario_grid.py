"""
test_scenario_grid.py — the Risk page's spot × vol scenario grid (C23).

The grid revalues a delta-hedged 30-day straddle on the headline Heston fit's
smile under sticky-strike and sticky-moneyness dynamics
(joint_vol_calibration/greeks/scenario_grid.py). Checked here:

  * the dashboard contract (dashboard/ARTIFACTS.md §5; desk_artifacts loader);
  * an independent oracle: sticky-moneyness at a zero vol shock is the Heston
    revaluation at the shocked spot with v0 held (the model is homogeneous in
    (S, K)), priced here with the Fourier pricer directly;
  * small shocks reproduce the second-order Greek expansion;
  * the committed file is reproduced from its own source calibration, which is
    the same fit as the committed Greeks surface (no value is typed here: the
    daily bot regenerates both from each new headline fit).
"""
import math
import pickle

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration.config import DATA_DIR, ROOT_DIR
from joint_vol_calibration.greeks import scenario_grid as sg
from joint_vol_calibration.greeks.risk_monitor import calibration_carry, heston_otm_prices
from joint_vol_calibration.models.implied_vol import bs_gamma, bs_vanna, bs_vega, bs_vomma

_CAL = DATA_DIR / "calibrations"
_HEADLINE = "2026-10-16"


@pytest.fixture(scope="module")
def cal():
    with open(_CAL / f"joint_cal_{_HEADLINE}.pkl", "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def grid(cal):
    return sg.compute_scenario_grid(cal)


def _heston_straddle(cal, S):
    """Heston call + put at the grid's strike, maturity and carry, at spot S."""
    p = cal["params"]
    T = sg.TENOR_DAYS / 365.0
    r_fn, q_fn, _ = calibration_carry(cal)
    r, q = r_fn(T), q_fn(T)
    S0 = float(cal["S"])
    K = S0 * math.exp((r - q) * T)
    otm, F, ok = heston_otm_prices(S, np.array([K]), T, r, q, p)
    assert ok
    return 2.0 * float(otm[0]) + math.exp(-r * T) * abs(F - K)


def _desk_artifacts():
    """dashboard/desk_artifacts.py, the Risk page's loader (dashboard/ is not a package)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("desk_artifacts_under_test",
                                                  ROOT_DIR / "dashboard" / "desk_artifacts.py")
    assert spec is not None
    assert spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_grid_meets_the_dashboard_contract(grid, tmp_path):
    da = _desk_artifacts()
    n = len(sg.SPOT_SHOCKS_PCT) * len(sg.VOL_SHOCKS_VP) * len(sg.DYNAMICS)
    assert len(grid) == n
    assert not grid.duplicated(["spot_shock_pct", "vol_shock_vp", "dynamics"]).any()
    assert np.isfinite(grid["pnl"]).all()
    origin = grid[(grid.spot_shock_pct == 0) & (grid.vol_shock_vp == 0)]
    assert len(origin) == 2
    assert (origin["pnl"] == 0).all()
    path = sg.save_scenario_grid(grid, tmp_path / "greeks" / "scenario_grid.parquet")
    assert not list(path.parent.glob(".*.tmp"))              # written atomically
    out = da.load_scenario_grid(tmp_path)
    assert out["state"] == "ok"
    prov = out["provenance"]
    assert {"book", "as_of_date", "units"} <= set(prov)
    assert prov["as_of_date"] == _HEADLINE
    assert prov["n_vol_floored"] == 0
    back = sg.load_scenario_grid(path)
    pd.testing.assert_frame_equal(back, grid, check_like=True)


def test_sticky_moneyness_is_the_heston_revaluation(cal, grid):
    """At a zero vol shock, the sticky-moneyness smile is the Heston smile at
    the new spot (v0 held): the grid must equal a direct Fourier repricing."""
    prov = grid.attrs["provenance"]
    S0, d0 = prov["S"], prov["hedge_delta"]
    base = _heston_straddle(cal, S0)
    assert base == pytest.approx(prov["straddle_value"], rel=1e-10)
    g = grid[(grid.dynamics == "sticky_moneyness") & (grid.vol_shock_vp == 0)]
    for s, pnl in zip(g["spot_shock_pct"], g["pnl"]):
        S1 = S0 * (1 + s / 100)
        want = _heston_straddle(cal, S1) - base - d0 * (S1 - S0)
        assert pnl == pytest.approx(want, rel=1e-8, abs=1e-8 * base), s


def test_small_shocks_reproduce_the_second_order_expansion(cal):
    """Delta-hedged at inception, so P&L ≈ ½ΓΔS² + νΔσ + ½ vomma Δσ² + vanna ΔSΔσ
    for a sticky-strike book; the remainder is third order."""
    g = sg.compute_scenario_grid(cal, spot_shocks_pct=(-0.2, 0.2), vol_shocks_vp=(-0.2, 0.2),
                                 dynamics=("sticky_strike",))
    prov = g.attrs["provenance"]
    S0, K, T, r, q, sig = (prov[k] for k in ("S", "strike", "T", "r", "q", "iv_atm_forward"))
    gam = 2 * float(bs_gamma(S0, K, T, r, q, sig))
    veg = 2 * float(bs_vega(S0, K, T, r, q, sig))
    vom = 2 * float(bs_vomma(S0, K, T, r, q, sig))
    van = 2 * float(bs_vanna(S0, K, T, r, q, sig))
    for row in g.itertuples():
        dS, dv = S0 * row.spot_shock_pct / 100, row.vol_shock_vp / 100
        approx = 0.5 * gam * dS ** 2 + veg * dv + 0.5 * vom * dv ** 2 + van * dS * dv
        assert row.pnl == pytest.approx(approx, rel=2e-3), row


def test_dynamics_and_vol_shocks_behave_like_a_long_straddle(grid):
    piv = {d: grid[grid.dynamics == d].pivot_table(index="vol_shock_vp", columns="spot_shock_pct",
                                                   values="pnl", aggfunc="first")
           for d in sg.DYNAMICS}
    # no spot move: the two dynamics are the same smile
    np.testing.assert_allclose(piv["sticky_strike"][0.0], piv["sticky_moneyness"][0.0], rtol=0, atol=0)
    for p in piv.values():
        assert (np.diff(p.to_numpy(), axis=0) > 0).all()       # long vega at every spot
    ss = piv["sticky_strike"].loc[0.0]
    assert (ss.drop(0.0) > 0).all()                           # long gamma, delta-hedged
    # sticky moneyness: after a small drop the fixed strike sits higher on the
    # (negatively skewed) smile, so its vol is lower than under sticky strike
    iv = grid[(grid.vol_shock_vp == 0)].pivot_table(index="spot_shock_pct", columns="dynamics",
                                                    values="iv_strike", aggfunc="first")
    assert iv.loc[-1.0, "sticky_moneyness"] < iv.loc[-1.0, "sticky_strike"]
    assert iv.loc[1.0, "sticky_moneyness"] > iv.loc[1.0, "sticky_strike"]


def test_shocked_vols_are_floored_and_counted(cal):
    g = sg.compute_scenario_grid(cal, spot_shocks_pct=(0.0,), vol_shocks_vp=(-40.0, 0.0),
                                 dynamics=("sticky_strike",))
    assert g.attrs["provenance"]["n_vol_floored"] == 1
    row = g[g.vol_shock_vp == -40.0].iloc[0]
    assert row.vol_floored
    assert row.iv_strike == sg.VOL_FLOOR
    with pytest.raises(ValueError, match="unknown dynamics"):
        sg.compute_scenario_grid(cal, dynamics=("sticky_delta_ish",))


def test_cli_writes_the_grid(tmp_path, capsys):
    out = tmp_path / "grid.parquet"
    assert sg.main(["--calibration", str(_CAL / f"joint_cal_{_HEADLINE}.pkl"),
                    "--out", str(out)]) == 0
    df = sg.load_scenario_grid(out)
    assert df.attrs["provenance"]["source"]["calibration_file"] == (
        f"data_store/calibrations/joint_cal_{_HEADLINE}.pkl")
    assert "108 cells" in capsys.readouterr().out


def test_committed_grid_reproduces_from_its_source_fit():
    """The shipped grid is compute_scenario_grid of the calibration it names,
    and that is the fit the shipped Greeks surface was built from."""
    from joint_vol_calibration.greeks.risk_monitor import load_greeks_surface
    path = DATA_DIR / "greeks" / "scenario_grid.parquet"
    if not path.exists():
        pytest.skip("scenario_grid.parquet not generated")
    shipped = sg.load_scenario_grid(path)
    prov = shipped.attrs["provenance"]
    src = ROOT_DIR / prov["source"]["calibration_file"]
    with open(src, "rb") as f:
        cal = pickle.load(f)
    again = sg.compute_scenario_grid(cal)
    keys = ["dynamics", "spot_shock_pct", "vol_shock_vp"]
    a = shipped.sort_values(keys).reset_index(drop=True)
    b = again.sort_values(keys).reset_index(drop=True)
    pd.testing.assert_frame_equal(a[keys], b[keys])
    np.testing.assert_allclose(a["pnl"], b["pnl"], rtol=1e-10, atol=1e-10)
    gprov = load_greeks_surface(DATA_DIR / "greeks" / "greeks_surface.parquet").attrs["provenance"]
    assert gprov["source"]["calibration_file"] == prov["source"]["calibration_file"]
    assert gprov["as_of_date"] == prov["as_of_date"]
    assert gprov["params"] == pytest.approx(prov["params"], rel=0, abs=0)
