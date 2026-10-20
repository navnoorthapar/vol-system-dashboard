"""The committed trials ledger parses, is internally consistent, and its
reproducible families match the committed artifacts they describe."""

import pandas as pd
import pytest

from joint_vol_calibration.stats import trials

pytest.importorskip("yaml")


@pytest.fixture(scope="module")
def led():
    return trials.load()


def test_counts_by_round(led):
    c = trials.count_trials(led)
    assert c["n_trials"] == sum(c["by_round"].values())
    c16 = trials.count_trials(led, from_round="C16")
    assert set(c16["by_round"]) == {"C16", "C17", "C18", "C19", "C20", "C21", "C22"}
    assert c16["n_trials"] < c["n_trials"]
    assert c16["by_round"]["C20"] == 0
    # C22: 15 bridge rows + 5 extra β rows, six sleeves each, + S3 direction (2) + pseudo-OOS (1)
    assert c16["by_round"]["C22"] == 15 * 6 + 5 * 6 + 2 + 1
    up = trials.count_trials(led, include_evaluations=True)
    assert up["n_trials_upper"] > up["n_trials"]


def test_reproducible_families_match_the_committed_artifacts(led):
    from joint_vol_calibration.backtest.backtest_engine import BACKTEST_DIR

    bridge = pd.read_csv(BACKTEST_DIR / "c22_bridge.csv")
    beta = pd.read_csv(BACKTEST_DIR / "beta_sensitivity.csv")
    fams = {f["name"]: f for r in led["rounds"] for f in r.get("families") or []}
    assert fams["c22_bridge"]["configurations"] == len(bridge) == led["reproducible"]["c22_bridge"]["rows"]
    betas = led["reproducible"]["beta_grid"]["betas"]
    assert sorted(betas) == sorted(beta["beta"].round(2).tolist())
    assert fams["beta_grid"]["configurations"] == len(betas) - 1          # 0.82 is bridge row 14


def test_validation_rejects_a_malformed_ledger(tmp_path):
    p = tmp_path / "bad.yaml"
    p.write_text("schema: trials_ledger/1\nrounds:\n  - id: C1\n    families:\n      - name: x\n"
                 "        sleeves: [s1]\n")
    with pytest.raises(ValueError, match="lacks"):
        trials.load(p)
    p.write_text("schema: other\nrounds: []\n")
    with pytest.raises(ValueError):
        trials.load(p)
