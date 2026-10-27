"""
capture_artifact_fixtures.py — refresh the dashboard tests' samples of the C23
artifacts from a data_store that holds the producers' real output.

The dashboard tests (test_dashboard.py, "C23 artifacts") render small, real
samples of what each producer writes, so a schema change on either side fails
in CI instead of on the site. This script trims them to a test-sized subset:

  * surface/<session>/residuals.parquet: 3 expiries (the first, the first
    >= 28 days, the first >= 85 days), every 4th quote, every model; option
    prices (bid / ask, repair_dprice) are dropped, the implied vols kept;
  * scorecard.json / surface.json / vix_replication.json: the fields the
    pages read, restricted to those expiries; surface/index.json: two sessions;
  * backtest/inference/{summary,benchmarks,forecast_eval}.json: the fields
    the Research page reads;
  * forward/ledger_*.json: copied as written by forward_ledger.write_ledger.

Usage (build-time only; never run by the site or the bot):
    python dashboard/capture_artifact_fixtures.py SRC_DATA_STORE \
        joint_vol_calibration/tests/fixtures/dashboard FORWARD_LEDGER_DIR

  * calibrations/diagnostics/<session>.json: the identifiability, multi-start,
    sigma-profile, SPX-only and joint rough-Heston blocks the Calibration page
    reads (per-run lists and timings dropped); diagnostics/summary.json: every
    session's row, restricted to the fields the pages read
    (desk_artifacts.DIAG_SUMMARY_FIELDS).

With --diagnostics-only, only the two diagnostics samples are refreshed
(python dashboard/capture_artifact_fixtures.py SRC_DATA_STORE FIXTURE_DIR
--diagnostics-only).
"""

import json
import shutil
import sys
from pathlib import Path

import pandas as pd

sess = "2026-10-16"


def capture_diagnostics(src: Path, fx: Path) -> None:
    """Trimmed real calibration-diagnostics samples (one session + the summary)."""
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from desk_artifacts import DIAG_SUMMARY_FIELDS

    out = fx / "diagnostics"
    out.mkdir(parents=True, exist_ok=True)
    d = json.loads((src / "calibrations" / "diagnostics" / f"{sess}.json").read_text())
    ident = {k: v for k, v in d["identifiability"].items() if k != "profile_check"}
    ms = d["trf"]["multistart"]
    keep = {
        "schema_version": d["schema_version"], "session": d["session"], "identifiability": ident,
        "trf": {"multistart": {k: ms[k] for k in ("n_starts", "n_starts_at_best", "n_distinct_basins",
                                                    "n_failed_starts", "best_label") if k in ms}},
        "sigma_profile": {k: d["sigma_profile"][k] for k in ("caps", "kappa_max", "production_box")},
        "spx_only": {}, "joint_rough_heston": None,
    }
    keep["sigma_profile"]["rows"] = [{k: v for k, v in r.items() if k != "spx_oos_rmse_by_expiry"}
                                     for r in d["sigma_profile"]["rows"]]
    drop = ("multistart", "identifiability", "spx_oos_rmse_by_expiry", "numerics", "settings")
    for m, r in d["spx_only"].items():
        keep["spx_only"][m] = {k: v for k, v in r.items() if k not in drop}
    keep["joint_rough_heston"] = {k: v for k, v in d["joint_rough_heston"].items() if k not in drop}
    (out / f"{sess}.json").write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n")
    summ = json.loads((src / "calibrations" / "diagnostics" / "summary.json").read_text())
    rows = [{k: r.get(k) for k in ("session", "on_bound", "joint_rough_on_bound", "relative_standard_errors",
                                   *DIAG_SUMMARY_FIELDS)} for r in summ["rows"]]
    (out / "summary.json").write_text(json.dumps(
        {"schema_version": summ["schema_version"], "generated_utc": summ["generated_utc"],
         "n_sessions": len(rows), "rows": rows}, indent=1, sort_keys=True) + "\n")
    print("diagnostics", sess, len(rows), "summary rows")


if "--diagnostics-only" in sys.argv:
    capture_diagnostics(Path(sys.argv[1]), Path(sys.argv[2]))
    sys.exit(0)

src, fx, fwd = (Path(a) for a in sys.argv[1:4])
res = pd.read_parquet(src / "surface" / sess / "residuals.parquet")
exps = sorted(res["expiry"].unique())
tdays = res.groupby("expiry")["T_days"].first()
keep_exp = [exps[0], next(e for e in exps if tdays[e] >= 28), next(e for e in exps if tdays[e] >= 85)]
q = res[res["expiry"].isin(keep_exp)]
first = q["model"].iloc[0]
ids = q[q["model"] == first][["expiry", "strike", "right"]].drop_duplicates()
ids = ids[ids.groupby("expiry").cumcount() % 4 == 0]
q = q.merge(ids, on=["expiry", "strike", "right"])
q = q.drop(columns=[c for c in ("bid", "ask", "repair_dprice") if c in q.columns])
out = fx / "surface" / sess
out.mkdir(parents=True, exist_ok=True)
q.reset_index(drop=True).to_parquet(out / "residuals.parquet", index=False)
print("residuals", q.shape, keep_exp, q.groupby("model").size().to_dict())
sc = json.loads((src / "surface" / sess / "scorecard.json").read_text())
for m in sc["models"]:
    met = m["metrics"]
    m["metrics"] = {
        k: met[k]
        for k in ("all", "in_sample", "out_of_sample", "excluding_arbitrage_flagged", "n_unpriced")
        if k in met
    }
    m["metrics"]["by_expiry"] = {e: v for e, v in met.get("by_expiry", {}).items() if e in keep_exp}
    m.pop("fit", None)
    m["params"] = {}
(out / "scorecard.json").write_text(json.dumps(sc, indent=1, sort_keys=True) + "\n")
su = json.loads((src / "surface" / sess / "surface.json").read_text())
su["arbitrage"].pop("largest_mid_violations", None)
su["term_structure"] = [t for t in su["term_structure"] if t["expiry"] in keep_exp]
su["surfaces"] = {k: {"dense_grid": v.get("dense_grid", {})} for k, v in su["surfaces"].items()}
su["inputs"].pop("chain_sha256", None)
(out / "surface.json").write_text(json.dumps(su, indent=1, sort_keys=True) + "\n")
vx = json.loads((src / "surface" / sess / "vix_replication.json").read_text())
vx["indices"] = {
    t: {
        k: v
        for k, v in r.items()
        if k in ("replicated", "official", "gap_vs_official", "headline_method", "official_inside_day_range")
    }
    for t, r in vx["indices"].items()
}
vx["strip"] = [
    {
        k: r[k]
        for k in (
            "expiry",
            "days",
            "var_swap_vol",
            "atm_vol_quotes",
            "vs_minus_atm_vp",
            "atm_over_vs",
            "n_puts",
            "n_calls",
            "settlement",
        )
        if k in r
    }
    for r in vx["strip"]
    if r["expiry"] in keep_exp
]
vx["gap_notes"] = vx["gap_notes"][:2]
(out / "vix_replication.json").write_text(json.dumps(vx, indent=1, sort_keys=True) + "\n")
idx = json.loads((src / "surface" / "index.json").read_text())
idx["sessions"] = [r for r in idx["sessions"] if r["session"] in ("2026-06-18", sess)]
(fx / "surface" / "index.json").write_text(json.dumps(idx, indent=1, sort_keys=True) + "\n")
inf = fx / "inference"
inf.mkdir(exist_ok=True)
s = json.loads((src / "backtest" / "inference" / "summary.json").read_text())
keep = {
    k: s[k]
    for k in (
        "schema",
        "n_sessions",
        "bootstrap",
        "bootstrap_settings",
        "closed_form",
        "nulls",
        "multiple_testing",
        "headline",
        "conventions",
    )
    if k in s
}
t = s["trials"]
keep["trials"] = {k: t[k] for k in ("dsr", "dsr_range", "n_choices", "ledger", "note") if k in t}
keep["nulls"].pop("attribution_vs_always_short", None)
(inf / "summary.json").write_text(json.dumps(keep, indent=1, sort_keys=True) + "\n")
b = json.loads((src / "backtest" / "inference" / "benchmarks.json").read_text())
bk = {
    "indices": b["indices"],
    "window": b["window"],
    "co_movement": {
        sl: {ix: {"all": b["co_movement"][sl][ix]["all"]} for ix in b["indices"]}
        for sl in ("portfolio", "s3", "always_short_straddle")
    },
    "implied_beta": {
        "canonical_beta": b["implied_beta"]["canonical_beta"],
        **{
            ix: {k: b["implied_beta"][ix][k] for k in ("overall", "by_year", "n_periods")}
            for ix in ("PUT", "WPUT")
        },
    },
}
(inf / "benchmarks.json").write_text(json.dumps(bk, indent=1, sort_keys=True) + "\n")
f = json.loads((src / "backtest" / "inference" / "forecast_eval.json").read_text())
bw = f["backtest_window"]
fk = {
    "models": f["models"],
    "backtest_window": {
        "losses": {"qlike": bw["losses"]["qlike"]},
        "rank": {"qlike": bw["rank"]["qlike"]},
        "r2_vol": bw["r2_vol"],
        "mcs": {"qlike": {"90": {"included": bw["mcs"]["qlike"]["90"]["included"]}}},
        **{k: bw[k] for k in ("start", "end", "n", "horizon", "mean_realised_vol", "mean_forecast_vol")},
    },
}
(inf / "forecast_eval.json").write_text(json.dumps(fk, indent=1, sort_keys=True) + "\n")
fw = fx / "forward"
fw.mkdir(exist_ok=True)
for p in sorted(fwd.glob("ledger_*.json")):
    shutil.copy(p, fw / p.name)
capture_diagnostics(src, fx)
