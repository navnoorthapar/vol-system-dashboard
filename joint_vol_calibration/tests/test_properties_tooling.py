"""
test_properties_tooling.py: property-based tests (Hypothesis) of the CI gates
in scripts/ci, which decide whether a change may merge.

baseline_gate.py (ruff / mypy against a committed per-(file, code) baseline):

  * a run identical to the baseline passes and reports nothing to shrink;
  * the gate fails exactly when some (file, code) count rises, and names
    exactly those pairs; fixed findings are reported as shrinkable, never
    as failures;
  * --update refuses to raise any count (the baseline can only shrink) and,
    when it succeeds, leaves a baseline the same run passes with nothing left
    to shrink;
  * the JSON baseline round-trips, and its 'total' is the sum of its counts;
  * the mypy and ruff parsers recover every finding from the tools' output
    formats (notes and summary lines are not findings);
  * a finding in a strict file always fails, whatever the baseline says.

reproduce.py (`make reproduce`: rebuilt artifacts against the committed ones):
the comparator measures deviations relative to each column's scale, never
passes a NaN mismatch, and skips / tolerates / notes exactly the keys it is
told to (properties listed above its tests below).
"""
from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from pathlib import Path

import pytest

pytest.importorskip("hypothesis")
from hypothesis import assume, given, settings
from hypothesis import strategies as st

from joint_vol_calibration.tests.test_properties_pricing import budget

pytestmark = pytest.mark.property

ROOT = Path(__file__).resolve().parents[2]


def _load(name: str):
    path = ROOT / "scripts" / "ci" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"_ci_{name}", path)
    if spec is None or spec.loader is None:
        raise ImportError(path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = mod            # dataclasses resolve their module here
    spec.loader.exec_module(mod)
    return mod


gate = _load("baseline_gate")

paths = st.sampled_from(["pkg/a.py", "pkg/b.py", "scripts/c.py", "dashboard/app.py", "x/y/z.py"])
codes = st.sampled_from(["E501", "B023", "arg-type", "union-attr", "S301", "UP045"])
keys = st.tuples(paths, codes)
counters = st.dictionaries(keys, st.integers(0, 5), max_size=12).map(Counter)


@settings(max_examples=budget(2.0))
@given(c=counters)
def test_an_unchanged_run_passes(c):
    res = gate.compare(c, c)
    assert res.ok
    assert not res.grown
    assert not res.shrunk


@settings(max_examples=budget(4.0))
@given(base=counters, cur=counters)
def test_the_gate_fails_exactly_on_growth(base, cur):
    res = gate.compare(base, cur)
    grown = {k for k in set(base) | set(cur) if cur.get(k, 0) > base.get(k, 0)}
    shrunk = {k for k in set(base) | set(cur) if cur.get(k, 0) < base.get(k, 0)}
    assert set(res.grown) == grown
    assert set(res.shrunk) == shrunk
    assert res.ok == (not grown)
    for k, (b, c) in res.grown.items():
        assert (b, c) == (base.get(k, 0), cur.get(k, 0))


@settings(max_examples=budget(4.0))
@given(base=counters, cur=counters)
def test_update_can_only_shrink_the_baseline(base, cur):
    grows = any(n > base.get(k, 0) for k, n in cur.items())
    if grows:
        with pytest.raises(ValueError, match="refusing to grow the baseline"):
            gate.updated_baseline(base, cur)
        new = gate.updated_baseline(base, cur, allow_growth=True)   # the reviewed re-baseline path
    else:
        new = gate.updated_baseline(base, cur)
        assert all(n <= base.get(k, 0) for k, n in new.items())
    res = gate.compare(new, cur)
    assert res.ok
    assert not res.shrunk
    assert all(n > 0 for n in new.values())


@settings(max_examples=budget(2.0))
@given(c=counters, version=st.text(min_size=1, max_size=20))
def test_baseline_json_round_trips(c, version):
    text = gate.to_json(c, "ruff", version)
    back, doc = gate.from_json(text)
    positive = Counter({k: n for k, n in c.items() if n > 0})
    assert back == positive
    assert doc["total"] == sum(positive.values())
    assert doc["version"] == version
    assert text == gate.to_json(back, "ruff", version)              # canonical: stable diffs


messages = st.text(st.characters(blacklist_categories=("Cc", "Cs"), blacklist_characters="[]\n"),
                   min_size=1, max_size=40).map(str.strip).filter(bool)


@settings(max_examples=budget(2.0))
@given(findings=st.lists(st.tuples(paths, st.integers(1, 5000), codes, messages), max_size=15),
       notes=st.lists(st.tuples(paths, st.integers(1, 99)), max_size=5))
def test_mypy_output_parses_back_to_its_findings(findings, notes):
    lines = [f"{p}:{n}: error: {m}  [{c}]" for p, n, c, m in findings]
    lines += [f"{p}:{n}: note: See https://mypy.rtfd.io" for p, n in notes]
    lines.append(f"Found {len(findings)} errors in 3 files (checked 9 source files)")
    parsed = gate.parse_mypy_output("\n".join(lines), ROOT)
    assert [(f.path, f.line, f.code, f.message) for f in parsed] == [
        (p, n, c, m) for p, n, c, m in findings]


@settings(max_examples=budget(2.0))
@given(findings=st.lists(st.tuples(paths, st.integers(1, 5000), codes), max_size=15))
def test_ruff_json_parses_back_to_repo_relative_findings(findings):
    doc = [{"filename": str(ROOT / p), "code": c, "location": {"row": n, "column": 1},
            "message": "msg"} for p, n, c in findings]
    parsed = gate.parse_ruff_json(json.dumps(doc), ROOT)
    assert gate.count(parsed) == Counter((p, c) for p, _, c in findings)


@settings(max_examples=budget(2.0))
@given(base=counters, cur=counters)
def test_a_strict_file_finding_always_fails(base, cur):
    strict = "pkg/a.py"
    res = gate.compare(base, cur, strict_files=(strict,))
    touched = any(k[0] == strict and (base.get(k, 0) or cur.get(k, 0)) for k in set(base) | set(cur))
    assert (not res.ok) if touched else (res.ok == (not res.grown))


# ── reproduce.py: the artifact comparator ────────────────────────────────────
#
#   * an artifact compared with itself passes with deviation 0;
#   * moving one numeric cell by a fraction d of its column's scale is reported
#     as deviation d (to rounding) at that column, and fails exactly when d
#     exceeds the tolerance;
#   * a NaN where a number is expected (or the reverse) always fails;
#   * skipped keys are not compared, optional keys may be missing, any other
#     missing key fails; provenance keys are compared but only noted.

repro = _load("reproduce")

finite = st.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False)


@st.composite
def frames(draw):
    import numpy as np
    import pandas as pd
    n = draw(st.integers(1, 12))
    cols = draw(st.lists(st.sampled_from(["nav", "pnl", "vega", "iv"]), min_size=1, max_size=4, unique=True))
    data = {c: np.array(draw(st.lists(finite, min_size=n, max_size=n))) for c in cols}
    return pd.DataFrame(data)


@settings(max_examples=budget(2.0))
@given(df=frames())
def test_an_artifact_matches_itself(df):
    check = repro.Check("x", 0.0)
    repro.compare_frames(df.copy(), df, check)
    assert check.ok
    assert check.deviation == 0.0


@settings(max_examples=budget(4.0))
@given(df=frames(), data=st.data(), d=st.floats(1e-14, 1e-3), tol=st.sampled_from([1e-10, 1e-6]))
def test_a_moved_cell_is_measured_relative_to_its_column_scale(df, data, d, tol):
    import numpy as np
    col = data.draw(st.sampled_from(list(df.columns)))
    scale = float(np.max(np.abs(df[col].to_numpy())))
    # d * scale must be a normal float: a perturbation of a subnormal column
    # (Hypothesis found 4.9e-324) underflows to no change at all
    assume(scale > 1e-280)
    row = data.draw(st.integers(0, len(df) - 1))
    got = df.copy()
    got.loc[row, col] = df.loc[row, col] + d * scale
    check = repro.Check("x", tol)
    repro.compare_frames(got, df, check)
    assert check.deviation == pytest.approx(d, rel=1e-6, abs=1e-15)
    assert check.where.startswith(col)
    assert check.ok == (check.deviation <= tol)


@settings(max_examples=budget())
@given(df=frames(), data=st.data())
def test_a_nan_mismatch_always_fails(df, data):
    col = data.draw(st.sampled_from(list(df.columns)))
    row = data.draw(st.integers(0, len(df) - 1))
    got = df.copy()
    got.loc[row, col] = float("nan")
    check = repro.Check("x", 1.0)
    repro.compare_frames(got, df, check)
    assert not check.ok


@settings(max_examples=budget(2.0))
@given(base=st.dictionaries(st.sampled_from(["kappa", "theta", "loss", "fit_time", "losses", "n"]),
                            finite, min_size=1, max_size=6),
       drop=st.sampled_from(["kappa", "theta", "loss", "fit_time", "losses", "n"]),
       text=st.sampled_from(["a", "b"]))
def test_skip_optional_and_provenance_keys(base, drop, text):
    """What compare_file does with a calibration pickle: 'fit_time' skipped,
    'losses' optional, 'methods' provenance (noted, never failed)."""
    import pickle
    import tempfile
    exp = dict(base, methods={"T": "a"})
    got = {k: v for k, v in exp.items() if k != drop}
    got["methods"] = {"T": text}
    if "fit_time" in got:
        got["fit_time"] = got["fit_time"] + 1.0
    with tempfile.TemporaryDirectory() as tmp:
        g, e = Path(tmp) / "got.pkl", Path(tmp) / "exp.pkl"
        g.write_bytes(pickle.dumps(got))
        e.write_bytes(pickle.dumps(exp))
        check = repro.compare_file(g, e, "x", 0.0, skip={"fit_time"}, optional={"losses"},
                                   provenance={"methods"})
    missing_required = drop in exp and drop not in ("fit_time", "losses")
    assert check.ok == (not missing_required)
    assert bool(check.notes) == (text != "a")
