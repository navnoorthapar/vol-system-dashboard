"""
Forward ledger (backtest/forward_ledger.py) on slices of the committed SPX
chains 2026-03-24 → 03-27 → 05-29 (tests/fixtures/forward_spx_chain_*.csv;
expiries 14-50 days out, strikes on the 25-point grid around spot) and the
committed SPX closes. Checks executable-price arithmetic against the quotes
themselves, settlement, the hash chain, determinism and the schema.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

from joint_vol_calibration.backtest import forward_ledger as fl

FIX = Path(__file__).parent / "fixtures"
SESSIONS = ("2026-03-24", "2026-03-27", "2026-05-29")
SPOT = {"2026-03-24": 6556.37, "2026-03-27": 6368.85, "2026-05-29": 7580.06}
RATE = {"2026-03-24": 0.036872, "2026-03-27": 0.036739, "2026-05-29": 0.036544}
# ^VIX9D/^VIX/^VIX3M/^VIX6M closes of 2026-03-27 (committed DB)
VIX_0327 = pd.Series({"^VIX9D": 30.639999, "^VIX": 31.049999, "^VIX3M": 29.27, "^VIX6M": 29.780001})

SIG_A = {"s1": {"position": -1, "kelly": 0.10}, "s1c": {"position": 1, "kelly": 0.12},
         "s2": {"position": 0, "kelly": 0.0}, "s3": {"position": 1, "kelly": 0.08},
         "s4": {"position": 1, "kelly": 0.10}}
# 03-27: S1 and S3 hold, S1C goes flat, S4 flips, S2 turns on
SIG_B = {"s1": {"position": -1, "kelly": 0.10}, "s1c": {"position": 0, "kelly": 0.0},
         "s2": {"position": 1, "kelly": 0.10}, "s3": {"position": 1, "kelly": 0.08},
         "s4": {"position": -1, "kelly": 0.10}}


def _chain(session):
    d = pd.read_csv(FIX / f"forward_spx_chain_{session}.csv", keep_default_na=False, na_values=[""])
    d["root"] = d["root"].fillna("")
    return d


@pytest.fixture(scope="module")
def closes():
    c = pd.read_csv(FIX / "forward_spx_closes_2026.csv")
    return dict(zip(c["date"], c["close"]))


@pytest.fixture(scope="module")
def ledgers(closes):
    l1 = fl.step(None, SESSIONS[0], _chain(SESSIONS[0]), SIG_A, spot=SPOT[SESSIONS[0]],
                 r_ref=RATE[SESSIONS[0]])
    l2 = fl.step(l1, SESSIONS[1], _chain(SESSIONS[1]), SIG_B, spot=SPOT[SESSIONS[1]], r_ref=RATE[SESSIONS[1]],
                 settlement_closes=closes, vix_row=VIX_0327)
    l3 = fl.step(l2, SESSIONS[2], _chain(SESSIONS[2]), SIG_B, spot=SPOT[SESSIONS[2]], r_ref=RATE[SESSIONS[2]],
                 settlement_closes=closes)
    return l1, l2, l3


def _quote(session, expiry, strike, right):
    d = _chain(session)
    row = d[(d["expiry"] == expiry) & (d["strike"] == strike) & (d["right"] == right)].iloc[0]
    return float(row["bid"]), float(row["ask"])


def test_contract_is_the_30_day_atm_forward_straddle():
    book = fl.observed_book(_chain(SESSIONS[0]))
    q = fl.select_straddle(book, SESSIONS[0], r_ref=RATE[SESSIONS[0]])
    days = {e: (pd.Timestamp(e) - pd.Timestamp(SESSIONS[0])).days for e in set(book["expiry"])}
    assert q["days_to_expiry"] == min((abs(d - 30), d) for d in days.values() if d >= 14)[1]
    strikes = sorted(set(book[book["expiry"] == q["expiry"]]["strike"]))
    assert abs(q["strike"] - q["parity_forward"]) <= min(abs(k - q["parity_forward"]) for k in strikes) + 1e-9
    cb, ca = _quote(SESSIONS[0], q["expiry"], q["strike"], "C")
    pb, pa = _quote(SESSIONS[0], q["expiry"], q["strike"], "P")
    assert q["bid"] == pytest.approx(cb + pb)
    assert q["ask"] == pytest.approx(ca + pa)
    # the forward implied by the straddle's own legs makes call and put IVs equal
    assert q["iv_call"] == pytest.approx(q["iv_put"], abs=1e-8)
    assert 0.1 < q["iv_mid"] < 0.5
    assert abs(q["delta_F"]) < 0.2


def test_entries_at_executable_prices_and_engine_sizing(ledgers):
    l1 = ledgers[0]
    by = {p["sleeve"]: p for p in l1["positions"]}
    assert set(by) == {"s1", "s1c", "s3", "s4"}                      # S2 flat
    assert by["s3"]["direction"] == -1                               # S3 on = short index straddle
    for p in by.values():
        e = p["entry"]
        assert e["price_exec"] == (e["ask"] if p["direction"] > 0 else e["bid"])
        n = max(1, int(p["kelly"] * 1e6 / (e["price_exec"] * 100)))
        if p["direction"] < 0:
            n = min(n, int(1e6 / (0.2 * p["contract"]["strike"] * 100)))
        assert p["contracts_per_1m"] == n
    assert all(a["action"] == "open" for a in l1["actions"])


def test_close_books_exit_bid_minus_entry_ask(ledgers):
    """S1C (long) goes flat on 03-27: sold at the 03-27 bids, bought at the
    03-24 asks; the gap to the mid P&L is exactly the two half-spreads."""
    _l1, l2, _ = ledgers
    c = {x["sleeve"]: x for x in l2["closed"]}["s1c"]
    exp, K = c["contract"]["expiry"], c["contract"]["strike"]
    cb1, ca1 = _quote(SESSIONS[0], exp, K, "C")
    pb1, pa1 = _quote(SESSIONS[0], exp, K, "P")
    cb2, ca2 = _quote(SESSIONS[1], exp, K, "C")
    pb2, pa2 = _quote(SESSIONS[1], exp, K, "P")
    n = c["contracts_per_1m"]
    assert c["closed"]["pnl_exec"] == pytest.approx(((cb2 + pb2) - (ca1 + pa1)) * 100 * n)
    half = ((ca1 + pa1) - (cb1 + pb1)) / 2 + ((ca2 + pa2) - (cb2 + pb2)) / 2
    assert c["closed"]["pnl_mid"] - c["closed"]["pnl_exec"] == pytest.approx(half * 100 * n)
    assert c["closed"]["pnl_net"] == pytest.approx(c["closed"]["pnl_exec"] - 4 * 0.65 * n)
    acts = {a["sleeve"]: a["action"] for a in l2["actions"]}
    assert acts == {"s1": "hold", "s1c": "close", "s2": "open", "s3": "hold", "s4": "close"}
    # a flip closes and waits a session before opening the other side (engine rule)
    assert "s4" not in {p["sleeve"] for p in l2["positions"]}


def test_marks_and_model_mark_on_a_held_short(ledgers):
    _, l2, _ = ledgers
    p = {x["sleeve"]: x for x in l2["positions"]}["s1"]
    m = p["mark"]
    assert m["status"] == "marked"
    assert m["close_value_exec"] == m["ask"]
    assert m["pnl_exec"] == pytest.approx(
        -(m["ask"] - p["entry"]["price_exec"]) * 100 * p["contracts_per_1m"])
    assert m["pnl_exec"] <= m["pnl_mid"]
    # the backtest's β·VIX value of the same contract sits beside the market mid
    assert m["model_sigma"] == pytest.approx(0.82 * 0.31, abs=0.01)
    assert abs(m["model_minus_mid"]) / m["mid"] < 0.25


def test_expired_straddles_settle_on_the_official_close_and_roll(ledgers, closes):
    _, _l2, l3 = ledgers
    st = {x["sleeve"]: x for x in l3["closed"]}
    s1 = st["s1"]
    K = s1["contract"]["strike"]
    S_T = closes[s1["contract"]["expiry"]]
    assert s1["closed"]["how"] == "settle"
    assert s1["closed"]["pnl_exec"] == pytest.approx(
        -1 * (abs(S_T - K) - s1["entry"]["price_exec"]) * 100 * s1["contracts_per_1m"])
    assert s1["closed"]["commission"] == pytest.approx(2 * 0.65 * s1["contracts_per_1m"])   # entry only
    acts = {(a["sleeve"], a["action"]) for a in l3["actions"]}
    assert ("s1", "settle") in acts
    assert ("s1", "roll") in acts
    assert ("s4", "open") in acts
    new = {p["sleeve"]: p for p in l3["positions"]}
    assert new["s1"]["entry"]["session"] == SESSIONS[2]
    # summary accumulates every realised trade
    for s in fl.SLEEVES:
        tot = sum(c["closed"]["pnl_exec"] for L in ledgers for c in L["closed"] if c["sleeve"] == s)
        assert l3["summary"][s]["realised_pnl_exec"] == pytest.approx(tot)


def test_settlement_waits_for_the_official_close(ledgers):
    _, l2, _ = ledgers
    l3 = fl.step(l2, SESSIONS[2], _chain(SESSIONS[2]), SIG_B, settlement_closes={})
    acts = {a["sleeve"]: a["action"] for a in l3["actions"]}
    assert acts["s1"] == "await_settlement"
    assert not l3["closed"]


def test_unmarkable_position_is_kept_not_priced(ledgers):
    l1 = ledgers[0]
    p = {x["sleeve"]: x for x in l1["positions"]}["s1c"]
    ch = _chain(SESSIONS[1])
    ch = ch[~((ch["expiry"] == p["contract"]["expiry"]) & (ch["strike"] == p["contract"]["strike"])
              & (ch["right"] == "P"))]
    l2 = fl.step(l1, SESSIONS[1], ch, SIG_B)
    kept = {x["sleeve"]: x for x in l2["positions"]}["s1c"]
    assert kept["mark"]["status"] == "unmarked"
    assert {a["sleeve"]: a["action"] for a in l2["actions"]}["s1c"] == "close_pending"
    assert "s1c" not in {c["sleeve"] for c in l2["closed"]}


def test_hash_chain_and_tamper_detection(ledgers):
    assert fl.verify_chain(list(ledgers)) == []
    l1 = json.loads(json.dumps(ledgers[0]))
    l1["positions"][0]["entry"]["price_exec"] += 0.05
    probs = fl.verify_chain([l1, ledgers[1], ledgers[2]])
    assert probs
    assert "2026-03-27" in probs[0]
    assert ledgers[1]["previous"]["sha256"] == fl.ledger_sha256(ledgers[0])
    assert len(ledgers[0]["inputs"]["chain_sha256"]) == 64


def test_deterministic_and_append_only_files(ledgers, tmp_path, closes):
    again = fl.step(None, SESSIONS[0], _chain(SESSIONS[0]), SIG_A, spot=SPOT[SESSIONS[0]],
                    r_ref=RATE[SESSIONS[0]])
    assert fl.canonical_json(again) == fl.canonical_json(ledgers[0])
    # the chain hash does not depend on row order
    shuffled = _chain(SESSIONS[0]).sample(frac=1.0, random_state=3)
    assert fl.step(None, SESSIONS[0], shuffled, SIG_A)["inputs"]["chain_sha256"] == \
        ledgers[0]["inputs"]["chain_sha256"]
    for L in ledgers:
        fl.write_ledger(L, tmp_path)
    loaded = [fl.load_ledger(fl.ledger_path(tmp_path, s)) for s in SESSIONS]
    assert [fl.canonical_json(x) for x in loaded] == [fl.canonical_json(x) for x in ledgers]
    assert fl.latest_ledger(tmp_path, before="2026-05-29")["session"] == "2026-03-27"
    fl.write_ledger(ledgers[0], tmp_path)                   # same content: idempotent
    other = fl.step(None, SESSIONS[0], _chain(SESSIONS[0]), SIG_B)
    with pytest.raises(FileExistsError):
        fl.write_ledger(other, tmp_path)
    with pytest.raises(ValueError, match="must follow"):
        fl.step(ledgers[1], SESSIONS[0], _chain(SESSIONS[0]), SIG_A)     # out of order


def test_schema_validation(ledgers):
    fl.validate_ledger(ledgers[2])
    bad = json.loads(json.dumps(ledgers[2]))
    del bad["positions"][0]["entry"]["bid"]
    bad["actions"][0]["action"] = "teleport"
    with pytest.raises(ValueError, match=r"entry.*bid.*|teleport"):
        fl.validate_ledger(bad)
    bad2 = json.loads(json.dumps(ledgers[0]))
    bad2["inputs"]["chain_sha256"] = "xyz"
    with pytest.raises(ValueError, match=r"chain_sha256.*does not match"):
        fl.validate_ledger(bad2)


def test_signals_for_session_reproduce_the_backtest_positions():
    """On sessions inside the backtest window the helper the bot calls gives
    the positions and sizes the canonical backtest traded."""
    from joint_vol_calibration.backtest.backtest_engine import RESULTS_PATH

    eq = pd.read_parquet(RESULTS_PATH)
    on = 0
    for session in ("2018-01-09", "2018-07-03", "2022-04-06", "2025-03-24"):
        sig = fl.signals_for_session(session)
        row = eq.loc[pd.Timestamp(session)]
        for s in fl.SLEEVES:
            assert sig[s]["position"] == pytest.approx(float(row[f"{s}_position"]))
            assert 0.0 <= sig[s]["kelly"] <= 0.2 + 1e-12
            on += sig[s]["position"] != 0
    assert on >= 6                       # the dates exercise every non-trivial state


def test_record_session_writes_a_chained_file(tmp_path):
    """The bot's one-call entry point on the committed DB copy: two sessions,
    the second linked to the first, VIX row and closes read from the DB."""
    p1 = fl.record_session(SESSIONS[0], _chain(SESSIONS[0]), spot=SPOT[SESSIONS[0]], r_ref=RATE[SESSIONS[0]],
                           directory=tmp_path, signals=SIG_A)
    p2 = fl.record_session(SESSIONS[1], _chain(SESSIONS[1]), spot=SPOT[SESSIONS[1]], r_ref=RATE[SESSIONS[1]],
                           directory=tmp_path, signals=SIG_B)
    a, b = fl.load_ledger(p1), fl.load_ledger(p2)
    assert fl.verify_chain([a, b]) == []
    held = {x["sleeve"]: x for x in b["positions"]}["s1"]
    assert held["mark"]["model_sigma"] > 0                  # the DB's 03-27 VIX row was used
    assert b["inputs"]["beta"] == pytest.approx(0.82)


def test_record_session_settles_after_a_long_gap_in_captures(tmp_path):
    """Positions opened on 03-24 expire in April; the next ledger is on
    2026-10-16, more than 120 days after those expiries. Each still settles on
    its own expiry's official close from the DB (not left awaiting a close
    outside the default look-back window)."""
    from joint_vol_calibration.data.database import get_spx_ohlcv

    flat = {s: {"position": 0, "kelly": 0.0} for s in fl.SLEEVES}
    p1 = fl.record_session(SESSIONS[0], _chain(SESSIONS[0]), spot=SPOT[SESSIONS[0]], r_ref=RATE[SESSIONS[0]],
                           directory=tmp_path, signals=SIG_A)
    first = fl.load_ledger(p1)
    late = "2026-10-16"
    assert all((pd.Timestamp(late) - pd.Timestamp(p["contract"]["expiry"])).days > 120
               for p in first["positions"])
    # the chain is only the hash input here: every sleeve is flat, nothing opens
    p2 = fl.record_session(late, _chain(SESSIONS[2]), directory=tmp_path, signals=flat)
    second = fl.load_ledger(p2)
    spx = get_spx_ohlcv(as_of_date=late, start_date="2026-04-01")
    close = dict(zip(spx["date"].astype(str), spx["close"].astype(float)))
    settled = {c["sleeve"]: c for c in second["closed"]}
    assert set(settled) == {p["sleeve"] for p in first["positions"]}
    assert not second["positions"]
    for c in settled.values():
        assert c["closed"]["how"] == "settle"
        assert c["mark"]["settle_close"] == pytest.approx(close[c["contract"]["expiry"]])
    assert fl.verify_chain([first, second]) == []


def test_a_book_quoted_before_the_signal_close_is_refused():
    """The session's signals use its 16:00 ET close: a book quoted earlier
    that day cannot be the one they trade on; a GTH book after it can."""
    ch = _chain(SESSIONS[0])
    with pytest.raises(ValueError, match="before the 2026-03-24 close"):
        fl.step(None, SESSIONS[0], ch, SIG_A, quote_time="2026-03-24T19:30:00Z")      # 15:30 ET
    ok = fl.step(None, SESSIONS[0], ch, SIG_A, quote_time="2026-03-25T02:30:00Z")     # 22:30 ET
    assert ok["inputs"]["quote_after_close"] is True
    assert ok["inputs"]["quote_time"] == "2026-03-25T02:30:00Z"
    assert fl.step(None, SESSIONS[0], ch, SIG_A)["inputs"]["quote_after_close"] is None
    # exactly at the close is allowed (EDT: 16:00 ET = 20:00 UTC on 2026-03-24)
    assert fl.quote_after_close(SESSIONS[0], "2026-03-24T20:00:00Z") is True
    with pytest.raises(ValueError, match="is before the 2026-03-24 close"):
        fl.quote_after_close(SESSIONS[0], "2026-03-24T19:59:59Z")


def test_committed_ledgers_form_one_valid_chain():
    """data_store/forward/ (written by the daily bot after each accepted
    session): every file is schema-valid, named after its session and linked
    to the one before it by SHA-256. An edited or missing link is an error
    here, before the bot commits, as it is on the Research page."""
    files = sorted((Path(__file__).resolve().parents[2] / "data_store" / "forward").glob("ledger_*.json"))
    if not files:
        pytest.skip("no committed forward ledger yet")
    ledgers = [fl.load_ledger(p) for p in files]
    for p, led in zip(files, ledgers):
        fl.validate_ledger(led)
        assert p.name == f"ledger_{led['session']}.json"
        # byte for byte what write_ledger writes (no hand edit, no reformat)
        assert p.read_text() == json.dumps(fl._clean(led), sort_keys=True, indent=1,
                                           ensure_ascii=False, allow_nan=False) + "\n"
    assert ledgers[0]["previous"] is None
    assert fl.verify_chain(ledgers) == []
