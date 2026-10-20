"""
Offline tests for the Cboe downloader (C22).

The old module could not return a single row from any Cboe endpoint: it asked
for 'SPX.json' (403; index symbols need a leading underscore), parsed fields
the payload does not have (every option carries only an OCC symbol), pointed
at two VX URLs that are dead S3 keys, renamed Settle onto an existing Close
column (TypeError on the real file), could not parse 'V (Oct 2026)', and dated
VX expiries by month-end.

Every fixture under tests/fixtures/cboe_* is a trimmed capture of the live
endpoint taken on 2026-10-12 (rows deleted, nothing edited):
  cboe_options_{SPX,VIX}_2026-10-09.json  delayed_quotes/options/_SPX.json, _VIX.json
  cboe_VX_<expiry>.csv                    VX per-contract history files
  cboe_vx_settlement_2026-10-{09,10}.csv  daily settlement CSV (10-10 = Saturday)
  cboe_<INDEX>_History.csv                index history files (VIX family, VVIX, VRO)
and, captured 2026-10-20 the same way (C23):
  cboe_{VIX1D,VIX1Y,SKEW,COR1M,COR3M,DSPX}_History.csv
No test touches the network.
"""
import copy
import hashlib
import json
import logging
import pathlib
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest

from joint_vol_calibration.config import VIX_INDEX_TENOR_DAYS
from joint_vol_calibration.data import cboe_downloader as cb
from joint_vol_calibration.data import database as db

FIX = pathlib.Path(__file__).parent / "fixtures"

# Fetch time of the captures: Monday 2026-10-12 08:15 UTC (04:15 ET, pre-open).
CAPTURED_AT = datetime(2026, 10, 12, 8, 15, 28, tzinfo=UTC)


def _text(name: str) -> str:
    return (FIX / name).read_text()


def _payload(und: str) -> dict:
    return json.loads(_text(f"cboe_options_{und}_2026-10-09.json"))


class _Resp:
    def __init__(self, text: str = "", status: int = 200):
        self.text, self.status_code = text, status

    def json(self):
        return json.loads(self.text)

    def raise_for_status(self):
        if self.status_code >= 400:
            raise cb.requests.HTTPError(f"{self.status_code}")


@pytest.fixture
def serve(monkeypatch):
    """Route cb._SESSION.get to fixture files; unknown URLs get a 403 like the dead S3 keys."""
    routes, seen = {}, []

    def fake_get(url, timeout=None):
        seen.append(url)
        for suffix, name in routes.items():
            if url.endswith(suffix):
                return _Resp(_text(name))
        return _Resp("<Error><Code>AccessDenied</Code></Error>", 403)

    monkeypatch.setattr(cb._SESSION, "get", fake_get)
    return routes, seen


# ── delayed option quotes ─────────────────────────────────────────────────────

@pytest.mark.parametrize("und", ["SPX", "VIX"])
def test_index_quote_url_has_leading_underscore(serve, und):
    routes, seen = serve
    routes[f"/options/_{und}.json"] = f"cboe_options_{und}_2026-10-09.json"
    df = cb.download_cboe_options_snapshot(und, now=CAPTURED_AT)
    assert seen == [f"https://cdn.cboe.com/api/global/delayed_quotes/options/_{und}.json"]
    assert not df.empty


def test_spx_payload_parses_every_occ_row():
    df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    assert len(df) == 17                          # the old parser returned 0 of these
    assert set(df["root"]) == {"SPX", "SPXW"}
    row = df[(df["contract_symbol"] == "SPXW261016C07700000")].iloc[0]
    assert (row["expiry"], row["strike"], row["right"]) == ("2026-10-16", 7700.0, "C")
    assert (row["bid"], row["ask"], row["mid_price"]) == (143.1, 144.2, pytest.approx(143.65))
    assert row["implied_vol"] == pytest.approx(0.1383)
    assert row["delta"] == pytest.approx(0.8737)
    assert row["open_interest"] == 1521


def test_am_and_pm_contracts_are_kept_apart():
    """SPX (AM) and SPXW (PM) share the 2026-10-16 expiry and strikes but are different contracts."""
    df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    oct16 = df[(df["expiry"] == "2026-10-16") & (df["strike"] == 7800.0) & (df["right"] == "C")]
    assert sorted(oct16["root"]) == ["SPX", "SPXW"]
    am = oct16[oct16["root"] == "SPX"].iloc[0]
    pm = oct16[oct16["root"] == "SPXW"].iloc[0]
    assert (am["settlement"], pm["settlement"]) == ("AM", "PM")
    # 16:00 ET Fri 10-09 to 09:30 ET Fri 10-16 vs to 16:00 ET: 6.5 hours apart.
    assert am["time_to_expiry"] == pytest.approx((7 * 1440 - 390) / 525_600)
    assert pm["time_to_expiry"] == pytest.approx(7 / 365)
    assert pm["mid_price"] > am["mid_price"]      # the later-settling call is worth more


def test_snapshot_is_dated_by_the_quote_session_not_the_clock():
    """Captured on Monday 10-12 before the open: the quotes are Friday 10-09's."""
    df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    assert set(df["snapshot_date"]) == {"2026-10-09"}
    assert set(df["underlying_price"]) == {7811.54}
    # the payload is stamped 08:15:07Z; its quotes are QUOTE_DELAY (15 min) older
    assert set(df["quote_time"]) == {"2026-10-12T08:00:07Z"}
    assert set(df["source"]) == {"cboe"}


def test_quote_time_is_the_payload_stamp_minus_the_feed_delay():
    """Measured 2026-10-20 on the live feed: a payload stamped 16:04:29Z held
    a latest option trade and an underlying last trade at 11:49:27 ET
    (15:49:27Z), 15m02s earlier. quote_time must be that instant, not the
    file's generation time."""
    assert timedelta(minutes=15) == cb.QUOTE_DELAY
    assert cb.STATED_DELAY_MINUTES == 15
    assert cb.delayed_quote_time("2026-10-20 16:04:29") == "2026-10-20T15:49:29Z"
    lag = pd.Timestamp("2026-10-20T16:04:29Z") - pd.Timestamp("2026-10-20 11:49:27", tz="America/New_York")
    assert abs(lag - cb.QUOTE_DELAY) <= timedelta(seconds=5)
    # no stamp: the fetch time minus the delay
    assert cb.delayed_quote_time(None, CAPTURED_AT) == "2026-10-12T08:00:28Z"
    p = _payload("SPX")
    p.pop("timestamp")
    df = cb.parse_cboe_options_payload(p, "SPX", now=CAPTURED_AT)
    assert set(df["quote_time"]) == {"2026-10-12T08:00:28Z"}


def test_one_sided_quotes_get_no_mid_and_zero_iv_is_missing():
    df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    row = df[df["contract_symbol"] == "SPX261016P00200000"].iloc[0]   # bid 0 / ask 0.05, iv 0
    assert pd.isna(row["mid_price"])
    assert pd.isna(row["implied_vol"])


def test_vix_payload_roots_are_am_settled():
    df = cb.parse_cboe_options_payload(_payload("VIX"), "VIX", now=CAPTURED_AT)
    assert len(df) == 8
    assert set(df["root"]) == {"VIX", "VIXW"}
    assert set(df["settlement"]) == {"AM"}
    assert set(df["underlying_price"]) == {14.84}             # data.close, not the GTH current_price
    assert set(df["snapshot_date"]) == {"2026-10-09"}


def test_unparseable_and_expired_rows_are_dropped(caplog):
    payload = copy.deepcopy(_payload("SPX"))
    good = payload["data"]["options"][0]
    payload["data"]["options"] += [
        dict(good, option="NOT-AN-OCC-SYMBOL"),
        dict(good, option="SPXW261009C07800000"),   # PM, expired at the snapshot close
    ]
    with caplog.at_level(logging.WARNING):
        df = cb.parse_cboe_options_payload(payload, "SPX", now=CAPTURED_AT)
    assert len(df) == 17
    assert "without an OCC symbol" in caplog.text
    assert (df["time_to_expiry"] > 0).all()


def test_payload_session_wins_over_clock_with_warning(caplog):
    late = datetime(2026, 10, 13, 15, 0, tzinfo=UTC)    # clock says Tuesday's session
    with caplog.at_level(logging.WARNING):
        df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=late)
    assert set(df["snapshot_date"]) == {"2026-10-09"}
    assert "clock says 2026-10-13" in caplog.text


def test_parsed_chain_round_trips_through_the_database(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_database()
    df = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    assert db.insert_options_snapshot(df, "SPX") == 17
    surf = db.get_options_surface("2026-10-09", "SPX", max_age_sessions=0)
    assert len(surf) == 17
    assert set(surf["root"]) == {"SPX", "SPXW"}


# ── VX futures ────────────────────────────────────────────────────────────────

def test_vx_contract_file_parses_with_settle_as_the_mark():
    df = cb.parse_vx_contract_csv(_text("cboe_VX_2026-10-21.csv"), "2026-10-21")
    assert len(df) == 7
    row = df[df["date"] == "2026-09-22"].iloc[0]
    assert row["settle"] == pytest.approx(17.3607)
    assert row["close"] == pytest.approx(17.40)
    assert row["days_to_expiry"] == 29
    assert (row["expiry_month"], row["is_weekly"]) == ("2026-10", 0)
    first = df.iloc[0]                       # 2026-01-26: no trade, Close/Open written as 0
    assert pd.isna(first["close"])
    assert pd.isna(first["open"])
    assert first["settle"] == 21.50


def test_vx_weekly_contract_is_flagged():
    df = cb.parse_vx_contract_csv(_text("cboe_VX_2026-10-14.csv"), "2026-10-14")
    assert set(df["is_weekly"]) == {1}
    assert set(df["expiry_month"]) == {"2026-10"}


def test_vx_early_2013_file_has_no_published_settle():
    df = cb.parse_vx_contract_csv(_text("cboe_VX_2013-01-16.csv"), "2013-01-16")
    assert df["settle"].isna().all()
    assert df["close"].notna().all()


def test_expiry_day_settle_equals_vro_final_settlement():
    contract = cb.parse_vx_contract_csv(_text("cboe_VX_2024-01-17.csv"), "2024-01-17")
    final = contract[contract["date"] == "2024-01-17"]["settle"].iloc[0]
    vro_df = pd.read_csv(FIX / "cboe_VRO_History.csv")
    assert final == pytest.approx(float(vro_df.loc[vro_df["DATE"] == "01/17/2024", "VRO"].iloc[0]))


def test_daily_settlement_csv_keeps_vx_only_with_exact_expiries():
    df = cb.parse_vx_settlement_csv(_text("cboe_vx_settlement_2026-10-09.csv"), "2026-10-09")
    assert len(df) == 14                                        # VXM/VA/... rows excluded
    monthly = df[df["is_weekly"] == 0]
    assert monthly["symbol"].tolist()[:2] == ["VX/V6", "VX/X6"]
    assert monthly["expiry_date"].tolist() == [
        "2026-10-21", "2026-11-18", "2026-12-16", "2027-01-20", "2027-02-17",
        "2027-03-17", "2027-04-21", "2027-05-18", "2027-06-16"]
    oct_ = df[df["symbol"] == "VX/V6"].iloc[0]
    assert (oct_["settle"], oct_["days_to_expiry"]) == (pytest.approx(17.1186), 12)
    assert set(df.loc[df["is_weekly"] == 1, "symbol"]) == {"VX41/V6", "VX43/V6", "VX44/X6", "VX45/X6", "VX47/X6"}


def test_daily_settlement_csv_on_a_closed_day_is_empty():
    assert cb.parse_vx_settlement_csv(_text("cboe_vx_settlement_2026-10-10.csv"), "2026-10-10").empty


@pytest.mark.parametrize(("year", "month", "expected"), [
    (2026, 9, "2026-09-16"), (2026, 10, "2026-10-21"), (2026, 11, "2026-11-18"),
    (2027, 3, "2027-03-17"),
    (2027, 5, "2027-05-18"),     # Fri 2027-06-18 is the observed Juneteenth holiday
    (2018, 2, "2018-02-14"), (2020, 3, "2020-03-18"), (2024, 1, "2024-01-17"),
])
def test_vx_monthly_expiry_rule(year, month, expected):
    # (the old month-end rule gave 2026-10-30 for Oct 2026)
    assert db.vx_monthly_expiry(year, month) == expected


def test_contract_names_normalise_to_year_month():
    out = cb._normalise_expiry_month(pd.Series(["V (Oct 2026)", "F (Jan 2024)", "Jan-24", "JAN/2024", "2024-01"]))
    assert out.tolist() == ["2026-10", "2024-01", "2024-01", "2024-01", "2024-01"]


def test_futures_history_requests_monthly_contract_files(serve, caplog):
    routes, seen = serve
    routes["/VX/VX_2026-10-21.csv"] = "cboe_VX_2026-10-21.csv"
    with caplog.at_level(logging.WARNING):
        df = cb.download_vix_futures_daily_history(start="2026-09-15", end="2026-10-09")
    files = [u.rsplit("/", 1)[-1] for u in seen]
    assert files[:3] == ["VX_2026-09-16.csv", "VX_2026-10-21.csv", "VX_2026-11-18.csv"]
    assert all(u.startswith("https://cdn.cboe.com/data/us/futures/market_statistics/historical_data/VX/")
               for u in seen)
    assert set(df["expiry_date"]) == {"2026-10-21"}            # the others were 403s in this test
    assert df["date"].between("2026-09-15", "2026-10-09").all()
    assert "VX contract 2026-09-16 unavailable" in caplog.text


def test_final_settlements_use_vro_only_on_monthly_expiry_dates(serve):
    """VRO_History also holds forward-filled non-expiry rows (2015-2020) and weeklies; neither is stored."""
    routes, _ = serve
    routes["/VRO_History.csv"] = "cboe_VRO_History.csv"
    df = cb.download_vix_futures_settlements(end="2026-10-09")
    assert df["settlement_date"].tolist() == ["2018-02-14", "2020-03-18", "2024-01-17", "2026-09-16"]
    assert df.set_index("settlement_date").loc["2018-02-14", "settlement_price"] == pytest.approx(21.87)
    assert set(df["is_weekly"]) == {0}
    assert set(df["source"]) == {"cboe_vro"}


def test_contract_expiry_settle_overrides_vro_and_logs_disagreement(serve, caplog):
    routes, _ = serve
    routes["/VRO_History.csv"] = "cboe_VRO_History.csv"
    hist = cb.parse_vx_contract_csv(_text("cboe_VX_2024-01-17.csv"), "2024-01-17")
    agree = cb.download_vix_futures_settlements(start="2024-01-01", end="2024-01-31", daily_history=hist)
    assert agree[["settlement_price", "source"]].values.tolist() == [[pytest.approx(15.13), "cboe_contract"]]
    hist.loc[hist["date"] == "2024-01-17", "settle"] = 14.99          # a disagreeing contract file
    with caplog.at_level(logging.WARNING):
        out = cb.download_vix_futures_settlements(start="2024-01-01", end="2024-01-31", daily_history=hist)
    assert out["settlement_price"].tolist() == [pytest.approx(14.99)]
    assert "contract file 14.9900 vs VRO 15.1300" in caplog.text


def test_settlements_and_daily_rows_store_without_key_collisions(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_database()
    daily = cb.parse_vx_settlement_csv(_text("cboe_vx_settlement_2026-10-09.csv"), "2026-10-09")
    assert db.insert_vix_futures_daily(daily) == 14             # weeklies no longer collide
    hist = cb.parse_vx_contract_csv(_text("cboe_VX_2026-10-21.csv"), "2026-10-21")
    db.insert_vix_futures_daily(hist)
    curve = db.get_vix_futures_curve("2026-10-09")
    assert curve["expiry_date"].tolist()[:2] == ["2026-10-21", "2026-11-18"]
    oct_ = curve.iloc[0]
    # merged: the settlement file's symbol and the contract file's last trade
    assert (oct_["symbol"], oct_["settle"], oct_["close"]) == ("VX/V6", pytest.approx(17.1186), pytest.approx(17.06))
    assert len(db.get_vix_futures_curve("2026-10-09", include_weeklies=True)) == 14


# ── index history files ───────────────────────────────────────────────────────

def test_index_history_parses_ohlc_and_single_value_files(serve):
    routes, _ = serve
    routes["/VIX_History.csv"] = "cboe_VIX_History.csv"
    routes["/VVIX_History.csv"] = "cboe_VVIX_History.csv"
    vix = cb.download_vix_history()
    assert vix.loc[vix["date"] == "2026-02-06", "close"].iloc[0] == pytest.approx(17.76)
    assert "2026-01-19" in set(vix["date"])                    # published on MLK day; kept as published
    vvix = cb.download_index_history("VVIX")
    assert vvix["open"].isna().all()
    assert vvix["close"].notna().all()
    assert set(vix["source"]) == {"cboe"}


def test_term_structure_history_uses_the_config_tenor_map(serve):
    routes, _ = serve
    for idx in ("VIX9D", "VIX", "VIX3M", "VIX6M", "VVIX"):
        routes[f"/{idx}_History.csv"] = f"cboe_{idx}_History.csv"
    ts = cb.download_vix_term_structure_history(start="2026-02-01", end="2026-02-28")
    tenors = ts.drop_duplicates("ticker").set_index("ticker")["tenor_days"].to_dict()
    assert tenors == {**VIX_INDEX_TENOR_DAYS, "^VVIX": 30}
    assert tenors["^VIX6M"] == 184
    assert set(ts["date"]) == {"2026-02-05", "2026-02-06", "2026-02-09"}


# ── C23: other Cboe indices ───────────────────────────────────────────────────

def _serve_indices(serve, *names):
    routes, seen = serve
    for n in names:
        routes[f"/{n}_History.csv"] = f"cboe_{n}_History.csv"
    return seen


def test_extra_indices_parse_ohlc_and_single_value_files(serve):
    seen = _serve_indices(serve, "VIX1D", "SKEW", "DSPX", "COR3M")
    df = cb.download_cboe_indices(["VIX1D", "^skew", "DSPX", "COR3M"], start="2026-10-15", end="2026-10-16")
    assert [u.rsplit("/", 1)[1] for u in seen] == ["VIX1D_History.csv", "SKEW_History.csv",
                                                   "DSPX_History.csv", "COR3M_History.csv"]
    assert list(df.columns) == ["date", "symbol", "open", "high", "low", "close", "source"]
    assert df[["date", "symbol"]].values.tolist() == [
        ["2026-10-15", "COR3M"], ["2026-10-15", "DSPX"], ["2026-10-15", "SKEW"], ["2026-10-15", "VIX1D"],
        ["2026-10-16", "COR3M"], ["2026-10-16", "DSPX"], ["2026-10-16", "SKEW"], ["2026-10-16", "VIX1D"]]
    vix1d = df[(df["symbol"] == "VIX1D") & (df["date"] == "2026-10-16")].iloc[0]
    assert (vix1d["open"], vix1d["high"], vix1d["low"], vix1d["close"]) == (7.77, 25.66, 7.69, 23.27)
    skew = df[df["symbol"] == "SKEW"]
    assert skew["open"].isna().all()
    assert skew["close"].tolist() == [160.45, 168.59]
    assert set(df["source"]) == {"cboe"}


def test_extra_index_catalog():
    assert set(cb.DEFAULT_EXTRA_INDICES) == {"VIX1D", "VIX1Y", "SKEW", "COR1M", "COR3M", "DSPX"}
    assert set(cb.DEFAULT_EXTRA_INDICES) <= set(cb.CBOE_INDEX_CATALOG)
    assert not set(cb.CBOE_INDEX_CATALOG) & set(cb._TERM_INDEX_FILES.values())   # VIX family lives elsewhere
    for (what, first) in cb.CBOE_INDEX_CATALOG.values():
        assert what
        assert pd.Timestamp(first) <= pd.Timestamp("2022-05-13")
    assert cb.CBOE_INDEX_CATALOG["VIX1D"][1] == "2022-05-13"
    assert cb.CBOE_INDEX_CATALOG["SKEW"][1] == "1990-01-02"


def test_refresh_cboe_indices_stores_what_downloads_and_reports_failures(serve, tmp_path, monkeypatch):
    from joint_vol_calibration.data import pipeline as pl
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_database()
    _serve_indices(serve, "VIX1D", "VIX1Y", "SKEW", "COR1M", "COR3M")         # DSPX -> 403
    res = pl.refresh_cboe_indices(start="2026-01-01")
    assert res["SKEW"] == {"rows": 4, "min_date": "2026-10-14", "max_date": "2026-10-19"}
    assert len(res["errors"]) == 1
    assert res["errors"][0].startswith("DSPX")
    got = db.get_cboe_index("2026-10-16", ["SKEW", "VIX1D", "COR3M"])
    assert got.groupby("symbol").size().to_dict() == {"COR3M": 3, "SKEW": 3, "VIX1D": 3}
    assert db.get_cboe_index("2026-10-19", ["VIX1D"])["close"].iloc[-1] == 11.93


# ── C23: timestamped chain capture ────────────────────────────────────────────

class _BytesResp:
    def __init__(self, path):
        self.content = pathlib.Path(path).read_bytes()
        self.text, self.status_code = self.content.decode(), 200

    def json(self):
        return json.loads(self.content)

    def raise_for_status(self):
        pass


@pytest.fixture
def serve_chains(monkeypatch):
    seen = []

    def fake_get(url, timeout=None):
        seen.append(url)
        for und in ("SPX", "VIX"):
            if url.endswith(f"/options/_{und}.json"):
                return _BytesResp(FIX / f"cboe_options_{und}_2026-10-09.json")
        raise cb.requests.HTTPError("403")
    monkeypatch.setattr(cb._SESSION, "get", fake_get)
    return seen


def test_capture_is_the_snapshot_schema_plus_timestamps_and_a_hash(serve_chains):
    cap = cb.capture_option_chain("SPX", now=CAPTURED_AT)
    same = cb.parse_cboe_options_payload(_payload("SPX"), "SPX", now=CAPTURED_AT)
    pd.testing.assert_frame_equal(cap.frame, same)
    assert set(cap.frame.columns) == set(db._OPTION_COLUMNS) - {"underlying"}
    assert set(cap.frame["quote_time"]) == {"2026-10-12T08:00:07Z"}
    assert set(cap.frame["source"]) == {"cboe"}
    assert cap.quote_time == "2026-10-12T08:00:07Z"
    raw = (FIX / "cboe_options_SPX_2026-10-09.json").read_bytes()
    assert cap.sha256 == hashlib.sha256(raw).hexdigest()
    assert cap.n_bytes == len(raw)
    s = cap.summary()
    assert (s["session"], s["book_state"], s["fetched_at"], s["payload_timestamp"], s["quote_time"]) == \
        ("2026-10-09", "gth", "2026-10-12T08:15:28Z", "2026-10-12T08:15:07Z", "2026-10-12T08:00:07Z")
    assert (s["rows"], s["two_sided_rows"], s["expiries"], s["roots"]) == (17, 16, 3, ["SPX", "SPXW"])
    assert s["underlying_last_trade_time"] == "2026-10-09T16:14:59"
    assert s["underlying_price"] == 7811.54
    assert s["stated_delay_minutes"] == 15
    assert s["url"].endswith("/options/_SPX.json")
    json.dumps(s)                                                          # JSON-ready


def test_capture_extras_are_optional_and_never_stored(serve_chains, tmp_path, monkeypatch):
    cap = cb.capture_option_chain("SPX", now=CAPTURED_AT, include_extras=True)
    extra = {"bid_size", "ask_size", "theo", "rho", "last_trade_price", "option_last_trade_time"}
    assert extra <= set(cap.frame.columns)
    assert len(cap.frame) == 17
    row = cap.frame.set_index("contract_symbol").loc["SPX261016P00200000"]
    assert (row["ask_size"], row["theo"], row["option_last_trade_time"]) == (351.0, 0.0178, "2026-09-22T11:05:39")
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "t.db")
    db.init_database()
    assert db.insert_options_snapshot(cap.frame, "SPX") == 17               # extras ignored by the schema


@pytest.mark.parametrize(("utc", "state"), [
    ("2026-10-09T15:00:00Z", "intraday"),      # Fri 11:00 ET
    ("2026-10-09T20:10:00Z", "intraday"),      # Fri 16:10 ET, before the 16:15 cutoff
    ("2026-10-09T20:20:00Z", "closing"),       # Fri 16:20 ET
    ("2026-10-09T21:00:00Z", "closing"),       # Fri 17:00 ET, after the 16:15 cutoff
    ("2026-10-10T16:00:00Z", "closing"),       # Saturday
    ("2026-10-12T00:30:00Z", "gth"),           # Sun 20:30 ET: GTH for Monday 10-12
    ("2026-10-12T13:20:00Z", "pre_open"),      # Mon 09:20 ET
])
def test_book_state(utc, state):
    when = datetime.fromisoformat(utc)
    assert cb.book_state_at(when, "2026-10-09") == state


def test_book_state_gth_agrees_with_the_calibrators_t_origin():
    """The capture's 'gth' label and option_chain.t_origin (which measures T
    from the quote time for a GTH book) must classify every instant alike."""
    from joint_vol_calibration.data import option_chain as oc
    for session in ("2026-10-09", "2026-11-25", "2026-12-31", "2027-01-15"):  # Fri, pre-holiday, NYE, pre-MLK
        start = pd.Timestamp(session + " 16:30", tz="America/New_York")
        for k in range(96 * 4):                        # 4 days in 15-minute steps
            when = (start + pd.Timedelta(minutes=15 * k)).to_pydatetime()
            if db.quote_session(when) != session and db.is_intraday(when):
                break                                         # the next session has opened
            gth = cb.book_state_at(when, session) == "gth"
            from_quote_time = oc.t_origin(db.utc_iso(when), session) != oc.session_close(session)
            assert gth == from_quote_time, (session, when)


@pytest.mark.parametrize(("stamp_utc", "fetched_state", "quoted_state"), [
    ("2026-10-12T00:25:00", "gth", "closing"),          # Sun 20:25 ET file shows 20:10 ET: GTH not open yet
    ("2026-10-09T20:25:00", "closing", "intraday"),     # Fri 16:25 ET file shows 16:10 ET: before the cutoff
    ("2026-10-12T13:40:00", "closing", "pre_open"),     # Mon 09:40 ET file shows 09:25 ET: RTH not open yet
    ("2026-10-12T08:15:07", "gth", "gth"),              # Mon 04:15 ET file shows 04:00 ET
])
def test_capture_book_state_is_classified_at_the_quote_instant(stamp_utc, fetched_state, quoted_state):
    """A delayed book is the market QUOTE_DELAY before the file was stamped:
    the label must follow the quotes (and so option_chain.t_origin of the
    frame's quote_time), not the fetch clock."""
    from joint_vol_calibration.data import option_chain as oc
    payload = _payload("SPX")
    payload["timestamp"] = stamp_utc.replace("T", " ")
    stamped = datetime.fromisoformat(stamp_utc).replace(tzinfo=UTC)
    cap = cb._capture_from_payload("SPX", payload, b"{}", "u", stamped)
    assert cb.book_state_at(stamped, cap.session) == fetched_state
    assert cap.book_state == quoted_state
    gth = oc.t_origin(cap.quote_time, cap.session) != oc.session_close(cap.session)
    assert gth == (quoted_state == "gth")


def test_capture_chains_writes_only_where_told(serve_chains, tmp_path, caplog):
    caps = cb.capture_option_chains(("SPX", "VIX"), now=CAPTURED_AT)
    assert set(caps) == {"SPX", "VIX"}
    assert not list(tmp_path.iterdir())
    out = tmp_path / "captures"
    with caplog.at_level(logging.WARNING):
        caps = cb.capture_option_chains(("SPX", "NDX", "VIX"), now=CAPTURED_AT, out_dir=out)
    assert set(caps) == {"SPX", "VIX"}
    assert "NDX chain capture failed" in caplog.text
    names = sorted(p.name for p in out.iterdir())
    assert names == ["spx_2026-10-09_081528Z.json", "spx_2026-10-09_081528Z.parquet",
                     "vix_2026-10-09_081528Z.json", "vix_2026-10-09_081528Z.parquet"]
    back = pd.read_parquet(out / "vix_2026-10-09_081528Z.parquet")
    assert len(back) == len(caps["VIX"].frame)
    assert set(back["root"]) <= {"VIX", "VIXW"}
    meta = json.loads((out / "spx_2026-10-09_081528Z.json").read_text())
    assert meta["sha256"] == caps["SPX"].sha256
