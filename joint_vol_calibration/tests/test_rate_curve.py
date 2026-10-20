"""
test_rate_curve.py — the C23 Treasury zero curve (data/rates.py).

Offline. The Treasury CSV and NY Fed JSON samples below are verbatim rows of
the live files fetched 2026-10-20 (Treasury fixture names cannot start with
cboe_/fred_, so they are inline); tests/fixtures/fred_graph_DGS_DTB3_SOFR_2026-10.csv
is a trimmed capture of fred.stlouisfed.org/graph/fredgraph.csv. The real-data
tests read the committed partitions (data_store/market/rates_daily/) and the
committed ^IRX history through the suite's private copy of data_store.
"""
import json
import math
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from joint_vol_calibration import config
from joint_vol_calibration.data import database as db
from joint_vol_calibration.data import rates

FIX = Path(__file__).parent / "fixtures"

BILLS_2026 = '''Date,"4 WEEKS BANK DISCOUNT","4 WEEKS COUPON EQUIVALENT","6 WEEKS BANK DISCOUNT","6 WEEKS COUPON EQUIVALENT","8 WEEKS BANK DISCOUNT","8 WEEKS COUPON EQUIVALENT","13 WEEKS BANK DISCOUNT","13 WEEKS COUPON EQUIVALENT","17 WEEKS BANK DISCOUNT","17 WEEKS COUPON EQUIVALENT","26 WEEKS BANK DISCOUNT","26 WEEKS COUPON EQUIVALENT","52 WEEKS BANK DISCOUNT","52 WEEKS COUPON EQUIVALENT"
10/19/2026,3.94,4.01,3.95,4.02,4.01,4.09,4.10,4.20,4.15,4.27,4.23,4.38,4.27,4.47
10/16/2026,3.95,4.02,3.96,4.03,4.02,4.10,4.08,4.18,4.14,4.26,4.20,4.35,4.24,4.44
10/09/2026,3.96,4.03,3.96,4.03,3.97,4.05,4.06,4.16,4.11,4.22,4.18,4.33,4.26,4.46
'''
PAR_2026 = '''Date,"1 Mo","1.5 Month","2 Mo","3 Mo","4 Mo","6 Mo","1 Yr","2 Yr","3 Yr","5 Yr","7 Yr","10 Yr","20 Yr","30 Yr"
10/19/2026,4.11,4.14,4.17,4.27,4.34,4.37,4.50,4.77,4.85,4.94,5.04,5.15,5.53,5.48
10/16/2026,4.12,4.14,4.17,4.26,4.32,4.34,4.46,4.74,4.82,4.89,4.98,5.09,5.47,5.43
10/09/2026,4.13,4.13,4.13,4.25,4.29,4.32,4.47,4.80,4.89,5.02,5.13,5.24,5.65,5.60
'''
BILLS_2010 = '''Date,"4 WEEKS BANK DISCOUNT","4 WEEKS COUPON EQUIVALENT","13 WEEKS BANK DISCOUNT","13 WEEKS COUPON EQUIVALENT","26 WEEKS BANK DISCOUNT","26 WEEKS COUPON EQUIVALENT","52 WEEKS BANK DISCOUNT","52 WEEKS COUPON EQUIVALENT"
12/31/2010,0.06,0.06,0.12,0.12,0.19,0.19,0.27,0.27
'''
SOFR_JSON = {"refRates": [
    {"effectiveDate": "2026-10-19", "type": "SOFR", "percentRate": 3.87, "volumeInBillions": 2888},
    {"effectiveDate": "2026-10-16", "type": "SOFR", "percentRate": 3.88, "volumeInBillions": 2925},
    {"effectiveDate": "2026-10-15", "type": "SOFR", "percentRate": 3.89, "volumeInBillions": 3025}]}


def _quotes() -> pd.DataFrame:
    return pd.concat([rates.parse_treasury_csv(BILLS_2026, "daily_treasury_bill_rates"),
                      rates.parse_treasury_csv(PAR_2026, "daily_treasury_yield_curve"),
                      rates.parse_nyfed_sofr(SOFR_JSON)], ignore_index=True)


def _day(q, date):
    return q[q["date"] == date].set_index("series")["value"]


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """A fresh database and an empty partition root; rates' caches cleared."""
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "r.db")
    monkeypatch.setattr(db, "MARKET_DIR", tmp_path / "market")
    db.init_database()
    rates._QUOTES_CACHE.clear()
    rates._curve_cache.clear()
    rates._index_cache.clear()
    yield tmp_path
    rates._QUOTES_CACHE.clear()
    rates._curve_cache.clear()
    rates._index_cache.clear()


# ── parsing ───────────────────────────────────────────────────────────────────

class TestParsing:
    def test_treasury_bill_file_maps_every_column_and_converts_percent(self):
        q = rates.parse_treasury_csv(BILLS_2026, "daily_treasury_bill_rates")
        assert len(q) == 3 * 14 and set(q["source"]) == {"treasury"}
        day = _day(q, "2026-10-19")
        assert day["UST_BILL_13W_BD"] == pytest.approx(0.0410) and day["UST_BILL_13W_CE"] == pytest.approx(0.0420)
        assert set(day.index) == {rates.bill_series(w, b) for w in rates.BILL_WEEKS for b in ("BD", "CE")}

    def test_older_files_carry_fewer_columns(self):
        q = rates.parse_treasury_csv(BILLS_2010, "daily_treasury_bill_rates")
        assert sorted(q["series"]) == sorted(rates.bill_series(w, b) for w in (4, 13, 26, 52) for b in ("BD", "CE"))

    def test_par_file_and_unknown_columns(self, caplog):
        q = rates.parse_treasury_csv(PAR_2026.replace('"30 Yr"', '"Mystery"'), "daily_treasury_yield_curve")
        assert "UST_PAR_1.5M" in set(q["series"]) and "UST_PAR_30Y" not in set(q["series"])
        assert "ignoring unknown column 'Mystery'" in caplog.text
        assert _day(q, "2026-10-16")["UST_PAR_2Y"] == pytest.approx(0.0474)

    def test_fred_ids_map_onto_the_treasury_series_and_gaps_are_dropped(self):
        q = rates.parse_fred_csv((FIX / "fred_graph_DGS_DTB3_SOFR_2026-10.csv").read_text())
        assert set(q["source"]) == {"fred"}
        assert "2026-10-12" not in set(q["date"])                    # Columbus Day: empty row
        assert set(q.loc[q["date"] == "2026-10-19", "series"]) == {"SOFR"}
        tsy = rates.parse_treasury_csv(PAR_2026, "daily_treasury_yield_curve")
        both = q.merge(tsy, on=["date", "series"], suffixes=("_fred", "_tsy"))
        assert len(both) == 10 and np.allclose(both["value_fred"], both["value_tsy"], rtol=0, atol=1e-15)

    def test_nyfed_sofr(self):
        q = rates.parse_nyfed_sofr(SOFR_JSON)
        assert q["date"].tolist() == ["2026-10-15", "2026-10-16", "2026-10-19"]
        assert np.allclose(q["value"], [0.0389, 0.0388, 0.0387], rtol=0, atol=1e-15)
        assert set(q["source"]) == {"nyfed"} and set(q["series"]) == {"SOFR"}


# ── instrument conversions ────────────────────────────────────────────────────

class TestConversions:
    def test_bank_discount_price(self):
        assert float(rates.price_from_discount(0.0410, 91)) == pytest.approx(1 - 0.0410 * 91 / 360, abs=1e-15)

    def test_coupon_equivalent_inverts_treasurys_formula(self):
        """Treasury's coupon-equivalent of a bill priced P (per 100) over n days:
        n <= 182: 365 d / (360 - d n) from the discount; n > 182: the root of
        (n/365 - 1/2)/2 y^2 + (n/365) y + (1 - 100/P) = 0."""
        for n, d in ((28, 0.0394), (91, 0.0410), (182, 0.0423), (364, 0.0427)):
            p = float(rates.price_from_discount(d, n))
            if n <= 182:
                ce = 365 * d / (360 - d * n)
            else:
                a = n / 365 - 0.5
                ce = (-2 * n / 365 + 2 * math.sqrt((n / 365) ** 2 - 2 * a * (1 - 1 / p))) / (2 * a)
            assert float(rates.price_from_coupon_equivalent(ce, n)) == pytest.approx(p, rel=1e-13)

    def test_both_published_quotes_of_a_bill_imply_the_same_rate(self):
        """Treasury rounds each quote to 0.01%: the bank-discount and coupon-
        equivalent prices of one bill agree to under 2 bp (all bills, every
        day 2010-2026: median 0.14-0.35 bp, max 1.98 bp; measured 2026-10-20)."""
        day = _day(rates.parse_treasury_csv(BILLS_2026, "daily_treasury_bill_rates"), "2026-10-19")
        for w in rates.BILL_WEEKS:
            n = 7 * w
            rb = rates.cc_rate(rates.price_from_discount(day[rates.bill_series(w, "BD")], n), n / 365)
            rc = rates.cc_rate(rates.price_from_coupon_equivalent(day[rates.bill_series(w, "CE")], n), n / 365)
            assert abs(rb - rc) < 2e-4, w

    def test_sofr_compounding(self):
        assert rates.sofr_to_cc(0.0388) == pytest.approx(math.log(1 + 0.0388 / 360) * 365, rel=1e-12)
        assert rates.sofr_to_cc(0.0388) == pytest.approx(0.0388 * 365 / 360, abs=3e-6)


# ── curve construction ────────────────────────────────────────────────────────

class TestCurve:
    def test_bill_nodes_are_exact_and_r91_is_the_13_week_bill(self):
        c = rates.build_zero_curve(_quotes(), "2026-10-19")
        assert c.curve_date == "2026-10-19" and c.short_end == "bills"
        r13 = -math.log(1 - 0.0410 * 91 / 360) * 365 / 91
        assert c.zero_rate(91 / 365) == pytest.approx(r13, abs=1e-12)
        for w in rates.BILL_WEEKS:
            n = 7 * w
            d = _day(_quotes(), "2026-10-19")[rates.bill_series(w, "BD")]
            assert c.discount_factor(n / 365) == pytest.approx(1 - d * n / 360, abs=1e-14)

    def test_log_linear_interpolation_is_piecewise_flat_forward(self):
        c = rates.build_zero_curve(_quotes(), "2026-10-19")
        t1, t2 = c.t[2], c.t[3]                                       # 8W .. 13W
        f = c.forward_rate(t1, t2)
        for a in np.linspace(0, 1, 7)[1:-1]:
            tm = t1 + a * (t2 - t1)
            assert c.forward_rate(t1, tm) == pytest.approx(f, abs=1e-12)
        T = np.array([1 / 365, 7 / 365, 0.25, 0.75, 1.5, 2.0, 10.0, 30.0, 40.0])
        assert np.allclose(c.zero_rate(T) * T, -np.log(c.discount_factor(T)), atol=1e-14)
        assert np.all(np.diff(c.discount_factor(np.linspace(0, 40, 400))) < 0)
        assert c.discount_factor(0.0) == 1.0 and c.discount_factor(-1.0) == 1.0

    def test_front_is_flat_at_the_first_bill_and_back_is_flat_forward(self):
        c = rates.build_zero_curve(_quotes(), "2026-10-19")
        r4 = c.zero_rate(28 / 365)
        assert c.zero_rate(1 / 365) == pytest.approx(r4, abs=1e-13) and c.zero_rate(0.0) == pytest.approx(r4)
        f_last = c.forward_rate(c.t[-2], c.t[-1])
        assert c.forward_rate(30.0, 35.0) == pytest.approx(f_last, abs=1e-12)

    def test_flat_par_curve_bootstraps_to_the_semiannual_zero(self):
        """A flat par yield y at every tenor (6M and 1Y as coupon-equivalents,
        2Y-30Y as semiannual par bonds) is the zero curve P(t) = (1 + y/2)^-2t
        on the 6-month grid: the bootstrap must reproduce it to rounding."""
        y = 0.045
        rows = [("2026-10-16", s, y, "treasury") for s, m in rates.PAR_MONTHS.items() if m in (6, 12) or m >= 24]
        c = rates.build_zero_curve(pd.DataFrame(rows, columns=["date", "series", "value", "source"]), "2026-10-16")
        assert c.short_end == "par"
        grid = np.arange(0.5, 30.01, 0.5)
        assert np.allclose(c.discount_factor(grid), (1 + y / 2) ** (-2 * grid), rtol=0, atol=1e-13)
        assert c.zero_rate(10.0) == pytest.approx(2 * math.log1p(y / 2), abs=1e-13)

    def test_too_few_bills_fall_back_to_the_par_short_end(self):
        q = _quotes()
        q = q[~q["series"].str.startswith("UST_BILL_") | (q["series"] == "UST_BILL_13W_BD")]
        c = rates.build_zero_curve(q, "2026-10-19")
        assert c.short_end == "par" and c.nodes["instrument"].iloc[0] == "UST_PAR_1M"
        y = 0.0411
        assert c.discount_factor(1 / 12) == pytest.approx(1 / (1 + y * (365 / 12) / 365), abs=1e-14)

    def test_the_bill_and_par_short_ends_disagree_in_2026(self):
        """Not a bug, a fact about the two Treasury products, kept visible: on
        2026-10-19 the 13-week bill's coupon equivalent is 4.20% and the
        3-month par yield 4.27% (median par-minus-bill gap at 3 months
        +0.25 bp 2018-01-02..2025-03-24, +4.8 bp after; measured 2026-10-20)."""
        bills = rates.build_zero_curve(_quotes(), "2026-10-19")
        q = _quotes()
        par = rates.build_zero_curve(q[~q["series"].str.startswith("UST_BILL_")], "2026-10-19")
        gap_bp = (par.zero_rate(0.25) - bills.zero_rate(0.25)) * 1e4
        assert 5.0 < gap_bp < 12.0


# ── look-ahead gating ─────────────────────────────────────────────────────────

class TestLookAhead:
    def test_one_quote_date_on_or_before_as_of(self):
        q = _quotes()
        assert rates.build_zero_curve(q, "2026-10-16").curve_date == "2026-10-16"
        assert rates.build_zero_curve(q, "2026-10-18").curve_date == "2026-10-16"     # Sunday
        assert rates.build_zero_curve(q, "2026-10-12").curve_date == "2026-10-09"     # Columbus Day
        assert rates.build_zero_curve(q, "2026-10-19", same_day=False).curve_date == "2026-10-16"
        with pytest.raises(LookupError):
            rates.build_zero_curve(q, "2026-10-08")
        with pytest.raises(db.StaleDataError):
            rates.build_zero_curve(q, "2026-10-30")

    def test_future_quotes_never_move_the_curve(self):
        q = _quotes()
        base = rates.build_zero_curve(q, "2026-10-16")
        poison = q[q["date"] == "2026-10-19"].assign(value=0.25)        # an absurd later print
        moved = rates.build_zero_curve(pd.concat([q[q["date"] < "2026-10-19"], poison]), "2026-10-16")
        T = np.linspace(0.01, 30, 50)
        assert np.array_equal(base.zero_rate(T), moved.zero_rate(T))
        # ...while poisoning the same day does move it (the gate is not vacuous)
        same = pd.concat([q[q["date"] != "2026-10-16"], q[q["date"] == "2026-10-16"].assign(value=0.25)])
        assert not np.allclose(rates.build_zero_curve(same, "2026-10-16").zero_rate(T), base.zero_rate(T))

    def test_sofr_is_used_only_once_published(self, tmp_store):
        db.insert_rates_daily(rates.parse_nyfed_sofr(SOFR_JSON))
        assert rates.sofr("2026-10-19")["date"] == "2026-10-16"         # 10-19's SOFR is out on 10-20
        assert rates.sofr("2026-10-20")["date"] == "2026-10-19"
        s = rates.sofr("2026-10-17")
        assert (s["date"], s["sofr"]) == ("2026-10-16", pytest.approx(0.0388))
        assert s["notice"] == rates.SOFR_NOTICE
        with pytest.raises(LookupError):
            rates.sofr("2026-10-15")

    def test_sofr_notice_carries_the_ny_fed_disclaimer(self):
        """The NY Fed Terms of Use require the notice AND the disclaimer (its
        template, "Use Restrictions" / "Reference Rates", read 2026-10-20)."""
        n = rates.SOFR_NOTICE
        assert n.startswith("The SOFR Data is subject to the Terms of Use posted at newyorkfed.org. ")
        assert "The New York Fed is not responsible for publication of the SOFR by " in n
        assert n.endswith("does not sanction or endorse any particular republication, "
                          "and has no liability for your use.")
        assert "[" not in n


class TestFredDownload:
    def test_ids_are_requested_twelve_at_a_time_and_a_dropped_id_raises(self, monkeypatch):
        """FRED's graph CSV serves at most 12 ids per request and silently
        drops the rest (measured 2026-10-20): download_fred chunks, and an id
        missing from a response is an error, not a silent gap."""
        text = (FIX / "fred_graph_DGS_DTB3_SOFR_2026-10.csv").read_text()
        cols = text.splitlines()[0].split(",")
        seen = []

        def fake_get(url, **kw):
            ids = url.split("id=", 1)[1].split(",")
            seen.append(ids)
            # serve the fixture's columns for the ids it has, fabricate none
            keep = [0] + [cols.index(i) for i in ids if i in cols]
            return "\n".join(",".join(r.split(",")[j] for j in keep) for r in text.splitlines()) + "\n"
        monkeypatch.setattr(rates, "_get_text", fake_get)
        ids = ["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DTB3", "SOFR"] * 2 + ["DGS1MO"]
        q = rates.download_fred(ids, start="2026-10-13")
        assert seen == [["DGS1MO", "DGS3MO", "DGS6MO", "DGS1", "DGS2", "DTB3", "SOFR"]]   # de-duplicated
        assert q["date"].min() == "2026-10-13" and "H15_BILL_3M_BD" in set(q["series"])
        many = [f"X{i}" for i in range(13)]
        seen.clear()
        with pytest.raises(ValueError, match="lacks requested series"):
            rates.download_fred(many)
        assert [len(c) for c in seen] == [12]
        assert rates.FRED_MAX_IDS_PER_REQUEST == 12


# ── the store: database rows over committed partitions ────────────────────────

class TestStore:
    def test_partitions_alone_serve_the_curve_and_the_database_wins(self, tmp_store):
        db.insert_rates_daily(_quotes())
        db.export_market_partitions(tables=["rates_daily"])
        con = sqlite3.connect(db.DB_PATH)
        con.execute("DELETE FROM rates_daily")
        con.commit()
        con.close()
        r = rates.zero_rate("2026-10-19", 91 / 365)
        assert r == pytest.approx(-math.log(1 - 0.0410 * 91 / 360) * 365 / 91, abs=1e-12)
        db.insert_rates_daily(pd.DataFrame({"date": ["2026-10-19"], "series": ["UST_BILL_13W_BD"],
                                            "value": [0.0450], "source": ["treasury"]}))
        assert rates.zero_rate("2026-10-19", 91 / 365) == pytest.approx(
            -math.log(1 - 0.0450 * 91 / 360) * 365 / 91, abs=1e-12)                 # cache invalidated
        assert rates.discount_factor("2026-10-19", 0.0) == 1.0

    def test_refresh_uses_treasury_and_nyfed_but_not_fred_by_default(self, tmp_store, monkeypatch):
        seen = []

        def fake_get(url, **kw):
            seen.append(url)
            if "daily_treasury_bill_rates" in url:
                return BILLS_2026
            if "daily_treasury_yield_curve" in url:
                return PAR_2026
            if "newyorkfed" in url:
                return json.dumps(SOFR_JSON)
            raise AssertionError(url)
        monkeypatch.setattr(rates, "_get_text", fake_get)
        res = rates.refresh_rates(2026, 2026)
        assert res == {"treasury": 3 * 14 + 3 * 14, "nyfed": 3, "fred": 0, "errors": []}
        assert not any("stlouisfed" in u for u in seen) and len(seen) == 3


# ── against independent real data (committed record) ──────────────────────────

def _committed_quotes():
    root = Path(db.MARKET_DIR)
    q = db.read_market_partitions("rates_daily", root)
    if q.empty:
        pytest.skip("no committed rates_daily partitions")
    return q


class TestCommittedRecord:
    def test_r91_tracks_the_independent_13_week_bill_print(self):
        """r(91 d) from the Treasury bills against Yahoo's ^IRX (another source
        and quote time, the rate the system used before C23) over every
        session 2018-01-02..2026-10-16: measured median +1.23 bp, 95th pct
        5.8 bp, worst 26 bp on 2022-09-19 (a bad ^IRX print, also 25.5 bp off
        FRED's DTB3)."""
        idx = rates.index_quotes(_committed_quotes())
        con = sqlite3.connect(f"file:{config.DB_PATH}?mode=ro", uri=True)
        irx = pd.read_sql("SELECT date, rate FROM tbill_rates WHERE date >= '2018-01-02' AND date <= '2026-10-16'",
                          con).set_index("date")["rate"]
        con.close()
        diff_bp = np.array([(rates.build_zero_curve(idx, d).zero_rate(91 / 365) - r) * 1e4
                            for d, r in irx.items()])
        assert len(diff_bp) > 2000
        assert abs(np.median(diff_bp)) < 2.0 and np.percentile(np.abs(diff_bp), 95) < 8.0

    def test_a_curve_exists_for_every_session_of_the_backtest_window(self):
        idx = rates.index_quotes(_committed_quotes())
        sessions = db.nyse_sessions("2018-01-02", config.BT_END)
        aged = {}
        for d in sessions:
            c = rates.build_zero_curve(idx, d.strftime("%Y-%m-%d"))
            if c.curve_date != d.strftime("%Y-%m-%d"):
                aged[d] = (d - pd.Timestamp(c.curve_date)).days
        # the bond market closes on Columbus Day and Veterans Day (observed) while
        # the NYSE trades: those sessions, and only those, use an older curve
        assert len(aged) == 13 and max(aged.values()) <= 3
        for d in aged:
            columbus = d.month == 10 and d.weekday() == 0 and 8 <= d.day <= 14
            veterans = d.month == 11 and 10 <= d.day <= 12
            assert columbus or veterans, d

    def test_interpolation_error_without_the_13_week_node(self):
        """Drop the 13-week bill and interpolate it from its neighbours (the
        8-week from 2018-10, else the 4-week; the 17-week from 2022-10, else
        the 26-week): measured median 1.65 bp, 95th pct 13.9 bp over every 5th
        date 2018-2026 (n=440)."""
        idx = rates.index_quotes(_committed_quotes())
        errs = []
        for d in idx.dates[(idx.dates >= "2018-01-02") & (idx.dates <= "2026-10-19")][::5]:
            full = rates.build_zero_curve(idx, d)
            day = idx.by_date[d].drop(["UST_BILL_13W_BD", "UST_BILL_13W_CE"])
            loo = rates.build_zero_curve(rates.QuoteIndex(np.array([d], dtype=object), {d: day}), d)
            errs.append(abs(loo.zero_rate(91 / 365) - full.zero_rate(91 / 365)) * 1e4)
        assert np.median(errs) < 3.0 and np.percentile(errs, 95) < 25.0


# ── parity-implied rates ──────────────────────────────────────────────────────

class TestParityComparison:
    def test_only_regression_rows_are_compared_in_basis_points(self):
        c = rates.build_zero_curve(_quotes(), "2026-10-16")
        carry = pd.DataFrame({"expiry": ["2026-12-18", "2027-01-15", "2026-10-30"],
                              "T": [63 / 365, 91 / 365, 14 / 365], "r": [0.0450, 0.0410, 0.0400],
                              "method": ["parity_regression", "parity_regression", "parity_fixed_D"],
                              "n_pairs": [40, 31, 9], "resid_rmse": [0.11, 0.09, 0.4]})
        out = rates.parity_rate_comparison(carry, "2026-10-16", curve=c, sofr_cc=rates.sofr_to_cc(0.0388))
        assert out["expiry"].tolist() == ["2026-12-18", "2027-01-15"]
        assert out["spread_bills_bp"].iloc[1] == pytest.approx((0.0410 - c.zero_rate(91 / 365)) * 1e4)
        assert out["spread_sofr_bp"].iloc[0] == pytest.approx((0.0450 - rates.sofr_to_cc(0.0388)) * 1e4)
        assert set(out["curve_date"]) == {"2026-10-16"}
        assert out["n_pairs"].tolist() == [40, 31] and out["days"].tolist() == [63.0, 91.0]

    def test_summary_buckets_are_half_open_medians_and_iqrs(self):
        cmp_ = pd.DataFrame({"as_of": ["a"] * 6 + ["b"] * 2,
                             "days": [5, 29.99, 30, 90, 91, 364.9, 365, 800],
                             "spread_bills_bp": [-100, 50, 10, 30, 40, 44, 20, 60.0]})
        out = rates.summarize_parity_comparison(cmp_).set_index("bucket")
        assert out.index.tolist() == ["0-30d", "30-91d", "91-365d", ">=365d"]
        assert out["n"].tolist() == [2, 2, 2, 2] and out.loc[">=365d", "sessions"] == 1
        assert out.loc["0-30d", "spread_bills_bp_median"] == pytest.approx(-25.0)
        assert out.loc["30-91d", "spread_bills_bp_iqr"] == pytest.approx(10.0)      # 25th/75th of {10, 30}
        assert out.loc[">=365d", "spread_bills_bp_median"] == pytest.approx(40.0)

    def test_committed_comparison_artifact_reproduces(self):
        """data_store/market/derived/spx_parity_rates_vs_treasury.csv is
        committed_chain_comparison() of the committed chains and curve."""
        path = Path(db.MARKET_DIR) / "derived" / "spx_parity_rates_vs_treasury.csv"
        if not path.exists():
            pytest.skip("artifact not committed")
        committed = pd.read_csv(path, dtype={"as_of": str, "curve_date": str, "expiry": str})
        rates._QUOTES_CACHE.clear()
        rates._curve_cache.clear()
        rates._index_cache.clear()
        fresh = rates.committed_chain_comparison(["2026-03-24", "2026-10-16"])
        sub = committed[committed["as_of"].isin(["2026-03-24", "2026-10-16"])].reset_index(drop=True)
        assert len(fresh) == len(sub) > 0
        num = ["T", "r_parity", "r_bills", "spread_bills_bp", "r_par", "spread_par_bp", "n_pairs", "r_irx"]
        assert (fresh["expiry"].tolist(), fresh["curve_date"].tolist()) == (sub["expiry"].tolist(),
                                                                            sub["curve_date"].tolist())
        assert np.allclose(fresh[num].to_numpy(float), sub[num].to_numpy(float), rtol=1e-6, atol=1e-6)
        assert fresh["calibrator_uses_parity"].tolist() == sub["calibrator_uses_parity"].tolist()
        # the flag is the calibrator's own rule (option_chain.PARITY_RATE_TOL around ^IRX)
        from joint_vol_calibration.data import option_chain as oc
        assert ((fresh["r_parity"] - fresh["r_irx"]).abs() <= oc.PARITY_RATE_TOL).tolist() == \
            fresh["calibrator_uses_parity"].tolist()

    def test_committed_summary_is_the_summary_of_the_committed_comparison(self):
        d = Path(db.MARKET_DIR) / "derived"
        if not (d / "spx_parity_rates_vs_treasury_summary.csv").exists():
            pytest.skip("artifact not committed")
        full = pd.read_csv(d / "spx_parity_rates_vs_treasury.csv")
        committed = pd.read_csv(d / "spx_parity_rates_vs_treasury_summary.csv")
        fresh = rates.summarize_parity_comparison(full)
        assert committed["bucket"].tolist() == fresh["bucket"].tolist()
        cols = [c for c in committed.columns if c != "bucket"]
        assert np.allclose(committed[cols].to_numpy(float), fresh[cols].to_numpy(float), atol=0.006)
        # beyond 3 months the options' box rate sits above the bill curve on every committed session
        long_ = full[full["days"] >= 91]
        assert (long_.groupby("as_of")["spread_bills_bp"].median() > 0).all()
