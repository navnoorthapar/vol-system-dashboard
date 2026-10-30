#!/usr/bin/env python3
"""
readme_results.py — the committed snapshot of every figure the README quotes.

The README's results, tables, data counts and limitation figures, and the
validation pack's per-session figures, are computed from data_store/. The
daily job rebuilds some of those artifacts after they are quoted: a method
change raises surface.report.REPORT_VERSION and the job's backfill rebuilds
every committed session's surface report (2026-10-26 moves from 1.60 to 1.66
half-spreads when rebuilt from its Cboe book). Checked against the live
data_store, the README turned CI red on every pull request until someone
edited it. The documents therefore quote this snapshot,
docs/results/readme_results.json, and say when it was taken ('as of <date>');
the live site has the newest numbers.

  claims      every figure as the documents print it, in the sentence or
              table row that carries it, with the document, the part of it the
              text must be in, and the artifacts it was computed from
  artifacts   each artifact a claim read: its path under data_store/ (with a
              '#<selector>' when the claim read a part of it: one session's row
              of an aggregate file, a table's rows up to a date, a directory
              listing), its sha256, and its report_version (surface artifacts;
              null elsewhere)
  generated   the date the documents state ('as of'), the commit whose
              data_store it was computed from, and the command
  params      the last session the results range over and the end date of the
              Data table; a later run recomputes over the same span

joint_vol_calibration/tests/test_readme.py checks (a) that every claim is in
its document verbatim and both documents state the snapshot's date, always,
and (b) that the snapshot is what the current data_store gives. A figure that
no longer recomputes to its text fails when none of its artifacts changed (the
code or the snapshot did) or when one the daily job never writes changed (a
commit did that); when every artifact under it that changed is one the job
writes (a path its commit steps in .github/workflows/daily_refresh.yml stage,
or the cache every run rewrites), it is a warning naming the rebuilt artifacts
and this command, never a failure.

A claim whose wording no longer holds (every chain a Yahoo download, after a
rebuild from a Cboe book) cannot be regenerated: this script stops and says
which, so the sentence is reworded here and in the document. The claims about
the C23 round (its five sessions' DE timings and option counts) stay pinned to
those sessions.

  python scripts/readme_results.py                  # recompute over the snapshot's span; write it
  python scripts/readme_results.py --update-docs    # also rewrite the figures that moved in the documents
  python scripts/readme_results.py --through newest # move the span and the Data table to the newest data
  python scripts/readme_results.py --check          # compare only; exit 1 on a failure
  make readme-results [THROUGH=newest]              # the first two together
"""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import pickle
import re
import sqlite3
import subprocess
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DATA = ROOT / "data_store"
SNAPSHOT = ROOT / "docs" / "results" / "readme_results.json"
WORKFLOW = ROOT / ".github" / "workflows" / "daily_refresh.yml"
COMMAND = "make readme-results"
SCHEMA_VERSION = 1
README = "README.md"
PACK = "docs/model_validation/README.md"
SOURCES_DOC = "DATA_SOURCES.md"
# where a claim's text must be: the README's first screen (above its first
# rule), a section from its heading to the next, or the one line that starts
# with a table row's first cells
FIRST_SCREEN = "first screen"
LIMITATIONS = "## Known limitations"
# the documents that state the snapshot's date, and where
AS_OF = ((README, FIRST_SCREEN), (PACK, ""))
# The cache every run of the daily job rewrites and never commits: each run
# restores the committed record into it, then stores the session in it (as
# daily_refresh.yml says where it commits the record; app.record_db does the
# same for a build). A figure it moves is the job's, like one a committed
# artifact moves: scripts/ci/simulate_next_session.py runs the suite on the
# tree the job leaves.
JOB_CACHE = ("data_store/vol_system.db",)
# The five sessions the C23 round measured: the claims about that round (the DE
# timings, the Data table's option counts) stay pinned to them.
C23_SESSIONS = ("2026-03-24", "2026-03-27", "2026-05-29", "2026-06-18", "2026-10-16")
C23_HEADLINE = "2026-10-16"
MINUS = "−"
WORDS = {2: "two", 3: "three", 4: "four", 5: "five", 6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten"}
MARKET_ROWS = (("SPX OHLCV", "spx_ohlcv"), ("VIX daily", "vix_daily"),
               ("VIX term structure", "vix_term_structure"), ("T-bill rate", "tbill_rates"),
               ("Treasury curve and SOFR", "rates_daily"))


def fmt(x: float, spec: str) -> str:
    return format(x, spec).replace("-", MINUS)


def rng(values: Iterable[float], spec: str) -> str:
    v = list(values)
    return f"{fmt(min(v), spec)}–{fmt(max(v), spec)}"


def word(n: int) -> str:
    return WORDS.get(n, str(n))


def usd(x: float) -> str:
    return f"{MINUS if x < 0 else '+'}${abs(x):,.0f}"


def collapse(text: str) -> str:
    """Whitespace collapsed: a re-wrapped paragraph must not fail a claim."""
    return " ".join(text.split())


class Untrue(Exception):
    """The document's wording no longer holds for the data: reword the sentence
    and its template here."""


class StaleSnapshot(UserWarning):
    """Artifacts the snapshot was computed from have changed since it was
    taken, and only artifacts the daily job writes moved a figure."""


# ── reading data_store, with what each claim read ─────────────────────────────

def _sha(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _canonical(obj: Any) -> bytes:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode()


class Store:
    """data_store/ as the snapshot reads it. Every read made while a claim is
    computed is recorded against the claim: the artifact's key, sha256 and
    report_version. A key is a path under data_store/, with '#<selector>' when
    the claim read a part of it (one session's row of an aggregate file the job
    rewrites daily, a table's rows up to a date, a directory listing).

    The β estimate behind one limitation reads the database at config.DB_PATH
    (the suite's private copy under pytest), not ``data``."""

    def __init__(self, data: Path = DATA) -> None:
        self.data = data
        self._parsed: dict[str, Any] = {}
        self._digests: dict[str, dict[str, Any]] = {}
        self.reads: dict[str, dict[str, Any]] | None = None

    def _note(self, key: str, sha: str, report_version: int | None = None) -> None:
        entry = {"sha256": sha, "report_version": report_version}
        self._digests[key] = entry
        if self.reads is not None:
            self.reads[key] = entry

    @staticmethod
    def _report_version(rel: str, doc: Any) -> int | None:
        if not rel.startswith("surface/"):
            return None
        from joint_vol_calibration.surface.report import stored_report_version
        return stored_report_version(doc)

    def _load(self, rel: str, parse: Callable[[Path], Any]) -> Any:
        path = self.data / rel
        if rel not in self._parsed:
            self._parsed[rel] = parse(path)
            self._digests.setdefault(f"data_store/{rel}", {"sha256": _sha(path.read_bytes()),
                                                           "report_version": None})
        return self._parsed[rel]

    def _whole(self, rel: str, report_version: int | None = None) -> None:
        self._note(f"data_store/{rel}", self._digests[f"data_store/{rel}"]["sha256"], report_version)

    def json(self, rel: str) -> Any:
        doc = self._load(rel, lambda p: json.loads(p.read_text()))
        self._whole(rel, self._report_version(rel, doc))
        return doc

    def pickle(self, rel: str) -> Any:
        def load(p: Path) -> Any:
            with open(p, "rb") as fh:
                return pickle.load(fh)  # noqa: S301 (the repository's own artifacts)
        obj = self._load(rel, load)
        self._whole(rel)
        return obj

    def parquet(self, rel: str) -> pd.DataFrame:
        df = self._load(rel, pd.read_parquet)
        self._whole(rel)
        return df

    def csv(self, rel: str) -> pd.DataFrame:
        df = self._load(rel, pd.read_csv)
        self._whole(rel)
        return df

    def file(self, rel: str) -> Path:
        """A file read by other code (the β estimate reads the database)."""
        self._load(rel, lambda p: None)
        self._whole(rel)
        return self.data / rel

    def row(self, rel: str, rows: str, session: str) -> dict[str, Any]:
        """One session's row of an aggregate file the job rewrites as sessions
        are added (surface/index.json, the diagnostics summary)."""
        doc = self._load(rel, lambda p: json.loads(p.read_text()))
        hit = [r for r in doc[rows] if r.get("session") == session]
        if len(hit) != 1:
            raise KeyError(f"{rel} has {len(hit)} rows for {session}")
        self._note(f"data_store/{rel}#session={session}", _sha(_canonical(hit[0])),
                   self._report_version(rel, hit[0]))
        return hit[0]

    def market(self, table: str, through: str) -> pd.DataFrame:
        """A market table's rows dated on or before ``through`` (the job appends
        later rows; a restated earlier row changes the digest). A table with no
        partition yet has no rows, and a digest that changes when the job
        writes its first one."""
        key = f"market/{table}#date<={through}"
        if key not in self._parsed:
            files = sorted((self.data / "market" / table).glob("*.parquet"))
            df = (pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files
                  else pd.DataFrame({"date": pd.Series([], dtype=str)}))
            df = df[pd.to_datetime(df["date"]) <= pd.Timestamp(through)]
            df = df.sort_values(list(df.columns)).reset_index(drop=True)
            digest = _sha(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
            self._parsed[key] = df
            self._digests[f"data_store/{key}"] = {"sha256": digest, "report_version": None}
        self._note(f"data_store/{key}", self._digests[f"data_store/{key}"]["sha256"])
        return self._parsed[key]

    def sql(self, name: str, query: str, params: tuple[Any, ...]) -> list[tuple[Any, ...]]:
        """Rows of the database cache (data_store/vol_system.db), read only."""
        key = f"vol_system.db#{name}"
        if key not in self._parsed:
            uri = f"file:{self.data / 'vol_system.db'}?mode=ro"
            with sqlite3.connect(uri, uri=True) as con:
                rows = [tuple(r) for r in con.execute(query, params).fetchall()]
            self._parsed[key] = rows
            self._digests[f"data_store/{key}"] = {"sha256": _sha(_canonical(rows)), "report_version": None}
        self._note(f"data_store/{key}", self._digests[f"data_store/{key}"]["sha256"])
        return self._parsed[key]

    def derived(self, key: str, compute: Callable[[], Any]) -> Any:
        """A fact derived from a directory listing (which sessions qualify, the
        first capture): the digest is of the fact itself."""
        if key not in self._parsed:
            self._parsed[key] = compute()
            self._digests[f"data_store/{key}"] = {"sha256": _sha(_canonical(self._parsed[key])),
                                                  "report_version": None}
        self._note(f"data_store/{key}", self._digests[f"data_store/{key}"]["sha256"])
        return self._parsed[key]

    def digest(self, key: str) -> dict[str, Any] | None:
        """The current digest of ``key`` if this run read it; for a whole file,
        the file's (None: missing, or a part no claim of this run read)."""
        if key in self._digests:
            return self._digests[key]
        path = self.data / key.removeprefix("data_store/")
        if "#" in key or not path.is_file():
            return None
        return {"sha256": _sha(path.read_bytes()), "report_version": None}


# ── claims ────────────────────────────────────────────────────────────────────

@dataclass
class Claim:
    id: str
    doc: str = README
    within: str = ""
    absent: bool = False
    text: str | None = None
    error: str | None = None
    sources: dict[str, dict[str, Any]] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        out: dict[str, Any] = {"id": self.id, "doc": self.doc}
        if self.within:
            out["within"] = self.within
        if self.absent:
            out["absent"] = True
        out["text"] = self.text
        out["sources"] = sorted(self.sources)
        return out


class _Computing:
    """``with claims.claim(...) as k: k.text = ...``: records what the block
    read; an exception inside it (Untrue, or a missing or reshaped artifact)
    becomes the claim's error instead of stopping the other claims."""

    def __init__(self, claims: Claims, claim: Claim) -> None:
        self.claims, self.claim = claims, claim

    def __enter__(self) -> Claim:
        self.claims.store.reads = {}
        return self.claim

    def __exit__(self, et: type[BaseException] | None, ev: BaseException | None,
                 tb: TracebackType | None) -> bool:
        self.claim.sources = self.claims.store.reads or {}
        self.claims.store.reads = None
        if ev is not None:
            self.claim.error = str(ev) if isinstance(ev, Untrue) else f"{type(ev).__name__}: {ev}"
        elif self.claim.text is None:
            self.claim.error = "no text was set"
        self.claims.out.append(self.claim)
        return et is None or issubclass(et, Exception)


class Claims:
    def __init__(self, store: Store, params: dict[str, str]) -> None:
        self.store, self.params = store, params
        self.out: list[Claim] = []

    def claim(self, cid: str, doc: str = README, within: str = "", absent: bool = False) -> _Computing:
        return _Computing(self, Claim(cid, doc, within, absent))

    # what several claims read

    @property
    def sessions(self) -> tuple[str, ...]:
        """Every stored session through params['sessions_through'] with a
        surface scorecard and a model_version 2 fit: the sessions the results
        range over."""
        through = self.params["sessions_through"]

        def listing() -> list[str]:
            out = []
            for d in sorted((self.store.data / "surface").iterdir()):
                pkl = self.store.data / "calibrations" / f"joint_cal_{d.name}.pkl"
                if not (d.name <= through and (d / "scorecard.json").exists() and pkl.exists()):
                    continue
                with open(pkl, "rb") as fh:
                    if pickle.load(fh).get("model_version") == 2:  # noqa: S301 (the repository's own fits)
                        out.append(d.name)
            return out
        out = self.store.derived(
            f"surface#sessions with a scorecard and a model_version 2 fit, through {through}", listing)
        if not out:
            raise Untrue(f"no session through {through} has a scorecard and a model_version 2 fit")
        return tuple(out)

    @property
    def last(self) -> str:
        return self.sessions[-1]

    def cal(self, s: str) -> dict[str, Any]:
        return self.store.pickle(f"calibrations/joint_cal_{s}.pkl")

    def scorecard(self, s: str) -> dict[str, Any]:
        return {m["model"]: m for m in self.store.json(f"surface/{s}/scorecard.json")["models"]}

    def surface(self, s: str) -> dict[str, Any]:
        return self.store.json(f"surface/{s}/surface.json")

    def diag(self, s: str) -> dict[str, Any]:
        return self.store.row("calibrations/diagnostics/summary.json", "rows", s)

    def index(self, s: str) -> dict[str, Any]:
        return self.store.row("surface/index.json", "sessions", s)

    def inference(self) -> dict[str, Any]:
        return self.store.json("backtest/inference/summary.json")

    def metrics(self) -> dict[str, Any]:
        return self.store.pickle("backtest/full_results.pkl")["metrics"]

    def vx_futures(self, through: str) -> list[tuple[str, str]]:
        """(date, expiry_month) of every VX futures row dated on or before
        ``through`` in the record: the cache's table and the partitions the
        daily job commits from FUND_GRADE_PLAN 1.12 on (data_store/market/
        vix_futures_daily/), merged on (date, expiry_date) as app.record_db
        merges them. The cache alone is what was committed before the job
        stored its first VX session: a clean checkout never sees the sessions
        the job commits, while the job's own runs add them to it."""
        rows = {(d, e): m for d, e, m in self.store.sql(
            f"vix_futures_daily dated on or before {through}",
            "SELECT date, expiry_date, expiry_month FROM vix_futures_daily WHERE date <= ? "
            "ORDER BY date, expiry_date", (through,))}
        part = self.store.market("vix_futures_daily", through)
        if len(part):
            rows.update({(str(d)[:10], str(e)[:10]): str(m) for d, e, m in
                         zip(part["date"], part["expiry_date"], part["expiry_month"])})
        return sorted((d, m) for (d, _), m in rows.items())

    def first_ledger(self) -> str:
        names = self.store.derived("forward#first ledger", lambda: sorted(
            p.stem.removeprefix("ledger_") for p in (self.store.data / "forward").glob("ledger_*.json"))[:1])
        if not names:
            raise Untrue("no forward-ledger entry is committed")
        return str(names[0])

    def first_captures(self) -> dict[str, Any]:
        def first() -> dict[str, Any]:
            recs = [json.loads(p.read_text()) for p in (self.store.data / "options" / "cboe").glob("*.json")]
            by_fetch: dict[str, Any] = {}
            for rec in sorted(recs, key=lambda r: r["fetched_at"]):
                by_fetch.setdefault(rec["underlying"], rec)
            spx = [r["session"] for r in recs if r["underlying"] == "SPX"]
            return {"first_spx_session": min(spx) if spx else None, "first_by_fetch": by_fetch}
        out = self.store.derived("options/cboe#first captures", first)
        if not out["first_spx_session"]:
            raise Untrue("no Cboe SPX book is stored")
        return out

    def capped(self) -> list[str]:
        from joint_vol_calibration.config import HESTON_BOUNDS
        cap = HESTON_BOUNDS["sigma"][1]
        out = [s for s in self.sessions if "sigma" in (self.cal(s).get("pinned_params") or [])]
        off = [s for s in out if abs(self.cal(s)["params"]["sigma"] - cap) > 1e-6 * cap]
        if off:
            raise Untrue(f"σ is reported pinned but is not at its {cap} cap on {off}")
        return out

    def feller_violated(self) -> int:
        params = [self.cal(s)["params"] for s in self.sessions]
        kept = [s for s, q in zip(self.sessions, params) if not 2 * q["kappa"] * q["theta"] < q["sigma"] ** 2]
        if kept:
            raise Untrue(f"Feller holds on {kept}")
        return len(params)

    def rough_counts(self) -> tuple[list[dict[str, Any]], int]:
        """The sessions where rough Heston is lower out of sample, and how many
        return H = ½ (Heston)."""
        diag = [self.diag(s) for s in self.sessions]
        lower = [r for r in diag if r["joint_rough_oos"] < r["sigma_cap2_oos"] - 0.01]
        return lower, sum(r["joint_rough_H"] >= 0.5 - 1e-9 for r in diag)

    def certified(self) -> None:
        for s in self.sessions:
            dg = self.scorecard(s)["svi_cal"]["dense_grid"]
            if dg["butterfly_violations"] or dg["calendar_violations"]:
                raise Untrue(f"the SVI surface of {s} has dense-grid violations")
            if self.surface(s)["surfaces"]["svi_cal"]["arbitrage_free_certified"] is not True:
                raise Untrue(f"the SVI surface of {s} is not certified arbitrage-free")

    def vix(self) -> dict[str, tuple[float, float]]:
        """Each session's replicated VIX and Cboe's official close
        (surface/index.json). A session the replication skipped (no candidate
        expiries on both sides of 30 days) has none: the sentences that list
        or range over the replicated values must say so, not stop on a
        missing number."""
        out = {s: (self.index(s)["vix_replicated"], self.index(s)["vix_official"]) for s in self.sessions}
        missing = [s for s, v in out.items() if None in v]
        if missing:
            raise Untrue(f"the VIX is not replicated on {missing}")
        return out

    def common(self) -> dict[str, dict[str, Any]]:
        """Heston's and SVI's scores on the quotes no model was fitted to
        (CLM-15), which must be the same quotes."""
        models = ("heston_v2", "svi_cal")
        out = {s: {m: self.scorecard(s)[m]["metrics"]["common_out_of_sample"] for m in models}
               for s in self.sessions}
        if any(v["svi_cal"]["n"] != v["heston_v2"]["n"] for v in out.values()):
            raise Untrue("Heston and SVI are not scored on the same quotes")
        return out


# ── the claims, one function per part of the documents ────────────────────────

def _span(c: Claims) -> None:
    with c.claim("span/results", within=FIRST_SCREEN) as k:
        ss = c.sessions
        k.text = (f"ranges over the {word(len(ss))} sessions with a stored chain and an accepted fit, "
                  f"{ss[0]} → {ss[-1]}, not confidence intervals")


def _records(c: Claims) -> None:
    for name, within in (("first_screen", FIRST_SCREEN), ("limitations", LIMITATIONS)):
        with c.claim(f"records/cboe_books_from/{name}", within=within) as k:
            k.text = f"Cboe's timestamped books are stored from {c.first_captures()['first_spx_session']}"
    with c.claim("records/vix_options_leg", within=FIRST_SCREEN) as k:
        weighted = [s for s in c.sessions if c.cal(s)["weights"]["w3"] != 0.0]
        if weighted:
            raise Untrue(f"the VIX-options leg has weight in the fits of {weighted}")
        k.text = "VIX options are a research leg, not in the fit"
    with c.claim("records/ledger/first_screen", within=FIRST_SCREEN) as k:
        k.text = f"the forward ledger at executable prices starts on {c.first_ledger()}"
    with c.claim("records/ledger/backtest") as k:
        k.text = f"Its first entry is {c.first_ledger()}"
    with c.claim("records/ledger/data_sources", doc=SOURCES_DOC) as k:
        k.text = f"first entry {c.first_ledger()}"
    for doc in (README, SOURCES_DOC):
        with c.claim(f"records/ledger/no_entry/{doc}", doc=doc, absent=True) as k:
            c.first_ledger()
            k.text = "no entry yet"


def _result_1(c: Claims) -> None:
    with c.claim("result_1/median", within=FIRST_SCREEN) as k:
        oos = [c.scorecard(s)["svi_cal"]["metrics"]["out_of_sample"] for s in c.sessions]
        k.text = f"**{rng((o['median_abs_hs'] for o in oos), '.2f')} half-spreads**"
    with c.claim("result_1/inside", within=FIRST_SCREEN) as k:
        oos = [c.scorecard(s)["svi_cal"]["metrics"]["out_of_sample"] for s in c.sessions]
        k.text = f"{rng((o['inside_spread_pct'] for o in oos), '.1f')}% inside the bid–ask"
    with c.claim("result_1/last", within=FIRST_SCREEN) as k:
        h = c.scorecard(c.last)["svi_cal"]["metrics"]["out_of_sample"]
        k.text = f"({c.last}: {fmt(h['median_abs_hs'], '.2f')}, {fmt(h['inside_spread_pct'], '.1f')}% inside)"
    with c.claim("result_1/certified", within=FIRST_SCREEN) as k:
        c.certified()
        k.text = "A certified arbitrage-free, calendar-ordered SVI surface"
    with c.claim("result_1/common_out_of_sample") as k:
        common = c.common()
        c.certified()
        n = rng((v["svi_cal"]["n"] for v in common.values()), ",d")
        hes = rng((v["heston_v2"]["median_abs_hs"] for v in common.values()), ".1f")
        svi = rng((v["svi_cal"]["median_abs_hs"] for v in common.values()), ".2f")
        k.text = (f"On the {n} quotes per session that no model was fitted to (`data_store/surface/<session>/"
                  f"scorecard.json`, `metrics.common_out_of_sample`), Heston's median |model − mid| is {hes} "
                  f"half-spreads against {svi} for the certified arbitrage-free SVI surface")


def _result_2(c: Claims) -> None:
    from joint_vol_calibration.config import HESTON_BOUNDS

    with c.claim("result_2/anchors", within=FIRST_SCREEN) as k:
        anchors = [sum(a["n"] for a in c.cal(s)["anchors_by_expiry"]) for s in c.sessions]
        k.text = f"fitted to {rng(anchors, 'd')} SPX anchors and the four Cboe VIX indices"
    with c.claim("result_2/heston", within=FIRST_SCREEN) as k:
        oos = rng((c.diag(s)["sigma_cap2_oos"] for s in c.sessions), ".2f")
        k.text = f"misses held-out quotes by **{oos} vol pts** RMSE"
    with c.claim("result_2/last", within=FIRST_SCREEN) as k:
        last = c.cal(c.last)
        bound = "no parameter on a bound" if not last.get("pinned_params") else "on a bound"
        k.text = (f"({c.last}: {fmt(last['leg_losses']['spx_oos_iv_rmse'], '.2f')} on "
                  f"{last['oos']['n_spx_oos']:,} quotes, {bound})")
    with c.claim("result_2/sigma_cap", within=FIRST_SCREEN) as k:
        capped = c.capped()
        k.text = (f"σ sits on its {HESTON_BOUNDS['sigma'][1]:.1f} cap in {len(capped)} of the "
                  f"{len(c.sessions)} fits ({', '.join(capped)})")
    with c.claim("result_2/rough", within=FIRST_SCREEN) as k:
        k.text = f"Rough Heston: **{rng((c.diag(s)['joint_rough_oos'] for s in c.sessions), '.2f')}**"
    with c.claim("result_2/verdict", within=FIRST_SCREEN) as k:
        lower, same = c.rough_counts()
        k.text = f"lower on {len(lower)} of {len(c.sessions)} sessions, H = ½ (Heston) on {same}"


def _result_3(c: Claims) -> None:
    with c.claim("result_3/sharpe", within=FIRST_SCREEN) as k:
        b = c.inference()["bootstrap"]["portfolio"]["sharpe"]
        k.text = f"Portfolio Sharpe **{fmt(b['point'], '.2f')}**"
    with c.claim("result_3/interval", within=FIRST_SCREEN) as k:
        b = c.inference()["bootstrap"]["portfolio"]["sharpe"]
        k.text = f"95% CI **{fmt(b['ci95'][0], '.2f')} to {fmt(b['ci95'][1], '.2f')}**"
    with c.claim("result_3/spa", within=FIRST_SCREEN) as k:
        k.text = f"(Hansen SPA p {c.inference()['multiple_testing']['spa_published']['p_consistent']:.2f})"
    with c.claim("result_3/s3_timing", within=FIRST_SCREEN) as k:
        s3 = c.inference()["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
        k.text = f"S3's entry timing has p {s3['p_value']:.3f}"
    with c.claim("result_3/s3_shifts", within=FIRST_SCREEN) as k:
        s3 = c.inference()["nulls"]["sleeves"]["s3"]["circular"]["total_pnl"]
        k.text = (f"against {s3['n_paths']:,} exposure-matched circular shifts "
                  f"(Holm over six series {s3['p_holm_6_sleeves']:.2f})")


def _backtest_table(c: Claims) -> None:
    rows = {"Portfolio (S1C + S3 + S4)": ("portfolio", None), "S1C contrarian PDV": ("s1c", "s1c"),
            "S3 dispersion, index leg": ("s3", "s3"), "S4 volatility risk premium": ("s4", "s4"),
            "S1 PDV short-VRP (reference)": ("s1", "s1"), "S2 VIX term structure (reference)": ("s2", "s2")}
    for label, (series, key) in rows.items():
        with c.claim(f"backtest_table/{series}") as k:
            m = c.metrics()
            if key is None:
                pnl, sharpe, trades = m["cumulative_return"] * 1_000_000, m["sharpe"], m["n_trades"]
            else:
                pnl, sharpe, trades = m[f"{key}_total_pnl"], m[f"{key}_sharpe"], m[f"{key}_n_trades"]
            ci = c.inference()["bootstrap"][series]["sharpe"]["ci95"]
            k.text = (f"| {label} | {usd(pnl)} | {fmt(sharpe, '.2f')} | {fmt(ci[0], '.2f')} to "
                      f"{fmt(ci[1], '.2f')} | {trades} |")
    with c.claim("backtest_table/nav") as k:
        m = c.metrics()
        nav = 1_000_000 * (1 + m["cumulative_return"])
        k.text = f"NAV ${nav:,.0f} ({fmt(m['cumulative_return'] * 100, '.1f')}%)"
    with c.claim("backtest_table/cash") as k:
        k.text = f"T-bills earned {c.metrics()['cash_cum_return'] * 100:.1f}%"
    with c.claim("backtest_table/beta_grid") as k:
        grid = c.store.csv("backtest/beta_sensitivity.csv").set_index("beta").sort_index()
        if not (grid["pnl"] < 0).all():
            raise Untrue("the portfolio is not negative at every β of the grid")
        lo, hi = grid.index[0], grid.index[-1]
        k.text = (f"moves S1 from {usd(grid.loc[hi, 's1_pnl'])} at β = {hi:.2f} to "
                  f"{usd(grid.loc[lo, 's1_pnl'])} at β = {lo:.2f}, and the portfolio is negative at every β "
                  f"from {lo:.2f} to {hi:.2f}")
    with c.claim("backtest_table/implied_beta") as k:
        bench = c.store.json("backtest/inference/benchmarks.json")["implied_beta"]
        put, wput = round(bench["PUT"]["overall"], 2), round(bench["WPUT"]["overall"], 2)
        if put != wput:
            raise Untrue(f"PUT and WPUT imply different β ({put:.2f}, {wput:.2f})")
        k.text = f"implies β = {put:.2f} for both"


def _backtest_prose(c: Claims) -> None:
    from joint_vol_calibration.backtest.backtest_engine import ATM_TO_VIX_RATIO

    def window() -> tuple[str, str, int]:
        eq = c.store.parquet("backtest/full_results.parquet")
        return str(pd.Timestamp(eq.index[0]).date()), str(pd.Timestamp(eq.index[-1]).date()), len(eq)

    with c.claim("backtest_prose/window") as k:
        start, end, n = window()
        k.text = f"{start} → {end} ({n:,} sessions)"
    with c.claim("backtest_prose/first_screen", within=FIRST_SCREEN) as k:
        start, end, _ = window()
        k.text = f"Backtest, {start} → {end}, straddles marked at β·VIX"
    with c.claim("backtest_prose/beta") as k:
        k.text = f"β = {ATM_TO_VIX_RATIO:.2f} estimated from"
    with c.claim("backtest_prose/dsr") as k:
        # STATS-3: the DSR at the ledger's counts, and the lenient bound with its basis
        t = c.inference()["trials"]
        s3 = t["dsr_range"]["s3"]
        led, n = s3["at_ledger_counts"], t["n_choices"]
        if not s3["argmax"].endswith("n_eigen_participation"):
            raise Untrue(f"S3's largest DSR is at {s3['argmax']}, not the eigenvalue participation ratio")
        counts = [n[x] for x in t["n_choices_ledger_based"]]
        k.text = (f"S3's deflated Sharpe ratio is {led['min']:.3f}–{led['max']:.3f} at the ledger's trial "
                  f"counts (N {min(counts):.0f}–{max(counts):.0f}); it reaches {s3['max']:.2f} only if the "
                  f"search is counted as {n['n_eigen_participation']:.1f} independent trials")


def _data_table(c: Claims) -> None:
    """Each row's count is taken as of the end date it states (the last date on
    or before params['data_through']), so the job appending rows after it does
    not move the number (CLM-03)."""
    through = c.params["data_through"]
    for label, table in MARKET_ROWS:
        with c.claim(f"data_table/{table}", within=f"| {label} |") as k:
            df = c.store.market(table, through)
            dates = pd.to_datetime(df["date"])
            text = f"| {dates.min().date()} → {dates.max().date()} | {len(df):,}"
            if table == "vix_term_structure":
                text += f" ({int((df['source'] == 'yfinance').sum())} Yahoo rows)"
            if table == "rates_daily":
                text += f" ({df['series'].nunique()} series)"
            k.text = text + " |"
    with c.claim("data_table/spx_options", within="| SPX options |") as k:
        k.text = f"| {sum(len(c.store.parquet(f'options/spx_{s}.parquet')) for s in C23_SESSIONS):,} |"
    with c.claim("data_table/vix_options", within="| VIX options |") as k:
        k.text = f"| {sum(len(c.store.parquet(f'options/vix_{s}.parquet')) for s in C23_SESSIONS[:4]):,} |"
    with c.claim("data_table/cboe_books", within="| SPX and VIX option books (C23) |") as k:
        first = c.first_captures()["first_by_fetch"]
        spx, vix = first["SPX"], first["VIX"]
        k.text = f"from {spx['session']} | {spx['rows']:,} SPX + {vix['rows']:,} VIX lines (first capture)"
    with c.claim("data_table/vx_futures", within="| VX futures |") as k:
        rows = c.vx_futures(through)
        days, months = [r[0] for r in rows], sorted({r[1] for r in rows})
        k.text = (f"({len(months)} monthly contracts expiring {months[0]} → {months[-1]}) | "
                  f"{min(days)} → {max(days)} | {len(rows):,} |")
    with c.claim("data_table/vx_settlements", within="| VX final settlements |") as k:
        rows = c.store.sql(f"vix_futures_settlements settled on or before {through}",
                           "SELECT settlement_date FROM vix_futures_settlements WHERE settlement_date <= ? "
                           "ORDER BY settlement_date", (through,))
        k.text = f"| {rows[0][0]} → {rows[-1][0]} | {len(rows):,} |"
    man = None
    with c.claim("data_table/strategy_indices", within="| Cboe strategy indices |") as k:
        man = c.store.json("benchmarks/manifest.json")["files"]
        spans = {(f["first_date"], f["last_date"]) for f in man.values()}
        if len(spans) != 1:
            raise Untrue(f"the strategy-index files cover different spans: {sorted(spans)}")
        a, b = spans.pop()
        k.text = f"| {a} → {b} |"
    for name in sorted(man or ()):
        with c.claim(f"data_table/strategy_indices/{name}", within="| Cboe strategy indices |") as k:
            c.store.json("benchmarks/manifest.json")
            k.text = name


def _limitations(c: Claims) -> None:
    from joint_vol_calibration.backtest import run_backtest as rb
    from joint_vol_calibration.backtest.backtest_engine import ATM_TO_VIX_RATIO
    from joint_vol_calibration.config import HESTON_BOUNDS

    with c.claim("limitations/beta", within=LIMITATIONS) as k:
        k.text = f"at one β ({ATM_TO_VIX_RATIO:.2f}) estimated from 2026 chains"
    with c.claim("limitations/later_chain", within=LIMITATIONS) as k:
        c.store.file("vol_system.db")              # the estimate reads the stored chains from the cache
        beta = rb.estimate_beta(include_live=False).set_index("session")["ratio"]
        k.text = f"The first later chain ({C23_HEADLINE}) gives {beta[C23_HEADLINE]:.2f}"
    with c.claim("limitations/put_index", within=LIMITATIONS) as k:
        put = c.store.json("backtest/inference/benchmarks.json")["implied_beta"]["PUT"]
        full = [v for y, v in put["by_year"].items() if y not in put["partial_years"]]
        k.text = f"the Cboe PUT index implies {rng(full, '.2f')} by full year"
    with c.claim("limitations/loss", within=LIMITATIONS) as k:
        m = c.metrics()
        k.text = (f"the portfolio loses {abs(m['cumulative_return']) * 100:.1f}% while T-bills earn "
                  f"{m['cash_cum_return'] * 100:.1f}%")
    with c.claim("limitations/vx_futures", within=LIMITATIONS) as k:
        vx = [v for s in c.sessions if (v := c.cal(s)["leg_losses"].get("vx_futures_rmse_oos")) is not None]
        k.text = f"misses real VX futures by {rng(vx, '.1f')} pts out of sample"
    with c.claim("limitations/sigma_cap", within=LIMITATIONS) as k:
        k.text = (f"σ sits on its {HESTON_BOUNDS['sigma'][1]:.1f} cap in {len(c.capped())} of the "
                  f"{len(c.sessions)} fits to {c.last}")
    with c.claim("limitations/vvix", within=LIMITATIONS) as k:
        diag = [c.diag(s) for s in c.sessions]
        below = [r["session"] for r in diag if not r["vvix_model_cap2"] > r["vvix_market"]]
        if below:
            raise Untrue(f"the boxed fit's VVIX is not above the market's on {below}")
        model = rng((r["vvix_model_cap2"] for r in diag), ".0f")
        market = rng((r["vvix_market"] for r in diag), ".0f")
        k.text = (f"the boxed fits price VVIX at {model} against a market ^VVIX of {market} on all "
                  f"{word(len(diag))}")
    with c.claim("limitations/quintic", within=LIMITATIONS) as k:
        q = sorted((c.store.data / "calibrations").glob("quintic_cal_*.pkl"))
        if c.store.pickle(f"calibrations/{q[-1].name}")["success"]:
            raise Untrue("the Quintic OU fit converged")
        k.text = "The Quintic OU fit that prices VIX options did not converge"
    with c.claim("limitations/quote_quality", within=LIMITATIONS) as k:
        other = [s for s in c.sessions if c.surface(s)["inputs"]["source"] != "yfinance"]
        if other:
            raise Untrue(f"not every chain is a Yahoo download: {other} were built from another source")
        clean = [s for s in c.sessions if c.index(s)["executable_arbitrage"] is not True]
        if clean:
            raise Untrue(f"no executable static arbitrage on {clean}")
        w = word(len(c.sessions))
        k.text = (f"The {w} chains the surfaces are built from are Yahoo end-of-day downloads without a "
                  "quote time; static arbitrage that is executable at the quoted bid and ask is present on "
                  f"all {w}")


def _calibration(c: Claims) -> None:
    with c.claim("calibration/feller") as k:
        k.text = f"Feller is violated in all {word(c.feller_violated())} and imposed in none"
    with c.claim("calibration/table_stops") as k:
        k.text = f"The table stops at {c.last}"
    for s in c.sessions:
        with c.claim(f"calibration/heston_row/{s}") as k:
            cal = c.cal(s)
            p, ll = cal["params"], cal["leg_losses"]
            star = "\\*" if "sigma" in c.diag(s)["on_bound"] else ""
            vx = ll.get("vx_futures_rmse_oos")                 # the 'VX futures, OOS' column
            k.text = (f"| {s} | {p['kappa']:.3f} | {p['theta']:.4f} | {p['sigma']:.3f}{star} | "
                      f"{fmt(p['rho'], '.3f')} | {p['v0']:.4f} | {ll['spx_iv_rmse']:.3f} / "
                      f"{ll['spx_oos_iv_rmse']:.3f} | {ll['vix_index_rmse']:.3f} | "
                      f"{f'{vx:.2f}' if vx is not None else 'n/a†'} |")
        with c.claim(f"calibration/rough_row/{s}") as k:
            r = c.diag(s)
            k.text = (f"| {s} | {fmt(r['sigma_cap2_oos'], '.2f')} | {fmt(r['joint_rough_oos'], '.2f')} | "
                      f"{r['joint_rough_H']:.2f} |")
    with c.claim("calibration/free_sigma") as k:
        cur = c.diag(C23_HEADLINE)
        k.text = (f"gives σ {cur['sigma_free']:.2f}, κ {cur['kappa_free']:.1f}, a loss "
                  f"{abs(cur['sigma_free_loss_rel']) * 100:.1f}% lower and an out-of-sample RMSE of "
                  f"{cur['sigma_free_oos']:.2f} instead of {cur['sigma_cap2_oos']:.2f}")
    with c.claim("calibration/box_cost") as k:
        first = c.diag(c.sessions[0])
        cost = abs(first["sigma_free_loss_rel"]) * 100
        k.text = f"on {first['session']} the box costs {cost:.1f}% of the loss"
    with c.claim("calibration/vvix") as k:
        diag = [c.diag(s) for s in c.sessions]
        k.text = (f"model's VVIX is {rng((r['vvix_model_cap2'] for r in diag), '.0f')} against a market "
                  f"^VVIX of {rng((r['vvix_market'] for r in diag), '.0f')}")
    with c.claim("calibration/condition") as k:
        diag = [c.diag(s) for s in c.sessions]
        cond = rng((r["rel_jtj_condition_number"] for r in diag), ".0f")
        k.text = (f"cond(JᵀJ) of the relative-sensitivity Jacobian is {cond} over the {word(len(diag))} "
                  "sessions")
    with c.claim("calibration/kappa_se") as k:
        se = rng((c.diag(s)["relative_standard_errors"]["kappa"] * 100 for s in c.sessions), ".0f")
        k.text = f"standard error is {se}% of κ"
    with c.claim("calibration/one_basin") as k:
        many = [s for s in c.sessions if c.diag(s)["trf_basins"] != 1]
        if many:
            raise Untrue(f"the trust-region starts find more than one basin on {many}")
        k.text = "finds one basin on every session"
    with c.claim("calibration/de_optima") as k:
        c23 = [c.diag(s) for s in c.sessions if s in C23_SESSIONS]   # the DE optima C23 timed TRF against
        above = [r["session"] for r in c23 if r["trf_loss_rel_diff"] > 0]
        if above:
            raise Untrue(f"the trust-region fit ends above the stored DE optimum on {above}")
        worst = f"{max(abs(r['trf_loss_rel_diff']) for r in c23):.1e}".replace("e-0", "e-")
        trf = rng((r["trf_seconds"] for r in c23), ".1f")
        de = rng((r["de_serial_seconds"] for r in c23), ".1f")
        k.text = (f"reaches the {word(len(c23))} stored differential-evolution optima to a relative loss of "
                  f"at most {worst} (slightly below them) in {trf} s, against {de} s")
    with c.claim("calibration/rough_lower") as k:
        lower, same = c.rough_counts()
        if len(lower) + same != len(c.sessions):
            raise Untrue(f"rough Heston is lower on {len(lower)} and returns Heston on {same} of "
                         f"{len(c.sessions)} sessions: 'the other' sessions are not all H = ½")
        k.text = (f"error on {len(lower)} of {len(c.sessions)} sessions and returns H = ½ (Heston) on the "
                  f"other {word(same)}")
    with c.claim("calibration/vix_gap") as k:
        gaps = [abs(rep - off) for rep, off in c.vix().values()]
        k.text = f"within {rng(gaps, '.2f')} index points of the official close"


def _pack(c: Claims) -> None:
    """The validation pack's figures that count or range over the sessions
    (VALIDATOR-10): its span, the §1 limitations, §2's identifiability and
    replication counts, and the §3 outcome rows."""
    from joint_vol_calibration.config import HESTON_BOUNDS

    def claim(cid: str, within: str = "", absent: bool = False) -> _Computing:
        return c.claim(f"pack/{cid}", doc=PACK, within=within, absent=absent)

    def ses() -> tuple[int, str, str]:
        return len(c.sessions), word(len(c.sessions)), c.last

    with claim("span") as k:
        k.text = (f"the {word(len(c.sessions))} sessions with a stored chain and an accepted fit, "
                  f"{c.sessions[0]} → {c.last}")
    with claim("sigma_cap") as k:
        n, _, last = ses()
        capped = c.capped()
        k.text = (f"σ on its {HESTON_BOUNDS['sigma'][1]:.1f} box ceiling in {len(capped)} of the {n} fits to "
                  f"{last} ({', '.join(capped)}); Feller violated in all {c.feller_violated()}")
    with claim("rough_half") as k:
        n, _, last = ses()
        k.text = f"½ on {c.rough_counts()[1]} of the {n} sessions to {last}"
    with claim("ledger") as k:
        k.text = f"first entry {c.first_ledger()}, not yet evidence"
    with claim("ledger/m18", within="| M18 | Forward test") as k:
        k.text = f"first entry {c.first_ledger()} (`data_store/forward/`)"
    with claim("ledger/no_entry", absent=True) as k:
        c.first_ledger()
        k.text = "no entry"
    with claim("condition") as k:
        _, w, last = ses()
        cond = rng((c.diag(s)["rel_jtj_condition_number"] for s in c.sessions), ".0f")
        k.text = f"cond(JᵀJ) of the relative-sensitivity Jacobian is {cond} over the {w} sessions to {last}"

    def se(param: str) -> str:
        return rng((c.diag(s)["relative_standard_errors"][param] * 100 for s in c.sessions), ".0f")

    with claim("standard_errors") as k:
        k.text = f"κ {se('kappa')}%, θ {se('theta')}%, ρ {se('rho')}%, v₀ {se('v0')}%"
    with claim("gauss_newton") as k:
        k.text = f"Gauss–Newton errors of {se('kappa')}% on κ"
    with claim("common_quotes") as k:
        common = c.common()
        k.text = (" / ".join(f"{common[s]['svi_cal']['n']:,}" for s in c.sessions)
                  + f" quotes, {c.sessions[0]} → {c.last}")
    with claim("vix_departures") as k:
        vr = c.store.json(f"surface/{c.last}/vix_replication.json")
        k.text = (f"on {c.last}, {len(vr.get('deviations') or [])} deviations from Cboe's rules and "
                  f"{len(vr.get('gap_notes') or [])} notes on the gap")
    with claim("m4_oos", within="| M4 | Out-of-sample SPX IV RMSE") as k:
        oos = " / ".join(fmt(c.cal(s)["leg_losses"]["spx_oos_iv_rmse"], ".2f") for s in c.sessions)
        dates = " / ".join([c.sessions[0], *(s[5:] for s in c.sessions[1:])])
        k.text = f"| {oos} vp for {dates} ("
    with claim("m4_oos/sessions", within="| M4 | Out-of-sample SPX IV RMSE") as k:
        k.text = f", {word(len(c.sessions))} v2 sessions |"
    with claim("m4_half_spreads", within="| M4 | Error in") as k:
        hes = [c.scorecard(s)["heston_v2"]["metrics"]["out_of_sample"] for s in c.sessions]
        k.text = (f"{rng((h['median_abs_hs'] for h in hes), '.1f')} half-spreads; "
                  f"{rng((h['inside_spread_pct'] for h in hes), '.1f')}% inside, {word(len(hes))} sessions")
    with claim("m14_certified", within="| M14 | The certified") as k:
        c.certified()
        svi = {s: c.scorecard(s)["svi_cal"]["metrics"]["out_of_sample"] for s in c.sessions}
        last = svi[c.last]
        k.text = (f"median {rng((o['median_abs_hs'] for o in svi.values()), '.2f')} half-spreads; "
                  f"{rng((o['inside_spread_pct'] for o in svi.values()), '.0f')}% inside the bid/ask, "
                  f"{word(len(svi))} sessions ({c.last}: {fmt(last['median_abs_hs'], '.2f')}, "
                  f"{fmt(last['inside_spread_pct'], '.1f')}%)")
    with claim("m14_common", within="| M14 | The certified") as k:
        common = c.common()
        svi_c = rng((v["svi_cal"]["median_abs_hs"] for v in common.values()), ".2f")
        hes_c = rng((v["heston_v2"]["median_abs_hs"] for v in common.values()), ".1f")
        k.text = f"on the common held-out quotes {svi_c} against Heston's {hes_c}"
    with claim("m14_arbitrage", within="| M14 | Static arbitrage") as k:
        arb = {s: c.surface(s)["arbitrage"] for s in c.sessions}
        none = [s for s, a in arb.items()
                if not a["executable_arbitrage_certificate"]["executable_arbitrage"]]
        if none:
            raise Untrue(f"no static arbitrage executable at the bid/ask on {none}")
        n_ex = rng((a["n_quotes_in_executable_violation"] for a in arb.values()), ",d")
        k.text = f"executable at the bid/ask on all {len(arb)} sessions; {n_ex} quotes per session"
    with claim("m14_vix", within="| M14 | VIX replicated") as k:
        vix = c.vix()
        rep = " / ".join(fmt(vix[s][0], ".2f") for s in c.sessions)
        off = " / ".join(fmt(vix[s][1], ".2f") for s in c.sessions)
        k.text = f"| {rep} against {off} ("
    with claim("m4_m15_rough", within="| M4, M15 | Rough Heston") as k:
        lower, same = c.rough_counts()
        pairs = ", ".join(f"{fmt(r['joint_rough_oos'], '.2f')} vs {fmt(r['sigma_cap2_oos'], '.2f')}"
                          for r in lower)
        k.text = (f"rough lower on {len(lower)} of {len(c.sessions)} sessions ({pairs} vp); H = ½, "
                  f"identical, on {same}")


TOPICS: dict[str, Callable[[Claims], None]] = {
    "span": _span, "records": _records, "result_1": _result_1, "result_2": _result_2,
    "result_3": _result_3, "backtest_table": _backtest_table, "backtest_prose": _backtest_prose,
    "data_table": _data_table, "limitations": _limitations, "calibration": _calibration, "pack": _pack,
}


def compute(params: dict[str, str], store: Store | None = None,
            topics: Iterable[str] | None = None) -> list[Claim]:
    """Every claim of ``topics`` (default: all) from ``store``, over the span
    ``params`` names."""
    c = Claims(store or Store(), params)
    for name in topics or TOPICS:
        TOPICS[name](c)
    return c.out


# ── the snapshot and the documents ────────────────────────────────────────────

def _git(*args: str) -> str:
    return subprocess.run(["git", "-C", str(ROOT), *args], capture_output=True,  # noqa: S603, S607
                          text=True, check=True).stdout.strip()


def newest_params(store: Store) -> dict[str, str]:
    """The newest span: the last stored session with a scorecard and a v2 fit,
    and the last SPX close."""
    c = Claims(store, {"sessions_through": "9999-12-31"})
    files = sorted((store.data / "market" / "spx_ohlcv").glob("*.parquet"))
    spx = pd.concat([pd.read_parquet(f) for f in files])
    return {"sessions_through": c.sessions[-1], "data_through": str(pd.to_datetime(spx["date"]).max().date())}


def build(params: dict[str, str], store: Store | None = None, *, date: str | None = None,
          commit: str | None = None) -> tuple[dict[str, Any], list[Claim]]:
    """The snapshot over ``params`` and the claims that could not be made
    (empty when the snapshot is complete)."""
    store = store or Store()
    claims = compute(params, store)
    bad = [k for k in claims if k.error]
    artifacts = {key: entry for k in claims for key, entry in k.sources.items()}
    snap = {
        "schema_version": SCHEMA_VERSION,
        "about": ("Every figure README.md quotes, and the validation pack's and DATA_SOURCES.md's figures "
                  "that range over the sessions, as printed, with the artifacts each was computed from. "
                  "Written by scripts/readme_results.py; joint_vol_calibration/tests/test_readme.py checks "
                  "the documents against it and it against data_store."),
        "generated": {"date": date or dt.datetime.now(dt.UTC).date().isoformat(),
                      "commit": commit or _git("rev-parse", "HEAD"), "command": COMMAND},
        "params": dict(params),
        "sessions": list(Claims(store, params).sessions),
        "artifacts": dict(sorted(artifacts.items())),
        "claims": [k.record() for k in claims],
    }
    return snap, bad


def spans(text: str, within: str) -> list[tuple[int, int]]:
    """Where in a document a claim's text must be: the whole document, the
    README's first screen, a '## ' section, or each line that starts with a
    table row's first cells."""
    if within == FIRST_SCREEN:
        end = text.find("\n---\n")
        return [(0, end if end >= 0 else len(text))]
    if within.startswith("## "):
        start = text.find(f"\n{within}\n")
        if start < 0:
            return []
        end = text.find("\n## ", start + 1)
        return [(start, end if end >= 0 else len(text))]
    if within.startswith("| "):
        out, pos = [], 0
        for line in text.split("\n"):
            if line.startswith(within):
                out.append((pos, pos + len(line)))
            pos += len(line) + 1
        return out
    return [(0, len(text))]


def region(text: str, within: str) -> list[str]:
    return [collapse(text[a:b]) for a, b in spans(text, within)]


def read_doc(doc: str) -> str:
    return (ROOT / doc).read_text()


def doc_problems(snapshot: dict[str, Any], read: Callable[[str], str] = read_doc) -> list[str]:
    """(a) What the documents say that the snapshot does not: a claim not in
    its place verbatim (or an absent one present), or a document that does not
    state the snapshot's date."""
    out = []
    texts: dict[str, str] = {}

    def doc(name: str) -> str:
        if name not in texts:
            texts[name] = read(name)
        return texts[name]

    for k in snapshot["claims"]:
        if k["text"] is None:
            out.append(f"{k['id']}: the snapshot has no text for it")
            continue
        parts = region(doc(k["doc"]), k.get("within", ""))
        if k.get("within", "").startswith("| ") and len(parts) != 1:
            out.append(f"{k['id']}: {len(parts)} lines of {k['doc']} start with {k['within']!r}, not one")
            continue
        found = any(collapse(k["text"]) in p for p in parts)
        if found == bool(k.get("absent")):
            where = f" ({k['within']})" if k.get("within") else ""
            state = "says" if k.get("absent") else "does not say"
            out.append(f"{k['id']}: {k['doc']}{where} {state} {k['text']!r}")
    asof = f"as of {snapshot['generated']['date']}"
    for name, within in AS_OF:
        if not any(asof in p for p in region(doc(name), within)):
            where = " (its first screen)" if within == FIRST_SCREEN else ""
            out.append(f"generated: {name}{where} does not say {asof!r}")
    return out


# ── comparing the snapshot with the current data_store ────────────────────────

def bot_written(workflow: Path = WORKFLOW) -> tuple[str, ...]:
    """The data_store/ paths the daily job writes: every path its commit steps
    (scripts/ci/commit_and_push.sh in daily_refresh.yml) stage, exclusions not
    counted, and JOB_CACHE."""
    text = workflow.read_text()
    paths: set[str] = set(JOB_CACHE)
    for m in re.finditer(r"scripts/ci/commit_and_push\.sh(.*?)(?=\n\s*\n|\n\s*- name:|\Z)", text, re.DOTALL):
        paths.update(re.findall(r"(?<![)\w/])(data_store/[\w./-]+)", m.group(1)))
    return tuple(sorted(p.rstrip("/.") for p in paths))


def _under(key: str, paths: Iterable[str]) -> bool:
    return any(key == p or key.startswith((f"{p}/", f"{p}#")) for p in paths)


@dataclass
class Finding:
    claim: str
    stale: bool                       # True: only artifacts the daily job writes changed under it
    message: str
    changed: list[str] = field(default_factory=list)


def _sha_was(snapshot: dict[str, Any], key: str) -> str | None:
    return (snapshot["artifacts"].get(key) or {}).get("sha256")


def _sha_now(store: Store, key: str) -> str | None:
    return (store.digest(key) or {}).get("sha256")


def compare(snapshot: dict[str, Any], claims: list[Claim], store: Store,
            bot: tuple[str, ...] | None = None) -> list[Finding]:
    """(b) Each snapshot claim against its recomputation. Same text in the same
    place: nothing. Otherwise the artifacts it was computed from that changed
    since decide: all written by the daily job, a stale snapshot (a warning);
    none, or one the job never writes, a failure. A claim the code adds or
    drops, or places elsewhere, is a code change and fails."""
    bot = bot_written() if bot is None else bot
    old = {k["id"]: k for k in snapshot["claims"]}
    new = {k.id: k for k in claims}
    out = []
    for cid in sorted(old.keys() | new.keys()):
        was, now = old.get(cid), new.get(cid)
        if was is None:
            out.append(Finding(cid, False, f"{cid} is computed but is not in the snapshot: regenerate with "
                                           f"{COMMAND}"))
            continue
        if now is None:
            out.append(Finding(cid, False, f"{cid} is in the snapshot but no longer computed: regenerate "
                                           f"with {COMMAND}"))
            continue
        place = (was["doc"], was.get("within", ""), bool(was.get("absent")))
        if place != (now.doc, now.within, now.absent):
            out.append(Finding(cid, False, f"{cid} is now placed at {(now.doc, now.within, now.absent)}, the "
                                           f"snapshot at {place}: regenerate with {COMMAND}"))
            continue
        if not now.error and was["text"] == now.text:
            continue
        changed = sorted(key for key in was["sources"] if _sha_was(snapshot, key) != _sha_now(store, key))
        what = (f"could not be recomputed ({now.error})" if now.error
                else f"now reads {now.text!r}, the snapshot {was['text']!r}")
        if changed and all(_under(key, bot) for key in changed):
            out.append(Finding(cid, True, f"{cid} {what}", changed))
        elif changed:
            other = ", ".join(key for key in changed if not _under(key, bot))
            out.append(Finding(cid, False, f"{cid} {what}, and {other} changed, which the daily job never "
                                           f"writes: regenerate with {COMMAND}", changed))
        else:
            out.append(Finding(cid, False, f"{cid} {what}, with every artifact it reads unchanged (the code "
                                           f"or the snapshot changed): regenerate with {COMMAND}"))
    return out


def changed_artifacts(snapshot: dict[str, Any], store: Store) -> list[str]:
    """The snapshot's artifacts whose sha256 is no longer the current one."""
    return sorted(key for key in snapshot["artifacts"] if _sha_was(snapshot, key) != _sha_now(store, key))


def stale_message(snapshot: dict[str, Any], findings: list[Finding], changed: list[str], store: Store,
                  bot: tuple[str, ...]) -> str:
    date = snapshot["generated"]["date"]
    lines = [(f"docs/results/readme_results.json (as of {date}) was computed from {len(changed)} "
              "artifact(s) that have changed since:")]
    for key in changed:
        was = (snapshot["artifacts"].get(key) or {}).get("report_version")
        now = (store.digest(key) or {}).get("report_version")
        note = [f"report_version {was} → {now}"] if was != now else []
        if not _under(key, bot):
            note.append("the daily job does not write it")
        lines.append(f"  {key}" + (f" ({'; '.join(note)})" if note else ""))
    moved = [f for f in findings if f.stale]
    if moved:
        lines.append("The daily job rebuilt them, and these figures the documents quote have moved:")
        lines += [f"  {f.message}" for f in moved]
    elif not any(f.changed for f in findings):
        lines.append("No figure the documents quote from them has moved.")
    lines.append(f"Regenerate with {COMMAND} (python scripts/readme_results.py --update-docs), which "
                 "rewrites the figures that moved in the documents, and review the diff.")
    return "\n".join(lines)


def check(snapshot: dict[str, Any], store: Store | None = None, topics: Iterable[str] | None = None,
          bot: tuple[str, ...] | None = None) -> list[str]:
    """(b) as a test runs it: warns StaleSnapshot when artifacts the snapshot
    read have changed and no figure moved for a reason other than the daily
    job's rebuilds; returns the failures."""
    store = store or Store()
    bot = bot_written() if bot is None else bot
    topics = list(topics or TOPICS)
    claims = compute(snapshot["params"], store, topics)
    prefixes = tuple(f"{t}/" for t in topics)
    part = [k for k in snapshot["claims"] if k["id"].startswith(prefixes)]
    read = {key for k in part for key in k["sources"]}
    sub = {**snapshot, "claims": part,
           "artifacts": {k: v for k, v in snapshot["artifacts"].items() if k in read}}
    findings = compare(sub, claims, store, bot)
    changed = changed_artifacts(sub, store)
    if changed:
        warnings.warn(stale_message(sub, findings, changed, store, bot), StaleSnapshot, stacklevel=2)
    return [f.message for f in findings if not f.stale]


def update_docs(old: dict[str, Any], new: dict[str, Any], root: Path = ROOT) -> list[str]:
    """Rewrite, in each document and only in the part of it the claim names,
    the claims whose text moved from ``old`` to ``new`` (and the 'as of'
    date), keeping the line breaks where the word count is unchanged."""
    olds = {k["id"]: k for k in old["claims"]}
    changes = [(olds[k["id"]], k["text"]) for k in new["claims"]
               if k["id"] in olds and not k.get("absent") and olds[k["id"]]["text"] != k["text"]
               and (olds[k["id"]]["doc"], olds[k["id"]].get("within", "")) == (k["doc"], k.get("within", ""))]
    for doc, within in AS_OF:
        was, now = f"as of {old['generated']['date']}", f"as of {new['generated']['date']}"
        if was != now:
            changes.append(({"doc": doc, "within": within, "text": was}, now))
    done = []
    for doc in sorted({was["doc"] for was, _ in changes}):
        path = root / doc
        text = path.read_text()
        for was, now in changes:
            if was["doc"] != doc:
                continue
            pattern = re.compile(r"\s+".join(map(re.escape, was["text"].split())))

            def swap(m: re.Match[str], new_text: str = now) -> str:
                gaps = re.findall(r"\s+", m.group(0))
                words = new_text.split()
                if len(gaps) != len(words) - 1:
                    return new_text
                return "".join(w + g for w, g in zip(words, [*gaps, ""]))
            n = 0
            for a, b in reversed(spans(text, was.get("within", ""))):
                part, k = pattern.subn(swap, text[a:b])
                text, n = text[:a] + part + text[b:], n + k
            if n:
                done.append(f"{doc}: {was['text']!r} → {now!r}")
        if text != path.read_text():
            path.write_text(text)
    return done


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])
    ap.add_argument("--through", help="a date or 'newest': the last session and the Data table's end date")
    ap.add_argument("--sessions-through", help="the last session the results range over")
    ap.add_argument("--data-through", help="the Data table's end date")
    ap.add_argument("--check", action="store_true",
                    help="compare the snapshot with data_store; write nothing")
    ap.add_argument("--update-docs", action="store_true",
                    help="rewrite the figures that moved in the documents")
    ap.add_argument("--allow-dirty", action="store_true",
                    help="write although data_store/ has uncommitted changes (the commit recorded is then "
                         "not the data's)")
    args = ap.parse_args(argv)
    store = Store()
    old = json.loads(SNAPSHOT.read_text()) if SNAPSHOT.exists() else None
    if args.check:
        if old is None:
            print(f"no snapshot at {SNAPSHOT.relative_to(ROOT)}: run {COMMAND}")
            return 1
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            failures = check(old, store)
        for w in caught:
            print(f"warning: {w.message}")
        problems = failures + doc_problems(old)
        for p in problems:
            print(f"FAIL {p}")
        print("ok" if not problems else f"{len(problems)} problem(s)")
        return 1 if problems else 0
    params = dict(old["params"]) if old else {}
    if args.through or not old:
        newest = newest_params(store)
        through = args.through if args.through and args.through != "newest" else None
        params = {k: through or v for k, v in newest.items()}
    if args.sessions_through:
        params["sessions_through"] = args.sessions_through
    if args.data_through:
        params["data_through"] = args.data_through
    dirty = _git("status", "--porcelain", "--", "data_store")
    if dirty and not args.allow_dirty:
        print("data_store/ has uncommitted changes; the snapshot names the commit its data came from:")
        print(dirty)
        return 1
    snap, bad = build(params, store)
    for k in bad:
        print(f"cannot claim {k.id}: {k.error}")
    if bad:
        print("nothing written: reword the claims above in the document and their template in this script")
        return 1
    SNAPSHOT.parent.mkdir(parents=True, exist_ok=True)
    SNAPSHOT.write_text(json.dumps(snap, indent=1, ensure_ascii=False, allow_nan=False) + "\n")
    print(f"wrote {SNAPSHOT.relative_to(ROOT)}: {len(snap['claims'])} claims over {len(snap['sessions'])} "
          f"sessions through {params['sessions_through']} (Data table through {params['data_through']})")
    if old is not None:
        olds = {k["id"]: k["text"] for k in old["claims"]}
        for k in snap["claims"]:
            if olds.get(k["id"]) != k["text"]:
                print(f"  {k['id']}: {olds.get(k['id'])!r} → {k['text']!r}")
        if args.update_docs:
            for line in update_docs(old, snap):
                print(f"  rewrote {line}")
    problems = doc_problems(snap)
    for p in problems:
        print(f"update the document: {p}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
