"""page_budget.py: the Surface page's weight against its budget (CI's a11y job).

    python scripts/ci/page_budget.py <site-dir>

surface.html may weigh at most 450,000 bytes at 45 listed expiries, 7,000 bytes more
for each expiry past 45 and 5,500 bytes less for each one short of 45 (D2): a full
Cboe book lists about 45, and each expiry adds a row to the five heatmaps, the delta
grid and the smile's tables (a 17-expiry Yahoo chain: 296,000 bytes; with a flat
budget below 45 such a page could have grown by 150 kB unnoticed). Two parts grow
with the data rather than the markup, and the budget grows with them: the default
expiry's quotes, which the smile draws from the page itself (about 53 bytes each; a
30-day expiry lists 100 to 600 of them), past DEFAULT_QUOTES, and the "Every session"
table, a row per stored session (about 300 bytes), past HISTORY_ROWS. All three counts
are read from the page itself (its smile's expiry tabs, the status line's quote count
and the table's rows), so a day with more listed expiries, a deeper default expiry or
one more stored session raises the budget with the page instead of failing the job,
while a heavier markup still does; the daily bot's deploy is never gated on it
(freeze_site.py does not read this). The gzipped size is printed at GZIP_LEVEL (zlib's
default; level 9 is about 3% smaller) and not gated: the server picks its own level.
Every other weight check runs in pytest on synthetic sessions (test_dashboard.py),
never on a stored one. Exits 1 when the page is over its budget (a page without the
market surface, so with no expiry tabs, is held to the 0-expiry budget).
"""
from __future__ import annotations

import re
import sys
import zlib
from pathlib import Path

BASE_BYTES = 450_000          # at BASE_EXPIRIES listed expiries
BASE_EXPIRIES = 45
PER_EXPIRY = 7_000            # each listed expiry past BASE_EXPIRIES
PER_EXPIRY_SHORT = 5_500      # each listed expiry short of BASE_EXPIRIES
DEFAULT_QUOTES = 200          # the default expiry's quotes BASE_BYTES allows for
PER_DEFAULT_QUOTE = 60        # each default-expiry quote past DEFAULT_QUOTES (about 53 bytes)
HISTORY_ROWS = 10             # "Every session" rows BASE_BYTES allows for
PER_HISTORY_ROW = 350         # each row past HISTORY_ROWS (about 300 bytes)
GZIP_LEVEL = 6
# the smile's expiry tabs (dashboard/templates/surface.html)
TAB = re.compile(r'<button class="btn[^"]*" type="button" role="tab" aria-selected="(?:true|false)" '
                 r'aria-controls="slices-panel" data-value="\d{4}-\d{2}-\d{2}"')
# the smile's status line names the default expiry's quote count; the history table's rows
STATUS = re.compile(r'id="slices-status"[^>]*>[^<]*?· ([\d,]+) quotes ·')
HISTORY = re.compile(r'<table[^>]*id="surface-history-table".*?<tbody>(.*?)</tbody>', re.DOTALL)


def budget(n_expiries: int, default_quotes: int = 0, history_rows: int = 0) -> int:
    """The Surface page's byte budget at n listed expiries, a default expiry of
    default_quotes quotes and history_rows rows in the "Every session" table."""
    d = n_expiries - BASE_EXPIRIES
    return (BASE_BYTES + (PER_EXPIRY if d > 0 else PER_EXPIRY_SHORT) * d
            + PER_DEFAULT_QUOTE * max(0, default_quotes - DEFAULT_QUOTES)
            + PER_HISTORY_ROW * max(0, history_rows - HISTORY_ROWS))


def counts(html: str) -> tuple[int, int, int]:
    """(listed expiries, default-expiry quotes, "Every session" rows) on the page."""
    q = STATUS.search(html)
    h = HISTORY.search(html)
    return (len(TAB.findall(html)), int(q.group(1).replace(",", "")) if q else 0,
            h.group(1).count("<tr") if h else 0)


def gzipped(data: bytes, level: int = GZIP_LEVEL) -> int:
    """Bytes of data gzipped at level (zlib's deflate in a gzip wrapper)."""
    z = zlib.compressobj(level, zlib.DEFLATED, 31)
    return len(z.compress(data) + z.flush())


def check(site: Path) -> tuple[bool, str]:
    html = (site / "surface.html").read_bytes()
    n, q, h = counts(html.decode("utf-8"))
    cap = budget(n, q, h)
    ok = len(html) <= cap
    return ok, (f"surface.html: {len(html):,} bytes at {n} listed expiries, a {q:,}-quote default "
                f"expiry and {h} stored sessions, budget {cap:,} ({BASE_BYTES:,} at {BASE_EXPIRIES}, "
                f"+ {PER_EXPIRY:,} an expiry past, − {PER_EXPIRY_SHORT:,} an expiry short, "
                f"+ {PER_DEFAULT_QUOTE} a default-expiry quote past {DEFAULT_QUOTES}, "
                f"+ {PER_HISTORY_ROW} a session past {HISTORY_ROWS}): {'ok' if ok else 'OVER'}; "
                f"{gzipped(html):,} bytes gzipped at level {GZIP_LEVEL} (not gated)")


def main(argv: list[str] | None = None) -> int:
    args = sys.argv[1:] if argv is None else argv
    if len(args) != 1:
        print(__doc__.strip().split("\n\n")[1], file=sys.stderr)
        return 2
    ok, msg = check(Path(args[0]))
    print(msg)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
