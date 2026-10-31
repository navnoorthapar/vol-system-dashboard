"""page_budget.py: the Surface page's weight against its budget (CI's a11y job).

    python scripts/ci/page_budget.py <site-dir>

surface.html may weigh at most 450,000 bytes at 45 listed expiries, 7,000 bytes more
for each expiry past 45 and 5,500 bytes less for each one short of 45 (D2): a full
Cboe book lists about 45, and each expiry adds a row to the five heatmaps, the delta
grid and the smile's tables (a 17-expiry Yahoo chain: 296,000 bytes; with a flat
budget below 45 such a page could have grown by 150 kB unnoticed). The page no longer
grows with the stored sessions it compares (at most four), only by the "Every
session" table's row each (about 300 bytes). The expiry count is read from the page
itself (its smile's expiry tabs), so a day with more listed expiries raises the budget
with the page instead of failing the job; the daily bot's deploy is never gated on it
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
GZIP_LEVEL = 6
# the smile's expiry tabs (dashboard/templates/surface.html)
TAB = re.compile(r'<button class="btn[^"]*" type="button" role="tab" aria-selected="(?:true|false)" '
                 r'aria-controls="slices-panel" data-value="\d{4}-\d{2}-\d{2}"')


def budget(n_expiries: int) -> int:
    """The Surface page's byte budget at n listed expiries."""
    d = n_expiries - BASE_EXPIRIES
    return BASE_BYTES + (PER_EXPIRY if d > 0 else PER_EXPIRY_SHORT) * d


def gzipped(data: bytes, level: int = GZIP_LEVEL) -> int:
    """Bytes of data gzipped at level (zlib's deflate in a gzip wrapper)."""
    z = zlib.compressobj(level, zlib.DEFLATED, 31)
    return len(z.compress(data) + z.flush())


def check(site: Path) -> tuple[bool, str]:
    html = (site / "surface.html").read_bytes()
    n = len(TAB.findall(html.decode("utf-8")))
    cap = budget(n)
    ok = len(html) <= cap
    return ok, (f"surface.html: {len(html):,} bytes at {n} listed expiries, budget {cap:,} "
                f"({BASE_BYTES:,} at {BASE_EXPIRIES}, + {PER_EXPIRY:,} an expiry past, "
                f"− {PER_EXPIRY_SHORT:,} an expiry short): {'ok' if ok else 'OVER'}; "
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
