"""
dashboard/desk_glossary.py: the units and sign conventions, the glossary and
the first-use links every page makes to it (Q14).

The linker is pure string work on rendered HTML; these tests pin which text it
may touch (running prose inside <main>) and which it must leave alone
(headings, tables, tiles, links, buttons, code, scripts, hidden and
screen-reader-only text), that only the first use of a term is linked, and that
every definition names code that exists.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
DASH = ROOT / "dashboard"
if str(DASH) not in sys.path:
    sys.path.insert(0, str(DASH))

import desk_glossary as gl  # noqa: E402


def _page(main: str) -> str:
    return (f"<html><body><header><p>PSR in the banner</p></header><main id='main'>{main}</main>"
            f"<footer><p>PSR in the footer</p></footer></body></html>")


def test_the_glossary_has_twenty_terms_with_unique_ids_and_full_entries():
    assert len(gl.TERMS) >= 20
    assert len(set(gl.TERM_IDS)) == len(gl.TERM_IDS)
    for t in gl.TERMS:
        assert re.fullmatch(r"[a-z0-9-]+", t["id"]), t["id"]
        for key in ("term", "text", "units", "used"):
            assert t[key], (t["id"], key)
        assert re.compile(t["pat"]).search(t["term"].replace("&", "&amp;")), t["id"]


def test_every_definition_names_code_that_exists():
    """A definition quotes the code's formula: the file it names exists, and a
    named function or field is in it."""
    for t in gl.TERMS:
        src = t.get("source")
        if not src:
            continue
        path = ROOT / src
        assert path.exists(), (t["id"], src)
        if path.is_file() and t.get("fn"):
            code = path.read_text(encoding="utf-8")
            for fn in re.split(r",\s*", t["fn"]):
                assert fn.split()[0] in code, (t["id"], fn)


def test_the_half_spread_definition_quotes_the_fits_floor():
    """The definition says what the fit does (Q14 honesty risk): the half-spread
    in vol terms, floored at 0.05 vol pts in the weights."""
    import inspect

    sys.path.insert(0, str(ROOT))
    from joint_vol_calibration.surface import essvi

    floor = inspect.signature(essvi.fit_essvi).parameters["h_floor"].default
    hs = next(t for t in gl.TERMS if t["id"] == "half-spread")
    assert f"floored at {floor * 100:.2f} vol pts" in hs["text"]
    assert "(ask IV − bid IV)/2" in hs["text"]


def test_only_the_first_use_in_running_prose_is_linked():
    html = _page("<h2>PSR heading</h2><table><tr><td>PSR cell</td></tr></table>"
                 "<p>The <b>PSR</b> is 0.31 and the PSR again; DSR too.</p><li>PSR in a list</li>")
    out = gl.link_first_uses(html, "model-risk.html")
    assert out.count('class="term"') == 2
    assert ('<b><a class="term" href="model-risk.html#def-psr"><abbr title="probabilistic Sharpe ratio">'
            'PSR</abbr></a></b>') in out
    assert "and the PSR again" in out
    assert "<li>PSR in a list</li>" in out
    assert '<a class="term" href="model-risk.html#def-dsr">' in out
    # outside <main>, in a heading and in a table cell: untouched
    for kept in ("<h2>PSR heading</h2>", "<td>PSR cell</td>", "<p>PSR in the banner</p>",
                 "<p>PSR in the footer</p>"):
        assert kept in out
    assert gl.first_uses(out) == {"psr": "model-risk.html#def-psr", "dsr": "model-risk.html#def-dsr"}


@pytest.mark.parametrize("wrap", [
    '<a href="x">{}</a>', "<button>{}</button>", "<code>{}</code>", '<span class="sr-only">{}</span>',
    '<span hidden>{}</span>', '<span aria-hidden="true">{}</span>', '<span class="chip">{}</span>',
    "<summary>{}</summary>", "<script>var s = '{}';</script>", "<svg><text>{}</text></svg>",
    "<!-- {} -->"])
def test_text_inside_controls_code_and_hidden_elements_is_never_linked(wrap):
    inner = "the vol pts and the PSR"
    out = gl.link_first_uses(_page("<p>" + wrap.format(inner) + "</p>"), "")
    assert 'class="term"' not in out
    # the next prose use is the first one linked
    out = gl.link_first_uses(_page("<p>" + wrap.format(inner) + "</p><p>the PSR</p>"), "")
    assert gl.first_uses(out) == {"psr": "#def-psr"}


def test_overlapping_terms_and_entities():
    out = gl.link_first_uses(_page("<p>|z| in half-spreads; the vomma P&amp;L ratio is 0.4</p>"), "m.html")
    assert gl.first_uses(out) == {"z": "m.html#def-z", "half-spread": "m.html#def-half-spread",
                                  "vomma-ratio": "m.html#def-vomma-ratio"}
    assert '<a class="term" href="m.html#def-half-spread">half-spreads</a>' in out
    # a plain phrase gets no <abbr>; an acronym's expansion is HTML-escaped
    assert "<abbr" not in out.split("def-half-spread")[1].split("</a>")[0]


def test_a_page_without_main_is_returned_unchanged_and_skip_is_honoured():
    assert gl.link_first_uses("<p>PSR</p>", "x") == "<p>PSR</p>"
    out = gl.link_first_uses(_page("<p>PSR and DSR</p>"), "", skip=("psr",))
    assert gl.first_uses(out) == {"dsr": "#def-dsr"}


def test_conventions_state_each_unit_once():
    keys = [k for k, _ in gl.CONVENTIONS]
    assert len(keys) == len(set(keys))
    text = " ".join(v for _, v in gl.CONVENTIONS)
    for must in ("vol pts", "index pts", "model − market", "implied − realised", "ln(K/F)",
                 "risk-neutral"):
        assert must in text, must
    assert not re.search(r"\bvp\b", text)
