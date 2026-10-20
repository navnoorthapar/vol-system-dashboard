"""
build_fonts.py — Rebuild the dashboard's self-hosted font subsets.

The site ships two OFL-1.1 fonts, subset for this dashboard and served from
dashboard/static/fonts (no font CDN, no third-party request):

  * Inter 4.1 (rsms/inter). The official variable file, instanced to weights
    400–700 at the text optical size (opsz 14) and subset to Latin, Greek,
    the math/arrow/shape glyphs the pages print, keeping the number features
    the desk needs: tabular figures (tnum), slashed zero (zero), the
    alternate one (cv01), case-sensitive forms, super/subscripts. The
    Fontsource and Google Fonts subsets strip `zero` and `cv01`.
  * IBM Plex Mono 2.5.0 (Regular, Medium). Subset to Latin-1 plus the minus
    sign, ± × ÷ ≈ ≤ ≥ ≠ and arrows (the official Latin-1 split file has no
    U+2212, so a signed number fell back to another font). IBM's licence
    reserves the font name "Plex", and a subset is a Modified Version under
    OFL-1.1 §3, so the subset is renamed "Desk Mono" in its name table.

Every source is pinned by URL and SHA-256; the script refuses a download
whose hash differs. The output hashes are recorded in static/fonts/fonts.json
and checked by the test suite, so a font file that no longer matches its
manifest fails CI.

Build-time only (needs fonttools and brotli, not runtime dependencies):
    python dashboard/build_fonts.py            # download, verify, subset, write
    python dashboard/build_fonts.py --src DIR  # use already-downloaded sources
(in DIR a source is read as `<key>-<file name>`, e.g. `inter_license-LICENSE.txt`,
else as its plain file name; the two licences share the name LICENSE.txt).
Rebuilt on 2026-10-20 with fonttools 4.60.1 and brotli: every output matched
fonts.json byte for byte.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

OUT = Path(__file__).resolve().parent / "static" / "fonts"

SOURCES = {
    "inter": {
        "url": "https://cdn.jsdelivr.net/gh/rsms/inter@v4.1/docs/font-files/InterVariable.woff2",
        "sha256": "693b77d4f32ee9b8bfc995589b5fad5e99adf2832738661f5402f9978429a8e3",
    },
    "inter_license": {
        "url": "https://cdn.jsdelivr.net/gh/rsms/inter@v4.1/LICENSE.txt",
        "sha256": "262481e844521b326f5ecd053e59b98c8b2da78c8ee1bdbb6e8174305e54935a",
    },
    "plex_400": {
        "url": "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/fonts/complete/woff2/IBMPlexMono-Regular.woff2",
        "sha256": "ba204497f16b6d334cee9d1e963a831b73e3a56e1d6300a8489d18df7214b350",
    },
    "plex_500": {
        "url": "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/fonts/complete/woff2/IBMPlexMono-Medium.woff2",
        "sha256": "33faf307fa6031fb4062276d7320a6d632de890cbb347576fd80cfa01077bc25",
    },
    "plex_license": {
        "url": "https://cdn.jsdelivr.net/npm/@ibm/plex-mono@2.5.0/LICENSE.txt",
        "sha256": "7e6b2818edbd8f6a01ae80641cc8f16a51080d08fb4e532be3a0b6f74adb07da",
    },
}

# Code points the pages print: Basic Latin, Latin-1, typographic punctuation,
# super/subscripts, Greek (κ θ σ ρ ν β Δ Σ ...), arrows, the minus sign and
# math operators, and the geometric shapes used as sign/status glyphs.
UNICODES = (
    "U+0020-007E,U+00A0-00FF,U+0131,U+0152-0153,U+02C6,U+02DA,U+02DC,"
    "U+0391-03A9,U+03B1-03C9,U+2000-206F,U+2070-209F,U+20AC,U+2122,U+2190-2199,"
    "U+2202,U+2206,U+2211,U+2212,U+2215,U+221A,U+221E,U+222B,U+2248,U+2260,"
    "U+2264,U+2265,U+25A0-25CF,U+2713,U+26A0"
)
INTER_FEATURES = ["kern", "calt", "ccmp", "locl", "mark", "mkmk", "case", "tnum", "pnum",
                  "lnum", "zero", "frac", "numr", "dnom", "sups", "subs", "cv01", "cv05", "ss02"]
MONO_FEATURES = ["kern", "calt", "ccmp", "locl", "zero", "tnum", "lnum", "ss02"]

OUTPUTS = {
    "inter": "InterDesk-4.1-wght400-700.woff2",
    "plex_400": "DeskMono-400.woff2",
    "plex_500": "DeskMono-500.woff2",
}
LICENSES = {"inter_license": "OFL-Inter.txt", "plex_license": "OFL-IBM-Plex-Mono.txt"}


def parse_unicodes(spec: str) -> list[int]:
    out: list[int] = []
    for part in spec.split(","):
        body = part.strip()[2:]
        if "-" in body:
            lo, hi = body.split("-")
            out.extend(range(int(lo, 16), int(hi, 16) + 1))
        else:
            out.append(int(body, 16))
    return out


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def fetch(key: str, src_dir: Path | None) -> bytes:
    spec = SOURCES[key]
    name = spec["url"].rsplit("/", 1)[-1]
    if src_dir is not None:
        data = (src_dir / f"{key}-{name}").read_bytes() if (src_dir / f"{key}-{name}").exists() \
            else (src_dir / name).read_bytes()
    else:
        with urllib.request.urlopen(spec["url"], timeout=60) as resp:   # noqa: S310 (pinned https URL)
            data = resp.read()
    got = sha256(data)
    if got != spec["sha256"]:
        raise SystemExit(f"{key}: sha256 {got} != pinned {spec['sha256']} ({spec['url']})")
    return data


def _rename(font, family: str, style: str) -> None:
    """Replace the family / full / PostScript names (OFL §3: no Reserved Font Name).

    Copyright (0) and trademark (7) notices are kept as published.
    """
    legacy_family = family if style in ("Regular", "Bold") else f"{family} {style}"
    ps = family.replace(" ", "") + "-" + style.replace(" ", "")
    names = {1: legacy_family, 2: "Regular" if style not in ("Regular", "Bold") else style,
             3: f"{ps};subset", 4: f"{family} {style}", 6: ps, 16: family, 17: style}
    table = font["name"]
    for rec in list(table.names):
        if rec.nameID in names:
            table.removeNames(nameID=rec.nameID)
    for name_id, text in names.items():
        table.setName(text, name_id, 3, 1, 0x409)


def subset_font(data: bytes, features: list[str], limits: dict | None = None,
                rename: tuple[str, str] | None = None) -> bytes:
    from fontTools import subset
    from fontTools.ttLib import TTFont
    from fontTools.varLib import instancer

    font = TTFont(io.BytesIO(data), lazy=False, recalcTimestamp=False)
    if limits:
        font = instancer.instantiateVariableFont(font, limits)
        buf = io.BytesIO()
        font.save(buf)
        font = TTFont(io.BytesIO(buf.getvalue()), lazy=False, recalcTimestamp=False)
    opts = subset.Options()
    opts.flavor = "woff2"
    opts.layout_features = features
    opts.name_IDs = ["*"]
    opts.notdef_outline = True
    opts.hinting = False
    opts.desubroutinize = True
    sub = subset.Subsetter(opts)
    sub.populate(unicodes=parse_unicodes(UNICODES))
    sub.subset(font)
    if rename:
        _rename(font, *rename)
    out = io.BytesIO()
    font.flavor = "woff2"
    font.save(out)
    return out.getvalue()


def build(src_dir: Path | None = None, out: Path = OUT) -> dict:
    import fontTools

    out.mkdir(parents=True, exist_ok=True)
    manifest = {"fonttools": fontTools.version, "unicodes": UNICODES, "files": {}}
    jobs = {
        "inter": (INTER_FEATURES, {"wght": (400, 700), "opsz": 14}, None),
        "plex_400": (MONO_FEATURES, None, ("Desk Mono", "Regular")),
        "plex_500": (MONO_FEATURES, None, ("Desk Mono", "Medium")),
    }
    for key, (features, limits, rename) in jobs.items():
        blob = subset_font(fetch(key, src_dir), features, limits, rename)
        (out / OUTPUTS[key]).write_bytes(blob)
        manifest["files"][OUTPUTS[key]] = {
            "source": SOURCES[key]["url"], "source_sha256": SOURCES[key]["sha256"],
            "sha256": sha256(blob), "bytes": len(blob), "features": features,
            "instance": {k: list(v) if isinstance(v, tuple) else v for k, v in (limits or {}).items()},
            "renamed_to": " ".join(rename) if rename else None, "license": LICENSES["inter_license" if key == "inter" else "plex_license"],
        }
        print(f"WROTE {OUTPUTS[key]}  {len(blob):,} bytes  sha256 {sha256(blob)[:16]}…")
    for key, name in LICENSES.items():
        (out / name).write_bytes(fetch(key, src_dir))
    (out / "fonts.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--src", type=Path, default=None, help="directory holding the pinned source files")
    args = ap.parse_args(argv[1:])
    build(args.src)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
