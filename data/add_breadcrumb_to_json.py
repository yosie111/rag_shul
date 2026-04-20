#!/usr/bin/env python3
"""
add_breadcrumb_to_json.py
==========================
Prepend a hierarchical breadcrumb (['אורח חיים' > 'הלכות ...' > 'דין ...'])
to the `text` field of every seif in a Shulchan Arukh RAG JSON.

Inputs
------
1. JSON file produced by `build_shulchan_aruch_rag.py`:
       shulchan_aruch_rag.json
   Expected structure:
       {"title": ..., "source": ..., "simanim": [
           {"siman": N, "seifim": [{"seif": M, "text": ..., ...}]}
       ]}

2. Text file with siman headings (table of contents):
       Shulchan_Aruch_Text_Headlines.txt
   Expected format:
       הלכות הנהגת אדם בבוקר               ← group heading
       סימן 1 - דין השכמת הבוקר. ובו 9 סעיפים:
       סימן 2 - דיני לבישת הבגדים. ובו 6 סעיפים:
       הלכות ציצית                          ← next group heading
       סימן 8 - ...

Output
------
Same JSON with the breadcrumb prepended to each seif's `text`:
    "[הלכות הנהגת אדם בבוקר > דין השכמת הבוקר] יתגבר כארי..."

Default output name is `{input-stem}_with_breadcrumb.json`.

Usage
-----
    # defaults: reads shulchan_aruch_rag.json + Shulchan_Aruch_Text_Headlines.txt
    python add_breadcrumb_to_json.py

    # explicit paths
    python add_breadcrumb_to_json.py \
        --json data/shulchan_aruch_rag.json \
        --headings data/Shulchan_Aruch_Text_Headlines.txt \
        --output data/shulchan_aruch_rag_with_breadcrumb.json

    # regression test (no real files needed)
    python add_breadcrumb_to_json.py --test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_JSON     = "shulchan_aruch_rag.json"
DEFAULT_HEADINGS = "Shulchan_Aruch_Text_Headlines.txt"
ROOT_LABEL       = "אורח חיים"


# ═════════════════════════════════════════════════════════════════════════════
# HEADINGS PARSER
# ═════════════════════════════════════════════════════════════════════════════

# "סימן 1 - דין השכמת הבוקר. ובו 9 סעיפים:"
SIMAN_WITH_DESC = re.compile(r'^סימן\s+(\d+)\s*-\s*(.+?)\.?\s*ובו')

# "סימן 625 - ובו סעיף אחד:"  or "סימן 625 ובו סעיף אחד:"
SIMAN_NO_DESC   = re.compile(r'^סימן\s+(\d+)\s*-?\s*ובו')

# "הלכות ציצית"
GROUP_HEADING   = re.compile(r'^הלכות\s')


def parse_headings(headings_path: Path) -> dict[str, dict[str, str]]:
    """
    Parse the table-of-contents text file.

    Returns a mapping:
        { "<siman_num_as_str>": {"hilchot_group": str, "siman_sign": str}, ... }
    """
    siman_map: dict[str, dict[str, str]] = {}
    current_group: str | None = None

    with open(headings_path, encoding='utf-8') as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue

        m = SIMAN_WITH_DESC.match(line)
        if m:
            siman_map[m.group(1)] = {
                'hilchot_group': current_group or '',
                'siman_sign':    m.group(2).strip().rstrip('.'),
            }
            continue

        m2 = SIMAN_NO_DESC.match(line)
        if m2:
            # No per-siman description → fall back to the group name
            siman_map[m2.group(1)] = {
                'hilchot_group': current_group or '',
                'siman_sign':    current_group or '',
            }
            continue

        if GROUP_HEADING.match(line):
            current_group = line
            continue

        # Any other line (title, blank, etc.) is ignored

    return siman_map


# ═════════════════════════════════════════════════════════════════════════════
# BREADCRUMB BUILDER
# ═════════════════════════════════════════════════════════════════════════════

def build_breadcrumb(info: dict[str, str] | None) -> str:
    """Compose the bracketed breadcrumb prefix for a siman."""
    parts: list[str] = []
    if info:
        if info.get('hilchot_group'):
            parts.append(info['hilchot_group'])
        # Avoid duplicating the group when siman_sign == group (fallback case)
        if info.get('siman_sign') and info['siman_sign'] != info.get('hilchot_group', ''):
            parts.append(info['siman_sign'])
    return ' > '.join(parts) if parts else ROOT_LABEL


def add_breadcrumbs(data: dict, siman_map: dict[str, dict[str, str]]
                    ) -> tuple[int, list[str]]:
    """
    Mutate `data` in place: prepend `[breadcrumb] ` to each seif's `text`.

    Returns (total_seifim_updated, list_of_missing_siman_numbers).
    """
    total = 0
    missing: list[str] = []

    for siman_obj in data.get('simanim', []):
        sn   = str(siman_obj['siman'])
        info = siman_map.get(sn)
        if info is None:
            missing.append(sn)

        breadcrumb = build_breadcrumb(info)
        prefix = f'[{breadcrumb}] '

        for seif_obj in siman_obj.get('seifim', []):
            txt = seif_obj.get('text')
            seif_obj['text'] = prefix + (txt if txt is not None else '')
            total += 1

    return total, missing


# ═════════════════════════════════════════════════════════════════════════════
# REGRESSION TEST (no external files needed)
# ═════════════════════════════════════════════════════════════════════════════

SAMPLE_HEADINGS = """\
שולחן ערוך, אורח חיים

הלכות הנהגת אדם בבוקר
סימן 1 - דין השכמת הבוקר. ובו 9 סעיפים:
סימן 2 - דיני לבישת הבגדים. ובו 6 סעיפים:

הלכות ציצית
סימן 8 - דין לבישת ציצית. ובו 17 סעיפים:

הלכות יום הכיפורים
סימן 625 - ובו סעיף אחד:
"""


SAMPLE_JSON = {
    "title": "שולחן ערוך, אורח חיים",
    "source": "Torat Emet 363",
    "simanim": [
        {"siman": 1, "seifim": [
            {"seif": 1, "text": "יתגבר כארי לעמוד בבוקר",
             "hagah": None, "text_raw": "יִתְגַּבֵּר"},
            {"seif": 2, "text": "לא יתבייש מפני המלעיגים",
             "hagah": None, "text_raw": "לֹא"},
        ]},
        {"siman": 2, "seifim": [
            {"seif": 1, "text": "ילבש בגדי חול ויתפלל",
             "hagah": None, "text_raw": "יִלְבַּשׁ"},
        ]},
        {"siman": 8, "seifim": [
            {"seif": 1, "text": "זמן עטיפת הציצית",
             "hagah": None, "text_raw": "זְמַן"},
        ]},
        {"siman": 625, "seifim": [
            {"seif": 1, "text": "ביום כיפור אסור באכילה",
             "hagah": None, "text_raw": "בְּיוֹם"},
        ]},
    ],
}


def run_tests() -> bool:
    import tempfile

    ok = True

    # Write sample headings to a temp file and parse
    with tempfile.NamedTemporaryFile('w', encoding='utf-8',
                                     suffix='.txt', delete=False) as fp:
        fp.write(SAMPLE_HEADINGS)
        headings_path = Path(fp.name)

    try:
        siman_map = parse_headings(headings_path)
        print("── parse_headings ──")

        # Note on siman 625: the FIRST regex (SIMAN_WITH_DESC) greedily matches
        # even "ובו סעיף אחד:" with an empty description (capture = " " → "").
        # The fallback regex SIMAN_NO_DESC is never reached in practice.
        # The breadcrumb still comes out correct because build_breadcrumb skips
        # empty siman_sign via the `if info['siman_sign']` guard. Preserved to
        # stay byte-identical with the original notebook.
        cases = [
            ('1',   'הלכות הנהגת אדם בבוקר', 'דין השכמת הבוקר'),
            ('2',   'הלכות הנהגת אדם בבוקר', 'דיני לבישת הבגדים'),
            ('8',   'הלכות ציצית',            'דין לבישת ציצית'),
            ('625', 'הלכות יום הכיפורים',     ''),  # first regex wins with empty sign
        ]
        for sn, exp_group, exp_sign in cases:
            info = siman_map.get(sn, {})
            g_ok = info.get('hilchot_group') == exp_group
            s_ok = info.get('siman_sign')    == exp_sign
            mark = '✓' if (g_ok and s_ok) else '✗'
            print(f'  {mark} siman {sn}: group={info.get("hilchot_group")!r}, '
                  f'sign={info.get("siman_sign")!r}')
            if not (g_ok and s_ok):
                print(f'     expected group={exp_group!r}, sign={exp_sign!r}')
                ok = False

        # Test breadcrumb builder on parsed siman_map
        print("\n── build_breadcrumb ──")
        bc_cases = [
            ('1',   '[הלכות הנהגת אדם בבוקר > דין השכמת הבוקר]'),
            ('8',   '[הלכות ציצית > דין לבישת ציצית]'),
            ('625', '[הלכות יום הכיפורים]'),  # sign==group → no duplicate
            ('999', '[אורח חיים]'),                       # missing → root only
        ]
        for sn, expected in bc_cases:
            bc = '[' + build_breadcrumb(siman_map.get(sn)) + ']'
            mark = '✓' if bc == expected else '✗'
            print(f'  {mark} siman {sn}: {bc}')
            if bc != expected:
                print(f'     expected: {expected}')
                ok = False

        # Test add_breadcrumbs end-to-end on SAMPLE_JSON
        print("\n── add_breadcrumbs ──")
        import copy
        sample = copy.deepcopy(SAMPLE_JSON)
        total, missing = add_breadcrumbs(sample, siman_map)
        print(f'  total seifim updated: {total}')
        print(f'  missing simanim: {missing}')

        expected_first = ('[הלכות הנהגת אדם בבוקר > דין השכמת הבוקר] '
                          'יתגבר כארי לעמוד בבוקר')
        got_first = sample['simanim'][0]['seifim'][0]['text']
        mark = '✓' if got_first == expected_first else '✗'
        print(f'  {mark} siman 1, seif 1: {got_first}')
        if got_first != expected_first:
            print(f'     expected: {expected_first}')
            ok = False

        # Ensure other seif in same siman also has it
        got_second = sample['simanim'][0]['seifim'][1]['text']
        if not got_second.startswith('[הלכות הנהגת אדם בבוקר > דין השכמת הבוקר] '):
            print(f'  ✗ siman 1, seif 2 missing breadcrumb: {got_second}')
            ok = False
        else:
            print(f'  ✓ siman 1, seif 2 prefix ok')

        # Ensure non-text fields are untouched
        raw_unchanged = (sample['simanim'][0]['seifim'][0]['text_raw']
                         == SAMPLE_JSON['simanim'][0]['seifim'][0]['text_raw'])
        mark = '✓' if raw_unchanged else '✗'
        print(f'  {mark} text_raw untouched')
        if not raw_unchanged: ok = False

    finally:
        headings_path.unlink(missing_ok=True)

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def process(json_path: Path, headings_path: Path, output_path: Path,
            quiet: bool = False) -> None:
    log = (lambda *a, **k: None) if quiet else print

    if not headings_path.exists():
        raise FileNotFoundError(f"Headings file not found: {headings_path}")
    if not json_path.exists():
        raise FileNotFoundError(f"JSON file not found: {json_path}")

    log(f"Headings: {headings_path.name}")
    siman_map = parse_headings(headings_path)
    log(f"  Mapped {len(siman_map)} simanim")

    # Spot-check
    for s in ['1', '8', '128', '625']:
        if s in siman_map:
            info = siman_map[s]
            log(f'  Siman {s}: [{info["hilchot_group"]} > {info["siman_sign"]}]')

    log(f"\nJSON:     {json_path.name}")
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    n_simanim = len(data.get('simanim', []))
    log(f"  {n_simanim} simanim")

    log("\nAdding breadcrumbs...")
    total, missing = add_breadcrumbs(data, siman_map)
    log(f"  Updated {total:,} seifim across {n_simanim} simanim")

    if missing:
        log(f"  WARNING: {len(missing)} simanim missing from headings "
            f"(received bare '{ROOT_LABEL}' prefix): {missing[:30]}"
            f"{' ...' if len(missing) > 30 else ''}")
    else:
        log("  Full coverage: every siman matched a heading")

    # Example
    sample = data['simanim'][0]['seifim'][0]
    log(f"\nExample (siman {data['simanim'][0]['siman']}, seif 1):")
    log(f'  {sample["text"][:150]}...')

    # Save (newline="\n" → consistent LF on Windows and Linux alike)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f"\nSaved: {output_path}  ({size_mb:.2f} MB)")


def main():
    ap = argparse.ArgumentParser(
        description="Add hierarchical breadcrumbs to a Shulchan Arukh RAG JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--json", "-j", default=DEFAULT_JSON,
                    help=f"input JSON file (default: {DEFAULT_JSON})")
    ap.add_argument("--headings", "-H", default=DEFAULT_HEADINGS,
                    help=f"headings TXT file (default: {DEFAULT_HEADINGS})")
    ap.add_argument("--output", "-o", default=None,
                    help="output JSON path "
                         "(default: <json-stem>_with_breadcrumb.json)")
    ap.add_argument("--test", action="store_true",
                    help="run regression tests on synthetic data and exit")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="suppress progress output")
    args = ap.parse_args()

    if args.test:
        ok = run_tests()
        sys.exit(0 if ok else 1)

    json_path     = Path(args.json)
    headings_path = Path(args.headings)
    output_path   = (Path(args.output) if args.output
                     else json_path.with_name(json_path.stem + "_with_breadcrumb.json"))

    try:
        process(json_path, headings_path, output_path, quiet=args.quiet)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
