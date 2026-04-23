#!/usr/bin/env python3
"""
add_breadcrumb_to_json.py
==========================
Add `hilchot_group` and `siman_sign` as new fields at the siman level of a
Shulchan Arukh RAG JSON. Seif-level fields (including `text`) are NOT modified.

Expected project layout (script lives in scripts/):
    <project_root>/
      ├── scripts/
      │   └── add_breadcrumb_to_json.py       ← this file
      ├── source_original/
      │   └── Shulchan_Aruch_Text_Headlines.txt   ← default headings file
      └── processed/
          ├── shulchan_aruch_rag.json                  ← default input JSON
          └── shulchan_aruch_rag_with_breadcrumb.json  ← default output

Resolving inputs/outputs:
    If `--json`, `--headings`, or `--output` aren't absolute paths, they're
    resolved against processed/ (for JSON files) or source_original/ (for the
    headings file) relative to this script. Running from any CWD still works.

Inputs
------
1. JSON produced by `build_shulchan_aruch_rag.py`:
       processed/shulchan_aruch_rag.json
   Expected structure:
       {"title": ..., "source": ..., "simanim": [
           {"siman": N, "seifim": [{"seif": M, "text": ..., ...}]}
       ]}

2. Text file with siman headings (table of contents):
       source_original/Shulchan_Aruch_Text_Headlines.txt
   Expected format:
       הלכות הנהגת אדם בבוקר               ← group heading
       סימן 1 - דין השכמת הבוקר. ובו 9 סעיפים:
       סימן 2 - דיני לבישת הבגדים. ובו 6 סעיפים:
       הלכות ציצית                          ← next group heading
       סימן 8 - ...

Output
------
Same JSON with two new fields per siman, placed right after `siman`:
    {
      "siman": 1,
      "hilchot_group": "הלכות הנהגת אדם בבוקר",
      "siman_sign":    "דין השכמת הבוקר",
      "seifim": [ ... unchanged ... ]
    }

Both fields are always present (empty string when unknown), so downstream
consumers can rely on a uniform schema. Default output name is
`{input-stem}_with_breadcrumb.json` in processed/.

Usage
-----
    # zero-config: reads processed/shulchan_aruch_rag.json + source_original/...
    python scripts/add_breadcrumb_to_json.py

    # explicit paths
    python scripts/add_breadcrumb_to_json.py \
        --json processed/shulchan_aruch_rag.json \
        --headings source_original/Shulchan_Aruch_Text_Headlines.txt \
        --output processed/shulchan_aruch_rag_with_breadcrumb.json

    # regression test (no real files needed)
    python scripts/add_breadcrumb_to_json.py --test
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from pathlib import Path


# ═════════════════════════════════════════════════════════════════════════════
# PROJECT LAYOUT — fixed relative to this script
# ═════════════════════════════════════════════════════════════════════════════

HERE          = Path(__file__).resolve().parent       # scripts/
PROJECT_ROOT  = HERE.parent                           # project root
SOURCE_DIR    = PROJECT_ROOT / "source_original"      # headings file
PROCESSED_DIR = PROJECT_ROOT / "processed"            # JSON files

DEFAULT_JSON_NAME     = "shulchan_aruch_rag.json"
DEFAULT_HEADINGS_NAME = "Shulchan_Aruch_Text_Headlines.txt"

DEFAULT_JSON     = str(PROCESSED_DIR / DEFAULT_JSON_NAME)
DEFAULT_HEADINGS = str(SOURCE_DIR / DEFAULT_HEADINGS_NAME)
ROOT_LABEL       = "אורח חיים"


def _resolve(path_arg: str, base_dir: Path) -> Path:
    """Resolve a CLI path argument against `base_dir` when relative.

    Rules:
      1. Absolute path — used as-is.
      2. Path that exists relative to CWD — used as-is.
      3. Bare filename — placed under `base_dir`.
      4. Relative path with directory — resolved against CWD.
    """
    p = Path(path_arg)
    if p.is_absolute():
        return p
    if p.exists():
        return p.resolve()
    if p.parent == Path("."):
        return (base_dir / p).resolve()
    return p.resolve()


def resolve_json(path_arg: str) -> Path:
    return _resolve(path_arg, PROCESSED_DIR)


def resolve_headings(path_arg: str) -> Path:
    return _resolve(path_arg, SOURCE_DIR)


def resolve_output(path_arg: str) -> Path:
    return _resolve(path_arg, PROCESSED_DIR)


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
    Mutate `data` in place: add `hilchot_group` and `siman_sign` as new fields
    at the siman level, immediately after the `siman` number and before `seifim`.
    Seif-level fields (including `text`) are NOT modified.

    Returns (total_simanim_updated, list_of_missing_siman_numbers).
    """
    total = 0
    missing: list[str] = []

    new_simanim = []
    for siman_obj in data.get('simanim', []):
        sn   = str(siman_obj['siman'])
        info = siman_map.get(sn)
        if info is None:
            missing.append(sn)
            hilchot_group = ''
            siman_sign    = ''
        else:
            hilchot_group = info.get('hilchot_group', '')
            siman_sign    = info.get('siman_sign', '')
            # If the sign is just a copy of the group (bare "ובו" case), drop it
            if siman_sign == hilchot_group:
                siman_sign = ''

        # Rebuild the siman dict with the new fields placed right after `siman`
        # and before `seifim`, preserving order of any other keys.
        new_obj: dict = {}
        for k, v in siman_obj.items():
            if k == 'seifim':
                new_obj['hilchot_group'] = hilchot_group
                new_obj['siman_sign']    = siman_sign
                new_obj[k] = v
            elif k in ('hilchot_group', 'siman_sign'):
                # Skip old values; re-inserted above
                continue
            else:
                new_obj[k] = v
        # Safety: if the siman had no `seifim` key for some reason
        if 'hilchot_group' not in new_obj:
            new_obj['hilchot_group'] = hilchot_group
            new_obj['siman_sign']    = siman_sign

        new_simanim.append(new_obj)
        total += 1

    data['simanim'] = new_simanim
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

        # Test breadcrumb builder (still useful for log messages / debugging)
        print("\n── build_breadcrumb ──")
        bc_cases = [
            ('1',   '[הלכות הנהגת אדם בבוקר > דין השכמת הבוקר]'),
            ('8',   '[הלכות ציצית > דין לבישת ציצית]'),
            ('625', '[הלכות יום הכיפורים]'),  # sign==group → no duplicate
            ('999', '[אורח חיים]'),           # missing → root only
        ]
        for sn, expected in bc_cases:
            bc = '[' + build_breadcrumb(siman_map.get(sn)) + ']'
            mark = '✓' if bc == expected else '✗'
            print(f'  {mark} siman {sn}: {bc}')
            if bc != expected:
                print(f'     expected: {expected}')
                ok = False

        # Test add_breadcrumbs end-to-end on SAMPLE_JSON
        print("\n── add_breadcrumbs (field injection) ──")
        import copy
        sample = copy.deepcopy(SAMPLE_JSON)
        total, missing = add_breadcrumbs(sample, siman_map)
        print(f'  total simanim updated: {total}')
        print(f'  missing simanim: {missing}')

        # Test 1: fields added correctly on siman 1
        s1 = sample['simanim'][0]
        expected_group = 'הלכות הנהגת אדם בבוקר'
        expected_sign  = 'דין השכמת הבוקר'
        g_ok = s1.get('hilchot_group') == expected_group
        s_ok = s1.get('siman_sign')    == expected_sign
        mark = '✓' if (g_ok and s_ok) else '✗'
        print(f'  {mark} siman 1 fields: group={s1.get("hilchot_group")!r}, '
              f'sign={s1.get("siman_sign")!r}')
        if not (g_ok and s_ok):
            ok = False

        # Test 2: key ordering — siman, hilchot_group, siman_sign, seifim
        keys = list(s1.keys())
        expected_keys = ['siman', 'hilchot_group', 'siman_sign', 'seifim']
        mark = '✓' if keys == expected_keys else '✗'
        print(f'  {mark} key ordering: {keys}')
        if keys != expected_keys:
            print(f'     expected: {expected_keys}')
            ok = False

        # Test 3: text of seifim is UNCHANGED
        orig_text = SAMPLE_JSON['simanim'][0]['seifim'][0]['text']
        got_text  = s1['seifim'][0]['text']
        mark = '✓' if got_text == orig_text else '✗'
        print(f'  {mark} seif text unchanged: {got_text!r}')
        if got_text != orig_text:
            print(f'     original: {orig_text!r}')
            ok = False

        # Test 4: text_raw is also untouched
        raw_unchanged = (s1['seifim'][0]['text_raw']
                         == SAMPLE_JSON['simanim'][0]['seifim'][0]['text_raw'])
        mark = '✓' if raw_unchanged else '✗'
        print(f'  {mark} seif text_raw unchanged')
        if not raw_unchanged: ok = False

        # Test 5: siman 625 — group present, sign empty (fallback behavior)
        s625 = next(s for s in sample['simanim'] if s['siman'] == 625)
        g_ok = s625.get('hilchot_group') == 'הלכות יום הכיפורים'
        s_ok = s625.get('siman_sign')    == ''
        mark = '✓' if (g_ok and s_ok) else '✗'
        print(f'  {mark} siman 625: group={s625.get("hilchot_group")!r}, '
              f'sign={s625.get("siman_sign")!r} (expected empty)')
        if not (g_ok and s_ok):
            ok = False

        # Test 6: missing siman → both fields empty
        sample_with_missing = copy.deepcopy(SAMPLE_JSON)
        sample_with_missing['simanim'].append(
            {"siman": 999, "seifim": [{"seif": 1, "text": "abc"}]}
        )
        _, missing2 = add_breadcrumbs(sample_with_missing, siman_map)
        s999 = next(s for s in sample_with_missing['simanim'] if s['siman'] == 999)
        g_ok = s999.get('hilchot_group') == ''
        s_ok = s999.get('siman_sign')    == ''
        m_ok = '999' in missing2
        mark = '✓' if (g_ok and s_ok and m_ok) else '✗'
        print(f'  {mark} missing siman (999): both fields empty, reported in missing list')
        if not (g_ok and s_ok and m_ok):
            ok = False

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

    log("\nAdding siman-level fields (hilchot_group, siman_sign)...")
    total, missing = add_breadcrumbs(data, siman_map)
    log(f"  Updated {total:,} simanim")

    if missing:
        log(f"  WARNING: {len(missing)} simanim missing from headings "
            f"(received empty fields): {missing[:30]}"
            f"{' ...' if len(missing) > 30 else ''}")
    else:
        log("  Full coverage: every siman matched a heading")

    # Example
    first = data['simanim'][0]
    log(f"\nExample (siman {first['siman']}):")
    log(f'  hilchot_group: {first.get("hilchot_group", "")!r}')
    log(f'  siman_sign:    {first.get("siman_sign", "")!r}')
    log(f'  seifim:        {len(first.get("seifim", []))} (unchanged)')

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

    json_path     = resolve_json(args.json)
    headings_path = resolve_headings(args.headings)
    output_path   = (resolve_output(args.output) if args.output
                     else json_path.with_name(json_path.stem + "_with_breadcrumb.json"))

    try:
        process(json_path, headings_path, output_path, quiet=args.quiet)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)


if __name__ == "__main__":
    main()
