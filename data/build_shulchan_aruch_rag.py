#!/usr/bin/env python3
"""
build_shulchan_aruch_rag.py
============================
Preprocess the raw Shulchan Arukh (Torat Emet) TXT into structured RAG JSON.

Input:  Shulchan Arukh, Orach Chayim - he - Torat Emet 363.txt
Output: shulchan_aruch_rag.json

Output structure:
    {
      "title":  "שולחן ערוך, אורח חיים",
      "source": "Torat Emet 363",
      "simanim": [
        {"siman": 1, "seifim": [
          {"seif": 1, "text": "...", "hagah": "..." | null, "text_raw": "..."}
        ]}
      ]
    }

Pipeline:
    1. Basic fixes   — orphaned </small>, parens completion, whitespace, seifim separation
    2. Small cleanup — strip depth-2+ citations, strip depth-1 mechaber refs
    3. Numbers       — expand gematria abbreviations (ד' → ארבע in context)
    4. Ktiv male     — convert nikud to full spelling (יִתְגַּבֵּר → יתגבר)
    5. Unification   — unify divine names, goy, delete reference phrases
    6. Final cleanup — punctuation, whitespace, blank lines
    7. Structure     — split into simanim → seifim with {text, hagah, text_raw}

Usage:
    python build_shulchan_aruch_rag.py
    python build_shulchan_aruch_rag.py --input path/to/torat_emet.txt --output out.json
    python build_shulchan_aruch_rag.py --test             # run regression tests only
    python build_shulchan_aruch_rag.py --quiet            # suppress progress reports
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Optional


# ═════════════════════════════════════════════════════════════════════════════
# DEFAULTS
# ═════════════════════════════════════════════════════════════════════════════

DEFAULT_INPUT  = "Shulchan Arukh, Orach Chayim - he - Torat Emet 363.txt"
DEFAULT_OUTPUT = "shulchan_aruch_rag.json"


# ═════════════════════════════════════════════════════════════════════════════
# NIKUD CONSTANTS & PRIMITIVES
# ═════════════════════════════════════════════════════════════════════════════

SHVA,   HATAF_S, HATAF_P, HATAF_Q  = '\u05B0', '\u05B1', '\u05B2', '\u05B3'
HIRIQ,  TSERE,   SEGOL,   PATAH    = '\u05B4', '\u05B5', '\u05B6', '\u05B7'
QAMATS, HOLAM,   HOLAM_H, QUBUTS   = '\u05B8', '\u05B9', '\u05BA', '\u05BB'
DAGESH, SHIN_D,  SIN_D,   QAMATS_Q = '\u05BC', '\u05C1', '\u05C2', '\u05C7'

NIKUD_RE = re.compile(r'[\u05B0-\u05C7]')
HEB_RE   = re.compile(r'[\u05D0-\u05EA]')


def strip_nikud(s: str) -> str:
    """Remove Hebrew nikud marks (U+0591–U+05C7)."""
    return re.sub(r'[\u0591-\u05C7]', '', s)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 1: BASIC FIXES  (notebook cell 5)
# ═════════════════════════════════════════════════════════════════════════════

def basic_fixes(lines: list[str]) -> tuple[list[str], dict]:
    """Orphaned </small>, parens completion, whitespace, seifim separation."""

    # 1.1 — Merge orphaned </small> tags
    fixed, orphan_count, i = [], 0, 0
    while i < len(lines):
        if lines[i].strip() == '</small>':
            if fixed:
                fixed[-1] = fixed[-1].rstrip('\n') + '</small>\n'
                orphan_count += 1
            i += 1
        else:
            fixed.append(lines[i]); i += 1
    lines = fixed

    # 1.2 — Complete missing parentheses at depth 2
    paren_count = 0
    for i in range(len(lines)):
        parts = re.split(r'(<small>|</small>)', lines[i])
        depth = d2_o = d2_c = 0
        has_d2 = False
        for p in parts:
            if p == '<small>': depth += 1; continue
            if p == '</small>': depth -= 1; continue
            if depth >= 2 and p.strip():
                has_d2 = True
                d2_o += p.count('('); d2_c += p.count(')')
        if not has_d2 or d2_o <= d2_c:
            continue
        missing = d2_o - d2_c
        depth, last = 0, -1
        new_parts = []
        for p in parts:
            new_parts.append(p)
            if p == '<small>': depth += 1
            elif p == '</small>': depth -= 1
            elif depth >= 2 and p.strip():
                last = len(new_parts) - 1
        if last >= 0:
            new_parts[last] = new_parts[last].rstrip() + ')' * missing
            lines[i] = ''.join(new_parts)
            paren_count += 1

    # 1.3 — Whitespace normalization
    ws_count = 0
    for i in range(len(lines)):
        if not lines[i].strip():
            lines[i] = '\n'; continue
        new = re.sub(r' {2,}', ' ', lines[i]).rstrip() + '\n'
        if new != lines[i]:
            ws_count += 1; lines[i] = new

    # 1.4 — Separate adjacent seifim (insert blank line between content lines)
    result, prev_c, sep_count, started = [], False, 0, False
    for line in lines:
        tr = line.strip()
        is_siman   = bool(re.match(r'^Siman \d+$', tr))
        is_header  = bool(re.match(r'^(Shulchan|שולחן|Torat|http)', tr))
        is_content = bool(tr) and not is_siman and not is_header
        if is_siman: started = True
        if started and is_content and prev_c:
            result.append('\n'); sep_count += 1
        result.append(line)
        if not tr or is_siman or is_header: prev_c = False
        elif is_content: prev_c = True
    lines = result

    balanced = (sum(l.count('<small>') for l in lines)
                == sum(l.count('</small>') for l in lines))

    stats = {
        'orphans':     orphan_count,
        'parens':      paren_count,
        'whitespace':  ws_count,
        'seif_sep':    sep_count,
        'tag_balance': 'OK' if balanced else 'unbalanced',
    }
    return lines, stats


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 2: <small> CLEANUP  (notebook cell 7)
# ═════════════════════════════════════════════════════════════════════════════

EXPLANATION_RE = re.compile(r"^\(\s*(פי'|פירוש)\s*")


def parse_small_ast(line: str):
    """Parse a line into a tree of (type, content):
       ('text', str) | ('small', [children]).
    """
    tokens = re.split(r'(<small>|</small>)', line)
    stack = [[]]
    for tok in tokens:
        if tok == '<small>':
            new = []
            stack[-1].append(('small', new))
            stack.append(new)
        elif tok == '</small>':
            if len(stack) > 1:
                stack.pop()
        else:
            stack[-1].append(('text', tok))
    return stack[0]


def is_citation(content: str) -> bool:
    """Heuristic classifier for content at depth ≥ 2."""
    stripped = content.strip()
    if not stripped:
        return True
    if EXPLANATION_RE.match(stripped):
        return False
    if len(stripped) <= 2 and not HEB_RE.search(stripped):
        return True
    if '(' in stripped or ')' in stripped:
        return True
    return False


def render_ast(ast, depth: int = 1, audit=None, ctx=None) -> str:
    """Re-render the AST, dropping depth ≥ 2 blocks classified as references."""
    out = []
    for node_type, content in ast:
        if node_type == 'text':
            out.append(content)
            continue
        inner = render_ast(content, depth + 1, audit, ctx)
        plain = re.sub(r'</?small>', '', inner)
        plain_norm = re.sub(r'\s+', ' ', plain).strip()
        if depth >= 2:
            drop = is_citation(plain_norm)
            if audit is not None:
                audit.append({
                    'action': 'DROP' if drop else 'KEEP',
                    'depth':  depth,
                    'siman':  ctx.get('siman') if ctx else None,
                    'line':   ctx.get('line_num') if ctx else None,
                    'content': plain_norm,
                })
            if drop:
                continue
        out.append('<small>' + inner + '</small>')
    return ''.join(out)


def strip_nested_citations(line: str, audit=None, ctx=None) -> str:
    ast = parse_small_ast(line)
    cleaned = render_ast(ast, depth=1, audit=audit, ctx=ctx)
    return re.sub(r'  +', ' ', cleaned)


def strip_depth1_refs(line: str) -> tuple[str, int]:
    """Drop <small> blocks at depth 1 that are mechaber references (not hagah).

    State machine:
      - block starts with 'הגה'          → state = rema       → keep
      - state == rema and block has text → continuation of rema → keep
      - otherwise                         → mechaber reference → drop
    """
    parts = re.split(r'(<small>|</small>)', line)
    depth = 0
    state = 'author'
    result = []
    ref_count = 0
    i = 0
    while i < len(parts):
        part = parts[i]
        if part == '<small>':
            depth += 1
            if depth == 1:
                # Collect the entire balanced block
                inner_parts = [part]
                j, d = i + 1, 1
                while j < len(parts) and d > 0:
                    inner_parts.append(parts[j])
                    if parts[j] == '<small>': d += 1
                    elif parts[j] == '</small>': d -= 1
                    j += 1
                plain = ''.join(p for p in inner_parts
                                if p not in ('<small>', '</small>'))
                plain = re.sub(r'[\u0591-\u05C7]', '', plain)
                plain = re.sub(r'\s+', ' ', plain).strip()
                if plain.startswith('הגה'):
                    state = 'rema'
                    result.extend(inner_parts)
                elif state == 'rema' and plain:
                    result.extend(inner_parts)
                else:
                    ref_count += 1
                i = j
                depth = 0
                continue
            else:
                result.append(part)
        elif part == '</small>':
            depth = max(0, depth - 1)
            result.append(part)
        else:
            if depth == 0 and part.strip() and HEB_RE.search(part):
                state = 'author'
            result.append(part)
        i += 1
    cleaned = ''.join(result)
    cleaned = re.sub(r'  +', ' ', cleaned)
    return cleaned, ref_count


def clean_small_tags(lines: list[str]) -> tuple[list[str], dict]:
    """Run depth-2+ citation strip, then depth-1 mechaber ref strip on all lines."""
    audit_entries = []
    current_siman = None
    clean_count = 0
    total_d1_refs = 0

    for i, line in enumerate(lines):
        tr = line.strip()
        m = re.match(r'^Siman (\d+)$', tr)
        if m:
            current_siman = int(m.group(1))
            continue
        if not tr or re.match(r'^(Shulchan|שולחן|Torat|http)', tr):
            continue
        ctx = {'siman': current_siman, 'line_num': i + 1}
        cleaned = strip_nested_citations(line, audit=audit_entries, ctx=ctx)
        cleaned, d1 = strip_depth1_refs(cleaned)
        total_d1_refs += d1
        if cleaned != line:
            clean_count += 1
            lines[i] = cleaned

    drops = sum(1 for e in audit_entries if e['action'] == 'DROP')
    keeps = sum(1 for e in audit_entries if e['action'] == 'KEEP')

    return lines, {
        'cleaned_lines': clean_count,
        'd2_total':      len(audit_entries),
        'd2_dropped':    drops,
        'd2_kept':       keeps,
        'd1_dropped':    total_d1_refs,
    }


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 3: KTIV MALE CONVERSION  (notebook cell 9)
# ═════════════════════════════════════════════════════════════════════════════

SHORT_WORDS_NO_VAV = {
    'לא','הלא','ולא','שלא','בלא','דלא','אלא',
    'כל','וכל','בכל','לכל','מכל','שכל','דכל','ככל',
    'כך','וכך','לכך','על','של','אל',
    'עד','גם','כן','פן','אם','עם','מן','זה','זו','את',
}

# "מים" exceptions — these keep their ktiv haser form
DUAL_SUFFIX_EXCEPTIONS = {'מים', 'שמים', 'ירושלים', 'מצרים'}

PE_ALEPH_FUTURE_PREFIXES = {'ת', 'י', 'נ'}


def tokenize(text: str) -> list[str]:
    return re.findall(r'[\u05D0-\u05EA][\u05D0-\u05EA\u0591-\u05C7]*|[^\u05D0-\u05EA]+', text)


def parse_word(word: str) -> list[tuple[str, set]]:
    tokens, i = [], 0
    while i < len(word):
        ch = word[i]
        if HEB_RE.match(ch):
            marks, j = [], i + 1
            while j < len(word) and NIKUD_RE.match(word[j]):
                marks.append(word[j]); j += 1
            tokens.append((ch, set(marks))); i = j
        else:
            i += 1
    return tokens


def has_em_kriah_after(letters, idx):
    return idx + 1 < len(letters) and letters[idx + 1][0] in 'אעה'


def is_consonantal_vav(letters, idx):
    ch, marks = letters[idx]
    if ch != 'ו': return False
    return bool(marks & {QAMATS, PATAH, SHVA, HIRIQ, TSERE, SEGOL,
                          HATAF_P, HATAF_Q, HATAF_S})


def has_chirik_yod_mem_ending(letters, idx):
    if idx < 1 or idx + 1 >= len(letters):
        return False
    _, prev_marks = letters[idx - 1]
    cur_ch, cur_marks = letters[idx]
    next_ch, _ = letters[idx + 1]
    if cur_ch == 'י' and HIRIQ in cur_marks and next_ch in 'םמ':
        if PATAH in prev_marks or QAMATS in prev_marks:
            return True
    return False


def convert_word(word: str) -> str:
    plain = strip_nikud(word)
    if plain in SHORT_WORDS_NO_VAV:
        return plain
    if plain in DUAL_SUFFIX_EXCEPTIONS:
        return plain
    letters = parse_word(word)
    if not letters:
        return plain
    result = []
    n = len(letters)
    for idx, (ch, marks) in enumerate(letters):
        next_ch    = letters[idx + 1][0] if idx + 1 < n else None
        next_marks = letters[idx + 1][1] if idx + 1 < n else set()
        prev_ch    = letters[idx - 1][0] if idx > 0 else None
        result.append(ch)
        # consonantal vav mid-word → double vav
        if ch == 'ו' and 0 < idx < n - 1 and is_consonantal_vav(letters, idx):
            if prev_ch != 'ו' and next_ch != 'ו':
                result.append('ו')
        # kubutz → vav
        if QUBUTS in marks and ch != 'ו' and next_ch != 'ו':
            result.append('ו')
        # holam → vav (with several exceptions)
        if HOLAM in marks and ch != 'ו':
            add = True
            if has_em_kriah_after(letters, idx):
                add = False
            if (idx == 0 and ch in PE_ALEPH_FUTURE_PREFIXES
                and n >= 3 and letters[1][0] == 'א'):
                add = False
            if next_ch == 'ו':
                add = False
            if add:
                result.append('ו')
        # kamatz katan → vav
        if QAMATS_Q in marks and ch != 'ו' and next_ch != 'ו':
            result.append('ו')
        # kamatz + shva (kamatz katan in practice) → vav
        # dagesh on the letter itself = not kamatz katan (pi'el/nif'al/hitpa'el)
        if QAMATS in marks and SHVA in next_marks and DAGESH not in marks:
            if ch != 'ו' and next_ch != 'ו' and plain not in SHORT_WORDS_NO_VAV:
                if ch not in 'בכלמשהוד':
                    if not has_em_kriah_after(letters, idx):
                        result.append('ו')
        # dual suffix ַיִם → add yod
        if has_chirik_yod_mem_ending(letters, idx):
            result.append('י')
        # consonantal yod mid-word → double yod (but not after prefixes, not before vav)
        if ch == 'י' and DAGESH in marks and 0 < idx < n - 1:
            if prev_ch != 'י' and next_ch != 'י' and next_ch != 'ו':
                all_prefix_before = all(letters[j][0] in 'בכלמשהוד' for j in range(idx))
                if not all_prefix_before:
                    result.append('י')
        # hirik before dagesh hazak → add yod (תפילה, אפילו)
        if idx > 0 and HIRIQ in marks and ch != 'י' and next_ch != 'י':
            if DAGESH in next_marks:
                result.append('י')
    return ''.join(result)


# Semantic homograph disambiguation: nikud → unambiguous form
DISAMBIG_TABLE = {
    'אחר': [
        (lambda L: len(L) >= 2 and PATAH in L[1][1], 'אחרי'),
    ],
    'קדוש': [
        (lambda L: HIRIQ in L[0][1], 'קידוש'),
    ],
    'אסור': [
        (lambda L: HIRIQ in L[0][1], 'איסור'),
    ],
    'עולה': [
        (lambda L: len(L) >= 3 and QAMATS in L[2][1], 'עולה'),
    ],
    'דין': [
        (lambda L: len(L) >= 2 and PATAH in L[0][1]
                   and (QAMATS in L[1][1] or PATAH in L[1][1]), 'דיין'),
    ],
    'מותר': [
        (lambda L: len(L) >= 3 and QAMATS in L[2][1] and HOLAM in L[0][1], 'מותר'),
    ],
}


def nikud_to_ktiv_male(text: str) -> str:
    tokens = tokenize(text)
    result = []
    for tok in tokens:
        if not any(HEB_RE.match(c) for c in tok):
            result.append(strip_nikud(tok))
            continue
        plain = strip_nikud(tok)
        if plain in DISAMBIG_TABLE:
            letters = parse_word(tok)
            replaced = False
            for cond, replacement in DISAMBIG_TABLE[plain]:
                if letters and cond(letters):
                    result.append(replacement)
                    replaced = True
                    break
            if not replaced:
                result.append(convert_word(tok))
        else:
            result.append(convert_word(tok))
    return ''.join(result)


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 4: SEMANTIC UNIFICATION  (notebook cell 11)
# ═════════════════════════════════════════════════════════════════════════════

PREFIX_CLASS    = r'[ובלמשכהד]'
PREFIX_UP_TO_3  = rf'{PREFIX_CLASS}{{0,3}}'
HEBREW          = r'[\u05D0-\u05EA]'

STEM_UNIFICATIONS = [
    # Divine names
    ('הקב"ה', 'אלוהים'),
    ('השי"ת', 'אלוהים'),
    ('אלקים', 'אלוהים'),
    ('אלהים', 'אלוהים'),
    # Non-Jewish
    ('עכו"ם', 'גוי'),
    ('כותי',  'גוי'),
    ('נכרי',  'גוי'),
    ('נכרים', 'גוים'),
]

DELETE_STEMS = ["סי'", 'סעיף', 'עיין']
DELETE_WORDS = ['עי"ש', 'וע"ש']

# Context where ה' is numeric / reference, not divine
HEY_NOT_GOD_PREV = {
    'סעיף', 'סעיפים', 'פרק', 'פרקים', "פ'", 'פ"ק', 'סימן', 'סימנים',
    'כמנין', 'מנין', "סי'", 'הלכה', 'הלכות', 'ביום', 'יום',
    'באות', 'אות', 'דף', 'שער', 'שורה', 'עמוד', 'כלל',
}
HEY_NOT_GOD_NEXT = {
    'טפחים', 'אמות', 'מילין', 'פרסאות', 'שנים', 'שנה',
    'ימים', 'חדשים', 'שבועות', 'שעות', 'דקות',
    'טפחים,', 'אמות,', 'ימים,', 'שנים,',
}


def _word_bnd_pattern(term: str) -> re.Pattern:
    return re.compile(rf'(?<!{HEBREW})' + re.escape(term) + rf'(?!{HEBREW})')


def _stem_with_prefix_pattern(stem: str) -> re.Pattern:
    return re.compile(rf'(?<!{HEBREW})({PREFIX_UP_TO_3}){re.escape(stem)}(?!{HEBREW})')


def disambiguate_hey(text: str) -> tuple[str, int, int]:
    """Replace ה' → אלוהים only when context is *not* numeric/reference."""
    replacements = 0
    kept = 0

    def repl(m):
        nonlocal replacements, kept
        start, end = m.start(), m.end()
        left = text[max(0, start - 20):start]
        prev_tokens = left.strip().split()
        prev_word = prev_tokens[-1] if prev_tokens else ''
        right = text[end:end + 20]
        next_tokens = right.strip().split()
        next_word = next_tokens[0] if next_tokens else ''
        if prev_word in HEY_NOT_GOD_PREV or next_word in HEY_NOT_GOD_NEXT:
            kept += 1
            return m.group(0)
        replacements += 1
        return 'אלוהים'

    new_text = _word_bnd_pattern("ה'").sub(repl, text)
    return new_text, replacements, kept


def apply_stem_replacement(text, stem, target, _unused=None) -> tuple[str, int]:
    pat = _stem_with_prefix_pattern(stem)
    count = len(pat.findall(text))
    def repl(m): return m.group(1) + target
    return pat.sub(repl, text), count


def apply_stem_deletion(text, stem) -> tuple[str, int]:
    pat = _stem_with_prefix_pattern(stem)
    count = len(pat.findall(text))
    return pat.sub('', text), count


def apply_synonym_unification(text: str) -> tuple[str, dict]:
    """Order of operations:
       1. Context-aware ה'
       2. Stem unification (divine names, goy)
       3. Full reference deletion (עיין/ע"ל/לקמן/לעיל + content)
       4. פי'/פירוש handling
       5. Stem deletion (סי'/סעיף/עיין)
       6. Whole-word deletions (עי"ש/וע"ש)
       7. ד'/ג'/ב' + measure words → ארבע/שלוש/שתי
       8. Section-name deletion (יו"ד/חו"מ/...)
       9. Abbreviation expansion (וה"ה → והוא הדין וכו')
    """
    stats = {}

    # 1. ה'
    text, hey_rep, hey_keep = disambiguate_hey(text)
    stats["HEY_→אלוהים"] = hey_rep
    stats["HEY_נשאר_מספרי"] = hey_keep

    # 2. stem unification
    for stem, tgt in STEM_UNIFICATIONS:
        text, count = apply_stem_replacement(text, stem, tgt)
        if count:
            stats[f'UNIFY_{stem}→{tgt}'] = count

    # 3. full references
    ref_in_small = re.compile(
        r'<small>\s*(?:ו?עיין|ו?ע"ל|ע"ל|ו?כד?לקמן|ו?כד?לעיל|ו?לקמן|ו?לעיל)\s*[^<]*?</small>'
    )
    c1 = len(ref_in_small.findall(text));  text = ref_in_small.sub('', text)

    ref_bare_ayin = re.compile(
        r'(?<![\u05D0-\u05EA])(?:ו?עיין|ו?ע"ל)\s+[^.:\n<]{1,80}[.:]?'
    )
    c2 = len(ref_bare_ayin.findall(text)); text = ref_bare_ayin.sub('', text)

    ref_lkm = re.compile(
        r'(?<![\u05D0-\u05EA])(?:ו?כד?לקמן|ו?כד?לעיל|ו?לקמן|ו?לעיל)\s+[^.,:;\n<]{0,80}[.,;:]?'
    )
    c3 = len(ref_lkm.findall(text));       text = ref_lkm.sub('', text)

    ref_standalone = re.compile(
        r'(?<![\u05D0-\u05EA])(?:ו?כדלקמן|ו?כדלעיל)(?![\u05D0-\u05EA])'
    )
    c4 = len(ref_standalone.findall(text)); text = ref_standalone.sub('', text)
    stats['DELETE_הפניות'] = c1 + c2 + c3 + c4

    # 4. פי'/פירוש
    expand_re = re.compile(r"פי'")
    expand_count = len(expand_re.findall(text))
    text = expand_re.sub('פירוש', text)

    delete_pirush = re.compile(r'(?<![\u05D0-\u05EA])פירוש\s*')
    delete_count = len(delete_pirush.findall(text))
    text = delete_pirush.sub('', text)

    stats["EXPAND_פי'→פירוש"] = expand_count
    stats["DELETE_פירוש"] = delete_count

    # 5. stem deletion
    for stem in DELETE_STEMS:
        text, count = apply_stem_deletion(text, stem)
        stats[f'DELETE_{stem}'] = count

    # 6. whole words
    for word in DELETE_WORDS:
        pat = _word_bnd_pattern(word)
        count = len(pat.findall(text))
        text = pat.sub('', text)
        if count:
            stats[f'DELETE_{word}'] = count

    # 7. numeric abbreviations + measure words
    MEASURE_WORDS_RE = (r'(?:אמות|טפחים|כוסות|מינים|כנפות|פרסאות|מילין|ציציות|'
                        r'ברכות|פעמים|שעות|ימים|חדשים|שבועות|אצבעות|זיתים|'
                        r'ביצים|גריסין|רביעיות)')
    for letter, number in [("ד'", 'ארבע'), ("ג'", 'שלוש'), ("ב'", 'שתי')]:
        pat = re.compile(re.escape(letter) + r'\s+(' + MEASURE_WORDS_RE + ')')
        count = len(pat.findall(text))
        text = pat.sub(number + r' \1', text)
        if count:
            stats[f'EXPAND_{letter}→{number}'] = count

    # 8. section refs
    SECTION_REFS = ['יו"ד', 'חו"מ', 'או"ח', 'אה"ע',
                    'יורה דעה', 'חושן משפט', 'אורח חיים', 'אבן העזר']
    for ref in SECTION_REFS:
        pat = _word_bnd_pattern(ref)
        c = len(pat.findall(text))
        text = pat.sub('', text)
        if c:
            stats[f'DELETE_{ref}'] = c

    # 9. known abbreviation expansions
    for abbr, expansion in [('וה"ה', 'והוא הדין'), ('ה"ה', 'הוא הדין'),
                             ('נ"ל', 'הנזכר לעיל'),
                             ('ג"ט', 'שלושה טפחים'), ('ד"ט', 'ארבעה טפחים')]:
        pat = _word_bnd_pattern(abbr)
        c = len(pat.findall(text))
        text = pat.sub(expansion, text)
        if c:
            stats[f'EXPAND_{abbr}'] = c

    # whitespace
    text = re.sub(r'  +', ' ', text)
    text = re.sub(r'^ +| +$', '', text, flags=re.MULTILINE)
    return text, stats


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 5: NUMERIC ABBREVIATION EXPANSION ON VOWELED TEXT  (cell 13 inline)
# ═════════════════════════════════════════════════════════════════════════════

GEMATRIA_V = {
    'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,
    'י':10,'כ':20,'ל':30,'מ':40,'נ':50,'ס':60,'ע':70,'פ':80,'צ':90,
    'ק':100,'ר':200,'ש':300,'ת':400,
}
U_M    = ['','אחד','שניים','שלושה','ארבעה','חמישה','שישה','שבעה','שמונה','תשעה']
T_M    = ['עשרה','אחד עשר','שנים עשר','שלושה עשר','ארבעה עשר','חמישה עשר',
          'שישה עשר','שבעה עשר','שמונה עשר','תשעה עשר']
TENS_W = ['','עשר','עשרים','שלושים','ארבעים','חמישים','ששים','שבעים','שמונים','תשעים']
HUND   = ['','מאה','מאתיים','שלוש מאות','ארבע מאות',
          'חמש מאות','שש מאות','שבע מאות','שמונה מאות','תשע מאות']

NOT_NUMBERS = {'נ"ל','ה"ה','וה"ה','ד"מ','ר"ת','ש"ע','ש"ץ','נ"ך','ס"ת','ע"ב','ח"ל'}
NIKUD_MARKS = set(chr(c) for c in range(0x05B0, 0x05C8))
NUM_ABBREV_NIKUD = re.compile(
    r'[\u05D0-\u05EA\u0591-\u05C7]{1,6}"[\u05D0-\u05EA\u0591-\u05C7]{1,3}'
)


def _gval(s: str) -> int:
    return sum(GEMATRIA_V.get(c, 0) for c in s if 'א' <= c <= 'ת')


def _n2h(n: int) -> Optional[str]:
    if n < 2 or n > 999:
        return None
    if n >= 100:
        h = n // 100
        if h >= len(HUND):
            return None
        rest = n % 100
        if rest == 0:
            return HUND[h]
        rest_h = _n2h(rest)
        return f'{HUND[h]} {rest_h}' if rest_h else None
    if 10 <= n <= 19:
        return T_M[n - 10]
    parts = []
    if n >= 20:
        parts.append(TENS_W[n // 10])
    if n % 10:
        parts.append(U_M[n % 10])
    return ' '.join(parts)


def _valid_gematria(chars: list[str]) -> bool:
    """Valid gematria = strict descending values (no ties, no zeros)."""
    if not chars or len(chars) > 5:
        return False
    vals = [GEMATRIA_V.get(c, 0) for c in chars]
    if any(v == 0 for v in vals):
        return False
    for i in range(len(vals) - 1):
        if vals[i] <= vals[i + 1]:
            return False
    return True


def expand_numeric_abbrev(line: str) -> tuple[str, int]:
    """Expand gematria abbreviations on voweled text (golden rule:
    if letters have nikud = sage name; no nikud = number)."""
    count = [0]

    def repl(m):
        tok = m.group(0)
        plain = strip_nikud(tok)
        if plain in NOT_NUMBERS:
            return tok
        parsed = []
        i = 0
        while i < len(tok):
            if '\u05D0' <= tok[i] <= '\u05EA':
                marks = set(); j = i + 1
                while j < len(tok) and tok[j] in NIKUD_MARKS:
                    marks.add(tok[j]); j += 1
                parsed.append((tok[i], marks)); i = j
            else:
                i += 1
        # split voweled prefix from unvoweled core
        prefix_end = 0
        for idx, (ch, marks) in enumerate(parsed):
            if marks and ch in 'בכלמשהוד':
                prefix_end = idx + 1
            else:
                break
        # prefix vav with kamatz = noun (וָי"ו), don't convert
        for idx in range(prefix_end):
            ch, marks = parsed[idx]
            if ch == 'ו' and QAMATS in marks:
                return tok
        core = parsed[prefix_end:]
        if any(marks for _, marks in core):
            return tok
        core_chars = [ch for ch, _ in core]
        if not _valid_gematria(core_chars):
            return tok
        val = _gval(''.join(core_chars))
        heb = _n2h(val)
        if heb and 2 <= val <= 99:
            count[0] += 1
            prefix_str = ''.join(ch for ch, _ in parsed[:prefix_end])
            return prefix_str + heb
        return tok

    return NUM_ABBREV_NIKUD.sub(repl, line), count[0]


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 6: MAIN PROCESSING LOOP + FINAL CLEANUP  (notebook cell 13)
# ═════════════════════════════════════════════════════════════════════════════

def run_pipeline(lines: list[str]) -> tuple[list[str], Counter]:
    processed_lines = []
    content_lines_count = 0
    total_stats = Counter()

    for line in lines:
        tr = line.strip()
        if not tr or re.match(r'^(Siman \d+|Shulchan|שולחן|Torat|http)', tr):
            processed_lines.append(line)
            continue
        content_lines_count += 1
        # step 0: numeric abbreviations on voweled text
        line, n_expanded = expand_numeric_abbrev(line)
        total_stats['EXPAND_מספרים'] += n_expanded
        # step 1: ktiv male (preserves <small>)
        converted = nikud_to_ktiv_male(line)
        # step 2: semantic unification
        unified, stats = apply_synonym_unification(converted)
        for k, v in stats.items():
            total_stats[k] += v
        if not unified.endswith('\n'):
            unified += '\n'
        processed_lines.append(unified)

    # Final cleanup
    for i, line in enumerate(processed_lines):
        # remove <small> with punctuation only (no Hebrew)
        line = re.sub(r'<small>\s*([^<]{0,5})\s*</small>',
                      lambda m: '' if not re.search(r'[\u05D0-\u05EA]', m.group(1)) else m.group(0),
                      line)
        line = re.sub(r'<small>\s*</small>', '', line)
        line = re.sub(r' +([.,:;])', r'\1', line)
        line = re.sub(r'([.,:;])\1+', r'\1', line)
        line = re.sub(r'  +', ' ', line)
        line = line.strip() + '\n' if line.strip() else '\n'
        processed_lines[i] = line

    # collapse multiple blank lines
    final_lines = []
    prev_blank = False
    for line in processed_lines:
        is_blank = not line.strip()
        if is_blank and prev_blank:
            continue
        final_lines.append(line)
        prev_blank = is_blank

    total_stats['_content_lines'] = content_lines_count
    total_stats['_output_lines']  = len(final_lines)
    return final_lines, total_stats


# ═════════════════════════════════════════════════════════════════════════════
# STAGE 7: STRUCTURED PARSING  (notebook cells 15–17)
# ═════════════════════════════════════════════════════════════════════════════

def merge_orphan_tags_s2(lines: list[str]) -> list[str]:
    fixed = []
    for line in lines:
        if line.strip() == '</small>' and fixed:
            fixed[-1] = fixed[-1].rstrip('\n') + '</small>\n'
        else:
            fixed.append(line)
    return fixed


def parse_torat_emet_to_seifim(lines_list: list[str],
                               fix_orphans: bool = False
                               ) -> list[tuple[int, list[str]]]:
    if fix_orphans:
        lines_list = merge_orphan_tags_s2(lines_list)
    result = []
    current_siman = None
    current_seifim: list[str] = []
    for line in lines_list:
        tr = line.strip()
        m = re.match(r'^Siman (\d+)$', tr)
        if m:
            if current_siman is not None:
                result.append((current_siman, current_seifim))
            current_siman = int(m.group(1))
            current_seifim = []
            continue
        if not tr or re.match(r'^(Shulchan|שולחן|Torat|http)', tr):
            continue
        if not re.sub(r'</?small>', '', tr).strip():
            continue
        current_seifim.append(line.rstrip('\n'))
    if current_siman is not None:
        result.append((current_siman, current_seifim))
    return result


def parse_seif(normalized_line: str) -> dict:
    """Split a normalized line into {text: mechaber, hagah: rema}."""
    parts = re.split(r'(<small>|</small>)', normalized_line)
    depth = 0
    mechaber_parts, hagah_parts = [], []
    state = 'author'  # 'author' | 'rema'

    for part in parts:
        if part == '<small>':
            depth += 1; continue
        if part == '</small>':
            depth = max(0, depth - 1); continue
        if depth == 0:
            mechaber_parts.append(part)
            if part.strip() and HEB_RE.search(part):
                state = 'author'
        elif depth == 1:
            stripped = part.strip()
            if stripped.startswith('הגה'):
                state = 'rema'
                hagah_parts.append(part)
            elif state == 'rema':
                hagah_parts.append(part)
            # depth-1 non-hagah = mechaber ref → skip

    def clean(ps):
        return re.sub(r'\s+', ' ', ''.join(ps)).strip()

    text  = clean(mechaber_parts).strip(': ')
    hagah = clean(hagah_parts)
    hagah = re.sub(r'(?<![\u05D0-\u05EA])הגה\s*:\s*', '', hagah).strip()

    return {'text': text or None, 'hagah': hagah or None}


def build_rag_json(raw_data, norm_data) -> dict:
    simanim = []
    for (siman_num, raw_lines), (_, norm_lines) in zip(raw_data, norm_data):
        seifim = []
        for i, (raw, norm) in enumerate(zip(raw_lines, norm_lines)):
            parsed = parse_seif(norm)
            seifim.append({
                'seif':     i + 1,
                'text':     parsed['text'],
                'hagah':    parsed['hagah'],
                'text_raw': raw,
            })
        simanim.append({'siman': siman_num, 'seifim': seifim})
    return {
        'title':   'שולחן ערוך, אורח חיים',
        'source':  'Torat Emet 363',
        'simanim': simanim,
    }


# ═════════════════════════════════════════════════════════════════════════════
# REGRESSION TESTS  (imported verbatim from notebook cells 9, 11, 15)
# ═════════════════════════════════════════════════════════════════════════════

KTIV_MALE_TESTS = [
    ("יִתְגַּבֵּר כַּאֲרִי לַעֲמֹד בַּבֹּקֶר", "יתגבר כארי לעמוד בבוקר"),
    ("חָכְמָה",    "חוכמה"),
    ("קָרְבָּן",   "קורבן"),
    ("אָמְנָם",    "אומנם"),
    ("מִצְוָה",    "מצווה"),
    ("תִּקְוָה",   "תקווה"),
    ("עֵינַיִם",   "עיניים"),
    ("רַגְלַיִם",  "רגליים"),
    ("יָדַיִם",    "ידיים"),
    ("אָזְנַיִם",  "אוזניים"),
    ("מַיִם",          "מים"),
    ("שָׁמַיִם",       "שמים"),
    ("יְרוּשָׁלַיִם",  "ירושלים"),
    ("מִצְרַיִם",      "מצרים"),
    ('עַכּוּ"ם',   'עכו"ם'),
    ('רַמְבַּ"ם',  'רמב"ם'),
    ("ה'",         "ה'"),
    ("בָּרְכוּ",     "ברכו"),
    ("הָרְצוּעוֹת",  "הרצועות"),
    ("אַחַר",        "אחרי"),
    ("אַחֵר",        "אחר"),
    ("קִדּוּשׁ",     "קידוש"),
    ("קָדוֹשׁ",      "קדוש"),
    ("אִסּוּר",      "איסור"),
    ("אָסוּר",       "אסור"),
    ("שְׁנַיִם",     "שניים"),
    ("שָׁנִים",       "שנים"),
    ("תְּפִלָּה",     "תפילה"),
    ("אֲפִלּוּ",      "אפילו"),
    ("תְּפִלִּין",    "תפילין"),
    ("חַיָּב",         "חייב"),
    ("נִתְפָּרְקוּ",    "נתפרקו"),
    ("גָּזְרוּ",        "גזרו"),
    ("בַּיּוֹם",         "ביום"),
    ("הַיּוֹם",          "היום"),
]


UNIFICATION_TESTS = [
    ('שהרי הקב"ה מקדים לברך',      'שהרי אלוהים מקדים לברך'),
    ('הבורא יתברך השי"ת ברא',       'הבורא יתברך אלוהים ברא'),
    ('להקב"ה אין גבול',             'לאלוהים אין גבול'),
    ("חייב לברך את ה' אלהינו",      'חייב לברך את אלוהים אלהינו'),
    ("סעיף ה' מסכם",                "ה' מסכם"),
    ("ה' טפחים",                    "ה' טפחים"),
    ("ברוך ה'",                     'ברוך אלוהים'),
    ('עכו"ם שבא לבית',               'גוי שבא לבית'),
    ('העכו"ם מחמתו',                 'הגוי מחמתו'),
    ('ולעכו"ם לא נותנים',            'ולגוי לא נותנים'),
    ('כותי אחד ונכרי אחר',           'גוי אחד וגוי אחר'),
    ('<small> וע"ל מ"ז י"ג. </small>',                 ''),
    ('טקסט לפני <small> ע"ל סימן ל"ג. </small> אחרי', 'טקסט לפני אחרי'),
    ('טקסט ועיין בסימן מ"ז.',                          'טקסט'),
    ('כדלקמן סימן רט"ו',                               ''),
    ('ברכות שמברך קודם נטילה כדלעיל סימן מ"ו.',        'ברכות שמברך קודם נטילה'),
    ('ביקר לעיל סימן י"א ב\'. טקסט.',                 'ביקר טקסט.'),
    ('אמר לקמן סימן כ"ה י"ב.',                          'אמר'),
    ("(פי' מקום שיכולים לראות)",      "(מקום שיכולים לראות)"),
    ("(פירוש תלמידים)",              "(תלמידים)"),
    ("רבנו תם פי' שהוא כעין עטרה",   "רבנו תם שהוא כעין עטרה"),
    ("דברי עצמו לפי' הטור",          "דברי עצמו לפירוש הטור"),
    ("ד' אמות",                      "ארבע אמות"),
    ("ג' פעמים",                     "שלוש פעמים"),
    ("ב' כוסות",                     "שתי כוסות"),
    ("ד' על ד' טפחים",              "ד' על ארבע טפחים"),
]


PARSE_SEIF_TEST = (
    'טקסט מחבר <small> הגה: דברי רמ"א חלק א </small> <small> הגה: דברי רמ"א חלק ב </small>',
    'דברי רמ"א חלק א דברי רמ"א חלק ב',
)


def run_tests(verbose: bool = True) -> bool:
    ok = True

    print("── Ktiv male tests ──")
    fails = 0
    for menukad, expected in KTIV_MALE_TESTS:
        got = nikud_to_ktiv_male(menukad)
        passing = got == expected
        fails += not passing
        if verbose or not passing:
            mark = '✓' if passing else '✗'
            print(f'  {mark} {strip_nikud(menukad):>16} → {got:<16} (expected: {expected})')
    print(f'  {len(KTIV_MALE_TESTS)-fails}/{len(KTIV_MALE_TESTS)} passing')
    if fails: ok = False

    print("\n── Unification tests ──")
    fails = 0
    for inp, expected in UNIFICATION_TESTS:
        out, _ = apply_synonym_unification(inp)
        passing = out.strip() == expected.strip()
        fails += not passing
        if verbose or not passing:
            mark = '✓' if passing else '✗'
            print(f'  {mark} {inp!r}')
            print(f'     → {out!r}')
            if not passing:
                print(f'     ✗ expected: {expected!r}')
    print(f'  {len(UNIFICATION_TESTS)-fails}/{len(UNIFICATION_TESTS)} passing')
    if fails: ok = False

    print("\n── parse_seif test ──")
    sample, expected_hagah = PARSE_SEIF_TEST
    p = parse_seif(sample)
    passing = p['hagah'] == expected_hagah
    print(f'  text:  {p["text"]}')
    print(f'  hagah: {p["hagah"]}')
    print(f'  {"✓" if passing else "✗"} Two hagah blocks merged without internal "הגה:"')
    if not passing:
        ok = False
        print(f'     expected: {expected_hagah!r}')

    return ok


# ═════════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═════════════════════════════════════════════════════════════════════════════

def process_file(input_path: Path, output_path: Path, quiet: bool = False) -> None:
    log = (lambda *a, **k: None) if quiet else print

    # 1. Load
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")

    log(f"Loading: {input_path.name}")
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    n_lines_orig = len(lines)
    n_siman = sum(1 for l in lines if re.match(r'^Siman \d+$', l.strip()))
    log(f'  {n_lines_orig:,} lines, {n_siman} simanim')

    # Keep unmodified copy for text_raw
    original_lines = list(lines)

    # 2. Basic fixes
    log("\nStage 1: basic fixes")
    lines, s1 = basic_fixes(lines)
    log(f'  orphaned </small>: {s1["orphans"]}')
    log(f'  parens completed:  {s1["parens"]}')
    log(f'  spaces cleaned:    {s1["whitespace"]:,}')
    log(f'  seif separations:  {s1["seif_sep"]}')
    log(f'  tag balance:       {s1["tag_balance"]}')

    # 3. <small> cleanup
    log("\nStage 2: <small> cleanup")
    lines, s2 = clean_small_tags(lines)
    log(f'  lines cleaned:     {s2["cleaned_lines"]:,}')
    log(f'  d2 fragments:      {s2["d2_total"]:,} '
        f'(dropped {s2["d2_dropped"]:,}, kept {s2["d2_kept"]:,})')
    log(f'  d1 refs dropped:   {s2["d1_dropped"]:,}')

    # 4. Main pipeline
    log("\nStage 3: ktiv male + semantic unification + number expansion")
    processed_lines, s3 = run_pipeline(lines)
    log(f'  content lines:     {s3["_content_lines"]:,}')
    log(f'  output lines:      {s3["_output_lines"]:,}')
    log('  top unifications:')
    for k, v in sorted(s3.items(), key=lambda kv: -kv[1]):
        if k.startswith('_') or v <= 0: continue
        log(f'    {v:>5}  {k}')

    # 5. Structured parse + alignment check
    log("\nStage 4: structure parsing")
    raw_data  = parse_torat_emet_to_seifim(original_lines,  fix_orphans=True)
    norm_data = parse_torat_emet_to_seifim(processed_lines, fix_orphans=False)

    if len(raw_data) != len(norm_data):
        raise RuntimeError(
            f'Siman count mismatch: {len(raw_data)} in source, {len(norm_data)} in processed')

    misaligned = []
    for (s_r, sf_r), (s_n, sf_n) in zip(raw_data, norm_data):
        if s_r != s_n:
            misaligned.append(f'siman number: {s_r} vs {s_n}')
        if len(sf_r) != len(sf_n):
            misaligned.append(f'siman {s_r}: {len(sf_r)} vs {len(sf_n)} seifim')
    if misaligned:
        for m in misaligned[:10]: log('  ' + m)
        raise RuntimeError(f'Alignment check failed: {len(misaligned)} mismatches')

    log(f'  simanim:           {len(norm_data)}')
    log(f'  seifim (total):    {sum(len(s) for _, s in norm_data):,}')
    log(f'  alignment:         OK')

    # 6. Build JSON
    log("\nStage 5: JSON build")
    output = build_rag_json(raw_data, norm_data)
    total_seifim = sum(len(s['seifim']) for s in output['simanim'])
    hagah_count  = sum(1 for s in output['simanim']
                       for sf in s['seifim'] if sf['hagah'])
    empty_text   = sum(1 for s in output['simanim']
                       for sf in s['seifim'] if not sf['text'])
    log(f'  simanim:           {len(output["simanim"])}')
    log(f'  seifim:            {total_seifim:,}')
    log(f'    with hagah:      {hagah_count:,}  ({100*hagah_count/total_seifim:.1f}%)')
    log(f'    empty text:      {empty_text}')

    # 7. Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8', newline='\n') as f:
        json.dump(output, f, ensure_ascii=False, indent=2)
    size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log(f'\nSaved: {output_path}  ({size_mb:.2f} MB)')


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess Shulchan Arukh TXT into RAG-ready JSON.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__.split("Usage:")[1] if "Usage:" in (__doc__ or "") else "",
    )
    ap.add_argument("--input",  "-i", default=DEFAULT_INPUT,
                    help=f"input TXT file (default: {DEFAULT_INPUT})")
    ap.add_argument("--output", "-o", default=DEFAULT_OUTPUT,
                    help=f"output JSON file (default: {DEFAULT_OUTPUT})")
    ap.add_argument("--test",   action="store_true",
                    help="run regression tests and exit")
    ap.add_argument("--quiet",  "-q", action="store_true",
                    help="suppress progress output")
    args = ap.parse_args()

    if args.test:
        ok = run_tests(verbose=not args.quiet)
        sys.exit(0 if ok else 1)

    try:
        process_file(Path(args.input), Path(args.output), quiet=args.quiet)
    except FileNotFoundError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(2)
    except RuntimeError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(3)


if __name__ == "__main__":
    main()
