"""
Step 1 — Data Loading, Cleaning & Chunking
==========================================
Input:  Shulchan Arukh, Orach Chayim - he - Torat Emet 363.txt
Output: chunks.json

What this script does:
  1. Load the raw text file
  2. Clean: strip HTML tags, remove Hebrew vowel marks, normalize punctuation
  3. Split into overlapping chunks that never cross siman (chapter) boundaries
  4. Save chunks.json

Usage:
  python step_01_data_loading.py                  # basic split → chunks_v1.json
  python step_01_data_loading.py --siman-boundary # siman-aware split → chunks_v3.json
  python step_01_data_loading.py --v4-clean       # v4 cleaning + siman-aware → chunks_v4.json
"""

import re
import json
import argparse
from pathlib import Path
from config import TEXT_FILE  # path defined in config.py


def extract_simanim(text: str) -> list[int]:
    """Return all siman numbers found in a chunk (format: 'Siman N')."""
    return [int(m) for m in re.findall(r'Siman (\d+)', text)]


# ─── Output paths ─────────────────────────────────────────────────────────────
OUTPUT_FILE_V1 = Path(__file__).parent / "chunks_v1.json"
OUTPUT_FILE_V2 = Path(__file__).parent / "chunks_v2.json"
OUTPUT_FILE_V3 = Path(__file__).parent / "chunks_v3.json"
OUTPUT_FILE_V4 = Path(__file__).parent / "chunks_v4.json"

# ─── Chunking parameters ──────────────────────────────────────────────────────
CHUNK_SIZE   = 200   # words per chunk (~400 tokens, safely under 512)
OVERLAP      = 0     # no overlap (v1/v2)
OVERLAP_V3   = 50    # overlap for v3+
STEP         = CHUNK_SIZE - OVERLAP  # sliding-window step = 200


# ─── Text cleaning functions ──────────────────────────────────────────────────

def remove_html_tags(text: str) -> str:
    """Strip all HTML tags (e.g. <small>, </small>)."""
    return re.sub(r'<[^>]+>', '', text)


def remove_niqqud(text: str) -> str:
    """Remove Hebrew vowel diacritics — Unicode range U+0591–U+05C7."""
    return re.sub(r'[\u0591-\u05C7]', '', text)


def normalize_whitespace(text: str) -> str:
    """Collapse multiple spaces / newlines into a single space."""
    return re.sub(r'\s+', ' ', text).strip()


def remove_parenthetical_refs(text: str) -> str:
    """Remove short parenthetical source citations (up to 50 chars).
    For example: (הרא"ש הל' תפלין), (שם סי' ל"ח).
    Short threshold avoids accidentally deleting long halakhic content."""
    return re.sub(r'\([^)]{1,50}\)', '', text)


def normalize_geresh(text: str) -> str:
    """Normalize Hebrew geresh/gershayim to ASCII apostrophe/quote.
    ׳ (U+05F3) → '    ״ (U+05F4) → "
    Improves tokenization consistency for the E5 model."""
    text = text.replace('\u05F3', "'")   # HEBREW PUNCTUATION GERESH → '
    text = text.replace('\u05F4', '"')   # HEBREW PUNCTUATION GERSHAYIM → "
    return text


def clean_text(text: str, expand_abbrev: bool = False, v4_clean: bool = False) -> str:
    """Full cleaning pipeline: HTML → niqqud → [abbreviations] → whitespace.

    v4_clean=True adds: remove parenthetical refs + normalize geresh.
    """
    text = remove_html_tags(text)
    text = remove_niqqud(text)
    if expand_abbrev:
        from abbreviations import apply_abbreviations
        text = apply_abbreviations(text)
    if v4_clean:
        text = remove_parenthetical_refs(text)
        text = normalize_geresh(text)
    text = normalize_whitespace(text)
    return text


# ─── Chunking functions ───────────────────────────────────────────────────────

def split_into_chunks(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = OVERLAP) -> list[dict]:
    """
    Split cleaned text into fixed-size overlapping chunks (sliding window).

    Args:
        text:       cleaned text string
        chunk_size: words per chunk
        overlap:    words shared between consecutive chunks (0 = no overlap)

    Returns:
        List of dicts, each containing:
            chunk_id     — sequential index (0, 1, 2, ...)
            start_word   — index of first word in the original word list
            end_word     — index past the last word (exclusive)
            text         — chunk content
            word_count   — actual word count
            simanim      — list of siman numbers whose header appears in this chunk
            siman_parent — the siman this chunk belongs to (carried forward if header
                           appeared in a previous chunk)
    """
    step = chunk_size - overlap
    words = text.split()
    chunks = []
    current_siman = None   # tracks the active siman as we slide through

    for i in range(0, len(words), step):
        chunk_words = words[i: i + chunk_size]
        if not chunk_words:
            break
        chunk_text = " ".join(chunk_words)
        simanim    = extract_simanim(chunk_text)
        if simanim:
            # Update active siman to the last one opened in this chunk
            current_siman = simanim[-1]
        chunks.append({
            "chunk_id":     len(chunks),
            "start_word":   i,
            "end_word":     i + len(chunk_words),
            "text":         chunk_text,
            "word_count":   len(chunk_words),
            "simanim":      simanim,
            "siman_parent": current_siman,
        })

    return chunks


def split_into_siman_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = OVERLAP_V3,
) -> list[dict]:
    """
    Split text by siman boundaries first, then chunk each siman individually
    with overlap. Guarantees that no chunk crosses a siman boundary.

    Returns:
        Same dict structure as split_into_chunks.
    """
    words = text.split()
    step  = chunk_size - overlap

    # Locate every "Siman N" marker in the word list
    siman_starts: list[tuple[int, int]] = []   # (word_index, siman_number)
    for i, word in enumerate(words):
        if word == "Siman" and i + 1 < len(words):
            try:
                siman_num = int(words[i + 1])
                siman_starts.append((i, siman_num))
            except ValueError:
                pass

    if not siman_starts:
        # Fallback: no siman markers found — use plain sliding window
        return split_into_chunks(text, chunk_size, overlap)

    # Build segments: (start_word, end_word, siman_number)
    segments: list[tuple[int, int, int | None]] = []

    # Text before the first siman header (preamble / introduction)
    if siman_starts[0][0] > 0:
        segments.append((0, siman_starts[0][0], None))

    for idx, (start, siman_num) in enumerate(siman_starts):
        end = siman_starts[idx + 1][0] if idx + 1 < len(siman_starts) else len(words)
        segments.append((start, end, siman_num))

    chunks: list[dict] = []

    for seg_start, seg_end, siman_num in segments:
        seg_words = words[seg_start:seg_end]
        if not seg_words:
            continue

        # Slide within this siman segment
        for i in range(0, len(seg_words), step):
            chunk_words = seg_words[i: i + chunk_size]
            if not chunk_words:
                break
            chunk_text = " ".join(chunk_words)
            simanim    = extract_simanim(chunk_text)
            chunks.append({
                "chunk_id":     len(chunks),
                "start_word":   seg_start + i,
                "end_word":     seg_start + i + len(chunk_words),
                "text":         chunk_text,
                "word_count":   len(chunk_words),
                "simanim":      simanim,
                "siman_parent": siman_num,
            })

    return chunks


# ─── I/O helpers ──────────────────────────────────────────────────────────────

def save_chunks(chunks: list[dict], output_path: Path) -> None:
    """Serialize chunks to a JSON file."""
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=2)
    print(f"Saved: {output_path}  ({output_path.stat().st_size / 1024:.1f} KB)")


def load_chunks(json_path: Path) -> list[dict]:
    """Load a chunks JSON file (used by downstream pipeline steps)."""
    with open(json_path, "r", encoding="utf-8") as f:
        return json.load(f)


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 1 — load, clean, and chunk the corpus")
    parser.add_argument("--abbrev", action="store_true",
                        help="expand Hebrew abbreviations (saves to chunks_v2.json)")
    parser.add_argument("--siman-boundary", action="store_true",
                        help="split by siman boundaries with overlap=50 (saves to chunks_v3.json)")
    parser.add_argument("--v4-clean", action="store_true",
                        help="v4 cleaning: siman-boundary + remove citations + normalize geresh (saves to chunks_v4.json)")
    args = parser.parse_args()

    if args.v4_clean:
        output_file = OUTPUT_FILE_V4
        version_tag = "v4 (siman-boundary + remove-refs + normalize-geresh)"
    elif args.siman_boundary:
        output_file = OUTPUT_FILE_V3
        version_tag = "v3 (siman-boundary, overlap=50)"
    elif args.abbrev:
        output_file = OUTPUT_FILE_V2
        version_tag = "v2 (with abbreviation expansion)"
    else:
        output_file = OUTPUT_FILE_V1
        version_tag = "v1 (baseline)"

    # 1. Load raw text
    print(f"Loading: {TEXT_FILE.name}")
    raw_text = TEXT_FILE.read_text(encoding="utf-8")
    words_before = len(raw_text.split())
    print(f"  Words before cleaning: {words_before:,}")
    print(f"  Version: {version_tag}")

    # 2. Clean
    use_v4   = args.v4_clean
    use_abbr = args.abbrev and not args.siman_boundary and not args.v4_clean
    clean = clean_text(raw_text, expand_abbrev=use_abbr, v4_clean=use_v4)
    words_after = len(clean.split())
    print(f"  Words after cleaning:  {words_after:,}  "
          f"(removed {words_before - words_after:,} tokens)")

    # 3. Split into chunks
    if args.v4_clean or args.siman_boundary:
        chunks = split_into_siman_chunks(clean, chunk_size=CHUNK_SIZE, overlap=OVERLAP_V3)
        overlap_label = f"overlap={OVERLAP_V3}"
        mode_label    = "siman-boundary"
        # Drop the preamble chunk (siman_parent=None) — no retrieval value
        chunks = [c for c in chunks if c["siman_parent"] is not None]
        # Re-index chunk_id after filtering
        for i, c in enumerate(chunks):
            c["chunk_id"] = i
    else:
        chunks = split_into_chunks(clean, chunk_size=CHUNK_SIZE, overlap=OVERLAP)
        overlap_label = "no overlap"
        mode_label    = "sliding-window"

    word_counts = [c["word_count"] for c in chunks]
    print(f"\nChunking (size={CHUNK_SIZE}, {overlap_label}, {mode_label}):")
    print(f"  Total chunks:   {len(chunks):,}")
    print(f"  Min words:      {min(word_counts):,}")
    print(f"  Max words:      {max(word_counts):,}")
    print(f"  Avg words:      {sum(word_counts)/len(word_counts):.1f}")

    # 4. Show first chunk as a sanity check
    print("\n--- chunk #0 (first) ---")
    print(chunks[0]["text"][:300], "...")

    # 5. Save
    print()
    save_chunks(chunks, output_file)
    print("Step 1 complete.")


if __name__ == "__main__":
    main()
