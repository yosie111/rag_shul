"""
chunker.py — Chunker for Shulchan Arukh RAG pipeline
=====================================================
Reads the RAG JSON and builds a flat DataFrame dispatched by mode:
  - seif          : one row per seif (default, backward compatible)
  - siman         : one row per siman (all seifim concatenated)
  - sliding_window: word-count windows across the full corpus

Output DataFrame columns:
    siman      (int)      — chapter number
    seif       (int|None) — sub-chapter number (None for siman/sliding_window)
    siman_seif (str)      — "סימן N, סעיף M" or "סימן N"
    text       (str)      — joined chunk text sent to the embedder
"""

import json
from pathlib import Path

import pandas as pd
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def _build_seif_chunks(schema: dict, chunk_fields: list[str]) -> list[dict]:
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]
            parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            rows.append({
                "siman":      siman_num,
                "seif":       seif_num,
                "siman_seif": f"סימן {siman_num}, סעיף {seif_num}",
                "text":       " ".join(parts),
            })
    return rows


def _build_siman_chunks(schema: dict, chunk_fields: list[str]) -> list[dict]:
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        parts = []
        for seif_data in siman_data["seifim"]:
            parts += [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
        rows.append({
            "siman":      siman_num,
            "seif":       None,
            "siman_seif": f"סימן {siman_num}",
            "text":       " ".join(parts),
        })
    return rows


def _build_sliding_window_chunks(schema: dict, chunk_fields: list[str]) -> list[dict]:
    cfg = load_config()["chunker"]
    chunk_size = cfg["chunk_size"]
    overlap = cfg["overlap"]
    step = chunk_size - overlap

    all_words: list[str] = []
    word_siman: list[int] = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        for seif_data in siman_data["seifim"]:
            parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            words = " ".join(parts).split()
            all_words.extend(words)
            word_siman.extend([siman_num] * len(words))

    rows = []
    for start in range(0, len(all_words), step):
        window = all_words[start:start + chunk_size]
        if not window:
            break
        siman_parent = word_siman[start]
        rows.append({
            "siman":      siman_parent,
            "seif":       None,
            "siman_seif": f"סימן {siman_parent}",
            "text":       " ".join(window),
        })
    return rows


_DISPATCH = {
    "seif":           _build_seif_chunks,
    "siman":          _build_siman_chunks,
    "sliding_window": _build_sliding_window_chunks,
}


def build_dataframe(schema: dict, chunk_fields: list[str] | None = None, mode: str | None = None) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame.

    Args:
        schema:       parsed JSON dict
        chunk_fields: seif fields to join into the text column (defaults to config)
        mode:         "seif" | "siman" | "sliding_window" (defaults to config)

    Returns:
        DataFrame with columns: siman, seif, siman_seif, text
        Sorted by siman, with a clean integer index.
    """
    cfg = load_config()["chunker"]
    if chunk_fields is None:
        chunk_fields = cfg["chunk_fields"]
    if mode is None:
        mode = cfg["mode"]

    if mode not in _DISPATCH:
        raise ValueError(f"Unknown chunker mode: {mode!r}. Choose from {list(_DISPATCH)}")

    rows = _DISPATCH[mode](schema, chunk_fields)
    df = pd.DataFrame(rows)
    return df.sort_values(["siman"]).reset_index(drop=True)
