"""
chunker.py — Seif-level chunker for Shulchan Arukh RAG pipeline
================================================================
Reads the RAG JSON and produces two CSVs in the run directory:

  1. chunks_DataFrame.csv  — debug/inspection view: every field that exists
                             in the source JSON becomes a column.
  2. chunks_v1.csv         — pipeline output, 3 columns only:
                             siman, seif, text
                             (text = chunk_fields joined by a single space)

Supported JSON variations
-------------------------
The same code handles multiple JSON structures (e.g. with or without
siman-level metadata like `hilchot_group` / `siman_sign`). Columns in
df_chunker reflect whatever fields actually appear in the file —
fields that are absent are simply not added.

Expected top-level structure:
    {
      "title":  "...",         (optional)
      "source": "...",         (optional)
      "simanim": [
        {
          "siman": <int>,
          "<siman-level field>": ...,    (optional, e.g. hilchot_group)
          ...
          "seifim": [
            {
              "seif": <int>,
              "<seif-level field>": ...,  (e.g. text, hagah, text_raw)
              ...
            }
          ]
        }
      ]
    }

Public API:
    load_schema(json_path)              → dict
    build_dataframe_chunker(schema)     → pd.DataFrame  (df_chunker)
    build_chunks_csv(json, csv, cfg)    → writes both CSVs, returns Path

Configuration (chunker_cfg, taken from exp_config.yaml `chunker:` block):
    chunk_size:    int   — accepted but currently unused (reserved)
    overlap:       int   — accepted but currently unused (reserved)
    mode:          str   — accepted but currently unused (reserved)
    chunk_fields:  list  — ordered list of field names to join into `text`
"""

import json
from pathlib import Path

import pandas as pd


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataframe_chunker(schema: dict) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame — one row per seif.

    Columns are derived from whatever fields exist in the JSON:
      • 'siman' and 'seif' first
      • Then siman-level fields (e.g. hilchot_group, siman_sign), duplicated
        across all seifim of the same siman
      • Then seif-level fields (e.g. text, hagah, text_raw)

    Fields not present in the JSON do not appear as columns. This keeps the
    function compatible with multiple JSON structures (basic / breadcrumb /
    future variants).
    """
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]

        # Siman-level fields = everything on the siman except 'siman' & 'seifim'
        siman_fields = {
            k: v for k, v in siman_data.items()
            if k not in ("siman", "seifim")
        }

        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]

            # Seif-level fields = everything on the seif except 'seif'
            seif_fields = {
                k: v for k, v in seif_data.items()
                if k != "seif"
            }

            row = {
                "siman": siman_num,
                "seif":  seif_num,
                **siman_fields,
                **seif_fields,
            }
            rows.append(row)

    df_chunker = pd.DataFrame(rows)
    return df_chunker.sort_values(["siman", "seif"]).reset_index(drop=True)


def _join_chunk_fields(row: pd.Series, chunk_fields: list[str]) -> str:
    """
    Join the values of `chunk_fields` from a row using a single space.
    Missing / NaN / empty values are skipped silently — keeps the output
    clean when a field (e.g. `hagah`) is null on some seifim.
    """
    parts: list[str] = []
    for field in chunk_fields:
        if field not in row.index:
            continue
        value = row[field]
        if value is None:
            continue
        if isinstance(value, float) and pd.isna(value):
            continue
        if value == "":
            continue
        parts.append(str(value))
    return " ".join(parts)


def build_chunks_csv(
    json_path:   str | Path,
    csv_path:    str | Path,
    chunker_cfg: dict,
) -> Path:
    """
    Pipeline entry point: JSON → chunks_v1.csv (+ chunks_DataFrame.csv).

    Args:
        json_path:   path to the RAG JSON.
        csv_path:    path to the final chunks CSV (typically chunks_v1.csv).
                     chunks_DataFrame.csv is written alongside it (same dir).
        chunker_cfg: the `chunker:` block from exp_config.yaml. Currently only
                     `chunk_fields` is consumed; `chunk_size`, `overlap`, and
                     `mode` are accepted but reserved for future use.

    Returns:
        Path to chunks_v1.csv.
    """
    chunk_fields = chunker_cfg.get("chunk_fields") or []

    schema = load_schema(json_path)
    df_chunker = build_dataframe_chunker(schema)

    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # 1. Debug view — full DataFrame with every JSON field as a column.
    debug_path = csv_path.parent / "chunks_DataFrame.csv"
    df_chunker.to_csv(debug_path, index=False, encoding="utf-8")

    # 2. Pipeline output — only siman, seif, text.
    df_final = pd.DataFrame({
        "siman": df_chunker["siman"],
        "seif":  df_chunker["seif"],
        "text":  df_chunker.apply(
            lambda r: _join_chunk_fields(r, chunk_fields), axis=1
        ),
    })
    df_final.to_csv(csv_path, index=False, encoding="utf-8")

    return csv_path
