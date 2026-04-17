"""
chunker.py — Seif-level chunker for Shulchan Arukh RAG pipeline
================================================================
Reads the RAG JSON (produced by Member 1) and builds a flat DataFrame
where each row is one seif — the unit sent to the embedder.

Input JSON structure:
    {
      "title": "שולחן ערוך, אורח חיים",
      "source": "Torat Emet 363",
      "simanim": [
        {
          "siman": 1,
          "seifim": [
            {
              "seif": 1,
              "text": "יתגבר כארי...",
              "hagah": "ועל כל פנים...",   (Rema commentary, or null)
              "text_raw": "..."             (raw text with HTML, not used)
            }
          ]
        }
      ]
    }

Output DataFrame columns:
    siman      (int)  — chapter number
    seif       (int)  — sub-chapter number
    siman_seif (str)  — "סימן N, סעיף M"  (matches מקור column in eval CSV)
    text       (str)  — text + hagah combined, sent to the embedder
"""

import json
from pathlib import Path

import pandas as pd


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(schema: dict) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame (one row per seif).

    Args:
        schema: parsed JSON dict

    Returns:
        DataFrame with columns: siman, seif, siman_seif, text
        Sorted by siman then seif, with a clean integer index.
        The text column combines the main text and hagah (Rema commentary).
    """
    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]
            text = seif_data["text"]
            hagah = seif_data.get("hagah")
            if hagah:
                text = text + " " + hagah
            rows.append({
                "siman":      siman_num,
                "seif":       seif_num,
                "siman_seif": f"סימן {siman_num}, סעיף {seif_num}",
                "text":       text,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["siman", "seif"]).reset_index(drop=True)

