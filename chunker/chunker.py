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
import yaml

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config_template.yaml"


def load_config() -> dict:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_schema(json_path: str | Path) -> dict:
    """Load the RAG JSON from disk."""
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def build_dataframe(schema: dict, chunk_fields: list[str] | None = None) -> pd.DataFrame:
    """
    Convert the RAG JSON dict into a flat DataFrame (one row per seif).

    Args:
        schema: parsed JSON dict
        chunk_fields: ordered list of seif fields to join into the text column.
                      Defaults to the chunk_fields list in config_template.yaml.

    Returns:
        DataFrame with columns: siman, seif, siman_seif, text
        Sorted by siman then seif, with a clean integer index.
    """
    if chunk_fields is None:
        chunk_fields = load_config()["chunker"]["chunk_fields"]

    rows = []
    for siman_data in schema["simanim"]:
        siman_num = siman_data["siman"]
        for seif_data in siman_data["seifim"]:
            seif_num = seif_data["seif"]
            parts = [seif_data.get(f) for f in chunk_fields if seif_data.get(f)]
            text = " ".join(parts)
            rows.append({
                "siman":      siman_num,
                "seif":       seif_num,
                "siman_seif": f"סימן {siman_num}, סעיף {seif_num}",
                "text":       text,
            })

    df = pd.DataFrame(rows)
    return df.sort_values(["siman", "seif"]).reset_index(drop=True)

