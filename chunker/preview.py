"""
preview.py — Preview the chunker DataFrame
===========================================
Demonstrates how the chunker output looks as a DataFrame.
Prints the first 15 rows to the terminal.

Usage:
    python -m chunker.preview
"""

from pathlib import Path
from .chunker import load_schema, build_dataframe

SCHEMA_PATH = Path(__file__).parent.parent / "data" / "shulchan_aruch_rag.json"


def main():
    schema = load_schema(SCHEMA_PATH)
    df = build_dataframe(schema)

    print(f"Shape:   {df.shape[0]} rows × {df.shape[1]} columns")
    print()

    # Truncate text column for clean display
    preview = df.head(15).copy()
    preview["text"] = preview["text"].str[:80] + "..."

    print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
