"""
chunker/main.py — CLI entry point for the chunker
==================================================
Loads Schema 2 JSON, builds the seif DataFrame, and saves it to CSV.

Usage:
    python -m chunker.main --input schema2.json
    python -m chunker.main --input schema2.json --output chunks.csv
"""

import argparse
from pathlib import Path

from .chunker import load_schema, build_dataframe

HERE = Path(__file__).parent.parent  # project root


def main():
    parser = argparse.ArgumentParser(description="Chunker — build seif DataFrame from Schema 2 JSON")
    parser.add_argument("--input",  required=True, help="path to Schema 2 JSON file")
    parser.add_argument("--output", default=str(HERE / "chunks.csv"), help="output CSV path")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading: {input_path}")
    schema = load_schema(input_path)

    df = build_dataframe(schema)
    print(f"Chunks:  {len(df)} seifim across {df['siman'].nunique()} simanim")
    print(df.head(3).to_string(index=False))

    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    print(f"\nSaved:   {output_path}")


if __name__ == "__main__":
    main()
