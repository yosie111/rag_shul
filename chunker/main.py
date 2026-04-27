"""
chunker/main.py — CLI entry point for the chunker
==================================================
Loads Schema 2 JSON, builds the chunk list, and saves it to JSON.
Input/output paths default to values in config/config.yaml (paths.schema_json / paths.chunks_json).

Usage:
    python -m chunker.main
    python -m chunker.main --input data/processed/shulchan_aruch_rag.json
    python -m chunker.main --input data/processed/shulchan_aruch_rag.json --output data/chunks.json
"""

import argparse
import json
from pathlib import Path

from .chunker import load_schema, build_dataframe, load_config

HERE = Path(__file__).parent.parent  # project root


def main():
    cfg       = load_config()
    run_mode  = cfg.get("run_mode", "full")
    cfg_paths = cfg.get("paths", {}).get(run_mode, cfg.get("paths", {}))
    default_input  = str(HERE / cfg_paths.get("schema_json", "data/processed/shulchan_aruch_rag.json"))
    default_output = str(HERE / cfg_paths.get("chunks_json", "data/chunks.json"))

    parser = argparse.ArgumentParser(description="Chunker — build chunk list from Schema 2 JSON")
    parser.add_argument("--input",  default=default_input,  help="path to Schema 2 JSON file")
    parser.add_argument("--output", default=default_output, help="output JSON path")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)

    print(f"Loading: {input_path}")
    schema = load_schema(input_path)

    df = build_dataframe(schema)
    records = [{"id": i, **row} for i, row in enumerate(df.to_dict(orient="records"))]

    print(f"Chunks:  {len(records)} chunks across {df['siman'].nunique()} simanim")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\nSaved:   {output_path}")


if __name__ == "__main__":
    main()
