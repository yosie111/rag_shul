"""
chunker/main.py — CLI entry point for the chunker
==================================================
Loads RAG JSON and writes:
  • chunks_v1.csv          — pipeline output (siman, seif, text)
  • chunks_DataFrame.csv   — debug view (every JSON field)

Reads the chunker configuration (chunk_fields, etc.) from
experiments/exp_config.yaml by default.

Usage:
    python -m chunker.main --input data/processed/shulchan_aruch_rag.json
    python -m chunker.main --input <json> --output chunks_v1.csv
    python -m chunker.main --input <json> --config path/to/exp_config.yaml
"""

import argparse
from pathlib import Path

import yaml

from .chunker import build_chunks_csv

HERE                = Path(__file__).parent.parent  # project root
DEFAULT_CONFIG_PATH = HERE / "experiments" / "exp_config.yaml"
DEFAULT_OUTPUT_PATH = HERE / "chunks_v1.csv"


def main():
    parser = argparse.ArgumentParser(
        description="Chunker — build chunks_v1.csv from RAG JSON"
    )
    parser.add_argument("--input",  required=True,
                        help="path to RAG JSON file")
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT_PATH),
                        help="output chunks CSV path "
                             "(chunks_DataFrame.csv is written alongside it)")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH),
                        help="YAML config (the `chunker:` block is consumed)")
    args = parser.parse_args()

    input_path  = Path(args.input)
    output_path = Path(args.output)
    config_path = Path(args.config)

    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    chunker_cfg = cfg.get("chunker", {})

    print(f"Config:  {config_path}")
    print(f"Input:   {input_path}")
    print(f"Fields:  {chunker_cfg.get('chunk_fields')}")

    csv_path = build_chunks_csv(
        json_path   = input_path,
        csv_path    = output_path,
        chunker_cfg = chunker_cfg,
    )

    print(f"\nSaved:   {csv_path}")
    print(f"Saved:   {csv_path.parent / 'chunks_DataFrame.csv'}")


if __name__ == "__main__":
    main()
