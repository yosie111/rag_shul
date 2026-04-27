"""
main_template2.py — RAG app template (YAML config)
====================================================
Loads all pipeline settings from config/config.yaml and runs an
interactive query loop over the Shulchan Arukh corpus.

Usage:
    python main_template2.py
    python main_template2.py --chunks chunks_v4.json
"""

import argparse
import json
from pathlib import Path

import yaml

from chunker import chunker
from embedder import load_model, load_embeddings
from retriever import retrieve

# ─── Load config ──────────────────────────────────────────────────────────────

HERE        = Path(__file__).parent
CONFIG_PATH = HERE / "config" / "config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

# Paths
TEXT_FILE = (HERE / cfg["paths"]["text_file"]).resolve()
XLSX_PATH = (HERE / cfg["paths"]["xlsx_path"]).resolve()

# Param dicts (passed as **kwargs to pipeline functions)
chunk_params     = cfg["chunker"]
embed_params     = cfg["embeddings"]
retrieval_params = cfg["retrieval"]
eval_params      = cfg["evaluation"]

# ─── Entry point ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG query loop — YAML config")
    parser.add_argument("--chunks", default=str(HERE / "chunks_v1.json"),
                        help="path to chunks JSON file")
    parser.add_argument("--topk", type=int, default=retrieval_params["top_k"],
                        help=f"results to return (default: {retrieval_params['top_k']})")
    args = parser.parse_args()

    chunks_path = Path(args.chunks)

    print(f"Config:    {CONFIG_PATH}")
    print(f"Text file: {TEXT_FILE}")
    print(f"Chunks:    {chunks_path}")
    print(f"Model:     {embed_params['model']}")
    print(f"Chunker:   {chunk_params}\n")

    with open(TEXT_FILE, encoding="utf-8") as f:
        data = f.read()

    chunks     = chunker(data, **chunk_params)
    embeddings = load_embeddings(chunks_path, **embed_params)
    model      = load_model(**embed_params)

    print(f"\nLoaded {len(chunks)} chunks. Ready.\n")

    while True:
        try:
            query = input("שאלה (Ctrl+C to exit): ").strip()
        except KeyboardInterrupt:
            print("\nBye.")
            break
        if not query:
            continue

        results = retrieve(query, model, chunks, embeddings, **{**retrieval_params, "top_k": args.topk})
        for r in results:
            print(f"\n[{r['rank']}] chunk #{r['chunk_id']}  score={r['score']}  siman={r['siman_parent']}")
            print(r["text"][:300], "...")
        print()


if __name__ == "__main__":
    main()
