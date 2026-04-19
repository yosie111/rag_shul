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

    if not chunks_path.exists():
        print(f"Chunk file {chunks_path} not found. Creating new chunks...")
        chunks = chunker(data, **chunk_params)
    else:
        print(f"Chunk file {chunks_path} found. Loading chunks...")
        with open(chunks_path, encoding="utf-8") as f:
            chunks = json.load(f)

    if not embeddings_exist := chunks_path.with_suffix(".embeddings.json").exists():
        print(f"Embeddings file not found. Computing embeddings for {len(chunks)} chunks...")
        embeddings = load_embeddings(chunks_path, **embed_params)
    else:
        print(f"Embeddings file found. Loading embeddings...")
        embeddings = load_embeddings(chunks_path, **embed_params)

    # Load the embedding model (if needed for retrieval)
    embedding_model = load_model(**embed_params)

    # load qa dataset:
    # TODO: add dataset loading function to utils.py and load from there

    for  query in queries:
        print(f"\nQuery: {query}")

        results = retrieve(query, chunks, embeddings, model, top_k=args.topk, **retrieval_params)
        for i, (chunk, score) in enumerate(results):
            print(f"Result {i+1}: (score: {score:.4f})\n{chunk}\n")

        eval_score = retrieve_evaluate(query, results, **eval_params)

        print(f"Evaluation score: {eval_score:.4f}")

    # Summarize results, save to file, etc.
    # TODO: add result saving function to utils.py and save results there

if __name__ == "__main__":
    main()
