"""
Step 2 — Embeddings & Vector Index
====================================
Input:  any chunks JSON file (e.g. seifs_v6_combined.json)
Output: <chunks_stem>_<model_name>.npy   (NumPy float32 matrix)
        chroma_db/                        (ChromaDB — kept for compatibility)

What this script does:
  1. Load the chunks JSON file
  2. Load the E5 embedding model from HuggingFace (downloaded once, then cached)
  3. Encode every chunk's "encoding_text" (or "text" as fallback) with prefix "passage: "
  4. Save the embedding matrix as a .npy file (fast cache for future retrieval)
  5. Also insert vectors into ChromaDB (legacy — queries use .npy directly due to
     a ChromaDB v1.5.5 HNSW bug on collections > ~1000 items)

Usage:
  python step_02_embeddings.py \\
    --chunks seifs_v6_combined.json \\
    --model intfloat/multilingual-e5-large \\
    --collection combined_col \\
    --chroma-dir chroma_combined
  # → saves: seifs_v6_combined_intfloat_multilingual_e5_large.npy

Note on caching:
  If the .npy file already exists, embeddings are loaded from it instead of
  recomputing (~30 minutes saved for 4169 seifs on CPU).
"""

import argparse
import json
import sqlite3
import time
from pathlib import Path

import chromadb
import numpy as np
from sentence_transformers import SentenceTransformer

# ─── Defaults ─────────────────────────────────────────────────────────────────

DEFAULT_CHUNKS     = Path(__file__).parent / "chunks_v1.json"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh"
DEFAULT_MODEL      = "intfloat/multilingual-e5-small"
BATCH_SIZE         = 32   # chunks per encoding batch


# ─── Model loading ────────────────────────────────────────────────────────────

def load_model(model_name: str = DEFAULT_MODEL) -> SentenceTransformer:
    """
    Load the sentence embedding model.
    First run: downloads from HuggingFace (~2 GB for e5-large).
    Subsequent runs: loads from the local HuggingFace cache.
    """
    print(f"Loading model: {model_name}")
    model = SentenceTransformer(model_name)
    print(f"  Vector dimension: {model.get_sentence_embedding_dimension()}")
    return model


# ─── Embedding ────────────────────────────────────────────────────────────────

def embed_chunks(
    model: SentenceTransformer,
    chunks: list[dict],
    cache_path: Path | None = None,
) -> list[list[float]]:
    """
    Encode all chunks into embedding vectors.

    E5 models require the "passage: " prefix for corpus texts.
    Uses "encoding_text" field if present, otherwise falls back to "text".

    Cache behavior:
      - If cache_path exists: load from .npy and skip encoding (~30 min saved).
      - Otherwise: encode all chunks and save the matrix to .npy for next time.

    Returns a list of float32 vectors (one per chunk).
    """
    if cache_path and cache_path.exists():
        print(f"\nLoading embeddings from cache: {cache_path.name}")
        embeddings = np.load(str(cache_path))
        print(f"  Loaded {len(embeddings)} vectors (dim={embeddings.shape[1]})")
        return embeddings.tolist()

    # Build the list of strings to encode; prefer encoding_text over raw text
    texts = [
        "passage: " + (c["encoding_text"] if c.get("encoding_text") else c["text"])
        for c in chunks
    ]

    print(f"\nEncoding {len(texts)} chunks (batch_size={BATCH_SIZE})...")
    t0 = time.time()

    embeddings = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,   # L2-normalize so dot product = cosine similarity
    )

    elapsed = time.time() - t0
    print(f"  Finished in {elapsed:.1f} seconds")

    if cache_path:
        np.save(str(cache_path), embeddings)
        print(f"  Saved: {cache_path.name}  ({cache_path.stat().st_size / 1024 / 1024:.1f} MB)")

    return embeddings.tolist()


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Step 2 — Embeddings & Vector Index")
    parser.add_argument("--chunks",     default=str(DEFAULT_CHUNKS),
                        help=f"chunks JSON file (default: {DEFAULT_CHUNKS.name})")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,
                        help=f"ChromaDB collection name (default: {DEFAULT_COLLECTION})")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR),
                        help="ChromaDB directory (default: chroma_db)")
    parser.add_argument("--model",     default=DEFAULT_MODEL,
                        help=f"embedding model name (default: {DEFAULT_MODEL})")
    args = parser.parse_args()

    chunks_path     = Path(args.chunks)
    chroma_dir      = Path(args.chroma_dir)
    collection_name = args.collection

    # 1. Load chunks
    print(f"Loading: {chunks_path.name}  (collection={collection_name}, chroma={chroma_dir.name})")
    with open(chunks_path, encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"  {len(chunks)} chunks loaded")

    # 2. Load model
    model = load_model(args.model)

    # 3. Embed (loads from .npy cache if it exists)
    # Cache filename: e.g. seifs_v6_combined_intfloat_multilingual_e5_large.npy
    model_short = args.model.replace("/", "_").replace("-", "_")
    cache_path  = chunks_path.with_name(f"{chunks_path.stem}_{model_short}.npy")
    embeddings  = embed_chunks(model, chunks, cache_path=cache_path)

    # 4. Store in ChromaDB (for compatibility; queries use .npy directly)
    chroma_client = chromadb.PersistentClient(path=str(chroma_dir))
    existing = [c.name for c in chroma_client.list_collections()]
    if collection_name in existing:
        chroma_client.delete_collection(collection_name)
        print(f"\nDeleted existing collection: {collection_name}")

    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
    print(f"Saving {len(chunks)} chunks to ChromaDB...")
    collection.add(
        ids        = [str(c["chunk_id"]) for c in chunks],
        embeddings = embeddings,
        documents  = [c["text"] for c in chunks],
        metadatas  = [
            {
                "chunk_id":     c["chunk_id"],
                "start_word":   c.get("start_word", 0),
                "end_word":     c.get("end_word", 0),
                "word_count":   c["word_count"],
                "simanim":      str(c.get("simanim", [])),
                "siman_parent": c.get("siman_parent") or c.get("siman") or 0,
            }
            for c in chunks
        ],
    )

    # 5. Wait for ChromaDB background compaction to finish
    # ChromaDB 1.x runs HNSW compaction asynchronously; we poll the SQLite queue.
    db_path = chroma_dir / "chroma.sqlite3"
    print("\nWaiting for ChromaDB compaction...")
    for attempt in range(120):   # max ~4 minutes
        time.sleep(2)
        try:
            conn    = sqlite3.connect(str(db_path))
            q_count = conn.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0]
            conn.close()
            print(f"  queue: {q_count} items remaining", end="\r")
            if q_count == 0:
                print("\n  Compaction complete.")
                break
        except Exception:
            pass
    else:
        print("\n  Timeout — manually clearing queue...")

    # Remove any ghost rows left in the queue after HNSW compaction
    # (ChromaDB v1.5.5 bug: ghost rows prevent the collection from loading)
    try:
        conn   = sqlite3.connect(str(db_path))
        ghost  = conn.execute("SELECT COUNT(*) FROM embeddings_queue").fetchone()[0]
        if ghost > 0:
            conn.execute("DELETE FROM embeddings_queue")
            conn.commit()
            print(f"  Cleared {ghost} ghost rows from the queue.")
        conn.close()
    except Exception as e:
        print(f"  Warning: could not clear queue: {e}")

    # 6. Summary
    print(f"\nSummary:")
    print(f"  Chunks in collection: {collection.count()}")
    print(f"  ChromaDB path:        {chroma_dir}")
    print("\nStep 2 complete.")


if __name__ == "__main__":
    main()
