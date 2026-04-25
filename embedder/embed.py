"""
embed.py — Embedding layer for Shulchan Arukh RAG
===================================================
Two output paths:
  1. NPY matrix (for the retrieval_npy retriever) — build_embeddings(...)
  2. ChromaDB   (for the legacy Chroma-based retriever) — store_in_chroma(...) via main()

Both paths share the same passage-text formula (prefix + text),
so a query embedded via encode_query() is compatible with either index.

Input CSV columns (produced by chunker.build_csv):
    siman, seif, text

Public API:
    build_embeddings(csv, npy, model, ...)     → writes NPY
    encode_query(query, model, ...)            → np.ndarray (1D, normalized)
    load_chunks(csv)                           → list[dict]       (legacy helper)
    embed(model, texts)                        → list[list[float]] (legacy helper)
    store_in_chroma(chunks, vectors, ...)      → writes ChromaDB   (legacy helper)

CLI (legacy Chroma path):
    python embed.py --chunks path/to/chunks.csv
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
# chromadb is imported lazily inside store_in_chroma — it's only needed for the
# legacy Chroma CLI path, not for the NPY pipeline used by exp_main.py.

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "intfloat/multilingual-e5-large"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh_seifs"
BATCH_SIZE         = 32


# ─── Shared helpers ────────────────────────────────────────────────────────────

def load_chunks(csv_path: Path) -> list[dict]:
    """Load chunks CSV produced by the chunker. Requires siman, seif, text."""
    df = pd.read_csv(csv_path)
    required = {"siman", "seif", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns in CSV: {missing}. "
            f"Expected at least {sorted(required)}."
        )
    print(f"  {len(df)} seifs loaded from {csv_path.name}")
    return df.to_dict("records")


def _build_passage_texts(chunks: list[dict], prefix_passage: str) -> list[str]:
    """
    Build the passage text for each chunk:
        "<prefix_passage><text>"
    """
    return [
        prefix_passage + row["text"]
        for row in chunks
    ]


# ─── Public API — NPY path (used by exp_main orchestrator) ─────────────────────

def build_embeddings(
    csv:            str | Path,
    npy:            str | Path,
    model:          str = DEFAULT_MODEL,
    batch_size:     int = BATCH_SIZE,
    prefix_passage: str = "passage: ",
) -> Path:
    """
    Pipeline entry point: chunks CSV → embeddings NPY.

    Loads chunks, prepends prefix_passage, encodes, saves .npy.
    Creates parent directories if needed. Returns the NPY path.
    """
    print(f"  Loading chunks from {Path(csv).name}")
    chunks = load_chunks(Path(csv))

    print(f"  Building passage texts (prefix={prefix_passage!r})")
    texts = _build_passage_texts(chunks, prefix_passage)

    print(f"  Loading model: {model}")
    m = SentenceTransformer(model)
    print(f"  Vector dim: {m.get_sentence_embedding_dimension()}")

    print(f"  Encoding {len(texts)} passages (batch_size={batch_size})...")
    t0 = time.time()
    vectors = m.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s — shape {vectors.shape}")

    npy = Path(npy)
    npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(npy), vectors)
    print(f"  Saved embeddings to {npy}")
    return npy


def encode_query(
    query:        str,
    model:        str | SentenceTransformer = DEFAULT_MODEL,
    prefix_query: str = "query: ",
) -> np.ndarray:
    """
    Encode a single query into a normalized 1D vector.
    Accepts either a model name (loads it) or an already-loaded SentenceTransformer.
    """
    m = model if isinstance(model, SentenceTransformer) else SentenceTransformer(model)
    text = prefix_query + query
    return m.encode(
        text,
        normalize_embeddings=True,
        convert_to_numpy=True,
    )


# ─── Legacy Chroma path (kept for backwards compatibility) ─────────────────────

def embed(model: SentenceTransformer, texts: list[str]) -> list[list[float]]:
    """Encode all texts into normalized float32 vectors."""
    print(f"  Encoding {len(texts)} seifs (batch_size={BATCH_SIZE})...")
    t0 = time.time()
    vectors = model.encode(
        texts,
        batch_size=BATCH_SIZE,
        show_progress_bar=True,
        convert_to_numpy=True,
        normalize_embeddings=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")
    return vectors.tolist()


def store_in_chroma(
    chunks: list[dict],
    vectors: list[list[float]],
    chroma_dir: Path,
    collection_name: str,
) -> None:
    """Store embeddings + metadata in ChromaDB."""
    import chromadb  # lazy — only needed for this legacy path

    client = chromadb.PersistentClient(path=str(chroma_dir))

    existing = [c.name for c in client.list_collections()]
    if collection_name in existing:
        client.delete_collection(collection_name)
        print(f"  Deleted existing collection: {collection_name}")

    collection = client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    collection.add(
        ids=[f"siman_{row['siman']}_seif_{row['seif']}" for row in chunks],
        embeddings=vectors,
        documents=[row["text"] for row in chunks],
        metadatas=[
            {
                "siman":      int(row["siman"]),
                "seif":       int(row["seif"]),
                "siman_seif": f"{int(row['siman'])}:{int(row['seif'])}",  # built on-the-fly
            }
            for row in chunks
        ],
    )
    print(f"  Stored {collection.count()} seifs in collection '{collection_name}'")
    print(f"  ChromaDB path: {chroma_dir}")


def main():
    parser = argparse.ArgumentParser(description="Embed Shulchan Arukh seifs into ChromaDB")
    parser.add_argument("--chunks",     required=True,                   help="Path to chunks.csv (chunker output)")
    parser.add_argument("--model",      default=DEFAULT_MODEL,           help=f"Embedding model (default: {DEFAULT_MODEL})")
    parser.add_argument("--chroma-dir", default=str(DEFAULT_CHROMA_DIR), help="ChromaDB directory")
    parser.add_argument("--collection", default=DEFAULT_COLLECTION,      help="ChromaDB collection name")
    parser.add_argument("--prefix-passage", default="passage: ",         help="Prefix used for passage encoding")
    args = parser.parse_args()

    csv_path   = Path(args.chunks)
    chroma_dir = Path(args.chroma_dir)

    print(f"\n1. Loading chunks...")
    chunks = load_chunks(csv_path)

    print(f"\n2. Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"   Vector dim: {model.get_sentence_embedding_dimension()}")

    print(f"\n3. Building encoding texts...")
    texts = _build_passage_texts(chunks, args.prefix_passage)

    print(f"\n4. Embedding...")
    vectors = embed(model, texts)

    print(f"\n5. Storing in ChromaDB...")
    store_in_chroma(chunks, vectors, chroma_dir, args.collection)

    print(f"\nDone. To query:\n"
          f"  client = chromadb.PersistentClient('{chroma_dir}')\n"
          f"  col = client.get_collection('{args.collection}')\n"
          f"  col.query(query_embeddings=[...], n_results=10)")


if __name__ == "__main__":
    main()
