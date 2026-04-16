"""
embed.py — Embedding layer for Shulchan Arukh RAG
===================================================
Reads chunks.csv (output of the chunker by Izar Dahan) and stores
sentence embeddings in ChromaDB.

Input CSV columns (produced by chunker):
    siman       (int)   — chapter number
    seif        (int)   — section number
    siman_seif  (str)   — "סימן N, סעיף M"
    text        (str)   — clean seif content

Usage:
    python embed.py --chunks path/to/chunks.csv
    python embed.py --chunks chunks.csv --model intfloat/multilingual-e5-large
    python embed.py --chunks chunks.csv --chroma-dir ./my_chroma
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import chromadb
from sentence_transformers import SentenceTransformer

# ─── Defaults ──────────────────────────────────────────────────────────────────
DEFAULT_MODEL      = "intfloat/multilingual-e5-large"
DEFAULT_CHROMA_DIR = Path(__file__).parent / "chroma_db"
DEFAULT_COLLECTION = "shulchan_arukh_seifs"
BATCH_SIZE         = 32


def load_chunks(csv_path: Path) -> list[dict]:
    """Load chunks CSV produced by the chunker."""
    df = pd.read_csv(csv_path)
    required = {"siman", "seif", "siman_seif", "text"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    print(f"  {len(df)} seifs loaded from {csv_path.name}")
    return df.to_dict("records")


def build_encoding_texts(chunks: list[dict]) -> list[str]:
    """
    Build the text string that gets embedded for each seif.
    E5 models require "passage: " prefix for corpus texts.
    Context prefix (siman_seif) is prepended so the model
    knows where in the text this seif comes from.
    """
    return [
        "passage: " + "שולחן ערוך אורח חיים, " + row["siman_seif"] + ": " + row["text"]
        for row in chunks
    ]


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
                "siman_seif": row["siman_seif"],
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
    args = parser.parse_args()

    csv_path   = Path(args.chunks)
    chroma_dir = Path(args.chroma_dir)

    print(f"\n1. Loading chunks...")
    chunks = load_chunks(csv_path)

    print(f"\n2. Loading model: {args.model}")
    model = SentenceTransformer(args.model)
    print(f"   Vector dim: {model.get_sentence_embedding_dimension()}")

    print(f"\n3. Building encoding texts...")
    texts = build_encoding_texts(chunks)

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
