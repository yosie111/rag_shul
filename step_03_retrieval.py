"""
Step 3 — Retrieval (ChromaDB)
==============================
Input:  a Hebrew query string
Output: top-k most relevant chunks from ChromaDB

What this script does:
  1. Load the ChromaDB collection and the embedding model
  2. Receive a question → encode it with the "query: " prefix
  3. Search for the most similar chunks (cosine similarity)
  4. Return top-k results with text and similarity score

Note: this script uses ChromaDB for retrieval.
For production use, prefer the numpy-based retriever in retrievers/
which avoids the ChromaDB v1.5.5 HNSW loading bug.
"""

import chromadb
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─── Constants ────────────────────────────────────────────────────────────────

CHROMA_DIR  = Path(__file__).parent / "chroma_db"
COLLECTION  = "shulchan_arukh"
EMBED_MODEL = "intfloat/multilingual-e5-small"
TOP_K       = 3   # default number of results to return


# ─── Initialization ───────────────────────────────────────────────────────────

def load_retriever():
    """
    Load the embedding model and connect to ChromaDB.
    Returns (model, collection) for reuse across multiple queries.
    """
    model         = SentenceTransformer(EMBED_MODEL)
    chroma_client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    collection    = chroma_client.get_collection(COLLECTION)
    return model, collection


# ─── Retrieval ────────────────────────────────────────────────────────────────

def retrieve(query: str, model: SentenceTransformer, collection, top_k: int = TOP_K) -> list[dict]:
    """
    Find the top_k most relevant chunks for the given query.

    Args:
        query:      question string (Hebrew)
        model:      loaded SentenceTransformer
        collection: ChromaDB collection object
        top_k:      number of results to return

    Returns:
        List of dicts, each containing:
            rank       — position (1 = most relevant)
            chunk_id   — unique chunk id
            score      — cosine similarity (0–1, higher = more relevant)
            text       — chunk text
    """
    # E5 requires the "query: " prefix for question vectors
    query_embedding = model.encode(
        "query: " + query,
        normalize_embeddings=True,
    ).tolist()

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for rank, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0],
    ), start=1):
        chunks.append({
            "rank":         rank,
            "chunk_id":     meta["chunk_id"],
            "score":        round(1 - dist, 4),   # ChromaDB returns distance; convert to similarity
            "text":         doc,
            "siman_parent": meta.get("siman_parent", 0),
        })

    return chunks


# ─── Interactive demo ─────────────────────────────────────────────────────────

def main():
    print("Loading model and ChromaDB...")
    model, collection = load_retriever()
    print(f"  Collection: {collection.count()} chunks\n")

    # Sample questions (Hebrew)
    test_queries = [
        "מה צריך האדם לעשות בבוקר",
        "כיצד נוטלים ידיים שחרית",
        "מה דין קריאת שמע",
    ]

    for query in test_queries:
        print(f"Query: {query}")
        print("-" * 50)
        results = retrieve(query, model, collection, top_k=TOP_K)
        for r in results:
            print(f"  [{r['rank']}] chunk #{r['chunk_id']}  score={r['score']}")
            print(f"       {r['text'][:150]}...")
        print()


if __name__ == "__main__":
    main()
