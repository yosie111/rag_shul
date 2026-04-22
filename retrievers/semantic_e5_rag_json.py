"""
SemanticE5RagJsonRetriever — semantic retrieval on shulchan_aruch_rag.json
==========================================================================
Matches your project layout:
    <project_root>/
      ├── data/processed/shulchan_aruch_rag.json        — nested JSON (simanim>seifim)
      └── experiments/
          ├── chunks_v1.csv                             — flat CSV (chunker output)
          ├── chunks_v1.embeddings.npy                  — embeddings matrix
          └── experiments_exp_main_Version6.py

What the retriever does:
    1. Loads the nested JSON and flattens it into a list of seifs
       (in the order of chunker.build_dataframe: sort by siman, seif)
    2. Loads the .npy matrix (its length must match the flattened list)
    3. For each query, encodes it via E5, computes cosine similarity, returns top_k

Returns dicts conforming to BaseRetriever.retrieve:
    rank, chunk_id, score, text, siman_parent
Additionally — helper fields for Version6 reports:
    seif, siman_seif
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

# ─── Paths & model ─────────────────────────────────────────────────────────────
# Project root = two levels above this file (retrievers/ → project_root)
_ROOT = Path(__file__).resolve().parents[1]

RAG_JSON_FILE   = _ROOT / "data" / "processed" / "shulchan_aruch_rag.json"
EMBEDDINGS_FILE = _ROOT / "experiments" / "chunks_v1.embeddings.npy"
EMBED_MODEL     = "intfloat/multilingual-e5-large"


def _flatten_schema(schema: dict) -> list[dict]:
    """
    Flattens the nested JSON into a list of seifs.
    The order must match chunker.build_dataframe exactly (sort by siman, seif)
    so that indices line up with the .npy matrix.

    Every seif gets a running chunk_id (0, 1, 2, ...) as in the original retriever.
    """
    rows = []
    simanim = sorted(schema["simanim"], key=lambda s: s["siman"])
    for siman_obj in simanim:
        siman_num = siman_obj["siman"]
        seifim = sorted(siman_obj.get("seifim", []), key=lambda x: x["seif"])
        for sf in seifim:
            seif_num = sf["seif"]
            parts = [sf.get("text"), sf.get("hagah")]
            text  = " ".join(p for p in parts if p)
            rows.append({
                "siman":      int(siman_num),
                "seif":       int(seif_num),
                "siman_seif": f"סימן {siman_num}, סעיף {seif_num}",
                "text":       text,
            })
    # chunk_id increments according to the flattened order
    for i, r in enumerate(rows):
        r["chunk_id"] = i
    return rows


class SemanticE5RagJsonRetriever(BaseRetriever):

    @property
    def name(self) -> str:
        return "semantic_e5_rag_json"

    def __init__(self):
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None
        self._seifs: list[dict] | None = None

    def _load(self) -> None:
        """Lazy load — happens once per instance."""
        if self._model is not None:
            return

        if not RAG_JSON_FILE.exists():
            raise FileNotFoundError(f"Not found: {RAG_JSON_FILE}")
        if not EMBEDDINGS_FILE.exists():
            raise FileNotFoundError(
                f"Embeddings matrix not found: {EMBEDDINGS_FILE}\n"
                f"Build it first (a single run of Version5/Version6 that builds the npy)."
            )

        # 1. JSON
        with open(RAG_JSON_FILE, encoding="utf-8") as f:
            schema = json.load(f)
        self._seifs = _flatten_schema(schema)

        # 2. Embeddings
        self._embeddings = np.load(str(EMBEDDINGS_FILE))

        # 3. Sanity check — lengths must match
        if len(self._seifs) != self._embeddings.shape[0]:
            raise RuntimeError(
                f"Mismatch: {len(self._seifs)} seifs in JSON vs. "
                f"{self._embeddings.shape[0]} rows in .npy.\n"
                f"The .npy may have been built from a different chunks version. Rebuild it."
            )

        # 4. Model — heaviest, loaded last
        self._model = SentenceTransformer(EMBED_MODEL)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        self._load()

        query_vec = self._model.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        scores = self._embeddings @ query_vec

        k = min(top_k, len(scores))
        top_indices = np.argpartition(scores, -k)[-k:]
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            s = self._seifs[idx]
            results.append({
                "rank":         rank,
                "chunk_id":     s["chunk_id"],
                "score":        round(float(scores[idx]), 4),
                "text":         s["text"],
                "siman_parent": s["siman"],      # BaseRetriever contract
                # Helper fields for the Version6 report
                "siman":        s["siman"],
                "seif":         s["seif"],
                "siman_seif":   s["siman_seif"],
            })
        return results
