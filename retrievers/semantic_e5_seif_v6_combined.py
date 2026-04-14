"""
SemanticE5SeifV6CombinedRetriever — semantic retrieval at seif level
     with GPT modern summary + GPT expert questions  (Experiment 028)

Each seif's encoding_text is built as:
    context_prefix + original_text + modern_summary + gpt_questions

Rationale:
    - modern_summary: bridges vocabulary gap between classical Hebrew and user queries (boosts R@10)
    - gpt_questions:  surface specific questions the seif answers (boosts R@3_seif)
    - combined:       best of both worlds — R@3 crossed 80% for the first time

Index file: seifs_v6_combined_intfloat_multilingual_e5_large.npy  (4169 × 1024)
"""

import json
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever

# ─── Paths and model name ──────────────────────────────────────────────────────
SEIFS_FILE      = Path(__file__).parent.parent / "seifs_v6_combined.json"
EMBEDDINGS_FILE = Path(__file__).parent.parent / "seifs_v6_combined_intfloat_multilingual_e5_large.npy"
EMBED_MODEL     = "intfloat/multilingual-e5-large"


class SemanticE5SeifV6CombinedRetriever(BaseRetriever):

    @property
    def name(self) -> str:
        return "semantic_e5_seif_v6_combined"

    def __init__(self):
        # All heavy resources are loaded lazily on first retrieve() call
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None = None
        self._seifs: list[dict] | None = None

    def _load(self) -> None:
        """Load model, embeddings matrix, and seif data (once per process)."""
        if self._model is not None:
            return  # already loaded

        self._model = SentenceTransformer(EMBED_MODEL)

        # Shape: (4169, 1024) — pre-computed, L2-normalized passage embeddings
        self._embeddings = np.load(str(EMBEDDINGS_FILE))

        with open(SEIFS_FILE, encoding="utf-8") as f:
            self._seifs = json.load(f)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Find the top_k most relevant seifs for the given query.

        Steps:
          1. Encode the query with "query: " prefix (required by E5)
          2. Compute cosine similarity against all 4,169 seif vectors
             (fast matrix multiply: embeddings @ query_vec)
          3. Pick the top_k indices and return their data
        """
        self._load()

        # Encode query; normalize_embeddings=True ensures cosine similarity = dot product
        query_vec = self._model.encode(
            "query: " + query,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )

        # Dot product with pre-normalized corpus vectors = cosine similarity for all seifs
        scores = self._embeddings @ query_vec  # shape: (4169,)

        # np.argpartition is faster than full argsort when top_k << total
        top_indices = np.argpartition(scores, -top_k)[-top_k:]
        # Sort only the top_k candidates (small sort)
        top_indices = top_indices[np.argsort(scores[top_indices])[::-1]]

        results = []
        for rank, idx in enumerate(top_indices, start=1):
            s = self._seifs[idx]
            results.append({
                "rank":            rank,
                "chunk_id":        s["chunk_id"],
                "score":           round(float(scores[idx]), 4),
                "text":            s["text"],            # original halakhic text
                "siman_parent":    s["siman"],           # chapter number (for evaluation)
                "seif_start":      s["seif"],
                "seif_end":        s["seif"],
                "seifim_in_chunk": [s["seif"]],          # used by seif-level Recall metric
                "summary":         s.get("summary", ""),
                "context_prefix":  s.get("context_prefix", ""),
                "modern_summary":  s.get("modern_summary", ""),
                "questions":       s.get("questions", []),
            })

        return results
