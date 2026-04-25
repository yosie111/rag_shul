"""
NpyRetriever — semantic retrieval on pre-built CSV + NPY artifacts
================================================================================
Layout this retriever expects:
    <project_root>/
      ├── data/processed/chunks_v1.csv           — flat CSV from chunker.build_csv
      └── data/processed/embeddings_v1.npy       — matrix from embed.build_embeddings
    (paths are passed in via kwargs — nothing is hardcoded)

What the retriever does:
    1. Loads the chunks CSV (siman, seif, text)
    2. Loads the .npy matrix  (must have the same number of rows as the CSV)
    3. For each query: calls embed.encode_query — same prefix + model as passages
    4. scores = embeddings @ query_vec, returns top_k

The retriever no longer owns any text-prep logic: chunking is done by the chunker,
passage encoding by embed.build_embeddings, query encoding by embed.encode_query.

Returns dicts conforming to BaseRetriever.retrieve:
    rank, chunk_id, score, text, siman_parent, siman, seif
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .base import BaseRetriever
from embedder.embed import encode_query

# Default model — overridden by the `model` kwarg from exp_config.yaml
DEFAULT_MODEL = "intfloat/multilingual-e5-large"


class NpyRetriever(BaseRetriever):

    @property
    def name(self) -> str:
        return "retrieval_npy"

    def __init__(
        self,
        chunks_csv:     str | Path | None = None,
        embeddings_npy: str | Path | None = None,
        model:          str = DEFAULT_MODEL,
        prefix_query:   str = "query: ",
        **_ignored,
    ):
        """
        All paths & params are passed explicitly (typically by exp_main via
        get_retriever(...)). Unknown kwargs (top_k, score_threshold, ...) are
        silently ignored — they're consumed by the evaluator, not here.
        """
        self._chunks_csv     = Path(chunks_csv)     if chunks_csv     else None
        self._embeddings_npy = Path(embeddings_npy) if embeddings_npy else None
        self._model_name     = model
        self._prefix_query   = prefix_query

        # Lazily loaded artifacts
        self._model: SentenceTransformer | None = None
        self._embeddings: np.ndarray | None     = None
        self._seifs: list[dict] | None          = None

    def _load(self) -> None:
        """Lazy load — happens once per instance."""
        if self._model is not None:
            return

        if self._chunks_csv is None or not self._chunks_csv.exists():
            raise FileNotFoundError(
                f"Chunks CSV not found: {self._chunks_csv}\n"
                f"Run the chunker (exp_main.py handles this automatically)."
            )
        if self._embeddings_npy is None or not self._embeddings_npy.exists():
            raise FileNotFoundError(
                f"Embeddings matrix not found: {self._embeddings_npy}\n"
                f"Run the embedder (exp_main.py handles this automatically)."
            )

        # 1. CSV — chunks already flat, no JSON flattening required
        df = pd.read_csv(self._chunks_csv)
        required = {"siman", "seif", "text"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {self._chunks_csv.name}: {missing}")
        self._seifs = df.to_dict("records")
        for i, s in enumerate(self._seifs):
            s["chunk_id"] = i

        # 2. Embeddings
        self._embeddings = np.load(str(self._embeddings_npy))

        # 3. Sanity check — lengths must match
        if len(self._seifs) != self._embeddings.shape[0]:
            raise RuntimeError(
                f"Mismatch: {len(self._seifs)} rows in {self._chunks_csv.name} vs. "
                f"{self._embeddings.shape[0]} rows in {self._embeddings_npy.name}.\n"
                f"The .npy was built from a different chunks version. "
                f"Re-run with --force-rebuild."
            )

        # 4. Model — heaviest, loaded last
        self._model = SentenceTransformer(self._model_name)

    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        self._load()

        # Encode query through the shared embed.encode_query — guarantees that
        # passage and query use the same model + normalization.
        query_vec = encode_query(
            query,
            model=self._model,
            prefix_query=self._prefix_query,
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
                "siman_parent": int(s["siman"]),   # BaseRetriever contract
                "siman":        int(s["siman"]),
                "seif":         int(s["seif"]),
            })
        return results
