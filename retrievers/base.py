"""
BaseRetriever — uniform interface for all retrieval methods.

Every new retriever must inherit from this class and implement retrieve().
The returned structure is always identical, so the evaluation pipeline
works with any retriever without modification.
"""

from abc import ABC, abstractmethod


class BaseRetriever(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique experiment name (e.g. 'semantic_e5', 'bm25')."""

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Retrieve the most relevant chunks for a given query.

        Args:
            query:  the question string (Hebrew)
            top_k:  number of chunks to return

        Returns:
            List of dicts, each containing:
                rank         — position in results (1 = most relevant)
                chunk_id     — unique chunk identifier
                score        — cosine similarity score (higher = better)
                text         — raw text of the chunk
                siman_parent — siman (chapter) number this chunk belongs to
        """
