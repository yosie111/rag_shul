"""
Retriever Registry
==================
Maps retriever names (strings) to their classes.

To add a new retriever:
    1. Create a new file in retrievers/  (e.g. my_retriever.py)
    2. Add an import and a REGISTRY entry below
    3. Run:  echo y | python run_experiment.py --retriever my_retriever --name exp_029 --no-gpt
"""

from .base import BaseRetriever
from .semantic_e5_seif_v6_combined import SemanticE5SeifV6CombinedRetriever

# Map of name → class for every available retriever
REGISTRY: dict[str, type[BaseRetriever]] = {
    "semantic_e5_seif_v6_combined": SemanticE5SeifV6CombinedRetriever,
}


def get_retriever(name: str) -> BaseRetriever:
    """Return an instance of the named retriever. Raises ValueError if unknown."""
    if name not in REGISTRY:
        available = list(REGISTRY.keys())
        raise ValueError(f"Retriever '{name}' not found. Available: {available}")
    return REGISTRY[name]()


def list_retrievers() -> list[str]:
    """Return the names of all registered retrievers."""
    return list(REGISTRY.keys())
