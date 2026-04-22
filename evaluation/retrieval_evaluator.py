"""
RetrievalEvaluator — retrieval-level evaluation only (Recall@K, MRR).

Computes:
  • Recall@K for a list of K values (at unique-siman granularity)
  • MRR (Mean Reciprocal Rank)
  • Run time

Called from Version6 in a loop; no changes to BaseRetriever contracts.
"""

import time
import warnings
from datetime import datetime

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        desc = kwargs.get("desc", "")
        if desc:
            print(desc)
        return iterable

from .base import BaseEvaluator


DEFAULT_K_VALUES   = [1, 3, 5, 10, 18, 30, 50]
DEFAULT_TARGET_K   = 50
DEFAULT_TARGET_REC = 0.85

# Known keys that may appear in the YAML evaluation block but are intended for
# other evaluators (e.g. LLMEvaluator). They are silently accepted.
_KNOWN_FOREIGN_KEYS = {"llm_model", "top_k_context", "sleep_between_calls"}


def _find_gt_rank_unique_siman(results: list[dict], gt_siman: int) -> int | None:
    """Rank at unique-siman granularity: counts only new simanim in the list."""
    seen: set[int] = set()
    unique_rank = 0
    for r in results:
        s = r["siman_parent"]
        if s in seen:
            continue
        seen.add(s)
        unique_rank += 1
        if s == gt_siman:
            return unique_rank
    return None


def _compute_recall_mrr(ranks: list, k_values: list[int]) -> dict:
    n = len(ranks)
    recall_at  = {k: 0 for k in k_values}
    reciprocal = 0.0
    for r in ranks:
        if r is None:
            continue
        reciprocal += 1.0 / r
        for k in k_values:
            if r <= k:
                recall_at[k] += 1
    recall_rate = {k: (recall_at[k] / n if n else 0.0) for k in k_values}
    mrr = reciprocal / n if n else 0.0
    return {
        "recall_at":   recall_at,
        "recall_rate": recall_rate,
        "mrr":         mrr,
        "n_total":     n,
    }


class RetrievalEvaluator(BaseEvaluator):

    @property
    def name(self) -> str:
        return "retrieval"

    def __init__(self,
                 k_values: list[int] | None = None,
                 target_k: int = DEFAULT_TARGET_K,
                 target_recall: float = DEFAULT_TARGET_REC,
                 retrieve_k: int | None = None,
                 **_unused):
        """
        Args from YAML (evaluation section):
            k_values:       list of K values to evaluate (or None = default)
            target_k:       which K to check against the target
            target_recall:  success threshold (0-1)
            retrieve_k:     how many results to retrieve per call (must be >= max(k_values))
            _unused:        additional YAML fields not relevant to this implementation
                            (e.g. llm_model/top_k_context — those are for LLMEvaluator)
        """
        self.k_values      = sorted(set(k_values)) if k_values else list(DEFAULT_K_VALUES)
        self.target_k      = int(target_k)
        self.target_recall = float(target_recall)
        max_k = max(self.k_values)
        self.retrieve_k    = max(retrieve_k or max_k, max_k)

        # Warn on truly unknown keys — ignore keys that belong to sibling evaluators
        unexpected = {k: v for k, v in _unused.items() if k not in _KNOWN_FOREIGN_KEYS}
        if unexpected:
            warnings.warn(
                f"RetrievalEvaluator received unknown parameters: {list(unexpected.keys())}. "
                f"Check for typos in exp_config.yaml (evaluation section).",
                stacklevel=2,
            )

    def evaluate(self, retriever, queries_df, **kwargs) -> dict:
        ranks: list = []
        t_start = time.perf_counter()

        iterator = tqdm(
            queries_df.itertuples(index=False),
            total=len(queries_df),
            desc="Dense",
            unit="q",
        )
        for row in iterator:
            query    = str(getattr(row, "question"))
            gt_siman = int(getattr(row, "siman"))
            results = retriever.retrieve(query, top_k=self.retrieve_k)
            ranks.append(_find_gt_rank_unique_siman(results, gt_siman))

        elapsed_sec = time.perf_counter() - t_start
        metrics = _compute_recall_mrr(ranks, self.k_values)

        target_rate   = metrics["recall_rate"].get(self.target_k, 0.0)
        target_passed = target_rate >= self.target_recall

        return {
            "evaluator":     self.name,
            "granularity":   "unique-siman",
            "metrics":       metrics,
            "n_questions":   metrics["n_total"],
            "elapsed_sec":   round(elapsed_sec, 3),
            "retrieve_k":    self.retrieve_k,
            "k_values":      self.k_values,
            "target_k":      self.target_k,
            "target_recall": self.target_recall,
            "target_passed": target_passed,
        }

    def format_report(self, result: dict, retriever_name: str = "",
                      ts_readable: str | None = None, **_meta) -> str:
        metrics  = result["metrics"]
        k_values = result["k_values"]
        n_total  = metrics["n_total"]
        elapsed  = result["elapsed_sec"]

        if ts_readable is None:
            ts_readable = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        lines = [
            f"Run: {ts_readable}",
            f"Evaluator: {self.name} (granularity: unique-siman)",
            f"Retriever: {retriever_name}",
            f"Questions: {n_total}",
            f"Elapsed:   {elapsed:.2f} sec",
            "",
            "Recall@K:",
        ]
        for k in k_values:
            rate = metrics["recall_rate"].get(k, 0.0)
            count = metrics["recall_at"].get(k, 0)
            lines.append(f"  K={k:<3} → {rate:.4f}  ({count}/{n_total})")
        lines.append("")
        lines.append(f"MRR: {metrics['mrr']:.4f}")
        lines.append("")
        status = "PASSED" if result["target_passed"] else "FAILED"
        target_rate = metrics["recall_rate"].get(self.target_k, 0.0)
        lines.append(
            f"Target: Recall@{self.target_k} >= {self.target_recall:.2f} "
            f"→ {target_rate:.4f} [{status}]"
        )
        return "\n".join(lines)
