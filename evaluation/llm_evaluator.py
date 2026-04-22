"""
LLMEvaluator — answer-level LLM evaluation (future).

For each question:
  1. Retrieves top_k_context chunks from the retriever
  2. Sends a prompt to the LLM (llm_model) with those chunks as context
  3. Receives an answer
  4. Compares against the "answer" column in the CSV (BLEU / ROUGE / F1)

Currently a stub — raises NotImplementedError until the implementation is ready.

YAML:
    evaluation:
      type: "llm_qa"
      llm_model: "gpt-4o"
      top_k_context: 3
      sleep_between_calls: 0.3
"""

import warnings
from datetime import datetime

from .base import BaseEvaluator


class LLMEvaluator(BaseEvaluator):

    @property
    def name(self) -> str:
        return "llm_qa"

    def __init__(self,
                 llm_model: str = "gpt-4o",
                 top_k_context: int = 3,
                 sleep_between_calls: float = 0.3,
                 **_unused):
        self.llm_model           = llm_model
        self.top_k_context       = int(top_k_context)
        self.sleep_between_calls = float(sleep_between_calls)

        # Warn on any unknown YAML keys — helps catch typos like `llm_modle`
        if _unused:
            warnings.warn(
                f"LLMEvaluator received unknown parameters: {list(_unused.keys())}. "
                f"Check for typos in exp_config.yaml (evaluation section).",
                stacklevel=2,
            )

    def evaluate(self, retriever, queries_df, **kwargs) -> dict:
        raise NotImplementedError(
            f"LLMEvaluator is not yet implemented. Only RetrievalEvaluator is available for now.\n"
            f"Future: will call {self.llm_model} with top_k_context={self.top_k_context} "
            f"chunks per question, compare against the 'answer' column in the CSV, "
            f"and return BLEU/ROUGE/F1."
        )

    def format_report(self, result: dict, **_meta) -> str:
        return (
            f"Run: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
            f"Evaluator: {self.name} (not yet implemented)\n"
        )
