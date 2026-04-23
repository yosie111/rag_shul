"""
BaseEvaluator — uniform interface for retrieval/answer evaluation.

Every evaluator must inherit from this and implement evaluate().
Output is uniform: dict with metrics + report_text + saving to file.
"""

from abc import ABC, abstractmethod
from pathlib import Path


class BaseEvaluator(ABC):

    @property
    @abstractmethod
    def name(self) -> str:
        """Unique name (e.g. 'retrieval', 'llm_qa')."""

    @abstractmethod
    def evaluate(
        self,
        retriever,
        queries_df,
        **kwargs,
    ) -> dict:
        """
        Runs the evaluation on the retriever against queries_df.

        Args:
            retriever:   object implementing BaseRetriever
            queries_df:  DataFrame with 'question', 'siman', 'seif'
            **kwargs:    additional parameters from YAML (evaluation section)

        Returns:
            dict with at least:
                "metrics":      dict of the main metrics
                "n_questions":  number of questions processed
                "elapsed_sec":  elapsed time in seconds
                "extra":        dict of additional info (for saving to JSON)
        """

    @abstractmethod
    def format_report(self, result: dict, **meta) -> str:
        """Formats a textual report for printing/saving."""

    def save(self, result: dict, report_text: str,
             output_dir: Path, filename_stem: str) -> dict:
        """Saves two files: .txt (the report) and .json (the metrics)."""
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        txt_path  = output_dir / f"{filename_stem}.txt"
        json_path = output_dir / f"{filename_stem}.json"

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(report_text)
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        return {"txt": txt_path, "json": json_path}
