"""
experiments_exp_main_Version6.py — Orchestrator
================================================
Connects: retriever (from retrievers/) → evaluator (from evaluation/) → report.

All logic for metrics computation, report formatting, and saving has moved to evaluation/.
Version6 only:
  1. Loads exp_config.yaml
  2. Gets retriever from registry (by --retriever)
  3. Loads questions from CSV
  4. Gets evaluator from registry (by evaluation.type in YAML)
  5. Runs evaluate() and saves report

Usage:
    python experiments_exp_main_Version6.py
    python experiments_exp_main_Version6.py --retriever semantic_e5_rag_json
    python experiments_exp_main_Version6.py --eval-type retrieval
    python experiments_exp_main_Version6.py --max-questions 50
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from retrievers import get_retriever, list_retrievers
from evaluation import get_evaluator, list_evaluators

# ─── Load config ───────────────────────────────────────────────────────────────
HERE        = Path(__file__).parent
CONFIG_PATH = HERE / "exp_config.yaml"

with open(CONFIG_PATH, encoding="utf-8") as f:
    cfg = yaml.safe_load(f)

CSV_PATH    = (HERE / cfg["paths"]["csv_path"]).resolve()
eval_params = cfg["evaluation"]


# ─── Helpers ───────────────────────────────────────────────────────────────────

def load_queries(csv_path: Path) -> pd.DataFrame:
    """Loads a questions CSV and normalizes column names."""
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        if lc in ("שאלה", "question", "query"):
            col_map[col] = "question"
        elif lc in ("סימן", "siman"):
            col_map[col] = "siman"
        elif lc in ("סעיף", "seif"):
            col_map[col] = "seif"
    df = df.rename(columns=col_map)
    required = {"question", "siman", "seif"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in CSV: {missing}")
    return df


def resolve_max_questions(cli_value, yaml_value) -> int | None:
    """CLI overrides. null/none/all/empty/<=0 → None (all)."""
    raw = cli_value if cli_value is not None else yaml_value
    if raw is None:
        return None
    if isinstance(raw, str):
        s = raw.strip().lower()
        if s in ("", "null", "none", "all", "-1"):
            return None
        try:
            raw = int(s)
        except ValueError:
            return None
    try:
        n = int(raw)
    except (TypeError, ValueError):
        return None
    return n if n > 0 else None


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation orchestrator")
    parser.add_argument("--retriever", default="semantic_e5_rag_json",
                        help=f"retriever from the registry. Available: {', '.join(list_retrievers())}")
    parser.add_argument("--eval-type", default=None,
                        help=f"Overrides evaluation.type in YAML. Available: {', '.join(list_evaluators())}")
    parser.add_argument("--max-questions", type=str, default=None,
                        help="Limit number of questions (null = from YAML; 'all' = all)")
    args = parser.parse_args()

    print(f"Config:    {CONFIG_PATH.name}")
    print(f"Retriever: {args.retriever}")
    print(f"CSV:       {CSV_PATH.name}")

    # ── 1. Retriever ──────────────────────────────────────────────────────────
    retriever = get_retriever(args.retriever)

    # ── 2. Queries ────────────────────────────────────────────────────────────
    queries_df      = load_queries(CSV_PATH)
    total_available = len(queries_df)
    print(f"\nLoaded {total_available} questions from {CSV_PATH.name}.")

    max_q = resolve_max_questions(args.max_questions, eval_params.get("max_questions"))
    if max_q is not None and max_q < total_available:
        queries_df = queries_df.head(max_q).reset_index(drop=True)
    n_running = len(queries_df)
    print(f"Running {n_running} out of {total_available} questions.\n")

    # ── 3. Evaluator — from YAML, unless overridden by --eval-type ───────────────
    eval_type = args.eval_type or eval_params.get("type", "retrieval")
    print(f"Evaluator: {eval_type}")

    # All evaluation YAML parameters are passed to the evaluator (it picks what it needs)
    eval_kwargs = {k: v for k, v in eval_params.items()
                   if k not in ("type", "max_questions")}
    evaluator = get_evaluator(eval_type, **eval_kwargs)

    # ── 4. Run evaluation ─────────────────────────────────────────────────────
    result = evaluator.evaluate(retriever, queries_df)

    # ── 5. Report ─────────────────────────────────────────────────────────────
    run_ts      = datetime.now()
    ts_filename = run_ts.strftime("%Y%m%d_%H%M%S")
    ts_readable = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    report_text = evaluator.format_report(
        result,
        retriever_name=args.retriever,
        ts_readable=ts_readable,
    )
    print("\n" + report_text)

    # ── 6. Save ───────────────────────────────────────────────────────────────
    result["timestamp"]       = ts_readable
    result["retriever"]       = args.retriever
    result["config"]          = str(CONFIG_PATH)
    result["total_available"] = total_available

    # JSON-serializable: retrieval metrics contain dicts with int keys — normalize them
    if "metrics" in result and "recall_at" in result["metrics"]:
        result["metrics"]["recall_at"]   = {str(k): v for k, v in result["metrics"]["recall_at"].items()}
        result["metrics"]["recall_rate"] = {str(k): v for k, v in result["metrics"]["recall_rate"].items()}

    name_safe = args.retriever.replace("/", "_")
    stem = f"exp_results_{eval_type}_{name_safe}_{ts_filename}"
    saved = evaluator.save(result, report_text, output_dir=HERE, filename_stem=stem)
    print(f"\nReport saved -> {saved['txt'].name}")
    print(f"JSON saved  -> {saved['json'].name}")


if __name__ == "__main__":
    main()
