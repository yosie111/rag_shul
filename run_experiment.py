"""
run_experiment.py — single entry point for running any experiment.

Usage:
    python run_experiment.py --retriever semantic_e5_seif_v6_combined --name exp_028_combined
    python run_experiment.py --retriever semantic_e5_seif_v6_combined --name exp_028 --no-gpt
    python run_experiment.py --retriever semantic_e5_seif_v6_combined --name test --test 50

What this script does:
  1. Load the named retriever from the registry
  2. Create experiments/{name}/ directory
  3. Save config.json with all parameters and a timestamp
  4. Run the full evaluation (step_05.run_evaluation)
  5. Save results.csv inside the experiment directory
  6. Update config.json with the final metrics
"""

import argparse
import json
import sys
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).parent))
from retrievers import get_retriever, list_retrievers
from step_05_evaluate_rag import run_evaluation, print_summary, XLSX_PATH

load_dotenv(Path(__file__).parent.parent / ".env")

EXPERIMENTS_DIR = Path(__file__).parent / "experiments"


def main():
    parser = argparse.ArgumentParser(description="Run a RAG experiment")
    parser.add_argument("--retriever", required=True,
                        help=f"retriever name. Available: {list_retrievers()}")
    parser.add_argument("--name",      required=True,
                        help="experiment name (e.g. exp_029_my_method)")
    parser.add_argument("--test",      type=int, default=None,
                        help="evaluate only the first N questions (quick check)")
    parser.add_argument("--no-gpt",    action="store_true",
                        help="retrieval metrics only, skip OpenAI calls")
    parser.add_argument("--topk",      type=int, default=3,
                        help="chunks to send to GPT for answer generation (default: 3)")
    parser.add_argument("--topk-retrieve", type=int, default=10,
                        help="chunks to retrieve for Recall@K computation (default: 10)")
    args = parser.parse_args()

    # ── Create experiment directory ──
    exp_dir = EXPERIMENTS_DIR / args.name
    if exp_dir.exists():
        print(f"[Warning] Directory already exists: {exp_dir}")
        ans = input("Overwrite? (y/n): ").strip().lower()
        if ans != "y":
            print("Cancelled.")
            return
    exp_dir.mkdir(parents=True, exist_ok=True)

    # ── Write initial config (metrics filled in at the end) ──
    config = {
        "name":           args.name,
        "retriever":      args.retriever,
        "top_k_retrieve": args.topk_retrieve,
        "top_k_ctx":      args.topk,
        "use_gpt":        not args.no_gpt,
        "test_n":         args.test,
        "timestamp":      datetime.now().isoformat(timespec="seconds"),
        "metrics":        None,   # filled in after evaluation
    }
    config_path = exp_dir / "config.json"
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    # ── Set up OpenAI client (if needed) ──
    use_gpt = not args.no_gpt
    client  = None
    if use_gpt:
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        client = OpenAI(api_key=api_key)
    else:
        print("--no-gpt mode: computing retrieval metrics only.\n")

    # ── Load retriever ──
    print(f"Loading retriever: {args.retriever}...")
    retriever = get_retriever(args.retriever)

    # ── Load questions ──
    df = pd.read_excel(XLSX_PATH)
    if args.test:
        df = df.head(args.test)
        print(f"Test mode: {args.test} questions\n")
    else:
        print(f"Processing {len(df)} questions...\n")

    # ── Run evaluation ──
    output_csv = exp_dir / "results.csv"
    results_df = run_evaluation(
        retriever,
        df,
        client=client,
        top_k_retrieve=args.topk_retrieve,
        top_k_ctx=args.topk,
        use_gpt=use_gpt,
        output_csv=output_csv,
    )

    # ── Save final metrics back to config.json ──
    config["metrics"] = {
        "n_questions":  len(results_df),
        "Recall@1":     round(results_df["Recall@1"].mean(),  4),
        "Recall@3":     round(results_df["Recall@3"].mean(),  4),
        "Recall@5":     round(results_df["Recall@5"].mean(),  4),
        "Recall@10":    round(results_df["Recall@10"].mean(), 4),
        "MRR":          round(results_df["MRR"].mean(),       4),
    }
    config_path.write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")

    print_summary(results_df, use_gpt)
    print(f"\nExperiment saved to: {exp_dir}")


if __name__ == "__main__":
    main()
