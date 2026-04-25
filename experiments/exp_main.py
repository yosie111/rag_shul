"""
exp_main.py — Pipeline Orchestrator
====================================
אורכסטרטור טהור. לא מכיל לוגיקה של chunking / embedding / retrieval / metrics —
רק קורא למודולים הייעודיים לפי הסדר, ומדלג על שלבים שכבר הופקו.

פיצול אחריות:
  • chunker.chunker.build_csv(...)        : JSON            → chunks CSV
  • embedder.embed.build_embeddings(...)  : chunks CSV      → embeddings NPY
  • retrievers.get_retriever(...)         : CSV + NPY + שאילתה → תוצאות
  • evaluation.get_evaluator(...)         : retriever + שאלות → מדדים

Skip logic (מבוצע כאן, ב-exp_main):
  • אם קיים embeddings_npy → מדלגים על chunker + embed
  • אם קיים chunks_csv בלבד → מדלגים רק על chunker
  • אחרת → מריצים את הפייפליין המלא
  • --force-rebuild → מוחק ומריץ מחדש

Usage:
    python exp_main.py
    python exp_main.py --mode mini
    python exp_main.py --mode full --retriever retrieval_npy
    python exp_main.py --eval-type retrieval
    python exp_main.py --max-questions 50
    python exp_main.py --force-rebuild
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

eval_params = cfg["evaluation"]


# ─── Config helpers ────────────────────────────────────────────────────────────

def resolve_paths(cfg: dict, mode: str) -> dict:
    """
    בוחר את בלוק הנתיבים לפי run_mode.
    תומך גם במבנה ה-flat הישן (paths: { json_file, csv_path }) לתאימות לאחור.
    """
    paths_cfg = cfg.get("paths", {})
    mode = (mode or "full").lower()

    # Legacy flat layout — no nested modes
    if "csv_path" in paths_cfg and mode not in paths_cfg:
        return paths_cfg

    if mode not in paths_cfg:
        available = [k for k in paths_cfg.keys() if isinstance(paths_cfg[k], dict)]
        raise ValueError(
            f"run_mode='{mode}' is not defined under 'paths' in YAML. "
            f"Available modes: {available}"
        )
    return paths_cfg[mode]


def load_queries(csv_path: Path) -> pd.DataFrame:
    """טוען CSV שאלות ומנרמל שמות עמודות."""
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


# ─── Pipeline stages ───────────────────────────────────────────────────────────


def ensure_chunks_csv(json_file: Path, chunks_csv: Path, chunker_cfg: dict) -> None:
    """
    שלב 1: מוודא שקובץ ה-chunks קיים. אם לא — קורא ל-chunker.build_chunks_csv.

    האחריות של שלב זה מוגבלת ל:
      • בדיקת קיום קובץ
      • קריאה ל-chunker עם הפרמטרים מ-YAML
    """
    if chunks_csv.exists():
        print(f"[1/3 chunker]  SKIP   — {chunks_csv.name} already exists")
        return

    if not json_file.exists():
        raise FileNotFoundError(
            f"Source JSON not found: {json_file}\n"
            f"Cannot build chunks without source data."
        )

    print(f"[1/3 chunker]  BUILD  — {json_file.name}  →  {chunks_csv.name}")
    # lazy import — only when we actually need to run the chunker
    from chunker.chunker import build_chunks_csv

    build_chunks_csv(
        json_path   = json_file,
        csv_path    = chunks_csv,
        chunker_cfg = chunker_cfg,
    )
    print(f"[1/3 chunker]  DONE   — wrote {chunks_csv.name}")
    
    
def ensure_embeddings_npy(chunks_csv: Path, embeddings_npy: Path, embed_cfg: dict) -> None:
    """
    שלב 2: מוודא שקובץ ה-embeddings (.npy) קיים. אם לא — קורא ל-embed.build_embeddings.

    דרישה מוקדמת: chunks_csv חייב להתקיים (מטופל ב-ensure_chunks_csv לפני שלב זה).
    """
    if embeddings_npy.exists():
        print(f"[2/3 embed]    SKIP   — {embeddings_npy.name} already exists")
        return

    if not chunks_csv.exists():
        raise FileNotFoundError(
            f"Chunks CSV not found: {chunks_csv}\n"
            f"This should have been created in stage 1."
        )

    print(f"[2/3 embed]    BUILD  — {chunks_csv.name}  →  {embeddings_npy.name}")
    # lazy import — only when we actually need to run the embedder
    from embedder.embed import build_embeddings

    build_embeddings(
        csv            = chunks_csv,
        npy            = embeddings_npy,
        model          = embed_cfg["model"],
        batch_size     = embed_cfg.get("batch_size", 32),
        prefix_passage = embed_cfg.get("prefix_passage", "passage: "),
    )
    print(f"[2/3 embed]    DONE   — wrote {embeddings_npy.name}")


# ─── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG pipeline orchestrator")
    parser.add_argument("--mode", choices=["full", "mini"], default=None,
                        help="Overrides run_mode in YAML (full = real data, mini = smoke test)")
    parser.add_argument("--retriever", default="retrieval_npy",
                        help=f"retriever from the registry. Available: {', '.join(list_retrievers())}")
    parser.add_argument("--eval-type", default=None,
                        help=f"Overrides evaluation.type in YAML. Available: {', '.join(list_evaluators())}")
    parser.add_argument("--max-questions", type=str, default=None,
                        help="Limit number of questions (null = from YAML; 'all' = all)")
    parser.add_argument("--force-rebuild", action="store_true",
                        help="Delete existing chunks_csv + embeddings_npy and rebuild from source JSON")
    args = parser.parse_args()

    # ── 0. Resolve run mode & paths ───────────────────────────────────────────
    run_mode     = (args.mode or cfg.get("run_mode", "full")).lower()
    active_paths = resolve_paths(cfg, run_mode)

    JSON_FILE      = (HERE / active_paths["json_file"]).resolve()
    CHUNKS_CSV     = (HERE / active_paths["chunks_csv"]).resolve()
    EMBEDDINGS_NPY = (HERE / active_paths["embeddings_npy"]).resolve()
    EVAL_CSV       = (HERE / active_paths["csv_path"]).resolve()

    print("=" * 72)
    print(f"Config:      {CONFIG_PATH.name}")
    print(f"Run mode:    {run_mode.upper()}")
    print(f"Retriever:   {args.retriever}")
    print(f"JSON:        {JSON_FILE.name}")
    print(f"Chunks CSV:  {CHUNKS_CSV.name}")
    print(f"Embeddings:  {EMBEDDINGS_NPY.name}")
    print(f"Eval CSV:    {EVAL_CSV.name}")
    print("=" * 72)

    # ── Optional: force rebuild ───────────────────────────────────────────────
    if args.force_rebuild:
        for p in (CHUNKS_CSV, EMBEDDINGS_NPY):
            if p.exists():
                print(f"[force]  removing {p.name}")
                p.unlink()

    # ── 1. Chunks CSV — build if missing ──────────────────────────────────────
    ensure_chunks_csv(JSON_FILE, CHUNKS_CSV, cfg["chunker"])

    # ── 2. Embeddings NPY — build if missing ──────────────────────────────────
    ensure_embeddings_npy(CHUNKS_CSV, EMBEDDINGS_NPY, cfg["embeddings"])

    # ── 3. Retriever — loads chunks_csv + embeddings_npy ──────────────────────
    print(f"[3/3 eval]     LOAD   — retriever: {args.retriever}")
    retriever_kwargs = {
        "chunks_csv":     str(CHUNKS_CSV),
        "embeddings_npy": str(EMBEDDINGS_NPY),
        "model":          cfg["embeddings"]["model"],
        "prefix_query":   cfg["embeddings"].get("prefix_query", "query: "),
        **cfg["retrieval"],   # top_k, top_k_retrieve, score_threshold
    }
    retriever = get_retriever(args.retriever, **retriever_kwargs)

    # ── 4. Queries ────────────────────────────────────────────────────────────
    queries_df      = load_queries(EVAL_CSV)
    total_available = len(queries_df)
    print(f"               loaded {total_available} questions from {EVAL_CSV.name}")

    max_q = resolve_max_questions(args.max_questions, eval_params.get("max_questions"))
    if max_q is not None and max_q < total_available:
        queries_df = queries_df.head(max_q).reset_index(drop=True)
    n_running = len(queries_df)
    print(f"               running {n_running} / {total_available} questions")

    # ── 5. Evaluator — from YAML, unless overridden by --eval-type ────────────
    eval_type = args.eval_type or eval_params.get("type", "retrieval")
    print(f"               evaluator: {eval_type}")

    eval_kwargs = {k: v for k, v in eval_params.items()
                   if k not in ("type", "max_questions")}
    evaluator = get_evaluator(eval_type, **eval_kwargs)

    # ── 6. Run evaluation ─────────────────────────────────────────────────────
    print("-" * 72)
    result = evaluator.evaluate(retriever, queries_df)

    # ── 7. Report ─────────────────────────────────────────────────────────────
    run_ts      = datetime.now()
    ts_filename = run_ts.strftime("%Y%m%d_%H%M%S")
    ts_readable = run_ts.strftime("%Y-%m-%d %H:%M:%S")

    report_text = evaluator.format_report(
        result,
        retriever_name=args.retriever,
        ts_readable=ts_readable,
    )
    print("\n" + report_text)

    # ── 8. Save ───────────────────────────────────────────────────────────────
    result["timestamp"]       = ts_readable
    result["retriever"]       = args.retriever
    result["config"]          = str(CONFIG_PATH)
    result["run_mode"]        = run_mode
    result["total_available"] = total_available

    # JSON-serializable: retrieval metrics contain dicts with int keys — normalize them
    if "metrics" in result and "recall_at" in result["metrics"]:
        result["metrics"]["recall_at"]   = {str(k): v for k, v in result["metrics"]["recall_at"].items()}
        result["metrics"]["recall_rate"] = {str(k): v for k, v in result["metrics"]["recall_rate"].items()}

    name_safe = args.retriever.replace("/", "_")
    stem = f"exp_results_{eval_type}_{run_mode}_{name_safe}_{ts_filename}"
    saved = evaluator.save(result, report_text, output_dir=HERE, filename_stem=stem)
    print(f"\nReport saved -> {saved['txt'].name}")
    print(f"JSON saved  -> {saved['json'].name}")


if __name__ == "__main__":
    main()
