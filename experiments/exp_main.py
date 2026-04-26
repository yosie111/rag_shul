"""
exp_main.py — Pipeline Orchestrator
====================================
Pure orchestrator. Contains no logic for chunking / embedding / retrieval / metrics —
just calls the dedicated modules in order, skipping stages whose output already exists.

Responsibility split:
  • chunker.chunker.build_csv(...)        : JSON            → chunks CSV
  • embedder.embed.build_embeddings(...)  : chunks CSV      → embeddings NPY
  • retrievers.get_retriever(...)         : CSV + NPY + query → results
  • evaluation.get_evaluator(...)         : retriever + questions → metrics

Skip logic (handled here, in exp_main):
  • If embeddings_npy exists  → skip chunker + embed
  • If only chunks_csv exists → skip the chunker stage only
  • Otherwise                 → run the full pipeline
  • --force-rebuild           → delete existing artifacts and rebuild

Usage:
    python exp_main.py
    python exp_main.py --mode mini
    python exp_main.py --mode full --retriever retrieval_npy
    python exp_main.py --eval-type retrieval
    python exp_main.py --max-questions 50
    python exp_main.py --force-rebuild
    python exp_main.py --dump-first-query
"""

import argparse
import json
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
    Pick the paths block matching run_mode.
    Also supports the legacy flat structure (paths: { json_file, csv_path }) for
    backwards compatibility.
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
    """Load the questions CSV and normalize the column names."""
    df = pd.read_csv(csv_path)
    col_map = {}
    for col in df.columns:
        lc = col.strip().lower()
        # Note: the Hebrew strings below are CSV header values to match in the
        # user's input file — they are data, not comments, and must stay as-is.
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


# ─── Inspection helpers ────────────────────────────────────────────────────────

def dump_first_query(
    retriever,
    queries_df: pd.DataFrame,
    output_dir: Path,
    *,
    retriever_name: str = "",
    run_mode: str = "",
    top_k: int = 10,
    ts_filename: str | None = None,
) -> Path:
    """
    Run the retriever on the first question in queries_df and write the
    **raw output** of retriever.retrieve(...) to a JSON file — exactly as
    returned, without formatting, ground-truth comparison, or derived fields.

    File structure:
        {
          "query":   "<text of the first question>",
          "top_k":   <value passed to retrieve>,
          "results": <list[dict] — whatever retriever.retrieve() returned>
        }

    Filename: exp_first_query_<run_mode>_<retriever>_<timestamp>.json
    """
    if len(queries_df) == 0:
        raise ValueError("queries_df is empty — nothing to dump.")

    question = str(queries_df.iloc[0]["question"])
    results  = retriever.retrieve(question, top_k=top_k)

    if ts_filename is None:
        ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S")

    name_safe = (retriever_name or "retriever").replace("/", "_")
    parts = ["exp_first_query"]
    if run_mode:
        parts.append(run_mode)
    parts.extend([name_safe, ts_filename])
    out_path = Path(output_dir) / f"{'_'.join(parts)}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "query":   question,
        "top_k":   top_k,
        "results": results,   # exactly what retriever.retrieve() returned
    }

    # default=str: insurance against non-JSON-native types (e.g. numpy.float32)
    # leaking out of a future retriever; converts them to string instead of crashing.
    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return out_path


# ─── Pipeline stages ───────────────────────────────────────────────────────────

def _ensure_stage(
    *,
    label:          str,
    output:         Path,
    input_path:     Path,
    input_label:    str,
    builder,                 # callable
    builder_kwargs: dict,
) -> None:
    """
    Generic template for a pipeline stage:
      • If the output already exists  → SKIP and return
      • If the input is missing       → raise FileNotFoundError with context
      • Otherwise                     → run the builder and print BUILD/DONE
    The builder import stays in the public functions (lazy) so we don't load
    modules that won't run in this stage.
    """
    if output.exists():
        print(f"{label}  SKIP   — {output.name} already exists")
        return

    if not input_path.exists():
        raise FileNotFoundError(
            f"{input_label} not found: {input_path}\n"
            f"Cannot proceed without it."
        )

    print(f"{label}  BUILD  — {input_path.name}  →  {output.name}")
    builder(**builder_kwargs)
    print(f"{label}  DONE   — wrote {output.name}")


def ensure_chunks_csv(json_file: Path, chunks_csv: Path, chunker_cfg: dict) -> None:
    """
    Stage 1: ensure the chunks file exists. If not — call chunker.build_chunks_csv.
    """
    # lazy import — only when we actually need to run the chunker
    from chunker.chunker import build_chunks_csv

    _ensure_stage(
        label          = "[1/3 chunker]",
        output         = chunks_csv,
        input_path     = json_file,
        input_label    = "Source JSON",
        builder        = build_chunks_csv,
        builder_kwargs = {
            "json_path":   json_file,
            "csv_path":    chunks_csv,
            "chunker_cfg": chunker_cfg,
        },
    )


def ensure_embeddings_npy(chunks_csv: Path, embeddings_npy: Path, embed_cfg: dict) -> None:
    """
    Stage 2: ensure the embeddings file (.npy) exists. If not — call
    embed.build_embeddings.

    Prerequisite: chunks_csv must exist (handled by ensure_chunks_csv before
    this stage).
    """
    # lazy import — only when we actually need to run the embedder
    from embedder.embed import build_embeddings

    _ensure_stage(
        label          = "[2/3 embed]  ",   # padding to align with '[1/3 chunker]'
        output         = embeddings_npy,
        input_path     = chunks_csv,
        input_label    = "Chunks CSV",
        builder        = build_embeddings,
        builder_kwargs = {
            "csv":            chunks_csv,
            "npy":            embeddings_npy,
            "model":          embed_cfg["model"],
            "batch_size":     embed_cfg.get("batch_size", 32),
            "prefix_passage": embed_cfg.get("prefix_passage", "passage: "),
        },
    )


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
    parser.add_argument("--dump-first-query", action="store_true",
                        help="Dump the raw retriever output for the first question to a JSON file")
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

    # ── 4b. Optional: dump raw retriever output for the first question ────────
    if args.dump_first_query:
        dump_top_k = int(cfg["retrieval"].get("top_k_retrieve", 10))
        dumped = dump_first_query(
            retriever,
            queries_df,
            output_dir     = HERE,
            retriever_name = args.retriever,
            run_mode       = run_mode,
            top_k          = dump_top_k,
        )
        print(f"               first-query dump -> {dumped.name}")

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
    # Experiment-level metadata — owned by the orchestrator, not the evaluator.
    # JSON-serializable normalization of the metrics is now handled inside
    # RetrievalEvaluator.
    result["timestamp"]       = ts_readable
    result["retriever"]       = args.retriever
    result["config"]          = str(CONFIG_PATH)
    result["run_mode"]        = run_mode
    result["total_available"] = total_available

    name_safe = args.retriever.replace("/", "_")
    stem = f"exp_results_{eval_type}_{run_mode}_{name_safe}_{ts_filename}"
    saved = evaluator.save(result, report_text, output_dir=HERE, filename_stem=stem)
    print(f"\nReport saved -> {saved['txt'].name}")
    print(f"JSON saved  -> {saved['json'].name}")


if __name__ == "__main__":
    main()
