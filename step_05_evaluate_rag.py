"""
Step 5 — RAG Evaluation
=======================
Runs all 596 test questions through the RAG system and computes:

  Retrieval metrics (by siman):
    • Recall@K    — is the correct chunk in the top-K results? (K=1,3,5,10)
    • MRR         — Mean Reciprocal Rank (average of 1/rank)

  Retrieval metrics (by siman + seif — finer grained):
    • Recall@K_seif  — same, but the correct seif must be in the chunk
    • MRR_seif

  Answer quality metrics (requires --no-gpt to be absent and OPENAI_API_KEY):
    • Exact Match — full match after normalization
    • F1          — unigram token overlap
    • BLEU        — adaptive (n adjusted to answer length)
    • ROUGE-1/2/L — n-gram overlap + LCS

Usage:
    python step_05_evaluate_rag.py                        # all 596 questions
    python step_05_evaluate_rag.py --test 10              # first 10 only (quick check)
    python step_05_evaluate_rag.py --no-gpt               # retrieval metrics only
    python step_05_evaluate_rag.py --retriever <name>     # choose retriever
"""

import argparse
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

sys.path.insert(0, str(Path(__file__).parent))
from retrievers import get_retriever, list_retrievers
from retrievers.base import BaseRetriever

load_dotenv(Path(__file__).parent.parent / ".env")

from config import XLSX_PATH   # path defined in config.py

# ─── Constants ────────────────────────────────────────────────────────────────
BASELINE_CSV     = Path(__file__).parent.parent / "results_chatgpt_prompts.csv"
OUTPUT_CSV       = Path(__file__).parent / "results_rag_eval.csv"
MODEL            = "gpt-4o"
TOP_K_RETRIEVE   = 10    # retrieve 10 for Recall@K / MRR computation
TOP_K_CONTEXT    = 3     # send only top-3 to GPT for answer generation
SLEEP            = 0.3   # seconds between OpenAI API calls (rate-limit safety)
SCORE_THRESHOLD  = 0.85  # fallback: score >= threshold -> treat as "relevant" when no ground truth

# System prompt for answer generation (intentionally in Hebrew — the corpus and answers are Hebrew)
RAG_SYSTEM_PROMPT = (
    "אתה פוסק הלכה. להלן קטעים רלוונטיים מהשולחן ערוך אורח חיים:\n\n"
    "{context}\n\n"
    "ענה על השאלה לפי הקטעים בלבד. "
    "תן תשובה קצרה ככל האפשר — עד 3 מילים, עדיף מילה אחת. "
    "ללא ניקוד, ללא פיסוק, ללא הסבר נוסף."
)


# ─── Text normalization ───────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """Strip niqqud and punctuation for comparison purposes."""
    text = str(text).strip()
    text = re.sub(r'[\u0591-\u05C7]', '', text)   # remove vowel diacritics
    text = re.sub(r'[.!?,;:\'"״׃]', '', text)      # remove punctuation
    return text.strip()


# ─── Answer quality metrics ───────────────────────────────────────────────────

def compute_bleu(reference: str, hypothesis: str) -> float:
    """Compute adaptive BLEU score (n-gram order adjusted to answer length)."""
    ref_tokens = normalize(reference).split()
    hyp_tokens = normalize(hypothesis).split()
    if not ref_tokens or not hyp_tokens:
        return 0.0
    n = min(len(ref_tokens), len(hyp_tokens), 4)
    weights = tuple([1.0 / n] * n + [0.0] * (4 - n))
    return sentence_bleu([ref_tokens], hyp_tokens,
                         weights=weights,
                         smoothing_function=SmoothingFunction().method1)


def _f1_tokens(ref_tokens: list, hyp_tokens: list) -> float:
    """Token-level F1: harmonic mean of unigram precision and recall."""
    if not ref_tokens or not hyp_tokens:
        return 0.0
    overlap = set(ref_tokens) & set(hyp_tokens)
    if not overlap:
        return 0.0
    p = len(overlap) / len(hyp_tokens)
    r = len(overlap) / len(ref_tokens)
    return 2 * p * r / (p + r)


def _lcs_length(a: list, b: list) -> int:
    """Compute the length of the Longest Common Subsequence between two token lists."""
    m, n = len(a), len(b)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            dp[i][j] = dp[i-1][j-1] + 1 if a[i-1] == b[j-1] else max(dp[i-1][j], dp[i][j-1])
    return dp[m][n]


def compute_rouge(reference: str, hypothesis: str) -> dict:
    """Compute ROUGE-1, ROUGE-2, and ROUGE-L scores."""
    ref = normalize(reference).split()
    hyp = normalize(hypothesis).split()

    r1 = _f1_tokens(ref, hyp)  # ROUGE-1: unigram overlap

    # ROUGE-2: bigram overlap
    if len(ref) < 2 or len(hyp) < 2:
        r2 = r1
    else:
        ref_bi = [f"{ref[i]} {ref[i+1]}" for i in range(len(ref) - 1)]
        hyp_bi = [f"{hyp[i]} {hyp[i+1]}" for i in range(len(hyp) - 1)]
        r2 = _f1_tokens(ref_bi, hyp_bi)

    # ROUGE-L: longest common subsequence F1
    lcs = _lcs_length(ref, hyp)
    if ref and hyp:
        p  = lcs / len(hyp)
        r  = lcs / len(ref)
        rl = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
    else:
        rl = 0.0

    return {"rouge1": round(r1, 4), "rouge2": round(r2, 4), "rougeL": round(rl, 4)}


# ─── Retrieval metrics ────────────────────────────────────────────────────────

# Gematria (Hebrew numeral) lookup table for siman parsing
_HEBREW_VALS = {
    'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,
    'י':10,'כ':20,'ל':30,'מ':40,'נ':50,'ס':60,'ע':70,'פ':80,'צ':90,
    'ק':100,'ר':200,'ש':300,'ת':400,
}

def _hebrew_to_int(s: str) -> int:
    """Convert a Hebrew gematria string to an integer. E.g. 'לב' -> 32."""
    # Special cases: 15 = טו, 16 = טז (avoid the blasphemous יה/יו)
    special = {'טו': 15, 'טז': 16}
    if s in special:
        return special[s]
    return sum(_HEBREW_VALS.get(c, 0) for c in s)


def extract_siman(source: str) -> str:
    """
    Parse a source citation string and return 'Siman N' (English format used in chunks).

    Handles five input formats:
      "(251, 2)"          -> "Siman 251"
      "סימן 128"          -> "Siman 128"
      "סימן א, סעיף א"    -> "Siman 1"
      'שמ"ה ,ב'           -> "Siman 345"  (gematria without 'סימן' keyword)
      'תרמ"ג'             -> "Siman 643"  (gematria with geresh)

    Bug fixed in exp_022: regex [א-ת]+ stopped at geresh '"' so 'תרמ"ג' -> 640 instead of 643.
    Fix: include geresh/gershayim characters in the match, then strip them before conversion.
    """
    src = str(source).strip()

    # Format: (N, M) — tuple of (siman, seif)
    m = re.match(r'^\((\d+),\s*\d+\)$', src)
    if m:
        return f"Siman {int(m.group(1))}"

    # Remove niqqud before pattern matching
    src_no_niq = re.sub(r'[\u0591-\u05C7]', '', src)

    # Format: "סימן X" — the word 'סימן' followed by a number or gematria
    # Include geresh characters in match to avoid splitting 'תרמ"ג' at the '"'
    match = re.search(r'סימן\s+((?:[א-ת"\'\u05F3\u05F4]+|\d+))', src_no_niq)
    if match:
        raw = re.sub(r'["\'\u05F3\u05F4]', '', match.group(1).strip())
        num = int(raw) if raw.isdigit() else _hebrew_to_int(raw)
        return f"Siman {num}" if num > 0 else ""

    # Format: gematria without 'סימן' keyword (e.g. 'שמ"ה ,ב' -> Siman 345)
    m2 = re.match(r'^([א-ת"\']+)\s*,', src_no_niq)
    if m2:
        letters = re.sub(r'["\'\u05F3\u05F4]', '', m2.group(1))
        num = _hebrew_to_int(letters)
        return f"Siman {num}" if num > 0 else ""

    return ""


# Lookup table for seif gematria (covers seifs 1–50)
_GEMATRIA_SEIF = {
    'א':1,'ב':2,'ג':3,'ד':4,'ה':5,'ו':6,'ז':7,'ח':8,'ט':9,
    'י':10,'יא':11,'יב':12,'יג':13,'יד':14,'טו':15,'טז':16,
    'יז':17,'יח':18,'יט':19,'כ':20,'כא':21,'כב':22,'כג':23,
    'כד':24,'כה':25,'כו':26,'כז':27,'כח':28,'כט':29,'ל':30,
    'לא':31,'לב':32,'לג':33,'לד':34,'לה':35,'לו':36,'לז':37,
    'לח':38,'לט':39,'מ':40,'מא':41,'מב':42,'מג':43,'מד':44,
    'מה':45,'מו':46,'מז':47,'מח':48,'מט':49,'נ':50,
}


def extract_seif(source: str) -> int:
    """
    Parse a source citation string and return the seif (paragraph) number.
    Returns 0 if no seif can be determined.

    Handles:
      "(242, 1)"                              -> 1
      "סימן 128, סעיף 1"                      -> 1
      "סימן א, סעיף ג"                        -> 3  (gematria)
      "שולחן ערוך, סימן מא, סעיף א"           -> 1
      'שמ"ה ,ב'  (gematria siman, letter seif) -> 2
    """
    src = str(source).strip()

    # Format: (N, M)
    m = re.match(r'^\((\d+),\s*(\d+)\)$', src)
    if m:
        return int(m.group(2))

    src_clean = re.sub(r'[\u0591-\u05C7\u05F3\u05F4]', '', src)

    # "סעיף X" — Arabic number or gematria
    m = re.search(r'סעיף\s+(\S+)', src_clean)
    if m:
        val = m.group(1).strip("',\".")
        try:
            return int(val)
        except ValueError:
            return _GEMATRIA_SEIF.get(val, 0)

    # Format: "שמ"ה ,ב" — gematria siman + comma + single letter seif
    m2 = re.search(r',\s*([א-ת]{1,3})\s*$', src_clean)
    if m2:
        return _GEMATRIA_SEIF.get(m2.group(1).strip(), 0)

    return 0


def is_chunk_relevant(chunk: dict, siman: str) -> bool:
    """
    Return True if this chunk belongs to the target siman.
    Primary check: siman_parent metadata field (fast, exact).
    Fallback:      text search (for old chunks without siman_parent).
    """
    if not siman:
        return False
    m = re.search(r'\d+', siman)
    if not m:
        return False
    target = int(m.group())
    if chunk.get("siman_parent"):
        return chunk["siman_parent"] == target
    # Old chunks (v1–v4) without siman_parent metadata
    return siman in chunk["text"]


def compute_recall_at_k(chunks: list[dict], siman: str, k: int) -> float:
    """Return 1.0 if a relevant chunk (by siman) is in the top-k results, else 0.0."""
    for chunk in chunks[:k]:
        if is_chunk_relevant(chunk, siman):
            return 1.0
    return 0.0


def compute_recall_at_k_score(chunks: list[dict], k: int,
                               threshold: float = SCORE_THRESHOLD) -> float:
    """
    Fallback Recall@K when no ground truth is available.
    A chunk is considered relevant if its similarity score >= threshold.
    """
    for chunk in chunks[:k]:
        if chunk["score"] >= threshold:
            return 1.0
    return 0.0


def compute_reciprocal_rank(chunks: list[dict], siman: str) -> float:
    """
    Return 1/rank for the first relevant chunk (by siman).
    Returns 0 if no relevant chunk is found within TOP_K_RETRIEVE results.
    """
    for rank, chunk in enumerate(chunks, start=1):
        if is_chunk_relevant(chunk, siman):
            return 1.0 / rank
    return 0.0


def is_chunk_relevant_seif(chunk: dict, siman: str, target_seif: int) -> bool:
    """
    Return True if this chunk belongs to the target siman AND contains the target seif.
    Falls back to siman-only matching when:
      - target_seif == 0 (seif not parseable from source)
      - chunk has no seifim_in_chunk field (pre-v5 chunks)
    """
    if not is_chunk_relevant(chunk, siman):
        return False
    if target_seif == 0:
        return True   # seif unknown — siman match is sufficient
    seifim = chunk.get("seifim_in_chunk")
    if not seifim:
        return True   # old chunks (v1–v4) without seif tracking
    return target_seif in seifim


def compute_recall_at_k_seif(chunks: list[dict], siman: str, target_seif: int, k: int) -> float:
    """Return 1.0 if a relevant chunk (by siman + seif) is in the top-k results."""
    for chunk in chunks[:k]:
        if is_chunk_relevant_seif(chunk, siman, target_seif):
            return 1.0
    return 0.0


def compute_reciprocal_rank_seif(chunks: list[dict], siman: str, target_seif: int) -> float:
    """Return 1/rank for the first chunk relevant by both siman and seif."""
    for rank, chunk in enumerate(chunks, start=1):
        if is_chunk_relevant_seif(chunk, siman, target_seif):
            return 1.0 / rank
    return 0.0


# ─── RAG answer generation ────────────────────────────────────────────────────

def build_context(chunks: list[dict]) -> str:
    """Format the top chunks into a numbered context string for the GPT prompt."""
    return "\n\n".join(f"[{i}] {c['text']}" for i, c in enumerate(chunks, 1))


def ask_with_rag(client: OpenAI, question: str, context: str) -> str:
    """Call GPT with the retrieved context and return its answer."""
    response = client.chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": RAG_SYSTEM_PROMPT.format(context=context)},
            {"role": "user",   "content": f"שאלה: {question}"},
        ],
        temperature=0,
        max_tokens=50,
    )
    return response.choices[0].message.content.strip()


# ─── Results summary ──────────────────────────────────────────────────────────

def print_summary(results_df: pd.DataFrame, use_gpt: bool) -> None:
    """Print a formatted summary of all evaluation metrics."""
    n = len(results_df)
    print("\n" + "=" * 60)
    print(f"RAG Evaluation Results — {n} questions")
    print("=" * 60)

    # Siman-level retrieval metrics
    print(f"\n[Retrieval by siman — {n} questions]")
    print(f"  Recall@1:  {results_df['Recall@1'].mean():.4f}")
    print(f"  Recall@3:  {results_df['Recall@3'].mean():.4f}")
    print(f"  Recall@5:  {results_df['Recall@5'].mean():.4f}")
    print(f"  Recall@10: {results_df['Recall@10'].mean():.4f}")
    print(f"  MRR:       {results_df['MRR'].mean():.4f}")

    # Seif-level retrieval metrics (finer grained)
    if "Recall@3_seif" in results_df.columns:
        print(f"\n[Retrieval by siman + seif — {n} questions]")
        print(f"  Recall@1_seif:  {results_df['Recall@1_seif'].mean():.4f}")
        print(f"  Recall@3_seif:  {results_df['Recall@3_seif'].mean():.4f}")
        print(f"  Recall@5_seif:  {results_df['Recall@5_seif'].mean():.4f}")
        print(f"  Recall@10_seif: {results_df['Recall@10_seif'].mean():.4f}")
        print(f"  MRR_seif:       {results_df['MRR_seif'].mean():.4f}")
        delta3 = results_df['Recall@3'].mean() - results_df['Recall@3_seif'].mean()
        print(f"  [Gap siman vs siman+seif R@3: {delta3:+.4f}]")

    if not use_gpt:
        print("=" * 60)
        return

    # Answer quality metrics
    exact = (results_df["RAG_answer"].apply(normalize) ==
             results_df["reference_answer"].apply(normalize)).sum()
    print("\n[Answer quality — GPT-4o + RAG]")
    print(f"  Exact Match: {exact}/{n} ({100*exact/n:.1f}%)")
    print(f"  F1:          {results_df['F1'].mean():.4f}")
    print(f"  BLEU:        {results_df['BLEU'].mean():.4f}")
    print(f"  ROUGE-1:     {results_df['ROUGE-1'].mean():.4f}")
    print(f"  ROUGE-2:     {results_df['ROUGE-2'].mean():.4f}")
    print(f"  ROUGE-L:     {results_df['ROUGE-L'].mean():.4f}")

    # Comparison vs. baseline (GPT without RAG)
    if BASELINE_CSV.exists():
        try:
            base = pd.read_csv(BASELINE_CSV, encoding="utf-8-sig")
            base.columns = base.columns.str.strip()
            base_best = base[base["prompt_variant"] == "קצר"]   # best baseline variant
            if not base_best.empty:
                ans_col    = "תשובת_GPT" if "תשובת_GPT" in base_best.columns else "תשובת_מודל"
                base_exact = (base_best[ans_col].apply(normalize) ==
                              base_best["תשובת_אמת"].apply(normalize)).sum()
                base_n = len(base_best)
                print("\n[Comparison vs. Baseline — GPT-4o without RAG]")
                def delta(rag_val, base_val):
                    d    = rag_val - base_val
                    sign = "+" if d >= 0 else ""
                    return f"{sign}{d:.4f}"
                r1_rag  = results_df["ROUGE-1"].mean()
                r1_base = base_best["ROUGE-1"].mean()
                em_rag  = exact / n
                em_base = base_exact / base_n
                print(f"  {'Metric':<14} {'Baseline':>10} {'RAG':>10} {'Delta':>10}")
                print(f"  {'-'*44}")
                print(f"  {'Exact Match':<14} {100*em_base:>9.1f}% {100*em_rag:>9.1f}% {delta(em_rag,em_base):>10}")
                print(f"  {'F1':<14} {base_best['F1'].mean():>10.4f} {results_df['F1'].mean():>10.4f} {delta(results_df['F1'].mean(), base_best['F1'].mean()):>10}")
                print(f"  {'BLEU':<14} {base_best['BLEU'].mean():>10.4f} {results_df['BLEU'].mean():>10.4f} {delta(results_df['BLEU'].mean(), base_best['BLEU'].mean()):>10}")
                print(f"  {'ROUGE-1':<14} {r1_base:>10.4f} {r1_rag:>10.4f} {delta(r1_rag, r1_base):>10}")
                print(f"  {'ROUGE-2':<14} {base_best['ROUGE-2'].mean():>10.4f} {results_df['ROUGE-2'].mean():>10.4f} {delta(results_df['ROUGE-2'].mean(), base_best['ROUGE-2'].mean()):>10}")
                print(f"  {'ROUGE-L':<14} {base_best['ROUGE-L'].mean():>10.4f} {results_df['ROUGE-L'].mean():>10.4f} {delta(results_df['ROUGE-L'].mean(), base_best['ROUGE-L'].mean()):>10}")
        except Exception as e:
            print(f"\n[Could not load baseline: {e}]")
    else:
        print(f"\n[Baseline file not found: {BASELINE_CSV}]")

    print("=" * 60)
    print("\nROUGE-1 reference points:")
    print("  0.8-1.0 = excellent  |  0.5-0.8 = good  |  0.3-0.5 = fair  |  <0.3 = poor")


# ─── Main evaluation loop ─────────────────────────────────────────────────────

def run_evaluation(
    retriever: "BaseRetriever",
    df: pd.DataFrame,
    client=None,
    top_k_retrieve: int = TOP_K_RETRIEVE,
    top_k_ctx: int = TOP_K_CONTEXT,
    use_gpt: bool = False,
    output_csv: Path = OUTPUT_CSV,
) -> pd.DataFrame:
    """
    Run retrieval (and optionally GPT answering) for every question in df.
    Saves results to output_csv and returns the results DataFrame.

    Works with any BaseRetriever — retriever-agnostic.
    """
    results = []
    skipped = 0
    start   = time.time()

    for i, row in df.iterrows():
        question  = str(row["שאלה"])          # Hebrew question text (column name from xlsx)
        reference = str(row["תשובה"])         # ground-truth answer
        source    = str(row.get("מקור", ""))  # source citation (siman + seif)

        # Parse siman; skip questions with unparseable source
        siman = extract_siman(source)
        if not siman:
            skipped += 1
            print(f"[{i+1}/{len(df)}] SKIP — cannot parse siman from: '{source[:60]}'")
            continue

        target_seif = extract_seif(source)

        # Retrieve top chunks for this question
        chunks_all = retriever.retrieve(question, top_k=top_k_retrieve)

        # Compute siman-level retrieval metrics
        r_at_1  = compute_recall_at_k(chunks_all, siman, 1)
        r_at_3  = compute_recall_at_k(chunks_all, siman, 3)
        r_at_5  = compute_recall_at_k(chunks_all, siman, 5)
        r_at_10 = compute_recall_at_k(chunks_all, siman, 10)
        mrr     = compute_reciprocal_rank(chunks_all, siman)

        # Compute seif-level retrieval metrics (finer grained)
        r_seif_1  = compute_recall_at_k_seif(chunks_all, siman, target_seif, 1)
        r_seif_3  = compute_recall_at_k_seif(chunks_all, siman, target_seif, 3)
        r_seif_5  = compute_recall_at_k_seif(chunks_all, siman, target_seif, 5)
        r_seif_10 = compute_recall_at_k_seif(chunks_all, siman, target_seif, 10)
        mrr_seif  = compute_reciprocal_rank_seif(chunks_all, siman, target_seif)

        # Optionally generate an answer with GPT
        if use_gpt and client:
            context = build_context(chunks_all[:top_k_ctx])
            answer  = ask_with_rag(client, question, context)
            bleu    = compute_bleu(reference, answer)
            rouge   = compute_rouge(reference, answer)
            f1      = _f1_tokens(normalize(reference).split(), normalize(answer).split())
            exact   = "[OK]" if normalize(answer) == normalize(reference) else "[X]"
            time.sleep(SLEEP)
        else:
            answer = ""
            bleu   = 0.0
            rouge  = {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
            f1     = 0.0
            exact  = ""

        # Progress line: shows retrieval status and estimated time remaining
        elapsed   = time.time() - start
        avg       = elapsed / (i + 1)
        remaining = avg * (len(df) - i - 1)
        top_score = chunks_all[0]["score"] if chunks_all else 0
        rel_mark  = "[+]" if r_at_3   else "[ ]"
        seif_mark = "s+"  if r_seif_3 else "s-"
        print(
            f"[{i+1}/{len(df)}] {rel_mark}{seif_mark} R@3={r_at_3:.0f}/sR@3={r_seif_3:.0f} "
            f"MRR={mrr:.2f} top={top_score:.3f}"
            + (f" | {exact} R1={rouge['rouge1']:.2f}" if use_gpt else "")
            + f" | ~{remaining/60:.1f} min remaining"
        )

        results.append({
            "ID":               row["#"],
            "question":         question,
            "reference_answer": reference,
            "source":           source,
            "siman":            siman,
            "seif":             target_seif,
            "RAG_answer":       answer,
            "top_score":        round(top_score, 4),
            "chunk_ids":        str([c["chunk_id"] for c in chunks_all[:top_k_ctx]]),
            "Recall@1":         r_at_1,
            "Recall@3":         r_at_3,
            "Recall@5":         r_at_5,
            "Recall@10":        r_at_10,
            "MRR":              round(mrr, 4),
            "Recall@1_seif":    r_seif_1,
            "Recall@3_seif":    r_seif_3,
            "Recall@5_seif":    r_seif_5,
            "Recall@10_seif":   r_seif_10,
            "MRR_seif":         round(mrr_seif, 4),
            "F1":               round(f1, 4),
            "BLEU":             round(bleu, 4),
            "ROUGE-1":          rouge["rouge1"],
            "ROUGE-2":          rouge["rouge2"],
            "ROUGE-L":          rouge["rougeL"],
        })

    results_df = pd.DataFrame(results)
    results_df.to_csv(output_csv, encoding="utf-8-sig", index=False)
    print(f"\nSaved: {output_csv}")
    if skipped:
        print(f"[Skipped {skipped} questions with unparseable source]")

    return results_df


# ─── CLI entry point ──────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="RAG evaluation — Recall@K, MRR, BLEU, ROUGE")
    parser.add_argument("--test",      type=int,  default=None,  help="evaluate only the first N questions")
    parser.add_argument("--no-gpt",    action="store_true",      help="retrieval metrics only, skip GPT calls")
    parser.add_argument("--topk",      type=int,  default=TOP_K_CONTEXT,
                        help=f"chunks to send to GPT (default: {TOP_K_CONTEXT})")
    parser.add_argument("--retriever", type=str,  default="semantic_e5_seif_v6_combined",
                        help=f"retriever name. Available: {list_retrievers()}")
    args = parser.parse_args()

    use_gpt = not args.no_gpt

    if use_gpt:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not found in .env")
        client = OpenAI(api_key=api_key)
    else:
        client = None
        print("--no-gpt mode: computing retrieval metrics only.\n")

    print(f"Loading retriever: {args.retriever}...")
    retriever = get_retriever(args.retriever)

    df = pd.read_excel(XLSX_PATH)
    if args.test:
        df = df.head(args.test)
        print(f"Test mode: {args.test} questions\n")
    else:
        print(f"Processing {len(df)} questions...\n")

    results_df = run_evaluation(
        retriever,
        df,
        client=client,
        top_k_ctx=args.topk,
        use_gpt=use_gpt,
        output_csv=OUTPUT_CSV,
    )

    print_summary(results_df, use_gpt)


if __name__ == "__main__":
    main()
