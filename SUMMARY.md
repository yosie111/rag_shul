# RAG System for Shulchan Arukh — Orach Chayim

> **Retrieval only** — no GPT answer generation required to run

---

## What This System Does

**RAG** (Retrieval-Augmented Generation) answers questions in two steps:
1. **Retrieval** — find the most relevant passages from a large text corpus
2. **Generation** — feed those passages to a language model (GPT-4) to produce an answer

**Corpus**: Shulchan Arukh, Orach Chayim (Torat Emet 363 edition) — 697 simanim (chapters) of Jewish law
**Test set**: 596 halakha questions with ground-truth answers and source references (siman + seif)
**Success metric**: Recall@K — is the correct seif (paragraph) found among the top-K results?

---

## Folder Structure

```
SH_H_RAG/
├── README.md                          ← this file
│
├── ── Pipeline steps ──
├── step_01_data_loading.py            ← load, clean, split into chunks
├── step_02_embeddings.py              ← encode to vectors → .npy cache
├── step_03_retrieval.py               ← retrieval functions (ChromaDB)
├── step_05_evaluate_rag.py            ← evaluation: Recall@K, MRR
│
├── ── Data builders ──
├── build_chunks_v6.py                 ← build seifs_v6.json
├── rebuild_seifs_index.py             ← rebuild with bug fixes
├── build_seifs_combined.py            ← build seifs_v6_combined.json ★
├── build_seifs_gpt_questions.py       ← build seifs with GPT questions
├── build_seifs_modern_summary.py      ← build seifs with modern summary
├── generate_seif_summaries.py         ← GPT summaries per seif
├── generate_seif_questions_local.py   ← rule-based questions per seif
│
├── ── Experiment management ──
├── run_experiment.py                  ← run experiment + evaluate
├── compare_experiments.py             ← compare all experiments
├── step_06_failure_analysis.py        ← failure analysis
├── analyze_seif_99.py                 ← per-siman analysis
├── config.py                          ← file paths
│
├── ── Retrievers ──
├── retrievers/
│   ├── __init__.py                    ← REGISTRY
│   ├── base.py                        ← BaseRetriever (ABC)
│   └── semantic_e5_seif_v6_combined.py  ← ★ best retriever (exp_028)
│
├── ── Data files (JSON) ──
├── seifs_v6_index.json                ← 4,169 seifs (baseline)
├── seifs_v6_combined.json             ← 4,169 seifs + summary + questions ★
│
├── ── Embedding files (NumPy) ──
├── seifs_v6_combined_intfloat_multilingual_e5_large.npy  ← (4169×1024) ★
│
└── shulchan_arukh_all_questions.xlsx  ← 596 test questions
```

---

## Step 1: Data Preparation

### File: `step_01_data_loading.py`

The raw text goes through a cleaning pipeline and is then split into chunks:

```
raw text
    ↓ remove_html_tags()           — strip HTML tags
    ↓ remove_niqqud()              — remove Hebrew vowel marks (U+0591–U+05C7)
    ↓ remove_parenthetical_refs()  — remove short citation parentheticals
    ↓ normalize_geresh()           — normalize Hebrew geresh/gershayim → ASCII
    ↓ normalize_whitespace()       — collapse multiple spaces
    ↓ split_into_siman_chunks()    — split without crossing siman boundaries
```

**Two granularities of chunks produced:**

| Type | File | Count | Avg size | Description |
|------|------|-------|----------|-------------|
| chunks_v5 | `chunks_v5.json` | 1,554 | 138 words | large chunks, include context_prefix + GPT summary |
| seifs_v6 | `seifs_v6_index.json` | **4,169** | ~40 words | **single seif** — smallest halakhic unit |

**Chunk structure:**
```json
{
  "chunk_id": 0,
  "siman_parent": 1,
  "text": "...",
  "word_count": 138
}
```

**Seif structure:**
```json
{
  "chunk_id": 0,
  "siman": 1,
  "seif": 1,
  "context_prefix": "Shulchan Arukh, Orach Chayim, Siman 1, Seif 1:",
  "encoding_text": "Shulchan Arukh, Orach Chayim, Siman 1, Seif 1: ...",
  "text": "...",
  "word_count": 42,
  "summary": "..."
}
```

---

## Step 2: Encoding to Vectors (Embeddings)

### File: `step_02_embeddings.py`

**The idea**: every text passage is converted into a vector of 1,024 numbers representing its semantic meaning.
A question that is semantically similar to a passage will receive a close vector (high cosine similarity).

```python
model = SentenceTransformer("intfloat/multilingual-e5-large")

# The "passage: " / "query: " prefixes are required by E5
corpus_vec = model.encode("passage: " + text)    # for each seif
query_vec  = model.encode("query: " + question)  # for each question
```

**Model**: `intfloat/multilingual-e5-large`
- 560M parameters
- Supports Hebrew, Arabic, English, and 100+ other languages
- Produces L2-normalized 1024-dim vectors suitable for cosine similarity

**Caching**: a `.npy` file is saved automatically — subsequent runs with the same data load in seconds instead of recomputing (~30 min saved).

**Building embeddings (if .npy does not exist):**
```bash
python step_02_embeddings.py \
  --chunks seifs_v6_combined.json \
  --model intfloat/multilingual-e5-large \
  --collection combined_col \
  --chroma-dir chroma_combined
# → saves: seifs_v6_combined_intfloat_multilingual_e5_large.npy
```

---

## Step 3: Retrieval

### Interface: `retrievers/base.py`

```python
class BaseRetriever(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...        # unique name for the experiment

    @abstractmethod
    def retrieve(self, query: str, top_k: int = 10) -> list[dict]:
        """
        Returns: [
            {
                "rank":         1,
                "chunk_id":     158,
                "score":        0.8763,   # cosine similarity (0–1, higher = better)
                "text":         "...",
                "siman_parent": 71,
            },
            ...
        ]
        """
```

### Best retriever: `retrievers/semantic_e5_seif_v6_combined.py`

**How retrieval works** (simplified):
```python
def retrieve(self, query, top_k=10):
    self._load()

    # 1. Encode the question into a vector
    qvec = self._model.encode("query: " + query, normalize_embeddings=True)

    # 2. Compute cosine similarity against all 4,169 seifs
    #    (matrix multiply: very fast with NumPy)
    scores = self._embeddings @ qvec    # shape: (4169,)

    # 3. Sort and select top-K
    top_idx = np.argsort(scores)[-top_k:][::-1]

    # 4. Return results
    return [{"rank": r+1, "chunk_id": ..., "score": ..., "text": ..., "siman_parent": ...}
            for r, idx in enumerate(top_idx)]
```

### REGISTRY: `retrievers/__init__.py`

Every retriever is registered here:
```python
REGISTRY = {
    "semantic_e5_seif_v6_combined": SemanticE5SeifV6CombinedRetriever,
    # add new retrievers here
}

def get_retriever(name) -> BaseRetriever:
    return REGISTRY[name]()
```

---

## Step 4: Evaluation

### File: `step_05_evaluate_rag.py`

**Evaluation loop** (596 questions):
```python
for question in questions:
    siman  = extract_siman(source)    # parse source string to siman number
    seif   = extract_seif(source)     # parse seif number for finer evaluation

    results = retriever.retrieve(question, top_k=10)

    # Is the correct siman among the top-K results?
    recall_1  = 1.0 if any(r["siman_parent"]==siman for r in results[:1])  else 0.0
    recall_3  = 1.0 if any(r["siman_parent"]==siman for r in results[:3])  else 0.0
    recall_10 = 1.0 if any(r["siman_parent"]==siman for r in results[:10]) else 0.0

    # MRR = 1 / rank of first correct result
    mrr = 1 / rank_of_first_correct
```

**`extract_siman` function** — handles 5 different source formats:

| Input | Output |
|-------|--------|
| `(251, 2)` | `Siman 251` |
| `סימן 128` | `Siman 128` |
| `סימן א, סעיף א` | `Siman 1` |
| `שמ"ה, סעיף ז'` | `Siman 345` (gematria!) |
| `תרמ"ג` | `Siman 643` |

> **Bug fixed (exp_022)**: regex `[א-ת]+` stopped at geresh `"` → `תרמ"ג` was parsed as 640 instead of 643.
> **Fix**: `[א-ת"'\u05F3\u05F4]+` — impact: **+13% on all metrics!**

---

## Experiment Timeline — What We Learned

```
exp_001–003: E5-small → E5-large          → R@3: 34% → 38%
exp_004–009: better cleaning + chunking    → R@3: 38% → 56%
exp_010–011: context_prefix + GPT summary  → R@3: 60%
exp_012–013: split to seifs (3950 seifs)   → R@3: 60.4%
             └ fixed seif_num+1 bug        → R@3_seif: 11% → 48%
───── discovery: exp_001–021 measured with geresh bug! ─────
exp_022: fixed extract_siman (geresh)      → R@3: 60.4% → 74.66% (true value)
exp_023: HyDE (GPT generates hypothesis)  → R@3: 69.1% ✗
exp_024: Doc2Query (rule-based questions) → R@3: 75.84%
exp_025: GPT Expert Questions             → R@3: 78.02%, R@3_seif: 62.75% ★
exp_026: GPT Modern Summary               → R@3: 78.02%, R@10: 90.10% ★
exp_027: English BGE + translation        → R@3: 62.25% ✗✗
exp_028: Combined (summary + questions)   → R@3: 80.03% ★★★
```

---

## Best Experiment: exp_028 — Combined

### File: `retrievers/semantic_e5_seif_v6_combined.py`
### Data: `seifs_v6_combined.json` + `seifs_v6_combined_intfloat_multilingual_e5_large.npy`

**The idea**: each seif is embedded with **three layers of information** in its `encoding_text`:

```
encoding_text = context_prefix + text + modern_summary + gpt_questions
```

| Part | Content | Purpose |
|------|---------|---------|
| `context_prefix` | `"Shulchan Arukh, Orach Chayim, Siman X, Seif Y:"` | textual anchoring |
| `text` | original halakhic text (Aramaic/classical Hebrew) | precise content |
| `modern_summary` | 2–3 sentences in modern Hebrew (GPT-4o-mini) | bridge to modern user vocabulary |
| `gpt_questions` | 3–5 questions the seif answers (GPT-4o-mini) | semantic precision |

**Example encoding_text:**
```
Shulchan Arukh, Orach Chayim, Siman 4, Seif 1:
ישתדל אדם לעמוד בבוקר כארי לעבודת בוראו...
[modern_summary] One should rise in the morning with energy to serve God...
[questions] When should one rise in the morning? How should one rise? Why is alacrity required?
```

**Average size**: 162 words/seif (vs. 44 words in baseline)

---

## Final Results — All Experiments

| Experiment | Retriever | R@1 | R@3 | R@10 | MRR | R@3_seif |
|------------|-----------|-----|-----|------|-----|----------|
| 001 | E5-small | 22.9% | 34.1% | 48.7% | 0.286 | — |
| 013 ⚠️ | E5-seif (eval bug) | 48.0% | 60.4% | 73.0% | 0.557 | 48.3% |
| **022** | E5-seif (fixed eval) | 59.23% | 74.66% | 88.93% | 0.686 | 57.05% |
| 023 | HyDE | 51.2% | 69.1% | 82.6% | 0.616 | 54.9% |
| 024 | Doc2Query | 61.24% | 75.84% | 87.58% | 0.698 | 57.55% |
| 025 | GPT Questions | 63.42% | 78.02% | 88.59% | 0.719 | 62.75% |
| 026 | Modern Summary | 63.26% | 78.02% | **90.10%** | 0.721 | 60.57% |
| 027 | BGE English | 45.64% | 62.25% | 74.50% | 0.554 | 40.60% |
| **028 ★** | **Combined** | **63.76%** | **80.03%** | 88.76% | **0.727** | **64.09%** |

> **Metric**: R@K = percentage of questions where the correct seif is found among the top-K results

> ⚠️ exp_001–021 were measured with a geresh parsing bug — values are ~13% too low.

---

## Why E5-large and Not Another Model?

| Model | R@3 | Why it failed |
|-------|-----|--------------|
| E5-small | 34% | too small (118M params) |
| mBERT | 12% | not a sentence encoder — wrong architecture for similarity |
| heBERT | 35% | Hebrew-specific but also not a sentence encoder |
| DictaBERT | 35% | same issue |
| BGE-large-en | 62% | English only + translation loses halakhic nuance |
| **E5-large** | **80%** | Multilingual sentence encoder, 1024 dim, cross-lingual alignment |

**Key reason**: E5-large was specifically trained on question-answer pairs in 100+ languages using contrastive learning — exactly what is needed here.

---

## Pipeline Architecture — Overview

```
Shulchan Arukh (raw text)
         │
         ▼
[step_01_data_loading.py]
  clean_text() → split_into_siman_chunks()
         │
         └──► [build_chunks_v6.py]
                insert_seif_markers()
                extract_seifs()
                │
                ▼
              seifs_v6.json  (4,169 seifs × 40 words)
                │
                ├──► [generate_seif_summaries.py]      → seif_summaries_cache.json
                ├──► [generate_seif_questions_gpt.py]  → seif_questions_gpt_cache.json
                └──► [generate_seif_modern_summary.py] → seif_modern_summary_cache.json
                                │
                                ▼
                         [build_seifs_combined.py]
                    encoding_text = prefix + text + summary + questions
                                │
                                ▼
                         seifs_v6_combined.json
                                │
                                ▼
                         [step_02_embeddings.py]
                           encode("passage: " + encoding_text)
                                │
                                ▼
               seifs_v6_combined_...e5_large.npy  (4169×1024)
                                │
                                ▼
                  [SemanticE5SeifV6CombinedRetriever]
                    encode("query: " + question)
                    scores = embeddings @ query_vec
                    top_k results
```

---

## Useful Commands

```bash
# Run an experiment with the best retriever
echo y | python run_experiment.py \
  --retriever semantic_e5_seif_v6_combined \
  --name exp_028_combined \
  --no-gpt

# Quick test run (10 questions only)
echo y | python run_experiment.py \
  --retriever semantic_e5_seif_v6_combined \
  --name test_run \
  --test 10 \
  --no-gpt

# Compare all experiments
python compare_experiments.py --sort recall3

# Direct Python query
from retrievers import get_retriever
r = get_retriever("semantic_e5_seif_v6_combined")
results = r.retrieve("What is the law for someone who forgot Ya'aleh ve-Yavo?", top_k=5)
for res in results:
    print(f"[{res['rank']}] Siman {res['siman_parent']}, score={res['score']:.4f}")
    print(res['text'][:100])
    print()
```

---

## Adding a New Retriever

```python
# 1. Create retrievers/my_retriever.py
from .base import BaseRetriever
import json, numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer

class MyRetriever(BaseRetriever):
    @property
    def name(self): return "my_retriever"

    def __init__(self):
        self._model = self._embeddings = self._chunks = None

    def _load(self):
        if self._model: return
        self._model = SentenceTransformer("intfloat/multilingual-e5-large")
        base = Path(__file__).parent.parent
        self._embeddings = np.load(str(base / "MY_FILE.npy"))
        with open(base / "MY_CHUNKS.json", encoding="utf-8") as f:
            self._chunks = json.load(f)

    def retrieve(self, query, top_k=10):
        self._load()
        qvec = self._model.encode("query: " + query, normalize_embeddings=True)
        scores = self._embeddings @ qvec
        top_idx = np.argsort(scores)[-top_k:][::-1]
        return [{"rank": r+1, "chunk_id": self._chunks[i]["chunk_id"],
                 "score": round(float(scores[i]), 4), "text": self._chunks[i]["text"],
                 "siman_parent": self._chunks[i].get("siman", 0)}
                for r, i in enumerate(top_idx)]

# 2. Register in retrievers/__init__.py:
# from .my_retriever import MyRetriever
# REGISTRY["my_retriever"] = MyRetriever

# 3. Run:
# echo y | python run_experiment.py --retriever my_retriever --name exp_029 --no-gpt
```

---

## Remaining Failure Points

**Glass ceiling at R@3 ≈ 80%**:
- **57 questions** about content-sparse simanim (siman 620 = 15 words, siman 600 = 72 words)
- **Siman 4** (handwashing): 15+ seifs with very similar content → the system confuses them
- **Next step suggestion**: Fine-tune E5 on question–seif pairs specific to Shulchan Arukh

---

## Developer Notes

1. **Windows + Hebrew output**: always add at the top of scripts that print Hebrew:
   ```python
   import sys, io
   sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
   ```

2. **ChromaDB v1.5.5 bug**: HNSW index breaks on collections > ~1000 items → **solution**: use `.npy` files directly for all queries (not ChromaDB).

3. **Geresh bug (FIXED)**: previously `תרמ"ג` was parsed as 640 instead of 643.
   Fixed regex in `extract_siman`: `[א-ת"'\u05F3\u05F4]+`

4. **Cache keys** in `seif_questions_gpt_cache.json`: `f"{siman}_{seif}"` (seif starts at 1)

5. **`run_experiment.py`**: requires interactive input → use `echo y | python run_experiment.py ...`
