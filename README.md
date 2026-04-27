# rag_shul
Run from the exp_main.py 
Files are created next to the exp_main.py file to examine the project flow.
config.yaml settings.

---

## Pipeline Overview

A RAG pipeline over the Shulchan Arukh (Orach Chaim), built in five stages:

```
Preprocess Data  →  Chunker  →  Embedder  →  Retriever  →  Evaluation
 data/scripts/      chunker/    embedder/    retrievers/   evaluation/
```

Each stage is an independent module. Read that module's section below before touching its code.

---

## Stage 1 — Preprocess Data (`data/scripts/`)

Converts the raw source text into a structured JSON file ready for chunking. Run the two scripts in order.

### Script A — `build_shulchan_aruch_rag.py`

Parses the raw Torat Emet TXT file and produces the canonical RAG JSON.

**Input**

`data/source_original/data_fixed.txt` — raw Shulchan Arukh text (Torat Emet edition)

**Output**

`data/processed/shulchan_aruch_rag.json`

```json
{
  "title": "שולחן ערוך, אורח חיים",
  "source": "Torat Emet 363",
  "simanim": [
    {
      "siman": 1,
      "seifim": [
        { "seif": 1, "text": "...", "hagah": "...", "text_raw": "..." }
      ]
    }
  ]
}
```

**Run (CLI)**

```bash
# zero-config (uses default paths)
python data/scripts/build_shulchan_aruch_rag.py

# explicit paths
python data/scripts/build_shulchan_aruch_rag.py --input path/to/raw.txt --output out.json

# regression test suite
python data/scripts/build_shulchan_aruch_rag.py --test
```

**Options**

| Flag | Description |
|------|-------------|
| `--input, -i` | Input TXT file (default: `data/source_original/data_fixed.txt`) |
| `--output, -o` | Output JSON file (default: `data/processed/shulchan_aruch_rag.json`) |
| `--test` | Run built-in regression tests |
| `--quiet, -q` | Suppress progress output |

> CLI-only — no Python API.

---

### Script B — `add_breadcrumb_to_json.py`

Enriches the RAG JSON with hierarchical headings (hilchot group + siman title).

**Input**

- `data/processed/shulchan_aruch_rag.json` (output of Script A)
- `data/source_original/Shulchan_Aruch_Text_Headlines.txt` — table of contents / siman headings

**Output**

`data/processed/shulchan_aruch_rag_with_breadcrumb.json`

Adds two fields to each siman:

```json
{
  "siman": 1,
  "hilchot_group": "הלכות הנהגת אדם בבוקר",
  "siman_sign": "דין השכמת הבוקר",
  "seifim": [ ... ]
}
```

**Run (CLI)**

```bash
# zero-config
python data/scripts/add_breadcrumb_to_json.py

# explicit paths
python data/scripts/add_breadcrumb_to_json.py \
    --json data/processed/shulchan_aruch_rag.json \
    --headings data/source_original/Shulchan_Aruch_Text_Headlines.txt \
    --output data/processed/shulchan_aruch_rag_with_breadcrumb.json

# regression tests
python data/scripts/add_breadcrumb_to_json.py --test
```

**Options**

| Flag | Description |
|------|-------------|
| `--json, -j` | Input JSON file |
| `--headings, -H` | Headings TXT file |
| `--output, -o` | Output JSON file |
| `--test` | Run built-in regression tests |
| `--quiet, -q` | Suppress progress output |

> CLI-only — no Python API.

---

## Stage 2 — Chunker (`chunker/`)

Reads the RAG JSON and produces a `chunks.json` file for the embedder.

**Input**

`data/processed/shulchan_aruch_rag.json`

```json
{
  "simanim": [
    {
      "siman": 1,
      "seifim": [
        { "seif": 1, "text": "...", "hagah": "..." }
      ]
    }
  ]
}
```

**Output**

`data/chunks.json` — a JSON array, one object per chunk:

```json
[
  { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
  { "id": 1, "siman": 1, "seif": null, "siman_seif": "סימן 1", "text": "..." }
]
```

`seif` is `null` for siman-level and sliding-window chunks.

**Options (`config/config.yaml`)**

```yaml
chunker:
  mode: seif            # seif | siman | sliding_window
  chunk_size: 200       # words per chunk (sliding_window only)
  overlap: 50           # overlapping words between chunks (sliding_window only)
  chunk_fields:
    - text              # always included
    # - hagah           # uncomment to append Rema commentary
    # - siman_title     # uncomment to prepend the siman heading
```

| Mode | Description |
|------|-------------|
| `seif` | One chunk per seif (default) |
| `siman` | One chunk per siman (all seifim merged) |
| `sliding_window` | Fixed word-count windows across the full corpus |

**Run (CLI)**

```bash
python -m chunker.main
# or with explicit paths:
python -m chunker.main --input data/processed/shulchan_aruch_rag.json --output data/chunks.json
```

**Use as API**

```python
from chunker import build_chunks

chunks = build_chunks("data/processed/shulchan_aruch_rag.json")
# returns list[dict] with id, siman, seif, siman_seif, text

# override mode or fields without touching config:
chunks = build_chunks("data/processed/shulchan_aruch_rag.json", mode="siman", chunk_fields=["text", "hagah"])
```

---

## Stage 3 — Embedder (`embedder/`)

Encodes the chunks into dense vector embeddings and saves them as a `.npy` matrix.

**Input**

`data/chunks.csv` — chunks table with columns: `siman`, `seif`, `text`

**Output**

`data/embeddings.npy` — normalized float32 matrix, one row per chunk (same order as CSV)

**Options (`config/config.yaml`)**

```yaml
embeddings:
  model: intfloat/multilingual-e5-large
  batch_size: 32
  prefix_passage: "passage: "
  prefix_query: "query: "
  enrich_fields: []        # optional extra fields to concatenate into the passage
  enrich_separator: " | "
```

**Run (CLI)**

```bash
python embedder/embed.py --chunks data/chunks.csv
# with explicit output path:
python embedder/embed.py --chunks data/chunks.csv --npy data/embeddings.npy --model intfloat/multilingual-e5-large
```

**Use as API**

```python
from embedder.embed import build_embeddings, encode_query

# build the full embeddings matrix
build_embeddings("data/chunks.csv", "data/embeddings.npy")

# encode a single query at retrieval time
query_vec = encode_query("מה דין ציצית?")
```

---

## Stage 4 — Retriever (`retrievers/`)

> **Status: implementation in progress.** This section describes the intended interface in general terms.

**Input**

- A user query (string)
- The embedded vector database (`.npy` matrix + chunks metadata from Stage 3)

**Output**

A ranked list of the most relevant chunks, each with a relevance score and source metadata (siman, seif, text).

**Expected behavior**

Given a query, embed it with the same model used in Stage 3, search the embedding space, and return the top-K most similar chunks.

**Options (`config/config.yaml`)**

| Key | Description |
|-----|-------------|
| `top_k` | Number of chunks returned to the caller |
| `score_threshold` | Minimum similarity score to include a result |

> Detailed API, available retriever types, and run commands will be added once the implementation is stable.

---

## Stage 5 — Evaluation (`evaluation/`)

> **Status: implementation in progress.** This section describes the intended interface in general terms.

**Input**

- A configured retriever (Stage 4)
- A benchmark CSV of questions with known correct source simanim

**Output**

Evaluation metrics report (Recall@K, MRR) saved as `.txt` and `.json`.

**Expected behavior**

For each benchmark question, run the retriever and check whether the correct siman appears within the top-K results. Aggregate across all questions into Recall@K and MRR scores.

**Options (`config/config.yaml`)**

| Key | Description |
|-----|-------------|
| `k_values` | List of K values to compute Recall@K for |
| `target_k` | K threshold for the pass/fail target |
| `target_recall` | Minimum recall rate to pass |
| `max_questions` | Limit evaluation to N questions (`null` = all) |

> Detailed API, available evaluator types, and run commands will be added once the implementation is stable.
