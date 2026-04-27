# Chunker

Reads the Shulchan Arukh RAG JSON and produces a `chunks.json` file for the embedder.

---

## Input

JSON file with this structure (`data/processed/shulchan_aruch_rag.json`):
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

---

## Output

`data/chunks.json` — a JSON array, one object per chunk:
```json
[
  { "id": 0, "siman": 1, "seif": 1, "siman_seif": "סימן 1, סעיף 1", "text": "..." },
  { "id": 1, "siman": 1, "seif": null, "siman_seif": "סימן 1", "text": "..." }
]
```
`seif` is `null` for siman-level and sliding-window chunks.

---

## Options (config/config.yaml)

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
|---|---|
| `seif` | One chunk per seif (default) |
| `siman` | One chunk per siman (all seifim merged) |
| `sliding_window` | Fixed word-count windows across the full corpus |

---

## Run (CLI)

```bash
python3 -m chunker.main
# or with explicit paths:
python3 -m chunker.main --input data/processed/shulchan_aruch_rag.json --output data/chunks.json
```

---

## Use as API

```python
from chunker import build_chunks

chunks = build_chunks("data/processed/shulchan_aruch_rag.json")
# returns list[dict] with id, siman, seif, siman_seif, text

# override mode or fields without touching config:
chunks = build_chunks("data/processed/shulchan_aruch_rag.json", mode="siman", chunk_fields=["text", "hagah"])
```
