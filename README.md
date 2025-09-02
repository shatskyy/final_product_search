# Product Hunt Similarity (final_product_search)

Self-contained release to search similar Product Hunt products.
Most users only need to run setup and search; rebuilding the index is optional.

## Contents

- `raw/` — input CSVs:
  - `product-hunt-prouducts-1-1-2014-to-12-31-2021.csv`
  - `product_hunt_2025_for_index.csv`
- `scripts/` — helper scripts
  - `setup.sh` — sets up a local virtual environment and installs deps
  - `build.sh` — merges CSVs and builds artifacts
  - `search.sh` — runs a search against the built artifacts
- `build_index.py` — self-contained index builder
- `search_cli.py` — CLI to query the index
- `artifacts/` — output directory for vectors and metadata (created after build)

## Quick start (search only)

1. Setup the virtual environment

```bash
cd final_product_search
bash scripts/setup.sh
```

2. Run a search (saves JSON under `./outputs/` with a unique filename)

```bash
bash scripts/search.sh "AI meeting notes summarizer with action items"
```

What you get:

- Top 10 matches with name, URL, tags, upvotes, release date (YYYY-MM-DD), and description
- A JSON file saved in `outputs/` (e.g., `outputs/search_YYYYMMDDTHHMMSSZ_<id>.json`)

Optional: direct JSON output

```bash
python search_cli.py \
  "AI meeting notes summarizer with action items" \
  --artifacts-dir . \
  --out-json outputs/example.json
```

## CLI reference

```bash
python search_cli.py "<your idea>" \
  --artifacts-dir . \
  [--top-n 10] [--k 50] [--quote-len 200] \
  [--model sentence-transformers/all-MiniLM-L6-v2] \
  [--out-json outputs/results.json]
```

## Optional: Rebuild the index

You do NOT need this to run searches. Rebuild only if you change CSVs or want to regenerate artifacts.

```bash
bash scripts/build.sh
```

This merges `raw/` CSVs and writes artifacts under `./artifacts/`:

- `artifacts/vectors/vectors.npy`
- `artifacts/meta/meta.parquet`
- `artifacts/meta/idf.json`
- `artifacts/meta/snapshot.json`

## Environment and troubleshooting

- Scripts auto-activate the local venv. Manual activation (optional):
  - `source .venv/bin/activate` (deactivate with `deactivate`)
- If commands are not found, ensure you ran `bash scripts/setup.sh` from this folder.
- If you get fewer than 10 results on tiny builds, increase recall: `bash scripts/search.sh "your query" --k 200`.
