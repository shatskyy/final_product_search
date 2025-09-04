# Product Hunt Similarity Search

**Find similar products to your startup idea instantly.** Search through 45,975 Product Hunt products from 2014-2025 using AI-powered semantic similarity.

## What it does

Enter a description of your product idea (e.g., "AI meeting notes summarizer") and get:

- **Top 10 similar products** with names, descriptions, and Product Hunt URLs
- **Upvotes and release dates** to gauge market validation
- **Matching keywords** that explain why products are similar
- **JSON export** for further analysis

Perfect for market research, competitor analysis, and idea validation.

## Quick start (2 minutes)

**1. Setup** (one-time)

```bash
git clone https://github.com/shatskyy/final_product_search.git
cd final_product_search
bash scripts/setup.sh
```

**2. Search**

```bash
bash scripts/search.sh "your product idea here"
```

**Example:**

```bash
bash scripts/search.sh "AI meeting notes summarizer with action items"
```

**Results appear instantly** — no building required, the search index is prebuilt.

## What you get

**Console output:**

```
1. Scribe AI  (score=0.598)
   URL: https://www.producthunt.com/posts/scribe-ai
   Tags: ARTIFICIAL INTELLIGENCE, PRODUCTIVITY, MEETINGS
   Upvotes: 146   Release: 2021-04-24
   similar intent to: 'Automate note-taking and share highlights...'
```

**JSON file:** Saved automatically in `outputs/` with timestamp (e.g., `search_20250902T101637Z_65156da0.json`)

## Advanced usage

**Direct CLI with custom options:**

```bash
python search_cli.py "your idea" --artifacts-dir . --top-n 15 --out-json results.json
```

**Options:**

- `--top-n` Number of results (default: 10)
- `--k` Search breadth for better recall (default: 50, try 200 for more results)
- `--quote-len` Max characters in similarity quotes (default: 200)

## Project structure

```
final_product_search/
├── README.md                 # This file
├── search_cli.py            # Main search engine (241 lines)
├── artifacts/               # Prebuilt search index (94MB)
│   ├── vectors/             # Product embeddings (67MB)
│   └── meta/                # Product metadata + keywords
├── outputs/                 # Your search results (JSON)
├── scripts/
│   ├── setup.sh             # Install dependencies
│   └── search.sh            # Search with auto-save
└── raw/                     # Source data (20MB CSV)
    └── product_hunt_2014_2025_merged.csv
```

## Data source

- **45,975 products** from Product Hunt (2014-2025)
- **Semantic embeddings** using sentence-transformers/all-MiniLM-L6-v2
- **TF-IDF keywords** for match explanations
- **Metadata:** name, description, tags, upvotes, release date, URL

## Rebuild index (optional)

Only needed if you modify the source data:

```bash
bash scripts/build.sh
```

## Troubleshooting

**"Command not found"**

- Run `bash scripts/setup.sh` first
- Or manually: `source .venv/bin/activate`

**Getting fewer than 10 results**

- Try: `bash scripts/search.sh "your query" --k 200`

**Search seems slow**

- First run downloads the AI model (~90MB), subsequent searches are fast

## Requirements

- Python 3.8+
- 150MB free space (for dependencies + model)
- Internet connection (first run only)

## License & Attribution

Built with sentence-transformers, pandas, and numpy. Product Hunt data used under fair use for research purposes.

---

**Questions?** Open an issue or check the code — it's well-commented and only ~600 lines total.
