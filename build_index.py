import argparse
import json
import math
import re
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


# -----------------------------
# Text normalization utilities
# -----------------------------

_URL_REGEX = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_HTML_TAG_REGEX = re.compile(r"<[^>]+>")
_WHITESPACE_REGEX = re.compile(r"\s+")
_PUNCT_SPACE_REGEX = re.compile(r"\s*([.,;:!?()\[\]{}\-])\s*")


ENGLISH_STOPWORDS = {
    "the",
    "be",
    "to",
    "of",
    "and",
    "a",
    "in",
    "that",
    "have",
    "I",
    "it",
    "for",
    "not",
    "on",
    "with",
    "he",
    "as",
    "you",
    "do",
    "at",
    "this",
    "but",
    "his",
    "by",
    "from",
}


def strip_html_and_urls(text: str) -> str:
    text = unescape(text)
    text = _HTML_TAG_REGEX.sub(" ", text)
    text = _URL_REGEX.sub(" ", text)
    text = _PUNCT_SPACE_REGEX.sub(r" \1 ", text)
    text = _WHITESPACE_REGEX.sub(" ", text)
    return text.strip()


def normalize_text_for_embedding(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = strip_html_and_urls(text)
    return text


def tokenized(text: str) -> List[str]:
    return [token for token in re.findall(r"[a-z0-9][a-z0-9\-]+", text.lower())]


def is_likely_english(text: str) -> bool:
    if not text:
        return False
    ascii_letters = sum(c.isalpha() and ord(c) < 128 for c in text)
    total_letters = sum(c.isalpha() for c in text)
    if total_letters == 0:
        return False
    ascii_ratio = ascii_letters / max(1, total_letters)
    toks = tokenized(text)
    if not toks:
        return False
    stopword_hits = sum(1 for t in toks if t in ENGLISH_STOPWORDS)
    stopword_ratio = stopword_hits / max(1, len(toks))
    return ascii_ratio >= 0.85 and stopword_ratio >= 0.01


# -----------------------------
# Data schema and helpers
# -----------------------------


@dataclass
class BuildConfig:
    input_csv: Path
    artifacts_dir: Path
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    keyphrases_top_k: int = 20
    english_only: bool = True
    min_description_chars: int = 40
    top_n_snippet_chars: int = 200
    limit_rows: Optional[int] = None


REQUIRED_COLUMNS = [
    "_id",
    "name",
    "product_description",
    "category_tags",
    "upvotes",
    "release_date",
]


def safe_parse_tags(raw_value: object) -> List[str]:
    if isinstance(raw_value, list):
        return [str(x) for x in raw_value]
    if not isinstance(raw_value, str) or raw_value.strip() == "":
        return []
    s = raw_value.strip()
    # Try JSON list first
    try:
        import json as _json

        v = _json.loads(s)
        if isinstance(v, list):
            return [str(x) for x in v]
    except Exception:
        pass
    # Fallback: comma-separated string
    if "," in s:
        return [part.strip() for part in s.split(",") if part.strip()]
    # Single tag string
    return [s]


def build_canonical_text(name: str, description: str, tags: List[str]) -> str:
    name = name or ""
    description = description or ""
    tags_text = " ".join([t.replace("#", " ").replace("_", " ") for t in tags])
    pieces = []
    if name:
        pieces.append(f"{name}.")
    if description:
        pieces.append(description)
    if tags_text:
        pieces.append(f"Tags: {tags_text}")
    return " ".join(pieces).strip()


def ensure_artifact_dirs(base: Path) -> Dict[str, Path]:
    vectors_dir = base / "artifacts" / "vectors"
    meta_dir = base / "artifacts" / "meta"
    vectors_dir.mkdir(parents=True, exist_ok=True)
    meta_dir.mkdir(parents=True, exist_ok=True)
    return {"vectors": vectors_dir, "meta": meta_dir}


def validate_columns(df: pd.DataFrame) -> None:
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def load_and_prepare_dataframe(cfg: BuildConfig) -> pd.DataFrame:
    df = pd.read_csv(cfg.input_csv)
    validate_columns(df)

    # Basic coercions
    df["name"] = df["name"].astype(str).fillna("")
    df["product_description"] = df["product_description"].astype(str).fillna("")
    df["upvotes"] = pd.to_numeric(df["upvotes"], errors="coerce").fillna(0).astype(int)
    # Keep release_date as string; optional parsing later
    if cfg.limit_rows is not None:
        df = df.head(cfg.limit_rows)

    # Parse tags
    df["tags_list"] = df["category_tags"].apply(safe_parse_tags)

    # Create canonical text with normalization
    df["name_norm"] = df["name"].apply(normalize_text_for_embedding)
    df["desc_norm"] = df["product_description"].apply(normalize_text_for_embedding)
    df["canon_text"] = [
        build_canonical_text(n, d, t)
        for n, d, t in zip(df["name_norm"], df["desc_norm"], df["tags_list"])
    ]

    # Optional filters
    if cfg.min_description_chars > 0:
        df = df[df["desc_norm"].str.len() >= cfg.min_description_chars]
    if cfg.english_only:
        tqdm.pandas(desc="english-filter")
        df = df[df["canon_text"].progress_apply(is_likely_english)]

    df = df.reset_index(drop=True)
    return df


def build_keyphrase_store(df: pd.DataFrame, cfg: BuildConfig) -> Tuple[TfidfVectorizer, np.ndarray, Dict[str, float], List[List[str]]]:
    vectorizer = TfidfVectorizer(
        max_features=None,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.9,
        lowercase=False,
        token_pattern=r"[a-z0-9][a-z0-9\-]+",
        strip_accents=None,
        sublinear_tf=True,
    )
    tfidf = vectorizer.fit_transform(df["canon_text"].tolist())
    feature_names = np.array(vectorizer.get_feature_names_out())
    idf_scores = dict(zip(feature_names.tolist(), vectorizer.idf_.tolist()))

    # Top tokens per row
    top_tokens_per_row: List[List[str]] = []
    for row_idx in range(tfidf.shape[0]):
        row = tfidf.getrow(row_idx)
        if row.nnz == 0:
            top_tokens_per_row.append([])
            continue
        data = row.data
        indices = row.indices
        if len(data) <= cfg.keyphrases_top_k:
            sorted_idx = np.argsort(-data)
        else:
            # partial top-k
            part = np.argpartition(-data, cfg.keyphrases_top_k - 1)[: cfg.keyphrases_top_k]
            sorted_idx = part[np.argsort(-data[part])]
        tokens = feature_names[indices[sorted_idx]].tolist()
        top_tokens_per_row.append(tokens)

    return vectorizer, tfidf, idf_scores, top_tokens_per_row


def l2_normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12
    return vectors / norms


def embed_corpus(texts: List[str], model_name: str) -> np.ndarray:
    model = SentenceTransformer(model_name)
    vectors = model.encode(texts, batch_size=256, show_progress_bar=True, normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype=np.float32)
    return vectors


def save_artifacts(
    cfg: BuildConfig,
    dirs: Dict[str, Path],
    vectors: np.ndarray,
    idf_scores: Dict[str, float],
    df_meta: pd.DataFrame,
) -> None:
    # Vectors
    np.save(dirs["vectors"] / "vectors.npy", vectors)

    # IDF and snapshot/meta
    with open(dirs["meta"] / "idf.json", "w", encoding="utf-8") as f:
        json.dump(idf_scores, f)

    snapshot = {
        "snapshot_id": f"ph_{int(time.time())}",
        "model": cfg.model_name,
        "dims": int(vectors.shape[1]),
        "num_rows": int(vectors.shape[0]),
        "build_time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    with open(dirs["meta"] / "snapshot.json", "w", encoding="utf-8") as f:
        json.dump(snapshot, f)

    # Meta: parquet
    df_meta.to_parquet(dirs["meta"] / "meta.parquet", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build PH similarity index")
    parser.add_argument(
        "--input-csv",
        type=Path,
        required=True,
        help="Path to Product Hunt CSV snapshot",
    )
    parser.add_argument(
        "--artifacts-dir",
        type=Path,
        required=True,
        help="Directory to write artifacts (vectors/meta)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="SentenceTransformer model name",
    )
    parser.add_argument("--top-keyphrases", type=int, default=20)
    parser.add_argument("--min-desc-chars", type=int, default=40)
    parser.add_argument("--no-english-only", action="store_true")
    parser.add_argument("--limit", type=int, default=None, help="Optional row limit for quick builds")

    args = parser.parse_args()
    cfg = BuildConfig(
        input_csv=args.input_csv,
        artifacts_dir=args.artifacts_dir,
        model_name=args.model,
        keyphrases_top_k=args.top_keyphrases,
        english_only=not args.no_english_only,
        min_description_chars=args.min_desc_chars,
        limit_rows=args.limit,
    )

    dirs = ensure_artifact_dirs(cfg.artifacts_dir)

    t0 = time.time()
    df = load_and_prepare_dataframe(cfg)
    print(f"Loaded and prepared {len(df)} rows in {time.time() - t0:.1f}s")

    print("Building keyphrase store (TF-IDF)…")
    vectorizer, tfidf, idf_scores, top_tokens_per_row = build_keyphrase_store(df, cfg)
    df["top_tokens"] = top_tokens_per_row

    print("Encoding corpus (embeddings)…")
    vectors = embed_corpus(df["canon_text"].tolist(), cfg.model_name)

    # Ensure normalized
    vectors = l2_normalize_rows(vectors)

    # Prepare display meta
    def first_chars(text: str, max_len: int) -> str:
        return (text or "")[: cfg.top_n_snippet_chars]

    df_meta = pd.DataFrame(
        {
            "id": df["_id"],
            "name": df["name"],
            "description": df["product_description"],
            "snippet": df["product_description"].apply(lambda s: first_chars(str(s), cfg.top_n_snippet_chars)),
            "tags": df["tags_list"],
            "url": df["_id"],  # `_id` appears to be the PH URL
            "upvotes": df["upvotes"],
            "release_date": df["release_date"],
            "top_tokens": df["top_tokens"],
        }
    )

    print("Saving artifacts…")
    save_artifacts(cfg, dirs, vectors, idf_scores, df_meta)

    dt = time.time() - t0
    print(f"Done. Saved {vectors.shape[0]} vectors of dim {vectors.shape[1]} in {dt:.1f}s")


if __name__ == "__main__":
    main()
