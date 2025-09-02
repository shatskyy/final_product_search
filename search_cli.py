import argparse
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer


TOKEN_REGEX = re.compile(r"[a-z0-9][a-z0-9\-]+")


def normalize(text: str) -> str:
    text = text or ""
    text = text.lower()
    # light normalization consistent with build
    text = re.sub(r"https?://\S+|www\.\S+", " ", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = re.sub(r"\s*([.,;:!?()\[\]{}\-])\s*", r" \1 ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def tokenize(text: str) -> List[str]:
    return TOKEN_REGEX.findall(text)


def cosine_topk(query_vec: np.ndarray, matrix: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray]:
    scores = matrix @ query_vec
    if matrix.shape[0] <= top_k:
        order = np.argsort(-scores)
    else:
        part = np.argpartition(-scores, top_k - 1)[:top_k]
        order = part[np.argsort(-scores[part])]
    return order, scores[order]


def format_release_date(raw: str) -> str:
    s = str(raw or "")
    m = re.search(r"\d{4}-\d{2}-\d{2}", s)
    return m.group(0) if m else s


def best_quote(query: str, description: str, model: SentenceTransformer, max_chars: int = 200) -> str:
    if not isinstance(description, str) or description.strip() == "":
        return ""
    # Simple sentence splitter
    sentences = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", description.strip())
    sentences = [s.strip() for s in sentences if len(s.strip()) > 0]
    if not sentences:
        return description[:max_chars]
    # Encode query and sentences
    q_vec = model.encode([query], normalize_embeddings=True)
    s_vecs = model.encode(sentences, normalize_embeddings=True)
    s_scores = (s_vecs @ q_vec.T).ravel()
    idx = int(np.argmax(s_scores))
    quote = sentences[idx]
    if len(quote) > max_chars:
        quote = quote[:max_chars].rstrip() + "…"
    return quote


def keyphrase_overlap(query_tokens: List[str], product_tokens: List[str], idf: Dict[str, float], max_terms: int = 4) -> List[str]:
    overlap = set(query_tokens).intersection(set(product_tokens))
    ranked = sorted(overlap, key=lambda t: idf.get(t, 0.0), reverse=True)
    return ranked[:max_terms]


def load_artifacts(artifacts_dir: Path) -> Tuple[np.ndarray, pd.DataFrame, Dict[str, float], dict]:
    vectors = np.load(artifacts_dir / "artifacts" / "vectors" / "vectors.npy")
    meta = pd.read_parquet(artifacts_dir / "artifacts" / "meta" / "meta.parquet")
    with open(artifacts_dir / "artifacts" / "meta" / "idf.json", "r", encoding="utf-8") as f:
        idf = json.load(f)
    with open(artifacts_dir / "artifacts" / "meta" / "snapshot.json", "r", encoding="utf-8") as f:
        snapshot = json.load(f)
    # Ensure required columns exist for robustness
    expected_cols = {"name", "url", "tags", "snippet", "upvotes", "release_date", "description", "top_tokens"}
    missing = expected_cols.difference(set(meta.columns))
    if missing:
        raise ValueError(f"Artifacts meta is missing required columns: {sorted(missing)}")
    return vectors, meta, idf, snapshot


def print_results(indices: np.ndarray, scores: np.ndarray, meta: pd.DataFrame, model: SentenceTransformer, query_norm: str, idf: Dict[str, float], top_n: int, quote_len: int, show_description: bool) -> None:
    q_tokens = tokenize(query_norm)
    for rank, (idx, score) in enumerate(zip(indices[:top_n], scores[:top_n]), start=1):
        row = meta.iloc[int(idx)]
        name = str(row.get("name", ""))
        url = str(row.get("url", ""))
        tags_raw = row.get("tags", [])
        # Coerce tags into a Python list of strings
        if tags_raw is None or (isinstance(tags_raw, float) and math.isnan(tags_raw)):
            tags = []
        elif isinstance(tags_raw, (list, tuple)):
            tags = [str(t) for t in tags_raw]
        elif hasattr(tags_raw, "tolist"):
            try:
                tags = [str(t) for t in tags_raw.tolist()]
            except Exception:
                tags = [str(tags_raw)]
        else:
            tags = [str(tags_raw)] if str(tags_raw) else []
        upvotes = int(row.get("upvotes", 0))
        release_date = format_release_date(str(row.get("release_date", "")))
        prod_tokens_raw = row.get("top_tokens", [])
        if isinstance(prod_tokens_raw, (list, tuple)):
            prod_tokens = [str(t) for t in prod_tokens_raw]
        elif hasattr(prod_tokens_raw, "tolist"):
            try:
                prod_tokens = [str(t) for t in prod_tokens_raw.tolist()]
            except Exception:
                prod_tokens = []
        else:
            prod_tokens = [str(prod_tokens_raw)] if prod_tokens_raw else []

        overlap_terms = keyphrase_overlap(q_tokens, prod_tokens, idf)
        quote = best_quote(query_norm, str(row.get("description", "")), model, max_chars=quote_len)

        print(f"{rank:2d}. {name}  (score={float(score):.3f})")
        if url:
            print(f"    URL: {url}")
        if len(tags) > 0:
            print(f"    Tags: {', '.join(map(str, tags))}")
        print(f"    Upvotes: {upvotes}   Release: {release_date}")
        if overlap_terms:
            print(f"    Explanation: matches on {', '.join(overlap_terms)};")
        if quote:
            print(f"      similar intent to: ‘{quote}’")
        if show_description:
            full_desc = str(row.get("description", ""))
            if full_desc:
                print()
                print("    Description:")
                # print as-is; leave formatting to terminal width
                for line in full_desc.splitlines():
                    print(f"      {line}")
        print()


def main() -> None:
    parser = argparse.ArgumentParser(description="Search similar Product Hunt products")
    parser.add_argument("query", type=str, help="Idea text (1–3 sentences)")
    parser.add_argument("--artifacts-dir", type=Path, required=True, help="Directory containing artifacts")
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--k", type=int, default=50, help="Top-K recall before selecting top-N")
    parser.add_argument("--quote-len", type=int, default=200)
    parser.add_argument("--no-description", action="store_true", help="Do not print full product descriptions")
    parser.add_argument("--model", type=str, default="sentence-transformers/all-MiniLM-L6-v2")
    parser.add_argument("--out-json", type=Path, default=None, help="Optional path to write results as JSON")

    args = parser.parse_args()

    vectors, meta, idf, snapshot = load_artifacts(args.artifacts_dir)
    model = SentenceTransformer(args.model)

    query_norm = normalize(args.query)
    q_vec = model.encode([query_norm], normalize_embeddings=True)
    q_vec = np.asarray(q_vec, dtype=np.float32).ravel()

    indices, scores = cosine_topk(q_vec, vectors, args.k)

    def dedupe_fill(idx_list: List[int], score_list: List[float], want: int) -> Tuple[List[int], List[float]]:
        seen = set()
        out_i: List[int] = []
        out_s: List[float] = []
        for i, sc in zip(idx_list, score_list):
            name = str(meta.iloc[int(i)]["name"]).strip().lower()
            if name in seen:
                continue
            seen.add(name)
            out_i.append(int(i))
            out_s.append(float(sc))
            if len(out_i) >= want:
                break
        return out_i, out_s

    # First pass: dedupe within top-k
    unique_indices, unique_scores = dedupe_fill(indices.tolist(), scores.tolist(), args.top_n)

    # Fallback: if fewer than requested, scan the full ranking
    if len(unique_indices) < args.top_n:
        full_order = np.argsort(- (vectors @ q_vec))
        full_scores = (vectors @ q_vec)[full_order]
        unique_indices, unique_scores = dedupe_fill(full_order.tolist(), full_scores.tolist(), args.top_n)

    # Optional JSON output
    if args.out_json is not None:
        results = []
        for idx, score in zip(unique_indices[: args.top_n], unique_scores[: args.top_n]):
            row = meta.iloc[int(idx)]
            # normalize tags to a plain list of strings for JSON
            tags_raw = row.get("tags", [])
            if tags_raw is None or (isinstance(tags_raw, float) and math.isnan(tags_raw)):
                tags = []
            elif isinstance(tags_raw, (list, tuple)):
                tags = [str(t) for t in tags_raw]
            elif hasattr(tags_raw, "tolist"):
                try:
                    tags = [str(t) for t in tags_raw.tolist()]
                except Exception:
                    tags = [str(tags_raw)]
            else:
                tags = [str(tags_raw)] if str(tags_raw) else []

            results.append({
                "name": str(row.get("name", "")),
                "url": str(row.get("url", "")),
                "tags": tags,
                "upvotes": int(row.get("upvotes", 0)),
                "release_date": format_release_date(str(row.get("release_date", ""))),
                "description": str(row.get("description", "")),
                "score": float(score),
            })
        args.out_json.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out_json, "w", encoding="utf-8") as f:
            json.dump({
                "query": args.query,
                "top_n": args.top_n,
                "results": results,
            }, f, ensure_ascii=False, indent=2)

    print_results(
        np.array(unique_indices),
        np.array(unique_scores),
        meta,
        model,
        query_norm,
        idf,
        args.top_n,
        args.quote_len,
        show_description=(not args.no_description),
    )


if __name__ == "__main__":
    main()


