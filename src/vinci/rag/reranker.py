# src/vinci/rag/reranker.py

import math
import re
from collections import Counter


def _tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens."""
    return re.findall(r"\b\w+\b", text.lower())


def _overlap_score(query: str, snippet: dict) -> float:
    """Calculate the number of unique matching query terms."""
    query_tokens = set(_tokenize(query))
    excerpt_tokens = set(_tokenize(snippet.get("excerpt", "")))

    return float(len(query_tokens & excerpt_tokens))


def _tfidf_scores(query: str, snippets: list[dict]) -> list[float]:
    """Calculate TF-IDF cosine similarity across all snippets."""

    query_tokens = Counter(_tokenize(query))

    if not query_tokens:
        return [0.0] * len(snippets)

    documents = [Counter(_tokenize(s.get("excerpt", ""))) for s in snippets]

    N = len(documents)

    if N == 0:
        return []

    # Calculate document frequency.
    df = Counter()

    for doc in documents:
        for token in doc:
            df[token] += 1

    # Calculate IDF.
    idf = {token: math.log((N + 1) / (freq + 1)) + 1 for token, freq in df.items()}

    # Query IDF uses the same document statistics.
    query_vector = {
        token: count * idf.get(token, math.log(N + 1) + 1)
        for token, count in query_tokens.items()
    }

    query_norm = math.sqrt(sum(weight**2 for weight in query_vector.values()))

    scores = []

    for doc in documents:
        # TF-IDF document vector.
        doc_vector = {token: count * idf[token] for token, count in doc.items()}

        dot = sum(
            weight * doc_vector.get(token, 0.0)
            for token, weight in query_vector.items()
        )

        doc_norm = math.sqrt(sum(weight**2 for weight in doc_vector.values()))

        if query_norm == 0 or doc_norm == 0:
            scores.append(0.0)
        else:
            scores.append(dot / (query_norm * doc_norm))

    return scores


def _bm25_scores(
    query: str, snippets: list[dict], k1: float = 1.5, b: float = 0.75
) -> list[float]:
    """Calculate BM25 relevance scores across all snippets."""

    query_tokens = _tokenize(query)

    if not query_tokens:
        return [0.0] * len(snippets)

    documents = [_tokenize(s.get("excerpt", "")) for s in snippets]

    N = len(documents)

    if N == 0:
        return []

    doc_lengths = [len(doc) for doc in documents]
    avgdl = sum(doc_lengths) / N

    if avgdl == 0:
        return [0.0] * N

    # Calculate document frequency.
    df = Counter()

    for doc in documents:
        for token in set(doc):
            df[token] += 1

    # Calculate BM25 IDF.
    idf = {
        token: math.log(1 + (N - freq + 0.5) / (freq + 0.5))
        for token, freq in df.items()
    }

    scores = []

    for doc, dl in zip(documents, doc_lengths):
        tf = Counter(doc)
        score = 0.0

        for token in query_tokens:
            freq = tf[token]

            if freq == 0:
                continue

            denominator = freq + k1 * (1 - b + b * dl / avgdl)

            score += idf.get(token, 0.0) * (freq * (k1 + 1) / denominator)

        scores.append(score)

    return scores


def rerank(snippets: list[dict], query: str, method: str = "overlap") -> list[dict]:
    """Rerank snippets by relevance to the query."""

    method = method.lower()

    if method == "overlap":
        scores = [_overlap_score(query, s) for s in snippets]

    elif method == "tfidf":
        scores = _tfidf_scores(query, snippets)

    elif method == "bm25":
        scores = _bm25_scores(query, snippets)

    else:
        raise ValueError(f"Unsupported reranking method: {method}")

    # Preserve original order when scores are equal.
    ranked = [{**snippet, "score": score} for snippet, score in zip(snippets, scores)]

    return sorted(ranked, key=lambda d: d["score"], reverse=True)
