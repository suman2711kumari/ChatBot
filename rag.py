"""Small, transparent RAG retriever over reviewed public self-help summaries.

Default: CPU-friendly TF-IDF retrieval; no dataset, GPU or model download.
Optional: MiniLM sentence embeddings, loaded only with RAG_BACKEND=semantic.
Retrieval results are reading suggestions, NOT clinical evidence or diagnosis.
"""
import json
import logging
import os
from functools import lru_cache
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

LOGGER = logging.getLogger(__name__)
DATA_FILE = Path(__file__).resolve().parent / "knowledge" / "resources.json"


@lru_cache(maxsize=1)
def _index():
    resources = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    for item in resources:
        if not item["url"].startswith("https://"):
            raise ValueError("Only HTTPS knowledge-base links are allowed")
    documents = [
        " ".join([
            item["title"], item["topic"],
            " ".join(item["keywords"]) * 2,
            item["summary"],
        ])
        for item in resources
    ]
    vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
    vectors = vectorizer.fit_transform(documents)
    return resources, documents, vectorizer, vectors


@lru_cache(maxsize=1)
def _semantic_model():
    # Optional model (~90 MB download plus libraries). CPU is sufficient.
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2", device="cpu")


@lru_cache(maxsize=1)
def _semantic_vectors():
    _, documents, _, _ = _index()
    return _semantic_model().encode(documents, normalize_embeddings=True)


def retrieve_resources(query: str, limit: int = 2) -> list[dict]:
    """Return related reading only when similarity clears the threshold."""
    if not query or not query.strip():
        return []
    resources, _, vectorizer, vectors = _index()
    method = "tfidf"
    if os.getenv("RAG_BACKEND", "tfidf").strip().lower() == "semantic":
        try:
            model = _semantic_model()
            query_vector = model.encode([query], normalize_embeddings=True)
            scores = cosine_similarity(query_vector, _semantic_vectors())[0]
            threshold = 0.35
            method = "semantic"
        except Exception:
            LOGGER.warning("Semantic RAG unavailable; using TF-IDF", exc_info=True)
            scores = cosine_similarity(vectorizer.transform([query]), vectors)[0]
            threshold = 0.13
    else:
        scores = cosine_similarity(vectorizer.transform([query]), vectors)[0]
        threshold = 0.13

    ranked = sorted(enumerate(scores), key=lambda pair: pair[1], reverse=True)
    return [
        dict(resources[index], similarity=round(float(score), 3), retrieval_method=method)
        for index, score in ranked if float(score) >= threshold
    ][:max(0, min(limit, 3))]


def reference_context(resources: list[dict]) -> str:
    """Treat summaries as reference data, not as instructions to the model."""
    if not resources:
        return "No relevant reference notes were found for this message."
    return "\n".join(
        f"[{i}] {item['title']} ({item['url']}): {item['summary']}"
        for i, item in enumerate(resources, start=1)
    )
