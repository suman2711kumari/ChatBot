"""MiniLM + FAISS retrieval over locally stored source-document chunks.

Documents are loaded and indexed by ingest_documents.py. This module only
loads the persisted index, embeds a user query, and returns relevant chunks.
Retrieved material is supportive background, not medical advice or diagnosis.
"""
from __future__ import annotations

import json
import logging
import os
from functools import lru_cache
from pathlib import Path

LOGGER = logging.getLogger(__name__)
BASE_DIR = Path(__file__).resolve().parent
VECTOR_STORE_DIR = BASE_DIR / "knowledge" / "vector_store"
INDEX_FILE = VECTOR_STORE_DIR / "index.faiss"
CHUNKS_FILE = VECTOR_STORE_DIR / "chunks.json"
MANIFEST_FILE = VECTOR_STORE_DIR / "manifest.json"
DEFAULT_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def is_index_ready() -> bool:
    """Return whether both the FAISS index and its chunk metadata are present."""
    return INDEX_FILE.is_file() and CHUNKS_FILE.is_file() and MANIFEST_FILE.is_file()


@lru_cache(maxsize=1)
def _load_assets():
    """Load the local FAISS index, chunk metadata and CPU MiniLM model once."""
    if not is_index_ready():
        raise FileNotFoundError(
            "RAG index is missing. Run 'python ingest_documents.py' first."
        )

    import faiss
    from sentence_transformers import SentenceTransformer

    chunks = json.loads(CHUNKS_FILE.read_text(encoding="utf-8"))
    manifest = json.loads(MANIFEST_FILE.read_text(encoding="utf-8"))
    model_name = os.getenv("RAG_EMBEDDING_MODEL", DEFAULT_MODEL).strip()
    if manifest.get("embedding_model") != model_name:
        raise ValueError(
            "The saved index was built with a different embedding model. "
            "Run python ingest_documents.py again after changing RAG_EMBEDDING_MODEL."
        )
    index = faiss.read_index(str(INDEX_FILE))
    if index.ntotal != len(chunks):
        raise ValueError(
            f"FAISS index contains {index.ntotal} vectors but "
            f"{len(chunks)} chunk records were found."
        )

    model = SentenceTransformer(model_name, device="cpu")
    return model, index, chunks


def retrieve_chunks(query: str, limit: int = 3) -> list[dict]:
    """Retrieve the highest-scoring source chunks using MiniLM and FAISS.

    MiniLM embeddings are normalized during indexing and query encoding.
    FAISS IndexFlatIP therefore returns cosine similarity via inner product.
    """
    if not query or not query.strip() or limit <= 0:
        return []

    try:
        import numpy as np

        model, index, chunks = _load_assets()
        if index.ntotal == 0:
            return []

        query_vector = model.encode(
            [query.strip()],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        query_vector = np.ascontiguousarray(query_vector, dtype=np.float32)
        scores, indices = index.search(query_vector, min(limit, index.ntotal))
    except FileNotFoundError as exc:
        LOGGER.warning("%s", exc)
        return []
    except ImportError:
        LOGGER.warning(
            "RAG dependencies are missing. Install requirements-rag-embeddings.txt."
        )
        return []
    except Exception:
        LOGGER.exception("Document retrieval failed; continuing without RAG context.")
        return []

    try:
        threshold = float(os.getenv("RAG_MIN_SIMILARITY", "0.30"))
    except ValueError:
        threshold = 0.30

    results = []
    for score, chunk_position in zip(scores[0], indices[0]):
        if chunk_position < 0 or float(score) < threshold:
            continue
        stored = chunks[int(chunk_position)]
        metadata = stored.get("metadata", {})
        results.append({
            "title": metadata.get("title") or metadata.get("source_file", "Source document"),
            "url": metadata.get("url") or "",
            "source_file": metadata.get("source_file") or "",
            "publisher": metadata.get("publisher") or "",
            "file_type": metadata.get("file_type") or "",
            "page": metadata.get("page"),
            "chunk_index": metadata.get("chunk_index"),
            "chunk_text": stored.get("page_content", ""),
            "similarity": round(float(score), 3),
            "retrieval_method": "minilm_faiss",
        })
    return results


def reference_context(chunks: list[dict]) -> str:
    """Format retrieved source chunks as reference data, never as instructions."""
    if not chunks:
        return "No relevant source-document chunks were retrieved."

    formatted = []
    for index, item in enumerate(chunks, start=1):
        location = f", page {item['page']}" if item.get("page") else ""
        source = f"Source: {item['url']}" if item.get("url") else (
            f"Local file: {item.get('source_file', 'uploaded document')}"
        )
        text = (item.get("chunk_text") or "").strip()[:1600]
        formatted.append(
            f"[{index}] {item.get('title', 'Source document')}{location}\n"
            f"{source}\n"
            f"Passage:\n{text}"
        )
    return "\n\n".join(formatted)
