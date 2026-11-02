"""Unit tests for chunk retrieval; the model and FAISS index are mocked."""
import numpy as np

import rag


class FakeModel:
    def encode(self, texts, **kwargs):
        assert texts == ["exam stress"]
        return np.asarray([[1.0, 0.0]], dtype=np.float32)


class FakeIndex:
    ntotal = 2

    def search(self, query, count):
        assert count == 2
        return (
            np.asarray([[0.82, 0.18]], dtype=np.float32),
            np.asarray([[0, 1]], dtype=np.int64),
        )


def sample_chunks():
    return [
        {
            "page_content": "Practical skills for coping with stress.",
            "metadata": {
                "title": "WHO: Stress guide",
                "url": "https://www.who.int/example",
                "source_file": "who_guide.pdf",
                "publisher": "WHO",
                "file_type": "pdf",
                "page": 12,
                "chunk_index": 1,
            },
        },
        {
            "page_content": "A passage about unrelated topics.",
            "metadata": {
                "title": "NIMH: Self-care",
                "url": "https://www.nimh.nih.gov/example",
                "source_file": "nimh.html",
                "publisher": "NIMH",
                "file_type": "html",
                "chunk_index": 2,
            },
        },
    ]


def test_retriever_returns_high_scoring_source_chunk(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.30")
    monkeypatch.setattr(
        rag, "_load_assets",
        lambda: (FakeModel(), FakeIndex(), sample_chunks()),
    )

    matches = rag.retrieve_chunks("exam stress", limit=2)

    assert len(matches) == 1
    assert matches[0]["title"] == "WHO: Stress guide"
    assert matches[0]["page"] == 12
    assert matches[0]["retrieval_method"] == "minilm_faiss"
    assert matches[0]["similarity"] == 0.82
    assert "Practical skills" in matches[0]["chunk_text"]


def test_retriever_filters_low_similarity(monkeypatch):
    monkeypatch.setenv("RAG_MIN_SIMILARITY", "0.90")
    monkeypatch.setattr(
        rag, "_load_assets",
        lambda: (FakeModel(), FakeIndex(), sample_chunks()),
    )

    assert rag.retrieve_chunks("exam stress", limit=2) == []


def test_blank_query_returns_no_chunks():
    assert rag.retrieve_chunks("   ") == []


def test_reference_context_contains_chunk_text_and_citation():
    context = rag.reference_context([{
        "title": "WHO: Stress guide",
        "url": "https://www.who.int/example",
        "page": 12,
        "source_file": "who_guide.pdf",
        "chunk_text": "A passage extracted from the original guide.",
    }])

    assert "WHO: Stress guide" in context
    assert "page 12" in context
    assert "https://www.who.int/example" in context
    assert "A passage extracted from the original guide." in context
    assert "summary" not in context.lower()


def test_empty_context_is_explicit():
    assert "No relevant source-document chunks" in rag.reference_context([])
