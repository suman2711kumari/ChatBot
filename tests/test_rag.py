"""Retrieval tests use the checked-in summaries and require no internet."""
from rag import retrieve_resources, reference_context


def test_study_stress_retrieves_authoritative_resource(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    matches = retrieve_resources("I am overwhelmed by exams and study deadlines")
    assert matches
    assert all(m["url"].startswith("https://") for m in matches)
    assert any("stress" in m["topic"] or "study" in m["topic"] for m in matches)


def test_greeting_does_not_fabricate_references(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    assert retrieve_resources("hello") == []


def test_reference_notes_are_bounded(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    matches = retrieve_resources("I cannot sleep at night", limit=2)
    assert len(matches) <= 2
    assert "http" in reference_context(matches)


def test_retriever_reports_actual_method(monkeypatch):
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    results = retrieve_resources("I feel overwhelmed about exams")
    assert results and all(x["retrieval_method"] == "tfidf" for x in results)
