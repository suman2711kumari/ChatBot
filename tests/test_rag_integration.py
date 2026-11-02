"""Offline API integration checks for document RAG, sentiment and admin trends."""
from fastapi.testclient import TestClient
import main2


def sample_chunks():
    return [{
        "title": "WHO: Stress guide",
        "url": "https://www.who.int/example",
        "source_file": "who_guide.pdf",
        "publisher": "WHO",
        "file_type": "pdf",
        "page": 12,
        "chunk_index": 1,
        "chunk_text": "Practical skills for coping with stress and difficult thoughts.",
        "similarity": 0.82,
        "retrieval_method": "minilm_faiss",
    }]


def patch_retrieval(monkeypatch):
    monkeypatch.setattr(main2, "retrieve_chunks", lambda query, limit=3: sample_chunks())
    monkeypatch.setattr(main2, "is_index_ready", lambda: True)


def test_references_with_gemini_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    patch_retrieval(monkeypatch)

    async def fake_reply(message, previous, references=None):
        assert references
        assert "Practical skills" in main2.reference_context(references)
        return "Thanks for sharing your exam worries.", "gemini"

    monkeypatch.setattr(main2, "generate_reply", fake_reply)
    with TestClient(main2.app) as client:
        response = client.post("/chat", json={
            "user_id": "student", "message": "I am overwhelmed by my exams and studies"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["references"][0]["title"].startswith("WHO:")
        assert data["rag_info"]["retrieved_count"] == 1
        assert data["rag_info"]["context_attached_to_gemini"] is True
        assert -1 <= data["sentiment_score"] <= 1
        stats = client.get("/admin/stats").json()
        assert stats["total_active_users"] == 1
        assert len(stats["sentiment_trend"]["dates"]) == 1
        assert len(stats["sentiment_trend"]["avg_scores"]) == 1


def test_fallback_does_not_claim_grounding(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    patch_retrieval(monkeypatch)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_DIALOGPT_FALLBACK", "true")
    monkeypatch.setattr(main2, "generate_local_reply",
                        lambda message, previous: "Local demo reply.")
    with TestClient(main2.app) as client:
        data = client.post("/chat", json={
            "user_id": "student", "message": "Exams are making me stressed"
        }).json()
        assert data["model_used"] == "dialogpt_fallback"
        assert data["references"] == []
        assert data["suggested_reading"][0]["title"].startswith("WHO:")


def test_crisis_blocks_generation_and_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    patch_retrieval(monkeypatch)
    with TestClient(main2.app) as client:
        data = client.post("/chat", json={
            "user_id": "student", "message": "I want to die"
        }).json()
        assert data["model_used"] == "safety_message"
        assert data["references"] == []
        assert data["suggested_reading"] == []


def test_suicide_intent_in_user_example_bypasses_gemini_and_rag(tmp_path, monkeypatch):
    """Regression: explicit first-person self-harm intent bypasses retrieval."""
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    monkeypatch.setattr(
        main2, "retrieve_chunks",
        lambda query, limit=3: (_ for _ in ()).throw(AssertionError("RAG must be bypassed")),
    )
    with TestClient(main2.app) as client:
        result = client.post("/chat", json={
            "user_id": "student",
            "message": "i want to suicide now. i know no one care for me",
        })
        assert result.status_code == 200
        payload = result.json()
        assert payload["model_used"] == "safety_message"
        assert payload["references"] == []
        assert payload["suggested_reading"] == []
        assert "112" in payload["reply"]


def test_safety_backstop_catches_some_direct_phrasings():
    for text in [
        "I am suicidal",
        "I'm thinking about suicide",
        "I want to hurt myself",
        "I have been thinking of killing myself",
        "I want to die",
    ]:
        assert main2.check_crisis_message(text), text
    assert not main2.check_crisis_message("I am writing a report on suicide prevention")
    assert not main2.check_crisis_message("I'm stressed about my assignment")


def test_retrieval_preview_returns_real_chunks_without_gemini_or_logging(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    patch_retrieval(monkeypatch)
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-preview", json={"query": "Ways to manage exam stress"})
        assert res.status_code == 200
        data = res.json()
        assert data["generator_called"] is False
        assert data["match_count"] == len(data["matches"])
        assert data["matches"][0]["retrieval_method"] == "minilm_faiss"
        assert "chunk_text" in data["matches"][0]
        assert 0 < data["matches"][0]["similarity"] <= 1
        stats = client.get("/admin/stats").json()
        assert stats["message_count"] == 0


def test_rag_preview_explains_missing_index(monkeypatch):
    monkeypatch.setattr(main2, "is_index_ready", lambda: False)
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-preview", json={"query": "Ways to manage exam stress"})
        assert res.status_code == 503
        assert "ingest_documents.py" in res.json()["detail"]


def test_comparison_is_opt_in_and_two_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    patch_retrieval(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "fake-key")
    calls = []

    async def fake_reply(query, reference_notes):
        calls.append(reference_notes)
        return "Test answer"

    monkeypatch.setattr(main2, "generate_demo_reply", fake_reply)
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-compare", json={"query": "Ways to manage exam stress"})
        assert res.status_code == 200
        assert res.json()["gemini_calls"] == 2
        assert calls[0] is None
        assert "WHO" in calls[1]
        assert "Practical skills" in calls[1]
        assert client.get("/admin/stats").json()["message_count"] == 0


def test_demo_does_not_allow_explicit_crisis_example():
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-preview", json={"query": "I want to suicide now"})
        assert res.status_code == 400
