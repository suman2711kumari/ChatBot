"""Offline API integration checks for RAG, sentiment and admin trends."""
from fastapi.testclient import TestClient
import main2


def fake_sentiment(message):
    return {
        "sentiment_score": -0.4, "sentiment_label": "negative",
        "sentiment_model": "roberta",
        "sentiment_probabilities": {"negative": .6, "neutral": .2, "positive": .2},
    }



def test_references_with_gemini_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    monkeypatch.setattr(main2, "score_sentiment", fake_sentiment)

    async def fake_reply(message, previous, references=None):
        assert references
        return "Thanks for sharing your exam worries.", "gemini"

    monkeypatch.setattr(main2, "generate_reply", fake_reply)
    with TestClient(main2.app) as client:
        response = client.post("/chat", json={
            "user_id": "student", "message": "I am overwhelmed by my exams and studies"
        })
        assert response.status_code == 200
        data = response.json()
        assert data["references"]
        assert data["suggested_reading"] == []
        assert -1 <= data["sentiment_score"] <= 1
        stats = client.get("/admin/stats").json()
        assert stats["total_active_users"] == 1
        assert len(stats["sentiment_trend"]["dates"]) == 1
        assert len(stats["sentiment_trend"]["avg_scores"]) == 1


def test_fallback_does_not_claim_grounding(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
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
        assert data["suggested_reading"]


def test_crisis_blocks_generation_and_reading(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    with TestClient(main2.app) as client:
        data = client.post("/chat", json={
            "user_id": "student", "message": "I want to die"
        }).json()
        assert data["model_used"] == "safety_message"
        assert data["references"] == []
        assert data["suggested_reading"] == []

def test_suicide_intent_in_user_example_bypasses_gemini_and_rag(tmp_path, monkeypatch):
    """Regression: earlier pattern checks missed the exact example."""
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
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


def test_retrieval_preview_does_not_call_gemini_or_log(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-preview", json={"query": "Ways to manage exam stress"})
        assert res.status_code == 200
        data = res.json()
        assert data["generator_called"] is False
        assert data["match_count"] == len(data["matches"])
        assert data["matches"][0]["retrieval_method"] == "tfidf"
        assert 0 < data["matches"][0]["similarity"] <= 1
        stats = client.get("/admin/stats").json()
        assert stats["message_count"] == 0


def test_comparison_is_opt_in_and_two_calls(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("RAG_BACKEND", "tfidf")
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
        assert "WHO" in calls[1] or "NIMH" in calls[1]
        assert client.get("/admin/stats").json()["message_count"] == 0


def test_demo_does_not_allow_explicit_crisis_example():
    with TestClient(main2.app) as client:
        res = client.post("/admin/rag-preview", json={"query": "I want to suicide now"})
        assert res.status_code == 400
