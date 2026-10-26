"""Offline API integration checks for RAG, sentiment and admin trends."""
from fastapi.testclient import TestClient
import main2


def test_references_with_gemini_mock(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    monkeypatch.setenv("RAG_BACKEND", "tfidf")

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
