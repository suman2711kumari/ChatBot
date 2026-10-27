"""Unit and API regression tests for RoBERTa-only text sentiment."""
import sqlite3
from datetime import datetime, timezone
import pytest
from fastapi.testclient import TestClient
import main2
import sentiment_engine as se


def test_probability_score_and_label(monkeypatch):
    def fake(message, **kwargs):
        assert kwargs["top_k"] is None
        return [{"label": "negative", "score": .80},
                {"label": "neutral", "score": .15},
                {"label": "positive", "score": .05}]
    monkeypatch.setattr(se, "get_roberta_classifier", lambda: fake)
    result = se.score_sentiment("Feeling hopeless")
    assert result["sentiment_model"] == "roberta"
    assert result["sentiment_score"] == -.75
    assert result["sentiment_label"] == "negative"
    assert result["sentiment_probabilities"]["neutral"] == .15


def test_normalize_numeric_labels():
    data = [{"label": "LABEL_0", "score": .2},
            {"label": "LABEL_1", "score": .5},
            {"label": "LABEL_2", "score": .3}]
    assert se.normalize_roberta_probabilities(data) == {
        "negative": .2, "neutral": .5, "positive": .3
    }


def test_bad_probabilities_raise():
    with pytest.raises(ValueError):
        se.normalize_roberta_probabilities([
            {"label": "negative", "score": .8},
            {"label": "neutral", "score": .8},
            {"label": "positive", "score": .1},
        ])


def test_no_fallback_on_missing_roberta(monkeypatch):
    def missing():
        raise OSError("model weights unavailable")
    monkeypatch.setattr(se, "get_roberta_classifier", missing)
    with pytest.raises(OSError):
        se.score_sentiment("Hello")


def test_only_roberta_scores_in_analytics(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "stats.db")
    main2.init_db()
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(main2.DB_PATH) as db:
        db.executemany("""
            INSERT INTO messages(user_id,message,reply,sentiment_score,sentiment_model,model_used,timestamp)
            VALUES (?,?,?,?,?,?,?)""", [
            ("old", "previous", "ok", .9, "legacy", "gemini", now),
            ("new", "current", "ok", -.4, "roberta", "gemini", now),
        ])
    with TestClient(main2.app) as client:
        result = client.get("/admin/stats").json()
        assert result["sentiment_model"] == "roberta"
        assert result["average_sentiment"] == -.4
        assert result["sentiment_message_count"] == 1
        assert result["message_count"] == 2
        assert client.get("/admin/stats?sentiment_model=legacy").status_code == 200


def test_inference_error_does_not_block_chat(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "chat.db")
    main2.init_db()
    def broken(message):
        raise RuntimeError("roberta failed")
    monkeypatch.setattr(main2, "score_sentiment", broken)
    async def reply(msg, turns, references=None):
        return "Thanks for sharing.", "gemini"
    monkeypatch.setattr(main2, "generate_reply", reply)
    with TestClient(main2.app) as client:
        res = client.post("/chat", json={"user_id": "a", "message": "Feeling worried"})
        assert res.status_code == 200
        payload = res.json()
        assert payload["sentiment_status"] == "unavailable"
        assert payload["sentiment_score"] is None
        assert payload["sentiment_model"] == "roberta"
        assert client.get("/admin/stats").json()["sentiment_message_count"] == 0


def test_crisis_bypasses_sentiment_inference(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "safety.db")
    main2.init_db()
    monkeypatch.setattr(main2, "score_sentiment",
                        lambda message: (_ for _ in ()).throw(AssertionError("scorer called")))
    with TestClient(main2.app) as client:
        data = client.post("/chat", json={"user_id": "a", "message": "I want to die"}).json()
        assert data["model_used"] == "safety_message"
        assert data["sentiment_status"] == "skipped_safety"
        assert data["sentiment_score"] is None


def test_legacy_migration_preserves_scores(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as db:
        db.execute("""CREATE TABLE messages(
            id INTEGER PRIMARY KEY, user_id TEXT, message TEXT, reply TEXT,
            sentiment_score REAL, model_used TEXT, timestamp TEXT)""")
        db.execute("INSERT INTO messages(id,sentiment_score) VALUES(1,0.5)")
    monkeypatch.setattr(main2, "DB_PATH", path)
    main2.init_db()
    with sqlite3.connect(path) as db:
        assert db.execute("SELECT sentiment_model FROM messages").fetchone()[0] == "legacy"
