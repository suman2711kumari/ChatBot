"""RoBERTa scoring, fallback and separated analytics without any model download."""
import sqlite3
from datetime import datetime, timezone

from fastapi.testclient import TestClient
import main2
import sentiment_engine as se


def test_score_mapping_and_all_three_probabilities(monkeypatch):
    calls = []
    def classifier(message, **kwargs):
        calls.append((message, kwargs))
        return [
            {"label": "negative", "score": 0.80},
            {"label": "neutral", "score": 0.15},
            {"label": "positive", "score": 0.05},
        ]
    monkeypatch.setattr(se, "get_roberta_classifier", lambda: classifier)
    monkeypatch.setenv("SENTIMENT_BACKEND", "roberta")
    result = se.score_sentiment("I'm losing hope")
    assert result["sentiment_model"] == "roberta"
    assert result["sentiment_score"] == -0.75
    assert result["sentiment_label"] == "negative"
    assert result["sentiment_probabilities"]["neutral"] == 0.15
    assert calls[0][1]["top_k"] is None


def test_numeric_labels_and_invalid_probabilities():
    data = [
        {"label": "LABEL_0", "score": 0.20},
        {"label": "LABEL_1", "score": 0.50},
        {"label": "LABEL_2", "score": 0.30},
    ]
    assert se.normalize_roberta_probabilities(data) == {
        "negative": 0.2, "neutral": 0.5, "positive": 0.3
    }


def test_unavailable_roberta_falls_back_to_vader(monkeypatch):
    monkeypatch.setenv("SENTIMENT_BACKEND", "roberta")
    def missing_model():
        raise OSError("No pretrained weights")
    monkeypatch.setattr(se, "get_roberta_classifier", missing_model)
    result = se.score_sentiment("I'm happy!")
    assert result["sentiment_model"] == "vader"
    assert result["sentiment_score"] > 0
    assert result["sentiment_probabilities"] is None


def test_default_vader_does_not_load_transformer(monkeypatch):
    monkeypatch.setenv("SENTIMENT_BACKEND", "vader")
    monkeypatch.setattr(se, "get_roberta_classifier",
                        lambda: (_ for _ in ()).throw(AssertionError("Unexpected RoBERTa")))
    assert se.score_sentiment("The exam starts at ten.")["sentiment_model"] == "vader"


def test_model_specific_dashboard_filters_scores(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "isolated.db")
    monkeypatch.setenv("SENTIMENT_BACKEND", "roberta")
    main2.init_db()
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(main2.DB_PATH) as conn:
        conn.executemany("""
            INSERT INTO messages
            (user_id,message,reply,sentiment_score,sentiment_model,model_used,timestamp)
            VALUES (?,?,?,?,?,?,?)
        """, [
            ("a", "legacy", "ok", 0.8, "vader", "gemini", stamp),
            ("b", "new", "ok", -0.4, "roberta", "gemini", stamp),
        ])
    with TestClient(main2.app) as client:
        transformer = client.get("/admin/stats").json()
        vader = client.get("/admin/stats?sentiment_model=vader").json()
        assert transformer["sentiment_model"] == "roberta"
        assert transformer["average_sentiment"] == -0.4
        assert transformer["sentiment_message_count"] == 1
        assert vader["average_sentiment"] == 0.8
        assert vader["sentiment_message_count"] == 1
        assert vader["message_count"] == transformer["message_count"] == 2
        assert client.get("/admin/stats?sentiment_model=unknown").status_code == 422


def test_database_migration_marks_only_legacy_vader_scores(tmp_path, monkeypatch):
    path = tmp_path / "upgrade.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE messages
            (id INTEGER PRIMARY KEY, user_id TEXT, message TEXT, reply TEXT,
            sentiment_score REAL, model_used TEXT, timestamp TEXT)""")
        conn.execute("INSERT INTO messages(id,sentiment_score) VALUES (1,0.6)")
        conn.execute("INSERT INTO messages(id,sentiment_score) VALUES (2,NULL)")
    monkeypatch.setattr(main2, "DB_PATH", path)
    main2.init_db()
    main2.init_db()
    with sqlite3.connect(path) as conn:
        assert conn.execute("SELECT sentiment_model FROM messages WHERE id=1").fetchone()[0] == "vader"
        assert conn.execute("SELECT sentiment_model FROM messages WHERE id=2").fetchone()[0] is None


def test_chat_response_and_database_store_actual_model(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "chat.db")
    main2.init_db()
    monkeypatch.setattr(main2, "score_sentiment", lambda message: {
        "sentiment_score": -0.75, "sentiment_model": "roberta",
        "sentiment_label": "negative",
        "sentiment_probabilities": {"negative": 0.8, "neutral": 0.15, "positive": 0.05},
    })
    async def fake_reply(message, history, references=None):
        return "Example response", "gemini"
    monkeypatch.setattr(main2, "generate_reply", fake_reply)
    with TestClient(main2.app) as client:
        res = client.post("/chat", json={
            "user_id": "demo", "message": "I'm losing hope"
        })
        assert res.status_code == 200
        data = res.json()
        assert data["sentiment_model"] == "roberta"
        assert data["sentiment_score"] == -0.75
        assert data["sentiment_probabilities"]["neutral"] == 0.15
    with sqlite3.connect(main2.DB_PATH) as conn:
        assert conn.execute("SELECT sentiment_model,sentiment_label,sentiment_score FROM messages").fetchone() == ("roberta", "negative", -0.75)
