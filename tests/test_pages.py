"""The two HTML pages keep the RAG chatbot and 24h sentiment contract."""
import sqlite3
from datetime import datetime, timedelta, timezone

from fastapi.testclient import TestClient
import main2


def test_chat_and_admin_are_independent_pages(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    with TestClient(main2.app) as client:
        chat = client.get("/")
        admin = client.get("/admin")
        assert chat.status_code == 200
        assert admin.status_code == 200
        assert "chatForm" in chat.text
        assert 'href="/admin"' in chat.text
        assert "sentimentChart" not in chat.text
        assert "sentimentChart" in admin.text
        assert "/static/admin.js" in admin.text
        assert "chatForm" not in admin.text
        assert "/admin/stats" not in chat.text
        for file in ("styles.css", "chat.js", "admin.js"):
            asset = client.get("/static/" + file)
            assert asset.status_code == 200
            assert len(asset.content) > 100


def test_admin_metrics_use_same_24h_window(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    now = datetime.now(timezone.utc)
    fresh = now.strftime("%Y-%m-%d %H:%M:%S")
    stale = (now - timedelta(hours=26)).strftime("%Y-%m-%d %H:%M:%S")
    with sqlite3.connect(main2.DB_PATH) as conn:
        conn.executemany(
            """INSERT INTO messages
                (user_id, message, reply, sentiment_score, model_used, timestamp)
                VALUES (?, ?, ?, ?, ?, ?)""",
            [
                ("a", "good", "hi", 0.8, "gemini", fresh),
                ("b", "bad", "hi", -0.4, "dialogpt_fallback", fresh),
                ("old", "yesterday", "hi", -1, "gemini", stale),
            ],
        )
    with TestClient(main2.app) as client:
        stats = client.get("/admin/stats")
        assert stats.status_code == 200
        data = stats.json()
        assert data["total_active_users"] == 2
        assert data["message_count"] == 2
        assert abs(data["average_sentiment"] - 0.2) < 0.001
        assert data["model_usage"] == {"dialogpt_fallback": 1, "gemini": 1}
        assert len(data["sentiment_trend"]["dates"]) >= 1
        assert len(data["sentiment_trend"]["dates"]) == len(data["sentiment_trend"]["avg_scores"])


def test_empty_dashboard_returns_safe_values(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()
    with TestClient(main2.app) as client:
        data = client.get("/admin/stats").json()
        assert data["message_count"] == 0
        assert data["total_active_users"] == 0
        assert data["average_sentiment"] is None
        assert data["model_usage"] == {}
        assert data["sentiment_trend"] == {"dates": [], "avg_scores": []}
