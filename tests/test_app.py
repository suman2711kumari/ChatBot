"""API smoke tests with a mocked Gemini response; no network/API key needed."""
import sqlite3
from fastapi.testclient import TestClient
import main2


def test_chat_and_dashboard(tmp_path, monkeypatch):
    monkeypatch.setattr(main2, "DB_PATH", tmp_path / "test.db")
    main2.init_db()

    async def fake_reply(message, previous):
        return "Thanks for sharing.", "gemini"

    monkeypatch.setattr(main2, "generate_reply", fake_reply)
    with TestClient(main2.app) as client:
        answer = client.post("/chat", json={
            "user_id": "student", "message": "I am happy!"
        })
        assert answer.status_code == 200
        assert answer.json()["sentiment_score"] > 0
        assert "depression_score" not in answer.json()
        stats = client.get("/admin/stats").json()
        assert stats["total_active_users"] == 1
        assert len(stats["sentiment_trend"]["dates"]) == 1


def test_legacy_table_migration(tmp_path, monkeypatch):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("""CREATE TABLE messages (
            id INTEGER PRIMARY KEY, user_id TEXT, message TEXT,
            reply TEXT, depression_score REAL, timestamp TEXT
        )""")
        conn.execute("""INSERT INTO messages VALUES (
            1, 'user', 'old', 'old reply', 4.2, '2025-01-01 00:00:00'
        )""")
    monkeypatch.setattr(main2, "DB_PATH", path)
    main2.init_db()
    with sqlite3.connect(path) as conn:
        old_score, new_score = conn.execute(
            "SELECT depression_score, sentiment_score FROM messages"
        ).fetchone()
        assert old_score == 4.2
        assert new_score is None


def test_dialogpt_fallback_when_no_gemini_key(monkeypatch):
    import asyncio
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_DIALOGPT_FALLBACK", "true")
    monkeypatch.setattr(
        main2, "generate_local_reply",
        lambda message, previous: "A local demo answer."
    )
    reply, source = asyncio.run(main2.generate_reply("How was your day?", []))
    assert source == "dialogpt_fallback"
    assert reply == "A local demo answer."


def test_safe_message_before_fallback(monkeypatch):
    import asyncio
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_DIALOGPT_FALLBACK", "true")
    reply, source = asyncio.run(main2.generate_reply("I want to die", []))
    assert source == "safety_message"
    assert "emergency" in reply.lower()


def test_no_fallback_if_user_disables_it(monkeypatch):
    import asyncio
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("ENABLE_DIALOGPT_FALLBACK", "false")
    reply, source = asyncio.run(main2.generate_reply("Hello", []))
    assert source == "unavailable"
    assert "disabled" in reply.lower()
