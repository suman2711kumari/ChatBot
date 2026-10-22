import sqlite3

import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def backend(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    (tmp_path / "static").mkdir()
    import main2
    main2.DB_PATH = tmp_path / "test.db"
    main2.init_db()

    class FakeModel:
        classes_ = [0, 1]

        def predict(self, x):
            return [1]

        def predict_proba(self, x):
            return np.asarray([[0.3, 0.7]])

    monkeypatch.setattr(main2, "get_stress_model", lambda: FakeModel())

    async def mock_reply(user_id, message, stress_label):
        return "I hear you.", "test_model"

    monkeypatch.setattr(main2, "generate_reply", mock_reply)
    return main2


def test_chat_and_dashboard(backend):
    with TestClient(backend.app) as client:
        result = client.post("/chat", json={
            "user_id": "u1", "message": "Feeling overwhelmed"
        })
        assert result.status_code == 200
        assert result.json()["stress_probability"] == 0.7
        assert "depression_score" not in result.json()
        stats = client.get("/admin/stats").json()
        assert stats["total_active_users"] == 1
        assert stats["stress_trend"]["avg_scores"] == [70.0]
        assert stats["model_usage"]["test_model"] == 1


def test_migrates_existing_database(backend, tmp_path):
    path = tmp_path / "legacy.db"
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, user_id TEXT, message TEXT, reply TEXT, depression_score REAL, timestamp DATETIME)")
        conn.execute("INSERT INTO messages VALUES (1,'u','a','b',9.8,'2025-01-01')")
    backend.DB_PATH = path
    backend.init_db()
    with sqlite3.connect(path) as conn:
        stored = conn.execute(
            "SELECT depression_score,stress_probability,model_used FROM messages"
        ).fetchone()
        assert stored == (9.8, None, None)
