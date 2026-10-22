"""FastAPI chatbot using a supervised stress-language classifier.

The ML output concerns wording in a text, not diagnosis or clinical risk.
Run `python train_stress_model.py` before serving chat requests.
"""
import logging
import os
import sqlite3
from contextlib import closing
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

import joblib
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from nlp_model import predict_stress

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DB_PATH = Path("chatbot.db")
MODEL_PATH = Path("models/stress_pipeline.joblib")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.mount("/static", StaticFiles(directory="static"), name="static")


def init_db():
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                reply TEXT,
                stress_probability REAL,
                model_used TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)
        # Keep any existing depression_score values as legacy data only.
        # Never reinterpret historical manual scores as ML predictions.
        columns = {row[1] for row in conn.execute("PRAGMA table_info(messages)")}
        for column, sql_type in (("stress_probability", "REAL"), ("model_used", "TEXT")):
            if column not in columns:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {column} {sql_type}")


init_db()


@lru_cache(maxsize=1)
def get_stress_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            "Model is not trained. Run 'python train_stress_model.py' first."
        )
    # Only load trusted, locally trained model artifacts; joblib uses pickle.
    return joblib.load(MODEL_PATH)


# This pre-existing safety backstop is not used to compute the ML score.
# Phrase matching is incomplete and NOT a clinically validated crisis detector.
CRITICAL_PHRASES = (
    "i want to die", "i want to kill myself",
    "i am going to hurt myself", "i can't go on", "i feel hopeless",
)
HELPLINE_MESSAGE = (
    "I'm concerned about your safety. Please contact a trusted person, "
    "a qualified mental-health professional, or local emergency services "
    "if you may be in immediate danger."
)


def check_critical_message(message):
    lower = message.casefold()
    return any(phrase in lower for phrase in CRITICAL_PHRASES)


@lru_cache(maxsize=1)
def get_local_model():
    """Load pretrained DialoGPT only when Gemini is not available."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(name)
    model = AutoModelForCausalLM.from_pretrained(name)
    model.eval()
    return tokenizer, model


chat_histories = {}


def generate_local_reply(user_id, message):
    try:
        import torch
        tokenizer, model = get_local_model()
        incoming = tokenizer.encode(message + tokenizer.eos_token, return_tensors="pt")
        previous = chat_histories.get(user_id)
        context = torch.cat((previous, incoming), dim=-1) if previous is not None else incoming
        context = context[:, -800:]
        with torch.inference_mode():
            output = model.generate(
                context, max_new_tokens=100, do_sample=True,
                top_p=0.9, temperature=0.7,
                pad_token_id=tokenizer.eos_token_id,
            )
        chat_histories[user_id] = output[:, -800:]
        reply = tokenizer.decode(
            output[0, context.shape[-1]:], skip_special_tokens=True
        ).strip()
        return reply or "I'm here to listen. Would you like to tell me more?"
    except Exception:
        logger.exception("Local fallback failed")
        return "I'm unable to respond right now. Please try again later."


async def generate_reply(user_id, message, stress_label):
    if check_critical_message(message):
        return HELPLINE_MESSAGE, "crisis_response"

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            client = genai.Client(api_key=api_key)
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=(
                    "User message:\n" + message + "\n\n"
                    "An experimental text classifier categorized this wording as "
                    + ("stress-related" if stress_label else "not stress-related")
                    + ". This is not a diagnosis; do not assume how the user feels. "
                    "Respond naturally to the user's own words."
                ),
                config={"system_instruction": (
                    "You are a supportive, non-clinical conversational assistant. "
                    "Respond empathetically and concisely. Do not diagnose or "
                    "suggest that an ML label measures someone's mental-health status. "
                    "For immediate danger, encourage real-world help."
                )},
            )
            reply = (response.text or "").strip()
            if reply:
                return reply, "gemini"
        except Exception:
            logger.exception("Gemini failed; trying local fallback")

    return generate_local_reply(user_id, message), "dialogpt_fallback"


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=5000)


@app.post("/chat")
async def chat_endpoint(chat_req: ChatRequest):
    message = chat_req.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message must not be blank")
    try:
        prediction = predict_stress(get_stress_model(), message)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    reply, model_used = await generate_reply(chat_req.user_id, message,
                                             prediction["label"])
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        conn.execute("""
            INSERT INTO messages
            (user_id, message, reply, stress_probability, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chat_req.user_id, message, reply, prediction["probability"],
            model_used, datetime.now(timezone.utc).isoformat()
        ))
    return {
        "reply": reply,
        "stress_probability": round(prediction["probability"], 4),
        "model_used": model_used,
    }


@app.get("/admin/stats")
async def admin_stats():
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    since_hour = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    with closing(sqlite3.connect(DB_PATH)) as conn, conn:
        rows = conn.execute("""
            SELECT date(timestamp), strftime('%H:00', timestamp),
                   AVG(stress_probability)
            FROM messages
            WHERE timestamp >= ? AND stress_probability IS NOT NULL
            GROUP BY date(timestamp), strftime('%H:00', timestamp)
            ORDER BY date(timestamp), strftime('%H:00', timestamp)
        """, (since,)).fetchall()
        users = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM messages
            WHERE timestamp >= ?
        """, (since_hour,)).fetchone()[0]
        model_stats = dict(conn.execute("""
            SELECT model_used, COUNT(*) FROM messages
            WHERE timestamp >= ? AND model_used IS NOT NULL
            GROUP BY model_used
        """, (since,)).fetchall())

    return {
        "total_active_users": users,
        "stress_trend": {
            "dates": [f"{date} {hour}" for date, hour, _ in rows],
            "avg_scores": [round(avg * 100, 2) for _, _, avg in rows],
        },
        "model_usage": model_stats,
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if MODEL_PATH.is_file() else "setup_required",
        "ml_model": "trained" if MODEL_PATH.is_file() else "not_trained",
        "gemini_api": "configured" if os.getenv("GEMINI_API_KEY") else "not_configured",
    }


@app.get("/")
async def root():
    index_path = Path("static/chatbot.html")
    if index_path.exists():
        return FileResponse(str(index_path))
    return {"message": "Chatbot API is running. Frontend not found."}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)
