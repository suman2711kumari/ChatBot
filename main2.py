"""Simple FastAPI chatbot: VADER sentiment, Gemini replies and SQLite dashboard.

Sentiment analysis is lexicon-based NLP, not depression prediction.
"""
import logging
from functools import lru_cache
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")
DB_PATH = BASE_DIR / "chatbot.db"
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
logger = logging.getLogger(__name__)

app = FastAPI()
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
analyzer = SentimentIntensityAnalyzer()


def init_db():
    """Create database or add sentiment columns without deleting old records."""
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT,
                message TEXT,
                reply TEXT,
                sentiment_score REAL,
                model_used TEXT,
                timestamp TEXT DEFAULT CURRENT_TIMESTAMP
            )
        """)
        columns = {column[1] for column in conn.execute("PRAGMA table_info(messages)")}
        for name, type_ in (("sentiment_score", "REAL"), ("model_used", "TEXT")):
            if name not in columns:
                conn.execute(f"ALTER TABLE messages ADD COLUMN {name} {type_}")


init_db()


class ChatRequest(BaseModel):
    user_id: str = Field(min_length=1, max_length=120)
    message: str = Field(min_length=1, max_length=2000)


# Minimal emergency backstop; it cannot reliably detect every crisis.
CRISIS_PHRASES = (
    "i want to kill myself", "i am going to kill myself", "i want to die",
    "i'm going to hurt myself", "i am going to hurt myself", "end my life",
)
CRISIS_REPLY = (
    "I'm sorry you're going through this. If you might hurt yourself, "
    "please contact local emergency services or a trusted person right now. "
    "A qualified mental-health professional can also help."
)


def recent_chat_context(user_id):
    """Use up to three previous conversational turns for context."""
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT message, reply FROM messages
            WHERE user_id = ? AND model_used IN ('gemini', 'dialogpt_fallback')
            ORDER BY id DESC LIMIT 3
        """, (user_id,)).fetchall()
    return list(reversed(rows))


@lru_cache(maxsize=1)
def get_local_dialogpt():
    """Load the model only if Gemini fails; cache it for later replies.

    The first use downloads weights unless they are already cached.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def generate_local_reply(message, previous_turns):
    """Pretrained DialoGPT inference; no fine-tuning or clinical guarantees."""
    import torch

    tokenizer, model = get_local_dialogpt()
    eos = tokenizer.eos_token
    # Build a little dialogue context from the last few stored turns.
    context = "".join(
        f"{user}{eos}{assistant}{eos}"
        for user, assistant in previous_turns
    ) + f"{message}{eos}"
    input_ids = tokenizer.encode(context, return_tensors="pt")[:, -600:]
    with torch.inference_mode():
        result = model.generate(
            input_ids,
            max_new_tokens=70,
            do_sample=True,
            top_p=0.85,
            temperature=0.7,
            no_repeat_ngram_size=3,
            pad_token_id=tokenizer.eos_token_id,
        )
    response_ids = result[0, input_ids.shape[-1]:]
    return tokenizer.decode(response_ids, skip_special_tokens=True).strip()


async def local_fallback(message, previous_turns):
    """Return an explicitly labelled offline response, or clear failure."""
    enabled = os.getenv("ENABLE_DIALOGPT_FALLBACK", "true").strip().lower()
    if enabled in ("false", "0", "no", "off"):
        return (
            "Gemini is unavailable. Local DialoGPT fallback is disabled.",
            "unavailable",
        )
    try:
        # Model inference can be slow on a laptop; don't block FastAPI's event loop.
        reply = await run_in_threadpool(
            generate_local_reply, message, previous_turns
        )
        if reply:
            return reply, "dialogpt_fallback"
    except Exception:
        logger.exception(
            "DialoGPT fallback failed. Install requirements-local.txt "
            "and download model weights while online."
        )
    return (
        "Neither Gemini nor the local DialoGPT model is available right now. "
        "Check the terminal for details.",
        "unavailable",
    )


async def generate_reply(message, previous_turns):
    """Gemini first, then optional local DialoGPT when Gemini cannot respond."""
    if any(phrase in message.casefold() for phrase in CRISIS_PHRASES):
        return CRISIS_REPLY, "safety_message"

    api_key = os.getenv("GEMINI_API_KEY")
    if api_key:
        try:
            from google import genai
            from google.genai import types
            client = genai.Client(api_key=api_key)
            history = "".join(
                f"User: {user}\nAssistant: {assistant}\n"
                for user, assistant in previous_turns
            )
            prompt = (
                "Previous conversation (if any):\n" + history
                + "\nLatest user message:\n" + message
            )
            response = await client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=prompt,
                config=types.GenerateContentConfig(
                    system_instruction=(
                        "You are a supportive, non-clinical chatbot for a student demo. "
                        "Reply naturally and briefly to the latest user message. "
                        "Use previous messages only for context. Do not diagnose depression, "
                        "give medical advice, or describe sentiment as a health assessment. "
                        "For immediate danger, encourage emergency or trusted-person help."
                    ),
                ),
            )
            if response.text and response.text.strip():
                return response.text.strip(), "gemini"
            raise RuntimeError("Gemini returned an empty response")
        except Exception:
            logger.exception("Gemini request failed. Trying local DialoGPT.")
    else:
        logger.info("No Gemini key found; using local fallback if available.")

    return await local_fallback(message, previous_turns)

@app.post("/chat")
async def chat(chat_request: ChatRequest):
    message = chat_request.message.strip()
    if not message:
        raise HTTPException(status_code=422, detail="Message cannot be blank")

    # Compound sentiment is in [-1, 1]. It is NOT a depression score.
    sentiment_score = analyzer.polarity_scores(message)["compound"]
    previous_turns = recent_chat_context(chat_request.user_id)
    reply, model_used = await generate_reply(message, previous_turns)

    with sqlite3.connect(DB_PATH) as conn:
        conn.execute("""
            INSERT INTO messages
            (user_id, message, reply, sentiment_score, model_used, timestamp)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            chat_request.user_id, message, reply, sentiment_score, model_used,
            datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S"),
        ))
    return {
        "reply": reply,
        "sentiment_score": sentiment_score,
        "model_used": model_used,
    }


@app.get("/admin/stats")
def admin_stats():
    """Hourly average sentiment and active users from the last 24 hours."""
    since = (datetime.now(timezone.utc) - timedelta(hours=24)).strftime(
        "%Y-%m-%d %H:%M:%S"
    )
    with sqlite3.connect(DB_PATH) as conn:
        rows = conn.execute("""
            SELECT strftime('%Y-%m-%d %H:00', timestamp), AVG(sentiment_score)
            FROM messages
            WHERE datetime(timestamp) >= datetime(?)
              AND sentiment_score IS NOT NULL
            GROUP BY strftime('%Y-%m-%d %H:00', timestamp)
            ORDER BY strftime('%Y-%m-%d %H:00', timestamp)
        """, (since,)).fetchall()
        active_users = conn.execute("""
            SELECT COUNT(DISTINCT user_id) FROM messages
            WHERE datetime(timestamp) >= datetime(?)
        """, (since,)).fetchone()[0]
    return {
        "total_active_users": active_users,
        "sentiment_trend": {
            "dates": [hour for hour, _ in rows],
            "avg_scores": [round(score, 3) for _, score in rows],
        },
    }


@app.get("/")
def homepage():
    return FileResponse(BASE_DIR / "static" / "chatbot.html")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main2:app", host="127.0.0.1", port=8000, reload=True)
