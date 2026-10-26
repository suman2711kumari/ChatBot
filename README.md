# Mental Health Chatbot: Grounded RAG + Sentiment Dashboard

Student demonstration using Python FastAPI, JavaScript, Gemini API, VADER sentiment, SQLite/Chart.js, and the retained pretrained local DialoGPT fallback.

## Resume wording (unchanged)

> Developed an AI-assisted mental health chatbot using Python (FastAPI), JavaScript and the Gemini API. Applied sentiment analysis and NLP techniques to analyse user conversations.
>
> Built an admin dashboard to visualize user’s interaction and mood trends over the last 24 hours.

## What the new branch does

- **RAG:** `knowledge/resources.json` contains seven short starter summaries linking to public WHO, NHS and NIMH guidance. `rag.py` converts the query and summaries into TF-IDF vectors and retrieves up to two relevant notes using cosine similarity. The notes become optional context for Gemini. Reference links are shown in the UI. Summaries are authored excerpts/interpretations, not copies of complete source documents; independently verify content and URLs.
- **Optional semantic embeddings:** install `requirements-rag-embeddings.txt`, set `RAG_BACKEND=semantic` for pretrained `all-MiniLM-L6-v2` CPU sentence embeddings. This needs a one-time download. Without optional packages the default TF-IDF retriever works on an 8 GB laptop.
- **Primary generator:** Gemini via `google-genai`, with up to three earlier conversational turns to improve continuity.
- **Fallback preserved:** `microsoft/DialoGPT-small` is still attempted if Gemini fails. It must have dependencies and model weights installed/cached first. Since its answer does **not** use RAG context, any relevant links are labeled **optional reading** rather than claimed citations.
- **NLP + dashboard preserved:** VADER compound sentiment score (-1 to +1) is stored in SQLite. `/admin/stats` retains `total_active_users` and `sentiment_trend.dates/avg_scores`; Chart.js plots hourly average sentiment over the last 24 hours. Previous legacy `depression_score` entries are not reinterpreted as a sentiment or depression measurement.
- **Safety:** a small phrase-based emergency backstop remains. This is incomplete and not a validated crisis detector; the chatbot never diagnoses or pretends to be a therapist.

## Do I need to download a dataset?

**No dataset download is necessary.** RAG reads a local reference knowledge base, not a labeled emotion/depression training set. The starter JSON is committed. To expand it, edit `knowledge/resources.json` with reviewed sources and summaries. This project does not train a new emotion classification model or an LLM.

## Run in VS Code on Windows 11

```powershell
git fetch origin
git switch feature/grounded-rag-support-2026
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Required for your DialoGPT fallback:
python -m pip install -r requirements-local.txt
Copy-Item .env.example .env
```

Fill in your own `GEMINI_API_KEY` in `.env`. Confirm your accessible Gemini model name in [Google AI Studio](https://aistudio.google.com/) and adjust `GEMINI_MODEL` if needed. Then run `python main2.py` and open <http://127.0.0.1:8000/>.

DialoGPT requires an initial download of pretrained weights and can be slow on CPU. After caching, local generation can run offline; do not count on its conversational quality for support in emergencies.

To use optional semantic embeddings:

```powershell
python -m pip install -r requirements-rag-embeddings.txt
```

Change `RAG_BACKEND=semantic` in `.env` and allow the initial sentence-transformer download. The default `tfidf` mode is simpler and faster.

## Architecture

```
JavaScript -> POST /chat -> VADER sentiment --------> SQLite
                      -> TF-IDF retrieval -> source notes -> Gemini
                                                          | API failure
                                                          v
                                                  local DialoGPT
                      -> JSON reply, score, source links
JavaScript Chart.js -> GET /admin/stats -> 24-hour hourly AVG(sentiment_score)
```

The POST response has `reply`, `sentiment_score`, `model_used`, `references` and `suggested_reading`. When Gemini generated the response, matching official-source links appear under references, but these are **not proof** that all generated claims are supported. When fallback generates, links are labeled **optional further reading**.

## Tests

```powershell
python -m pip install pytest httpx
python -m pytest -q tests
```

Tests avoid external APIs by mocking Gemini/DialoGPT. They check retrieval, crisis fallback behavior, and 24-hour dashboard aggregation. Also test live Gemini, actual cached DialoGPT fallback, and the browser chart locally.

## Important limits

**Not therapy, diagnosis, or a clinical decision system.** VADER is lexicon/rule-based NLP, not a trained depression predictor. TF-IDF retrieval is not a supervised classifier. Gemini and DialoGPT are existing pretrained LLMs.

Do not deploy this as a real patient app. The admin dashboard lacks authentication; messages are stored in plaintext and sent to Gemini when enabled; consent, user access, deletion, encryption, safety evaluation, and stronger crisis handling would be required before real-world use. Only test with fictional messages. This repo has historical commits containing a `chatbot.db`; ignoring that file now does not erase old history. Check it separately for sensitive content.

The **main branch and existing experimental branches remain untouched**.
