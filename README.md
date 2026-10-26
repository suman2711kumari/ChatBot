# Mental Health Chatbot: Grounded RAG + Sentiment Dashboard

A simple demonstration using Python FastAPI, JavaScript, Gemini API, VADER sentiment, SQLite/Chart.js, and the retained pretrained local DialoGPT fallback.

## Resume wording (unchanged)

> Developed an AI-assisted mental health chatbot using Python (FastAPI), JavaScript and the Gemini API. Applied sentiment analysis and NLP techniques to analyse user conversations.
>
> Built an admin dashboard to visualize user’s interaction and mood trends over the last 24 hours.

## What the new branch does

## Chat readability and safety update

- Sentiment scores refer to **the user's incoming text**; they are attached to that user's bubble only after `POST /chat` completes. A VADER score measures wording, not a person's emotions, and may misread context.
- Automatic resource and suggested-reading link panels were removed from the chat UI. **RAG retrieval still runs** and its reference notes still reach Gemini. The API continues returning `references` and `suggested_reading` for compatibility, but the browser does not render them. Gemini is instructed not to add source lists unless asked.
- Explicit first-person suicidal statements such as "I want to suicide now" now trigger the immediate-safety response rather than Gemini or DialoGPT. This small pattern matcher is incomplete and is **not validated crisis detection**.
- SQLite storage, the independent `/admin` page, the 24-hour Chart.js sentiment trend, and the local DialoGPT fallback are preserved.

- **RAG:** `knowledge/resources.json` contains seven short starter summaries linking to public WHO, NHS and NIMH guidance. `rag.py` converts the query and summaries into TF-IDF vectors and retrieves up to two relevant notes using cosine similarity. The notes become optional context for Gemini. Reference links are shown in the UI. Summaries are authored excerpts/interpretations, not copies of complete source documents; independently verify content and URLs.
- **Optional semantic embeddings:** install `requirements-rag-embeddings.txt`, set `RAG_BACKEND=semantic` for pretrained `all-MiniLM-L6-v2` CPU sentence embeddings. This needs a one-time download. Without optional packages the default TF-IDF retriever works on an 8 GB laptop.
- **Primary generator:** Gemini via `google-genai`, with up to three earlier conversational turns to improve continuity.
- **Fallback preserved:** `microsoft/DialoGPT-small` is still attempted if Gemini fails. It must have dependencies and model weights installed/cached first. Since its answer does **not** use RAG context, any relevant links are labeled **optional reading** rather than claimed citations.
- **NLP + dashboard preserved:** VADER compound sentiment score (-1 to +1) is stored in SQLite. `/admin/stats` retains `total_active_users` and `sentiment_trend.dates/avg_scores`; Chart.js plots hourly average sentiment over the last 24 hours. Previous legacy `depression_score` entries are not reinterpreted as a sentiment or depression measurement.
- **Safety:** a small phrase-based emergency backstop remains. This is incomplete and not a validated crisis detector; the chatbot never diagnoses or pretends to be a therapist.

## Do I need to download a dataset?

**No dataset download is necessary.** RAG reads a local reference knowledge base, not a labeled emotion/depression training set. The starter JSON is committed. To expand it, edit `knowledge/resources.json` with reviewed sources and summaries. This project does not train a new emotion classification model or an LLM.

## Updated frontend: two separate pages

The responsive interface now uses a calm two-panel chat layout, suggested prompts,
conversation bubbles, message composer with Enter/Shift+Enter handling, visible response
engine labels and optional reference links.

- **Chat:** `http://127.0.0.1:8000/`
- **Admin dashboard:** `http://127.0.0.1:8000/admin`

The admin page displays a **24-hour** sentiment trend, distinct active users, total
messages, mean VADER score and generator usage. It still calls `GET /admin/stats` and
retains `total_active_users` and `sentiment_trend`. The analytics endpoint adds
`message_count`, `average_sentiment` and `model_usage`.

Files: `static/chatbot.html`, `static/admin.html`, `static/styles.css`,
`static/chat.js` and `static/admin.js`. No frontend npm dependencies required.
The chart uses Chart.js from a CDN, so the chart library itself requires
internet access unless you choose to serve it locally.

**Security:** The admin page and statistics endpoint remain **public in this student
demo**, not authenticated. A separate page is *not* access control. Do not deploy
it publicly or store real private conversations until authorization, privacy and
data-retention controls are implemented.

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

## Demo how RAG works (without cluttering the chat)

The separate `/admin` page now has a **RAG Inspector**. For a *fictional*
question such as "How can I manage exam stress?" the inspector calls
`POST /admin/rag-preview` and shows the actual retrieved summaries and
cosine similarity scores. **The preview doesn't call Gemini or log queries.**

The optional **Compare ON vs OFF** button performs exactly two Gemini requests
when clicked: one without knowledge-base context and one with the retrieved
summaries. It will fail clearly if there are no relevant notes or no Gemini API key.
The comparison sends the typed text to Gemini, so use fictional demonstrations.
Different responses are not by themselves evidence of higher quality; a proper
evaluation needs human-labeled references.

The `/chat` API also includes `rag_info.retrieved_count`,
`rag_info.retrieval_method`, and
`rag_info.context_attached_to_gemini`. This records whether the application
supplied relevant notes on the successful Gemini route, **not** whether Gemini
actually relied on them. These diagnostics are not displayed in the chat bubbles.

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
