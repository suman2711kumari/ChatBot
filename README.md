# MindSpace | RoBERTa + RAG Support Chatbot

Student demonstration built with Python FastAPI, JavaScript, pretrained
**RoBERTa** sentiment inference, a Gemini response generator, TF-IDF RAG,
SQLite and Chart.js. A local pretrained DialoGPT-small remains an optional
fallback for **reply generation**, not sentiment scoring.

This is NOT therapy, a mental-health diagnosis, distress scoring, or a
validated crisis detector. Use fictional example messages only. Admin
endpoints have no authentication and chat text is saved unencrypted.

## Architecture

```text
Web UI (static/chat.js) -- POST /chat --> main2.py
   |                                     |
   |                                     +--> sentiment_engine.py: RoBERTa
   |                                     |     Hugging Face + PyTorch, CPU
   |                                     |     P(pos) - P(neg) --> SQLite
   |                                     |
   |                                     +--> rag.py: TF-IDF + cosine similarity
   |                                     |     knowledge/resources.json
   |                                     |     --> reference notes for Gemini
   |                                     |
   |                                     +--> Gemini API --> conversational reply
   |                                           on API failure: local DialoGPT
   |
Admin dashboard (static/admin.js) <-- /admin/stats -- SQLite
   Only RoBERTa messages contribute to sentiment graph and average.
```

## Install & run on Windows

```powershell
git fetch origin
# After merging this branch into main:
git switch main
git pull --ff-only
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python main2.py
```

Copy `.env.example` to `.env` only if you don't already have `.env`.
Keep your real `GEMINI_API_KEY` there; it must never be committed.
Do not replace your existing `.env` or delete your SQLite database.

- Chat: http://127.0.0.1:8000/
- Admin: http://127.0.0.1:8000/admin

### RoBERTa text sentiment

Pretrained checkpoint:
`cardiffnlp/twitter-roberta-base-sentiment-latest`.
The Hugging Face text-classification pipeline runs locally on CPU after
initial model weights download (about 500 MB). It predicts probabilities
for three classes: negative, neutral and positive. Softmax converts the
classifier logits to probabilities summing approximately to 1.

```text
Custom sentiment_score = P(positive) - P(negative)
Example probabilities:
  negative: 0.80
  neutral:  0.15
  positive: 0.05
Score: 0.05 - 0.80 = -0.75
```

The label is the class with maximum probability. Scores fall in [-1,1],
but they do NOT measure emotions, depression, or health status.
These are pretrained weights: no local model training or fine-tuning.

If the model cannot be loaded or inference fails, the app still generates
a response using its independent reply engine. Its text sentiment score
is `null`, with `sentiment_status="unavailable"`. No alternative
sentiment calculation is performed, no misleading score is substituted.
An immediate-safety reply skips sentiment inference.

Old sentiment rows remain stored in SQLite and are *not deleted or
recalculated*. Their scores are excluded from the RoBERTa-only chart.
The admin page reports only the last 24 hours of RoBERTa scores.
`POST /chat` returns `sentiment_score`, `sentiment_label`,
`sentiment_model`, `sentiment_probabilities`, `sentiment_status`
and reply/RAG metadata.

## RAG

`rag.py` reads curated resource summaries from
`knowledge/resources.json`. The default uses Scikit-learn TF-IDF
unigram/bigram vectors and cosine similarity to retrieve up to two
relevant notes. The notes become optional context in the Gemini prompt
in `main2.py`. This is prompt grounding, not training Gemini.
Optional MiniLM sentence-embedding retrieval can be enabled with
`RAG_BACKEND=semantic`, requiring
`python -m pip install -r requirements-rag-embeddings.txt`.
No vector database or paid dataset is required.

## Other code

- `main2.py`: FastAPI, Gemini API, DialoGPT response fallback,
  phrase-based immediate-safety backstop, recent conversation context,
  SQLite and 24-hour analytics.
- `sentiment_engine.py`: the only sentiment model, RoBERTa.
- `rag.py` and `knowledge/resources.json`: retrieval + local notes.
- `static/chat.js`, `static/chatbot.html`: chat UI and model score.
- `static/admin.js`, `static/admin.html`: 24h trends and counts.
- `static/rag-inspector.js`: retrieval inspector and opt-in two-request
  Gemini context comparison.
- `tests/`: mocked unit and endpoint checks.

## Test

```powershell
python -m pip install pytest httpx
python -m pytest -q tests
```

Live RoBERTa inference and the Gemini API need separate manual smoke
tests; mocked unit tests cannot verify model weights or live quotas.

## Limitations

The sentiment model was fine-tuned for social-media polarity, not
counseling or clinical use. It cannot diagnose depression or reliably
detect crises. Immediate-safety phrase recognition is incomplete.
There is no administrator authentication; do not deploy this prototype
for real private conversations. The independent RAG Inspector is a
development demonstration, not a measure of clinical correctness.
