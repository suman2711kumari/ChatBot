# MindSpace | Mental health support chatbot

FastAPI + JavaScript + Gemini API + TF-IDF retrieval (RAG) + SQLite/Chart.js,
with an optional local DialoGPT response fallback and optional RoBERTa
sentiment inference. This is a **non-clinical student demonstration**, not
therapy, a depression diagnosis or validated crisis-risk assessment.

## Main components

- **Generation:** Gemini API (subject to provider's pricing, quota and privacy
  terms), or optional local pretrained DialoGPT-small when Gemini fails.
- **Retrieval:** TF-IDF/cosine similarity over seven concise, curated public
  WHO/NHS/NIMH summaries in `knowledge/resources.json`, optionally
  sentence-embedding retrieval (`requirements-rag-embeddings.txt`,
  `RAG_BACKEND=semantic`). Retrieved notes are attached as optional Gemini
  context; the API has `rag_info` diagnostics and a separate admin RAG Inspector.
- **Sentiment:** VADER's compound lexicon score by default. Optional
  three-class RoBERTa with a separately defined polarity score. These do **not**
  measure mental health, depression, distress or user satisfaction.
- **Analytics:** last 24h message count, distinct users, generator usage,
  and a sentiment trend that **never averages different models together**.

## Run in VS Code (Windows)

```powershell
# If the virtual environment already exists, activate it instead:
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Only for optional local DialoGPT fallback:
python -m pip install -r requirements-local.txt
# First-time setup only: Copy-Item .env.example .env
python main2.py
```

Set `GEMINI_API_KEY` in your local `.env`, and confirm your accessible
`GEMINI_MODEL`.  Visit
http://127.0.0.1:8000/ for chat, http://127.0.0.1:8000/admin for analytics.

The admin page and stats API are **not password protected**. SQLite stores
chat messages as plaintext, and Gemini receives messages when enabled.
**Use fictional data locally**. Do not deploy for real patient use or collect
private conversations without consent/security/access controls.

## How to enable OPTIONAL RoBERTa sentiment

```powershell
python -m pip install -r requirements-roberta.txt
```

In your local `.env` file, set:

```text
SENTIMENT_BACKEND=roberta
```

Start `python main2.py`. RoBERTa runs locally on CPU with pretrained
`cardiffnlp/twitter-roberta-base-sentiment-latest`. No model training,
GPU or paid API is required for sentiment inference. The first request
downloads/loads weights (~500 MB download), and CPU inference may be slow,
especially with the optional DialoGPT fallback on an 8 GB laptop. For a
lighter default set `SENTIMENT_BACKEND=vader`.

If RoBERTa can't load/run, that message explicitly falls back to VADER:
its API output and stored `sentiment_model` indicate the actual backend.
Model errors also appear in server logs. Normal app startup does not
download RoBERTa automatically.

### Exactly how the score is calculated

The pretrained transformer encoder represents the whole message using
learned contextual features. Its classification head produces **three
logits** for `negative`, `neutral`, and `positive`. Hugging Face's
`pipeline("text-classification", top_k=None)` applies **Softmax**:

```text
P(class i) = exp(logit[i]) / sum(exp(all three logits))

sentiment_score = P(positive) - P(negative)

Illustrative probabilities:
negative=0.80, neutral=0.15, positive=0.05
sentiment_score = 0.05 - 0.80 = -0.75
```

The chosen class label is the one with maximum probability; its polarity
score lies in [-1, +1]. Near zero may also mean positive/negative
uncertainty or mixed sentiment rather than neutral wording. Model outputs
are not calibrated measures of emotion or mental health. **RoBERTa's
probability-difference score is not interchangeable with VADER's compound.**

`POST /chat` returns `sentiment_score`, `sentiment_model`,
`sentiment_label` and `sentiment_probabilities` (null for VADER). The
database stores the score, label and actual sentiment backend per message.
Existing VADER sentiment records are safely tagged VADER during schema
migration. Old `depression_score` data is left alone.

### Separate analytics in the admin dashboard

The dashboard's **Sentiment model** dropdown lets you choose which model's
records feed the average and the trend. It never averages VADER and
RoBERTa together. If there are no messages scored with the selected
model in the previous 24h, those metrics are empty. The counts of users,
messages and response engines still include *all* message records.

The API defaults to the model configured in `.env` or you can select
`GET /admin/stats?sentiment_model=vader` or
`GET /admin/stats?sentiment_model=roberta`. The additional
`sentiment_message_count` indicates how many message scores contributed.


### Compare VADER and RoBERTa (optional experiment)

```powershell
# Lightweight VADER baseline only
python evaluate_sentiment_threeclass.py --dataset tweet_eval --limit 90

# With RoBERTa (uses cached model weights if available)
python evaluate_sentiment_threeclass.py --dataset tweet_eval --models roberta --limit 90

# Chat-style, hand-authored examples
python evaluate_sentiment_threeclass.py --dataset challenges --models roberta
```

User-reported results:

| Model | TweetEval accuracy (90) | Authored chat examples (36) |
| --- | ---: | ---: |
| VADER | 55.56% | 94.44% |
| RoBERTa (three-class) | 68.89% | 100% |

The authored examples are **not** an independent benchmark. RoBERTa was
fine-tuned on TweetEval's training set, giving it an in-domain advantage
on its held-out TweetEval tests. Tweet sentiment differs from emotional
support conversations. None of these results establishes clinical
reliability or overall real-world performance.

- https://github.com/cardiffnlp/tweeteval
- https://huggingface.co/cardiffnlp/twitter-roberta-base-sentiment-latest

## RAG Inspector

The independent `/admin` page includes a RAG Inspector that previews
retrieved source summaries and similarity scores without a Gemini call
or conversation logging. The optional RAG ON/OFF comparison performs
**two Gemini API requests** when explicitly clicked, so it can consume
free API quota; only use fictional prompts. This does not prove one
answer is more correct than the other. No reference links are appended
automatically to chat bubbles.

The safety backstop recognizes some explicit self-harm phrases but is
**not a validated detector**. It is independent of the sentiment score.

## Tests

```powershell
python -m pip install pytest httpx
python -m pytest -q tests
```

The tests mock Gemini, DialoGPT and optional RoBERTa inference;
real pretrained model weights are not needed for unit tests. Verify
a real chat request, dashboard model selector and actual cached model
on your local Windows laptop before demoing to interviewers.



## Two-day interview study focus

For the sentiment classifier, only study **RoBERTa** as the transformer:
tokenization, transformer encoder and self-attention, classifier logits,
softmax, and `sentiment_score = P(positive) - P(negative)`. This repo uses
a pretrained model: **no transformer training or fine-tuning was performed**.

VADER is a fast lexicon-based comparison and an automatic fallback if
RoBERTa cannot load. You only need its basic definition and why it
can misread context, not an additional deep-learning tutorial.

Gemini generates responses; optional pretrained DialoGPT is the separate
response-generation fallback. TF-IDF is the default RAG retriever.
