# MindSpace | Student mental health support chatbot

FastAPI + JavaScript + Gemini API + document-based RAG (MiniLM + FAISS) + SQLite/Chart.js,
with an optional local DialoGPT response fallback and optional RoBERTa
sentiment inference. This is a **non-clinical student demonstration**, not
therapy, a depression diagnosis or validated crisis-risk assessment.

## Main components

- **Generation:** Gemini API (subject to provider's pricing, quota and privacy
  terms), or optional local pretrained DialoGPT-small when Gemini fails.
- **Retrieval:** Original WHO PDF and NIMH webpage sources are downloaded into
  `knowledge/documents/`, parsed, split into overlapping chunks with LangChain's
  `RecursiveCharacterTextSplitter`, embedded with `all-MiniLM-L6-v2`, and indexed
  in a local FAISS vector store. Relevant chunks are attached as optional Gemini
  context; the API exposes `rag_info` diagnostics and an admin RAG Inspector.
- **Sentiment:** VADER's compound lexicon score by default. Optional
  three-class RoBERTa with a separately defined polarity score. These do **not**
  measure mental health, depression, distress or user satisfaction.
- **Analytics:** last 24h message count, distinct users, generator usage,
  and a sentiment trend that **never averages different models together**.

## Run in VS Code (Windows)

```powershell
git fetch origin
git switch feature/document-chunk-rag
git pull --ff-only
# If the virtual environment already exists, activate it instead:
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
# Install parsing, chunking, MiniLM and FAISS dependencies:
python -m pip install -r requirements-rag-embeddings.txt
# Download source documents, parse, chunk, embed and build the index:
python ingest_documents.py
# Only for optional local DialoGPT fallback:
python -m pip install -r requirements-local.txt
# First-time setup only: Copy-Item .env.example .env
python main2.py
```

Set `GEMINI_API_KEY` in your local `.env`, and confirm your accessible
`GEMINI_MODEL`. Never commit your key. Visit
http://127.0.0.1:8000/ for chat, http://127.0.0.1:8000/admin for analytics.

The admin page and stats API are **not password protected**. SQLite stores
chat messages as plaintext, and Gemini receives messages when enabled.
**Use fictional data locally**. Do not deploy for real patient use or collect
private conversations without consent/security/access controls.


## Document-based RAG pipeline

The implementation follows the two-stage pattern: document ingestion first, then query-time retrieval and generation.

### Build the local knowledge base

```powershell
python -m pip install -r requirements-rag-embeddings.txt
python ingest_documents.py
```

The ingestion script downloads the seed sources listed in `knowledge/sources.json` into `knowledge/documents/`. It also accepts your own PDF, HTML, TXT and Markdown files placed in that folder.

```powershell
# Rebuild using only files already in the documents folder:
python ingest_documents.py --local-only

# Re-download the seed files and rebuild:
python ingest_documents.py --force-download
```

### What ingestion does

- **Loading and parsing:** PyMuPDF extracts PDF text page by page. BeautifulSoup extracts main HTML content and removes navigation, scripts and other non-content elements. TXT and Markdown files are read directly.
- **Document structure:** Each page/document becomes a LangChain `Document` with text and metadata such as title, publisher, source filename, URL and page number.
- **Chunking:** `RecursiveCharacterTextSplitter` uses a 1,000-character chunk size and 200-character overlap while preserving metadata.
- **Embeddings:** `sentence-transformers/all-MiniLM-L6-v2` creates a dense vector for each chunk and runs on CPU.
- **Vector storage:** FAISS `IndexFlatIP` persists normalized vectors in `knowledge/vector_store/index.faiss`. Because vectors are normalized, inner product is equivalent to cosine similarity. Text and source metadata are stored separately in `chunks.json`.

The generated vector store is ignored by Git because it can be rebuilt from the source documents. Keep document copies only when their reuse terms permit it. WHO's guide has a non-commercial Creative Commons licence; NIMH's text is public domain unless otherwise indicated. Cite both publishers and do not imply endorsement.

### Query-time retrieval

`rag.py` embeds the user query with the same MiniLM model, searches FAISS, applies `RAG_MIN_SIMILARITY` (default `0.30`) and returns up to three source chunks. `main2.py` attaches retrieved passages to Gemini's prompt. Crisis-message handling runs before retrieval. The optional DialoGPT fallback does not receive retrieved context and is not described as grounded.

The admin RAG Inspector previews actual chunk text, source metadata and similarity scores without calling Gemini or logging a conversation. Its optional ON/OFF comparison makes two Gemini API requests.

## How to enable OPTIONAL RoBERTa sentiment


```powershell
python -m pip install -r requirements-sentiment-experiment.txt
```

In your local `.env` file, set:

```text
SENTIMENT_BACKEND=roberta
```

Restart `python main2.py`. RoBERTa runs locally on CPU with pretrained
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

## What was evaluated, and what it means

```powershell
# Original binary VADER vs DistilBERT (20 illustrative authored examples):
python evaluate_sentiment.py --transformer

# Balanced official TweetEval test split, 90 messages:
python evaluate_sentiment_threeclass.py --dataset tweet_eval --models all --limit 90

# Authored student-chat stress tests, 36 examples:
python evaluate_sentiment_threeclass.py --dataset challenges --models all
```

User-reported results:

| Model | TweetEval accuracy (90) | Authored chat examples (36) |
| --- | ---: | ---: |
| VADER | 55.56% | 94.44% |
| DistilBERT (binary) | 48.89% | 66.67% |
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
retrieved source-document chunks, metadata and similarity scores without a
Gemini call or conversation logging. The optional RAG ON/OFF comparison performs
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

## Repository cleanup

Removed from the working feature branch: unused
`merged_nrc_lexicon.json`, unused `mentalchat.json` experimental
dataset, the bundled `get-pip.py` installer, and a machine-specific
`.vscode/settings.json`. Production files, optional experiments, tests,
`knowledge/sources.json`, the ingestion script and all required dependency manifests remain.

Git history is retained. The `main` branch is **not modified**.

**Accurate interview wording:** "I implemented a document-based RAG
pipeline that loads PDF and HTML sources, extracts text with PyMuPDF and
BeautifulSoup, splits documents into overlapping chunks using LangChain,
generates MiniLM embeddings, stores them in a local FAISS index and retrieves
relevant passages for Gemini. I also implemented sentiment analysis with
VADER and optional pretrained RoBERTa inference; I did not train or fine-tune
the transformer."
