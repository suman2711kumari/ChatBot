# Emotional Support Chatbot: supervised NLP + FastAPI

A **non-clinical prototype** preserving the original FastAPI chat, SQLite storage, Gemini generation, optional DialoGPT fallback, and Chart.js dashboard. This change **replaces** the old hand-weighted depression formula, VADER adjustment, and manually selected emotion replies with a supervised text classifier.

## What changed

- `train_stress_model.py` trains a **TF-IDF + NRC features + Logistic Regression** text classifier on **labeled Dreaddit data**. The regression weights are learned from labeled training examples.
- Your `merged_nrc_lexicon.json` still contributes ten emotion/sentiment association frequency features. The code intentionally ignores the arbitrary `1.5` sadness multiplier: original NRC associations are binary.
- The **new** `stress_probability` comes from `predict_proba()`; the dashboard charts hourly mean predicted stress-language signal. It is **not** a validated depression score or clinical risk indicator.
- For ordinary user messages, Gemini generates a reply; if it fails or has no key, the existing pretrained DialoGPT model is attempted locally.
- The original hardcoded crisis phrase check remains only as an **incomplete safety backstop**, not as an NLP classifier.

## Install and train (Python 3.11 or 3.12 recommended)

Run these commands **from the repository root**:

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements-train.txt
python train_stress_model.py
```

The script downloads the public [Dreaddit text dataset](https://huggingface.co/datasets/asmaab/dreaddit) using Hugging Face Datasets. It fits on the **train** split, then reports validation/test precision, recall, F1, accuracy, ROC-AUC and confusion matrices. It writes:

- `models/stress_pipeline.joblib` (trusted local trained classifier)
- `models/evaluation.json` (actual evaluation results)

The model is **not** pre-trained or bundled in this repository; you must run the training command before chatting. An internet connection is needed for the initial dataset download, and a Gemini API key is optional for LLM responses. No Kaggle token or paid dataset is required.

```bash
# Copy .env.example to .env and put your key in GEMINI_API_KEY if using Gemini
python main2.py
# Open http://127.0.0.1:8000/
```

If the trained model is missing, `POST /chat` returns HTTP 503 with training instructions instead of inventing a score.

## API and database

- `POST /chat`: `{"user_id":"demo","message":"I'm overwhelmed with assignments"}` → `reply`, `stress_probability`, `model_used`.
- `GET /admin/stats`: last-hour active users; `stress_trend` contains hourly mean probabilities as percentages; model usage counts.
- `GET /health`: whether the trained artifact exists and Gemini is configured.

The existing SQLite table is migrated **non-destructively** with an added `stress_probability` column. Historical `depression_score` values are preserved but excluded from the new chart, since they came from an unrelated manual calculation.

## What the model does and doesn't do

The text is represented by (1) TF-IDF unigrams/bigrams and (2) normalized frequencies of original NRC word-emotion **associations**. Logistic Regression learns weights from a labeled Reddit stress dataset. This is **genuine supervised machine learning**, unlike the original hand-built scoring rule.

It detects **stress-related wording**, not actual psychological stress, clinical depression, suicidality, or suitability for treatment. Performance must be evaluated against independent labels; do not treat `predict_proba` as a medically calibrated risk probability. The separate `mentalchat.json` contains conversation examples but **is not used to train this classifier**.

## Limitations before production use

This is not a real-world clinical service. User messages may be sent to Google if an API key is configured. The `/admin/stats` endpoint is not authenticated, and local DialoGPT is not safety-validated. The phrase-based crisis check can miss unsafe requests or give false positives. Sensitive chat logs require consent, access control, encryption, retention limits, and deletion controls before deployment.

The old `chatbot.db` is still tracked in earlier commits: adding it to `.gitignore` **does not remove it from Git history**. Do not store real personal conversations in a public repository; audit and clean Git history separately after backing up any real data.

## Testing

```bash
python -m pip install pytest httpx
python -m pytest -q tests/
```

Tests cover NRC feature extraction, a small learned-model smoke test, API response and admin aggregation, and legacy schema migration. These are functional tests, **not** Dreaddit evaluation results.

## Attribution

- NRC Emotion Lexicon by Saif M. Mohammad and Peter Turney. Respect its [usage and redistribution terms](https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm).
- Dreaddit: Reddit stress-related language dataset. See the [original publication](https://aclanthology.org/D19-6213/).
- DialoGPT-small: Microsoft / Hugging Face Transformers. Gemini: Google Gen AI API.
