# AI-Assisted Emotional Support Chatbot (beginner-friendly)

A small, **non-clinical student project** using FastAPI, JavaScript, Gemini, VADER sentiment analysis, SQLite and Chart.js.

## Simple workflow
1. JavaScript sends a message to FastAPI using POST /chat.
2. VADER analyzes the sentence's *sentiment*, returning a compound score from -1 to +1.
3. Gemini generates a reply, with up to three earlier successful chat turns for context.
4. SQLite stores the conversation, sentiment score and timestamp.
5. The dashboard shows hourly average sentiment and distinct users over the last 24 hours.

**No machine-learning training required.** VADER uses a lexicon and rules; Gemini is a pretrained model accessed through an API. The project **does not diagnose depression**.

The original NRC preprocessing dataset `merged_nrc_lexicon.json` and `mentalchat.json` are kept in this branch as **earlier experiments but are not used in the running application**. Your original implementation remains in `main`, and the separate classifier experiment remains in `ml-supervised-nlp-replacement`.

## Run in VS Code on Windows
From this branch's folder with Python 3.11 or later:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
Copy-Item .env.example .env
```

Put your Gemini API key in `.env`, then run:

```powershell
python main2.py
```

Open http://127.0.0.1:8000/

The default `GEMINI_MODEL=gemini-3.6-flash` is currently listed by Google, but your key still needs access and quota. If Gemini fails, the app explains the outage rather than silently falling back to unreliable DialoGPT replies. Check the VS Code terminal for the underlying Gemini error. Available models can be checked in [Google AI Studio](https://aistudio.google.com/).

## Interview explanation
**How is sentiment calculated?** VADER evaluates words, negations and emphasis, and outputs `compound = S / sqrt(S² + 15)`, with a result between -1 and +1. This is **not a depression score**.

**How are dashboard trends calculated?** Each hourly point is the arithmetic mean of the VADER compound scores for the messages in that hour. Old database depression_score entries remain untouched but are not combined with sentiment.

**Did you train an ML model?** No. This version integrates Gemini for conversational generation and uses VADER for lexicon-based NLP.

**What about DialoGPT?** The earlier `main` branch uses pretrained `microsoft/DialoGPT-small`, without fine-tuning. It sometimes gives irrelevant replies. It is excluded from this simplified app. You can study it separately as an experiment, but fine-tuning it on a tiny conversation sample is not a reliable way to create a mental-health chatbot.

## Boundaries
This is a student demonstration, not therapy or a crisis service. The limited crisis phrase backstop is not a validated detector. VADER doesn't understand all context and sarcasm. The dashboard has no authentication, and Gemini receives user message text. Don't deploy with real private health messages without proper security and consent.

A `chatbot.db` existed in the original Git history. It is ignored and removed from the **current branch contents**, but not from earlier Git commits. `.gitignore` alone does not erase past committed data.
