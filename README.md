# AI-Assisted Emotional Support Chatbot (beginner-friendly)

A small, **non-clinical student project** using FastAPI, JavaScript, Gemini, VADER sentiment analysis, SQLite and Chart.js.

## Simple workflow
1. JavaScript sends a message to FastAPI using POST /chat.
2. VADER analyzes the sentence's *sentiment*, returning a compound score from -1 to +1.
3. Gemini generates a reply when available. If its API fails, a locally cached **pretrained DialoGPT-small** model can produce a labelled offline fallback (quality is limited). The last three conversational turns provide context.
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
# Optional, for local DialoGPT fallback (larger download):
python -m pip install -r requirements-local.txt
Copy-Item .env.example .env
```

Put your Gemini API key in `.env`, then run:

```powershell
python main2.py
```

Open http://127.0.0.1:8000/

The default `GEMINI_MODEL=gemini-3.6-flash` is currently listed by Google, but your key still needs access and quota. If Gemini fails, the app **tries the locally cached DialoGPT-small** model when `ENABLE_DIALOGPT_FALLBACK=true`. The browser explicitly labels replies from that model. If the optional dependencies or weights are unavailable, you get a clear unavailable-service message instead. Check the VS Code terminal for the underlying Gemini error. Available models can be checked in [Google AI Studio](https://aistudio.google.com/).

## Interview explanation
**How is sentiment calculated?** VADER evaluates words, negations and emphasis, and outputs `compound = S / sqrt(S² + 15)`, with a result between -1 and +1. This is **not a depression score**.

**How are dashboard trends calculated?** Each hourly point is the arithmetic mean of the VADER compound scores for the messages in that hour. Old database depression_score entries remain untouched but are not combined with sentiment.

**Did you train an ML model?** No. This version integrates Gemini for conversational generation and uses VADER for lexicon-based NLP.

**What about DialoGPT?** The app keeps pretrained `microsoft/DialoGPT-small` as an *optional fallback*, not an independently trained mental-health model. It is not fine-tuned and may produce irrelevant or inappropriate replies. Install `requirements-local.txt` while connected to the internet; the first actual fallback also downloads model weights. Once cached, the fallback can run without a Gemini key or network. If memory is tight, set `ENABLE_DIALOGPT_FALLBACK=false` in `.env`.

## Boundaries
The phrase-based safety check is limited and can miss high-risk messages. DialoGPT is **not a safe or validated mental-health counselor**, and its offline replies may be misleading even when correctly generated. Do not rely on it for crisis advice. If Gemini is unavailable, use this only as a clearly-labelled student demo.
This is a student demonstration, not therapy or a crisis service. The limited crisis phrase backstop is not a validated detector. VADER doesn't understand all context and sarcasm. The dashboard has no authentication, and Gemini receives user message text. Don't deploy with real private health messages without proper security and consent.

A `chatbot.db` existed in the original Git history. It is ignored and removed from the **current branch contents**, but not from earlier Git commits. `.gitignore` alone does not erase past committed data.
