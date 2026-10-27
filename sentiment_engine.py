"""Optional three-class RoBERTa sentiment, with VADER as a safe low-RAM default.

This only analyzes TEXT POLARITY, not distress, diagnoses or crisis risk.
It never fine-tunes or trains a model. Generation and RAG are separate.
"""
import logging
import math
import os
import threading
from functools import lru_cache

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

LOGGER = logging.getLogger(__name__)
ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ("negative", "neutral", "positive")
_MODEL_LOCK = threading.Lock()


def configured_sentiment_backend():
    backend = os.getenv("SENTIMENT_BACKEND", "vader").strip().lower()
    if backend not in ("vader", "roberta"):
        LOGGER.warning("Unknown SENTIMENT_BACKEND=%r, using VADER", backend)
        return "vader"
    return backend


@lru_cache(maxsize=1)
def get_vader_analyzer():
    return SentimentIntensityAnalyzer()


@lru_cache(maxsize=1)
def get_roberta_classifier():
    """Lazy local CPU inference; dependencies and model weights are optional."""
    from transformers import pipeline
    return pipeline("text-classification", model=ROBERTA_MODEL, device=-1)


def vader_sentiment(message):
    score = float(get_vader_analyzer().polarity_scores(message)["compound"])
    label = "positive" if score >= 0.05 else (
        "negative" if score <= -0.05 else "neutral"
    )
    return {
        "sentiment_score": score,
        "sentiment_label": label,
        "sentiment_model": "vader",
        "sentiment_probabilities": None,
    }


def normalize_roberta_probabilities(outputs):
    """Decode full 3-way softmax output from Hugging Face pipeline(top_k=None)."""
    if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
        outputs = outputs[0]
    if not isinstance(outputs, list):
        raise ValueError("Expected three RoBERTa output probabilities")
    numeric_labels = {
        "label_0": "negative", "label_1": "neutral", "label_2": "positive"
    }
    probabilities = {}
    for output in outputs:
        key = str(output["label"]).strip().lower()
        label = numeric_labels.get(key, key)
        if label not in LABELS or label in probabilities:
            raise ValueError("Unknown or repeated RoBERTa label: " + label)
        probability = float(output["score"])
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid RoBERTa output probability")
        probabilities[label] = probability
    if set(probabilities) != set(LABELS):
        raise ValueError("RoBERTa must return negative, neutral and positive")
    if abs(sum(probabilities.values()) - 1.0) > 0.02:
        raise ValueError("RoBERTa output probabilities do not sum to one")
    return probabilities


def roberta_sentiment(message):
    # A lock avoids concurrent model load/inference spikes on low-RAM CPUs.
    with _MODEL_LOCK:
        outputs = get_roberta_classifier()(
            message, truncation=True, max_length=256, top_k=None
        )
    probabilities = normalize_roberta_probabilities(outputs)
    label = max(LABELS, key=lambda key: probabilities[key])
    return {
        "sentiment_score": round(
            probabilities["positive"] - probabilities["negative"], 4
        ),
        "sentiment_label": label,
        "sentiment_model": "roberta",
        "sentiment_probabilities": {
            label: round(probabilities[label], 4) for label in LABELS
        },
    }


def score_sentiment(message):
    """Always report the backend ACTUALLY used on this message."""
    if configured_sentiment_backend() == "roberta":
        try:
            return roberta_sentiment(message)
        except Exception:
            LOGGER.exception(
                "RoBERTa sentiment unavailable; using VADER for this message. "
                "Check local transformers/torch dependencies and CPU memory."
            )
    return vader_sentiment(message)
