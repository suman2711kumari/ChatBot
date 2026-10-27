"""RoBERTa-only pretrained sentiment inference for text polarity.

Transformer inference runs locally on CPU. No alternative sentiment scorer
is used on errors. These outputs are not mental-health assessments.
"""
import math
import threading
from functools import lru_cache

ROBERTA_MODEL = "cardiffnlp/twitter-roberta-base-sentiment-latest"
LABELS = ("negative", "neutral", "positive")
_MODEL_LOCK = threading.Lock()


@lru_cache(maxsize=1)
def get_roberta_classifier():
    """Lazy-load RoBERTa from the local Hugging Face model cache."""
    from transformers import pipeline
    return pipeline("text-classification", model=ROBERTA_MODEL, device=-1)


def normalize_roberta_probabilities(outputs):
    """Read all three probabilities after the classifier's Softmax."""
    if isinstance(outputs, list) and len(outputs) == 1 and isinstance(outputs[0], list):
        outputs = outputs[0]
    if not isinstance(outputs, list):
        raise ValueError("RoBERTa output must contain three class probabilities")
    label_map = {"label_0": "negative", "label_1": "neutral",
                 "label_2": "positive"}
    probabilities = {}
    for item in outputs:
        raw = str(item["label"]).strip().lower()
        label = label_map.get(raw, raw)
        if label not in LABELS or label in probabilities:
            raise ValueError("Unexpected or repeated RoBERTa label: " + label)
        score = float(item["score"])
        if not math.isfinite(score) or not 0 <= score <= 1:
            raise ValueError("Invalid RoBERTa probability")
        probabilities[label] = score
    if set(probabilities) != set(LABELS):
        raise ValueError("Expected negative, neutral and positive probabilities")
    if abs(sum(probabilities.values()) - 1.0) > 0.02:
        raise ValueError("RoBERTa probabilities do not sum to 1")
    return probabilities


def score_sentiment(message):
    """Sentiment score = P(positive) - P(negative), within [-1, 1].

    Raises on model errors rather than substituting another algorithm.
    """
    with _MODEL_LOCK:
        outputs = get_roberta_classifier()(
            message, truncation=True, max_length=256, top_k=None
        )
    probabilities = normalize_roberta_probabilities(outputs)
    return {
        "sentiment_score": round(probabilities["positive"] - probabilities["negative"], 4),
        "sentiment_label": max(LABELS, key=lambda key: probabilities[key]),
        "sentiment_model": "roberta",
        "sentiment_probabilities": {
            label: round(probabilities[label], 4) for label in LABELS
        },
    }
