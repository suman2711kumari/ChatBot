"""Supervised stress-language model with TF-IDF and NRC features.

The NRC annotations are inputs to the model, not manually weighted distress
scores. The classifier learns its coefficients from labeled examples.
"""
import json
import re

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

EMOTIONS = (
    "anger", "anticipation", "disgust", "fear", "joy",
    "sadness", "surprise", "trust", "positive", "negative",
)


class NRCFeatures(BaseEstimator, TransformerMixin):
    """Normalized binary NRC word-emotion association frequencies."""

    def __init__(self, lexicon_path="merged_nrc_lexicon.json"):
        self.lexicon_path = lexicon_path

    def fit(self, X, y=None):
        with open(self.lexicon_path, encoding="utf-8") as handle:
            lexicon = json.load(handle)
        # Deliberately ignore the old, arbitrarily increased sadness value.
        self.word_labels_ = {
            word.lower(): frozenset(labels)
            for word, labels in lexicon.items()
        }
        return self

    def transform(self, X):
        if not hasattr(self, "word_labels_"):
            raise RuntimeError("NRCFeatures must be fitted before transform")
        rows = []
        for text in X:
            words = re.findall(r"[a-z]+", str(text).lower())
            counts = np.zeros(len(EMOTIONS), dtype=np.float64)
            for word in words:
                associated = self.word_labels_.get(word, ())
                for idx, label in enumerate(EMOTIONS):
                    if label in associated:
                        counts[idx] += 1
            rows.append(counts / max(len(words), 1))
        return csr_matrix(np.asarray(rows).reshape(-1, len(EMOTIONS)))

    def get_feature_names_out(self, input_features=None):
        return np.asarray(["nrc_" + label for label in EMOTIONS], dtype=object)


def build_stress_pipeline(lexicon_path="merged_nrc_lexicon.json"):
    features = FeatureUnion([
        ("text", TfidfVectorizer(
            ngram_range=(1, 2), min_df=2,
            max_features=20000, sublinear_tf=True,
        )),
        ("nrc", Pipeline([
            ("extract", NRCFeatures(lexicon_path)),
            ("scale", StandardScaler(with_mean=False)),
        ])),
    ])
    return Pipeline([
        ("features", features),
        ("classifier", LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=42
        )),
    ])


def predict_stress(model, text):
    """Prediction of stress-related wording, not clinical risk."""
    classes = list(model.classes_)
    if 1 not in classes:
        raise ValueError("Trained classifier must have positive class 1")
    probability = float(model.predict_proba([text])[0, classes.index(1)])
    prediction = int(model.predict([text])[0])
    return {"label": prediction, "probability": probability}
