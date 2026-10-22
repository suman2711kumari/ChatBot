"""Train and evaluate the supervised text model on labeled Dreaddit data."""
import json
from pathlib import Path

import joblib
from datasets import load_dataset
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    f1_score, precision_score, recall_score, roc_auc_score,
)

from nlp_model import build_stress_pipeline

MODEL_DIR = Path("models")
DATASET = "asmaab/dreaddit"


def evaluate(model, split):
    truth = list(split["label"])
    text = list(split["text"])
    predicted = model.predict(text)
    probabilities = model.predict_proba(text)[:, list(model.classes_).index(1)]
    return {
        "samples": len(truth),
        "accuracy": accuracy_score(truth, predicted),
        "precision": precision_score(truth, predicted, zero_division=0),
        "recall": recall_score(truth, predicted, zero_division=0),
        "f1": f1_score(truth, predicted, zero_division=0),
        "roc_auc": roc_auc_score(truth, probabilities),
        "confusion_matrix": confusion_matrix(
            truth, predicted, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            truth, predicted, zero_division=0
        ),
    }


def main():
    dataset = load_dataset(DATASET)
    for name in ("train", "validation", "test"):
        if name not in dataset:
            raise ValueError(f"Required dataset split missing: {name}")

    model = build_stress_pipeline()
    model.fit(dataset["train"]["text"], dataset["train"]["label"])
    results = {
        "dataset": DATASET,
        "task": "Stress-related language classification, not diagnosis",
        "training_examples": len(dataset["train"]),
        "validation": evaluate(model, dataset["validation"]),
        "test": evaluate(model, dataset["test"]),
    }
    MODEL_DIR.mkdir(exist_ok=True)
    joblib.dump(model, MODEL_DIR / "stress_pipeline.joblib")
    (MODEL_DIR / "evaluation.json").write_text(
        json.dumps(results, indent=2), encoding="utf-8"
    )
    print(json.dumps(results, indent=2))
    print("Model saved: models/stress_pipeline.joblib")


if __name__ == "__main__":
    main()
