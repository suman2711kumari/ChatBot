"""Opt-in, READ-ONLY sentiment benchmark; it never changes the running chatbot.

Default command runs only VADER. With --transformer, download and evaluate
a pretrained, sentiment-fine-tuned DistilBERT model for comparison.

Important: This small, hand-authored sample is *not* a representative or
independent clinical evaluation. Labels are illustrative text polarity only.
"""
import argparse
import json
from collections import Counter
from pathlib import Path

BENCHMARK = Path(__file__).resolve().parent / "evaluation" / "sentiment_examples.json"
DEFAULT_MODEL = "distilbert-base-uncased-finetuned-sst-2-english"


def read_examples(path: Path):
    items = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(items, list) or not items:
        raise ValueError("Sentiment examples must be a non-empty list")
    for entry in items:
        if entry.get("label") not in {"positive", "negative"} or not entry.get("text"):
            raise ValueError("Each example requires text and a positive/negative label")
    return items


def vader_predict(text, analyzer):
    """Map VADER compound to a *binary* polarity label for this benchmark.

    Neutral compound scores are deliberately classified as 'negative' only
    for a two-class comparison. Real chatbot stores raw compound unchanged.
    """
    compound = analyzer.polarity_scores(text)["compound"]
    # Nonpositive becomes the comparison's negative class, not a diagnosis.
    return ("positive" if compound > 0.05 else "negative"), float(compound)


def transformer_predict(text, classifier):
    output = classifier(text, truncation=True, max_length=256)[0]
    label = output["label"].lower()
    if label.startswith("pos") or label == "label_1":
        return "positive", float(output["score"])
    if label.startswith("neg") or label == "label_0":
        return "negative", float(output["score"])
    raise ValueError("Expected a binary positive/negative classifier, got " + label)


def evaluate(examples, predictions):
    correct = sum(row["label"] == pred for row, pred in zip(examples, predictions))
    by_label = {}
    for label in ("positive", "negative"):
        tp = sum(row["label"] == label and pred == label
                 for row, pred in zip(examples, predictions))
        fp = sum(row["label"] != label and pred == label
                 for row, pred in zip(examples, predictions))
        fn = sum(row["label"] == label and pred != label
                 for row, pred in zip(examples, predictions))
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        by_label[label] = {"precision": precision, "recall": recall, "f1": f1}
    return {"accuracy": correct / len(examples),
            "macro_f1": sum(x["f1"] for x in by_label.values()) / len(by_label),
            "by_label": by_label, "correct": correct, "total": len(examples)}


def main():
    parser = argparse.ArgumentParser(description="Compare VADER and optional DistilBERT (no app changes)")
    parser.add_argument("--transformer", action="store_true",
                        help="Download and test DistilBERT; needs transformers and torch")
    parser.add_argument("--examples", type=Path, default=BENCHMARK)
    args = parser.parse_args()
    examples = read_examples(args.examples)

    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    vader_results = [vader_predict(row["text"], analyzer) for row in examples]
    print("\nVADER (existing production backend)")
    print(json.dumps(evaluate(examples, [x[0] for x in vader_results]), indent=2))

    transformer_results = None
    if args.transformer:
        print("\nLoading pretrained DistilBERT classifier; first run downloads model weights.")
        from transformers import pipeline
        classifier = pipeline(
            "text-classification", model=DEFAULT_MODEL, device=-1
        )
        transformer_results = [transformer_predict(row["text"], classifier) for row in examples]
        print("\nDistilBERT (experimental only)")
        print(json.dumps(evaluate(examples, [x[0] for x in transformer_results]), indent=2))

    print("\nExamples (human-provided label, VADER prediction, optional DistilBERT prediction)")
    for i, row in enumerate(examples):
        print(f'{i+1:02d} {row["label"]:8} | VADER={vader_results[i][0]:8} '
              + (f'| DistilBERT={transformer_results[i][0]:8} ' if transformer_results else "")
              + "| " + row["text"])

    print("\nLIMITATIONS: These are 20 authored examples, not an independent test set.")
    print("The selected DistilBERT model learned movie-review polarity (SST-2),")
    print("so it may perform poorly on mental-health wording and has no neutral class.")
    print("The transformer confidence and VADER compound scores are NOT comparable.")
    print("Nothing here trains a model, diagnoses anyone, edits the database,")
    print("or changes the chatbot's default VADER/24-hour dashboard.")


if __name__ == "__main__":
    main()
