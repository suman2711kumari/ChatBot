"""Compare three sentiment approaches without changing the chatbot.

- VADER uses standard -0.05 / +0.05 cutoffs for three labels.
- DistilBERT is a binary SST-2 classifier (cannot predict neutral).
- RoBERTa predicts negative / neutral / positive.
- Official TweetEval test data is downloaded to a local ignored cache on demand.
- The optional authored challenge set is separate and NOT an independent benchmark.

No training, Gemini API calls, SQLite writes, clinical prediction, or app changes.
"""
import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parent
CHALLENGE_FILE = ROOT / "evaluation" / "sentiment_challenges_3class.json"
CACHE = ROOT / "evaluation" / "cache" / "tweeteval" / "sentiment"
TWEETEVAL_URL = "https://raw.githubusercontent.com/cardiffnlp/tweeteval/main/datasets/sentiment/"
LABELS = ("negative", "neutral", "positive")
LABEL_IDS = {"0": "negative", "1": "neutral", "2": "positive"}
MODEL_NAMES = {
    "distilbert": "distilbert-base-uncased-finetuned-sst-2-english",
    "roberta": "cardiffnlp/twitter-roberta-base-sentiment-latest",
}


def read_challenges(path=CHALLENGE_FILE):
    """Authored stress-test examples, with subjective (not clinical) labels."""
    examples = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(examples, list) or not examples:
        raise ValueError("Expected a nonempty list of challenge messages")
    for item in examples:
        if not isinstance(item, dict) or not isinstance(item.get("text"), str):
            raise ValueError("Each challenge requires a text string")
        if not item["text"].strip() or item.get("label") not in LABELS:
            raise ValueError("Invalid challenge text/label")
    return examples


def ensure_tweeteval(cache_dir=CACHE, opener=urlopen):
    """Download official TweetEval sentiment test files, caching them locally."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in ("test_text.txt", "test_labels.txt"):
        path = cache_dir / name
        if not path.exists():
            url = TWEETEVAL_URL + name
            try:
                with opener(url, timeout=40) as response:
                    data = response.read()
            except (OSError, URLError) as exc:
                raise RuntimeError(
                    "Couldn't download TweetEval. Connect to the internet once "
                    "or retry later. Previously downloaded files work offline."
                ) from exc
            if not data or len(data) > 15_000_000:
                raise ValueError("TweetEval file is missing or unexpectedly large: " + name)
            temporary = path.with_suffix(path.suffix + ".tmp")
            temporary.write_bytes(data)
            temporary.replace(path)
        paths.append(path)
    return paths[0], paths[1]


def load_tweeteval(text_path, labels_path):
    """Read aligned official text / label files; no manual labels added."""
    texts = Path(text_path).read_text(encoding="utf-8").splitlines()
    labels = Path(labels_path).read_text(encoding="utf-8").splitlines()
    if len(texts) != len(labels) or not texts:
        raise ValueError("TweetEval text/label counts do not match")
    examples = []
    for i, (text, raw) in enumerate(zip(texts, labels)):
        if raw.strip() not in LABEL_IDS:
            raise ValueError(f"Invalid TweetEval label on row {i + 1}: {raw!r}")
        if text.strip():
            examples.append({"text": text, "label": LABEL_IDS[raw.strip()],
                             "source": f"tweet_eval_test_{i + 1}"})
    return examples


def balanced_sample(examples, limit=90, seed=42):
    """Select equal count from all classes in a repeatable, shuffled sample."""
    if limit < 3 or limit % 3:
        raise ValueError("--limit must be a multiple of 3 and at least 3")
    groups = {label: [] for label in LABELS}
    for item in examples:
        if item["label"] not in groups:
            raise ValueError("Unknown sentiment label " + str(item["label"]))
        groups[item["label"]].append(item)
    per_label = limit // 3
    rng = random.Random(seed)
    selected = []
    for label in LABELS:
        candidates = list(groups[label])
        if len(candidates) < per_label:
            raise ValueError(f"Not enough {label} examples for --limit {limit}")
        rng.shuffle(candidates)
        selected.extend(candidates[:per_label])
    rng.shuffle(selected)
    return selected


def vader_label(compound):
    """Official 3-class VADER thresholds; raw compound still used in app."""
    if compound >= 0.05:
        return "positive"
    if compound <= -0.05:
        return "negative"
    return "neutral"


def predict_vader(examples):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    return [vader_label(analyzer.polarity_scores(item["text"])["compound"])
            for item in examples]


def decode_model_label(raw, model_key):
    """Normalize Hugging Face labels, including numeric model-config labels."""
    key = str(raw).strip().lower()
    if key in LABELS:
        label = key
    else:
        mapping = {
            "distilbert": {"label_0": "negative", "label_1": "positive"},
            "roberta": {"label_0": "negative", "label_1": "neutral",
                        "label_2": "positive"},
        }
        label = mapping.get(model_key, {}).get(key)
    if label not in LABELS:
        raise ValueError(f"Unexpected label {raw!r} from model {model_key}")
    if model_key == "distilbert" and label == "neutral":
        raise ValueError("The chosen DistilBERT checkpoint has no neutral class")
    return label


def predict_transformer(examples, model_key, batch_size=4):
    """Run one model at a time on CPU to reduce memory pressure."""
    if model_key not in MODEL_NAMES:
        raise ValueError("Unknown model: " + model_key)
    from transformers import pipeline
    model = MODEL_NAMES[model_key]
    print(f"Loading {model} (first run downloads pretrained weights)...", flush=True)
    classifier = pipeline("text-classification", model=model, device=-1)
    texts = [item["text"] for item in examples]
    predictions = []
    try:
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            outputs = classifier(batch, truncation=True, max_length=128,
                                 batch_size=batch_size)
            for output in outputs:
                # Some transformers versions return a nested singleton list.
                if isinstance(output, list):
                    output = output[0]
                predictions.append(decode_model_label(output["label"], model_key))
            if (start + len(batch)) % 30 == 0:
                print(f"  {model_key}: {start + len(batch)}/{len(texts)} evaluated", flush=True)
    finally:
        del classifier
        gc.collect()
    if len(predictions) != len(examples):
        raise RuntimeError("Model returned a different number of predictions")
    return predictions


def report(examples, predicted, class_labels=LABELS):
    """Accuracy, macro F1, per-label recall, and actual/predicted matrix."""
    if len(examples) != len(predicted) or not examples:
        raise ValueError("Expected one prediction per example")
    correct = 0
    confusion = {actual: {label: 0 for label in class_labels}
                 for actual in class_labels}
    for item, guess in zip(examples, predicted):
        true = item["label"]
        if true not in confusion or guess not in class_labels:
            raise ValueError("Unknown true or predicted class")
        confusion[true][guess] += 1
        correct += int(true == guess)
    f1_scores = {}
    recall = {}
    for label in class_labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in class_labels if other != label)
        fn = sum(confusion[label][other] for other in class_labels if other != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall[label] = tp / (tp + fn) if tp + fn else 0.0
        f1_scores[label] = (2 * precision * recall[label] / (precision + recall[label])
                            if precision + recall[label] else 0.0)
    return {
        "accuracy": round(correct / len(examples), 4),
        "macro_f1": round(sum(f1_scores.values()) / len(class_labels), 4),
        "per_class_recall": {label: round(value, 4) for label, value in recall.items()},
        "correct": correct, "total": len(examples),
        "confusion_actual_rows_predicted_columns": confusion,
    }


def selected_models(value):
    """VADER is always a baseline, and transformers run only if requested."""
    if value == "all":
        return ("vader", "distilbert", "roberta")
    if value == "vader":
        return ("vader",)
    if value == "roberta":
        return ("vader", "roberta")
    if value == "distilbert":
        return ("vader", "distilbert")
    raise ValueError("Invalid --models selection")


def main():
    parser = argparse.ArgumentParser(description="Three-class sentiment benchmark; live chatbot unchanged")
    parser.add_argument("--dataset", choices=("tweet_eval", "challenges"),
                        default="tweet_eval", help="TweetEval test set or authored examples")
    parser.add_argument("--models", choices=("vader", "distilbert", "roberta", "all"),
                        default="vader", help="Default only runs VADER; 'all' downloads two pretrained models")
    parser.add_argument("--limit", type=int, default=90,
                        help="Balanced TweetEval sample size; multiple of 3 (default 90)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Repeatable TweetEval sample selection")
    parser.add_argument("--errors", type=int, default=6,
                        help="Print up to this many incorrect examples per model (default 6)")
    args = parser.parse_args()

    if args.dataset == "tweet_eval":
        if args.limit > 600:
            parser.error("Use at most --limit 600 for this lightweight laptop experiment")
        try:
            text_path, labels_path = ensure_tweeteval()
            examples = balanced_sample(load_tweeteval(text_path, labels_path),
                                       limit=args.limit, seed=args.seed)
        except (OSError, ValueError, RuntimeError) as exc:
            parser.error(str(exc))
        print("Dataset: official TweetEval sentiment TEST split (public tweets), "
              "balanced sampled with seed", args.seed)
    else:
        examples = read_challenges()
        print("Dataset: AUTHORED chat-style stress tests; labels are subjective, "
              "NOT independently verified")
    print("Class counts:", dict(Counter(x["label"] for x in examples)))
    print("Interpretation: positive/neutral/negative text, not psychological status.\n")

    predictions_by_model = {}
    for name in selected_models(args.models):
        predictions = (predict_vader(examples) if name == "vader"
                       else predict_transformer(examples, name))
        predictions_by_model[name] = predictions
        result = report(examples, predictions)
        print(f"\n{name.upper()} (three-class; {len(examples)} messages)")
        print(json.dumps(result, indent=2))
        if name == "distilbert":
            print("NOTE: This SST-2 classifier CANNOT predict neutral; "
                  "three-class accuracy is structurally limited.")
        mistakes = [(item, guess) for item, guess in zip(examples, predictions)
                    if item["label"] != guess]
        for item, guess in mistakes[:max(0, args.errors)]:
            snippet = item["text"].replace("\n", " ")[:135]
            print(f"  Incorrect: expected={item['label']} predicted={guess} | {snippet}")
        if name != "vader":
            gc.collect()

    if "distilbert" in predictions_by_model:
        # Fairer secondary check: only evaluate true positive/negative cases.
        indices = [i for i, item in enumerate(examples) if item["label"] != "neutral"]
        subset = [examples[i] for i in indices]
        print("\nNON-NEUTRAL SUBSET ONLY: DistilBERT vs other models")
        for name, predicted in predictions_by_model.items():
            guesses = [predicted[i] for i in indices]
            # A three-class model's neutral prediction counts as a mistake.
            # Compute binary accuracy directly, not a renormalized score.
            binary_acc = sum(row["label"] == guess for row, guess
                             in zip(subset, guesses)) / len(subset)
            print(f"  {name}: {binary_acc:.3f} accuracy on {len(subset)} true non-neutral examples")
    print("\nLIMITATIONS: TweetEval measures short social-media sentiment, NOT "
          "mental-health evaluation. RoBERTa was fine-tuned on TweetEval's "
          "training split, so it has an in-domain advantage. This is a "
          "pretrained-inference experiment, not BERT training.")
    print("No VADER backend, dashboard, SQLite data, or chatbot code has been changed.")


if __name__ == "__main__":
    main()
