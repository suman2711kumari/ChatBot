"""Three-class evaluator unit tests. No network/model download in tests."""
import json

import pytest
from evaluate_sentiment_threeclass import (
    LABELS, balanced_sample, decode_model_label, ensure_tweeteval,
    load_tweeteval, read_challenges, report, selected_models, vader_label
)


def test_vader_thresholds():
    assert vader_label(0.06) == "positive"
    assert vader_label(-0.06) == "negative"
    assert vader_label(0.0) == "neutral"
    assert vader_label(0.05) == "positive"
    assert vader_label(-0.05) == "negative"


def test_hf_label_mapping():
    assert decode_model_label("POSITIVE", "roberta") == "positive"
    assert decode_model_label("Neutral", "roberta") == "neutral"
    assert decode_model_label("LABEL_0", "roberta") == "negative"
    assert decode_model_label("LABEL_1", "roberta") == "neutral"
    assert decode_model_label("LABEL_2", "roberta") == "positive"
    assert decode_model_label("LABEL_0", "distilbert") == "negative"
    assert decode_model_label("LABEL_1", "distilbert") == "positive"
    with pytest.raises(ValueError):
        decode_model_label("neutral", "distilbert")


def test_balanced_repeatable_sample():
    cases = [{"text": str(i), "label": label} for label in LABELS
             for i in range(10)]
    first = balanced_sample(cases, limit=15, seed=42)
    second = balanced_sample(cases, limit=15, seed=42)
    assert first == second
    assert len(first) == 15
    assert all(sum(x["label"] == label for x in first) == 5 for label in LABELS)
    with pytest.raises(ValueError):
        balanced_sample(cases, limit=16)


def test_alignment_of_downloaded_tweet_files(tmp_path):
    texts = tmp_path / "test_text.txt"
    labels = tmp_path / "test_labels.txt"
    texts.write_text("This was fine\nVery bad\nWonderful!\n", encoding="utf-8")
    labels.write_text("1\n0\n2\n", encoding="utf-8")
    rows = load_tweeteval(texts, labels)
    assert [x["label"] for x in rows] == ["neutral", "negative", "positive"]
    labels.write_text("1\n0\n", encoding="utf-8")
    with pytest.raises(ValueError):
        load_tweeteval(texts, labels)


def test_cache_downloader_runs_only_for_missing_files(tmp_path):
    calls = []

    class FakeResponse:
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            pass
        def read(self):
            return b"some text\n"

    def opener(url, timeout=0):
        calls.append(url)
        return FakeResponse()

    ensure_tweeteval(tmp_path, opener)
    ensure_tweeteval(tmp_path, opener)
    assert len(calls) == 2
    assert (tmp_path / "test_text.txt").exists()


def test_metrics_threeclass():
    examples = [{"label": x} for x in LABELS]
    perfect = report(examples, list(LABELS))
    assert perfect["accuracy"] == 1
    assert perfect["macro_f1"] == 1
    assert perfect["per_class_recall"]["neutral"] == 1
    wrong_neutral = report(examples, ["negative", "negative", "positive"])
    assert wrong_neutral["accuracy"] == 0.6667
    assert wrong_neutral["per_class_recall"]["neutral"] == 0
    assert wrong_neutral["confusion_actual_rows_predicted_columns"]["neutral"]["negative"] == 1


def test_challenge_set_is_three_class():
    rows = read_challenges()
    assert len(rows) >= 30
    assert {x["label"] for x in rows} == set(LABELS)
    assert selected_models("roberta") == ("vader", "roberta")
    assert selected_models("all") == ("vader", "distilbert", "roberta")
