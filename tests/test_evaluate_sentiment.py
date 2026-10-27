"""Evaluation logic unit tests; no transformer weight download."""
from evaluate_sentiment import evaluate, read_examples, transformer_predict, vader_predict, BENCHMARK


class StubAnalyzer:
    def polarity_scores(self, text):
        return {"compound": 0.8 if "good" in text else -0.4}


def test_binary_vader_mapping():
    assert vader_predict("good news", StubAnalyzer())[0] == "positive"
    assert vader_predict("bad news", StubAnalyzer())[0] == "negative"


def test_transformer_labels():
    assert transformer_predict("great", lambda text, **kw: [{"label": "POSITIVE", "score": 0.98}])[0] == "positive"
    assert transformer_predict("bad", lambda text, **kw: [{"label": "NEGATIVE", "score": 0.95}])[0] == "negative"


def test_f1_balanced_perfect():
    examples = [{"label": "positive"}, {"label": "negative"}]
    result = evaluate(examples, ["positive", "negative"])
    assert result["accuracy"] == 1
    assert result["macro_f1"] == 1


def test_examples_are_illustrative_binary_labels():
    rows = read_examples(BENCHMARK)
    assert len(rows) >= 20
    assert {r["label"] for r in rows} == {"positive", "negative"}
