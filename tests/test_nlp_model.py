import json

import numpy as np

from nlp_model import NRCFeatures, build_stress_pipeline, predict_stress


def test_nrc_uses_binary_associations_not_manual_weights(tmp_path):
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"sad": {"sadness": 1.5}, "fear": {"fear": 1}}))
    features = NRCFeatures(str(path)).fit(["sad", "fear"]).transform(
        ["sad", "sad fear", "unknown"]
    ).toarray()
    assert features.shape == (3, 10)
    assert features[0, 5] == 1.0
    assert features[1, 5] == 0.5
    assert np.all(features[2] == 0)


def test_pipeline_learns_and_predicts(tmp_path):
    path = tmp_path / "lexicon.json"
    path.write_text(json.dumps({"sad": {"sadness": 1}, "happy": {"joy": 1}}))
    text = ["happy calm today", "happy great work", "calm amazing joy",
            "great calm day", "sad worried deadline", "sad pressure today",
            "worried deadlines at work", "pressure and worried"]
    labels = [0, 0, 0, 0, 1, 1, 1, 1]
    trained = build_stress_pipeline(str(path)).fit(text, labels)
    result = predict_stress(trained, "sad and worried")
    assert result["label"] in (0, 1)
    assert 0 <= result["probability"] <= 1
