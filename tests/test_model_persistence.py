import hashlib
import json

import joblib
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier

from finance360.modeling.feature_contract import MODEL_FEATURES
from finance360.modeling.save_model import (
    MODEL_VERSION,
    calculate_sha256,
    save_champion_model,
)


def test_calculate_sha256_matches_expected_digest(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"finance360")

    expected = hashlib.sha256(b"finance360").hexdigest()

    assert calculate_sha256(file_path) == expected


def test_calculate_sha256_detects_file_changes(tmp_path):
    file_path = tmp_path / "sample.bin"
    file_path.write_bytes(b"version-one")

    original = calculate_sha256(file_path)

    file_path.write_bytes(b"version-two")

    assert calculate_sha256(file_path) != original


def test_model_version_is_defined():
    assert MODEL_VERSION == "1.0.0"


def test_saved_model_metadata_and_reload(
    tmp_path,
    monkeypatch,
):
    rows = 100
    rng = np.random.default_rng(42)

    features = pd.DataFrame(
        {
            feature: rng.normal(size=rows)
            for feature in MODEL_FEATURES
        }
    )

    target = pd.Series([0, 1] * (rows // 2))

    class FakeSplit:
        X_train = features.iloc[:80]
        X_test = features.iloc[80:]
        y_train = target.iloc[:80]
        y_test = target.iloc[80:]

    monkeypatch.setattr(
        "finance360.modeling.save_model.load_modeling_data",
        lambda project_root: (features, target),
    )

    monkeypatch.setattr(
        "finance360.modeling.save_model.create_modeling_split",
        lambda X, y: FakeSplit(),
    )

    monkeypatch.setattr(
        "finance360.modeling.save_model.create_xgboost_challenger",
        lambda: DummyClassifier(
            strategy="constant",
            constant=0,
        ),
    )

    metadata = save_champion_model(tmp_path)

    model_path = (
        tmp_path
        / "artifacts"
        / "models"
        / "finance360_xgboost_v1.joblib"
    )

    metadata_path = (
        tmp_path
        / "artifacts"
        / "models"
        / "finance360_xgboost_v1_metadata.json"
    )

    assert model_path.exists()
    assert metadata_path.exists()

    saved_metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    assert metadata == saved_metadata
    assert metadata["model_version"] == MODEL_VERSION
    assert metadata["training_rows"] == 80
    assert metadata["held_out_test_rows"] == 20
    assert metadata["feature_count"] == 24
    assert metadata["feature_names"] == list(MODEL_FEATURES)
    assert metadata["test_set_evaluated"] is False
    assert metadata["artifact_sha256"] == calculate_sha256(
        model_path
    )
    assert metadata["maximum_reload_probability_error"] == 0.0

    reloaded = joblib.load(model_path)

    assert isinstance(reloaded, DummyClassifier)

    probabilities = reloaded.predict_proba(
        features.iloc[:5]
    )

    assert probabilities.shape == (5, 2)
    assert np.allclose(probabilities[:, 1], 0.0)