import json

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from finance360.evaluation.held_out_evaluation import evaluate_held_out_model
from finance360.modeling.feature_contract import MODEL_FEATURES
from finance360.modeling.save_model import calculate_sha256


@pytest.fixture
def evaluation_project(tmp_path, monkeypatch):
    rng = np.random.default_rng(42)
    rows = 100

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

    split = FakeSplit()

    pipeline = LogisticRegression(max_iter=200)
    pipeline.fit(split.X_train, split.y_train)

    model_dir = tmp_path / "artifacts" / "models"
    model_dir.mkdir(parents=True)

    model_path = model_dir / "finance360_xgboost_v1.joblib"
    metadata_path = model_dir / "finance360_xgboost_v1_metadata.json"

    joblib.dump(pipeline, model_path)

    metadata = {
        "model_name": "xgboost_challenger",
        "model_version": "1.0.0",
        "selection_status": "provisional",
        "training_rows": 80,
        "held_out_test_rows": 20,
        "feature_names": list(MODEL_FEATURES),
        "test_set_evaluated": False,
        "artifact_sha256": calculate_sha256(model_path),
    }

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        "finance360.evaluation.held_out_evaluation.load_modeling_data",
        lambda project_root: (features, target),
    )

    monkeypatch.setattr(
        "finance360.evaluation.held_out_evaluation.create_modeling_split",
        lambda X, y: split,
    )

    return tmp_path, model_path, metadata_path


def test_held_out_evaluation_writes_valid_report(evaluation_project):
    project_root, _, _ = evaluation_project

    result = evaluate_held_out_model(project_root)

    assert result["test_rows"] == 20
    assert result["training_rows"] == 80
    assert result["test_positive_count"] == 10
    assert result["test_set_evaluated"] is True
    assert 0 <= result["roc_auc"] <= 1
    assert 0 <= result["average_precision"] <= 1
    assert 0 <= result["brier_score"] <= 1

    confusion = result["confusion_matrix"]

    assert sum(confusion.values()) == 20

    report_path = (
        project_root
        / "reports"
        / "modeling"
        / "held_out_test_evaluation.json"
    )

    assert report_path.exists()

    saved_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved_report == result


def test_held_out_evaluation_rejects_modified_model(evaluation_project):
    project_root, model_path, _ = evaluation_project

    with model_path.open("ab") as file:
        file.write(b"tampered")

    with pytest.raises(
        ValueError,
        match="Model artifact checksum mismatch",
    ):
        evaluate_held_out_model(project_root)


def test_held_out_evaluation_rejects_feature_mismatch(evaluation_project):
    project_root, _, metadata_path = evaluation_project

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    metadata["feature_names"] = ["wrong_feature"]

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Saved model feature contract mismatch",
    ):
        evaluate_held_out_model(project_root)


def test_held_out_evaluation_rejects_prior_evaluation(evaluation_project):
    project_root, _, metadata_path = evaluation_project

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    metadata["test_set_evaluated"] = True

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Model metadata indicates prior test evaluation",
    ):
        evaluate_held_out_model(project_root)


def test_held_out_evaluation_rejects_training_count_mismatch(
    evaluation_project,
):
    project_root, _, metadata_path = evaluation_project

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    metadata["training_rows"] = 79

    metadata_path.write_text(
        json.dumps(metadata),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Training row count mismatch",
    ):
        evaluate_held_out_model(project_root)