import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finance360.modeling.train_baseline import evaluate_logistic_cv


@pytest.fixture
def synthetic_training_data():
    rows = 120

    features = pd.DataFrame(
        {
            "income": np.arange(rows, dtype=float),
            "credit": np.arange(rows, dtype=float) * 2,
        }
    )

    target = pd.Series([0] * 108 + [1] * 12)

    return features, target


class DummyProbabilityModel:
    def fit(self, features, target):
        self.training_rows = len(features)
        self.positive_rate = float(target.mean())
        return self

    def predict_proba(self, features):
        probabilities = np.clip(
            0.05 + features["income"].to_numpy() / 1000,
            0.01,
            0.99,
        )
        return np.column_stack([1 - probabilities, probabilities])


def test_cross_validation_report_structure(
    tmp_path,
    synthetic_training_data,
):
    features, target = synthetic_training_data

    with (
        patch(
            "finance360.modeling.train_baseline.load_modeling_data",
            return_value=(features, target),
        ),
        patch(
            "finance360.modeling.train_baseline.create_logistic_baseline",
            side_effect=DummyProbabilityModel,
        ),
    ):
        result = evaluate_logistic_cv(tmp_path)

    assert result["model"] == "logistic_regression_baseline"
    assert result["validation_strategy"] == "stratified_3_fold_training_only"
    assert result["training_rows"] == 96
    assert result["held_out_test_rows"] == 24
    assert result["training_positives"] == 10
    assert result["test_set_evaluated"] is False
    assert len(result["fold_results"]) == 3

    validation_positives = []

    for fold in result["fold_results"]:
        assert fold["training_rows"] == 64
        assert fold["validation_rows"] == 32
        assert fold["validation_positives"] in (3, 4)
        assert 0 <= fold["roc_auc"] <= 1
        assert 0 <= fold["average_precision"] <= 1
        assert 0 <= fold["brier_score"] <= 1

        validation_positives.append(fold["validation_positives"])

    assert sorted(validation_positives) == [3, 3, 4]
    assert sum(validation_positives) == result["training_positives"]

    assert set(result["summary"]) == {
        "roc_auc",
        "average_precision",
        "brier_score",
    }


def test_cross_validation_saves_json_report(
    tmp_path,
    synthetic_training_data,
):
    features, target = synthetic_training_data

    with (
        patch(
            "finance360.modeling.train_baseline.load_modeling_data",
            return_value=(features, target),
        ),
        patch(
            "finance360.modeling.train_baseline.create_logistic_baseline",
            side_effect=DummyProbabilityModel,
        ),
    ):
        result = evaluate_logistic_cv(tmp_path)

    report_path = (
        tmp_path
        / "reports"
        / "modeling"
        / "logistic_baseline_cv.json"
    )

    assert report_path.is_file()

    saved_report = json.loads(report_path.read_text(encoding="utf-8"))

    assert saved_report == result
    assert saved_report["test_set_evaluated"] is False