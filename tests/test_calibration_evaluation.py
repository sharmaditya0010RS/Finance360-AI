import json
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finance360.evaluation.evaluate_calibration import (
    evaluate_model_calibration,
)


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
        return self

    def predict_proba(self, features):
        probabilities = np.clip(
            0.05 + features["income"].to_numpy() / 1000,
            0.01,
            0.99,
        )

        return np.column_stack(
            [1 - probabilities, probabilities]
        )


def test_calibration_evaluation_report_structure(
    tmp_path,
    synthetic_training_data,
):
    features, target = synthetic_training_data

    factories = {
        "logistic_regression_baseline": DummyProbabilityModel,
        "xgboost_challenger": DummyProbabilityModel,
    }

    with (
        patch(
            "finance360.evaluation.evaluate_calibration.load_modeling_data",
            return_value=(features, target),
        ),
        patch(
            "finance360.evaluation.evaluate_calibration.MODEL_FACTORIES",
            factories,
        ),
    ):
        result = evaluate_model_calibration(tmp_path)

    assert result["evaluation"] == "out_of_fold_probability_calibration"
    assert result["validation_strategy"] == "stratified_3_fold_training_only"
    assert result["training_rows"] == 96
    assert result["held_out_test_rows"] == 24
    assert result["test_set_evaluated"] is False

    assert set(result["models"]) == {
        "logistic_regression_baseline",
        "xgboost_challenger",
    }

    for model_result in result["models"].values():
        assert 0 <= model_result["roc_auc"] <= 1
        assert 0 <= model_result["average_precision"] <= 1

        calibration = model_result["calibration"]

        assert calibration["rows"] == 96
        assert calibration["positive_count"] == 10
        assert calibration["requested_bins"] == 10
        assert sum(
            item["count"] for item in calibration["bins"]
        ) == 96

        assert 0 <= calibration["brier_score"] <= 1
        assert 0 <= calibration["expected_calibration_error"] <= 1


def test_calibration_evaluation_saves_json(
    tmp_path,
    synthetic_training_data,
):
    features, target = synthetic_training_data

    factories = {
        "logistic_regression_baseline": DummyProbabilityModel,
        "xgboost_challenger": DummyProbabilityModel,
    }

    with (
        patch(
            "finance360.evaluation.evaluate_calibration.load_modeling_data",
            return_value=(features, target),
        ),
        patch(
            "finance360.evaluation.evaluate_calibration.MODEL_FACTORIES",
            factories,
        ),
    ):
        result = evaluate_model_calibration(tmp_path)

    report_path = (
        tmp_path
        / "reports"
        / "modeling"
        / "model_calibration_oof.json"
    )

    assert report_path.is_file()

    saved_report = json.loads(
        report_path.read_text(encoding="utf-8")
    )

    assert saved_report == result
    assert saved_report["test_set_evaluated"] is False


def test_calibration_evaluation_rejects_incomplete_predictions(
    tmp_path,
    synthetic_training_data,
):
    features, target = synthetic_training_data

    class InvalidProbabilityModel:
        def fit(self, features, target):
            return self

        def predict_proba(self, features):
            return np.full((len(features), 2), np.nan)

    with (
        patch(
            "finance360.evaluation.evaluate_calibration.load_modeling_data",
            return_value=(features, target),
        ),
        patch(
            "finance360.evaluation.evaluate_calibration.MODEL_FACTORIES",
            {"invalid_model": InvalidProbabilityModel},
        ),
        pytest.raises(
            ValueError,
            match="Incomplete out-of-fold predictions",
        ),
    ):
        evaluate_model_calibration(tmp_path)