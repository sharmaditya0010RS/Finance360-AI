from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from finance360.evaluation.feature_importance import (
    calculate_feature_importance,
)
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger


@pytest.fixture
def fitted_xgboost_pipeline():
    rng = np.random.default_rng(42)
    rows = 120

    features = pd.DataFrame(
        {
            feature: rng.normal(size=rows)
            for feature in NUMERICAL_FEATURES
        }
    )

    for feature in CATEGORICAL_FEATURES:
        features[feature] = rng.choice(
            ["Category_A", "Category_B"],
            size=rows,
        )

    target = pd.Series([0, 1] * (rows // 2))

    pipeline = create_xgboost_challenger()
    pipeline.fit(features, target)

    return pipeline


def test_importance_contains_all_source_features(
    fitted_xgboost_pipeline,
):
    result = calculate_feature_importance(fitted_xgboost_pipeline)

    expected_features = set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )

    assert len(result) == 24
    assert {item["feature"] for item in result} == expected_features


def test_importance_normalizes_to_one(
    fitted_xgboost_pipeline,
):
    result = calculate_feature_importance(fitted_xgboost_pipeline)

    assert sum(
        item["normalized_gain"]
        for item in result
    ) == pytest.approx(1.0)


def test_importance_ranks_features_in_descending_order(
    fitted_xgboost_pipeline,
):
    result = calculate_feature_importance(fitted_xgboost_pipeline)

    gains = [
        item["normalized_gain"]
        for item in result
    ]

    assert gains == sorted(gains, reverse=True)
    assert [item["rank"] for item in result] == list(range(1, 25))


def test_importance_tracks_encoded_column_counts(
    fitted_xgboost_pipeline,
):
    result = calculate_feature_importance(fitted_xgboost_pipeline)

    by_feature = {
        item["feature"]: item
        for item in result
    }

    for feature in NUMERICAL_FEATURES:
        assert by_feature[feature]["encoded_columns"] == 1

    for feature in CATEGORICAL_FEATURES:
        assert by_feature[feature]["encoded_columns"] == 2


def test_importance_rejects_missing_gain(
    fitted_xgboost_pipeline,
):
    classifier = fitted_xgboost_pipeline.named_steps["classifier"]
    booster = classifier.get_booster()

    with (
        patch.object(booster, "get_score", return_value={}),
        pytest.raises(
            ValueError,
            match="Feature importance contains no positive gain",
        ),
    ):
        calculate_feature_importance(fitted_xgboost_pipeline)