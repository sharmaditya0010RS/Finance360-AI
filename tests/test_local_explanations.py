import numpy as np
import pandas as pd
import pytest

from finance360.evaluation.local_explanations import explain_application
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger


@pytest.fixture(scope="module")
def fitted_pipeline_and_application():
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

    return pipeline, features.iloc[[0]]


@pytest.fixture(scope="module")
def explanation(fitted_pipeline_and_application):
    pipeline, application = fitted_pipeline_and_application

    return explain_application(
        pipeline,
        application,
        top_n=5,
    )


def test_explanation_contains_all_source_features(explanation):
    assert explanation["source_feature_count"] == 24
    assert len(explanation["all_contributions"]) == 24

    assert {
        item["feature"]
        for item in explanation["all_contributions"]
    } == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)


def test_explanation_reconstructs_probability(explanation):
    assert explanation["explanation_space"] == "raw_log_odds"

    assert explanation["reconstruction_error"] <= 1e-4

    assert explanation["predicted_probability"] == pytest.approx(
        explanation["reconstructed_probability"],
        abs=1e-4,
    )


def test_explanation_returns_requested_top_features(explanation):
    assert len(explanation["top_contributions"]) == 5

    values = [
        abs(item["shap_log_odds"])
        for item in explanation["all_contributions"]
    ]

    assert values == sorted(values, reverse=True)

    assert [
        item["rank"]
        for item in explanation["all_contributions"]
    ] == list(range(1, 25))


def test_explanation_directions_are_consistent(explanation):
    for item in explanation["all_contributions"]:
        value = item["shap_log_odds"]

        expected_direction = (
            "increases_predicted_risk"
            if value > 0
            else (
                "decreases_predicted_risk"
                if value < 0
                else "neutral"
            )
        )

        assert item["direction"] == expected_direction


def test_explanation_rejects_multiple_applications(
    fitted_pipeline_and_application,
):
    pipeline, application = fitted_pipeline_and_application

    multiple = pd.concat(
        [application, application],
        ignore_index=True,
    )

    with pytest.raises(
        ValueError,
        match="Exactly one application is required",
    ):
        explain_application(pipeline, multiple)


def test_explanation_rejects_missing_feature(
    fitted_pipeline_and_application,
):
    pipeline, application = fitted_pipeline_and_application

    invalid = application.drop(
        columns=[NUMERICAL_FEATURES[0]]
    )

    with pytest.raises(
        ValueError,
        match="Application feature contract mismatch",
    ):
        explain_application(pipeline, invalid)


@pytest.mark.parametrize("top_n", [0, 25, -1])
def test_explanation_rejects_invalid_top_n(
    fitted_pipeline_and_application,
    top_n,
):
    pipeline, application = fitted_pipeline_and_application

    with pytest.raises(
        ValueError,
        match="Invalid number of explanation features",
    ):
        explain_application(
            pipeline,
            application,
            top_n=top_n,
        )


def test_explanation_rejects_non_dataframe(
    fitted_pipeline_and_application,
):
    pipeline, application = fitted_pipeline_and_application

    with pytest.raises(
        TypeError,
        match="Application must be a pandas DataFrame",
    ):
        explain_application(
            pipeline,
            application.to_numpy(),
        )