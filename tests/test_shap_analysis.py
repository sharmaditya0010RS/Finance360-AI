import numpy as np
import pandas as pd
import pytest

from finance360.evaluation.shap_analysis import (
    calculate_shap_summary,
    get_source_feature_mapping,
)
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger


@pytest.fixture(scope="module")
def fitted_shap_pipeline():
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

    target = pd.Series(
        [0, 1] * (rows // 2)
    )

    pipeline = create_xgboost_challenger()
    pipeline.fit(features, target)

    return pipeline, features


@pytest.fixture(scope="module")
def shap_summary(fitted_shap_pipeline):
    pipeline, features = fitted_shap_pipeline

    return calculate_shap_summary(
        pipeline,
        features.iloc[:20],
    )


def test_shap_mapping_covers_all_transformed_columns(
    fitted_shap_pipeline,
):
    pipeline, _ = fitted_shap_pipeline
    preprocessor = pipeline.named_steps["preprocessor"]

    mapping = get_source_feature_mapping(preprocessor)

    assert len(mapping) == len(
        preprocessor.get_feature_names_out()
    )

    assert set(mapping) == set(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    )


def test_shap_summary_contains_all_source_features(shap_summary):
    importance = shap_summary["global_feature_importance"]

    assert shap_summary["sample_rows"] == 20
    assert shap_summary["source_feature_count"] == 24
    assert len(importance) == 24

    assert {
        item["feature"]
        for item in importance
    } == set(NUMERICAL_FEATURES + CATEGORICAL_FEATURES)


def test_shap_reconstructs_model_probabilities(shap_summary):
    assert shap_summary["shap_output_space"] == "raw_log_odds"

    assert (
        shap_summary["maximum_probability_reconstruction_error"]
        <= 1e-4
    )


def test_shap_importance_is_sorted(shap_summary):
    importance = shap_summary["global_feature_importance"]

    values = [
        item["mean_absolute_shap"]
        for item in importance
    ]

    assert values == sorted(values, reverse=True)

    assert [
        item["rank"]
        for item in importance
    ] == list(range(1, 25))


def test_shap_importance_values_are_finite(shap_summary):
    importance = shap_summary["global_feature_importance"]

    for item in importance:
        assert np.isfinite(item["mean_absolute_shap"])
        assert np.isfinite(item["mean_signed_shap"])
        assert item["mean_absolute_shap"] >= 0


def test_shap_rejects_missing_input_feature(fitted_shap_pipeline):
    pipeline, features = fitted_shap_pipeline

    invalid_features = features.iloc[:5].drop(
        columns=[NUMERICAL_FEATURES[0]]
    )

    with pytest.raises(ValueError):
        calculate_shap_summary(
            pipeline,
            invalid_features,
        )