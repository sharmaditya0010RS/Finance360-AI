import numpy as np
import pandas as pd
import shap
from scipy.special import expit

from finance360.evaluation.shap_analysis import (
    get_source_feature_mapping,
)
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def explain_application(
    pipeline,
    application: pd.DataFrame,
    top_n: int = 5,
) -> dict:
    if not isinstance(application, pd.DataFrame):
        raise TypeError("Application must be a pandas DataFrame")

    if len(application) != 1:
        raise ValueError("Exactly one application is required")

    if top_n < 1 or top_n > len(
        NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    ):
        raise ValueError("Invalid number of explanation features")

    required_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    if set(application.columns) != set(required_features):
        raise ValueError("Application feature contract mismatch")

    application = application.loc[:, required_features]

    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(application)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    source_mapping = get_source_feature_mapping(preprocessor)

    if transformed.shape != (1, len(source_mapping)):
        raise ValueError("Transformed application shape mismatch")

    explainer = shap.TreeExplainer(classifier)

    shap_values = np.asarray(
        explainer.shap_values(transformed),
        dtype=float,
    )

    if shap_values.shape != transformed.shape:
        raise ValueError("Unexpected SHAP explanation shape")

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP explanation contains non-finite values")

    expected_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    contributions = {
        feature: 0.0
        for feature in required_features
    }

    for index, feature in enumerate(source_mapping):
        contributions[feature] += float(shap_values[0, index])

    reconstructed_probability = float(
        expit(expected_value + sum(contributions.values()))
    )

    predicted_probability = float(
        classifier.predict_proba(transformed)[0, 1]
    )

    reconstruction_error = abs(
        reconstructed_probability - predicted_probability
    )

    if reconstruction_error > 1e-4:
        raise ValueError("Local explanation reconstruction failed")

    ranked = sorted(
        contributions.items(),
        key=lambda item: abs(item[1]),
        reverse=True,
    )

    feature_contributions = []

    for rank, (feature, value) in enumerate(ranked, start=1):
        feature_contributions.append(
            {
                "rank": rank,
                "feature": feature,
                "shap_log_odds": value,
                "direction": (
                    "increases_predicted_risk"
                    if value > 0
                    else (
                        "decreases_predicted_risk"
                        if value < 0
                        else "neutral"
                    )
                ),
            }
        )

    return {
        "predicted_probability": predicted_probability,
        "expected_value_log_odds": expected_value,
        "reconstructed_probability": reconstructed_probability,
        "reconstruction_error": reconstruction_error,
        "explanation_space": "raw_log_odds",
        "source_feature_count": len(required_features),
        "top_contributions": feature_contributions[:top_n],
        "all_contributions": feature_contributions,
    }