import json
from pathlib import Path

import numpy as np
import shap
from scipy.special import expit

from finance360.modeling.data_split import create_modeling_split
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    load_modeling_data,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger

SAMPLE_SIZE = 500
RANDOM_STATE = 42


def get_source_feature_mapping(preprocessor) -> list[str]:
    transformed_names = preprocessor.get_feature_names_out()

    numeric_pipeline = preprocessor.named_transformers_["numeric"]
    categorical_pipeline = preprocessor.named_transformers_["categorical"]

    numeric_names = numeric_pipeline.get_feature_names_out(
        NUMERICAL_FEATURES
    )

    categorical_names = categorical_pipeline.get_feature_names_out(
        CATEGORICAL_FEATURES
    )

    encoder = categorical_pipeline.named_steps["encoder"]

    category_counts = [
        len(categories)
        for categories in encoder.categories_
    ]

    if len(numeric_names) != len(NUMERICAL_FEATURES):
        raise ValueError("Numerical feature mapping is inconsistent")

    if len(category_counts) != len(CATEGORICAL_FEATURES):
        raise ValueError("Categorical feature mapping is inconsistent")

    if sum(category_counts) != len(categorical_names):
        raise ValueError("Categorical output columns do not reconcile")

    mapping = list(NUMERICAL_FEATURES)

    for feature_name, category_count in zip(
        CATEGORICAL_FEATURES,
        category_counts,
        strict=True,
    ):
        mapping.extend([feature_name] * category_count)

    if len(mapping) != len(transformed_names):
        raise ValueError("SHAP feature mapping is incomplete")

    return mapping


def calculate_shap_summary(
    pipeline,
    sample_features,
) -> dict:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed = preprocessor.transform(sample_features)

    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    transformed = np.asarray(transformed, dtype=float)

    source_mapping = get_source_feature_mapping(preprocessor)

    if transformed.shape[1] != len(source_mapping):
        raise ValueError("SHAP input feature count does not reconcile")

    explainer = shap.TreeExplainer(classifier)
    shap_values = np.asarray(
        explainer.shap_values(transformed),
        dtype=float,
    )

    if shap_values.shape != transformed.shape:
        raise ValueError("Unexpected SHAP output shape")

    if not np.isfinite(shap_values).all():
        raise ValueError("SHAP values contain non-finite values")

    expected_value = float(
        np.asarray(explainer.expected_value).reshape(-1)[0]
    )

    source_features = NUMERICAL_FEATURES + CATEGORICAL_FEATURES

    aggregated_values = np.zeros(
        (len(sample_features), len(source_features)),
        dtype=float,
    )

    source_indices = {
        feature_name: index
        for index, feature_name in enumerate(source_features)
    }

    for column_index, feature_name in enumerate(source_mapping):
        aggregated_values[:, source_indices[feature_name]] += (
            shap_values[:, column_index]
        )

    mean_absolute_values = np.mean(
        np.abs(aggregated_values),
        axis=0,
    )

    mean_signed_values = np.mean(
        aggregated_values,
        axis=0,
    )

    ranking = []

    for index, feature_name in enumerate(source_features):
        ranking.append(
            {
                "feature": feature_name,
                "mean_absolute_shap": float(
                    mean_absolute_values[index]
                ),
                "mean_signed_shap": float(
                    mean_signed_values[index]
                ),
            }
        )

    ranking.sort(
        key=lambda item: item["mean_absolute_shap"],
        reverse=True,
    )

    for rank, item in enumerate(ranking, start=1):
        item["rank"] = rank

    reconstructed_margins = (
        expected_value + shap_values.sum(axis=1)
    )

    reconstructed_probabilities = expit(
        reconstructed_margins
    )

    model_probabilities = classifier.predict_proba(
        transformed
    )[:, 1]

    maximum_probability_error = float(
        np.max(
            np.abs(
                reconstructed_probabilities
                - model_probabilities
            )
        )
    )

    if maximum_probability_error > 1e-4:
        raise ValueError(
            "SHAP contributions do not reconstruct model predictions"
        )

    return {
        "sample_rows": len(sample_features),
        "transformed_feature_count": transformed.shape[1],
        "source_feature_count": len(source_features),
        "shap_output_space": "raw_log_odds",
        "expected_value_log_odds": expected_value,
        "maximum_probability_reconstruction_error": (
            maximum_probability_error
        ),
        "global_feature_importance": ranking,
    }


def generate_shap_report(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    if len(split.X_train) < SAMPLE_SIZE:
        raise ValueError("Training partition is smaller than SHAP sample")

    pipeline = create_xgboost_challenger()
    pipeline.fit(split.X_train, split.y_train)

    sample_features = split.X_train.sample(
        n=SAMPLE_SIZE,
        random_state=RANDOM_STATE,
    )

    summary = calculate_shap_summary(
        pipeline,
        sample_features,
    )

    result = {
        "model": "xgboost_challenger",
        "explanation_method": "tree_shap",
        "sampling_strategy": "random_training_sample",
        "random_state": RANDOM_STATE,
        "training_rows": len(split.X_train),
        "held_out_test_rows": len(split.X_test),
        "test_set_evaluated": False,
        **summary,
    }

    report_path = (
        project_root
        / "reports"
        / "modeling"
        / "xgboost_shap_summary.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = generate_shap_report(project_root)

    print(f"Training applications: {result['training_rows']}")
    print(f"SHAP sample: {result['sample_rows']}")
    print(
        "Probability reconstruction error:",
        result["maximum_probability_reconstruction_error"],
    )

    print("Top 10 SHAP features:")

    for item in result["global_feature_importance"][:10]:
        print(
            f"{item['rank']:2d}. "
            f"{item['feature']:35s} "
            f"{item['mean_absolute_shap']:.6f}"
        )


if __name__ == "__main__":
    main()