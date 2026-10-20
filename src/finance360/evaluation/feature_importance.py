import json
from pathlib import Path

import numpy as np

from finance360.modeling.data_split import create_modeling_split
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
    load_modeling_data,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger


def calculate_feature_importance(pipeline) -> list[dict]:
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    transformed_names = preprocessor.get_feature_names_out()
    booster = classifier.get_booster()
    gain_scores = booster.get_score(importance_type="gain")

    if len(transformed_names) != classifier.n_features_in_:
        raise ValueError("Transformed feature count does not match model")

    if len(set(transformed_names)) != len(transformed_names):
        raise ValueError("Transformed feature names must be unique")

    numeric_output_names = preprocessor.named_transformers_[
        "numeric"
    ].get_feature_names_out(NUMERICAL_FEATURES)

    categorical_output_names = preprocessor.named_transformers_[
        "categorical"
    ].get_feature_names_out(CATEGORICAL_FEATURES)

    encoder = preprocessor.named_transformers_[
        "categorical"
    ].named_steps["encoder"]

    categorical_counts = [
        len(categories)
        for categories in encoder.categories_
    ]

    if len(categorical_counts) != len(CATEGORICAL_FEATURES):
        raise ValueError("Categorical feature mapping is inconsistent")

    if sum(categorical_counts) != len(categorical_output_names):
        raise ValueError("Categorical output columns do not reconcile")

    if len(numeric_output_names) != len(NUMERICAL_FEATURES):
        raise ValueError("Numerical output columns do not reconcile")

    source_features = list(NUMERICAL_FEATURES)

    for feature_name, category_count in zip(
        CATEGORICAL_FEATURES,
        categorical_counts,
        strict=True,
    ):
        source_features.extend([feature_name] * category_count)

    if len(source_features) != len(transformed_names):
        raise ValueError("Source feature mapping is incomplete")

    model_feature_names = booster.feature_names

    if model_feature_names is not None:
        if len(model_feature_names) != len(transformed_names):
            raise ValueError("XGBoost feature names do not reconcile")

        feature_identifiers = model_feature_names
    else:
        feature_identifiers = [
            f"f{index}"
            for index in range(len(transformed_names))
        ]

    aggregated = {
        feature_name: {
            "total_gain": 0.0,
            "encoded_columns": 0,
            "used_encoded_columns": 0,
        }
        for feature_name in NUMERICAL_FEATURES + CATEGORICAL_FEATURES
    }

    for identifier, source_feature in zip(
        feature_identifiers,
        source_features,
        strict=True,
    ):
        gain = float(gain_scores.get(identifier, 0.0))

        if not np.isfinite(gain) or gain < 0:
            raise ValueError("Invalid XGBoost gain score")

        aggregated[source_feature]["total_gain"] += gain
        aggregated[source_feature]["encoded_columns"] += 1

        if gain > 0:
            aggregated[source_feature]["used_encoded_columns"] += 1

    total_gain = sum(
        values["total_gain"]
        for values in aggregated.values()
    )

    if not np.isfinite(total_gain) or total_gain <= 0:
        raise ValueError("Feature importance contains no positive gain")

    results = []

    for feature_name, values in aggregated.items():
        results.append(
            {
                "feature": feature_name,
                "total_gain": float(values["total_gain"]),
                "normalized_gain": float(
                    values["total_gain"] / total_gain
                ),
                "encoded_columns": values["encoded_columns"],
                "used_encoded_columns": values["used_encoded_columns"],
            }
        )

    results.sort(
        key=lambda item: item["normalized_gain"],
        reverse=True,
    )

    for rank, item in enumerate(results, start=1):
        item["rank"] = rank

    return results


def generate_feature_importance(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    pipeline = create_xgboost_challenger()
    pipeline.fit(split.X_train, split.y_train)

    importance = calculate_feature_importance(pipeline)

    result = {
        "model": "xgboost_challenger",
        "importance_type": "gain",
        "aggregation": "sum_of_encoded_column_gain_scores",
        "training_rows": len(split.X_train),
        "held_out_test_rows": len(split.X_test),
        "test_set_evaluated": False,
        "feature_count": len(importance),
        "feature_importance": importance,
    }

    report_path = (
        project_root
        / "reports"
        / "modeling"
        / "xgboost_feature_importance.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = generate_feature_importance(project_root)

    print(f"Training applications: {result['training_rows']}")
    print(f"Features analyzed: {result['feature_count']}")

    for item in result["feature_importance"][:10]:
        print(
            f"{item['rank']:2d}. "
            f"{item['feature']:35s} "
            f"{item['normalized_gain']:.4%}"
        )


if __name__ == "__main__":
    main()