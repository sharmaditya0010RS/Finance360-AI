import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from finance360.evaluation.calibration import calculate_calibration
from finance360.modeling.baseline import create_logistic_baseline
from finance360.modeling.data_split import RANDOM_STATE, create_modeling_split
from finance360.modeling.feature_contract import load_modeling_data
from finance360.modeling.train_baseline import N_SPLITS
from finance360.modeling.xgboost_model import create_xgboost_challenger

MODEL_FACTORIES = {
    "logistic_regression_baseline": create_logistic_baseline,
    "xgboost_challenger": create_xgboost_challenger,
}


def evaluate_model_calibration(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = list(
        cross_validation.split(split.X_train, split.y_train)
    )

    model_results = {}

    for model_name, model_factory in MODEL_FACTORIES.items():
        out_of_fold_probabilities = np.full(
            len(split.X_train),
            np.nan,
            dtype=float,
        )

        for fold_number, (train_indices, validation_indices) in enumerate(
            folds,
            start=1,
        ):
            pipeline = model_factory()

            pipeline.fit(
                split.X_train.iloc[train_indices],
                split.y_train.iloc[train_indices],
            )

            probabilities = pipeline.predict_proba(
                split.X_train.iloc[validation_indices]
            )[:, 1]

            out_of_fold_probabilities[validation_indices] = probabilities

            print(
                f"{model_name}: fold {fold_number}/{N_SPLITS} completed",
                flush=True,
            )

        if not np.isfinite(out_of_fold_probabilities).all():
            raise ValueError(
                f"Incomplete out-of-fold predictions for {model_name}"
            )

        calibration = calculate_calibration(
            split.y_train,
            out_of_fold_probabilities,
            n_bins=10,
        )

        model_results[model_name] = {
            "roc_auc": float(
                roc_auc_score(
                    split.y_train,
                    out_of_fold_probabilities,
                )
            ),
            "average_precision": float(
                average_precision_score(
                    split.y_train,
                    out_of_fold_probabilities,
                )
            ),
            "calibration": calibration,
        }

    result = {
        "evaluation": "out_of_fold_probability_calibration",
        "validation_strategy": "stratified_3_fold_training_only",
        "training_rows": len(split.X_train),
        "held_out_test_rows": len(split.X_test),
        "test_set_evaluated": False,
        "models": model_results,
    }

    report_path = (
        project_root
        / "reports"
        / "modeling"
        / "model_calibration_oof.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = evaluate_model_calibration(project_root)

    for model_name, metrics in result["models"].items():
        calibration = metrics["calibration"]

        print(
            f"{model_name}: "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"AP={metrics['average_precision']:.4f}, "
            f"Brier={calibration['brier_score']:.5f}, "
            f"ECE={calibration['expected_calibration_error']:.5f}",
            flush=True,
        )


if __name__ == "__main__":
    main()