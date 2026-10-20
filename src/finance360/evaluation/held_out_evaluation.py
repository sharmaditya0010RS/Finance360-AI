import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from finance360.evaluation.calibration import calculate_calibration
from finance360.modeling.data_split import create_modeling_split
from finance360.modeling.feature_contract import (
    MODEL_FEATURES,
    load_modeling_data,
)
from finance360.modeling.save_model import calculate_sha256


def evaluate_held_out_model(project_root: Path) -> dict:
    model_dir = project_root / "artifacts" / "models"
    model_path = model_dir / "finance360_xgboost_v1.joblib"
    metadata_path = model_dir / "finance360_xgboost_v1_metadata.json"

    metadata = json.loads(
        metadata_path.read_text(encoding="utf-8")
    )

    if metadata["artifact_sha256"] != calculate_sha256(model_path):
        raise ValueError("Model artifact checksum mismatch")

    if metadata["feature_names"] != list(MODEL_FEATURES):
        raise ValueError("Saved model feature contract mismatch")

    if metadata["test_set_evaluated"] is not False:
        raise ValueError("Model metadata indicates prior test evaluation")

    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    if len(split.X_train) != metadata["training_rows"]:
        raise ValueError("Training row count mismatch")

    if len(split.X_test) != metadata["held_out_test_rows"]:
        raise ValueError("Held-out row count mismatch")

    pipeline = joblib.load(model_path)

    probabilities = np.asarray(
        pipeline.predict_proba(split.X_test)[:, 1],
        dtype=float,
    )

    if probabilities.shape != (len(split.X_test),):
        raise ValueError("Invalid probability output shape")

    if not np.isfinite(probabilities).all():
        raise ValueError("Non-finite predicted probabilities")

    if np.any((probabilities < 0) | (probabilities > 1)):
        raise ValueError("Predicted probabilities outside [0, 1]")

    predictions = (probabilities >= 0.5).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        split.y_test,
        predictions,
        labels=[0, 1],
    ).ravel()

    calibration = calculate_calibration(
        split.y_test,
        probabilities,
        n_bins=10,
    )

    result = {
        "evaluation_type": "single_held_out_test",
        "model_name": metadata["model_name"],
        "model_version": metadata["model_version"],
        "selection_status": metadata["selection_status"],
        "training_rows": len(split.X_train),
        "test_rows": len(split.X_test),
        "test_positive_count": int(split.y_test.sum()),
        "test_positive_rate": float(split.y_test.mean()),
        "decision_threshold": 0.5,
        "roc_auc": float(
            roc_auc_score(split.y_test, probabilities)
        ),
        "average_precision": float(
            average_precision_score(split.y_test, probabilities)
        ),
        "brier_score": calibration["brier_score"],
        "expected_calibration_error": calibration[
            "expected_calibration_error"
        ],
        "precision": float(
            precision_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "recall": float(
            recall_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "f1_score": float(
            f1_score(
                split.y_test,
                predictions,
                zero_division=0,
            )
        ),
        "confusion_matrix": {
            "true_negative": int(tn),
            "false_positive": int(fp),
            "false_negative": int(fn),
            "true_positive": int(tp),
        },
        "calibration": calibration,
        "model_artifact_sha256": metadata["artifact_sha256"],
        "test_set_evaluated": True,
        "limitations": [
            "TARGET indicates repayment difficulty, not necessarily formal default.",
            "Predicted probabilities are not regulatory probability-of-default estimates.",
            "The dataset has no event dates, so temporal validation is unavailable.",
            "The 0.50 threshold is illustrative and not a credit decision policy.",
            "Evaluation does not establish fairness, causal validity, or production readiness.",
        ],
    }

    report_dir = project_root / "reports" / "modeling"
    report_dir.mkdir(parents=True, exist_ok=True)

    report_path = report_dir / "held_out_test_evaluation.json"

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = evaluate_held_out_model(project_root)

    print("Model:", result["model_name"])
    print("Test applications:", result["test_rows"])
    print("Test positives:", result["test_positive_count"])
    print(f"ROC-AUC: {result['roc_auc']:.6f}")
    print(f"Average precision: {result['average_precision']:.6f}")
    print(f"Brier score: {result['brier_score']:.6f}")
    print(
        "Expected calibration error:",
        f"{result['expected_calibration_error']:.6f}",
    )
    print(f"Precision at 0.50: {result['precision']:.6f}")
    print(f"Recall at 0.50: {result['recall']:.6f}")
    print(f"F1 at 0.50: {result['f1_score']:.6f}")
    print("Confusion matrix:", result["confusion_matrix"])
    print("Held-out evaluation completed:", result["test_set_evaluated"])


if __name__ == "__main__":
    main()