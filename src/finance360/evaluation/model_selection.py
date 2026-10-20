import json
from pathlib import Path

MODEL_NAMES = (
    "logistic_regression_baseline",
    "xgboost_challenger",
)


def select_provisional_model(project_root: Path) -> dict:
    report_directory = project_root / "reports" / "modeling"

    baseline_path = report_directory / "logistic_baseline_cv.json"
    challenger_path = report_directory / "xgboost_challenger_cv.json"
    calibration_path = report_directory / "model_calibration_oof.json"

    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    challenger = json.loads(challenger_path.read_text(encoding="utf-8"))
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))

    reports = {
        "logistic_regression_baseline": baseline,
        "xgboost_challenger": challenger,
    }

    if set(calibration["models"]) != set(MODEL_NAMES):
        raise ValueError("Calibration report has unexpected model names")

    for model_name, report in reports.items():
        if report["model"] != model_name:
            raise ValueError("Cross-validation model identity mismatch")

        if report["test_set_evaluated"]:
            raise ValueError("Held-out test set was already evaluated")

        if report["training_rows"] != calibration["training_rows"]:
            raise ValueError("Training row counts do not reconcile")

        if report["held_out_test_rows"] != calibration["held_out_test_rows"]:
            raise ValueError("Held-out test row counts do not reconcile")

    if calibration["test_set_evaluated"]:
        raise ValueError("Calibration report includes test-set evaluation")

    metrics = {}

    for model_name, report in reports.items():
        model_calibration = calibration["models"][model_name]["calibration"]

        metrics[model_name] = {
            "cv_roc_auc": report["summary"]["roc_auc"]["mean"],
            "cv_average_precision": report["summary"]["average_precision"]["mean"],
            "cv_brier_score": report["summary"]["brier_score"]["mean"],
            "oof_expected_calibration_error": model_calibration[
                "expected_calibration_error"
            ],
        }

    baseline_metrics = metrics["logistic_regression_baseline"]
    challenger_metrics = metrics["xgboost_challenger"]

    challenger_improves_ranking_and_brier = (
        challenger_metrics["cv_roc_auc"]
        > baseline_metrics["cv_roc_auc"]
        and challenger_metrics["cv_average_precision"]
        > baseline_metrics["cv_average_precision"]
        and challenger_metrics["cv_brier_score"]
        < baseline_metrics["cv_brier_score"]
    )

    if challenger_improves_ranking_and_brier:
        selected_model = "xgboost_challenger"
    else:
        selected_model = "logistic_regression_baseline"

    result = {
        "selection_status": "provisional",
        "selected_model": selected_model,
        "benchmark_model": "logistic_regression_baseline",
        "selection_rule": (
            "Select XGBoost only if it improves cross-validation "
            "ROC-AUC, average precision, and Brier score"
        ),
        "recalibration_status": "not_applied",
        "recalibration_reason": (
            "Observed calibration differences are small, and "
            "high-probability bins have limited sample sizes"
        ),
        "training_rows": calibration["training_rows"],
        "held_out_test_rows": calibration["held_out_test_rows"],
        "test_set_evaluated": False,
        "model_metrics": metrics,
        "limitations": [
            "Selection uses three stratified cross-validation folds",
            "Equal-width expected calibration error depends on binning",
            "Sparse high-risk bins have uncertain observed rates",
            "No temporal validation is possible with available dates",
            "TARGET represents recorded repayment difficulty",
            "No regulatory-grade probability-of-default claim",
        ],
    }

    output_path = report_directory / "model_selection.json"

    output_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = select_provisional_model(project_root)

    print(
        json.dumps(
            {
                "selection_status": result["selection_status"],
                "selected_model": result["selected_model"],
                "recalibration_status": result["recalibration_status"],
                "test_set_evaluated": result["test_set_evaluated"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()