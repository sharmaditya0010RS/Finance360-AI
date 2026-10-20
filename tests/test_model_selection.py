import json

import pytest

from finance360.evaluation.model_selection import select_provisional_model


def create_test_reports(
    project_root,
    challenger_auc=0.75,
    challenger_ap=0.23,
    challenger_brier=0.068,
    test_set_evaluated=False,
):
    report_directory = project_root / "reports" / "modeling"
    report_directory.mkdir(parents=True, exist_ok=True)

    baseline = {
        "model": "logistic_regression_baseline",
        "training_rows": 96,
        "held_out_test_rows": 24,
        "test_set_evaluated": test_set_evaluated,
        "summary": {
            "roc_auc": {"mean": 0.74},
            "average_precision": {"mean": 0.22},
            "brier_score": {"mean": 0.069},
        },
    }

    challenger = {
        "model": "xgboost_challenger",
        "training_rows": 96,
        "held_out_test_rows": 24,
        "test_set_evaluated": False,
        "summary": {
            "roc_auc": {"mean": challenger_auc},
            "average_precision": {"mean": challenger_ap},
            "brier_score": {"mean": challenger_brier},
        },
    }

    calibration = {
        "training_rows": 96,
        "held_out_test_rows": 24,
        "test_set_evaluated": False,
        "models": {
            "logistic_regression_baseline": {
                "calibration": {
                    "expected_calibration_error": 0.001,
                }
            },
            "xgboost_challenger": {
                "calibration": {
                    "expected_calibration_error": 0.002,
                }
            },
        },
    }

    reports = {
        "logistic_baseline_cv.json": baseline,
        "xgboost_challenger_cv.json": challenger,
        "model_calibration_oof.json": calibration,
    }

    for filename, report in reports.items():
        (report_directory / filename).write_text(
            json.dumps(report),
            encoding="utf-8",
        )

    return report_directory


def test_selects_xgboost_when_all_primary_metrics_improve(tmp_path):
    report_directory = create_test_reports(tmp_path)

    result = select_provisional_model(tmp_path)

    assert result["selection_status"] == "provisional"
    assert result["selected_model"] == "xgboost_challenger"
    assert result["benchmark_model"] == "logistic_regression_baseline"
    assert result["recalibration_status"] == "not_applied"
    assert result["training_rows"] == 96
    assert result["held_out_test_rows"] == 24
    assert result["test_set_evaluated"] is False

    output_path = report_directory / "model_selection.json"

    assert output_path.is_file()
    assert json.loads(output_path.read_text(encoding="utf-8")) == result


@pytest.mark.parametrize(
    ("challenger_auc", "challenger_ap", "challenger_brier"),
    [
        (0.73, 0.23, 0.068),
        (0.75, 0.21, 0.068),
        (0.75, 0.23, 0.070),
        (0.74, 0.23, 0.068),
        (0.75, 0.22, 0.068),
        (0.75, 0.23, 0.069),
    ],
)
def test_selects_baseline_if_challenger_fails_any_metric(
    tmp_path,
    challenger_auc,
    challenger_ap,
    challenger_brier,
):
    create_test_reports(
        tmp_path,
        challenger_auc=challenger_auc,
        challenger_ap=challenger_ap,
        challenger_brier=challenger_brier,
    )

    result = select_provisional_model(tmp_path)

    assert result["selected_model"] == "logistic_regression_baseline"


def test_rejects_prior_test_set_evaluation(tmp_path):
    create_test_reports(
        tmp_path,
        test_set_evaluated=True,
    )

    with pytest.raises(
        ValueError,
        match="Held-out test set was already evaluated",
    ):
        select_provisional_model(tmp_path)


def test_rejects_inconsistent_training_rows(tmp_path):
    report_directory = create_test_reports(tmp_path)

    calibration_path = report_directory / "model_calibration_oof.json"
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )

    calibration["training_rows"] = 95

    calibration_path.write_text(
        json.dumps(calibration),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Training row counts do not reconcile",
    ):
        select_provisional_model(tmp_path)


def test_rejects_unexpected_calibration_models(tmp_path):
    report_directory = create_test_reports(tmp_path)

    calibration_path = report_directory / "model_calibration_oof.json"
    calibration = json.loads(
        calibration_path.read_text(encoding="utf-8")
    )

    del calibration["models"]["xgboost_challenger"]

    calibration_path.write_text(
        json.dumps(calibration),
        encoding="utf-8",
    )

    with pytest.raises(
        ValueError,
        match="Calibration report has unexpected model names",
    ):
        select_provisional_model(tmp_path)