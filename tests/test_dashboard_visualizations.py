from pathlib import Path

import pytest

from finance360.dashboard.data_access import load_dashboard_data
from finance360.dashboard.visualizations import (
    applicant_vs_credit_share_chart,
    confusion_matrix_chart,
    contract_risk_chart,
    contract_volume_chart,
    credit_share_chart,
    exposure_chart,
    held_out_calibration_chart,
    income_risk_chart,
    income_volume_chart,
    model_evidence_chart,
    observed_vs_predicted_chart,
    portfolio_outcome_chart,
    risk_distribution_chart,
    risk_outcome_composition_chart,
    risk_score_range_chart,
    score_distribution_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def dashboard_data():
    return load_dashboard_data(PROJECT_ROOT)


CHART_CASES = [
    ("portfolio_summary", portfolio_outcome_chart),
    ("contract_performance", contract_risk_chart),
    ("contract_performance", contract_volume_chart),
    ("income_type_performance", income_risk_chart),
    ("income_type_performance", income_volume_chart),
    ("risk_segmentation", risk_distribution_chart),
    ("risk_segmentation", observed_vs_predicted_chart),
    ("risk_segmentation", risk_score_range_chart),
    ("risk_segmentation", risk_outcome_composition_chart),
    ("portfolio_exposure", exposure_chart),
    ("portfolio_exposure", credit_share_chart),
    ("portfolio_exposure", applicant_vs_credit_share_chart),
    ("held_out_evaluation", model_evidence_chart),
    ("held_out_evaluation", confusion_matrix_chart),
    ("held_out_evaluation", held_out_calibration_chart),
    ("score_monitoring", score_distribution_chart),
]


@pytest.mark.parametrize("data_key,chart_function", CHART_CASES)
def test_dashboard_chart_renders(dashboard_data, data_key, chart_function):
    figure = chart_function(dashboard_data[data_key])

    assert len(figure.data) > 0
    assert figure.layout.title.text
    assert figure.layout.height > 0

    serialized = figure.to_plotly_json()

    assert "data" in serialized
    assert "layout" in serialized


def test_dashboard_chart_count():
    assert len(CHART_CASES) == 16


def test_confusion_matrix_matches_held_out_population(dashboard_data):
    evaluation = dashboard_data["held_out_evaluation"]
    matrix = evaluation["confusion_matrix"]

    assert sum(matrix.values()) == evaluation["test_rows"]
    assert matrix["false_negative"] == 4895
    assert matrix["true_positive"] == 70


def test_risk_exposure_population_reconciles(dashboard_data):
    risk = dashboard_data["risk_segmentation"]
    exposure = dashboard_data["portfolio_exposure"]

    assert risk["training_rows"] == exposure["application_count"]
    assert sum(
        segment["application_count"] for segment in risk["segments"]
    ) == risk["training_rows"]


def test_historical_monitoring_groups_reconcile(dashboard_data):
    monitoring = dashboard_data["score_monitoring"]

    assert (
        monitoring["reference_count"] + monitoring["comparison_count"]
        == monitoring["training_rows"]
    )
    assert len(monitoring["psi_result"]["bins"]) == 10
    assert monitoring["production_drift_claim"] is False