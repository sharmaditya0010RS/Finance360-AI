import numpy as np
import pytest

from finance360.monitoring.score_monitoring import (
    calculate_score_monitoring,
)


def test_score_monitoring_counts():
    scores = np.array([0.05, 0.15, 0.25, 0.35])
    groups = np.array([0, 1, 0, 1])

    result = calculate_score_monitoring(scores, groups)

    assert result["reference_count"] == 2
    assert result["comparison_count"] == 2
    assert result["psi_result"]["reference_count"] == 2
    assert result["psi_result"]["comparison_count"] == 2


def test_score_monitoring_mean_scores():
    scores = np.array([0.05, 0.15, 0.25, 0.35])
    groups = np.array([0, 1, 0, 1])

    result = calculate_score_monitoring(scores, groups)

    assert result["reference_mean_score"] == pytest.approx(0.15)
    assert result["comparison_mean_score"] == pytest.approx(0.25)


def test_identical_groups_have_zero_psi():
    scores = np.array([0.05, 0.15, 0.05, 0.15])
    groups = np.array([0, 0, 1, 1])

    result = calculate_score_monitoring(scores, groups)

    assert result["psi_result"]["psi"] == pytest.approx(0.0)


def test_shifted_groups_have_positive_psi():
    scores = np.array([0.05] * 80 + [0.15] * 20 + [0.05] * 40 + [0.15] * 60)
    groups = np.array([0] * 100 + [1] * 100)

    result = calculate_score_monitoring(scores, groups)

    assert result["psi_result"]["psi"] > 0


def test_monitoring_is_not_production_claim():
    scores = np.array([0.05, 0.15])
    groups = np.array([0, 1])

    result = calculate_score_monitoring(scores, groups)

    assert result["production_drift_claim"] is False
    assert result["comparison_type"] == "non_temporal_training_oof_demonstration"


@pytest.mark.parametrize(
    ("scores", "groups", "error"),
    [
        ([], [], "nonempty and aligned"),
        ([0.1, 0.2], [0], "nonempty and aligned"),
        ([[0.1, 0.2]], [0, 1], "one-dimensional"),
        ([0.1, 0.2], [[0, 1]], "one-dimensional"),
        ([np.nan, 0.2], [0, 1], "finite"),
        ([np.inf, 0.2], [0, 1], "finite"),
        ([-0.1, 0.2], [0, 1], "between 0 and 1"),
        ([1.1, 0.2], [0, 1], "between 0 and 1"),
        ([0.1, 0.2], [0, 2], "only 0 and 1"),
        ([0.1, 0.2], [0, 0], "Both comparison groups"),
        ([0.1, 0.2], [1, 1], "Both comparison groups"),
    ],
)
def test_invalid_monitoring_inputs(scores, groups, error):
    with pytest.raises(ValueError, match=error):
        calculate_score_monitoring(scores, groups)