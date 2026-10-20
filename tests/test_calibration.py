import numpy as np
import pandas as pd
import pytest

from finance360.evaluation.calibration import calculate_calibration


def test_perfect_calibration():
    targets = pd.Series([0, 0, 1, 1])
    probabilities = np.array([0.0, 0.0, 1.0, 1.0])

    result = calculate_calibration(targets, probabilities)

    assert result["rows"] == 4
    assert result["positive_count"] == 2
    assert result["observed_positive_rate"] == 0.5
    assert result["mean_predicted_probability"] == 0.5
    assert result["brier_score"] == 0.0
    assert result["expected_calibration_error"] == 0.0
    assert result["populated_bins"] == 2
    assert sum(item["count"] for item in result["bins"]) == 4


def test_calibration_with_constant_probabilities():
    targets = np.array([0, 0, 1, 1])
    probabilities = np.array([0.25, 0.25, 0.25, 0.25])

    result = calculate_calibration(targets, probabilities)

    assert result["populated_bins"] == 1
    assert result["expected_calibration_error"] == pytest.approx(0.25)
    assert result["brier_score"] == pytest.approx(0.3125)


@pytest.mark.parametrize(
    ("targets", "probabilities"),
    [
        ([0, 1], [0.5]),
        ([], []),
        ([0, 2], [0.2, 0.8]),
        ([0, 1], [-0.1, 0.8]),
        ([0, 1], [0.2, 1.1]),
        ([0, 1], [0.2, np.nan]),
        ([0, 1], [0.2, np.inf]),
    ],
)
def test_calibration_rejects_invalid_inputs(targets, probabilities):
    with pytest.raises(ValueError):
        calculate_calibration(
            np.asarray(targets),
            np.asarray(probabilities),
        )


def test_calibration_rejects_invalid_bin_count():
    with pytest.raises(ValueError):
        calculate_calibration(
            np.array([0, 1]),
            np.array([0.2, 0.8]),
            n_bins=1,
        )