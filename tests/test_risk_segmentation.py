import numpy as np
import pytest

from finance360.monitoring.risk_segmentation import (
    RISK_BANDS,
    RISK_QUANTILES,
    calculate_risk_bands,
)


@pytest.fixture
def sample_risk_data():
    probabilities = np.linspace(0.001, 0.999, 100)
    targets = np.array([0] * 80 + [1] * 20)

    return probabilities, targets


def test_risk_band_configuration():
    assert RISK_BANDS == ("Low", "Moderate", "High", "Very High")
    assert RISK_QUANTILES == (0.50, 0.75, 0.90)


def test_risk_band_boundaries_match_quantiles(sample_risk_data):
    probabilities, targets = sample_risk_data

    result = calculate_risk_bands(probabilities, targets)

    expected = np.quantile(probabilities, RISK_QUANTILES)

    assert result["boundaries"] == pytest.approx(expected.tolist())


def test_risk_band_counts_reconcile(sample_risk_data):
    probabilities, targets = sample_risk_data

    result = calculate_risk_bands(probabilities, targets)

    segments = result["segments"]

    assert len(segments) == 4
    assert [item["application_count"] for item in segments] == [
        50,
        25,
        15,
        10,
    ]
    assert sum(item["application_count"] for item in segments) == 100
    assert sum(item["positive_count"] for item in segments) == 20
    assert sum(item["portfolio_share"] for item in segments) == pytest.approx(
        1.0
    )


def test_risk_band_difficulty_rates_increase(sample_risk_data):
    probabilities, targets = sample_risk_data

    result = calculate_risk_bands(probabilities, targets)

    rates = [
        segment["observed_difficulty_rate"]
        for segment in result["segments"]
    ]

    assert rates == pytest.approx([0.0, 0.0, 2 / 3, 1.0])


def test_risk_band_probability_order(sample_risk_data):
    probabilities, targets = sample_risk_data

    result = calculate_risk_bands(probabilities, targets)

    means = [
        segment["mean_predicted_probability"]
        for segment in result["segments"]
    ]

    assert means == sorted(means)


@pytest.mark.parametrize(
    ("probabilities", "targets", "error"),
    [
        ([], [], "nonempty and aligned"),
        ([0.1, 0.2], [0], "nonempty and aligned"),
        ([[0.1, 0.2]], [0, 1], "one-dimensional"),
        ([0.1, np.nan], [0, 1], "finite"),
        ([0.1, np.inf], [0, 1], "finite"),
        ([-0.1, 0.2], [0, 1], "between 0 and 1"),
        ([0.1, 1.1], [0, 1], "between 0 and 1"),
        ([0.1, 0.2], [0, 2], "binary"),
    ],
)
def test_risk_band_invalid_inputs(probabilities, targets, error):
    with pytest.raises(ValueError, match=error):
        calculate_risk_bands(probabilities, targets)


def test_risk_band_rejects_duplicate_boundaries():
    probabilities = np.full(100, 0.2)
    targets = np.array([0, 1] * 50)

    with pytest.raises(
        ValueError,
        match="strictly increasing",
    ):
        calculate_risk_bands(probabilities, targets)