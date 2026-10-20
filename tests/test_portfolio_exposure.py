import numpy as np
import pytest

from finance360.monitoring.portfolio_exposure import (
    calculate_portfolio_exposure,
)


@pytest.fixture
def sample_portfolio():
    scores = np.array(
        [0.02, 0.04, 0.06, 0.08, 0.12, 0.15, 0.20, 0.30]
    )
    targets = np.array([0, 0, 0, 0, 0, 1, 1, 1])
    credit = np.array(
        [100, 200, 300, 400, 500, 600, 700, 800],
        dtype=float,
    )
    boundaries = np.array([0.05, 0.10, 0.18])

    return scores, targets, credit, boundaries


def test_portfolio_counts_reconcile(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)

    assert result["application_count"] == 8
    assert result["positive_count"] == 3
    assert result["total_requested_credit"] == pytest.approx(3600)
    assert sum(
        segment["application_count"] for segment in result["segments"]
    ) == 8


def test_portfolio_band_assignments(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)

    assert [
        segment["application_count"] for segment in result["segments"]
    ] == [2, 2, 2, 2]

    assert [
        segment["positive_count"] for segment in result["segments"]
    ] == [0, 0, 1, 2]


def test_portfolio_credit_by_band(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)

    assert [
        segment["total_requested_credit"] for segment in result["segments"]
    ] == pytest.approx([300, 700, 1100, 1500])


def test_portfolio_shares_reconcile(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)

    assert sum(
        segment["application_share"] for segment in result["segments"]
    ) == pytest.approx(1.0)

    assert sum(
        segment["requested_credit_share"] for segment in result["segments"]
    ) == pytest.approx(1.0)


def test_elevated_risk_summary(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)
    elevated = result["high_and_very_high"]

    assert elevated["application_count"] == 4
    assert elevated["application_share"] == pytest.approx(0.5)
    assert elevated["total_requested_credit"] == pytest.approx(2600)
    assert elevated["requested_credit_share"] == pytest.approx(
        2600 / 3600
    )


def test_portfolio_average_credit(sample_portfolio):
    result = calculate_portfolio_exposure(*sample_portfolio)

    assert [
        segment["average_requested_credit"] for segment in result["segments"]
    ] == pytest.approx([150, 350, 550, 750])


@pytest.mark.parametrize(
    ("scores", "targets", "credit", "boundaries", "error"),
    [
        ([], [], [], [0.05, 0.1, 0.2], "nonempty and aligned"),
        ([0.1, 0.2], [0], [100, 200], [0.05, 0.1, 0.2], "nonempty and aligned"),
        ([[0.1, 0.2]], [0, 1], [100, 200], [0.05, 0.1, 0.2], "one-dimensional"),
        ([0.1, 0.2], [0, 1], [100, 200], [0.1, 0.2], "three risk-band"),
        ([0.1, np.nan], [0, 1], [100, 200], [0.05, 0.1, 0.2], "finite"),
        ([0.1, 0.2], [0, 1], [100, np.inf], [0.05, 0.1, 0.2], "finite"),
        ([-0.1, 0.2], [0, 1], [100, 200], [0.05, 0.1, 0.2], "between 0 and 1"),
        ([0.1, 1.1], [0, 1], [100, 200], [0.05, 0.1, 0.2], "between 0 and 1"),
        ([0.1, 0.2], [0, 1], [-100, 200], [0.05, 0.1, 0.2], "negative"),
        ([0.1, 0.2], [0, 2], [100, 200], [0.05, 0.1, 0.2], "binary"),
        ([0.1, 0.2], [0, 1], [100, 200], [0.05, 0.05, 0.2], "strictly increasing"),
        ([0.1, 0.2], [0, 1], [100, 200], [0.0, 0.1, 0.2], "between 0 and 1"),
        ([0.1, 0.2], [0, 1], [0, 0], [0.05, 0.1, 0.2], "must be positive"),
    ],
)
def test_portfolio_invalid_inputs(
    scores,
    targets,
    credit,
    boundaries,
    error,
):
    with pytest.raises(ValueError, match=error):
        calculate_portfolio_exposure(
            scores,
            targets,
            credit,
            boundaries,
        )


def test_portfolio_rejects_empty_band():
    with pytest.raises(ValueError, match="Every risk band"):
        calculate_portfolio_exposure(
            np.array([0.01, 0.02, 0.03, 0.04]),
            np.array([0, 0, 1, 1]),
            np.array([100, 200, 300, 400]),
            np.array([0.05, 0.10, 0.20]),
        )