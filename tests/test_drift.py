import numpy as np
import pytest

from finance360.monitoring.drift import (
    DEFAULT_BIN_EDGES,
    MINIMUM_PROPORTION,
    calculate_psi,
)


def test_default_configuration():
    assert len(DEFAULT_BIN_EDGES) == 11
    assert DEFAULT_BIN_EDGES[0] == 0.0
    assert DEFAULT_BIN_EDGES[-1] == 1.0
    assert MINIMUM_PROPORTION == 1e-6


def test_identical_distributions_have_zero_psi():
    scores = np.array([0.01, 0.05, 0.15, 0.25, 0.45, 0.85])

    result = calculate_psi(scores, scores.copy())

    assert result["psi"] == pytest.approx(0.0)
    assert result["reference_count"] == 6
    assert result["comparison_count"] == 6
    assert len(result["bins"]) == 10


def test_equal_proportions_have_zero_psi():
    reference = np.array([0.05, 0.05, 0.15, 0.15])
    comparison = np.array([0.05, 0.15])

    result = calculate_psi(reference, comparison)

    assert result["psi"] == pytest.approx(0.0)


def test_shifted_distribution_has_positive_psi():
    reference = np.array([0.05] * 80 + [0.15] * 20)
    comparison = np.array([0.05] * 40 + [0.15] * 60)

    result = calculate_psi(reference, comparison)

    assert result["psi"] > 0
    assert sum(
        item["psi_contribution"] for item in result["bins"]
    ) == pytest.approx(result["psi"])


def test_psi_matches_manual_calculation():
    reference = np.array([0.05] * 75 + [0.15] * 25)
    comparison = np.array([0.05] * 50 + [0.15] * 50)

    result = calculate_psi(reference, comparison)

    expected = (
        (0.50 - 0.75) * np.log(0.50 / 0.75)
        + (0.50 - 0.25) * np.log(0.50 / 0.25)
    )

    assert result["psi"] == pytest.approx(expected)


def test_custom_bin_edges():
    reference = np.array([0.1, 0.3, 0.6, 0.9])
    comparison = np.array([0.1, 0.4, 0.7, 0.9])

    result = calculate_psi(
        reference,
        comparison,
        bin_edges=np.array([0.0, 0.5, 1.0]),
    )

    assert len(result["bins"]) == 2
    assert result["psi"] == pytest.approx(0.0)


def test_boundary_probabilities_are_counted():
    reference = np.array([0.0, 0.1, 0.5, 1.0])
    comparison = np.array([0.0, 0.1, 0.5, 1.0])

    result = calculate_psi(reference, comparison)

    assert sum(item["reference_count"] for item in result["bins"]) == 4
    assert sum(item["comparison_count"] for item in result["bins"]) == 4


def test_empty_bins_use_smoothing():
    reference = np.array([0.05] * 10)
    comparison = np.array([0.95] * 10)

    result = calculate_psi(reference, comparison)

    assert np.isfinite(result["psi"])
    assert result["psi"] > 0
    assert result["smoothing_floor"] == MINIMUM_PROPORTION


def test_psi_is_nonnegative():
    reference = np.array([0.05, 0.05, 0.15, 0.35, 0.75])
    comparison = np.array([0.05, 0.25, 0.25, 0.45, 0.95])

    result = calculate_psi(reference, comparison)

    assert result["psi"] >= 0


@pytest.mark.parametrize(
    ("reference", "comparison", "error"),
    [
        ([], [0.1], "nonempty"),
        ([0.1], [], "nonempty"),
        ([[0.1]], [0.1], "one-dimensional"),
        ([0.1], [[0.1]], "one-dimensional"),
        ([np.nan], [0.1], "finite"),
        ([0.1], [np.inf], "finite"),
        ([-0.1], [0.1], "between 0 and 1"),
        ([1.1], [0.1], "between 0 and 1"),
        ([0.1], [-0.1], "between 0 and 1"),
        ([0.1], [1.1], "between 0 and 1"),
    ],
)
def test_invalid_score_arrays(reference, comparison, error):
    with pytest.raises(ValueError, match=error):
        calculate_psi(reference, comparison)


@pytest.mark.parametrize(
    ("edges", "error"),
    [
        ([0.0, 1.0], "at least two bins"),
        ([[0.0, 0.5, 1.0]], "at least two bins"),
        ([0.0, np.nan, 1.0], "finite"),
        ([0.1, 0.5, 1.0], "span 0 to 1"),
        ([0.0, 0.5, 0.9], "span 0 to 1"),
        ([0.0, 0.5, 0.5, 1.0], "strictly increasing"),
        ([0.0, 0.8, 0.4, 1.0], "strictly increasing"),
    ],
)
def test_invalid_bin_edges(edges, error):
    with pytest.raises(ValueError, match=error):
        calculate_psi(
            np.array([0.1, 0.9]),
            np.array([0.2, 0.8]),
            bin_edges=edges,
        )