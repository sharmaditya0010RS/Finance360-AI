import pandas as pd
import pytest

from finance360.dashboard.hotspots import analyze_hotspots


@pytest.fixture
def segments():
    return pd.DataFrame([
        {"dimension": "Income type", "segment": "A", "applications": 2000, "difficulty_cases": 300, "portfolio_applications": 10000, "portfolio_difficulty_cases": 1000},
        {"dimension": "Income type", "segment": "B", "applications": 500, "difficulty_cases": 10, "portfolio_applications": 10000, "portfolio_difficulty_cases": 1000},
    ])


def test_rates_and_benchmark(segments):
    result = analyze_hotspots(segments, 100)
    assert result.iloc[0]["difficulty_rate"] == pytest.approx(.15)
    assert result.iloc[0]["portfolio_rate"] == pytest.approx(.10)
    assert result.iloc[0]["difference_pp"] == pytest.approx(5)
    assert result.iloc[0]["relative_rate"] == pytest.approx(1.5)


def test_sample_threshold(segments):
    assert analyze_hotspots(segments, 1000)["segment"].tolist() == ["A"]


@pytest.mark.parametrize("minimum", [0, -1, True, 1.5])
def test_invalid_threshold(segments, minimum):
    with pytest.raises(ValueError, match="minimum_applications"):
        analyze_hotspots(segments, minimum)


def test_invalid_counts(segments):
    segments.loc[0, "difficulty_cases"] = 3000
    with pytest.raises(ValueError, match="exceed"):
        analyze_hotspots(segments)


def test_inconsistent_benchmark(segments):
    segments.loc[0, "portfolio_applications"] = 9000
    with pytest.raises(ValueError, match="Inconsistent"):
        analyze_hotspots(segments)


def test_does_not_mutate_input(segments):
    original = segments.copy(deep=True)
    analyze_hotspots(segments)
    pd.testing.assert_frame_equal(segments, original)
