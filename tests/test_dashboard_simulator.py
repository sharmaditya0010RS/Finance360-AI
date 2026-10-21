"""Unit tests for Finance360 scenario simulation."""

from pathlib import Path
from unittest.mock import patch

import pytest

from finance360.dashboard.simulator import compare_scenarios


@pytest.fixture
def applicant():
    return {
        "NAME_CONTRACT_TYPE": "Cash loans",
        "NAME_INCOME_TYPE": "Working",
        "NAME_EDUCATION_TYPE": "Secondary / secondary special",
        "NAME_FAMILY_STATUS": "Married",
        "NAME_HOUSING_TYPE": "House / apartment",
        "FLAG_OWN_CAR": "N",
        "FLAG_OWN_REALTY": "Y",
        "AMT_INCOME_TOTAL": 180000.0,
        "AMT_CREDIT": 450000.0,
        "AMT_ANNUITY": 25000.0,
        "AMT_GOODS_PRICE": 450000.0,
        "CNT_CHILDREN": 1,
        "CNT_FAM_MEMBERS": 3,
        "AGE_YEARS": 35.0,
        "EMPLOYMENT_YEARS": 8.0,
        "EXT_SOURCE_1": None,
        "EXT_SOURCE_2": None,
        "EXT_SOURCE_3": None,
    }


def fake_prediction(profile, root):
    probability = profile["AMT_CREDIT"] / 10000000
    return {"repayment_difficulty_probability": probability, "risk_band": "Moderate"}


def test_comparison_and_no_mutation(applicant):
    original_credit = applicant["AMT_CREDIT"]
    with patch("finance360.dashboard.simulator.predict_applicant", side_effect=fake_prediction) as predictor:
        result = compare_scenarios(applicant, {"AMT_CREDIT": 500000}, Path("."))
    assert predictor.call_count == 2
    assert result["baseline"]["repayment_difficulty_probability"] == pytest.approx(0.045)
    assert result["scenario"]["repayment_difficulty_probability"] == pytest.approx(0.05)
    assert result["percentage_point_delta"] == pytest.approx(0.5)
    assert applicant["AMT_CREDIT"] == original_credit


@pytest.mark.parametrize("changes", [{}, {"AGE_YEARS": 40}, {"AMT_CREDIT": -1}, {"AMT_CREDIT": float("nan")}, {"AMT_CREDIT": "500000"}])
def test_invalid_changes(applicant, changes):
    with pytest.raises((ValueError, TypeError)):
        compare_scenarios(applicant, changes, Path("."))


def test_invalid_baseline(applicant):
    applicant["AMT_INCOME_TOTAL"] = 0
    with pytest.raises(ValueError, match="Income"):
        compare_scenarios(applicant, {"AMT_CREDIT": 500000}, Path("."))
