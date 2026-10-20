from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from finance360.dashboard.prediction import (
    assign_risk_band,
    predict_applicant,
    prepare_applicant_features,
    safe_ratio,
)
from finance360.modeling.feature_contract import MODEL_FEATURES


@pytest.fixture
def applicant() -> dict:
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
        "EXT_SOURCE_1": 0.5,
        "EXT_SOURCE_2": 0.6,
        "EXT_SOURCE_3": 0.7,
    }


def test_safe_ratio() -> None:
    assert safe_ratio(10, 2) == 5.0
    assert np.isnan(safe_ratio(10, 0))
    assert np.isnan(safe_ratio(10, -1))


def test_prepared_features_match_model_contract(applicant: dict) -> None:
    features = prepare_applicant_features(applicant)

    assert isinstance(features, pd.DataFrame)
    assert features.shape == (1, 24)
    assert features.columns.tolist() == MODEL_FEATURES


def test_financial_ratios_are_derived(applicant: dict) -> None:
    features = prepare_applicant_features(applicant)
    row = features.iloc[0]

    assert row["CREDIT_TO_INCOME_RATIO"] == pytest.approx(2.5)
    assert row["ANNUITY_TO_INCOME_RATIO"] == pytest.approx(25000 / 180000)
    assert row["CREDIT_TO_GOODS_RATIO"] == pytest.approx(1.0)
    assert row["INCOME_PER_FAMILY_MEMBER"] == pytest.approx(60000)


def test_external_source_features_are_derived(applicant: dict) -> None:
    features = prepare_applicant_features(applicant)
    row = features.iloc[0]

    assert row["EXT_SOURCE_AVAILABLE_COUNT"] == 3
    assert row["EXT_SOURCE_MEAN"] == pytest.approx(0.6)


def test_missing_external_sources_are_supported(applicant: dict) -> None:
    applicant.pop("EXT_SOURCE_1")
    applicant.pop("EXT_SOURCE_2")
    applicant.pop("EXT_SOURCE_3")

    features = prepare_applicant_features(applicant)
    row = features.iloc[0]

    assert row["EXT_SOURCE_AVAILABLE_COUNT"] == 0
    assert pd.isna(row["EXT_SOURCE_MEAN"])
    assert pd.isna(row["EXT_SOURCE_1"])


def test_missing_required_field_is_rejected(applicant: dict) -> None:
    applicant.pop("AMT_CREDIT")

    with pytest.raises(ValueError, match="Missing required"):
        prepare_applicant_features(applicant)


def test_invalid_income_is_rejected(applicant: dict) -> None:
    applicant["AMT_INCOME_TOTAL"] = 0

    with pytest.raises(ValueError, match="Income"):
        prepare_applicant_features(applicant)


def test_invalid_numerical_type_is_rejected(applicant: dict) -> None:
    applicant["AMT_CREDIT"] = "not a number"

    with pytest.raises(TypeError, match="Invalid numerical"):
        prepare_applicant_features(applicant)


def test_invalid_external_source_is_rejected(applicant: dict) -> None:
    applicant["EXT_SOURCE_2"] = 1.5

    with pytest.raises(ValueError, match="External source"):
        prepare_applicant_features(applicant)


def test_risk_band_assignment() -> None:
    boundaries = [0.05, 0.10, 0.18]

    assert assign_risk_band(0.01, boundaries) == "Low"
    assert assign_risk_band(0.05, boundaries) == "Moderate"
    assert assign_risk_band(0.10, boundaries) == "High"
    assert assign_risk_band(0.18, boundaries) == "Very High"


def test_invalid_risk_boundaries_are_rejected() -> None:
    with pytest.raises(ValueError, match="sorted"):
        assign_risk_band(0.1, [0.2, 0.1, 0.3])

    with pytest.raises(ValueError, match="Exactly three"):
        assign_risk_band(0.1, [0.2])


def test_saved_model_produces_valid_prediction(applicant: dict) -> None:
    project_root = Path(__file__).resolve().parents[1]

    model_path = (
        project_root
        / "artifacts"
        / "models"
        / "finance360_xgboost_v1.joblib"
    )

    if not model_path.is_file():
        pytest.skip("Local trained model artifact is unavailable")

    result = predict_applicant(applicant, project_root)

    assert 0 <= result["repayment_difficulty_probability"] <= 1
    assert result["risk_band"] in {
        "Low",
        "Moderate",
        "High",
        "Very High",
    }
    assert result["missing_external_sources"] == 0
    assert result["decision"] == "Decision support only"