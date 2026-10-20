import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERICAL_FEATURES,
)
from finance360.monitoring.risk_segmentation import RISK_BANDS

MODEL_PATH = Path("artifacts/models/finance360_xgboost_v1.joblib")
RISK_REPORT_PATH = Path("reports/monitoring/risk_segmentation.json")


def safe_ratio(numerator: float, denominator: float) -> float:
    if not np.isfinite(numerator) or not np.isfinite(denominator):
        return float("nan")

    if denominator <= 0:
        return float("nan")

    return float(numerator / denominator)


def prepare_applicant_features(applicant: dict) -> pd.DataFrame:
    required_fields = (
        "NAME_CONTRACT_TYPE",
        "NAME_INCOME_TYPE",
        "NAME_EDUCATION_TYPE",
        "NAME_FAMILY_STATUS",
        "NAME_HOUSING_TYPE",
        "FLAG_OWN_CAR",
        "FLAG_OWN_REALTY",
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
    )

    missing_fields = [
        field for field in required_fields if field not in applicant
    ]

    if missing_fields:
        raise ValueError(
            f"Missing required applicant fields: {missing_fields}"
        )

    features = dict(applicant)

    for field in CATEGORICAL_FEATURES:
        value = features[field]

        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"Invalid categorical field: {field}")

        features[field] = value.strip()

    for field in (
        "AMT_INCOME_TOTAL",
        "AMT_CREDIT",
        "AMT_ANNUITY",
        "AMT_GOODS_PRICE",
        "CNT_CHILDREN",
        "CNT_FAM_MEMBERS",
        "AGE_YEARS",
        "EMPLOYMENT_YEARS",
    ):
        value = features[field]

        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"Invalid numerical field: {field}")

        if not np.isfinite(value):
            raise ValueError(f"Non-finite numerical field: {field}")

        features[field] = float(value)

    if features["AMT_INCOME_TOTAL"] <= 0:
        raise ValueError("Income must be greater than zero")

    if features["AMT_CREDIT"] <= 0:
        raise ValueError("Requested credit must be greater than zero")

    if features["AMT_ANNUITY"] <= 0:
        raise ValueError("Annuity must be greater than zero")

    if features["AMT_GOODS_PRICE"] <= 0:
        raise ValueError("Goods price must be greater than zero")

    if features["CNT_CHILDREN"] < 0:
        raise ValueError("Children count cannot be negative")

    if features["CNT_FAM_MEMBERS"] < 1:
        raise ValueError("Family member count must be at least one")

    if not 18 <= features["AGE_YEARS"] <= 100:
        raise ValueError("Age must be between 18 and 100")

    if features["EMPLOYMENT_YEARS"] < 0:
        raise ValueError("Employment years cannot be negative")

    if features["EMPLOYMENT_YEARS"] > features["AGE_YEARS"] - 14:
        raise ValueError("Employment years are inconsistent with age")

    external_sources = []

    for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        value = features.get(field)

        if value is None:
            features[field] = float("nan")
            continue

        if isinstance(value, bool) or not isinstance(
            value, (int, float, np.integer, np.floating)
        ):
            raise TypeError(f"Invalid external source: {field}")

        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f"External source must be between 0 and 1: {field}")

        features[field] = float(value)
        external_sources.append(float(value))

    income = features["AMT_INCOME_TOTAL"]
    credit = features["AMT_CREDIT"]
    annuity = features["AMT_ANNUITY"]
    goods_price = features["AMT_GOODS_PRICE"]
    family_members = features["CNT_FAM_MEMBERS"]

    features["CREDIT_TO_INCOME_RATIO"] = safe_ratio(credit, income)
    features["ANNUITY_TO_INCOME_RATIO"] = safe_ratio(annuity, income)
    features["CREDIT_TO_GOODS_RATIO"] = safe_ratio(credit, goods_price)
    features["INCOME_PER_FAMILY_MEMBER"] = safe_ratio(
        income,
        family_members,
    )
    features["EXT_SOURCE_AVAILABLE_COUNT"] = len(external_sources)
    features["EXT_SOURCE_MEAN"] = (
        float(np.mean(external_sources))
        if external_sources
        else float("nan")
    )

    frame = pd.DataFrame(
        [{field: features[field] for field in MODEL_FEATURES}]
    )

    for field in NUMERICAL_FEATURES:
        frame[field] = pd.to_numeric(frame[field], errors="raise")

    for field in CATEGORICAL_FEATURES:
        frame[field] = frame[field].astype(object)

    return frame[MODEL_FEATURES]


def assign_risk_band(
    probability: float,
    boundaries: list[float],
) -> str:
    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Probability must be between 0 and 1")

    if len(boundaries) != 3:
        raise ValueError("Exactly three risk boundaries are required")

    if not all(
        np.isfinite(boundary) and 0 <= boundary <= 1
        for boundary in boundaries
    ):
        raise ValueError("Risk boundaries must be finite probabilities")

    if list(boundaries) != sorted(boundaries):
        raise ValueError("Risk boundaries must be sorted")

    band_index = int(np.searchsorted(boundaries, probability, side="right"))

    return RISK_BANDS[band_index]


def predict_applicant(
    applicant: dict,
    project_root: Path,
) -> dict:
    features = prepare_applicant_features(applicant)

    model = joblib.load(project_root / MODEL_PATH)

    with (project_root / RISK_REPORT_PATH).open(
        encoding="utf-8"
    ) as file:
        risk_report = json.load(file)

    probability = float(model.predict_proba(features)[0, 1])

    if not np.isfinite(probability) or not 0 <= probability <= 1:
        raise ValueError("Model returned an invalid probability")

    boundaries = risk_report["boundaries"]
    risk_band = assign_risk_band(probability, boundaries)

    missing_external_sources = sum(
        pd.isna(features[field].iloc[0])
        for field in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3")
    )

    return {
        "repayment_difficulty_probability": probability,
        "repayment_difficulty_percentage": round(probability * 100, 2),
        "risk_band": risk_band,
        "risk_boundaries": boundaries,
        "model": "finance360_xgboost_v1",
        "missing_external_sources": int(missing_external_sources),
        "decision": "Decision support only",
        "limitations": (
            "This is an estimated repayment-difficulty probability, "
            "not an approval decision. Risk bands are relative to "
            "training-data score distributions. Missing anonymized "
            "external-source inputs can reduce prediction reliability."
        ),
    }