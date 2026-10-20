from pathlib import Path

import pandas as pd

TARGET_COLUMN = "TARGET"
IDENTIFIER_COLUMN = "SK_ID_CURR"

CATEGORICAL_FEATURES = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

NUMERICAL_FEATURES = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "AGE_YEARS",
    "EMPLOYMENT_YEARS",
    "CREDIT_TO_INCOME_RATIO",
    "ANNUITY_TO_INCOME_RATIO",
    "CREDIT_TO_GOODS_RATIO",
    "INCOME_PER_FAMILY_MEMBER",
    "EXT_SOURCE_AVAILABLE_COUNT",
    "EXT_SOURCE_MEAN",
]

MODEL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES

EXPECTED_ROWS = 307511
EXPECTED_POSITIVES = 24825


def load_modeling_data(project_root: Path) -> tuple[pd.DataFrame, pd.Series]:
    silver_path = (
        project_root / "data" / "silver" / "credit_applications_silver.parquet"
    )

    if not silver_path.is_file():
        raise FileNotFoundError(f"Silver dataset not found: {silver_path}")

    data = pd.read_parquet(silver_path)

    required_columns = [IDENTIFIER_COLUMN, TARGET_COLUMN] + MODEL_FEATURES
    missing_columns = [
        column for column in required_columns if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(f"Missing modeling columns: {missing_columns}")

    if len(data) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} applications, found {len(data)}"
        )

    if data[IDENTIFIER_COLUMN].isna().any():
        raise ValueError("Application identifiers contain missing values")

    if not data[IDENTIFIER_COLUMN].is_unique:
        raise ValueError("Application identifiers are not unique")

    if data[TARGET_COLUMN].isna().any():
        raise ValueError("Target contains missing values")

    if not data[TARGET_COLUMN].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    if int(data[TARGET_COLUMN].sum()) != EXPECTED_POSITIVES:
        raise ValueError("Unexpected repayment-difficulty target count")

    features = data[MODEL_FEATURES].copy()
    target = data[TARGET_COLUMN].astype(int).copy()

    for column in NUMERICAL_FEATURES:
        features[column] = pd.to_numeric(features[column], errors="raise")

    for column in CATEGORICAL_FEATURES:
        features[column] = features[column].astype("object")

    return features, target