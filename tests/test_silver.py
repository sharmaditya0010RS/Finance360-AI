import numpy as np
import pandas as pd
import pytest

from finance360.ingestion.home_credit import SOURCE_COLUMNS
from finance360.transformations.silver import (
    safe_divide,
    transform_applications,
)


def make_application() -> pd.DataFrame:
    row = dict.fromkeys(SOURCE_COLUMNS)

    row.update(
        {
            "SK_ID_CURR": 100001,
            "TARGET": 0,
            "AMT_INCOME_TOTAL": 120000.0,
            "AMT_CREDIT": 240000.0,
            "AMT_ANNUITY": 10000.0,
            "AMT_GOODS_PRICE": 200000.0,
            "DAYS_BIRTH": -14610,
            "DAYS_EMPLOYED": 365243,
            "CNT_FAM_MEMBERS": 2.0,
            "EXT_SOURCE_1": 0.2,
            "EXT_SOURCE_2": 0.4,
            "EXT_SOURCE_3": np.nan,
        }
    )

    return pd.DataFrame([row])


def test_safe_divide_handles_zero_denominator():
    numerator = pd.Series([10.0, 20.0])
    denominator = pd.Series([2.0, 0.0])

    result = safe_divide(numerator, denominator)

    assert result.iloc[0] == 5.0
    assert pd.isna(result.iloc[1])


def test_employment_sentinel_becomes_missing():
    result = transform_applications(make_application())

    assert result["EMPLOYMENT_SENTINEL_FLAG"].iloc[0] == 1
    assert pd.isna(result["EMPLOYMENT_YEARS"].iloc[0])
    assert result["DAYS_EMPLOYED"].iloc[0] == 365243


def test_financial_ratios():
    result = transform_applications(make_application())

    assert result["CREDIT_TO_INCOME_RATIO"].iloc[0] == 2.0
    assert result["ANNUITY_TO_INCOME_RATIO"].iloc[0] == 1.0
    assert result["CREDIT_TO_GOODS_RATIO"].iloc[0] == 1.2
    assert result["INCOME_PER_FAMILY_MEMBER"].iloc[0] == 60000.0


def test_external_score_aggregation():
    result = transform_applications(make_application())

    assert result["EXT_SOURCE_AVAILABLE_COUNT"].iloc[0] == 2
    assert result["EXT_SOURCE_MEAN"].iloc[0] == pytest.approx(0.3)


def test_original_columns_and_target_preserved():
    source = make_application()
    result = transform_applications(source)

    assert result.loc[:, SOURCE_COLUMNS].equals(source)
    assert result["TARGET"].tolist() == [0]


def test_duplicate_application_ids_are_rejected():
    source = pd.concat([make_application(), make_application()], ignore_index=True)

    with pytest.raises(ValueError, match="unique"):
        transform_applications(source)


def test_invalid_income_is_flagged():
    source = make_application()
    source.loc[0, "AMT_INCOME_TOTAL"] = 0

    result = transform_applications(source)

    assert result["INCOME_INVALID_FLAG"].iloc[0] == 1
    assert pd.isna(result["CREDIT_TO_INCOME_RATIO"].iloc[0])
    assert result["FINANCIAL_DATA_QUALITY_FLAG"].iloc[0] == 1