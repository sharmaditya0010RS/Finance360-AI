import json
from pathlib import Path

import numpy as np
import pandas as pd

from finance360.ingestion.home_credit import SOURCE_COLUMNS


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    valid_denominator = denominator.where(denominator > 0)
    result = numerator.div(valid_denominator)
    return result.replace([np.inf, -np.inf], np.nan)


def transform_applications(bronze: pd.DataFrame) -> pd.DataFrame:
    missing_columns = [
        column for column in SOURCE_COLUMNS if column not in bronze.columns
    ]

    if missing_columns:
        raise ValueError(f"Required source columns missing: {missing_columns}")

    silver = bronze.loc[:, SOURCE_COLUMNS].copy()

    if silver["SK_ID_CURR"].isna().any():
        raise ValueError("Application identifiers contain null values")

    if silver["SK_ID_CURR"].duplicated().any():
        raise ValueError("Application identifiers must be unique")

    if silver["TARGET"].isna().any():
        raise ValueError("Target contains null values")

    if not silver["TARGET"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    silver["EMPLOYMENT_SENTINEL_FLAG"] = (
        silver["DAYS_EMPLOYED"].eq(365243).astype("int8")
    )

    silver["AGE_YEARS"] = -silver["DAYS_BIRTH"] / 365.25

    silver["EMPLOYMENT_YEARS"] = (
        -silver["DAYS_EMPLOYED"].replace(365243, np.nan) / 365.25
    )

    silver["AGE_INVALID_FLAG"] = (
        silver["AGE_YEARS"].isna()
        | ~silver["AGE_YEARS"].between(18, 100)
    ).astype("int8")

    silver["EMPLOYMENT_INVALID_FLAG"] = (
        silver["EMPLOYMENT_YEARS"].notna()
        & (
            (silver["EMPLOYMENT_YEARS"] < 0)
            | (silver["EMPLOYMENT_YEARS"] > 80)
        )
    ).astype("int8")

    silver["INCOME_INVALID_FLAG"] = (
        silver["AMT_INCOME_TOTAL"].isna()
        | (silver["AMT_INCOME_TOTAL"] <= 0)
    ).astype("int8")

    silver["CREDIT_INVALID_FLAG"] = (
        silver["AMT_CREDIT"].isna()
        | (silver["AMT_CREDIT"] <= 0)
    ).astype("int8")

    silver["ANNUITY_INVALID_FLAG"] = (
        silver["AMT_ANNUITY"].notna()
        & (silver["AMT_ANNUITY"] <= 0)
    ).astype("int8")

    silver["CREDIT_TO_INCOME_RATIO"] = safe_divide(
        silver["AMT_CREDIT"],
        silver["AMT_INCOME_TOTAL"],
    )

    silver["ANNUITY_TO_INCOME_RATIO"] = safe_divide(
        silver["AMT_ANNUITY"] * 12,
        silver["AMT_INCOME_TOTAL"],
    )

    silver["CREDIT_TO_GOODS_RATIO"] = safe_divide(
        silver["AMT_CREDIT"],
        silver["AMT_GOODS_PRICE"],
    )

    silver["INCOME_PER_FAMILY_MEMBER"] = safe_divide(
        silver["AMT_INCOME_TOTAL"],
        silver["CNT_FAM_MEMBERS"],
    )

    external_columns = [
        "EXT_SOURCE_1",
        "EXT_SOURCE_2",
        "EXT_SOURCE_3",
    ]

    silver["EXT_SOURCE_AVAILABLE_COUNT"] = (
        silver[external_columns].notna().sum(axis=1).astype("int8")
    )

    silver["EXT_SOURCE_MEAN"] = silver[external_columns].mean(axis=1)

    silver["EXT_SOURCE_INVALID_FLAG"] = (
        (
            silver[external_columns].notna()
            & ~silver[external_columns].ge(0)
        )
        | (
            silver[external_columns].notna()
            & ~silver[external_columns].le(1)
        )
    ).any(axis=1).astype("int8")

    silver["FINANCIAL_DATA_QUALITY_FLAG"] = (
        silver[
            [
                "AGE_INVALID_FLAG",
                "EMPLOYMENT_INVALID_FLAG",
                "INCOME_INVALID_FLAG",
                "CREDIT_INVALID_FLAG",
                "ANNUITY_INVALID_FLAG",
                "EXT_SOURCE_INVALID_FLAG",
            ]
        ]
        .any(axis=1)
        .astype("int8")
    )

    return silver


def build_silver_dataset(root: Path) -> dict:
    source = root / "data" / "bronze" / "home_credit_applications.parquet"
    destination = root / "data" / "silver"
    silver_path = destination / "credit_applications_silver.parquet"
    report_directory = root / "reports" / "data_quality"

    if not source.is_file():
        raise FileNotFoundError(f"Bronze dataset not found: {source}")

    bronze = pd.read_parquet(source)
    silver = transform_applications(bronze)

    if len(silver) != len(bronze):
        raise ValueError("Transformation changed the application count")

    if not silver["SK_ID_CURR"].equals(bronze["SK_ID_CURR"]):
        raise ValueError("Transformation changed application identifiers")

    if not silver["TARGET"].equals(bronze["TARGET"]):
        raise ValueError("Transformation changed target values")

    destination.mkdir(parents=True, exist_ok=True)
    report_directory.mkdir(parents=True, exist_ok=True)

    silver.to_parquet(silver_path, index=False)

    report = {
        "bronze_rows": len(bronze),
        "silver_rows": len(silver),
        "original_columns_retained": len(SOURCE_COLUMNS),
        "silver_columns": len(silver.columns),
        "employment_sentinel_count": int(
            silver["EMPLOYMENT_SENTINEL_FLAG"].sum()
        ),
        "financial_quality_flag_count": int(
            silver["FINANCIAL_DATA_QUALITY_FLAG"].sum()
        ),
        "missing_values": {
            column: int(count)
            for column, count in silver.isna().sum().items()
        },
        "target_positive_count": int(silver["TARGET"].sum()),
        "target_positive_rate": float(silver["TARGET"].mean()),
    }

    report_path = report_directory / "silver_quality_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    return report


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    report = build_silver_dataset(root)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()