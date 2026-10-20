import hashlib
import json
from pathlib import Path

import pandas as pd

SOURCE_COLUMNS = [
    "SK_ID_CURR",
    "TARGET",
    "NAME_CONTRACT_TYPE",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
]

EXPECTED_ROWS = 307511
EXPECTED_SOURCE_COLUMNS = 122


def calculate_sha256(path: Path) -> str:
    digest = hashlib.sha256()

    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def ingest_application_data(project_root: Path) -> dict:
    source_path = project_root / "data" / "external" / "application_train.csv"
    bronze_directory = project_root / "data" / "bronze"
    bronze_path = bronze_directory / "home_credit_applications.parquet"
    manifest_path = bronze_directory / "ingestion_manifest.json"

    if not source_path.is_file():
        raise FileNotFoundError(f"Source dataset not found: {source_path}")

    source_header = pd.read_csv(source_path, nrows=0).columns.tolist()

    if len(source_header) != EXPECTED_SOURCE_COLUMNS:
        raise ValueError(
            f"Expected {EXPECTED_SOURCE_COLUMNS} source columns, found {len(source_header)}"
        )

    if len(source_header) != len(set(source_header)):
        raise ValueError("Source dataset contains duplicate column names")

    missing_columns = [
        column for column in SOURCE_COLUMNS if column not in source_header
    ]

    if missing_columns:
        raise ValueError(f"Required columns missing: {missing_columns}")

    frame = pd.read_csv(source_path, usecols=SOURCE_COLUMNS)
    frame = frame.loc[:, SOURCE_COLUMNS]

    if len(frame) != EXPECTED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_ROWS} rows, found {len(frame)}"
        )

    if frame.shape[1] != len(SOURCE_COLUMNS):
        raise ValueError("Curated dataset must contain exactly 20 columns")

    if frame["SK_ID_CURR"].isna().any():
        raise ValueError("Application identifiers contain null values")

    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError("Application identifiers are not unique")

    if frame["TARGET"].isna().any():
        raise ValueError("Target contains null values")

    if not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("Target must contain only 0 and 1")

    bronze_directory.mkdir(parents=True, exist_ok=True)

    frame.to_parquet(bronze_path, index=False)

    manifest = {
        "dataset": "Home Credit Default Risk",
        "source_file": source_path.name,
        "source_sha256": calculate_sha256(source_path),
        "source_column_count": len(source_header),
        "curated_column_count": len(SOURCE_COLUMNS),
        "row_count": len(frame),
        "target_positive_count": int(frame["TARGET"].sum()),
        "target_positive_rate": float(frame["TARGET"].mean()),
        "bronze_file": bronze_path.name,
        "selected_columns": SOURCE_COLUMNS,
    }

    manifest_path.write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )

    return manifest


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    manifest = ingest_application_data(project_root)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()