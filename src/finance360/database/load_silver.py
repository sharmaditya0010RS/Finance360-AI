import io
import json
from pathlib import Path

import pandas as pd

from finance360.database.connection import get_connection
from finance360.ingestion.home_credit import SOURCE_COLUMNS


EXPECTED_ROWS = 307511
EXPECTED_COLUMNS = 36
EXPECTED_POSITIVE_TARGETS = 24825


def validate_silver(frame: pd.DataFrame) -> None:
    if frame.shape != (EXPECTED_ROWS, EXPECTED_COLUMNS):
        raise ValueError(f"Unexpected Silver shape: {frame.shape}")

    if frame.columns.duplicated().any():
        raise ValueError("Silver contains duplicate column names")

    if frame["SK_ID_CURR"].isna().any():
        raise ValueError("Application identifiers contain missing values")

    if frame["SK_ID_CURR"].duplicated().any():
        raise ValueError("Application identifiers are duplicated")

    if frame["TARGET"].isna().any():
        raise ValueError("Target contains missing values")

    if not frame["TARGET"].isin([0, 1]).all():
        raise ValueError("Target contains invalid values")

    if int(frame["TARGET"].sum()) != EXPECTED_POSITIVE_TARGETS:
        raise ValueError("Target total does not match the source baseline")

    if frame.columns[:20].tolist() != SOURCE_COLUMNS:
        raise ValueError("Original Silver column order has changed")


def load_silver(root: Path) -> dict:
    source = root / "data" / "silver" / "credit_applications_silver.parquet"
    schema_file = root / "sql" / "staging" / "create_credit_applications.sql"

    if not source.is_file():
        raise FileNotFoundError(f"Silver dataset not found: {source}")

    if not schema_file.is_file():
        raise FileNotFoundError(f"SQL schema not found: {schema_file}")

    frame = pd.read_parquet(source)
    validate_silver(frame)

    columns = [column.lower() for column in frame.columns]
    column_list = ", ".join(columns)

    copy_statement = (
        f"COPY staging.credit_applications ({column_list}) "
        "FROM STDIN WITH (FORMAT CSV, NULL '\\N')"
    )

    with get_connection(root) as connection, connection.cursor() as cursor:
        cursor.execute(schema_file.read_text(encoding="utf-8"))

        cursor.execute(
            """
            SELECT column_name
            FROM information_schema.columns
            WHERE table_schema = 'staging'
              AND table_name = 'credit_applications'
            ORDER BY ordinal_position
            """
        )

        database_columns = [row[0] for row in cursor.fetchall()]

        if database_columns != columns:
            raise ValueError("Database schema does not match Silver columns")

        cursor.execute("TRUNCATE TABLE staging.credit_applications")

        with cursor.copy(copy_statement) as copy:
            for start in range(0, len(frame), 10000):
                batch = frame.iloc[start : start + 10000]
                buffer = io.StringIO()
                batch.to_csv(
                    buffer,
                    index=False,
                    header=False,
                    na_rep="\\N",
                    lineterminator="\n",
                )
                copy.write(buffer.getvalue())

        cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT sk_id_curr),
                SUM(target),
                SUM(employment_sentinel_flag),
                SUM(financial_data_quality_flag)
            FROM staging.credit_applications
            """
        )

        row_count, unique_ids, positive_targets, sentinels, quality_flags = (
            cursor.fetchone()
        )

        expected_sentinels = int(frame["EMPLOYMENT_SENTINEL_FLAG"].sum())
        expected_flags = int(frame["FINANCIAL_DATA_QUALITY_FLAG"].sum())

        if row_count != len(frame):
            raise ValueError("PostgreSQL row count does not match Silver")

        if unique_ids != len(frame):
            raise ValueError("PostgreSQL application IDs are not unique")

        if positive_targets != int(frame["TARGET"].sum()):
            raise ValueError("PostgreSQL target count does not match Silver")

        if sentinels != expected_sentinels:
            raise ValueError("Employment sentinel reconciliation failed")

        if quality_flags != expected_flags:
            raise ValueError("Data-quality flag reconciliation failed")

    return {
        "database": "finance360",
        "table": "staging.credit_applications",
        "rows_loaded": row_count,
        "unique_application_ids": unique_ids,
        "positive_target_count": positive_targets,
        "employment_sentinel_count": sentinels,
        "financial_quality_flag_count": quality_flags,
        "column_count": len(columns),
        "reconciliation": "PASSED",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result = load_silver(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()