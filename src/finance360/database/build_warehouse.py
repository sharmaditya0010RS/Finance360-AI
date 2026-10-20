import json
from pathlib import Path

from finance360.database.connection import get_connection

EXPECTED_APPLICATIONS = 307511
EXPECTED_POSITIVE_TARGETS = 24825

EXPECTED_DIMENSIONS = {
    "dim_contract_type": 2,
    "dim_income_type": 8,
    "dim_education": 5,
    "dim_housing": 6,
    "dim_family_status": 6,
}


def validate_metrics(staging: tuple, warehouse: tuple) -> None:
    if staging != warehouse:
        raise ValueError(
            f"Warehouse reconciliation failed: staging={staging}, "
            f"warehouse={warehouse}"
        )

    if staging[0] != EXPECTED_APPLICATIONS:
        raise ValueError("Unexpected staging application count")

    if staging[1] != EXPECTED_POSITIVE_TARGETS:
        raise ValueError("Unexpected staging repayment-difficulty count")


def build_warehouse(root: Path) -> dict:
    script_path = root / "sql" / "facts" / "load_warehouse.sql"

    if not script_path.is_file():
        raise FileNotFoundError(f"Warehouse ETL SQL not found: {script_path}")

    metric_sql = """
        SELECT
            COUNT(*),
            SUM(target),
            SUM(amt_credit)::NUMERIC,
            SUM(amt_income_total)::NUMERIC,
            SUM(amt_annuity)::NUMERIC,
            SUM(employment_sentinel_flag),
            SUM(financial_data_quality_flag)
        FROM {}
    """

    with get_connection(root) as connection, connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM staging.credit_applications
            WHERE name_contract_type IS NULL
               OR name_income_type IS NULL
               OR name_education_type IS NULL
               OR name_housing_type IS NULL
               OR name_family_status IS NULL
            """
        )

        missing_categories = cursor.fetchone()[0]

        if missing_categories:
            raise ValueError(
                f"{missing_categories} staging rows have missing dimension categories"
            )

        cursor.execute(metric_sql.format("staging.credit_applications"))
        staging_metrics = cursor.fetchone()

        if staging_metrics[0] != EXPECTED_APPLICATIONS:
            raise ValueError("Unexpected staging application count")

        if staging_metrics[1] != EXPECTED_POSITIVE_TARGETS:
            raise ValueError("Unexpected staging repayment-difficulty count")

        cursor.execute(script_path.read_text(encoding="utf-8"))

        cursor.execute(metric_sql.format("warehouse.fact_credit_application"))
        warehouse_metrics = cursor.fetchone()

        validate_metrics(staging_metrics, warehouse_metrics)

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM warehouse.fact_credit_application
            """
        )

        fact_count = cursor.fetchone()[0]

        if fact_count != EXPECTED_APPLICATIONS:
            raise ValueError("Warehouse fact count is incorrect")

        dimension_counts = {}

        for table_name, expected_count in EXPECTED_DIMENSIONS.items():
            cursor.execute(f"SELECT COUNT(*) FROM warehouse.{table_name}")
            actual_count = cursor.fetchone()[0]
            dimension_counts[table_name] = actual_count

            if actual_count != expected_count:
                raise ValueError(
                    f"{table_name}: expected {expected_count}, got {actual_count}"
                )

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM warehouse.fact_credit_application AS fact
            JOIN warehouse.dim_family_status AS family
                ON fact.family_status_key = family.family_status_key
            WHERE family.family_status_name = 'Unknown'
            """
        )

        unknown_family_count = cursor.fetchone()[0]

        if unknown_family_count != 2:
            raise ValueError("Unknown family-status records were not preserved")

        cursor.execute(
            """
            SELECT COUNT(*)
            FROM staging.credit_applications AS s
            FULL JOIN warehouse.fact_credit_application AS f
                ON s.sk_id_curr = f.sk_id_curr
            WHERE s.sk_id_curr IS NULL
               OR f.sk_id_curr IS NULL
               OR s.target IS DISTINCT FROM f.target
               OR s.amt_credit IS DISTINCT FROM f.amt_credit
               OR s.amt_income_total IS DISTINCT FROM f.amt_income_total
               OR s.amt_annuity IS DISTINCT FROM f.amt_annuity
            """
        )

        mismatched_applications = cursor.fetchone()[0]

        if mismatched_applications:
            raise ValueError(
                f"{mismatched_applications} applications failed row reconciliation"
            )

    return {
        "warehouse": "finance360.warehouse",
        "fact_table": "warehouse.fact_credit_application",
        "fact_rows": fact_count,
        "dimension_counts": dimension_counts,
        "repayment_difficulty_count": warehouse_metrics[1],
        "total_requested_credit": str(warehouse_metrics[2]),
        "employment_sentinel_count": warehouse_metrics[5],
        "financial_quality_flag_count": warehouse_metrics[6],
        "unknown_family_status_applications": unknown_family_count,
        "row_reconciliation_mismatches": mismatched_applications,
        "reconciliation": "PASSED",
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    result = build_warehouse(root)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()