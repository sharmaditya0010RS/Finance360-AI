from decimal import Decimal

import pytest

from finance360.database.build_warehouse import (
    EXPECTED_APPLICATIONS,
    EXPECTED_DIMENSIONS,
    EXPECTED_POSITIVE_TARGETS,
    validate_metrics,
)
from finance360.database.connection import get_connection


@pytest.fixture(scope="module")
def warehouse_connection():
    with get_connection() as connection:
        yield connection


def test_warehouse_fact_grain(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*),
                COUNT(DISTINCT sk_id_curr),
                COUNT(*) FILTER (WHERE sk_id_curr IS NULL),
                SUM(target)
            FROM warehouse.fact_credit_application
            """
        )
        total, unique_ids, missing_ids, positive_targets = cursor.fetchone()

    assert total == EXPECTED_APPLICATIONS
    assert unique_ids == EXPECTED_APPLICATIONS
    assert missing_ids == 0
    assert positive_targets == EXPECTED_POSITIVE_TARGETS


def test_warehouse_dimension_counts(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        for table_name, expected_count in EXPECTED_DIMENSIONS.items():
            cursor.execute(f"SELECT COUNT(*) FROM warehouse.{table_name}")
            assert cursor.fetchone()[0] == expected_count


def test_warehouse_financial_reconciliation(warehouse_connection):
    query = """
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

    with warehouse_connection.cursor() as cursor:
        cursor.execute(query.format("staging.credit_applications"))
        staging = cursor.fetchone()

        cursor.execute(query.format("warehouse.fact_credit_application"))
        warehouse = cursor.fetchone()

    validate_metrics(staging, warehouse)
    assert warehouse[2] == Decimal("184207084195.50")
    assert warehouse[5] == 55374
    assert warehouse[6] == 0


def test_warehouse_preserves_unknown_family_status(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT COUNT(*)
            FROM warehouse.fact_credit_application AS f
            JOIN warehouse.dim_family_status AS d
                ON f.family_status_key = d.family_status_key
            WHERE d.family_status_name = 'Unknown'
            """
        )
        unknown_count = cursor.fetchone()[0]

    assert unknown_count == 2


def test_warehouse_preserves_application_keys_and_outcomes(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
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
            """
        )
        mismatches = cursor.fetchone()[0]

    assert mismatches == 0


def test_warehouse_portfolio_view(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                application_count,
                repayment_difficulty_count,
                repayment_difficulty_rate_pct,
                total_requested_credit
            FROM analytics.warehouse_portfolio_summary
            """
        )
        applications, positives, rate, credit = cursor.fetchone()

    assert applications == EXPECTED_APPLICATIONS
    assert positives == EXPECTED_POSITIVE_TARGETS
    assert rate == Decimal("8.0729")
    assert credit == Decimal("184207084195.50")


def test_warehouse_contract_view_reconciles(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*),
                SUM(application_count),
                SUM(repayment_difficulty_count),
                SUM(total_requested_credit)
            FROM analytics.warehouse_contract_performance
            """
        )
        groups, applications, positives, credit = cursor.fetchone()

    assert groups == 2
    assert applications == EXPECTED_APPLICATIONS
    assert positives == EXPECTED_POSITIVE_TARGETS
    assert credit == Decimal("184207084195.50")


def test_warehouse_income_view_reconciles(warehouse_connection):
    with warehouse_connection.cursor() as cursor:
        cursor.execute(
            """
            SELECT
                COUNT(*),
                SUM(application_count),
                SUM(repayment_difficulty_count),
                SUM(total_requested_credit)
            FROM analytics.warehouse_income_type_performance
            """
        )
        groups, applications, positives, credit = cursor.fetchone()

    assert groups == 8
    assert applications == EXPECTED_APPLICATIONS
    assert positives == EXPECTED_POSITIVE_TARGETS
    assert credit == Decimal("184207084195.50")


def test_warehouse_rejects_mismatched_metrics():
    staging = (307511, 24825, Decimal("100.00"))
    warehouse = (307511, 24825, Decimal("99.00"))

    with pytest.raises(ValueError, match="Warehouse reconciliation failed"):
        validate_metrics(staging, warehouse)