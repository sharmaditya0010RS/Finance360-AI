"""Historical segment-level repayment-difficulty analysis."""

from __future__ import annotations

import math

import pandas as pd

DIMENSIONS = {
    "Contract type": ("dim_contract_type", "contract_type_key", "contract_type_name"),
    "Income type": ("dim_income_type", "income_type_key", "income_type_name"),
    "Education": ("dim_education", "education_key", "education_name"),
    "Housing": ("dim_housing", "housing_key", "housing_name"),
    "Family status": ("dim_family_status", "family_status_key", "family_status_name"),
}


def analyze_hotspots(frame: pd.DataFrame, minimum_applications: int = 1000) -> pd.DataFrame:
    """Calculate segment rates using an explicit overall benchmark.

    Input must include dimension, segment, applications, difficulty_cases,
    portfolio_applications and portfolio_difficulty_cases. All portfolio totals
    must refer to the same population, independent of segment filtering.
    """
    if isinstance(minimum_applications, bool) or not isinstance(minimum_applications, int) or minimum_applications < 1:
        raise ValueError("minimum_applications must be a positive integer")
    required = {"dimension", "segment", "applications", "difficulty_cases", "portfolio_applications", "portfolio_difficulty_cases"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing hotspot columns: {sorted(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=["dimension", "segment", "applications", "difficulty_cases", "difficulty_rate", "portfolio_rate", "difference_pp", "relative_rate"])
    result = frame.copy()
    for col in ("applications", "difficulty_cases", "portfolio_applications", "portfolio_difficulty_cases"):
        result[col] = pd.to_numeric(result[col], errors="raise")
        if not result[col].map(lambda v: math.isfinite(float(v)) and float(v).is_integer()).all():
            raise ValueError(f"Invalid integer counts in {col}")
    if (result["applications"] < 0).any() or (result["difficulty_cases"] < 0).any():
        raise ValueError("Counts cannot be negative")
    if (result["difficulty_cases"] > result["applications"]).any():
        raise ValueError("Difficulty cases exceed applications")
    if (result["portfolio_applications"] <= 0).any() or (result["portfolio_difficulty_cases"] < 0).any() or (result["portfolio_difficulty_cases"] > result["portfolio_applications"]).any():
        raise ValueError("Invalid portfolio benchmark counts")
    if result["portfolio_applications"].nunique() != 1 or result["portfolio_difficulty_cases"].nunique() != 1:
        raise ValueError("Inconsistent portfolio benchmarks")
    result = result.loc[result["applications"] >= minimum_applications].copy()
    result["difficulty_rate"] = result["difficulty_cases"] / result["applications"]
    result["portfolio_rate"] = result["portfolio_difficulty_cases"] / result["portfolio_applications"]
    result["difference_pp"] = 100 * (result["difficulty_rate"] - result["portfolio_rate"])
    result["relative_rate"] = result["difficulty_rate"] / result["portfolio_rate"].replace(0, float("nan"))
    return result.sort_values(["difference_pp", "applications"], ascending=[False, False]).reset_index(drop=True)


def load_hotspot_segments(project_root) -> pd.DataFrame:
    """Fetch segment counts from the warehouse with a single overall benchmark."""
    from finance360.database.connection import get_connection

    query_template = """
        SELECT %s AS dimension, COALESCE(d.{label}, 'Unknown') AS segment,
               COUNT(*)::bigint AS applications,
               COUNT(*) FILTER (WHERE f.target = 1)::bigint AS difficulty_cases
        FROM warehouse.fact_credit_application AS f
        LEFT JOIN warehouse.{table} AS d ON f.{key} = d.{key}
        GROUP BY d.{label}
    """
    rows = []
    with get_connection(project_root) as connection, connection.cursor() as cursor:
        cursor.execute("SELECT COUNT(*)::bigint, COUNT(*) FILTER (WHERE target = 1)::bigint FROM warehouse.fact_credit_application")
        total, positives = cursor.fetchone()
        for dimension, (table, key, label) in DIMENSIONS.items():
            # Identifiers are constants declared in this module, not user input.
            cursor.execute(query_template.format(table=table, key=key, label=label), (dimension,))
            rows.extend((*record, total, positives) for record in cursor.fetchall())
    return pd.DataFrame(rows, columns=["dimension", "segment", "applications", "difficulty_cases", "portfolio_applications", "portfolio_difficulty_cases"])
