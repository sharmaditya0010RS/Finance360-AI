import json
from pathlib import Path

import pandas as pd

from finance360.database.connection import get_connection

ANALYTICS_VIEWS = {
    "portfolio_summary": "analytics.warehouse_portfolio_summary",
    "contract_performance": "analytics.warehouse_contract_performance",
    "income_type_performance": "analytics.warehouse_income_type_performance",
}

REPORT_PATHS = {
    "held_out_evaluation": "reports/modeling/held_out_test_evaluation.json",
    "risk_segmentation": "reports/monitoring/risk_segmentation.json",
    "portfolio_exposure": "reports/monitoring/portfolio_exposure.json",
    "score_monitoring": "reports/monitoring/score_distribution_monitoring.json",
}


def load_analytics_view(
    view_name: str,
    project_root: Path,
) -> pd.DataFrame:
    if view_name not in ANALYTICS_VIEWS:
        raise ValueError(f"Unknown analytics view: {view_name}")

    query = f"SELECT * FROM {ANALYTICS_VIEWS[view_name]}"

    with get_connection(project_root) as connection, connection.cursor() as cursor:
        cursor.execute(query)
        rows = cursor.fetchall()
        columns = [column.name for column in cursor.description]

    return pd.DataFrame(rows, columns=columns)


def load_dashboard_report(
    report_name: str,
    project_root: Path,
) -> dict:
    if report_name not in REPORT_PATHS:
        raise ValueError(f"Unknown dashboard report: {report_name}")

    report_path = project_root / REPORT_PATHS[report_name]

    with report_path.open(encoding="utf-8") as file:
        report = json.load(file)

    if not isinstance(report, dict):
        raise TypeError(f"Invalid dashboard report: {report_name}")

    return report


def load_dashboard_data(project_root: Path) -> dict:
    return {
        "portfolio_summary": load_analytics_view(
            "portfolio_summary",
            project_root,
        ),
        "contract_performance": load_analytics_view(
            "contract_performance",
            project_root,
        ),
        "income_type_performance": load_analytics_view(
            "income_type_performance",
            project_root,
        ),
        "held_out_evaluation": load_dashboard_report(
            "held_out_evaluation",
            project_root,
        ),
        "risk_segmentation": load_dashboard_report(
            "risk_segmentation",
            project_root,
        ),
        "portfolio_exposure": load_dashboard_report(
            "portfolio_exposure",
            project_root,
        ),
        "score_monitoring": load_dashboard_report(
            "score_monitoring",
            project_root,
        ),
    }