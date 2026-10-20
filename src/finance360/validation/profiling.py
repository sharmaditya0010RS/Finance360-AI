import json
from pathlib import Path

import pandas as pd


def profile_data(frame: pd.DataFrame) -> dict:
    columns = {}

    for column in frame.columns:
        series = frame[column]
        numeric = pd.api.types.is_numeric_dtype(series)

        details = {
            "dtype": str(series.dtype),
            "missing_count": int(series.isna().sum()),
            "missing_rate": float(series.isna().mean()),
            "unique_count": int(series.nunique(dropna=True)),
        }

        if numeric:
            details["minimum"] = (
                float(series.min()) if series.notna().any() else None
            )
            details["maximum"] = (
                float(series.max()) if series.notna().any() else None
            )
            details["median"] = (
                float(series.median()) if series.notna().any() else None
            )

        columns[column] = details

    return {
        "row_count": len(frame),
        "column_count": len(frame.columns),
        "duplicate_application_ids": int(frame["SK_ID_CURR"].duplicated().sum()),
        "employment_sentinel_count": int(
            frame["DAYS_EMPLOYED"].eq(365243).sum()
        ),
        "columns": columns,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[3]
    source = root / "data" / "bronze" / "home_credit_applications.parquet"
    destination = root / "reports" / "data_quality"

    if not source.is_file():
        raise FileNotFoundError(f"Bronze dataset not found: {source}")

    frame = pd.read_parquet(source)
    report = profile_data(frame)

    destination.mkdir(parents=True, exist_ok=True)
    report_path = destination / "bronze_profile.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Rows: {report['row_count']}")
    print(f"Columns: {report['column_count']}")
    print(f"Employment sentinels: {report['employment_sentinel_count']}")
    print(f"Report: {report_path}")


if __name__ == "__main__":
    main()