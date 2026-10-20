import pandas as pd
import pytest

from finance360.database.load_silver import (
    EXPECTED_COLUMNS,
    EXPECTED_POSITIVE_TARGETS,
    EXPECTED_ROWS,
    validate_silver,
)
from finance360.ingestion.home_credit import SOURCE_COLUMNS


def make_valid_frame() -> pd.DataFrame:
    columns = SOURCE_COLUMNS + [
        f"DERIVED_{index}"
        for index in range(EXPECTED_COLUMNS - len(SOURCE_COLUMNS))
    ]

    frame = pd.DataFrame(index=range(EXPECTED_ROWS), columns=columns)
    frame["SK_ID_CURR"] = range(100000, 100000 + EXPECTED_ROWS)
    frame["TARGET"] = 0
    frame.loc[: EXPECTED_POSITIVE_TARGETS - 1, "TARGET"] = 1

    return frame


def test_loader_accepts_valid_schema():
    validate_silver(make_valid_frame())


def test_loader_rejects_duplicate_identifiers():
    frame = make_valid_frame()
    frame.loc[1, "SK_ID_CURR"] = frame.loc[0, "SK_ID_CURR"]

    with pytest.raises(ValueError, match="duplicated"):
        validate_silver(frame)


def test_loader_rejects_wrong_target_total():
    frame = make_valid_frame()
    frame.loc[0, "TARGET"] = 0

    with pytest.raises(ValueError, match="Target total"):
        validate_silver(frame)


def test_loader_rejects_wrong_column_count():
    frame = make_valid_frame().drop(columns="DERIVED_0")

    with pytest.raises(ValueError, match="Silver shape"):
        validate_silver(frame)