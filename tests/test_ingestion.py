from pathlib import Path

import pandas as pd
import pytest

from finance360.ingestion import home_credit


def test_selected_schema_has_exactly_twenty_columns():
    assert len(home_credit.SOURCE_COLUMNS) == 20
    assert len(set(home_credit.SOURCE_COLUMNS)) == 20


def test_source_dataset_is_required(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        home_credit.ingest_application_data(tmp_path)


def test_sha256_is_deterministic(tmp_path: Path):
    sample = tmp_path / "sample.txt"
    sample.write_text("finance360", encoding="utf-8")

    first_hash = home_credit.calculate_sha256(sample)
    second_hash = home_credit.calculate_sha256(sample)

    assert first_hash == second_hash
    assert len(first_hash) == 64


def test_invalid_source_column_count_is_rejected(tmp_path: Path):
    source_directory = tmp_path / "data" / "external"
    source_directory.mkdir(parents=True)

    pd.DataFrame({"SK_ID_CURR": [100001], "TARGET": [0]}).to_csv(
        source_directory / "application_train.csv",
        index=False,
    )

    with pytest.raises(ValueError, match="source columns"):
        home_credit.ingest_application_data(tmp_path)