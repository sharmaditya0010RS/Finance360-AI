import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib

from finance360.modeling.data_split import create_modeling_split
from finance360.modeling.feature_contract import (
    MODEL_FEATURES,
    load_modeling_data,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger

MODEL_VERSION = "1.0.0"


def calculate_sha256(file_path: Path) -> str:
    digest = hashlib.sha256()

    with file_path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)

    return digest.hexdigest()


def save_champion_model(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    pipeline = create_xgboost_challenger()
    pipeline.fit(split.X_train, split.y_train)

    artifact_dir = project_root / "artifacts" / "models"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_path = artifact_dir / "finance360_xgboost_v1.joblib"
    metadata_path = artifact_dir / "finance360_xgboost_v1_metadata.json"

    joblib.dump(pipeline, model_path)

    reloaded_pipeline = joblib.load(model_path)

    sample = split.X_train.iloc[:100]

    original_probabilities = pipeline.predict_proba(sample)[:, 1]
    reloaded_probabilities = reloaded_pipeline.predict_proba(sample)[:, 1]

    import numpy as np

    maximum_reload_error = float(
        np.max(
            np.abs(
                original_probabilities - reloaded_probabilities
            )
        )
    )

    if maximum_reload_error > 1e-10:
        raise ValueError("Reloaded model predictions do not match")

    metadata = {
        "model_name": "xgboost_challenger",
        "model_version": MODEL_VERSION,
        "selection_status": "provisional",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "training_rows": len(split.X_train),
        "held_out_test_rows": len(split.X_test),
        "test_set_evaluated": False,
        "feature_count": len(MODEL_FEATURES),
        "feature_names": list(MODEL_FEATURES),
        "artifact_filename": model_path.name,
        "artifact_sha256": calculate_sha256(model_path),
        "maximum_reload_probability_error": maximum_reload_error,
        "serialization_format": "joblib",
    }

    metadata_path.write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )

    return metadata


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = save_champion_model(project_root)

    print("Model:", result["model_name"])
    print("Version:", result["model_version"])
    print("Training applications:", result["training_rows"])
    print("Features:", result["feature_count"])
    print(
        "Maximum reload error:",
        result["maximum_reload_probability_error"],
    )
    print("Test set evaluated:", result["test_set_evaluated"])


if __name__ == "__main__":
    main()