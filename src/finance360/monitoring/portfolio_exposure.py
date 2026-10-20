import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

from finance360.modeling.data_split import RANDOM_STATE, create_modeling_split
from finance360.modeling.feature_contract import load_modeling_data
from finance360.modeling.train_baseline import N_SPLITS
from finance360.modeling.xgboost_model import create_xgboost_challenger
from finance360.monitoring.risk_segmentation import RISK_BANDS


def calculate_portfolio_exposure(
    probabilities: np.ndarray,
    targets: np.ndarray,
    requested_credit: np.ndarray,
    boundaries: np.ndarray,
) -> dict:
    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray(targets)
    credit = np.asarray(requested_credit, dtype=float)
    thresholds = np.asarray(boundaries, dtype=float)

    if any(array.ndim != 1 for array in (scores, labels, credit, thresholds)):
        raise ValueError("Inputs must be one-dimensional")

    if len(scores) == 0 or len(scores) != len(labels) or len(scores) != len(credit):
        raise ValueError("Application arrays must be nonempty and aligned")

    if len(thresholds) != len(RISK_BANDS) - 1:
        raise ValueError("Expected three risk-band boundaries")

    if not np.isfinite(scores).all() or not np.isfinite(credit).all():
        raise ValueError("Scores and requested credit must be finite")

    if not np.isfinite(thresholds).all():
        raise ValueError("Risk-band boundaries must be finite")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1")

    if np.any(credit < 0):
        raise ValueError("Requested credit cannot be negative")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Targets must be binary")

    if np.any((thresholds <= 0) | (thresholds >= 1)):
        raise ValueError("Risk-band boundaries must be between 0 and 1")

    if not np.all(np.diff(thresholds) > 0):
        raise ValueError("Risk-band boundaries must be strictly increasing")

    assignments = np.searchsorted(thresholds, scores, side="right")
    total_credit = float(credit.sum())

    if total_credit <= 0:
        raise ValueError("Total requested credit must be positive")

    segments = []

    for index, band_name in enumerate(RISK_BANDS):
        mask = assignments == index
        count = int(mask.sum())

        if count == 0:
            raise ValueError("Every risk band must contain applications")

        band_credit = credit[mask]
        band_labels = labels[mask]

        segments.append(
            {
                "risk_band": band_name,
                "application_count": count,
                "application_share": float(count / len(scores)),
                "positive_count": int(band_labels.sum()),
                "observed_difficulty_rate": float(band_labels.mean()),
                "total_requested_credit": float(band_credit.sum()),
                "requested_credit_share": float(
                    band_credit.sum() / total_credit
                ),
                "average_requested_credit": float(band_credit.mean()),
                "mean_predicted_probability": float(scores[mask].mean()),
            }
        )

    if sum(segment["application_count"] for segment in segments) != len(scores):
        raise ValueError("Application counts do not reconcile")

    if not np.isclose(
        sum(segment["total_requested_credit"] for segment in segments),
        total_credit,
    ):
        raise ValueError("Requested credit does not reconcile")

    elevated_segments = segments[2:]

    return {
        "application_count": len(scores),
        "positive_count": int(labels.sum()),
        "total_requested_credit": total_credit,
        "risk_band_boundaries": thresholds.tolist(),
        "segments": segments,
        "high_and_very_high": {
            "application_count": sum(
                item["application_count"] for item in elevated_segments
            ),
            "application_share": float(
                sum(item["application_count"] for item in elevated_segments)
                / len(scores)
            ),
            "total_requested_credit": float(
                sum(item["total_requested_credit"] for item in elevated_segments)
            ),
            "requested_credit_share": float(
                sum(item["total_requested_credit"] for item in elevated_segments)
                / total_credit
            ),
        },
    }


def generate_portfolio_exposure(project_root: Path) -> dict:
    risk_path = (
        project_root
        / "reports"
        / "monitoring"
        / "risk_segmentation.json"
    )

    risk_report = json.loads(risk_path.read_text(encoding="utf-8"))

    if risk_report["model"] != "xgboost_challenger":
        raise ValueError("Risk report model does not match XGBoost")

    if risk_report["test_set_evaluated"]:
        raise ValueError("Risk boundaries must come from training only")

    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    if risk_report["training_rows"] != len(split.X_train):
        raise ValueError("Risk report training rows do not match")

    silver = pd.read_parquet(
        project_root / "data" / "silver" / "credit_applications_silver.parquet",
        columns=["AMT_CREDIT"],
    )

    requested_credit = silver.loc[split.X_train.index, "AMT_CREDIT"]

    if not requested_credit.index.equals(split.X_train.index):
        raise ValueError("Requested credit index does not align")

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.full(len(split.X_train), np.nan, dtype=float)

    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validation.split(split.X_train, split.y_train),
        start=1,
    ):
        pipeline = create_xgboost_challenger()

        pipeline.fit(
            split.X_train.iloc[train_indices],
            split.y_train.iloc[train_indices],
        )

        oof_probabilities[validation_indices] = pipeline.predict_proba(
            split.X_train.iloc[validation_indices]
        )[:, 1]

        print(
            f"Portfolio exposure fold {fold_number}/{N_SPLITS} completed",
            flush=True,
        )

    if not np.isfinite(oof_probabilities).all():
        raise ValueError("Incomplete out-of-fold predictions")

    result = calculate_portfolio_exposure(
        oof_probabilities,
        split.y_train.to_numpy(),
        requested_credit.to_numpy(),
        np.asarray(risk_report["boundaries"], dtype=float),
    )

    for actual, expected in zip(
        result["segments"],
        risk_report["segments"],
        strict=True,
    ):
        if actual["risk_band"] != expected["risk_band"]:
            raise ValueError("Risk-band names do not reconcile")

        if actual["application_count"] != expected["application_count"]:
            raise ValueError("Risk-band application counts do not reconcile")

        if actual["positive_count"] != expected["positive_count"]:
            raise ValueError("Risk-band positive counts do not reconcile")

    result.update(
        {
            "model": "xgboost_challenger",
            "population": "training_only_out_of_fold",
            "held_out_test_rows": len(split.X_test),
            "test_set_evaluated": False,
            "exposure_definition": "original_requested_credit_amount",
            "limitations": [
                "Requested credit is not actual outstanding exposure or EAD.",
                "Source currency units are not established.",
                "Training OOF segments are not production monitoring results.",
                "No held-out test outcomes were used in this analysis.",
            ],
        }
    )

    report_path = (
        project_root
        / "reports"
        / "monitoring"
        / "portfolio_exposure.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = generate_portfolio_exposure(project_root)

    print("Applications:", result["application_count"])
    print("Total requested credit:", result["total_requested_credit"])

    for segment in result["segments"]:
        print(
            segment["risk_band"],
            "applications:",
            segment["application_count"],
            "requested credit:",
            f"{segment['total_requested_credit']:.2f}",
            "credit share:",
            f"{segment['requested_credit_share']:.2%}",
        )

    print(
        "High + Very High credit share:",
        f"{result['high_and_very_high']['requested_credit_share']:.2%}",
    )
    print("Test set evaluated:", result["test_set_evaluated"])


if __name__ == "__main__":
    main()