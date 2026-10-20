import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from finance360.modeling.data_split import RANDOM_STATE, create_modeling_split
from finance360.modeling.feature_contract import load_modeling_data
from finance360.modeling.train_baseline import N_SPLITS
from finance360.modeling.xgboost_model import create_xgboost_challenger

RISK_BANDS = ("Low", "Moderate", "High", "Very High")
RISK_QUANTILES = (0.50, 0.75, 0.90)


def calculate_risk_bands(
    probabilities: np.ndarray,
    targets: np.ndarray,
) -> dict:
    scores = np.asarray(probabilities, dtype=float)
    labels = np.asarray(targets)

    if scores.ndim != 1 or labels.ndim != 1:
        raise ValueError("Scores and targets must be one-dimensional")

    if len(scores) == 0 or len(scores) != len(labels):
        raise ValueError("Scores and targets must be nonempty and aligned")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Targets must be binary")

    boundaries = np.quantile(scores, RISK_QUANTILES)

    if not np.all(np.diff(boundaries) > 0):
        raise ValueError("Risk-band boundaries must be strictly increasing")

    assignments = np.searchsorted(
        boundaries,
        scores,
        side="right",
    )

    segments = []

    for index, band_name in enumerate(RISK_BANDS):
        mask = assignments == index
        count = int(mask.sum())

        if count == 0:
            raise ValueError("Every risk band must contain applications")

        band_scores = scores[mask]
        band_targets = labels[mask]

        segments.append(
            {
                "risk_band": band_name,
                "application_count": count,
                "portfolio_share": float(count / len(scores)),
                "positive_count": int(band_targets.sum()),
                "observed_difficulty_rate": float(
                    band_targets.mean()
                ),
                "mean_predicted_probability": float(
                    band_scores.mean()
                ),
                "minimum_predicted_probability": float(
                    band_scores.min()
                ),
                "maximum_predicted_probability": float(
                    band_scores.max()
                ),
            }
        )

    if sum(item["application_count"] for item in segments) != len(scores):
        raise ValueError("Risk-band counts do not reconcile")

    return {
        "method": "training_oof_score_quantiles",
        "quantiles": list(RISK_QUANTILES),
        "boundaries": [float(value) for value in boundaries],
        "training_rows": len(scores),
        "training_positives": int(labels.sum()),
        "segments": segments,
    }


def generate_risk_segmentation(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    oof_probabilities = np.full(
        len(split.X_train),
        np.nan,
        dtype=float,
    )

    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validation.split(split.X_train, split.y_train),
        start=1,
    ):
        pipeline = create_xgboost_challenger()

        pipeline.fit(
            split.X_train.iloc[train_indices],
            split.y_train.iloc[train_indices],
        )

        probabilities = pipeline.predict_proba(
            split.X_train.iloc[validation_indices]
        )[:, 1]

        oof_probabilities[validation_indices] = probabilities

        print(
            f"Risk segmentation fold {fold_number}/{N_SPLITS} completed",
            flush=True,
        )

    if not np.isfinite(oof_probabilities).all():
        raise ValueError("Incomplete out-of-fold predictions")

    result = calculate_risk_bands(
        oof_probabilities,
        split.y_train.to_numpy(),
    )

    result.update(
        {
            "model": "xgboost_challenger",
            "validation_strategy": "stratified_3_fold_training_only",
            "random_state": RANDOM_STATE,
            "held_out_test_rows": len(split.X_test),
            "test_set_evaluated": False,
            "interpretation": (
                "Relative repayment-difficulty risk segments based "
                "on training-only out-of-fold predicted probabilities."
            ),
            "limitations": [
                "Risk bands are descriptive and not regulatory credit grades.",
                "Observed difficulty rates are training OOF estimates.",
                "No held-out test outcomes were used to define boundaries.",
                "Score quantiles do not imply equal business exposure.",
            ],
        }
    )

    report_path = (
        project_root
        / "reports"
        / "monitoring"
        / "risk_segmentation.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = generate_risk_segmentation(project_root)

    print("Model:", result["model"])
    print("Training applications:", result["training_rows"])
    print("Boundaries:", result["boundaries"])

    for segment in result["segments"]:
        print(
            segment["risk_band"],
            "count:",
            segment["application_count"],
            "difficulty rate:",
            f"{segment['observed_difficulty_rate']:.4%}",
        )

    print("Test set evaluated:", result["test_set_evaluated"])


if __name__ == "__main__":
    main()