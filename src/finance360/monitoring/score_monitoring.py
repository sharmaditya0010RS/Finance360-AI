import json
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from finance360.modeling.data_split import RANDOM_STATE, create_modeling_split
from finance360.modeling.feature_contract import load_modeling_data
from finance360.modeling.train_baseline import N_SPLITS
from finance360.modeling.xgboost_model import create_xgboost_challenger
from finance360.monitoring.drift import calculate_psi


def calculate_score_monitoring(
    probabilities: np.ndarray,
    group_assignments: np.ndarray,
) -> dict:
    scores = np.asarray(probabilities, dtype=float)
    groups = np.asarray(group_assignments)

    if scores.ndim != 1 or groups.ndim != 1:
        raise ValueError("Scores and groups must be one-dimensional")

    if len(scores) == 0 or len(scores) != len(groups):
        raise ValueError("Scores and groups must be nonempty and aligned")

    if not np.isfinite(scores).all():
        raise ValueError("Scores must be finite")

    if np.any((scores < 0) | (scores > 1)):
        raise ValueError("Scores must be probabilities between 0 and 1")

    if not np.isin(groups, [0, 1]).all():
        raise ValueError("Group assignments must contain only 0 and 1")

    reference = scores[groups == 0]
    comparison = scores[groups == 1]

    if len(reference) == 0 or len(comparison) == 0:
        raise ValueError("Both comparison groups must be nonempty")

    psi_result = calculate_psi(reference, comparison)

    return {
        "comparison_type": "non_temporal_training_oof_demonstration",
        "reference_group": "deterministic_group_0",
        "comparison_group": "deterministic_group_1",
        "reference_count": len(reference),
        "comparison_count": len(comparison),
        "reference_mean_score": float(reference.mean()),
        "comparison_mean_score": float(comparison.mean()),
        "psi_result": psi_result,
        "production_drift_claim": False,
    }


def generate_score_monitoring(project_root: Path) -> dict:
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

        oof_probabilities[validation_indices] = pipeline.predict_proba(
            split.X_train.iloc[validation_indices]
        )[:, 1]

        print(
            f"Score monitoring fold {fold_number}/{N_SPLITS} completed",
            flush=True,
        )

    if not np.isfinite(oof_probabilities).all():
        raise ValueError("Incomplete out-of-fold predictions")

    rng = np.random.default_rng(RANDOM_STATE)
    permutation = rng.permutation(len(oof_probabilities))

    group_assignments = np.zeros(
        len(oof_probabilities),
        dtype=int,
    )

    group_assignments[permutation[len(permutation) // 2 :]] = 1

    result = calculate_score_monitoring(
        oof_probabilities,
        group_assignments,
    )

    result.update(
        {
            "model": "xgboost_challenger",
            "training_rows": len(split.X_train),
            "held_out_test_rows": len(split.X_test),
            "random_state": RANDOM_STATE,
            "grouping_method": "seeded_random_half_split",
            "test_set_evaluated": False,
            "limitations": [
                "This comparison uses random subsets of training OOF scores.",
                "The dataset does not provide application event dates.",
                "The PSI result is not evidence of production or temporal drift.",
                "A real monitoring process requires future scored populations.",
            ],
        }
    )

    report_path = (
        project_root
        / "reports"
        / "monitoring"
        / "score_distribution_monitoring.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)

    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = generate_score_monitoring(project_root)

    print("Model:", result["model"])
    print("Reference count:", result["reference_count"])
    print("Comparison count:", result["comparison_count"])
    print("Reference mean score:", result["reference_mean_score"])
    print("Comparison mean score:", result["comparison_mean_score"])
    print("PSI:", result["psi_result"]["psi"])
    print("Production drift claim:", result["production_drift_claim"])
    print("Test set evaluated:", result["test_set_evaluated"])


if __name__ == "__main__":
    main()