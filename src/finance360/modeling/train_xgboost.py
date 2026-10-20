import json
from pathlib import Path

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

from finance360.modeling.data_split import RANDOM_STATE, create_modeling_split
from finance360.modeling.feature_contract import load_modeling_data
from finance360.modeling.train_baseline import N_SPLITS
from finance360.modeling.xgboost_model import create_xgboost_challenger


def evaluate_xgboost_cv(project_root: Path) -> dict:
    features, target = load_modeling_data(project_root)
    split = create_modeling_split(features, target)

    cross_validation = StratifiedKFold(
        n_splits=N_SPLITS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    fold_results = []

    for fold_number, (train_indices, validation_indices) in enumerate(
        cross_validation.split(split.X_train, split.y_train),
        start=1,
    ):
        X_fold_train = split.X_train.iloc[train_indices]
        X_fold_validation = split.X_train.iloc[validation_indices]
        y_fold_train = split.y_train.iloc[train_indices]
        y_fold_validation = split.y_train.iloc[validation_indices]

        pipeline = create_xgboost_challenger()
        pipeline.fit(X_fold_train, y_fold_train)

        probabilities = pipeline.predict_proba(X_fold_validation)[:, 1]

        fold_result = {
            "fold": fold_number,
            "training_rows": len(X_fold_train),
            "validation_rows": len(X_fold_validation),
            "validation_positives": int(y_fold_validation.sum()),
            "roc_auc": float(
                roc_auc_score(y_fold_validation, probabilities)
            ),
            "average_precision": float(
                average_precision_score(y_fold_validation, probabilities)
            ),
            "brier_score": float(
                brier_score_loss(y_fold_validation, probabilities)
            ),
        }

        fold_results.append(fold_result)

        print(
            f"Fold {fold_number}/{N_SPLITS} completed: "
            f"ROC-AUC={fold_result['roc_auc']:.4f}, "
            f"AP={fold_result['average_precision']:.4f}, "
            f"Brier={fold_result['brier_score']:.4f}",
            flush=True,
        )

    metric_names = ["roc_auc", "average_precision", "brier_score"]

    summary = {
        metric: {
            "mean": float(np.mean([fold[metric] for fold in fold_results])),
            "std": float(np.std([fold[metric] for fold in fold_results])),
        }
        for metric in metric_names
    }

    result = {
        "model": "xgboost_challenger",
        "validation_strategy": "stratified_3_fold_training_only",
        "random_state": RANDOM_STATE,
        "training_rows": len(split.X_train),
        "held_out_test_rows": len(split.X_test),
        "training_positives": int(split.y_train.sum()),
        "test_set_evaluated": False,
        "fold_results": fold_results,
        "summary": summary,
    }

    report_path = (
        project_root
        / "reports"
        / "modeling"
        / "xgboost_challenger_cv.json"
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    return result


def main() -> None:
    project_root = Path(__file__).resolve().parents[3]
    result = evaluate_xgboost_cv(project_root)
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()