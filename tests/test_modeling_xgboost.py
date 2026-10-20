import numpy as np
import pandas as pd

from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)
from finance360.modeling.xgboost_model import create_xgboost_challenger


def build_synthetic_features(rows):
    data = {
        column: np.linspace(1.0, float(rows), rows)
        for column in NUMERICAL_FEATURES
    }

    for column in CATEGORICAL_FEATURES:
        data[column] = ["Category A", "Category B"] * (rows // 2)

    return pd.DataFrame(data)


def test_xgboost_pipeline_predicts_probabilities():
    features = build_synthetic_features(60)
    target = pd.Series([0, 1] * 30)

    features.loc[0, NUMERICAL_FEATURES[0]] = np.nan
    features.loc[1, CATEGORICAL_FEATURES[0]] = None

    pipeline = create_xgboost_challenger()
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)

    assert probabilities.shape == (60, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_xgboost_handles_unseen_categories():
    features = build_synthetic_features(40)
    target = pd.Series([0, 1] * 20)

    pipeline = create_xgboost_challenger()
    pipeline.fit(features, target)

    unseen = features.iloc[:1].copy()
    unseen[CATEGORICAL_FEATURES[0]] = "New Category"

    probabilities = pipeline.predict_proba(unseen)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()