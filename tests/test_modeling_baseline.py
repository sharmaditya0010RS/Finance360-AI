import numpy as np
import pandas as pd

from finance360.modeling.baseline import create_logistic_baseline
from finance360.modeling.feature_contract import (
    CATEGORICAL_FEATURES,
    NUMERICAL_FEATURES,
)


def test_baseline_pipeline_predicts_probabilities():
    rows = 60

    data = {
        column: np.linspace(1.0, 60.0, rows)
        for column in NUMERICAL_FEATURES
    }

    for column in CATEGORICAL_FEATURES:
        data[column] = ["Category A", "Category B"] * 30

    features = pd.DataFrame(data)
    features.loc[0, NUMERICAL_FEATURES[0]] = np.nan
    features.loc[1, CATEGORICAL_FEATURES[0]] = None

    target = pd.Series([0, 1] * 30)

    pipeline = create_logistic_baseline()
    pipeline.fit(features, target)

    probabilities = pipeline.predict_proba(features)

    assert probabilities.shape == (rows, 2)
    assert np.isfinite(probabilities).all()
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_baseline_handles_unseen_categories():
    rows = 40

    data = {
        column: np.linspace(1.0, 40.0, rows)
        for column in NUMERICAL_FEATURES
    }

    for column in CATEGORICAL_FEATURES:
        data[column] = ["Category A", "Category B"] * 20

    features = pd.DataFrame(data)
    target = pd.Series([0, 1] * 20)

    pipeline = create_logistic_baseline()
    pipeline.fit(features, target)

    unseen = features.iloc[:1].copy()
    unseen[CATEGORICAL_FEATURES[0]] = "New Category"

    probabilities = pipeline.predict_proba(unseen)

    assert probabilities.shape == (1, 2)
    assert np.isfinite(probabilities).all()