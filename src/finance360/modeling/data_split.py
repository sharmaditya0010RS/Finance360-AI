from dataclasses import dataclass

import pandas as pd
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.20


@dataclass
class ModelingSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series


def create_modeling_split(
    features: pd.DataFrame,
    target: pd.Series,
) -> ModelingSplit:
    if len(features) != len(target):
        raise ValueError("Feature and target row counts do not match")

    if features.empty:
        raise ValueError("Cannot split an empty feature dataset")

    if target.isna().any():
        raise ValueError("Target contains missing values")

    if not target.isin([0, 1]).all():
        raise ValueError("Target must contain only binary values")

    if target.nunique() != 2:
        raise ValueError("Both target classes must be present")

    if target.value_counts().min() < 2:
        raise ValueError("Each target class must contain at least two records")

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    if len(X_train) + len(X_test) != len(features):
        raise ValueError("Split row counts do not reconcile")

    if set(X_train.index).intersection(set(X_test.index)):
        raise ValueError("Training and test indices overlap")

    return ModelingSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
    )