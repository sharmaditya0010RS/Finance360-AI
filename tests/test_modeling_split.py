import pandas as pd
import pytest

from finance360.modeling.data_split import create_modeling_split


@pytest.fixture
def sample_modeling_data():
    features = pd.DataFrame(
        {
            "income": [float(i * 1000) for i in range(100)],
            "credit": [float(i * 500) for i in range(100)],
            "contract_type": ["Cash loans", "Revolving loans"] * 50,
        }
    )

    target = pd.Series([0] * 90 + [1] * 10)

    return features, target


def test_split_preserves_all_records(sample_modeling_data):
    features, target = sample_modeling_data
    split = create_modeling_split(features, target)

    assert len(split.X_train) == 80
    assert len(split.X_test) == 20
    assert len(split.y_train) == 80
    assert len(split.y_test) == 20


def test_split_preserves_class_distribution(sample_modeling_data):
    features, target = sample_modeling_data
    split = create_modeling_split(features, target)

    assert int(split.y_train.sum()) == 8
    assert int(split.y_test.sum()) == 2
    assert split.y_train.mean() == pytest.approx(target.mean())
    assert split.y_test.mean() == pytest.approx(target.mean())


def test_split_has_no_overlapping_indices(sample_modeling_data):
    features, target = sample_modeling_data
    split = create_modeling_split(features, target)

    train_indices = set(split.X_train.index)
    test_indices = set(split.X_test.index)

    assert train_indices.isdisjoint(test_indices)
    assert train_indices | test_indices == set(features.index)


def test_split_is_reproducible(sample_modeling_data):
    features, target = sample_modeling_data

    first = create_modeling_split(features, target)
    second = create_modeling_split(features, target)

    assert first.X_train.index.tolist() == second.X_train.index.tolist()
    assert first.X_test.index.tolist() == second.X_test.index.tolist()
    assert first.y_train.tolist() == second.y_train.tolist()
    assert first.y_test.tolist() == second.y_test.tolist()


def test_split_rejects_mismatched_lengths(sample_modeling_data):
    features, target = sample_modeling_data

    with pytest.raises(ValueError, match="row counts do not match"):
        create_modeling_split(features.iloc[:-1], target)


def test_split_rejects_empty_features(sample_modeling_data):
    features, target = sample_modeling_data

    with pytest.raises(ValueError, match="empty feature dataset"):
        create_modeling_split(features.iloc[:0], target.iloc[:0])


def test_split_rejects_missing_targets(sample_modeling_data):
    features, target = sample_modeling_data
    invalid_target = target.astype(float).copy()
    invalid_target.iloc[0] = float("nan")

    with pytest.raises(ValueError, match="missing values"):
        create_modeling_split(features, invalid_target)


def test_split_rejects_nonbinary_targets(sample_modeling_data):
    features, target = sample_modeling_data
    invalid_target = target.copy()
    invalid_target.iloc[0] = 2

    with pytest.raises(ValueError, match="binary values"):
        create_modeling_split(features, invalid_target)


def test_split_rejects_single_class(sample_modeling_data):
    features, target = sample_modeling_data
    invalid_target = pd.Series([0] * len(target))

    with pytest.raises(ValueError, match="Both target classes"):
        create_modeling_split(features, invalid_target)


def test_split_rejects_insufficient_minority_records(sample_modeling_data):
    features, _target = sample_modeling_data
    invalid_target = pd.Series([0] * 99 + [1])

    with pytest.raises(ValueError, match="at least two records"):
        create_modeling_split(features, invalid_target)