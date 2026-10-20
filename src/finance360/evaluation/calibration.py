import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def calculate_calibration(
    targets: pd.Series | np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict:
    actual = np.asarray(targets)
    predicted = np.asarray(probabilities, dtype=float)

    if actual.ndim != 1 or predicted.ndim != 1:
        raise ValueError("Targets and probabilities must be one-dimensional")

    if len(actual) != len(predicted):
        raise ValueError("Target and probability lengths do not match")

    if len(actual) == 0:
        raise ValueError("Calibration data cannot be empty")

    if n_bins < 2:
        raise ValueError("At least two calibration bins are required")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Targets must contain only binary values")

    if not np.isfinite(predicted).all():
        raise ValueError("Probabilities must be finite")

    if ((predicted < 0) | (predicted > 1)).any():
        raise ValueError("Probabilities must lie between zero and one")

    bin_indices = np.minimum(
        np.floor(predicted * n_bins).astype(int),
        n_bins - 1,
    )

    bins = []
    expected_calibration_error = 0.0

    for bin_number in range(n_bins):
        mask = bin_indices == bin_number
        count = int(mask.sum())

        if count == 0:
            continue

        mean_probability = float(predicted[mask].mean())
        observed_rate = float(actual[mask].mean())
        absolute_gap = abs(mean_probability - observed_rate)

        expected_calibration_error += (
            count / len(actual)
        ) * absolute_gap

        bins.append(
            {
                "bin": bin_number + 1,
                "lower_bound": bin_number / n_bins,
                "upper_bound": (bin_number + 1) / n_bins,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_gap": absolute_gap,
            }
        )

    return {
        "rows": len(actual),
        "positive_count": int(actual.sum()),
        "observed_positive_rate": float(actual.mean()),
        "mean_predicted_probability": float(predicted.mean()),
        "brier_score": float(brier_score_loss(actual, predicted)),
        "expected_calibration_error": float(expected_calibration_error),
        "binning_strategy": "equal_width",
        "requested_bins": n_bins,
        "populated_bins": len(bins),
        "bins": bins,
    }