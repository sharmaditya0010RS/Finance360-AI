import numpy as np

DEFAULT_BIN_EDGES = np.linspace(0.0, 1.0, 11)
MINIMUM_PROPORTION = 1e-6


def calculate_psi(
    reference_scores: np.ndarray,
    comparison_scores: np.ndarray,
    bin_edges: np.ndarray | None = None,
) -> dict:
    reference = np.asarray(reference_scores, dtype=float)
    comparison = np.asarray(comparison_scores, dtype=float)

    if bin_edges is None:
        edges = DEFAULT_BIN_EDGES.copy()
    else:
        edges = np.asarray(bin_edges, dtype=float)

    if reference.ndim != 1 or comparison.ndim != 1:
        raise ValueError("Score arrays must be one-dimensional")

    if len(reference) == 0 or len(comparison) == 0:
        raise ValueError("Score arrays must be nonempty")

    if not np.isfinite(reference).all() or not np.isfinite(comparison).all():
        raise ValueError("Scores must be finite")

    if np.any((reference < 0) | (reference > 1)):
        raise ValueError("Reference scores must be between 0 and 1")

    if np.any((comparison < 0) | (comparison > 1)):
        raise ValueError("Comparison scores must be between 0 and 1")

    if edges.ndim != 1 or len(edges) < 3:
        raise ValueError("Bin edges must define at least two bins")

    if not np.isfinite(edges).all():
        raise ValueError("Bin edges must be finite")

    if edges[0] != 0 or edges[-1] != 1:
        raise ValueError("Bin edges must span 0 to 1")

    if not np.all(np.diff(edges) > 0):
        raise ValueError("Bin edges must be strictly increasing")

    reference_counts, _ = np.histogram(reference, bins=edges)
    comparison_counts, _ = np.histogram(comparison, bins=edges)

    reference_proportions = reference_counts / len(reference)
    comparison_proportions = comparison_counts / len(comparison)

    adjusted_reference = np.maximum(
        reference_proportions,
        MINIMUM_PROPORTION,
    )
    adjusted_comparison = np.maximum(
        comparison_proportions,
        MINIMUM_PROPORTION,
    )

    psi_contributions = (
        adjusted_comparison - adjusted_reference
    ) * np.log(adjusted_comparison / adjusted_reference)

    bins = []

    for index in range(len(reference_counts)):
        bins.append(
            {
                "bin": index + 1,
                "lower_bound": float(edges[index]),
                "upper_bound": float(edges[index + 1]),
                "reference_count": int(reference_counts[index]),
                "comparison_count": int(comparison_counts[index]),
                "reference_share": float(reference_proportions[index]),
                "comparison_share": float(comparison_proportions[index]),
                "psi_contribution": float(psi_contributions[index]),
            }
        )

    return {
        "metric": "population_stability_index",
        "reference_count": len(reference),
        "comparison_count": len(comparison),
        "binning_strategy": "fixed_probability_bins",
        "smoothing_floor": MINIMUM_PROPORTION,
        "psi": float(psi_contributions.sum()),
        "bins": bins,
        "interpretation": (
            "PSI measures score-distribution differences and does not "
            "establish a cause or prove model performance deterioration."
        ),
    }