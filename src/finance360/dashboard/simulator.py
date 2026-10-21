"""What-if scenario comparisons using the existing Finance360 inference pipeline."""

from copy import deepcopy
from pathlib import Path

import numpy as np

from finance360.dashboard.prediction import predict_applicant, prepare_applicant_features

EDITABLE_FIELDS = frozenset({"AMT_INCOME_TOTAL", "AMT_CREDIT", "AMT_ANNUITY"})


def compare_scenarios(applicant: dict, changes: dict, project_root: Path) -> dict:
    """Compare baseline and hypothetical model estimates, not causal outcomes."""
    if not isinstance(applicant, dict) or not isinstance(changes, dict):
        raise TypeError("Applicant and changes must be dictionaries")
    if not changes:
        raise ValueError("Provide at least one scenario change")
    unsupported = set(changes) - EDITABLE_FIELDS
    if unsupported:
        raise ValueError(f"Unsupported scenario fields: {sorted(unsupported)}")

    baseline = deepcopy(applicant)
    scenario = deepcopy(applicant)
    for field, value in changes.items():
        if isinstance(value, (bool, str)) or not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(f"Scenario value must be numeric: {field}")
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f"Scenario value must be finite and positive: {field}")
        scenario[field] = float(value)

    # Validate both profiles before loading model artifacts.
    prepare_applicant_features(baseline)
    prepare_applicant_features(scenario)
    baseline_result = predict_applicant(baseline, project_root)
    scenario_result = predict_applicant(scenario, project_root)
    delta = (
        scenario_result["repayment_difficulty_probability"]
        - baseline_result["repayment_difficulty_probability"]
    )
    return {
        "baseline": baseline_result,
        "scenario": scenario_result,
        "probability_delta": float(delta),
        "percentage_point_delta": float(delta * 100),
        "changes": {field: {"before": baseline[field], "after": scenario[field]} for field in changes},
        "interpretation": (
            "Model sensitivity to hypothetical inputs only; not a causal effect, "
            "credit decision, or guarantee of real-world repayment outcomes."
        ),
    }
