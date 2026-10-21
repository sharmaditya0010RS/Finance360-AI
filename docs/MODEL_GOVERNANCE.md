# Model Governance and Responsible Use

## Intended use

Finance360 AI is a portfolio demonstration for historical credit-risk analytics and model-assisted exploration. It is **not** an automated lending decision system and must not be used to approve, deny, price, or otherwise determine an individual's credit eligibility.

## Model and data lineage

- Dataset: Home Credit Default Risk, historical application records.
- Target: `TARGET=1` represents recorded repayment difficulty in the source dataset.
- Model inputs: See `src/finance360/modeling/feature_contract.py` (24 modeled features, including derived ratios).
- Inference: `src/finance360/dashboard/prediction.py` prepares inputs and invokes the saved model.
- Model file: `artifacts/models/finance360_xgboost_v1.joblib` (local artifact).
- Segmentation: `reports/monitoring/risk_segmentation.json` defines boundaries from training-only out-of-fold scores.

## Key interpretation limits

1. Estimated probability is not a guaranteed individual outcome; calibration and generalization should be evaluated on relevant populations.
2. Relative risk bands are descriptive score groupings, not statutory or regulatory grades.
3. What-if simulation changes model inputs, not the real-world applicant. Differences in model output do not demonstrate causal effects or justify applicant actions.
4. Missing anonymized external scores can affect reliability; do not fabricate scores.
5. Requested credit is not a disbursement or loss amount. Currency should not be assumed.
6. Historical risk hotspots may reflect composition, confounding, sampling variability or missingness. Show sample sizes and avoid ranking very small segments as reliable hotspots.
7. Historical PSI comparisons between random groups are not time-based production monitoring.

## Controls and validation checklist

- [ ] Verify source dataset provenance, license and access rights.
- [ ] Confirm the exact training/validation/test split and leakage protections.
- [ ] Record held-out discrimination and calibration metrics from evaluation reports; do not invent values.
- [ ] Test model input schema, missingness and extreme-value handling.
- [ ] Validate that the full automated test suite passes.
- [ ] Review subgroup performance, bias and fairness limitations before any real-world use.
- [ ] Document model and data versions, environment, owner and retraining triggers.
- [ ] Define monitoring thresholds using future, timestamped data before claiming production monitoring.
- [ ] Keep credentials, personal data and sensitive artifacts out of version control.
- [ ] Obtain legal, compliance and domain-expert review before considering real lending applications.

## Reproducibility notes

A fresh clone may not include raw data, database contents, generated reports or trained artifacts. Setup instructions must identify these dependencies explicitly. Do not report tests or deployment as successful without running them in the target environment.
