# Dashboard Guide

## Streamlit

Launch from the repository root:

```powershell
python -m streamlit run app/Executive_Risk_Command_Center.py
```

Available pages include the executive command center, credit risk predictor, portfolio analytics, model performance/explainability, risk monitoring, data quality/governance, decision simulator and historical risk hotspots.

### Decision Simulator

Compare a baseline application profile with a hypothetical variation. Review both model-estimated repayment-difficulty probabilities, their percentage-point difference and their relative risk bands. Treat comparisons as **model sensitivity**, not causal interventions or approval advice.

### Historical Risk Hotspots

Explore segment-level historical application counts, recorded difficulty counts and observed rates relative to the overall portfolio. Apply a minimum sample-size threshold and inspect category counts before interpreting elevated rates.

### Data dependencies

The application uses local model/report artifacts and PostgreSQL analytics views. A database connection or missing local artifact may prevent a page from loading. No live scoring or production monitoring is implied.

## Power BI

Open the local `powerbi/Finance360_Executive_Dashboard.pbix` report in Power BI Desktop and configure access to the PostgreSQL `finance360` database as required.

Exactly three report pages:

1. **Executive Overview** — application counts, repayment difficulty, requested credit and composition.
2. **Credit Risk Analysis** — historical difficulty-rate comparisons across applicant segments.
3. **Data Quality & Health** — sentinel and financial-quality indicators.

Five dimensional relationships connect the application fact table with contract type, income type, education, housing and family status. Filters and slicers should be checked for expected interactions. A date-based trend must not be inferred without a validated date field.

## Interpretation

- Historical `TARGET=1` means recorded repayment difficulty, not confirmed monetary loss.
- `amt_credit` means requested credit, not necessarily disbursed funds.
- Values remain in source units where currency is not verified.
- Data-quality flags should be understood from their implemented definitions before making operational claims.
