# Finance360 AI

**Enterprise Credit Risk Intelligence & Predictive Analytics Platform**

Finance360 AI is an end-to-end portfolio project for exploring historical credit-application data, building a PostgreSQL analytics warehouse, estimating repayment-difficulty risk with machine learning, and presenting results through Streamlit and Power BI.

> **Educational decision support only.** Model scores are estimates, not lending decisions, guarantees of repayment behavior, regulatory risk grades, or causal effects. Historical application outcomes are not evidence of actual financial losses.

## Project capabilities

- **Data engineering:** Python-based preparation and quality checks for the Home Credit Default Risk dataset.
- **Analytics warehouse:** PostgreSQL fact/dimension model and analytics views.
- **Machine learning:** Scikit-learn/XGBoost pipeline, held-out evaluation, feature importance and SHAP-based analysis.
- **Risk segmentation:** Training-only out-of-fold score quantiles used for relative risk bands.
- **Streamlit:** Executive command center, applicant risk predictor, portfolio analytics, model explainability, monitoring, data quality, what-if simulator and historical hotspot explorer.
- **Power BI:** Three-page executive report covering overview, credit risk and data quality.
- **Automated checks:** Pytest suite covering core analytics and dashboard behavior.

## Data and interpretation

Source: **Home Credit Default Risk** training data. The modeling data contract expects **307,511 applications**, including **24,825 records with `TARGET=1`** (approximately **8.07%**). `TARGET=1` is described here as *recorded repayment difficulty*; it should not be labeled a verified loss or proven default event.

Requested credit is an application amount, **not confirmed disbursement**. Monetary values remain in source-data units unless a verified currency definition is available. No transaction-date time series or live production scoring feed is claimed.

## Technology

Python 3.12, pandas, NumPy, PyArrow, PostgreSQL, psycopg, SQLAlchemy, scikit-learn, XGBoost, SHAP, Plotly, Streamlit, Power BI, pytest and Ruff.


```

## Local setup (Windows PowerShell)

Prerequisites: **Python 3.12** and a configured **PostgreSQL** instance. Database initialization, input datasets and trained artifacts must be provisioned separately; cloning this repository alone may not reproduce them.

```powershell
# From the repository root
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"

# Create local environment files using the supplied template.
# Do not commit passwords, connection strings or secrets.
Copy-Item .env.example .env

# Run automated tests
python -m pytest -q

# Start the Streamlit app
python -m streamlit run app/Executive_Risk_Command_Center.py
```

Configure `.env` and the PostgreSQL database according to the variables and SQL scripts in your local project. The trained inference artifact is expected at `artifacts/models/finance360_xgboost_v1.joblib`, and risk-band definitions at `reports/monitoring/risk_segmentation.json`.

**Note:** Tests requiring local data, database services or model artifacts may not run on a fresh clone until those dependencies are available.

## Application pages

| Page | Purpose |
| --- | --- |
| Executive Risk Command Center | Main Streamlit entry point |
| AI Credit Risk Predictor | Individual application probability estimate |
| Portfolio Risk Analytics | Historical risk bands and requested-credit exposure |
| Model Performance & Explainability | Model evaluation and explanation |
| Risk Monitoring Center | Historical score-distribution comparison |
| Data Quality & Governance | Quality diagnostics and governance |
| Decision Simulator | Baseline versus hypothetical what-if model scores |
| Historical Risk Hotspots | Segment-level observed difficulty comparisons |

## Power BI report

`powerbi/Finance360_Executive_Dashboard.pbix` is the intended local report path. The report contains **three pages**: Executive Overview, Credit Risk Analysis, and Data Quality & Health. It uses PostgreSQL warehouse tables and manually authored DAX measures. The report may need refreshed local database credentials and data access after opening on another computer.

## Model governance

- **Prediction target:** Historical recorded repayment difficulty (`TARGET`).
- **Feature contract:** Defined in `src/finance360/modeling/feature_contract.py`.
- **Model artifact:** `artifacts/models/finance360_xgboost_v1.joblib`.
- **Risk bands:** Relative training-score quantiles; not regulatory grades.
- **External scores:** Anonymized dataset-specific inputs; missing values may reduce confidence in an individual estimate.
- **Simulator:** Compares model predictions for hypothetical inputs. Differences do **not** establish causality.
- **Monitoring:** Historical demonstration; **not** production drift surveillance.
- **Hotspots:** Descriptive segment-level associations; sample size and selection effects matter.

See [`docs/MODEL_GOVERNANCE.md`](docs/MODEL_GOVERNANCE.md) for a fuller checklist and limitations.

## Verification

Run:

```powershell
python -m pytest -q
python -m ruff check src app tests
```

The last independently reported project test run was **211 passed**, before the hotspot additions. Update this statement only after the complete suite has been rerun locally.

## Security and reproducibility

Do not commit `.env`, `.env.database`, credentials, raw datasets, personal information, virtual environments or model artifacts unless an explicit and appropriate publication policy has been established. Check `.gitignore` and staged files before publishing.

## Status

**Phases 1â€“9:** Implemented locally, according to project checkpoints.
**Phase 10:** Documentation and governance in progress.
**Phase 11:** Repository polish, publication and deployment preparation pending.
