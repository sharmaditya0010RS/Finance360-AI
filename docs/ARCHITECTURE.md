# System Architecture

```text
Home Credit historical application data
                |
                v
      Python data preparation
       /                  \
      v                    v
PostgreSQL warehouse   Modeling features
(fact + dimensions)          |
      |                       v
      v               Scikit-learn / XGBoost
Analytics views                |
      |                        v
      |                Saved model + reports
      |                        |
      +------------+-----------+
                   |
                   v
           Streamlit dashboards
           /                 \
   Portfolio analysis     Prediction / simulator

PostgreSQL warehouse ---> Power BI (3-page report)
```

## Core components

| Layer | Location | Responsibility |
| --- | --- | --- |
| Data pipeline | `src/finance360/` | Data preparation and validation |
| Warehouse | `sql/` and PostgreSQL | Fact/dimension model and analytics views |
| Model contract | `src/finance360/modeling/feature_contract.py` | Features and data expectations |
| Inference | `src/finance360/dashboard/prediction.py` | Input validation and model scoring |
| Monitoring | `src/finance360/monitoring/` | Historical segmentation and score comparisons |
| App | `app/` | Streamlit interactive analysis |
| BI | `powerbi/` | Executive reporting |
| Tests | `tests/` | Automated regression checks |

## Boundaries

The Power BI report uses PostgreSQL warehouse data. The Streamlit application combines warehouse analytics with saved model/report artifacts. This project does not claim real-time ingestion, live production monitoring, an operational decision API or automated lending decisions.

## Deployment note

The project may require a local PostgreSQL instance, source data and generated model/report artifacts. Containerization and continuous integration should be documented as operational only after their configuration and successful execution are verified.
