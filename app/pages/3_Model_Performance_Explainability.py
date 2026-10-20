import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from finance360.dashboard.data_access import load_dashboard_data
from finance360.dashboard.theme import apply_theme
from finance360.dashboard.visualizations import (
    COLORS,
    confusion_matrix_chart,
    held_out_calibration_chart,
    model_evidence_chart,
    style_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODELING_ROOT = PROJECT_ROOT / "reports" / "modeling"

st.set_page_config(
    page_title="Model Performance & Explainability | Finance360 AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(ttl=300, show_spinner="Loading model intelligence...")
def get_data() -> dict:
    dashboard_data = load_dashboard_data(PROJECT_ROOT)

    with (MODELING_ROOT / "xgboost_feature_importance.json").open(
        "r", encoding="utf-8"
    ) as file:
        importance = json.load(file)

    with (MODELING_ROOT / "xgboost_shap_summary.json").open(
        "r", encoding="utf-8"
    ) as file:
        shap_summary = json.load(file)

    with (MODELING_ROOT / "model_selection.json").open(
        "r", encoding="utf-8"
    ) as file:
        selection = json.load(file)

    return {
        "evaluation": dashboard_data["held_out_evaluation"],
        "importance": importance,
        "shap_summary": shap_summary,
        "selection": selection,
    }


def feature_importance_chart(frame: pd.DataFrame) -> go.Figure:
    selected = frame.sort_values("normalized_gain").tail(12)

    fig = go.Figure(
        go.Bar(
            x=selected["normalized_gain"].astype(float) * 100,
            y=selected["feature"],
            orientation="h",
            marker_color=COLORS["blue"],
            hovertemplate=(
                "%{y}<br>Share of total gain: %{x:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(title_text="Share of total gain (%)")

    return style_chart(
        fig,
        "Top 12 Features by XGBoost Gain",
        height=500,
    )


def shap_importance_chart(frame: pd.DataFrame) -> go.Figure:
    selected = frame.sort_values("mean_absolute_shap").tail(12)

    fig = go.Figure(
        go.Bar(
            x=selected["mean_absolute_shap"].astype(float),
            y=selected["feature"],
            orientation="h",
            marker_color=COLORS["purple"],
            hovertemplate=(
                "%{y}<br>Mean absolute SHAP: %{x:.4f} log-odds"
                "<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(
        title_text="Mean absolute SHAP contribution (log-odds)"
    )

    return style_chart(
        fig,
        "Top 12 Features by Global SHAP Magnitude",
        height=500,
    )


st.sidebar.title("FINANCE360 AI")
st.sidebar.caption("Model Intelligence Center")
st.sidebar.divider()
st.sidebar.caption("Validated historical evidence | Not production monitoring")

st.title("Model Performance & Explainability")
st.caption(
    "Independent held-out evaluation, global feature influence "
    "and responsible interpretation of XGBoost risk predictions."
)

try:
    data = get_data()
except (OSError, ValueError, TypeError, KeyError, psycopg.Error) as error:
    st.error(f"Unable to load model intelligence reports: {error}")
    st.stop()

evaluation = data["evaluation"]
importance_report = data["importance"]
shap_report = data["shap_summary"]
selection = data["selection"]

importance_frame = pd.DataFrame(importance_report["feature_importance"])
shap_frame = pd.DataFrame(shap_report["global_feature_importance"])

if (
    len(importance_frame) != int(importance_report["feature_count"])
    or len(shap_frame) != int(shap_report["source_feature_count"])
):
    st.error("Feature attribution counts do not reconcile with source reports.")
    st.stop()

if int(evaluation["test_rows"]) != int(selection["held_out_test_rows"]):
    st.error("Model selection and held-out evaluation populations do not match.")
    st.stop()

st.subheader("Held-Out Model Performance")

metric_a, metric_b, metric_c, metric_d = st.columns(4)

with metric_a:
    st.metric("ROC-AUC", f"{float(evaluation['roc_auc']):.4f}")

with metric_b:
    st.metric(
        "Average precision",
        f"{float(evaluation['average_precision']):.4f}",
    )

with metric_c:
    st.metric("Brier score", f"{float(evaluation['brier_score']):.4f}")

with metric_d:
    st.metric(
        "Recall at 50% threshold",
        f"{float(evaluation['recall']):.2%}",
    )

st.warning(
    "At the 50% classification threshold, recall is approximately 1.41%. "
    "Most recorded repayment-difficulty cases are not flagged at this "
    "threshold. This is not a validated lending decision policy."
)

st.caption(
    f"Held-out test applications: {int(evaluation['test_rows']):,} | "
    f"Recorded difficulty cases: {int(evaluation['test_positive_count']):,} | "
    f"Classification threshold: {float(evaluation['decision_threshold']):.0%}"
)

performance_tab, calibration_tab, explainability_tab, governance_tab = st.tabs(
    [
        "Performance Diagnostics",
        "Probability Calibration",
        "Feature Explainability",
        "Model Governance",
    ]
)

with performance_tab:
    st.subheader("Held-Out Performance Diagnostics")

    st.plotly_chart(
        model_evidence_chart(evaluation),
        width="stretch",
    )

    st.caption(
        "ROC-AUC evaluates ranking, average precision evaluates "
        "precision-recall behavior, and Brier score evaluates "
        "probability prediction error. These metrics are not interchangeable."
    )

    st.plotly_chart(
        confusion_matrix_chart(evaluation),
        width="stretch",
    )

    matrix = evaluation["confusion_matrix"]

    st.info(
        f"At the 50% threshold: {int(matrix['true_positive']):,} "
        f"recorded difficulty cases were identified, while "
        f"{int(matrix['false_negative']):,} were missed."
    )

    st.caption(
        "Source: reports/modeling/held_out_test_evaluation.json"
    )

with calibration_tab:
    st.subheader("Predicted Probability vs Observed Difficulty")

    st.plotly_chart(
        held_out_calibration_chart(evaluation),
        width="stretch",
    )

    st.metric(
        "Expected calibration error",
        f"{float(evaluation['expected_calibration_error']):.4f}",
    )

    st.markdown(
        "The dashed diagonal represents perfect calibration. "
        "Each model point summarizes a populated probability bin "
        "from the independent held-out test set."
    )

    st.caption(
        "Good aggregate calibration does not guarantee reliable "
        "probabilities for every subgroup or individual applicant."
    )

with explainability_tab:
    st.subheader("Global Feature Importance")

    left, right = st.columns(2)

    with left:
        st.plotly_chart(
            feature_importance_chart(importance_frame),
            width="stretch",
        )

    with right:
        st.plotly_chart(
            shap_importance_chart(shap_frame),
            width="stretch",
        )

    st.info(
        "XGBoost gain measures how much features improve tree splits. "
        "Mean absolute SHAP measures average contribution magnitude "
        "to model output. These methods answer different questions."
    )

    st.subheader("Explore Individual Features")

    features = sorted(
        set(importance_frame["feature"])
        & set(shap_frame["feature"])
    )

    selected_feature = st.selectbox(
        "Choose a feature",
        options=features,
        index=(
            features.index("EXT_SOURCE_MEAN")
            if "EXT_SOURCE_MEAN" in features
            else 0
        ),
    )

    gain_row = importance_frame.loc[
        importance_frame["feature"] == selected_feature
    ].iloc[0]

    shap_row = shap_frame.loc[
        shap_frame["feature"] == selected_feature
    ].iloc[0]

    feature_a, feature_b, feature_c = st.columns(3)

    with feature_a:
        st.metric(
            "Share of model gain",
            f"{float(gain_row['normalized_gain']):.2%}",
        )

    with feature_b:
        st.metric(
            "Mean absolute SHAP",
            f"{float(shap_row['mean_absolute_shap']):.4f}",
        )

    with feature_c:
        st.metric(
            "Mean signed SHAP",
            f"{float(shap_row['mean_signed_shap']):+.4f}",
        )

    st.caption(
        "SHAP values are in log-odds, not percentage points. "
        "A signed global average can hide opposing contributions "
        "across different applicants. Feature influence is not causation."
    )

    st.dataframe(
        importance_frame[
            ["feature", "rank", "normalized_gain", "total_gain"]
        ].sort_values("rank"),
        width="stretch",
        hide_index=True,
    )

    st.caption(
        "Sources: reports/modeling/xgboost_feature_importance.json "
        "and reports/modeling/xgboost_shap_summary.json"
    )

with governance_tab:
    st.subheader("Model Evidence & Responsible Use")

    st.markdown(
        f"**Model:** {evaluation['model_name']}  \n"
        f"**Version:** {evaluation['model_version']}  \n"
        f"**Training applications:** {int(evaluation['training_rows']):,}  \n"
        f"**Held-out test applications:** {int(evaluation['test_rows']):,}"
    )

    st.markdown(
        "**Model selection:** Cross-validation results were used "
        "to choose the champion before independent held-out evaluation."
    )

    st.markdown(
        "**Global explanations:** Feature gain and SHAP summaries "
        "describe model behavior, not causal relationships."
    )

    st.markdown(
        "**Known limitations:**"
    )

    for limitation in evaluation["limitations"]:
        st.markdown(f"- {limitation}")

    st.warning(
        "TARGET represents recorded repayment difficulty, not necessarily "
        "formal default. The source lacks verified event dates and "
        "loss-given-default information. No automated lending decisions "
        "should be based on this demonstration."
    )

st.divider()
st.caption(
    "Finance360 AI | Historical XGBoost model intelligence | "
    "Educational and analytical use only."
)