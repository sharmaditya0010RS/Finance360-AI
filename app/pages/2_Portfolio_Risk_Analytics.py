from pathlib import Path

import pandas as pd
import psycopg
import streamlit as st

from finance360.dashboard.data_access import load_dashboard_data
from finance360.dashboard.theme import apply_theme
from finance360.dashboard.visualizations import (
    applicant_vs_credit_share_chart,
    observed_vs_predicted_chart,
    risk_outcome_composition_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RISK_ORDER = ["Low", "Moderate", "High", "Very High"]

st.set_page_config(
    page_title="Portfolio Risk Analytics | Finance360 AI",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(ttl=300, show_spinner="Loading portfolio risk intelligence...")
def get_data() -> dict:
    return load_dashboard_data(PROJECT_ROOT)


st.sidebar.title("FINANCE360 AI")
st.sidebar.caption("Portfolio Risk Analytics")
st.sidebar.divider()
st.sidebar.caption("Historical training population | Not live production data")

st.title("Portfolio Risk Analytics")
st.caption(
    "Explore model-derived risk segments, observed repayment difficulty "
    "and requested credit concentration using validated historical evidence."
)

try:
    data = get_data()
except (OSError, ValueError, TypeError, KeyError, psycopg.Error) as error:
    st.error(f"Unable to load portfolio analytics: {error}")
    st.stop()

risk_report = data["risk_segmentation"]
exposure_report = data["portfolio_exposure"]

risk_frame = pd.DataFrame(risk_report["segments"])
exposure_frame = pd.DataFrame(exposure_report["segments"])

if (
    int(risk_frame["application_count"].sum()) != int(risk_report["training_rows"])
    or int(exposure_frame["application_count"].sum())
    != int(exposure_report["application_count"])
):
    st.error("Risk segment populations do not reconcile with source reports.")
    st.stop()

if int(risk_report["training_rows"]) != int(exposure_report["application_count"]):
    st.error("Risk segmentation and exposure populations do not match.")
    st.stop()

selected_bands = st.multiselect(
    "Select risk bands to investigate",
    options=RISK_ORDER,
    default=RISK_ORDER,
)

if not selected_bands:
    st.warning("Select at least one risk band to continue.")
    st.stop()

risk_selected = risk_frame[risk_frame["risk_band"].isin(selected_bands)].copy()
exposure_selected = exposure_frame[
    exposure_frame["risk_band"].isin(selected_bands)
].copy()

selected_applications = int(risk_selected["application_count"].sum())
selected_positives = int(risk_selected["positive_count"].sum())
selected_credit = float(exposure_selected["total_requested_credit"].sum())
selected_difficulty_rate = selected_positives / selected_applications * 100

metric_a, metric_b, metric_c, metric_d = st.columns(4)

with metric_a:
    st.metric("Selected applications", f"{selected_applications:,}")

with metric_b:
    st.metric("Recorded difficulty", f"{selected_positives:,}")

with metric_c:
    st.metric("Observed difficulty rate", f"{selected_difficulty_rate:.2f}%")

with metric_d:
    st.metric("Requested credit", f"{selected_credit / 1_000_000_000:,.2f}B")

st.caption(
    "Requested credit is shown in source-data units. "
    "No verified currency conversion or loss estimate is implied."
)

st.divider()

st.subheader("Risk Band Comparison")

comparison = risk_selected[
    [
        "risk_band",
        "application_count",
        "positive_count",
        "observed_difficulty_rate",
        "mean_predicted_probability",
    ]
].merge(
    exposure_selected[
        [
            "risk_band",
            "total_requested_credit",
            "requested_credit_share",
        ]
    ],
    on="risk_band",
    validate="one_to_one",
)

comparison["Observed difficulty (%)"] = (
    comparison["observed_difficulty_rate"] * 100
)
comparison["Mean model score (%)"] = (
    comparison["mean_predicted_probability"] * 100
)
comparison["Credit share (%)"] = comparison["requested_credit_share"] * 100
comparison["Requested credit (billions)"] = (
    comparison["total_requested_credit"] / 1_000_000_000
)

display_frame = comparison[
    [
        "risk_band",
        "application_count",
        "positive_count",
        "Observed difficulty (%)",
        "Mean model score (%)",
        "Credit share (%)",
        "Requested credit (billions)",
    ]
].rename(
    columns={
        "risk_band": "Risk band",
        "application_count": "Applications",
        "positive_count": "Recorded difficulty",
    }
)

st.dataframe(
    display_frame,
    width="stretch",
    hide_index=True,
)

st.download_button(
    "Download selected risk comparison (CSV)",
    data=display_frame.to_csv(index=False).encode("utf-8"),
    file_name="finance360_risk_comparison.csv",
    mime="text/csv",
)

st.caption(
    "Credit share percentages refer to the full training portfolio, "
    "not only the selected risk bands."
)

st.divider()

st.subheader("Historical Risk Evidence")

left, right = st.columns(2)

with left:
    st.plotly_chart(
        observed_vs_predicted_chart(risk_report),
        width="stretch",
    )
    st.caption(
        "Observed and predicted rates use training out-of-fold scores. "
        "The chart displays all four bands for consistent comparison."
    )

with right:
    st.plotly_chart(
        applicant_vs_credit_share_chart(exposure_report),
        width="stretch",
    )
    st.caption(
        "Applicant and requested-credit shares are measured against "
        "the full training portfolio."
    )

st.plotly_chart(
    risk_outcome_composition_chart(risk_report),
    width="stretch",
)

st.caption(
    "The stacked outcome chart shows historical recorded difficulty "
    "and non-difficulty counts across all risk bands."
)

with st.expander("Methodology, evidence and limitations"):
    st.markdown(
        "Risk bands are determined using quantiles of out-of-fold "
        "XGBoost predictions on the training population."
    )
    st.markdown(
        f"**Training applications:** {risk_report['training_rows']:,}"
    )
    st.markdown(
        "**Sources:** "
        "`reports/monitoring/risk_segmentation.json` and "
        "`reports/monitoring/portfolio_exposure.json`."
    )
    st.warning(
        "These results are historical decision-support analytics. "
        "They do not establish production drift, expected credit loss, "
        "regulatory capital or applicant approval eligibility."
    )

st.divider()
st.caption(
    "Finance360 AI | Historical portfolio analysis | "
    "Educational and analytical use only."
)