from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import psycopg
import streamlit as st

from finance360.dashboard.data_access import load_dashboard_data
from finance360.dashboard.theme import apply_theme
from finance360.dashboard.visualizations import (
    COLORS,
    score_distribution_chart,
    style_chart,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="Risk Monitoring Center | Finance360 AI",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(ttl=300, show_spinner="Loading historical monitoring evidence...")
def get_data() -> dict:
    return load_dashboard_data(PROJECT_ROOT)


def psi_contribution_chart(frame: pd.DataFrame) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=frame["Score range"],
            y=frame["PSI contribution"],
            marker_color=COLORS["purple"],
            hovertemplate=(
                "Score range: %{x}<br>"
                "PSI contribution: %{y:.8f}<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(title_text="Predicted probability bin")
    fig.update_yaxes(title_text="PSI contribution")

    return style_chart(
        fig,
        "PSI Contribution by Historical Score Bin",
        height=420,
    )


st.sidebar.title("FINANCE360 AI")
st.sidebar.caption("Risk Monitoring Center")
st.sidebar.divider()
st.sidebar.caption("Historical demonstration | No production scoring feed")

st.title("Risk Monitoring Center")
st.caption(
    "Explore score-distribution comparisons and PSI diagnostics "
    "using two random subsets of historical training predictions."
)

try:
    data = get_data()
except (OSError, ValueError, TypeError, KeyError, psycopg.Error) as error:
    st.error(f"Unable to load monitoring evidence: {error}")
    st.stop()

monitoring = data["score_monitoring"]
psi_result = monitoring["psi_result"]

reference_count = int(monitoring["reference_count"])
comparison_count = int(monitoring["comparison_count"])
training_rows = int(monitoring["training_rows"])
psi_value = float(psi_result["psi"])

if reference_count + comparison_count != training_rows:
    st.error("Monitoring populations do not reconcile with training data.")
    st.stop()

if monitoring["production_drift_claim"] is not False:
    st.error("Monitoring report contains an unexpected production drift claim.")
    st.stop()

bins = pd.DataFrame(psi_result["bins"])

required = {
    "bin",
    "lower_bound",
    "upper_bound",
    "reference_count",
    "comparison_count",
    "reference_share",
    "comparison_share",
    "psi_contribution",
}

if not required.issubset(bins.columns):
    st.error("Monitoring report is missing required PSI bin fields.")
    st.stop()

if (
    int(bins["reference_count"].sum()) != reference_count
    or int(bins["comparison_count"].sum()) != comparison_count
):
    st.error("PSI bin populations do not reconcile.")
    st.stop()

if (
    abs(float(bins["reference_share"].sum()) - 1) > 0.001
    or abs(float(bins["comparison_share"].sum()) - 1) > 0.001
):
    st.error("PSI bin distributions do not sum to 100%.")
    st.stop()

if abs(float(bins["psi_contribution"].sum()) - psi_value) > 0.000001:
    st.error("PSI contributions do not reconcile with the reported PSI.")
    st.stop()

bins = bins.sort_values("bin").copy()

bins["Score range"] = [
    f"{lower:.0%}–{upper:.0%}"
    for lower, upper in zip(
        bins["lower_bound"],
        bins["upper_bound"],
        strict=True,
    )
]

bins["Reference share (%)"] = bins["reference_share"] * 100
bins["Comparison share (%)"] = bins["comparison_share"] * 100
bins["PSI contribution"] = bins["psi_contribution"]

st.info(
    "Historical random-half comparison only. "
    "This is not time-based drift monitoring and does not measure "
    "performance after deployment."
)

metric_a, metric_b, metric_c, metric_d = st.columns(4)

with metric_a:
    st.metric("Reference applications", f"{reference_count:,}")

with metric_b:
    st.metric("Comparison applications", f"{comparison_count:,}")

with metric_c:
    st.metric("Historical PSI", f"{psi_value:.6f}")

with metric_d:
    st.metric(
        "Mean score difference",
        (
            f"{(float(monitoring['comparison_mean_score']) - float(monitoring['reference_mean_score'])) * 100:+.4f}"
            " pp"
        ),
    )

st.caption(
    "Mean score difference is comparison minus reference, "
    "expressed in percentage points. "
    "PSI measures distribution differences, not predictive accuracy."
)

distribution_tab, contribution_tab, governance_tab = st.tabs(
    [
        "Score Distribution",
        "PSI Bin Explorer",
        "Monitoring Governance",
    ]
)

with distribution_tab:
    st.subheader("Historical Score Distribution")

    st.plotly_chart(
        score_distribution_chart(monitoring),
        width="stretch",
    )

    st.caption(
        "Both groups are random halves of training out-of-fold scores. "
        "They are not separate time periods."
    )

    left, right = st.columns(2)

    with left:
        st.metric(
            "Reference mean probability",
            f"{float(monitoring['reference_mean_score']):.3%}",
        )

    with right:
        st.metric(
            "Comparison mean probability",
            f"{float(monitoring['comparison_mean_score']):.3%}",
        )

    st.markdown(
        "A small historical PSI in this comparison means these two "
        "randomly selected score distributions are similar. "
        "It does not establish that future populations will remain similar."
    )

with contribution_tab:
    st.subheader("PSI Contribution Analysis")

    st.plotly_chart(
        psi_contribution_chart(bins),
        width="stretch",
    )

    selected_bins = st.multiselect(
        "Select probability bins to investigate",
        options=bins["Score range"].tolist(),
        default=bins["Score range"].tolist(),
    )

    if selected_bins:
        selected = bins[
            bins["Score range"].isin(selected_bins)
        ].copy()

        display = selected[
            [
                "Score range",
                "reference_count",
                "comparison_count",
                "Reference share (%)",
                "Comparison share (%)",
                "PSI contribution",
            ]
        ].rename(
            columns={
                "reference_count": "Reference applications",
                "comparison_count": "Comparison applications",
            }
        )

        st.dataframe(
            display,
            width="stretch",
            hide_index=True,
        )

        st.download_button(
            "Download selected PSI bins (CSV)",
            data=display.to_csv(index=False).encode("utf-8"),
            file_name="finance360_historical_psi_bins.csv",
            mime="text/csv",
        )

        st.caption(
            "Selecting bins filters this table and CSV only. "
            "The chart and headline PSI always represent all ten bins."
        )
    else:
        st.warning("Select at least one probability bin to view its details.")

with governance_tab:
    st.subheader("Monitoring Methodology & Limitations")

    st.markdown(
        f"**Comparison type:** `{monitoring['comparison_type']}`"
    )

    st.markdown(
        f"**Grouping method:** {monitoring['grouping_method']}"
    )

    st.markdown(
        f"**Reference group:** {monitoring['reference_group']}"
    )

    st.markdown(
        f"**Comparison group:** {monitoring['comparison_group']}"
    )

    st.markdown(
        f"**Training applications:** {training_rows:,}"
    )

    st.markdown(
        "**Production drift evaluated:** No"
    )

    st.markdown("**Recorded limitations:**")

    for limitation in monitoring["limitations"]:
        st.markdown(f"- {limitation}")

    st.warning(
        "A production monitoring system would require timestamped "
        "scoring records, future applicant populations, score history, "
        "observed outcomes and a defined monitoring policy. "
        "These inputs are not available in the current historical dataset."
    )

    st.caption(
        "Source: reports/monitoring/score_distribution_monitoring.json"
    )

st.divider()
st.caption(
    "Finance360 AI | Historical monitoring demonstration | "
    "Not a live production monitoring service."
)