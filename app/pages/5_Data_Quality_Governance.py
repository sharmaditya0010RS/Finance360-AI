import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance360.dashboard.theme import apply_theme
from finance360.dashboard.visualizations import COLORS, style_chart

PROJECT_ROOT = Path(__file__).resolve().parents[2]
QUALITY_ROOT = PROJECT_ROOT / "reports" / "data_quality"
MANIFEST_PATH = PROJECT_ROOT / "data" / "bronze" / "ingestion_manifest.json"

st.set_page_config(
    page_title="Data Quality & Governance | Finance360 AI",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()


@st.cache_data(ttl=300, show_spinner="Loading data governance evidence...")
def get_reports() -> dict:
    paths = {
        "bronze": QUALITY_ROOT / "bronze_profile.json",
        "silver": QUALITY_ROOT / "silver_quality_report.json",
        "manifest": MANIFEST_PATH,
    }

    reports = {}

    for name, path in paths.items():
        with path.open("r", encoding="utf-8") as file:
            reports[name] = json.load(file)

    return reports


def missing_values_chart(frame: pd.DataFrame) -> go.Figure:
    selected = frame.sort_values("Missing count").tail(15)

    fig = go.Figure(
        go.Bar(
            x=selected["Missing (%)"],
            y=selected["Column"],
            orientation="h",
            marker_color=COLORS["amber"],
            customdata=selected["Missing count"],
            hovertemplate=(
                "%{y}<br>Missing: %{customdata:,}<br>"
                "Missing rate: %{x:.2f}%<extra></extra>"
            ),
        )
    )

    fig.update_xaxes(title_text="Missing values (%)")

    return style_chart(
        fig,
        "Top 15 Columns by Missing-Value Rate",
        height=560,
    )


def pipeline_chart(bronze_rows: int, silver_rows: int) -> go.Figure:
    fig = go.Figure(
        go.Bar(
            x=["Bronze", "Silver"],
            y=[bronze_rows, silver_rows],
            marker_color=[COLORS["blue"], COLORS["teal"]],
            text=[f"{bronze_rows:,}", f"{silver_rows:,}"],
            textposition="outside",
            hovertemplate="%{x}: %{y:,} applications<extra></extra>",
        )
    )

    fig.update_yaxes(
        title_text="Application count",
        range=[0, max(bronze_rows, silver_rows) * 1.15],
    )

    return style_chart(
        fig,
        "Bronze-to-Silver Population Reconciliation",
        height=380,
    )


st.sidebar.title("FINANCE360 AI")
st.sidebar.caption("Data Quality & Governance")
st.sidebar.divider()
st.sidebar.caption("Historical dataset | Validated pipeline evidence")

st.title("Data Quality & Governance")
st.caption(
    "Explore ingestion lineage, pipeline reconciliation, missing values "
    "and documented data-quality controls."
)

try:
    reports = get_reports()
    bronze = reports["bronze"]
    silver = reports["silver"]
    manifest = reports["manifest"]
except (OSError, ValueError, TypeError, KeyError) as error:
    st.error(f"Unable to load data-quality evidence: {error}")
    st.stop()

bronze_rows = int(bronze["row_count"])
silver_rows = int(silver["silver_rows"])
manifest_rows = int(manifest["row_count"])
original_columns = int(silver["original_columns_retained"])
silver_columns = int(silver["silver_columns"])

if (
    bronze_rows != silver_rows
    or bronze_rows != manifest_rows
    or int(silver["bronze_rows"]) != bronze_rows
):
    st.error("Bronze, Silver and ingestion manifest row counts do not reconcile.")
    st.stop()

if (
    int(bronze["column_count"]) != original_columns
    or original_columns != int(manifest["curated_column_count"])
    or silver_columns < original_columns
):
    st.error("Curated column counts do not reconcile.")
    st.stop()

if (
    int(bronze["employment_sentinel_count"])
    != int(silver["employment_sentinel_count"])
):
    st.error("Employment sentinel counts do not reconcile.")
    st.stop()

missing = pd.DataFrame(
    [
        {
            "Column": column,
            "Missing count": int(count),
            "Missing (%)": int(count) / silver_rows * 100,
        }
        for column, count in silver["missing_values"].items()
    ]
)

if len(missing) != silver_columns:
    st.error("Silver missing-value profile does not cover all Silver columns.")
    st.stop()

if (missing["Missing count"] > silver_rows).any():
    st.error("Missing-value counts exceed the application population.")
    st.stop()

st.subheader("Pipeline Quality Overview")

metric_a, metric_b, metric_c, metric_d = st.columns(4)

with metric_a:
    st.metric("Bronze applications", f"{bronze_rows:,}")

with metric_b:
    st.metric("Silver applications", f"{silver_rows:,}")

with metric_c:
    st.metric(
        "Duplicate application IDs",
        f"{int(bronze['duplicate_application_ids']):,}",
    )

with metric_d:
    st.metric(
        "Financial quality flags",
        f"{int(silver['financial_quality_flag_count']):,}",
    )

st.caption(
    "Financial quality flags represent the pipeline's defined validation "
    "rules. They are not a universal measure of data completeness."
)

pipeline_tab, missing_tab, controls_tab, lineage_tab = st.tabs(
    [
        "Pipeline Reconciliation",
        "Missing-Value Explorer",
        "Quality Controls",
        "Source Lineage & Governance",
    ]
)

with pipeline_tab:
    st.subheader("Bronze-to-Silver Reconciliation")

    st.plotly_chart(
        pipeline_chart(bronze_rows, silver_rows),
        width="stretch",
    )

    left, right, third = st.columns(3)

    with left:
        st.metric("Curated source columns", original_columns)

    with right:
        st.metric("Silver columns", silver_columns)

    with third:
        st.metric(
            "Engineered columns",
            silver_columns - original_columns,
        )

    st.success(
        "All source applications were retained during the "
        "Bronze-to-Silver transformation."
    )

    st.caption(
        "Source evidence: bronze_profile.json, "
        "silver_quality_report.json and ingestion_manifest.json."
    )

with missing_tab:
    st.subheader("Silver Dataset Completeness")

    st.plotly_chart(
        missing_values_chart(missing),
        width="stretch",
    )

    show_missing_only = st.toggle(
        "Show only columns with missing values",
        value=True,
    )

    filtered = missing.copy()

    if show_missing_only:
        filtered = filtered[filtered["Missing count"] > 0]

    filtered = filtered.sort_values(
        "Missing count",
        ascending=False,
    )

    st.dataframe(
        filtered,
        width="stretch",
        hide_index=True,
    )

    st.download_button(
        "Download missing-value profile (CSV)",
        data=filtered.to_csv(index=False).encode("utf-8"),
        file_name="finance360_silver_missing_values.csv",
        mime="text/csv",
    )

    st.info(
        "Some missing values are expected. For example, the employment "
        "sentinel is converted into missing employment years, while "
        "anonymized external scores are not available for every applicant."
    )

with controls_tab:
    st.subheader("Validated Quality Controls")

    sentinel_count = int(silver["employment_sentinel_count"])
    sentinel_share = sentinel_count / silver_rows * 100
    target_count = int(silver["target_positive_count"])
    target_rate = float(silver["target_positive_rate"]) * 100

    left, right = st.columns(2)

    with left:
        st.metric(
            "Employment sentinel records",
            f"{sentinel_count:,}",
        )
        st.caption(
            f"{sentinel_share:.2f}% of applications contained "
            "the original employment sentinel."
        )

    with right:
        st.metric(
            "Recorded repayment difficulty",
            f"{target_count:,}",
        )
        st.caption(
            f"{target_rate:.2f}% of applications have TARGET = 1."
        )

    controls = pd.DataFrame(
        [
            {
                "Control": "Bronze-to-Silver row preservation",
                "Observed value": f"{silver_rows:,} / {bronze_rows:,}",
                "Result": "Reconciled",
            },
            {
                "Control": "Duplicate application identifiers",
                "Observed value": str(
                    int(bronze["duplicate_application_ids"])
                ),
                "Result": (
                    "No duplicates observed"
                    if int(bronze["duplicate_application_ids"]) == 0
                    else "Review required"
                ),
            },
            {
                "Control": "Financial quality flags",
                "Observed value": str(
                    int(silver["financial_quality_flag_count"])
                ),
                "Result": "Pipeline-defined flags",
            },
            {
                "Control": "Employment sentinel tracking",
                "Observed value": f"{sentinel_count:,}",
                "Result": "Recorded",
            },
            {
                "Control": "Original source columns retained",
                "Observed value": str(original_columns),
                "Result": "Reconciled",
            },
        ]
    )

    st.dataframe(
        controls,
        width="stretch",
        hide_index=True,
    )

    st.warning(
        "Zero duplicate IDs and zero financial quality flags do not "
        "establish that the dataset is free of bias, missing values "
        "or unsuitable features for real lending decisions."
    )

with lineage_tab:
    st.subheader("Dataset Provenance")

    st.markdown(
        f"**Dataset:** {manifest['dataset']}"
    )

    st.markdown(
        f"**Original source file:** `{manifest['source_file']}`"
    )

    st.markdown(
        f"**Bronze artifact:** `{manifest['bronze_file']}`"
    )

    st.markdown(
        f"**Original source columns:** "
        f"{int(manifest['source_column_count'])}"
    )

    st.markdown(
        f"**Curated source columns:** "
        f"{int(manifest['curated_column_count'])}"
    )

    st.markdown("**Source SHA-256 fingerprint:**")

    st.code(manifest["source_sha256"])

    st.markdown(
        "**Pipeline lineage:** Original CSV → curated Bronze Parquet "
        "→ validated Silver Parquet → PostgreSQL warehouse "
        "→ modeling and analytical reports → Streamlit application."
    )

    with st.expander("View curated source columns"):
        st.dataframe(
            pd.DataFrame(
                {"Source column": manifest["selected_columns"]}
            ),
            width="stretch",
            hide_index=True,
        )

    st.markdown("**Governance limitations**")

    st.warning(
        "The source is historical and anonymized. Application event dates, "
        "production scoring history, verified currency units and "
        "regulatory decision approvals are not available. "
        "TARGET denotes repayment difficulty, not necessarily formal default."
    )

    st.caption(
        "Evidence sources: data/bronze/ingestion_manifest.json, "
        "reports/data_quality/bronze_profile.json and "
        "reports/data_quality/silver_quality_report.json."
    )

st.divider()
st.caption(
    "Finance360 AI | Historical data governance | "
    "Educational and analytical use only."
)