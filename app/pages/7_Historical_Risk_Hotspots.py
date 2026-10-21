"""Streamlit historical segment hotspot explorer."""
from pathlib import Path

import plotly.express as px
import psycopg
import streamlit as st

from finance360.dashboard.hotspots import DIMENSIONS, analyze_hotspots, load_hotspot_segments
from finance360.dashboard.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]
st.set_page_config(page_title="Historical Risk Hotspots | Finance360 AI", page_icon="📊", layout="wide")
apply_theme()
st.title("Historical Risk Hotspots")
st.caption("Historical segment associations only; not a causal analysis or credit approval tool.")

@st.cache_data(ttl=300, show_spinner="Loading warehouse segments...")
def get_segments():
    return load_hotspot_segments(PROJECT_ROOT)

try:
    raw = get_segments()
except (OSError, ValueError, psycopg.Error) as error:
    st.error(f"Could not load hotspot data: {error}")
    st.stop()

dimension = st.selectbox("Segment dimension", list(DIMENSIONS))
minimum = st.number_input("Minimum applications per segment", min_value=1, max_value=307511, value=1000, step=100)
filtered = raw.loc[raw["dimension"] == dimension]
try:
    result = analyze_hotspots(filtered, int(minimum))
except (ValueError, TypeError) as error:
    st.error(str(error))
    st.stop()

if result.empty:
    st.warning("No segments meet the minimum sample size.")
    st.stop()

portfolio_rate = float(result["portfolio_rate"].iloc[0])
above = result[result["difference_pp"] > 0]
a, b, c = st.columns(3)
a.metric("Historical portfolio difficulty rate", f"{portfolio_rate:.2%}")
b.metric("Eligible segments", f"{len(result):,}")
c.metric("Above-benchmark segments", f"{len(above):,}")

chart_data = result.copy()
chart_data["Difficulty rate (%)"] = chart_data["difficulty_rate"] * 100
fig = px.bar(chart_data, x="segment", y="Difficulty rate (%)", color="difference_pp", color_continuous_scale="RdYlBu_r", hover_data=["applications", "difficulty_cases", "relative_rate"], labels={"segment": dimension, "difference_pp": "Difference (pp)"}, title=f"Observed difficulty rate by {dimension.lower()}")
fig.add_hline(y=portfolio_rate * 100, line_dash="dash", annotation_text="Full portfolio benchmark")
fig.update_layout(height=450, xaxis_tickangle=-25)
st.plotly_chart(fig, width="stretch")

show = result[["segment", "applications", "difficulty_cases", "difficulty_rate", "difference_pp", "relative_rate"]].copy()
show["difficulty_rate"] = (show["difficulty_rate"] * 100).round(2)
show["difference_pp"] = show["difference_pp"].round(2)
show["relative_rate"] = show["relative_rate"].round(2)
show.columns = ["Segment", "Applications", "Difficulty cases", "Difficulty rate (%)", "Difference vs portfolio (pp)", "Relative rate (x)"]
st.dataframe(show, width="stretch", hide_index=True)
st.download_button("Download hotspot comparison (CSV)", show.to_csv(index=False).encode("utf-8"), file_name="finance360_historical_hotspots.csv", mime="text/csv")
st.info("A rate above the historical portfolio average does not imply causation, future outcomes, or an approval/rejection recommendation. Smaller segments can have unstable rates. Counts and labels come from the current warehouse snapshot.")
