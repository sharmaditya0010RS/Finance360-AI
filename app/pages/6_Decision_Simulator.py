"""Interactive hypothetical credit-risk scenario comparison."""

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from finance360.dashboard.simulator import compare_scenarios
from finance360.dashboard.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(page_title="Decision Simulator | Finance360 AI", page_icon="📊", layout="wide")
apply_theme()
st.title("Decision Simulator")
st.caption("Compare baseline and hypothetical model-estimated repayment difficulty.")
st.warning("Educational decision support only. Scenario differences are not causal effects or loan approval recommendations.")

with st.form("scenario_comparison"):
    st.subheader("Applicant profile")
    c1, c2, c3 = st.columns(3)
    with c1:
        contract = st.selectbox("Loan type", ["Cash loans", "Revolving loans"])
        income_type = st.selectbox("Income source", ["Working", "Commercial associate", "Pensioner", "State servant", "Unemployed", "Student", "Businessman", "Maternity leave"])
        education = st.selectbox("Education", ["Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"])
        family = st.selectbox("Family status", ["Married", "Single / not married", "Civil marriage", "Separated", "Widow", "Unknown"])
        housing = st.selectbox("Housing arrangement", ["House / apartment", "With parents", "Municipal apartment", "Rented apartment", "Office apartment", "Co-op apartment"])
    with c2:
        car = st.selectbox("Owns a car?", ["N", "Y"])
        realty = st.selectbox("Owns real estate?", ["Y", "N"])
        goods = st.number_input("Goods price (source units)", min_value=1.0, value=450000.0)
        children = st.number_input("Children", min_value=0, max_value=20, value=1)
        members = st.number_input("Family members", min_value=1, max_value=30, value=3)
        age = st.number_input("Age (years)", min_value=18, max_value=100, value=35)
        employment = st.number_input("Employment duration (years)", min_value=0.0, max_value=80.0, value=8.0)
    with c3:
        income = st.number_input("Baseline income", min_value=1.0, value=180000.0)
        credit = st.number_input("Baseline requested credit", min_value=1.0, value=450000.0)
        annuity = st.number_input("Baseline annuity", min_value=1.0, value=25000.0)
        st.caption("External scores are anonymized dataset-specific inputs; leave blank unless genuine values are available.")
        ext1 = st.number_input("External score 1", min_value=0.0, max_value=1.0, value=None)
        ext2 = st.number_input("External score 2", min_value=0.0, max_value=1.0, value=None)
        ext3 = st.number_input("External score 3", min_value=0.0, max_value=1.0, value=None)

    st.subheader("Hypothetical scenario")
    s1, s2, s3 = st.columns(3)
    with s1:
        scenario_income = st.number_input("Scenario income", min_value=1.0, value=200000.0)
    with s2:
        scenario_credit = st.number_input("Scenario requested credit", min_value=1.0, value=450000.0)
    with s3:
        scenario_annuity = st.number_input("Scenario annuity", min_value=1.0, value=25000.0)
    submitted = st.form_submit_button("Compare scenarios", type="primary")

if submitted:
    applicant = {
        "NAME_CONTRACT_TYPE": contract, "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education, "NAME_FAMILY_STATUS": family,
        "NAME_HOUSING_TYPE": housing, "FLAG_OWN_CAR": car,
        "FLAG_OWN_REALTY": realty, "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit, "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods, "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": members, "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": employment, "EXT_SOURCE_1": ext1,
        "EXT_SOURCE_2": ext2, "EXT_SOURCE_3": ext3,
    }
    try:
        result = compare_scenarios(
            applicant,
            {"AMT_INCOME_TOTAL": scenario_income, "AMT_CREDIT": scenario_credit, "AMT_ANNUITY": scenario_annuity},
            PROJECT_ROOT,
        )
    except (ValueError, TypeError, OSError, KeyError) as error:
        st.error(f"Unable to compare scenarios: {error}")
        st.stop()

    baseline = result["baseline"]
    scenario = result["scenario"]
    a, b, c = st.columns(3)
    a.metric("Baseline estimated difficulty", f"{baseline['repayment_difficulty_probability']:.2%}")
    b.metric("Scenario estimated difficulty", f"{scenario['repayment_difficulty_probability']:.2%}")
    c.metric("Change (percentage points)", f"{result['percentage_point_delta']:+.2f} pp")
    st.write(f"Relative risk bands: **{baseline['risk_band']}** → **{scenario['risk_band']}**")
    fig = go.Figure(go.Bar(
        x=["Baseline", "Scenario"],
        y=[baseline["repayment_difficulty_probability"] * 100, scenario["repayment_difficulty_probability"] * 100],
        text=[f"{baseline['repayment_difficulty_probability']:.2%}", f"{scenario['repayment_difficulty_probability']:.2%}"],
        textposition="auto",
    ))
    fig.update_layout(title="Model-estimated repayment difficulty", yaxis_title="Probability (%)", yaxis_range=[0, 100])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(pd.DataFrame([
        {"Input": "Reported income", "Baseline": income, "Scenario": scenario_income},
        {"Input": "Requested credit", "Baseline": credit, "Scenario": scenario_credit},
        {"Input": "Annuity", "Baseline": annuity, "Scenario": scenario_annuity},
    ]), hide_index=True, use_container_width=True)
    if baseline.get("missing_external_sources", 0):
        st.warning("Some anonymized external scores were unavailable; interpret estimates cautiously.")
    st.caption(result["interpretation"])
