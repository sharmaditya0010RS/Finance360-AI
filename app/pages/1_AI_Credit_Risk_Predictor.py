from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from finance360.dashboard.prediction import predict_applicant
from finance360.dashboard.theme import apply_theme

PROJECT_ROOT = Path(__file__).resolve().parents[2]

st.set_page_config(
    page_title="AI Credit Risk Predictor | Finance360 AI",
    page_icon="🧠",
    layout="wide",
)

apply_theme()

st.title("AI Credit Risk Predictor")
st.caption(
    "Explore an estimated probability of repayment difficulty "
    "using the trained Finance360 XGBoost model."
)

st.info(
    "This is an educational decision-support tool, not a loan approval "
    "or rejection system. The model was trained on historical application data."
)

with st.form("applicant_form"):
    st.subheader("Applicant Information")

    left, right = st.columns(2)

    with left:
        contract_type = st.selectbox(
            "Loan type",
            ["Cash loans", "Revolving loans"],
        )
        income_type = st.selectbox(
            "Income source",
            [
                "Working",
                "Commercial associate",
                "Pensioner",
                "State servant",
                "Unemployed",
                "Student",
                "Businessman",
                "Maternity leave",
            ],
        )
        education = st.selectbox(
            "Education",
            [
                "Secondary / secondary special",
                "Higher education",
                "Incomplete higher",
                "Lower secondary",
                "Academic degree",
            ],
        )
        family_status = st.selectbox(
            "Family status",
            [
                "Married",
                "Single / not married",
                "Civil marriage",
                "Separated",
                "Widow",
                "Unknown",
            ],
        )
        housing_type = st.selectbox(
            "Housing arrangement",
            [
                "House / apartment",
                "With parents",
                "Municipal apartment",
                "Rented apartment",
                "Office apartment",
                "Co-op apartment",
            ],
        )
        owns_car = st.selectbox("Owns a car?", ["N", "Y"])
        owns_realty = st.selectbox("Owns real estate?", ["Y", "N"])

    with right:
        income = st.number_input(
            "Reported annual income (source units)",
            min_value=1.0,
            value=180000.0,
            step=10000.0,
        )
        credit = st.number_input(
            "Requested credit (source units)",
            min_value=1.0,
            value=450000.0,
            step=10000.0,
        )
        annuity = st.number_input(
            "Reported annuity amount (source units)",
            min_value=1.0,
            value=25000.0,
            step=1000.0,
        )
        goods_price = st.number_input(
            "Goods price (source units)",
            min_value=1.0,
            value=450000.0,
            step=10000.0,
        )
        children = st.number_input(
            "Number of children",
            min_value=0,
            max_value=20,
            value=1,
            step=1,
        )
        family_members = st.number_input(
            "Family members",
            min_value=1,
            max_value=30,
            value=3,
            step=1,
        )
        age = st.number_input(
            "Age (years)",
            min_value=18,
            max_value=100,
            value=35,
            step=1,
        )
        employment_years = st.number_input(
            "Employment duration (years)",
            min_value=0.0,
            max_value=80.0,
            value=8.0,
            step=1.0,
        )

    with st.expander("Optional anonymized external scores"):
        st.caption(
            "These are dataset-specific anonymized scores. "
            "Leave them blank if you do not have genuine values."
        )
        ext_1 = st.number_input(
            "External score 1",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
        )
        ext_2 = st.number_input(
            "External score 2",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
        )
        ext_3 = st.number_input(
            "External score 3",
            min_value=0.0,
            max_value=1.0,
            value=None,
            step=0.01,
        )

    submitted = st.form_submit_button(
        "Estimate Repayment Difficulty",
        type="primary",
        width="stretch",
    )

if submitted:
    applicant = {
        "NAME_CONTRACT_TYPE": contract_type,
        "NAME_INCOME_TYPE": income_type,
        "NAME_EDUCATION_TYPE": education,
        "NAME_FAMILY_STATUS": family_status,
        "NAME_HOUSING_TYPE": housing_type,
        "FLAG_OWN_CAR": owns_car,
        "FLAG_OWN_REALTY": owns_realty,
        "AMT_INCOME_TOTAL": income,
        "AMT_CREDIT": credit,
        "AMT_ANNUITY": annuity,
        "AMT_GOODS_PRICE": goods_price,
        "CNT_CHILDREN": children,
        "CNT_FAM_MEMBERS": family_members,
        "AGE_YEARS": age,
        "EMPLOYMENT_YEARS": employment_years,
        "EXT_SOURCE_1": ext_1,
        "EXT_SOURCE_2": ext_2,
        "EXT_SOURCE_3": ext_3,
    }

    try:
        result = predict_applicant(applicant, PROJECT_ROOT)
    except (ValueError, TypeError, OSError, KeyError) as error:
        st.error(f"Unable to calculate prediction: {error}")
        st.stop()

    probability = result["repayment_difficulty_probability"]
    risk_band = result["risk_band"]

    st.divider()
    st.subheader("AI Risk Assessment")

    left, right = st.columns(2)

    with left:
        st.metric(
            "Estimated repayment-difficulty probability",
            f"{probability:.2%}",
        )
        st.metric("Indicative risk segment", risk_band)

        st.write(
            "This percentage represents the model's estimated probability "
            "of repayment difficulty for the entered application profile. "
            "It is not a guarantee that repayment difficulty will occur."
        )

    with right:
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "valueformat": ".2f"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": "#38bdf8"},
                    "steps": [
                        {"range": [0, 25], "color": "#134e4a"},
                        {"range": [25, 50], "color": "#365314"},
                        {"range": [50, 75], "color": "#78350f"},
                        {"range": [75, 100], "color": "#7f1d1d"},
                    ],
                },
            )
        )
        fig.update_layout(
            height=310,
            margin={"t": 30, "b": 20, "l": 20, "r": 20},
        )
        st.plotly_chart(fig, width="stretch")

    if result["missing_external_sources"] > 0:
        st.warning(
            f"{result['missing_external_sources']} of 3 anonymized external "
            "scores were unavailable. The model used its trained missing-value "
            "handling, so interpret this individual estimate cautiously."
        )

    with st.expander("How was this risk band determined?"):
        boundaries = result["risk_boundaries"]

        st.write(
            "The four bands divide historical training-model scores "
            "using fixed score boundaries. They indicate relative "
            "risk compared with the training population."
        )
        st.write(
            f"Low: below {boundaries[0]:.2%} | "
            f"Moderate: {boundaries[0]:.2%} to {boundaries[1]:.2%} | "
            f"High: {boundaries[1]:.2%} to {boundaries[2]:.2%} | "
            f"Very High: {boundaries[2]:.2%} and above"
        )

    st.caption(result["limitations"])