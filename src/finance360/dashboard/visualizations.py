import pandas as pd
import plotly.graph_objects as go

COLORS = {
    "blue": "#7398E8",
    "teal": "#42C6A4",
    "green": "#58C9A8",
    "amber": "#DDBB79",
    "orange": "#D99A72",
    "red": "#DF8290",
    "purple": "#AA99DE",
    "muted": "#9BAEC5",
    "text": "#EAF1FB",
    "background": "#101C2D",
}

RISK_COLORS = {
    "Low": "#42C6A4",
    "Moderate": "#7398E8",
    "High": "#DDBB79",
    "Very High": "#DF8290",
}

def style_chart(fig: go.Figure, title: str, height: int = 380) -> go.Figure:
    is_horizontal = any(
        isinstance(trace, go.Bar) and trace.orientation == "h"
        for trace in fig.data
    )

    category_count = max(
        (
            len(trace.y)
            for trace in fig.data
            if isinstance(trace, go.Bar)
            and trace.orientation == "h"
            and trace.y is not None
        ),
        default=0,
    )

    longest_label = max(
        (
            len(str(value))
            for trace in fig.data
            if isinstance(trace, go.Bar)
            and trace.orientation == "h"
            and trace.y is not None
            for value in trace.y
        ),
        default=0,
    )

    calculated_height = height

    if is_horizontal:
        calculated_height = max(height, 150 + category_count * 42)

    left_margin = min(240, max(70, longest_label * 7 + 24)) if is_horizontal else 75

    fig.update_layout(
        title={
            "text": title,
            "x": 0.035,
            "xanchor": "left",
            "y": 0.96,
            "font": {
                "size": 17,
                "color": "#F4F7FC",
                "family": "Arial, sans-serif",
            },
        },
        height=calculated_height,
        autosize=True,
        paper_bgcolor="#101C2D",
        plot_bgcolor="#101C2D",
        font={
            "color": "#C5D2E3",
            "family": "Arial, sans-serif",
            "size": 12,
        },
        margin={
            "l": left_margin,
            "r": 85,
            "t": 90,
            "b": 90,
            "autoexpand": True,
        },
        uniformtext={
            "minsize": 10,
            "mode": "hide",
        },
        legend={
            "orientation": "h",
            "x": 0,
            "xanchor": "left",
            "y": -0.15,
            "yanchor": "top",
            "font": {"size": 11, "color": "#B8C8DC"},
            "bgcolor": "rgba(0,0,0,0)",
        },
        hoverlabel={
            "bgcolor": "#17283D",
            "bordercolor": "#3C5875",
            "font": {
                "color": "#F8FAFC",
                "size": 12,
                "family": "Arial, sans-serif",
            },
        },
        bargap=0.30,
        hovermode="closest",
        separators=".,",
    )

    fig.update_xaxes(
        showgrid=True,
        gridcolor="rgba(148,163,184,0.11)",
        gridwidth=1,
        zeroline=False,
        automargin=True,
        tickfont={"size": 11, "color": "#AFC0D4"},
        title_font={"size": 12, "color": "#AFC0D4"},
    )

    fig.update_yaxes(
        showgrid=False,
        zeroline=False,
        automargin=True,
        tickfont={"size": 11, "color": "#AFC0D4"},
        title_font={"size": 12, "color": "#AFC0D4"},
    )

    for trace in fig.data:
        if isinstance(trace, go.Bar):
            trace.update(
                cliponaxis=False,
                textfont={"size": 11, "color": "#EAF1FB"},
            )

    return fig

def portfolio_outcome_chart(summary: pd.DataFrame) -> go.Figure:
    row = summary.iloc[0]
    total = int(row["application_count"])
    difficulty = int(row["repayment_difficulty_count"])
    without_difficulty = total - difficulty

    if total <= 0 or not 0 <= difficulty <= total:
        raise ValueError("Invalid portfolio totals")

    fig = go.Figure(
        go.Pie(
            labels=[
                "No recorded repayment difficulty",
                "Recorded repayment difficulty",
            ],
            values=[without_difficulty, difficulty],
            hole=0.68,
            marker={
                "colors": [COLORS["teal"], COLORS["red"]],
                "line": {"color": COLORS["background"], "width": 3},
            },
            textinfo="percent",
            hovertemplate="%{label}<br>%{value:,} applications<br>%{percent}<extra></extra>",
        )
    )
    fig.add_annotation(
        text=f"{total:,}<br><span style='font-size:12px'>Applications</span>",
        x=0.5,
        y=0.5,
        showarrow=False,
        font={"size": 23, "color": COLORS["text"]},
    )
    return style_chart(fig, "Portfolio Outcome Composition")


def contract_risk_chart(contracts: pd.DataFrame) -> go.Figure:
    required = {
        "contract_type_name",
        "application_count",
        "repayment_difficulty_rate_pct",
    }
    if not required.issubset(contracts.columns):
        raise ValueError("Missing contract analytics columns")

    frame = contracts.sort_values("repayment_difficulty_rate_pct")
    fig = go.Figure(
        go.Bar(
            x=frame["repayment_difficulty_rate_pct"].astype(float),
            y=frame["contract_type_name"],
            orientation="h",
            marker_color=COLORS["blue"],
            customdata=frame[["application_count"]].to_numpy(),
            text=frame["repayment_difficulty_rate_pct"].astype(float).map(
                lambda value: f"{value:.2f}%"
            ),
            textposition="outside",
            hovertemplate=(
                "%{y}<br>Observed difficulty rate: %{x:.2f}%"
                "<br>Applications: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Observed repayment-difficulty rate (%)")
    return style_chart(fig, "Repayment Difficulty by Loan Type")


def income_risk_chart(income_groups: pd.DataFrame) -> go.Figure:
    required = {
        "income_type_name",
        "application_count",
        "repayment_difficulty_rate_pct",
    }
    if not required.issubset(income_groups.columns):
        raise ValueError("Missing income analytics columns")

    frame = income_groups.copy()
    frame["repayment_difficulty_rate_pct"] = frame[
        "repayment_difficulty_rate_pct"
    ].astype(float)
    frame = frame.sort_values("repayment_difficulty_rate_pct")

    colors = [
        COLORS["red"] if count < 100 else COLORS["purple"]
        for count in frame["application_count"]
    ]

    fig = go.Figure(
        go.Bar(
            x=frame["repayment_difficulty_rate_pct"],
            y=frame["income_type_name"],
            orientation="h",
            marker_color=colors,
            customdata=frame[["application_count"]].to_numpy(),
            hovertemplate=(
                "%{y}<br>Observed difficulty rate: %{x:.2f}%"
                "<br>Sample size: %{customdata[0]:,}<extra></extra>"
            ),
        )
    )
    fig.update_xaxes(title_text="Observed repayment-difficulty rate (%)")
    return style_chart(fig, "Income Source Risk Comparison", height=440)


def risk_distribution_chart(risk_report: dict) -> go.Figure:
    segments = risk_report["segments"]
    if isinstance(segments, dict):
        records = list(segments.values())
    else:
        records = segments

    frame = pd.DataFrame(records)
    required = {"risk_band", "application_count"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing risk segmentation columns")

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    if frame["application_count"].isna().any():
        raise ValueError("Missing risk band")

    fig = go.Figure(
        go.Bar(
            x=order,
            y=frame["application_count"],
            marker_color=[RISK_COLORS[band] for band in order],
            text=frame["application_count"].map(lambda value: f"{int(value):,}"),
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} applications<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Training applications")
    return style_chart(fig, "Training Population by Risk Band")


def exposure_chart(exposure_report: dict) -> go.Figure:
    segments = exposure_report["segments"]
    if isinstance(segments, dict):
        records = list(segments.values())
    else:
        records = segments

    frame = pd.DataFrame(records)
    required = {"risk_band", "total_requested_credit"}
    if not required.issubset(frame.columns):
        raise ValueError("Missing credit exposure columns")

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    if frame["total_requested_credit"].isna().any():
        raise ValueError("Missing credit exposure band")

    credit = frame["total_requested_credit"].astype(float) / 1_000_000_000

    fig = go.Figure(
        go.Bar(
            x=order,
            y=credit,
            marker_color=[RISK_COLORS[band] for band in order],
            text=[f"{value:.2f}B" for value in credit],
            textposition="outside",
            hovertemplate=(
                "%{x}<br>Requested credit: %{y:.2f} billion "
                "source units<extra></extra>"
            ),
        )
    )
    fig.update_yaxes(title_text="Requested credit (billions of source units)")
    return style_chart(fig, "Requested Credit Exposure by Risk Band")


def model_evidence_chart(evaluation: dict) -> go.Figure:
    metrics = [
        ("ROC-AUC", float(evaluation["roc_auc"])),
        ("Average precision", float(evaluation["average_precision"])),
        ("Brier score", float(evaluation["brier_score"])),
    ]

    fig = go.Figure(
        go.Bar(
            x=[name for name, _ in metrics],
            y=[value for _, value in metrics],
            marker_color=[
                COLORS["blue"],
                COLORS["teal"],
                COLORS["amber"],
            ],
            text=[f"{value:.4f}" for _, value in metrics],
            textposition="outside",
            hovertemplate="%{x}<br>Value: %{y:.4f}<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Metric value", range=[0, 1])
    return style_chart(fig, "Held-Out Model Evaluation")


def contract_volume_chart(contracts: pd.DataFrame) -> go.Figure:
    frame = contracts.sort_values("application_count", ascending=False)

    fig = go.Figure(
        go.Bar(
            x=frame["contract_type_name"],
            y=frame["application_count"],
            marker_color=[COLORS["blue"], COLORS["teal"]],
            text=frame["application_count"].map(lambda value: f"{int(value):,}"),
            textposition="outside",
            hovertemplate="%{x}<br>%{y:,} applications<extra></extra>",
        )
    )
    fig.update_yaxes(title_text="Number of applications")
    return style_chart(fig, "Application Volume by Loan Type")


def income_volume_chart(income_groups: pd.DataFrame) -> go.Figure:
    frame = income_groups.sort_values("application_count")

    fig = go.Figure(
        go.Bar(
            x=frame["application_count"],
            y=frame["income_type_name"],
            orientation="h",
            marker_color=COLORS["purple"],
            text=frame["application_count"].map(lambda value: f"{int(value):,}"),
            textposition="outside",
            hovertemplate="%{y}<br>%{x:,} applications<extra></extra>",
        )
    )
    fig.update_xaxes(title_text="Number of applications")
    return style_chart(fig, "Application Volume by Income Source", height=440)


def observed_vs_predicted_chart(risk_report: dict) -> go.Figure:
    frame = pd.DataFrame(risk_report["segments"])

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    observed = frame["observed_difficulty_rate"].astype(float) * 100
    predicted = frame["mean_predicted_probability"].astype(float) * 100

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Observed difficulty rate",
            x=order,
            y=observed,
            marker_color=COLORS["teal"],
            hovertemplate="%{x}<br>Observed: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Average predicted probability",
            x=order,
            y=predicted,
            marker_color=COLORS["purple"],
            hovertemplate="%{x}<br>Predicted: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Repayment-difficulty rate (%)")

    return style_chart(fig, "Observed vs Predicted Risk by Segment")


def credit_share_chart(exposure_report: dict) -> go.Figure:
    frame = pd.DataFrame(exposure_report["segments"])

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    fig = go.Figure(
        go.Pie(
            labels=order,
            values=frame["total_requested_credit"].astype(float),
            hole=0.58,
            marker={
                "colors": [RISK_COLORS[band] for band in order],
                "line": {"color": COLORS["background"], "width": 3},
            },
            textinfo="label+percent",
            hovertemplate=(
                "%{label}<br>Requested credit: %{value:,.0f} source units"
                "<br>Share: %{percent}<extra></extra>"
            ),
        )
    )

    return style_chart(fig, "Requested Credit Share by Risk Band")


def confusion_matrix_chart(evaluation: dict) -> go.Figure:
    matrix = evaluation["confusion_matrix"]

    tn = int(matrix["true_negative"])
    fp = int(matrix["false_positive"])
    fn = int(matrix["false_negative"])
    tp = int(matrix["true_positive"])

    if min(tn, fp, fn, tp) < 0:
        raise ValueError("Confusion matrix counts cannot be negative")

    if tn + fp + fn + tp != int(evaluation["test_rows"]):
        raise ValueError("Confusion matrix does not match test population")

    values = [[tn, fp], [fn, tp]]

    fig = go.Figure(
        go.Heatmap(
            z=values,
            x=["Predicted no difficulty", "Predicted difficulty"],
            y=["Actual no difficulty", "Actual difficulty"],
            colorscale=[
                [0, "#12233A"],
                [0.5, "#2563EB"],
                [1, "#38BDF8"],
            ],
            showscale=False,
            text=[[f"{value:,}" for value in row] for row in values],
            texttemplate="%{text}",
            textfont={"size": 19},
            hovertemplate="%{y}<br>%{x}<br>Cases: %{z:,}<extra></extra>",
        )
    )

    fig.update_yaxes(autorange="reversed")
    return style_chart(fig, "Held-Out Confusion Matrix at 50% Threshold", height=440)


def applicant_vs_credit_share_chart(exposure_report: dict) -> go.Figure:
    frame = pd.DataFrame(exposure_report["segments"])

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    applicant_share = frame["application_share"].astype(float) * 100
    credit_share = frame["requested_credit_share"].astype(float) * 100

    if abs(applicant_share.sum() - 100) > 0.01:
        raise ValueError("Applicant shares do not reconcile")

    if abs(credit_share.sum() - 100) > 0.01:
        raise ValueError("Credit shares do not reconcile")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Applicant share",
            x=order,
            y=applicant_share,
            marker_color=COLORS["blue"],
            hovertemplate="%{x}<br>Applicants: %{y:.2f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Requested credit share",
            x=order,
            y=credit_share,
            marker_color=COLORS["amber"],
            hovertemplate="%{x}<br>Requested credit: %{y:.2f}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    fig.update_yaxes(title_text="Share of training population (%)")

    return style_chart(fig, "Applicant Share vs Requested Credit Share")


def risk_score_range_chart(risk_report: dict) -> go.Figure:
    frame = pd.DataFrame(risk_report["segments"])

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    minimum = frame["minimum_predicted_probability"].astype(float) * 100
    maximum = frame["maximum_predicted_probability"].astype(float) * 100

    if (minimum > maximum).any():
        raise ValueError("Risk score minimum exceeds maximum")

    fig = go.Figure()

    for band in order:
        fig.add_trace(
            go.Scatter(
                x=[float(minimum[band]), float(maximum[band])],
                y=[band, band],
                mode="lines+markers",
                name=band,
                line={"color": RISK_COLORS[band], "width": 8},
                marker={"size": 12},
                hovertemplate=(
                    f"{band}<br>Score: %{{x:.2f}}%<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.update_xaxes(title_text="Predicted repayment-difficulty probability (%)")
    return style_chart(fig, "Observed Model Score Range by Risk Band")


def score_distribution_chart(monitoring_report: dict) -> go.Figure:
    psi_result = monitoring_report["psi_result"]
    bins = pd.DataFrame(psi_result["bins"])

    required = {
        "lower_bound",
        "upper_bound",
        "reference_share",
        "comparison_share",
    }

    if not required.issubset(bins.columns):
        raise ValueError("Missing score-distribution bin columns")

    if abs(bins["reference_share"].sum() - 1) > 0.001:
        raise ValueError("Reference distribution does not reconcile")

    if abs(bins["comparison_share"].sum() - 1) > 0.001:
        raise ValueError("Comparison distribution does not reconcile")

    labels = [
        f"{lower:.0%}-{upper:.0%}"
        for lower, upper in zip(
            bins["lower_bound"],
            bins["upper_bound"],
            strict=True,
        )
    ]

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="Historical reference half",
            x=labels,
            y=bins["reference_share"].astype(float) * 100,
            marker_color=COLORS["blue"],
            hovertemplate="%{x}<br>Reference: %{y:.3f}%<extra></extra>",
        )
    )

    fig.add_trace(
        go.Bar(
            name="Historical comparison half",
            x=labels,
            y=bins["comparison_share"].astype(float) * 100,
            marker_color=COLORS["teal"],
            hovertemplate="%{x}<br>Comparison: %{y:.3f}%<extra></extra>",
        )
    )

    fig.update_layout(barmode="group")
    fig.update_xaxes(title_text="Predicted probability bin")
    fig.update_yaxes(title_text="Share of group (%)")

    return style_chart(fig, "Historical Score Distribution Comparison", height=440)


def held_out_calibration_chart(evaluation: dict) -> go.Figure:
    calibration = evaluation["calibration"]
    bins = pd.DataFrame(calibration["bins"])

    required = {
        "mean_predicted_probability",
        "observed_positive_rate",
    }

    if not required.issubset(bins.columns):
        raise ValueError("Missing calibration bin columns")

    frame = bins.dropna(
        subset=["mean_predicted_probability", "observed_positive_rate"]
    ).copy()

    if frame.empty:
        raise ValueError("No populated calibration bins")

    predicted = frame["mean_predicted_probability"].astype(float) * 100
    observed = frame["observed_positive_rate"].astype(float) * 100

    if (
        predicted.lt(0).any()
        or predicted.gt(100).any()
        or observed.lt(0).any()
        or observed.gt(100).any()
    ):
        raise ValueError("Calibration probabilities must be between 0 and 1")

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=[0, 100],
            y=[0, 100],
            mode="lines",
            name="Perfect calibration",
            line={"color": COLORS["muted"], "dash": "dash", "width": 2},
            hoverinfo="skip",
        )
    )

    fig.add_trace(
        go.Scatter(
            x=predicted,
            y=observed,
            mode="lines+markers",
            name="Held-out calibration",
            line={"color": COLORS["teal"], "width": 3},
            marker={"size": 10},
            hovertemplate=(
                "Mean predicted: %{x:.2f}%"
                "<br>Observed difficulty: %{y:.2f}%"
                "<extra></extra>"
            ),
        )
    )

    upper = max(float(predicted.max()), float(observed.max()), 1.0)
    axis_limit = min(100.0, upper * 1.15)

    fig.update_xaxes(
        title_text="Mean predicted probability (%)",
        range=[0, axis_limit],
    )
    fig.update_yaxes(
        title_text="Observed repayment-difficulty rate (%)",
        range=[0, axis_limit],
        scaleanchor="x",
        scaleratio=1,
    )

    return style_chart(fig, "Held-Out Probability Calibration", height=440)


def risk_outcome_composition_chart(risk_report: dict) -> go.Figure:
    frame = pd.DataFrame(risk_report["segments"])

    order = ["Low", "Moderate", "High", "Very High"]
    frame = frame.set_index("risk_band").reindex(order)

    applications = frame["application_count"].astype(int)
    difficulty = frame["positive_count"].astype(int)
    no_difficulty = applications - difficulty

    if (
        applications.le(0).any()
        or difficulty.lt(0).any()
        or no_difficulty.lt(0).any()
    ):
        raise ValueError("Invalid risk-band outcome counts")

    if int(applications.sum()) != int(risk_report["training_rows"]):
        raise ValueError("Risk-band counts do not match training population")

    fig = go.Figure()

    fig.add_trace(
        go.Bar(
            name="No recorded difficulty",
            x=order,
            y=no_difficulty,
            marker_color=COLORS["teal"],
            hovertemplate=(
                "%{x}<br>No recorded difficulty: %{y:,}"
                "<extra></extra>"
            ),
        )
    )

    fig.add_trace(
        go.Bar(
            name="Recorded difficulty",
            x=order,
            y=difficulty,
            marker_color=COLORS["red"],
            hovertemplate=(
                "%{x}<br>Recorded difficulty: %{y:,}"
                "<extra></extra>"
            ),
        )
    )

    fig.update_layout(barmode="stack")
    fig.update_yaxes(title_text="Training applications")

    return style_chart(fig, "Historical Outcomes by Risk Band")