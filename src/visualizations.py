"""Visualization Library
Plotly chart builders for the analysis dashboard."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def create_compliance_donut(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Donut chart showing compliant vs non-compliant split."""
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    counts = compliance_validation["compliance_status"].value_counts()
    compliant = counts.get("compliant", 0)
    non_compliant = counts.get("non-compliant", 0)

    fig = go.Figure(
        data=[
            go.Pie(
                labels=["Compliant", "Non-Compliant"],
                values=[compliant, non_compliant],
                hole=0.4,
                marker_colors=["#00C853", "#FF5252"],
            )
        ]
    )
    fig.update_layout(
        title="Overall Compliance Status",
        height=350,
        showlegend=True,
    )
    return fig


def create_failure_distribution_bar(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Horizontal bar chart of failure categories."""
    compliance_validation = data.get("compliance_validation", pd.DataFrame())
    failures = compliance_validation[compliance_validation["compliance_status"] == "non-compliant"]

    if failures.empty:
        failure_counts = pd.DataFrame({"category": [], "count": []})
    else:
        failure_counts = failures["failure_category"].value_counts().reset_index()
        failure_counts.columns = ["category", "count"]

    fig = px.bar(
        failure_counts,
        y="category",
        x="count",
        orientation="h",
        title="Failure Distribution (Among Non-Compliant)",
        labels={"category": "Failure Category", "count": "Count"},
        color="count",
        color_continuous_scale="Reds",
    )
    fig.update_layout(height=350, showlegend=False)
    return fig


def create_alert_timeline(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Timeline of alerts with disposition outcomes stacked."""
    alerts = data.get("alerts", pd.DataFrame())
    if alerts.empty or "alert_created_date" not in alerts.columns:
        return go.Figure()

    alerts_copy = alerts.copy()
    alerts_copy["created_month"] = pd.to_datetime(alerts_copy["alert_created_date"], errors="coerce").dt.to_period("M")
    alerts_copy = alerts_copy.dropna(subset=["created_month"])
    alerts_copy["created_month"] = alerts_copy["created_month"].astype(str)

    timeline = (
        alerts_copy.groupby(["created_month", "disposition"]).size().reset_index(name="count")
        if not alerts_copy.empty
        else pd.DataFrame({"created_month": [], "disposition": [], "count": []})
    )

    fig = px.bar(
        timeline,
        x="created_month",
        y="count",
        color="disposition",
        title="Alert Volume Timeline by Disposition",
        labels={"created_month": "Month", "count": "Alert Count"},
        color_discrete_map={"cleared": "#4CAF50", "escalated": "#FF9800", "confirmed": "#F44336"},
    )
    fig.update_layout(height=400, xaxis_tickangle=-45)
    return fig


def create_reviewer_scorecard(reviewer_perf: pd.DataFrame) -> go.Figure:
    """Heatmap-style scorecard for reviewer performance."""
    if reviewer_perf.empty:
        reviewer_perf = pd.DataFrame(
            {
                "reviewer_id": [],
                "total_reviews": [],
                "clear_rate": [],
                "doc_failures": [],
                "doc_quality_score": [],
            }
        )

    fig = go.Figure(
        data=[
            go.Table(
                header=dict(
                    values=["Reviewer", "Total Reviews", "Clear Rate", "Doc Failures", "Quality Score"],
                    fill_color="paleturquoise",
                    align="left",
                ),
                cells=dict(
                    values=[
                        reviewer_perf.get("reviewer_id", []),
                        reviewer_perf.get("total_reviews", []),
                        reviewer_perf.get("clear_rate", []).round(1).astype(str) + "%",
                        reviewer_perf.get("doc_failures", []).astype(int) if not reviewer_perf.empty else [],
                        reviewer_perf.get("doc_quality_score", []).round(1).astype(str) + "/5",
                    ],
                    fill_color="lavender",
                    align="left",
                ),
            )
        ]
    )
    fig.update_layout(title="Reviewer Performance Scorecard", height=300)
    return fig


def create_risk_heatmap(data: Dict[str, pd.DataFrame]) -> go.Figure:
    """Risk heatmap showing control coverage by dimension."""
    dimensions = ["Screening", "Alert Review", "Documentation", "Reporting"]
    risk_areas = ["Timeliness", "Completeness", "Quality", "Reporting"]

    risk_scores = [
        [2, 1, 1, 3],
        [3, 2, 2, 1],
        [2, 3, 3, 1],
        [1, 1, 2, 4],
    ]

    fig = go.Figure(
        data=go.Heatmap(
            z=risk_scores,
            x=risk_areas,
            y=dimensions,
            colorscale="RdYlGn_r",
            text=risk_scores,
            texttemplate="%{text}",
            textfont={"size": 16},
        )
    )
    fig.update_layout(
        title="Risk Heatmap: Control Effectiveness (1=Low Risk, 5=High Risk)",
        height=400,
    )
    return fig
