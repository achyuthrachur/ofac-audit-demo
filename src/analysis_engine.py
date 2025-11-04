"""Analysis Engine
Core business logic for compliance testing."""

from __future__ import annotations

from typing import Dict, List

import numpy as np
import pandas as pd


def _safe_to_datetime(series: pd.Series) -> pd.Series:
    """Convert a pandas series to datetime, coercing errors to NaT."""
    return pd.to_datetime(series, errors="coerce")


def calculate_summary_metrics(data: Dict[str, pd.DataFrame]) -> Dict[str, float]:
    """Calculate high-level KPIs for the executive dashboard."""
    policyholders = data.get("policyholders", pd.DataFrame())
    alerts = data.get("alerts", pd.DataFrame())
    ofac_reporting = data.get("ofac_reporting", pd.DataFrame())
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    total_policies = len(policyholders)
    compliant_count = len(
        compliance_validation[compliance_validation["compliance_status"] == "compliant"]
    )
    compliance_rate = (compliant_count / total_policies * 100) if total_policies else 0.0

    policies_with_alerts = alerts["policy_id"].nunique()
    confirmed_matches = (alerts["disposition"] == "confirmed").sum()
    ofac_reports_filed = len(ofac_reporting)

    late_ofac = len(
        compliance_validation[
            compliance_validation["failure_category"] == "ofac_reporting_delay"
        ]
    )

    return {
        "total_policies": total_policies,
        "compliance_rate": compliance_rate,
        "compliant_count": compliant_count,
        "non_compliant_count": total_policies - compliant_count,
        "policies_with_alerts": policies_with_alerts,
        "total_alerts": len(alerts),
        "confirmed_matches": confirmed_matches,
        "ofac_reports_filed": ofac_reports_filed,
        "late_ofac_reports": late_ofac,
    }


def run_screening_checks(data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Evaluate screening timeliness compliance."""
    policyholders = data.get("policyholders", pd.DataFrame()).copy()
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    screening_failures = compliance_validation[
        compliance_validation["failure_category"] == "screening_timeliness"
    ]

    total = len(policyholders)
    failed = len(screening_failures)
    passed = total - failed
    pass_rate = (passed / total * 100) if total else 0.0

    exceptions: List[Dict[str, object]] = []
    if not policyholders.empty:
        policy_lookup = policyholders.set_index("policy_id")
        for _, row in screening_failures.iterrows():
            policy_id = row["policy_id"]
            detail = policy_lookup.get(policy_id)
            if detail is None:
                continue
            exceptions.append(
                {
                    "policy_id": policy_id,
                    "holder_name": detail.get("holder_name"),
                    "inforce_date": detail.get("inforce_date"),
                    "last_screen_date": detail.get("last_screen_date"),
                    "screen_performed": detail.get("screen_performed_flag"),
                    "failure_reason": row.get("failure_details"),
                }
            )

    exceptions_df = pd.DataFrame(exceptions)

    return {
        "total": total,
        "passed": passed,
        "failed": failed,
        "pass_rate": pass_rate,
        "exceptions": exceptions_df,
    }


def run_alert_review_checks(data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Test alert review timeliness and documentation quality."""
    alerts = data.get("alerts", pd.DataFrame()).copy()
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    review_failures = compliance_validation[
        compliance_validation["failure_category"].isin(
            ["alert_review_timeliness", "documentation_quality"]
        )
    ]

    total_alerts = len(alerts[alerts["reviewed_flag"].str.upper() == "Y"])
    timely_reviews = total_alerts - len(
        compliance_validation[compliance_validation["failure_category"] == "alert_review_timeliness"]
    )

    alerts["created_dt"] = _safe_to_datetime(alerts.get("alert_created_date"))
    alerts["reviewed_dt"] = _safe_to_datetime(alerts.get("reviewed_date"))
    alerts["review_days"] = (alerts["reviewed_dt"] - alerts["created_dt"]).dt.days
    alerts_with_reviews = alerts[alerts["review_days"].notna()]
    avg_review_days = alerts_with_reviews["review_days"].mean() if not alerts_with_reviews.empty else 0.0

    exceptions: List[Dict[str, object]] = []
    alert_lookup = alerts.set_index("policy_id") if not alerts.empty else pd.DataFrame()
    for _, row in review_failures.iterrows():
        policy_id = row["policy_id"]
        if policy_id in alert_lookup.index:
            alert_detail = alert_lookup.loc[policy_id]
            if isinstance(alert_detail, pd.DataFrame):
                alert_detail = alert_detail.iloc[0]
            exceptions.append(
                {
                    "policy_id": policy_id,
                    "alert_id": alert_detail.get("alert_id"),
                    "alert_created": alert_detail.get("alert_created_date"),
                    "reviewed_date": alert_detail.get("reviewed_date"),
                    "failure_category": row.get("failure_category"),
                    "failure_details": row.get("failure_details"),
                }
            )

    exceptions_df = pd.DataFrame(exceptions)

    return {
        "total": total_alerts,
        "timely_count": max(timely_reviews, 0),
        "avg_review_days": float(avg_review_days) if not np.isnan(avg_review_days) else 0.0,
        "avg_doc_quality": 3.5,  # Placeholder for future LLM integration
        "exceptions": exceptions_df,
    }


def run_ofac_reporting_checks(data: Dict[str, pd.DataFrame]) -> Dict[str, object]:
    """Validate OFAC reporting timeliness."""
    alerts = data.get("alerts", pd.DataFrame())
    ofac_reporting = data.get("ofac_reporting", pd.DataFrame())
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    confirmed_matches = alerts[alerts["disposition"] == "confirmed"]
    total_required = len(confirmed_matches)

    reporting_failures = compliance_validation[
        compliance_validation["failure_category"] == "ofac_reporting_delay"
    ]

    late = len(reporting_failures)
    on_time = max(len(ofac_reporting) - late, 0)
    on_time_pct = (on_time / total_required * 100) if total_required else 0.0

    exceptions: List[Dict[str, object]] = []
    alert_lookup = alerts.set_index("policy_id") if not alerts.empty else pd.DataFrame()
    report_lookup = ofac_reporting.set_index("policy_id") if not ofac_reporting.empty else pd.DataFrame()

    for _, row in reporting_failures.iterrows():
        policy_id = row["policy_id"]
        if policy_id in alert_lookup.index and policy_id in report_lookup.index:
            alert_detail = alert_lookup.loc[policy_id]
            if isinstance(alert_detail, pd.DataFrame):
                alert_detail = alert_detail.iloc[0]
            report_detail = report_lookup.loc[policy_id]
            if isinstance(report_detail, pd.DataFrame):
                report_detail = report_detail.iloc[0]
            exceptions.append(
                {
                    "policy_id": policy_id,
                    "alert_id": alert_detail.get("alert_id"),
                    "escalation_date": alert_detail.get("escalation_date"),
                    "report_date": report_detail.get("report_date"),
                    "report_reference": report_detail.get("report_reference"),
                    "failure_details": row.get("failure_details"),
                }
            )

    exceptions_df = pd.DataFrame(exceptions)

    return {
        "total_required": total_required,
        "on_time": on_time,
        "late": late,
        "on_time_pct": on_time_pct,
        "exceptions": exceptions_df,
    }


def calculate_reviewer_performance(data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compute reviewer performance metrics."""
    alerts = data.get("alerts", pd.DataFrame()).copy()
    compliance_validation = data.get("compliance_validation", pd.DataFrame())

    if alerts.empty:
        return pd.DataFrame(
            columns=["reviewer_id", "total_reviews", "clear_rate", "doc_failures", "doc_quality_score"]
        )

    alerts["reviewer_id"] = alerts["reviewer_id"].fillna("Unknown")

    reviewer_stats = (
        alerts.groupby("reviewer_id")
        .agg(total_reviews=("alert_id", "count"), cleared_count=("disposition", lambda x: (x == "cleared").sum()))
        .reset_index()
    )

    reviewer_stats["clear_rate"] = np.where(
        reviewer_stats["total_reviews"] > 0,
        reviewer_stats["cleared_count"] / reviewer_stats["total_reviews"] * 100,
        0.0,
    )

    doc_failures = compliance_validation[
        compliance_validation["failure_category"] == "documentation_quality"
    ]

    failed_alert_ids: List[str] = []
    for detail in doc_failures.get("failure_details", []):
        if isinstance(detail, str) and "ALT-" in detail:
            fragment = detail.split("ALT-")[1].split(":")[0]
            failed_alert_ids.append(f"ALT-{fragment}")

    alerts_with_failures = alerts[alerts["alert_id"].isin(failed_alert_ids)]
    failure_counts = alerts_with_failures.groupby("reviewer_id").size().reset_index(name="doc_failures")

    merged = reviewer_stats.merge(failure_counts, on="reviewer_id", how="left")
    merged["doc_failures"] = merged["doc_failures"].fillna(0)
    merged["doc_quality_score"] = np.clip(
        5 - (merged["doc_failures"] / merged["total_reviews"].replace({0: np.nan}) * 5), 0, 5
    ).fillna(5)

    return merged[["reviewer_id", "total_reviews", "clear_rate", "doc_failures", "doc_quality_score"]]
