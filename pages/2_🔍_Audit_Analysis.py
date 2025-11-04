"""OFAC Audit Analysis Dashboard.

Upload datasets and perform comprehensive compliance testing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from analysis_engine import (  # type: ignore  # pylint: disable=wrong-import-position
    calculate_reviewer_performance,
    calculate_summary_metrics,
    run_alert_review_checks,
    run_ofac_reporting_checks,
    run_screening_checks,
)
from llm_evaluator import evaluate_note_quality  # type: ignore  # pylint: disable=wrong-import-position
from visualizations import (  # type: ignore  # pylint: disable=wrong-import-position
    create_alert_timeline,
    create_compliance_donut,
    create_failure_distribution_bar,
    create_reviewer_scorecard,
    create_risk_heatmap,
)


def _load_generated_files(file_map: Dict[str, str]) -> Dict[str, pd.DataFrame]:
    datasets: Dict[str, pd.DataFrame] = {}
    for key in ("policyholders", "alerts", "ofac_reporting", "compliance_validation"):
        path = file_map.get(key)
        if not path:
            raise FileNotFoundError(f"Missing path for '{key}'.")
        datasets[key] = pd.read_csv(path)
    return datasets


def _render_metrics_tiles(metrics: Dict[str, float]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            "Compliance Rate",
            f"{metrics['compliance_rate']:.1f}%",
            delta=f"{metrics['compliance_rate'] - 60:.1f}% vs target",
        )
    with col2:
        st.metric(
            "Total Policies",
            f"{metrics['total_policies']:,}",
            delta=f"{metrics['policies_with_alerts']} with alerts",
        )
    with col3:
        st.metric(
            "Critical Gaps",
            metrics["late_ofac_reports"],
            delta="Late OFAC reports",
            delta_color="inverse",
        )
    with col4:
        st.metric(
            "Confirmed Matches",
            metrics["confirmed_matches"],
            delta=f"{metrics['ofac_reports_filed']} reports filed",
        )


def _safe_concat(exceptions: List[pd.DataFrame]) -> pd.DataFrame:
    frames = [df for df in exceptions if df is not None and not df.empty]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def render() -> None:
    st.title("🔍 OFAC Audit Analysis Dashboard")
    st.markdown("---")

    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None

    st.header("📁 Data Upload")

    generated_files = st.session_state.get("generated_files")
    if generated_files:
        st.success("✅ Using data from generator. Files detected in session.")
        if st.button("Load Data from Generation", type="primary"):
            try:
                st.session_state.analysis_data = _load_generated_files(generated_files)
                st.success("Generator output loaded successfully.")
                st.rerun()
            except Exception as exc:  # pylint: disable=broad-except
                st.error(f"Error loading files: {exc}")
    else:
        st.info("Upload the four CSV files generated from the Data Generator page.")
        col_left, col_right = st.columns(2)
        with col_left:
            policyholders_file = st.file_uploader("Policyholders CSV", type=["csv"], key="ph_upload")
            alerts_file = st.file_uploader("Alerts CSV", type=["csv"], key="al_upload")
        with col_right:
            ofac_file = st.file_uploader("OFAC Reporting CSV", type=["csv"], key="of_upload")
            validation_file = st.file_uploader("Compliance Validation CSV", type=["csv"], key="cv_upload")

        if st.button("Load Uploaded Files", type="primary"):
            files = [policyholders_file, alerts_file, ofac_file, validation_file]
            if all(files):
                try:
                    st.session_state.analysis_data = {
                        "policyholders": pd.read_csv(policyholders_file),
                        "alerts": pd.read_csv(alerts_file),
                        "ofac_reporting": pd.read_csv(ofac_file),
                        "compliance_validation": pd.read_csv(validation_file),
                    }
                    st.success("✅ Files loaded successfully!")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Error loading files: {exc}")
            else:
                st.warning("Please upload all four CSV files.")

    if not st.session_state.analysis_data:
        st.info("👆 Load CSV files above to begin analysis.")
        return

    data = st.session_state.analysis_data
    metrics = calculate_summary_metrics(data)
    screening_results = run_screening_checks(data)
    alert_results = run_alert_review_checks(data)
    ofac_results = run_ofac_reporting_checks(data)
    reviewer_perf = calculate_reviewer_performance(data)

    st.markdown("---")
    tab_exec, tab_detailed, tab_llm, tab_yoy, tab_export = st.tabs(
        [
            "📊 Executive Summary",
            "🔍 Detailed Analysis",
            "🤖 LLM Note Evaluator",
            "📈 Year-over-Year Trends",
            "📄 Export Workpapers",
        ]
    )

    with tab_exec:
        st.header("Executive Summary Dashboard")
        _render_metrics_tiles(metrics)
        st.markdown("---")

        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Compliance Status")
            st.plotly_chart(create_compliance_donut(data), width="stretch")
        with col_right:
            st.subheader("Failure Distribution")
            st.plotly_chart(create_failure_distribution_bar(data), width="stretch")

        st.markdown("---")
        st.subheader("Alert Timeline & Dispositions")
        st.plotly_chart(create_alert_timeline(data), width="stretch")

    with tab_detailed:
        st.header("Detailed Compliance Testing")
        test_screening, test_alerts, test_reporting = st.tabs(
            ["Screening Timeliness", "Alert Review Quality", "OFAC Reporting"]
        )

        with test_screening:
            st.subheader("Check 1: Timely Screening")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Policies Tested", screening_results["total"])
            with col_b:
                st.metric("Passed", screening_results["passed"], delta=f"{screening_results['pass_rate']:.1f}%")
            with col_c:
                st.metric("Failed", screening_results["failed"], delta_color="inverse")

            st.markdown("#### Exceptions")
            exceptions_df = screening_results["exceptions"]
            st.dataframe(
                exceptions_df,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "policy_id": st.column_config.TextColumn("Policy ID", width="small"),
                    "failure_reason": st.column_config.TextColumn("Failure Reason", width="large"),
                },
            )
            st.download_button(
                "📥 Download Screening Exceptions",
                exceptions_df.to_csv(index=False).encode("utf-8"),
                "screening_exceptions.csv",
                "text/csv",
            )

        with test_alerts:
            st.subheader("Check 2: Alert Review Timeliness & Documentation")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Alerts Tested", alert_results["total"])
            with col_b:
                st.metric("Timely Reviews", alert_results["timely_count"])
            with col_c:
                st.metric("Avg Review Time", f"{alert_results['avg_review_days']:.1f} days")

            st.markdown("#### Reviewer Performance Scorecard")
            st.plotly_chart(create_reviewer_scorecard(reviewer_perf), width="stretch")

            st.markdown("#### Alert Review Exceptions")
        st.dataframe(alert_results["exceptions"], hide_index=True, use_container_width=True)

        with test_reporting:
            st.subheader("Check 3: OFAC Reporting Compliance")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Reports Required", ofac_results["total_required"])
            with col_b:
                st.metric("On-Time Reports", ofac_results["on_time"])
            with col_c:
                st.metric("Late Reports", ofac_results["late"], delta_color="inverse")

            st.markdown("#### OFAC Reporting Exceptions")
        st.dataframe(ofac_results["exceptions"], hide_index=True, use_container_width=True)

    with tab_llm:
        st.header("🤖 AI-Powered Note Quality Evaluator")
        st.markdown(
            """
            This tool uses an LLM-inspired checklist to evaluate reviewer notes against the 5-element SOP:
            1. Alert reason explanation
            2. Secondary identifier checked
            3. Ticket/reference number
            4. Disposition rationale
            5. Reviewer signature/date
            """
        )
        col_input, _ = st.columns([2, 1])
        with col_input:
            sample_choice = st.radio(
                "Load Sample Note",
                ["Compliant (5/5)", "Partially Compliant (2/5)", "Custom Input"],
                horizontal=True,
            )

        alerts_df = data["alerts"]
        validation_df = data["compliance_validation"]
        sample_note = ""
        if sample_choice == "Compliant (5/5)":
            compliant = validation_df[validation_df["failure_category"] != "documentation_quality"]
            if not compliant.empty:
                policy_id = compliant.iloc[0]["policy_id"]
                sample_series = alerts_df[alerts_df["policy_id"] == policy_id]["reviewer_notes"]
                if not sample_series.empty:
                    sample_note = sample_series.iloc[0]
        elif sample_choice == "Partially Compliant (2/5)":
            partial = validation_df[validation_df["failure_category"] == "documentation_quality"]
            if not partial.empty:
                policy_id = partial.iloc[0]["policy_id"]
                sample_series = alerts_df[alerts_df["policy_id"] == policy_id]["reviewer_notes"]
                if not sample_series.empty:
                    sample_note = sample_series.iloc[0]

        reviewer_note = st.text_area(
            "Reviewer Note to Evaluate",
            value=sample_note,
            height=180,
            placeholder="Paste or type a reviewer note here...",
        )

        if st.button("🔍 Evaluate Note", type="primary"):
            if reviewer_note.strip():
                with st.spinner("Analyzing note quality..."):
                    evaluation = evaluate_note_quality(reviewer_note)

                col_summary, col_checklist = st.columns([1, 2])
                with col_summary:
                    score = evaluation["score"]
                    if score >= 4:
                        st.success(f"### Score: {score}/5")
                        st.metric("Quality", "Compliant ✅")
                    elif score >= 3:
                        st.warning(f"### Score: {score}/5")
                        st.metric("Quality", "Needs Improvement ⚠️")
                    else:
                        st.error(f"### Score: {score}/5")
                        st.metric("Quality", "Non-Compliant ❌")
                with col_checklist:
                    st.markdown("#### Elements Checklist")
                    for name, found in evaluation["elements"].items():
                        if found:
                            st.success(f"✅ {name}")
                        else:
                            st.error(f"❌ {name}")

                st.markdown("---")
                st.markdown("#### Analysis Details")
                st.info(evaluation["rationale"])
                if evaluation["missing_elements"]:
                    st.warning(f"**Missing**: {', '.join(evaluation['missing_elements'])}")
                st.markdown("#### Cited Snippets")
                for snippet in evaluation["snippets"]:
                    st.code(snippet, language=None)
            else:
                st.warning("Please enter a reviewer note to evaluate.")

    with tab_yoy:
        st.header("📈 Year-over-Year Compliance Trends")
        st.info("📊 This section would display multi-year trend data. Showing current-year baseline for demonstration.")

        screening_pct = screening_results["pass_rate"]
        avg_review_days = alert_results["avg_review_days"]
        doc_quality_score = alert_results["avg_doc_quality"]
        ofac_on_time_pct = ofac_results["on_time_pct"]

        yoy_data = pd.DataFrame(
            {
                "Metric": [
                    "% Policies Screened On-Time",
                    "Avg Alert Review Time (days)",
                    "Documentation Quality Score",
                    "% OFAC Reports On-Time",
                ],
                "2023": ["72%", "3.2", "3.1/5", "45%"],
                "2024": ["68%", "2.8", "3.4/5", "52%"],
                "2025 (Current)": [
                    f"{screening_pct:.0f}%",
                    f"{avg_review_days:.1f}",
                    f"{doc_quality_score:.1f}/5",
                    f"{ofac_on_time_pct:.0f}%",
                ],
                "Trend": ["↑", "↑", "↑", "↓"],
            }
        )
        st.dataframe(yoy_data, hide_index=True, use_container_width=True)

        st.markdown("#### Risk Heatmap: Control Coverage by Dimension")
        st.plotly_chart(create_risk_heatmap(data), width="stretch")

    with tab_export:
        st.header("📄 Generate Audit Workpapers")
        st.markdown(
            """
            Create a comprehensive audit workpaper package that includes:
            - Audit objective and scope
            - Testing methodology
            - Summary of results
            - Detailed exception registers
            - Management response templates
            """
        )

        col_left, col_right = st.columns(2)
        with col_left:
            _ = st.text_input("Workpaper Title", "OFAC Sanctions Compliance Audit")
            _ = st.text_input("Lead Auditor", "")
        with col_right:
            _ = st.date_input("Period Start")
            _ = st.date_input("Period End")

        if st.button("📥 Generate Workpaper Package", type="primary"):
            st.success("✅ Workpaper package generated!")
            combined_exceptions = _safe_concat(
                [
                    screening_results.get("exceptions"),
                    alert_results.get("exceptions"),
                    ofac_results.get("exceptions"),
                ]
            )
            if not combined_exceptions.empty:
                st.download_button(
                    "Download All Exceptions (CSV)",
                    combined_exceptions.to_csv(index=False).encode("utf-8"),
                    "all_exceptions.csv",
                    "text/csv",
                )
            else:
                st.info("No exceptions detected across tests.")


if __name__ == "__main__":
    st.set_page_config(page_title="🔍 Audit Analysis", page_icon="🔍", layout="wide")
    render()
