"""OFAC Audit Analysis dashboard page."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List

import pandas as pd
import streamlit as st

import sys

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

COMPLIANT_SAMPLE_NOTE = (
    "Alert triggered on name similarity (87% match). Verified SSN last 4 (3321) "
    "and passport number (no match). Documented in TICK-2025-1234. Cleared as false "
    "positive based on corroborating identifiers. - J.Smith, 2025-10-15"
)

PARTIAL_SAMPLE_NOTE = (
    "Alert triggered on name similarity. Reviewed primary name only; no supporting "
    "identifiers captured. Cleared based on low residual risk."
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


def _resolve_sample(sample_choice: str, data: Dict[str, pd.DataFrame]) -> str:
    alerts = data["alerts"]
    validation = data["compliance_validation"]
    if sample_choice == "Compliant (5/5)":
        candidates = validation[validation["failure_category"] != "documentation_quality"]
        if not candidates.empty:
            policy = candidates.iloc[0]["policy_id"]
            notes = alerts[alerts["policy_id"] == policy]["reviewer_notes"]
            if not notes.empty:
                return str(notes.iloc[0])
        return COMPLIANT_SAMPLE_NOTE
    if sample_choice == "Partially Compliant (2/5)":
        candidates = validation[validation["failure_category"] == "documentation_quality"]
        if not candidates.empty:
            policy = candidates.iloc[0]["policy_id"]
            notes = alerts[alerts["policy_id"] == policy]["reviewer_notes"]
            if not notes.empty:
                return str(notes.iloc[0])
        return PARTIAL_SAMPLE_NOTE
    return ""


def _build_workpaper_zip(
    metrics: Dict[str, float],
    screening_results: Dict[str, object],
    alert_results: Dict[str, object],
    ofac_results: Dict[str, object],
    workpaper_title: str,
    lead_auditor: str,
    period_start: str,
    period_end: str,
) -> io.BytesIO:
    combined_exceptions = _safe_concat(
        [
            screening_results.get("exceptions"),
            alert_results.get("exceptions"),
            ofac_results.get("exceptions"),
        ]
    )
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        summary_lines = [
            f"# {workpaper_title or 'OFAC Sanctions Compliance Audit'}",
            "",
            f"- Lead auditor: {lead_auditor or 'N/A'}",
            f"- Period covered: {period_start} to {period_end}",
            "",
            "## Key Metrics",
            f"- Compliance rate: {metrics['compliance_rate']:.1f}%",
            f"- Total policies: {metrics['total_policies']}",
            f"- Policies with alerts: {metrics['policies_with_alerts']}",
            f"- Confirmed matches: {metrics['confirmed_matches']}",
            f"- Late OFAC reports: {metrics['late_ofac_reports']}",
            "",
            "## Test Summary",
            f"- Screening failures: {screening_results['failed']}",
            f"- Alert review exceptions: {len(alert_results['exceptions'])}",
            f"- OFAC reporting exceptions: {len(ofac_results['exceptions'])}",
        ]
        archive.writestr("summary.md", "\n".join(summary_lines))

        metrics_df = pd.DataFrame(
            [
                {"Metric": "Compliance Rate", "Value": f"{metrics['compliance_rate']:.1f}%"},
                {"Metric": "Total Policies", "Value": metrics["total_policies"]},
                {"Metric": "Policies with Alerts", "Value": metrics["policies_with_alerts"]},
                {"Metric": "Confirmed Matches", "Value": metrics["confirmed_matches"]},
                {"Metric": "Late OFAC Reports", "Value": metrics["late_ofac_reports"]},
            ]
        )
        archive.writestr("metrics.csv", metrics_df.to_csv(index=False))

        archive.writestr(
            "screening_exceptions.csv",
            screening_results["exceptions"].to_csv(index=False),
        )
        archive.writestr(
            "alert_review_exceptions.csv",
            alert_results["exceptions"].to_csv(index=False),
        )
        archive.writestr(
            "ofac_reporting_exceptions.csv",
            ofac_results["exceptions"].to_csv(index=False),
        )
        archive.writestr(
            "combined_exceptions.csv",
            combined_exceptions.to_csv(index=False),
        )
    buffer.seek(0)
    return buffer


def render() -> None:
    st.title("OFAC Audit Analysis Dashboard")
    st.markdown("---")

    st.markdown(
        """
        <style>
        .analysis-high-contrast, .analysis-high-contrast * {
            color: #111111 !important;
        }
        [data-testid="stMetricValue"] {
            color: #111111 !important;
            font-weight: 700 !important;
        }
        [data-testid="stMetricDelta"] {
            color: #0b4f26 !important;
            font-weight: 600 !important;
        }
        .high-contrast-table div[data-testid="stDataFrame"] tbody tr td {
            color: #111111 !important;
            font-weight: 500;
        }
        .analysis-high-contrast h1,
        .analysis-high-contrast h2,
        .analysis-high-contrast h3,
        .analysis-high-contrast h4 {
            color: #101010 !important;
        }
        .stTabs [role="tablist"] [role="tab"] p {
            color: #2b2b2b !important;
            font-weight: 600 !important;
        }
        .stTabs [role="tab"][aria-selected="true"] p {
            color: #000000 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
    st.markdown('<div class="analysis-high-contrast">', unsafe_allow_html=True)
    if "analysis_data" not in st.session_state:
        st.session_state.analysis_data = None

    st.header("Data Upload")

    generated_files = st.session_state.get("generated_files")
    if generated_files:
        st.success("Using dataset generated in the previous step.")
        if st.button("Load Data from Generation"):
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

        if st.button("Load Uploaded Files"):
            files = [policyholders_file, alerts_file, ofac_file, validation_file]
            if all(files):
                try:
                    st.session_state.analysis_data = {
                        "policyholders": pd.read_csv(policyholders_file),
                        "alerts": pd.read_csv(alerts_file),
                        "ofac_reporting": pd.read_csv(ofac_file),
                        "compliance_validation": pd.read_csv(validation_file),
                    }
                    st.success("Files loaded successfully.")
                    st.rerun()
                except Exception as exc:  # pylint: disable=broad-except
                    st.error(f"Error loading files: {exc}")
            else:
                st.warning("Please upload all four CSV files.")

    if not st.session_state.analysis_data:
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
            "Executive Summary",
            "Detailed Analysis",
            "LLM Note Evaluator",
            "Year-over-Year Trends",
            "Export Workpapers",
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
        st.subheader("Alert Timeline and Dispositions")
        st.plotly_chart(create_alert_timeline(data), width="stretch")

    with tab_detailed:
        st.header("Detailed Compliance Testing")
        tab_screen, tab_alerts, tab_reporting = st.tabs(
            ["Screening Timeliness", "Alert Review Quality", "OFAC Reporting"]
        )

        with tab_screen:
            st.subheader("Check 1: Timely Screening")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Policies Tested", screening_results["total"])
            with col_b:
                st.metric("Passed", screening_results["passed"], delta=f"{screening_results['pass_rate']:.1f}%")
            with col_c:
                st.metric("Failed", screening_results["failed"], delta_color="inverse")

            st.markdown("#### Exceptions")
        screen_df = screening_results["exceptions"]
        if screen_df.empty:
            st.success("No screening timeliness exceptions detected.")
        else:
            st.dataframe(screen_df, hide_index=True, use_container_width=True, height=400)
            st.download_button(
                "Download Screening Exceptions",
                screen_df.to_csv(index=False).encode("utf-8"),
                "screening_exceptions.csv",
                "text/csv",
            )

        with tab_alerts:
            st.subheader("Check 2: Alert Review Timeliness and Documentation")
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
            if alert_results["exceptions"].empty:
                st.success("No alert review exceptions identified.")
            else:
                st.dataframe(alert_results["exceptions"], hide_index=True, use_container_width=True, height=400)

        with tab_reporting:
            st.subheader("Check 3: OFAC Reporting Compliance")
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("Reports Required", ofac_results["total_required"])
            with col_b:
                st.metric("On-Time Reports", ofac_results["on_time"])
            with col_c:
                st.metric("Late Reports", ofac_results["late"], delta_color="inverse")

            st.markdown("#### OFAC Reporting Exceptions")
            if ofac_results["exceptions"].empty:
                st.success("All required OFAC reports were submitted on time.")
            else:
                st.dataframe(ofac_results["exceptions"], hide_index=True, use_container_width=True, height=300)

    with tab_llm:
        st.header("LLM Note Evaluator")
        st.markdown("Evaluate reviewer documentation against the SOP checklist.")
        sample_choice = st.radio(
            "Load Sample Note",
            ["Compliant (5/5)", "Partially Compliant (2/5)", "Custom Input"],
            horizontal=True,
        )
        reviewer_note = st.text_area(
            "Reviewer Note to Evaluate",
            value=_resolve_sample(sample_choice, data),
            height=180,
            placeholder="Paste or type a reviewer note here...",
        )

        if st.button("Evaluate Note"):
            if reviewer_note.strip():
                with st.spinner("Analyzing note quality..."):
                    evaluation = evaluate_note_quality(reviewer_note)

                col_summary, col_checklist = st.columns([1, 2])
                with col_summary:
                    score = evaluation["score"]
                    if score >= 4:
                        st.success(f"Score: {score}/5")
                        st.metric("Quality", "Compliant ✅")
                    elif score >= 3:
                        st.warning(f"Score: {score}/5")
                        st.metric("Quality", "Needs Improvement ⚠️")
                    else:
                        st.error(f"Score: {score}/5")
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
                    st.warning(f"Missing: {', '.join(evaluation['missing_elements'])}")
                st.markdown("#### Cited Snippets")
                for snippet in evaluation["snippets"]:
                    st.code(snippet, language=None)
            else:
                st.warning("Please enter a reviewer note to evaluate.")

    with tab_yoy:
        st.header("Year-over-Year Trends")
        st.info("Illustrative comparison for current versus prior periods.")
        yoy_data = pd.DataFrame(
            [
                {
                    "Metric": "% Policies Screened On-Time",
                    "2023": "72%",
                    "2024": "68%",
                    "2025 (Current)": f"{screening_results['pass_rate']:.0f}%",
                },
                {
                    "Metric": "Avg Alert Review Time (days)",
                    "2023": "3.2",
                    "2024": "2.8",
                    "2025 (Current)": f"{alert_results['avg_review_days']:.1f}",
                },
                {
                    "Metric": "Documentation Quality Score",
                    "2023": "3.1/5",
                    "2024": "3.4/5",
                    "2025 (Current)": f"{alert_results['avg_doc_quality']:.1f}/5",
                },
                {
                    "Metric": "% OFAC Reports On-Time",
                    "2023": "45%",
                    "2024": "52%",
                    "2025 (Current)": f"{ofac_results['on_time_pct']:.0f}%",
                },
            ]
        )
        st.dataframe(yoy_data, hide_index=True, use_container_width=True)
        st.markdown("#### Risk Heatmap: Control Coverage by Dimension")
        st.plotly_chart(create_risk_heatmap(data), width="stretch")

    with tab_export:
        st.header("Export Workpapers")
        st.markdown("Generate a ZIP package containing summary metrics and exception registers.")
        col_left, col_right = st.columns(2)
        with col_left:
            workpaper_title = st.text_input("Workpaper Title", "OFAC Sanctions Compliance Audit")
            lead_auditor = st.text_input("Lead Auditor", "")
        with col_right:
            period_start = st.date_input("Period Start").isoformat()
            period_end = st.date_input("Period End").isoformat()

        if st.button("Generate Workpaper Package"):
            package = _build_workpaper_zip(
                metrics,
                screening_results,
                alert_results,
                ofac_results,
                workpaper_title,
                lead_auditor,
                period_start,
                period_end,
            )
            filename = f"{(workpaper_title or 'ofac_workpaper').lower().replace(' ', '_')}_package.zip"
            st.download_button(
                "Download Workpaper Package",
                data=package.getvalue(),
                file_name=filename,
                mime="application/zip",
            )
    st.markdown("</div>", unsafe_allow_html=True)
