"""Data Generator page for the OFAC sanctions audit demo."""

from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import altair as alt
import pandas as pd
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from data_generator import (  # type: ignore  # pylint: disable=wrong-import-position
    GeneratorConfig,
    OFACDatasetGenerator,
    load_config,
    merge_overrides,
    save_datasets,
    validate_config,
)

CONFIG_PATH = ROOT_DIR / "config.yaml"
OUTPUT_DIR = ROOT_DIR / "output"


@st.cache_data(show_spinner=False)
def get_base_config() -> GeneratorConfig:
    return load_config(CONFIG_PATH)


def make_overrides(
    total_policies: int, compliance_ratio_pct: float, failures_pct: Dict[str, float]
) -> Dict[str, Dict]:
    compliance_ratio = compliance_ratio_pct / 100.0
    failure_distribution = {k: v / 100.0 for k, v in failures_pct.items()}
    return {
        "population": {
            "total_policies": total_policies,
            "compliance_ratio": compliance_ratio,
        },
        "failure_distribution": failure_distribution,
    }


def generate_datasets(
    overrides: Dict[str, Dict], seed: int | None
) -> Tuple[Dict[str, pd.DataFrame], GeneratorConfig]:
    base_config = get_base_config()
    config = merge_overrides(base_config, overrides)
    validate_config(config)
    generator = OFACDatasetGenerator(config, seed=seed)
    datasets = generator.generate()
    return datasets, config


def prepare_download(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def show_generation_page() -> None:
    st.subheader("Dataset Generator")
    base_config = get_base_config()

    population = base_config.population
    failures = base_config.failure_distribution

    st.markdown("Configure the synthetic dataset parameters before generating output CSV files.")
    total_policies = st.slider(
        "Total policies",
        min_value=100,
        max_value=5000,
        value=int(population["total_policies"]),
        step=50,
    )
    compliance_ratio_pct = st.slider(
        "Compliance ratio (%)",
        min_value=10,
        max_value=95,
        value=int(population["compliance_ratio"] * 100),
    )

    st.markdown("**Failure distribution (% of non-compliant policies)**")
    cols = st.columns(len(failures))
    failure_inputs: Dict[str, float] = {}
    for col, (name, value) in zip(cols, failures.items()):
        label = name.replace("_", " ").title()
        failure_inputs[name] = col.slider(
            label,
            min_value=0,
            max_value=100,
            value=int(value * 100),
            step=5,
            key=f"dist_{name}",
        )

    failure_total = sum(failure_inputs.values())
    seed = st.number_input(
        "Random seed (optional)",
        min_value=0,
        max_value=999_999,
        step=1,
        value=42,
    )

    generate_disabled = failure_total != 100
    if generate_disabled:
        st.warning("Failure distribution must sum to 100%. Adjust the sliders before generating.")

    if st.button("Generate Dataset", disabled=generate_disabled, type="primary"):
        with st.spinner("Generating synthetic datasets..."):
            progress = st.progress(0)
            overrides = make_overrides(total_policies, compliance_ratio_pct, failure_inputs)
            progress.progress(30)
            datasets, config = generate_datasets(overrides, seed)
            progress.progress(60)
            save_datasets(datasets, OUTPUT_DIR)
            progress.progress(80)
            st.session_state["datasets"] = datasets
            st.session_state["active_config"] = config
            st.session_state["generation_summary"] = build_summary(datasets, config)
            st.session_state["generated_files"] = {
                "policyholders": str(OUTPUT_DIR / "policyholders.csv"),
                "alerts": str(OUTPUT_DIR / "alerts.csv"),
                "ofac_reporting": str(OUTPUT_DIR / "ofac_reporting.csv"),
                "compliance_validation": str(OUTPUT_DIR / "compliance_validation.csv"),
                "generation_params": {
                    "total_policies": total_policies,
                    "compliance_ratio": compliance_ratio_pct / 100.0,
                    "seed": seed,
                    "timestamp": datetime.utcnow().isoformat(),
                },
            }
            progress.progress(100)
            st.success("Dataset generation complete.")

    if "datasets" in st.session_state:
        summary = st.session_state.get("generation_summary", {})
        st.subheader("Summary Statistics")
        metrics_cols = st.columns(4)
        metrics_cols[0].metric("Policies", f"{summary.get('policies', 0)}")
        metrics_cols[1].metric("Alerts", f"{summary.get('alerts', 0)}")
        metrics_cols[2].metric("Confirmed matches", f"{summary.get('confirmed_matches', 0)}")
        metrics_cols[3].metric("Compliance ratio", f"{summary.get('compliance_ratio', 0):.0%}")

        st.markdown("**Compliance Breakdown**")
        st.dataframe(summary.get("compliance_breakdown"))

        st.markdown("**Download CSV Outputs**")
        datasets = st.session_state["datasets"]
        downloads = st.columns(4)
        for col, (name, df) in zip(
            downloads,
            [
                ("policyholders.csv", datasets["policyholders"]),
                ("alerts.csv", datasets["alerts"]),
                ("ofac_reporting.csv", datasets["ofac_reporting"]),
                ("compliance_validation.csv", datasets["compliance_validation"]),
            ],
        ):
            col.download_button(
                label=f"Download {name}",
                data=prepare_download(df),
                file_name=name,
                mime="text/csv",
                use_container_width=True,
            )

        st.markdown("**Preview (first 10 rows)**")
        policy_df = datasets["policyholders"].head(10)
        alert_df = datasets["alerts"].head(10)
        report_df = datasets["ofac_reporting"].head(10)
        validation_df = datasets["compliance_validation"].head(10)

        st.write("Policyholders")
        st.dataframe(policy_df)
        st.write("Alerts")
        st.dataframe(alert_df)
        st.write("OFAC Reporting")
        st.dataframe(report_df)
        st.write("Compliance Validation")
        st.dataframe(validation_df)

        if st.session_state.get("generated_files"):
            st.markdown("---")
            st.success("✅ Data generation complete! Files are ready for analysis.")
            col_go, col_hint = st.columns([1, 3])
            with col_go:
                if st.button("🔍 Go to Analysis", type="primary", use_container_width=True):
                    st.session_state["nav_selection"] = "🔍 Audit Analysis"
            with col_hint:
                st.info("Or download the files above and proceed to the Analysis page manually.")


def show_inspection_page() -> None:
    st.subheader("Data Inspection")
    if "datasets" not in st.session_state:
        st.info("Generate a dataset first to explore the data.")
        return

    datasets: Dict[str, pd.DataFrame] = st.session_state["datasets"]
    compliance_df = datasets["compliance_validation"]
    policy_df = datasets["policyholders"].copy()
    policy_df["compliance_status"] = policy_df["policy_id"].map(
        compliance_df.set_index("policy_id")["compliance_status"]
    )
    policy_df["failure_category"] = policy_df["policy_id"].map(
        compliance_df.set_index("policy_id")["failure_category"]
    )

    tab_policies, tab_alerts, tab_reporting, tab_validation = st.tabs(
        ["Policyholders", "Alerts", "OFAC Reporting", "Validation Log"]
    )

    with tab_policies:
        status_filter = st.multiselect(
            "Policy status",
            options=sorted(policy_df["status"].unique()),
            default=sorted(policy_df["status"].unique()),
            key="policy_status_filter",
        )
        compliance_filter = st.multiselect(
            "Compliance status",
            options=["compliant", "non-compliant"],
            default=["compliant", "non-compliant"],
            key="policy_compliance_filter",
        )
        available_failures = sorted([c for c in policy_df["failure_category"].dropna().unique()])
        failure_filter = st.multiselect(
            "Failure category",
            options=available_failures,
            default=available_failures,
            key="policy_failure_filter",
        )
        include_compliant = st.checkbox("Include compliant policies", value=True)

        filtered_policies = policy_df[
            policy_df["status"].isin(status_filter) & policy_df["compliance_status"].isin(compliance_filter)
        ]
        if failure_filter:
            filtered_policies = filtered_policies[
                filtered_policies["failure_category"].isin(failure_filter)
                | (filtered_policies["failure_category"].isna() & include_compliant)
            ]
        elif not include_compliant:
            filtered_policies = filtered_policies[filtered_policies["failure_category"].notna()]

        st.dataframe(filtered_policies.drop(columns=["failure_category"]))

    with tab_alerts:
        alerts_df = datasets["alerts"]
        disposition_filter = st.multiselect(
            "Disposition",
            options=sorted(alerts_df["disposition"].unique()),
            default=sorted(alerts_df["disposition"].unique()),
            key="alert_disposition_filter",
        )
        reviewer_filter = st.multiselect(
            "Reviewer",
            options=sorted(alerts_df["reviewer_id"].unique()),
            default=sorted(alerts_df["reviewer_id"].unique()),
            key="alert_reviewer_filter",
        )
        filtered_alerts = alerts_df[
            alerts_df["disposition"].isin(disposition_filter) & alerts_df["reviewer_id"].isin(reviewer_filter)
        ]
        st.dataframe(filtered_alerts)

    with tab_reporting:
        report_df = datasets["ofac_reporting"]
        report_type_filter = st.multiselect(
            "Report type",
            options=sorted(report_df["report_type"].unique()),
            default=sorted(report_df["report_type"].unique()),
            key="report_type_filter",
        )
        filtered_reports = report_df[report_df["report_type"].isin(report_type_filter)]
        st.dataframe(filtered_reports)

    with tab_validation:
        available_failures = sorted(compliance_df["failure_category"].dropna().unique())
        fail_filter = st.multiselect(
            "Failure category",
            options=available_failures,
            default=available_failures,
            key="validation_failure_filter",
        )
        validation_status_filter = st.multiselect(
            "Compliance status",
            options=sorted(compliance_df["compliance_status"].unique()),
            default=sorted(compliance_df["compliance_status"].unique()),
            key="validation_status_filter",
        )
        filtered_validation = compliance_df[
            compliance_df["compliance_status"].isin(validation_status_filter)
        ]
        if fail_filter:
            filtered_validation = filtered_validation[
                filtered_validation["failure_category"].isin(fail_filter)
            ]
        st.dataframe(filtered_validation)

    st.subheader("Sample Record Viewer")
    dataset_name = st.selectbox(
        "Dataset",
        options=["Policyholders", "Alerts", "OFAC Reporting", "Compliance Validation"],
    )
    dataset_map = {
        "Policyholders": policy_df.drop(columns=["failure_category"]),
        "Alerts": datasets["alerts"],
        "OFAC Reporting": datasets["ofac_reporting"],
        "Compliance Validation": compliance_df,
    }
    selected_df = dataset_map[dataset_name]
    if selected_df.empty:
        st.info("No records match the current filters.")
        return

    record_index = st.slider(
        "Record index",
        min_value=0,
        max_value=len(selected_df) - 1,
        value=0,
    )
    record = selected_df.iloc[record_index].to_dict()
    with st.expander("Record details", expanded=True):
        st.json(record)


def show_summary_page() -> None:
    st.subheader("Compliance Summary")
    if "datasets" not in st.session_state:
        st.info("Generate a dataset first to view compliance insights.")
        return

    datasets: Dict[str, pd.DataFrame] = st.session_state["datasets"]
    validation = datasets["compliance_validation"]

    status_counts = validation["compliance_status"].value_counts().reset_index()
    status_counts.columns = ["status", "count"]
    st.write("Compliant vs Non-compliant Policies")
    st.bar_chart(data=status_counts.set_index("status"))

    non_compliant = validation[validation["compliance_status"] == "non-compliant"]
    if not non_compliant.empty:
        failure_counts = non_compliant["failure_category"].value_counts().reset_index()
        failure_counts.columns = ["failure_category", "count"]
        st.write("Failure Category Distribution")
        pie_chart = (
            alt.Chart(failure_counts)
            .mark_arc()
            .encode(theta="count", color="failure_category", tooltip=["failure_category", "count"])
        )
        st.altair_chart(pie_chart, width="stretch")
    else:
        st.info("All policies are compliant in the current dataset.")

    st.write("Data Quality Metrics")
    summary = st.session_state.get("generation_summary", {})
    metrics_df = pd.DataFrame(
        [
            {"Metric": "Policies generated", "Value": summary.get("policies", 0)},
            {"Metric": "Alerts generated", "Value": summary.get("alerts", 0)},
            {"Metric": "Alerts per policy", "Value": summary.get("alerts_per_policy", 0.0)},
            {"Metric": "Confirmed match rate", "Value": f"{summary.get('confirmed_rate', 0.0):.1%}"},
            {"Metric": "Compliance ratio", "Value": f"{summary.get('compliance_ratio', 0.0):.1%}"},
        ]
    )
    metrics_df["Value"] = metrics_df["Value"].astype(str)
    st.table(metrics_df)


def build_summary(datasets: Dict[str, pd.DataFrame], config: GeneratorConfig) -> Dict[str, float]:
    policy_df = datasets["policyholders"]
    alerts_df = datasets["alerts"]
    validation = datasets["compliance_validation"]

    confirmed_matches = alerts_df[alerts_df["disposition"] == "confirmed"]
    compliance_counts = (
        validation["compliance_status"].value_counts().rename_axis("status").reset_index(name="count")
    )
    compliance_table = compliance_counts.set_index("status")

    summary = {
        "policies": len(policy_df),
        "alerts": len(alerts_df),
        "confirmed_matches": len(confirmed_matches),
        "compliance_ratio": float(config.population["compliance_ratio"]),
        "alerts_per_policy": len(alerts_df) / max(len(policy_df), 1),
        "confirmed_rate": len(confirmed_matches) / max(len(alerts_df), 1),
        "compliance_breakdown": compliance_table,
    }
    return summary


def render_page() -> None:
    st.title("📊 Data Generator")
    generator_tab, inspection_tab, summary_tab = st.tabs(
        ["Generator", "Inspection", "Compliance Summary"]
    )
    with generator_tab:
        show_generation_page()
    with inspection_tab:
        show_inspection_page()
    with summary_tab:
        show_summary_page()


if __name__ == "__main__":
    st.set_page_config(page_title="📊 Data Generator", page_icon="📊", layout="wide")
    render_page()
