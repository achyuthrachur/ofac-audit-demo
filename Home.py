"""OFAC Sanctions Audit Demo
Main landing page with navigation."""

from __future__ import annotations

import streamlit as st

st.set_page_config(
    page_title="OFAC Audit Demo",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🛡️ OFAC Sanctions Compliance Audit Demo")
st.markdown("---")

st.markdown(
    """
## Welcome to the AI-Assisted OFAC Audit Demonstration

This application demonstrates how Internal Audit can use AI to independently verify
sanctions screening controls at scale while maintaining audit rigor and explainability.

### 🎯 Demo Workflow

1. **📊 Data Generator** - Create synthetic OFAC compliance datasets with configurable
   compliance ratios and failure distributions

2. **🔍 Audit Analysis** - Upload generated datasets and perform comprehensive
   compliance testing with interactive dashboards

### 🚀 Getting Started

Use the sidebar to navigate between pages:
- Start with **Data Generator** to create your test dataset
- Then proceed to **Audit Analysis** to run compliance checks

### 📋 What This Demo Tests

- ✅ **Timely Screening**: Are policyholders screened at onboarding and every 30 days?
- ✅ **Alert Review Quality**: Are alerts investigated within 2 business days with complete documentation?
- ✅ **OFAC Reporting**: Are confirmed matches reported within 10 business days?
"""
)

col1, col2 = st.columns(2)

with col1:
    st.info(
        "**For Conference Demos**: Start with Data Generator using default settings "
        "(500 policies, 60% compliant)"
    )

with col2:
    st.warning(
        "**For Deep Dives**: Experiment with different compliance ratios and failure "
        "distributions"
    )

st.markdown("---")
st.caption("Demo uses 100% synthetic data | No PII | Built for Internal Audit professionals")
