# OFAC Sanctions Audit Demo

Synthetic data generator and Streamlit dashboard for demonstrating OFAC sanctions monitoring, testing, and AI-assisted documentation review.

## Project Overview
This project creates repeatable synthetic datasets that mimic insurance policy screening, alert handling, and OFAC reporting processes. The generator intentionally injects control failures across four categories (screening timeliness, alert review timeliness, documentation quality, and reporting timeliness) according to configurable ratios. A multi-page Streamlit application lets analysts tune parameters, generate data, upload or auto-load the resulting CSVs, and run an end-to-end audit analysis.

## Quick Start
1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Generate datasets via CLI (optional)**
   ```bash
   python src/data_generator.py --seed 42
   ```
   CSVs are written to `output/`.

3. **Launch the Streamlit demo**
   ```bash
   streamlit run Home.py
   ```
   The app opens on a landing page that links the full workflow.

## Configuration
Generator behaviour is controlled by `config.yaml`:

| Section | Description |
| --- | --- |
| `population.total_policies` | Base number of policies to create (default 500). |
| `population.compliance_ratio` | Fraction of policies expected to be compliant (default 0.60). |
| `failure_distribution` | Weights applied to non-compliant policies (must sum to 1.0). |
| `alert_rates` | Expected rates for alert generation, escalations, and confirmed matches. |
| `business_rules` | Screening cadence, alert review SLA, and OFAC reporting SLA in business days. |

Overrides can be supplied through the Streamlit UI or via CLI flags (e.g., `--total-policies`, `--compliance-ratio`, `--failure-distribution screening_timeliness=0.4 ...`).

## Generated Outputs
Four CSV files are produced for each run:

### `policyholders.csv`
| Column | Description |
| --- | --- |
| `policy_id` | Unique identifier (e.g., `POL-000123`). |
| `holder_name`, `dob`, `ssn_last4`, `passport_no` | Synthetic customer attributes generated with Faker. |
| `address_country`, `policy_type`, `inforce_date`, `status` | Policy profile details with realistic distributions. |
| `screen_performed_flag`, `last_screen_date`, `screen_method`, `watchlist_version` | Screening metadata aligned with cadence rules. |

### `alerts.csv`
| Column | Description |
| --- | --- |
| `alert_id`, `policy_id`, `alert_created_date`, `alert_reason`, `similarity_score` | Alert metadata and similarity metrics. |
| `reviewed_flag`, `reviewed_date`, `reviewer_id`, `reviewer_notes` | Review workflow evidence, including required documentation elements. |
| `disposition`, `escalation_date`, `senior_reviewer_id`, `supporting_docs` | Escalation outcomes and references. |

### `ofac_reporting.csv`
| Column | Description |
| --- | --- |
| `policy_id`, `report_required_flag`, `report_type`, `report_date`, `report_reference` | OFAC filing details for confirmed matches. |
| `ack_received_flag`, `ack_date` | Acknowledgement tracking for submitted reports. |

### `compliance_validation.csv`
| Column | Description |
| --- | --- |
| `policy_id` | Links back to the policy record. |
| `compliance_status` | `compliant` or `non-compliant`. |
| `failure_category` | Failure type (if any). |
| `failure_details` | Narrative describing the precise rule violation. |

## Streamlit App Features
- **Home** – Overview of the demo workflow and navigation hints.
- **📊 Data Generator** – Parameter sliders, progress feedback, preview tables, CSV downloads, and a direct link into the analysis experience. The page also provides granular inspection tabs and summary visuals for generated datasets.
- **🔍 Audit Analysis** – Upload or auto-load datasets, run compliance checks, visualise insights, review AI note scoring, compare YoY trends, and export workpaper artifacts.

## Running the Application

### Launch the Multi-Page App
```bash
streamlit run Home.py
```

The app will open in your browser with:
- **Home**: Landing page with navigation
- **📊 Data Generator**: Create synthetic datasets
- **🔍 Audit Analysis**: Upload and analyze compliance data

### Demo Workflow
1. Navigate to **Data Generator**
2. Configure parameters (or use defaults)
3. Click **Generate Dataset**
4. Download CSVs (optional)
5. Click **Go to Analysis**
6. Explore the five analysis tabs:
   - Executive Summary
   - Detailed Analysis
   - LLM Note Evaluator
   - YoY Trends
   - Export Workpapers

### Manual Upload Mode
If you have pre-generated CSV files:
1. Go directly to **Audit Analysis**
2. Upload all four CSV files
3. Click **Load Uploaded Files**
4. Begin analysis

## Compliance Rules
Centralized checks are implemented in `src/compliance_rules.py`:

| Function | Purpose |
| --- | --- |
| `check_screening_timeliness(policy_row)` | Verifies screenings occur within the 30-business-day cadence and after policy inception. |
| `check_alert_review_sla(alert_row)` | Ensures analyst reviews complete within the 2-business-day SLA. |
| `check_note_completeness(notes_text)` | Confirms reviewer notes contain all five documentation elements. |
| `check_ofac_reporting_timeliness(reporting_row, escalation_date)` | Validates OFAC reports are filed within 10 business days of escalation. |

These utilities feed both the data generator and the downstream analytics.

## Implementation Notes
- Session state persists generated file paths, enabling one-click transitions between pages.
- Business-day arithmetic excludes weekends and US federal holidays.
- Faker seeds the random state for reproducible demos when a seed is provided.
- Failure assignment honours the configured distribution while ensuring mutually exclusive failure categories per policy.
- Referential integrity is enforced: alerts always reference existing policies and OFAC reporting rows always link to confirmed alerts.
- Edge cases include policies with multiple alerts and confirmed matches that require follow-up OFAC reports.
- The LLM evaluator is rule-based for demo purposes; swap in a real LLM provider for production pilots.
- Workpaper export currently downloads exception registers as CSV; integrate PDF/Word generation for production use.

Refer to `docs/SOP.md` for the full sanctions monitoring SOP reflected in both generation and analytics logic.
