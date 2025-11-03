# OFAC Sanctions Audit Demo

Synthetic data generator and Streamlit app for demonstrating OFAC sanctions monitoring and compliance testing workflows.

## Project Overview
This project creates repeatable synthetic datasets that mimic insurance policy screening, alert handling, and OFAC reporting processes. The generator intentionally injects control failures across four categories (screening timeliness, alert review timeliness, documentation quality, and reporting timeliness) according to configurable ratios. A Streamlit application lets analysts tune parameters, run the generator, explore the resulting datasets, and review compliance summaries.

## Quick Start
1. **Install dependencies**
   ```bash
   python -m venv .venv
   .venv\Scripts\activate
   pip install -r requirements.txt
   ```
2. **Generate datasets via CLI**
   ```bash
   python src/data_generator.py --seed 42
   ```
   CSVs are written to `output/`.
3. **Launch the Streamlit demo**
   ```bash
   streamlit run src/demo_app.py
   ```
   Use the sidebar to switch between the generator, inspection, and compliance summary views.

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

## Compliance Rules
Centralized checks are implemented in `src/compliance_rules.py`:

| Function | Purpose |
| --- | --- |
| `check_screening_timeliness(policy_row)` | Verifies screenings occur within the 30-business-day cadence and after policy inception. |
| `check_alert_review_sla(alert_row)` | Ensures analyst reviews complete within the 2-business-day SLA. |
| `check_note_completeness(notes_text)` | Confirms reviewer notes contain all five documentation elements. |
| `check_ofac_reporting_timeliness(reporting_row, escalation_date)` | Validates OFAC reports are filed within 10 business days of escalation. |

These utilities are used both during data generation (to guarantee targeted failures) and can be reused by downstream auditing workflows.

## Streamlit App Features
- **Data Generator** – Sliders to control population size, compliance ratio, and failure distribution, with progress feedback, preview tables, and CSV downloads.
- **Data Inspection** – Filterable tables for each dataset, plus a sample record viewer for detailed inspection.
- **Compliance Summary** – Bar and pie charts summarizing compliance status and failure categories, alongside key data quality metrics.

## Implementation Notes
- Business-day arithmetic excludes weekends and US federal holidays.
- Faker seeds the random state for reproducible demos when a seed is provided.
- Failure assignment honours the configured distribution while ensuring mutually exclusive failure categories per policy.
- Referential integrity is enforced: alerts always reference existing policies and OFAC reporting rows always link to confirmed alerts.
- Edge cases include policies with multiple alerts and confirmed matches that require follow-up OFAC reports.

Refer to `docs/SOP.md` for the full sanctions monitoring SOP reflected in the generator logic.
