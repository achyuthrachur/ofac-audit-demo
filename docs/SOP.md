# OFAC Sanctions Monitoring SOP

This standard operating procedure (SOP) describes how sanctions screening, alert handling, documentation, and OFAC reporting should be executed for the synthetic audit dataset.

## 1. Screening Cadence
- Every in-force policy must be screened against the OFAC SDN list at least once every **30 business days**.
- Screenings may be supported by batch, real-time, or vendor API methods.
- Each screening must reference the watchlist version used and be recorded against the policy.
- Policies that have lapsed must receive one final screening before the lapse date.

## 2. Alert Review Service-Level Agreement
- Alerts are generated when a screening match exceeds similarity thresholds.
- Analysts must review each alert within **2 business days** of creation.
- Reviews include confirming the triggering attribute, comparing secondary identifiers, and deciding on a disposition.
- Alerts that cannot be cleared must be escalated to a senior reviewer with supporting documentation.

## 3. Documentation Standards
Alert reviewers must capture the following five documentation elements:
1. **Alert reason explanation** – describe the triggering attribute and similarity score.
2. **Secondary identifier checked** – identify any secondary attributes (SSN last 4, DOB, passport number, etc.) that were compared.
3. **Ticket/reference number** – record the tracking case or ticket number tying the alert to workflow evidence (e.g., `TICK-2025-1234`).
4. **Disposition rationale** – state why the alert was cleared, escalated, or confirmed.
5. **Reviewer signature/date** – provide reviewer initials and the review date (e.g., `- J.Smith, 2025-10-15`).

Missing one or more elements constitutes a **documentation quality failure**.

## 4. Escalation Workflow
- Alerts escalated to senior review must include the analyst’s notes, ticket number, and supporting documents.
- Senior reviewers capture their employee ID, escalation date, and decision (cleared, confirmed).
- Confirmed matches trigger OFAC reporting requirements.

## 5. OFAC Reporting Requirements
- **Report Required Flag** must be `Y` for every confirmed match.
- Initial reports must be submitted within **10 business days** of escalation.
- Follow-up reports are issued when OFAC requests additional information or when material updates occur.
- Reporting acknowledgements should be received within 7–14 calendar days; lack of acknowledgement must be logged and remediated.

## 6. Data Quality Metrics
- Maintain a documented compliance ratio target (default 60% compliant, 40% non-compliant policies).
- Track failure categories across:
  - Screening timeliness
  - Alert review timeliness
  - Documentation quality
  - OFAC reporting delay
- Keep referential integrity between policyholders, alerts, and reporting records.

These SOP guidelines are embedded in the synthetic data generator and compliance rules to support prototyping AI audit workflows.
