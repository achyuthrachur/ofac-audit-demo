"""Compliance rule helpers for OFAC sanctions audit datasets."""

from __future__ import annotations

import datetime as dt
import re
from typing import Dict, List, Optional

import pandas as pd
from pandas.tseries.holiday import USFederalHolidayCalendar
from pandas.tseries.offsets import CustomBusinessDay

BUSINESS_CALENDAR = USFederalHolidayCalendar()
BUSINESS_DAY = CustomBusinessDay(calendar=BUSINESS_CALENDAR)

NOTE_ELEMENT_PATTERNS = {
    "Alert reason explanation": re.compile(
        r"(alert\s+(triggered|generated|raised)|reason)", re.IGNORECASE
    ),
    "Secondary identifier checked": re.compile(
        r"(ssn|passport|tax\s?id|dob|date of birth|secondary identifier)", re.IGNORECASE
    ),
    "Ticket/reference number": re.compile(r"(tick-\d{4}-\d{3,}|case-\d+|ref-\d+)", re.IGNORECASE),
    "Disposition rationale": re.compile(
        r"(cleared|escalated|confirmed).*(false positive|true hit|per|due|because)",
        re.IGNORECASE,
    ),
    "Reviewer signature/date": re.compile(r"-\s*[a-z][\w.\s]+\d{4}", re.IGNORECASE),
}


def _to_date(value: object) -> Optional[dt.date]:
    if value in (None, "", pd.NaT):
        return None
    return pd.Timestamp(value).date()


def _business_days_between(start: object, end: object) -> Optional[int]:
    start_date = _to_date(start)
    end_date = _to_date(end)
    if start_date is None or end_date is None:
        return None
    start_ts = pd.Timestamp(start_date)
    end_ts = pd.Timestamp(end_date)
    if end_ts <= start_ts:
        return 0
    business_range = pd.date_range(start=start_ts, end=end_ts, freq=BUSINESS_DAY)
    return max(0, len(business_range) - 1)


def check_screening_timeliness(
    policy_row: Dict[str, object],
    cadence_days: int = 30,
    reference_date: Optional[dt.date] = None,
) -> Dict[str, object]:
    """Verify that screening occurred within the cadence and after policy inception."""
    reference = reference_date or dt.date.today()
    last_screen = _to_date(policy_row.get("last_screen_date"))
    inforce_date = _to_date(policy_row.get("inforce_date"))
    flag = (policy_row.get("screen_performed_flag") or "").upper()

    if flag != "Y":
        return {
            "pass": False,
            "details": "Screening flag is 'N' or missing; cadence not met.",
        }

    if last_screen is None:
        return {"pass": False, "details": "Last screening date is missing."}

    if inforce_date and last_screen < inforce_date:
        return {
            "pass": False,
            "details": "Last screening date predates policy inforce date.",
        }

    elapsed = _business_days_between(last_screen, reference)
    if elapsed is None:
        return {"pass": False, "details": "Unable to calculate business day cadence."}

    if elapsed > cadence_days:
        return {
            "pass": False,
            "details": (
                f"Last screening on {last_screen.isoformat()} "
                f"was {elapsed} business days ago (>{cadence_days})."
            ),
        }

    return {
        "pass": True,
        "details": (
            f"Last screening on {last_screen.isoformat()} "
            f"was within {elapsed} business days."
        ),
    }


def check_alert_review_sla(
    alert_row: Dict[str, object],
    sla_days: int = 2,
) -> Dict[str, object]:
    """Ensure alerts are reviewed within SLA using business day calculations."""

    created = _to_date(alert_row.get("alert_created_date"))
    reviewed = _to_date(alert_row.get("reviewed_date"))
    reviewed_flag = (alert_row.get("reviewed_flag") or "").upper()

    if reviewed_flag != "Y" or reviewed is None:
        return {
            "pass": False,
            "days_elapsed": None,
            "details": "Alert has not been reviewed.",
        }

    elapsed = _business_days_between(created, reviewed)
    if elapsed is None:
        return {
            "pass": False,
            "days_elapsed": None,
            "details": "Cannot compute business days between alert and review.",
        }

    if elapsed > sla_days:
        return {
            "pass": False,
            "days_elapsed": elapsed,
            "details": f"Review completed in {elapsed} business days (> {sla_days}).",
        }

    return {
        "pass": True,
        "days_elapsed": elapsed,
        "details": f"Review completed in {elapsed} business days.",
    }


def check_note_completeness(notes_text: str) -> Dict[str, object]:
    """Validate reviewer notes against the five required documentation elements."""

    text = notes_text or ""
    elements_found: List[str] = [
        name for name, pattern in NOTE_ELEMENT_PATTERNS.items() if pattern.search(text)
    ]
    missing = [name for name in NOTE_ELEMENT_PATTERNS.keys() if name not in elements_found]

    return {
        "pass": not missing,
        "elements_found": elements_found,
        "score": len(elements_found),
        "details": (
            "All required documentation elements present."
            if not missing
            else f"Missing elements: {', '.join(missing)}."
        ),
    }


def check_ofac_reporting_timeliness(
    reporting_row: Dict[str, object],
    escalation_date: object,
    sla_days: int = 10,
) -> Dict[str, object]:
    """Confirm OFAC reporting occurred within the SLA following escalation."""

    required_flag = (reporting_row.get("report_required_flag") or "").upper()
    report_date = _to_date(reporting_row.get("report_date"))
    escalation_dt = _to_date(escalation_date)

    if required_flag != "Y":
        return {"pass": True, "days_elapsed": 0, "details": "Reporting not required."}

    if report_date is None or escalation_dt is None:
        return {
            "pass": False,
            "days_elapsed": None,
            "details": "Report date or escalation date missing.",
        }

    elapsed = _business_days_between(escalation_dt, report_date)
    if elapsed is None:
        return {
            "pass": False,
            "days_elapsed": None,
            "details": "Unable to calculate business day difference.",
        }

    if elapsed > sla_days:
        return {
            "pass": False,
            "days_elapsed": elapsed,
            "details": f"Report submitted in {elapsed} business days (> {sla_days}).",
        }

    return {
        "pass": True,
        "days_elapsed": elapsed,
        "details": f"Report submitted in {elapsed} business days.",
    }
