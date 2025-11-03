"""Synthetic data generator for OFAC sanctions compliance demo."""

from __future__ import annotations

import argparse
import datetime as dt
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd
import yaml
from dateutil.relativedelta import relativedelta
from faker import Faker

from compliance_rules import (
    BUSINESS_DAY,
    check_alert_review_sla,
    check_note_completeness,
    check_ofac_reporting_timeliness,
    check_screening_timeliness,
)

ALERT_REASONS = [
    ("name_match", 0.60),
    ("dob_match", 0.25),
    ("address_match", 0.15),
]

POLICY_TYPES = [("Life", 0.60), ("Annuity", 0.30), ("Health", 0.10)]
POLICY_STATUS = [("Active", 0.85), ("Lapsed", 0.15)]
SCREEN_METHODS = [("batch", 0.50), ("realtime", 0.30), ("vendor_api", 0.20)]
COUNTRY_DOMESTIC_RATE = 0.90
PASSPORT_RATE = 0.30

REVIEWER_POOL = [f"EMP-{idx:05d}" for idx in range(1, 9)]
SENIOR_POOL = [f"EMP-{idx:05d}" for idx in range(1001, 1009)]

REQUIRED_NOTE_ELEMENTS = [
    "Alert reason explanation",
    "Secondary identifier checked",
    "Ticket/reference number",
    "Disposition rationale",
    "Reviewer signature/date",
]


@dataclass
class GeneratorConfig:
    population: Dict[str, Any]
    failure_distribution: Dict[str, float]
    alert_rates: Dict[str, float]
    business_rules: Dict[str, Any]


def load_config(path: Path) -> GeneratorConfig:
    with path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    required_sections = {"population", "failure_distribution", "alert_rates", "business_rules"}
    missing = required_sections - raw.keys()
    if missing:
        raise ValueError(f"Config file missing sections: {', '.join(sorted(missing))}")
    return GeneratorConfig(
        population=raw["population"],
        failure_distribution=raw["failure_distribution"],
        alert_rates=raw["alert_rates"],
        business_rules=raw["business_rules"],
    )


def merge_overrides(config: GeneratorConfig, overrides: Optional[Dict[str, Any]]) -> GeneratorConfig:
    if not overrides:
        return config
    merged = {
        "population": {**config.population},
        "failure_distribution": {**config.failure_distribution},
        "alert_rates": {**config.alert_rates},
        "business_rules": {**config.business_rules},
    }
    for section, values in overrides.items():
        if section not in merged:
            continue
        merged[section].update(values)
    return GeneratorConfig(**merged)


def validate_config(config: GeneratorConfig) -> None:
    compliance_ratio = float(config.population["compliance_ratio"])
    if not 0 < compliance_ratio < 1:
        raise ValueError("compliance_ratio must be between 0 and 1 (exclusive).")

    failure_total = sum(config.failure_distribution.values())
    if not math.isclose(failure_total, 1.0, rel_tol=1e-4, abs_tol=1e-4):
        raise ValueError(
            f"failure_distribution must sum to 1.0, received {failure_total:.4f}."
        )

    for key in ("alert_generation", "escalation_rate", "confirmed_match_rate"):
        rate = config.alert_rates.get(key, 0)
        if not 0 <= rate <= 1:
            raise ValueError(f"alert_rates['{key}'] must be between 0 and 1.")


class OFACDatasetGenerator:
    """Build synthetic OFAC datasets with controlled compliance failures."""

    def __init__(self, config: GeneratorConfig, seed: Optional[int] = None) -> None:
        self.config = config
        self.seed = seed
        self.random = random.Random(seed)
        self.rng = np.random.default_rng(seed)
        self.fake = Faker()
        if seed is not None:
            Faker.seed(seed)
        self.today = dt.date.today()
        self.alert_counter = 1
        self.report_counter = 1

    def generate(self) -> Dict[str, pd.DataFrame]:
        population_cfg = self.config.population
        total_policies = int(population_cfg["total_policies"])
        compliance_ratio = float(population_cfg["compliance_ratio"])
        num_compliant = int(math.floor(total_policies * compliance_ratio))
        num_non_compliant = total_policies - num_compliant

        failure_assignments = self._allocate_failures(num_non_compliant)
        policy_ids = [f"POL-{idx:06d}" for idx in range(1, total_policies + 1)]

        statuses = ["compliant"] * num_compliant + failure_assignments
        self.random.shuffle(statuses)

        alert_plan = self._plan_alert_distribution(policy_ids, statuses)

        policy_rows: List[Dict[str, Any]] = []
        alert_rows: List[Dict[str, Any]] = []
        report_rows: List[Dict[str, Any]] = []
        validation_rows: List[Dict[str, Any]] = []

        failure_counts: Dict[str, int] = {}

        for policy_id, failure_type in zip(policy_ids, statuses):
            results = self._create_policy_record(
                policy_id=policy_id,
                failure_type=failure_type,
                alerts_for_policy=alert_plan["alerts_per_policy"].get(policy_id, 0),
                multi_alert_flags=alert_plan["multi_alert_policies"],
                disposition_pool=alert_plan["disposition_pool"],
            )
            policy_rows.append(results["policy"])
            alert_rows.extend(results["alerts"])
            report_rows.extend(results["reports"])
            validation_rows.append(results["validation"])

            if failure_type != "compliant":
                failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        self._ensure_follow_up_reports(report_rows)

        datasets = {
            "policyholders": pd.DataFrame(policy_rows),
            "alerts": pd.DataFrame(alert_rows),
            "ofac_reporting": pd.DataFrame(report_rows),
            "compliance_validation": pd.DataFrame(validation_rows),
        }
        return datasets

    # ------------------------------------------------------------------
    # Planning helpers
    # ------------------------------------------------------------------
    def _allocate_failures(self, non_compliant_count: int) -> List[str]:
        if non_compliant_count <= 0:
            return []
        categories = list(self.config.failure_distribution.keys())
        weights = np.array([self.config.failure_distribution[c] for c in categories])
        counts = self.rng.multinomial(non_compliant_count, weights)
        assignments: List[str] = []
        for category, count in zip(categories, counts):
            assignments.extend([category] * int(count))
        self.random.shuffle(assignments)
        return assignments

    def _plan_alert_distribution(
        self, policy_ids: List[str], statuses: List[str]
    ) -> Dict[str, Any]:
        total_policies = len(policy_ids)
        alert_rate = float(self.config.alert_rates["alert_generation"])
        escalation_rate = float(self.config.alert_rates["escalation_rate"])
        confirmed_rate = float(self.config.alert_rates["confirmed_match_rate"])

        alert_required_statuses = {"alert_review_timeliness", "documentation_quality", "ofac_reporting_delay"}
        alert_required_indices = [
            idx for idx, status in enumerate(statuses) if status in alert_required_statuses
        ]
        target_alert_policies = max(
            len(alert_required_indices),
            max(1, int(round(total_policies * alert_rate))),
        )

        optional_indices = [
            idx
            for idx, status in enumerate(statuses)
            if status == "compliant" and idx not in alert_required_indices
        ]
        self.random.shuffle(optional_indices)
        extra_needed = max(0, target_alert_policies - len(alert_required_indices))
        chosen_optional = optional_indices[:extra_needed]

        alert_policy_indices = set(alert_required_indices + chosen_optional)
        alerts_per_policy: Dict[str, int] = {}
        for idx in alert_policy_indices:
            alerts_per_policy[policy_ids[idx]] = 1

        # Select 2-3 policies to have multiple alerts.
        multi_target = min(3, len(alert_policy_indices))
        multi_alert_policies = set()
        if multi_target > 0:
            multi_candidates = list(alert_policy_indices)
            self.random.shuffle(multi_candidates)
            for i, idx in enumerate(multi_candidates[:multi_target]):
                policy_id = policy_ids[idx]
                multi_alert_policies.add(policy_id)
                alerts_per_policy[policy_id] = 3 if i == 0 else 2

        expected_alerts = sum(alerts_per_policy.values())
        mandatory_confirmed = sum(1 for status in statuses if status == "ofac_reporting_delay")
        target_escalations = max(
            mandatory_confirmed, int(round(expected_alerts * escalation_rate))
        )
        target_confirmed = max(
            mandatory_confirmed,
            min(target_escalations, int(round(target_escalations * confirmed_rate))),
        )

        disposition_pool = (
            ["confirmed"] * target_confirmed
            + ["escalated"] * max(0, target_escalations - target_confirmed)
            + ["cleared"] * max(0, expected_alerts - target_escalations)
        )
        self.random.shuffle(disposition_pool)

        return {
            "alerts_per_policy": alerts_per_policy,
            "multi_alert_policies": multi_alert_policies,
            "disposition_pool": disposition_pool,
        }

    # ------------------------------------------------------------------
    # Record builders
    # ------------------------------------------------------------------
    def _create_policy_record(
        self,
        policy_id: str,
        failure_type: str,
        alerts_for_policy: int,
        multi_alert_flags: Iterable[str],
        disposition_pool: List[str],
    ) -> Dict[str, Any]:
        base_policy = self._build_policy_base(policy_id, failure_type)
        alerts: List[Dict[str, Any]] = []
        reports: List[Dict[str, Any]] = []
        validation_detail: Dict[str, Any]

        screen_cadence = int(self.config.business_rules["screening_cadence_days"])
        alert_sla = int(self.config.business_rules["alert_review_sla_days"])
        ofac_sla = int(self.config.business_rules["ofac_reporting_sla_days"])

        if failure_type == "screening_timeliness":
            self._apply_screening_failure(base_policy, screen_cadence)
            rule_result = check_screening_timeliness(base_policy, screen_cadence, self.today)
            validation_detail = self._build_validation_row(
                policy_id, False, failure_type, rule_result["details"]
            )

        elif failure_type == "alert_review_timeliness":
            alerts = self._generate_alerts_for_policy(
                base_policy,
                count=max(1, alerts_for_policy),
                disposition_pool=disposition_pool,
                enforce_sla_failure=True,
                alert_sla=alert_sla,
                multi_policy=policy_id in multi_alert_flags,
            )
            failing_alert = next(a for a in alerts if a["sla_failure"])
            rule_result = check_alert_review_sla(failing_alert, alert_sla)
            validation_detail = self._build_validation_row(
                policy_id,
                False,
                failure_type,
                f"Alert {failing_alert['alert_id']}: {rule_result['details']}",
            )

        elif failure_type == "documentation_quality":
            alerts = self._generate_alerts_for_policy(
                base_policy,
                count=max(1, alerts_for_policy),
                disposition_pool=disposition_pool,
                enforce_note_failure=True,
                alert_sla=alert_sla,
                multi_policy=policy_id in multi_alert_flags,
            )
            failing_alert = next(a for a in alerts if a["note_failure"])
            notes_result = check_note_completeness(failing_alert["reviewer_notes"])
            validation_detail = self._build_validation_row(
                policy_id,
                False,
                failure_type,
                f"Alert {failing_alert['alert_id']}: {notes_result['details']}",
            )

        elif failure_type == "ofac_reporting_delay":
            alerts = self._generate_alerts_for_policy(
                base_policy,
                count=max(1, alerts_for_policy),
                disposition_pool=disposition_pool,
                enforce_ofac_failure=True,
                alert_sla=alert_sla,
                ofac_sla=ofac_sla,
                multi_policy=policy_id in multi_alert_flags,
            )
            failing_alert = next(a for a in alerts if a["ofac_failure"])
            reports = self._build_reporting_rows_for_alerts(alerts, ofac_sla)
            failing_report = next(r for r in reports if r["report_failure"])
            rule_result = check_ofac_reporting_timeliness(
                failing_report, failing_report["escalation_date"], ofac_sla
            )
            validation_detail = self._build_validation_row(
                policy_id,
                False,
                failure_type,
                f"Report {failing_report['report_reference']}: {rule_result['details']}",
            )

        else:
            alerts = self._generate_alerts_for_policy(
                base_policy,
                count=alerts_for_policy,
                disposition_pool=disposition_pool,
                alert_sla=alert_sla,
                multi_policy=policy_id in multi_alert_flags,
            )
            reports = self._build_reporting_rows_for_alerts(alerts, ofac_sla)
            screening_result = check_screening_timeliness(base_policy, screen_cadence, self.today)
            validation_detail = self._build_validation_row(
                policy_id,
                screening_result["pass"],
                "" if screening_result["pass"] else "screening_timeliness",
                screening_result["details"],
            )

        # Strip helper keys before dataframe conversion.
        for alert in alerts:
            alert.pop("sla_failure", None)
            alert.pop("note_failure", None)
            alert.pop("ofac_failure", None)

        for report in reports:
            report.pop("report_failure", None)
            report.pop("escalation_date", None)

        return {
            "policy": base_policy,
            "alerts": alerts,
            "reports": reports,
            "validation": validation_detail,
        }

    def _build_policy_base(self, policy_id: str, failure_type: str) -> Dict[str, Any]:
        policy_type = self._weighted_choice(POLICY_TYPES)
        status = self._weighted_choice(POLICY_STATUS)
        method = self._weighted_choice(SCREEN_METHODS)
        watchlist_version = self._generate_watchlist_version()

        dob = self.fake.date_between_dates(dt.date(1950, 1, 1), dt.date(2000, 12, 31))
        inforce_start = self.today - relativedelta(years=3)
        inforce_date = self.fake.date_between_dates(inforce_start, self.today - dt.timedelta(days=1))
        screen_date = self._random_recent_business_day(self.today, 0, self.config.business_rules["screening_cadence_days"] - 1)

        policy = {
            "policy_id": policy_id,
            "holder_name": self.fake.name(),
            "dob": dob.isoformat(),
            "ssn_last4": f"{self.random.randint(0, 9999):04d}",
            "passport_no": self._maybe_passport(),
            "address_country": self._generate_country(),
            "policy_type": policy_type,
            "inforce_date": inforce_date.isoformat(),
            "status": status,
            "screen_performed_flag": "Y",
            "last_screen_date": screen_date.isoformat(),
            "screen_method": method,
            "watchlist_version": watchlist_version,
        }

        # Ensure screening date is not before inforce.
        if dt.date.fromisoformat(policy["last_screen_date"]) < inforce_date:
            policy["last_screen_date"] = inforce_date.isoformat()

        # For policies marked as lapsed, optionally adjust screening cadence to be slightly older.
        if status == "Lapsed" and failure_type == "compliant":
            older_screen = self._random_recent_business_day(self.today, 5, 20)
            policy["last_screen_date"] = older_screen.isoformat()

        return policy

    def _apply_screening_failure(self, policy: Dict[str, Any], cadence: int) -> None:
        if self.random.random() < 0.4:
            policy["screen_performed_flag"] = "N"
            policy["last_screen_date"] = ""
            return

        overdue_days = self.random.randint(cadence + 5, cadence + 45)
        last_screen = self._subtract_business_days(self.today, overdue_days)
        if self.random.random() < 0.3:
            inforce = dt.date.fromisoformat(policy["inforce_date"])
            last_screen = inforce - dt.timedelta(days=self.random.randint(1, 10))
        policy["last_screen_date"] = last_screen.isoformat()

    def _generate_alerts_for_policy(
        self,
        policy: Dict[str, Any],
        count: int,
        disposition_pool: List[str],
        alert_sla: int,
        multi_policy: bool,
        enforce_sla_failure: bool = False,
        enforce_note_failure: bool = False,
        enforce_ofac_failure: bool = False,
        ofac_sla: Optional[int] = None,
    ) -> List[Dict[str, Any]]:
        alerts: List[Dict[str, Any]] = []
        if count <= 0:
            return alerts

        last_screen_date = (
            dt.date.fromisoformat(policy["last_screen_date"])
            if policy.get("last_screen_date")
            else dt.date.fromisoformat(policy["inforce_date"])
        )

        sla_failure_applied = False
        note_failure_applied = False
        ofac_failure_applied = False

        for index in range(count):
            alert_id = f"ALT-{self.alert_counter:06d}"
            self.alert_counter += 1

            reason = self._weighted_choice(ALERT_REASONS)
            base_offset = 1 + (2 * index if multi_policy else 0)
            created_date = self._add_business_days(
                last_screen_date, self.random.randint(base_offset, base_offset + 8)
            )
            similarity = round(self.random.uniform(0.75, 0.98), 2)
            reviewer_id = self.random.choice(REVIEWER_POOL)

            disposition_forced: Optional[str] = None
            alert_ofac_failure = False
            if enforce_ofac_failure and not ofac_failure_applied:
                disposition_forced = "confirmed"
                alert_ofac_failure = True
                ofac_failure_applied = True

            disposition = self._pull_disposition(disposition_pool, disposition_forced)

            if disposition in {"escalated", "confirmed"} or enforce_note_failure:
                reviewed_flag = "Y"
            else:
                reviewed_flag = "Y" if self.random.random() < 0.95 else "N"

            reviewed_date: Optional[dt.date] = None
            sla_failure = False
            if reviewed_flag == "Y":
                if enforce_sla_failure and not sla_failure_applied:
                    review_delay = alert_sla + self.random.randint(1, 4)
                    sla_failure_applied = True
                    sla_failure = True
                else:
                    review_delay = self.random.randint(0, max(alert_sla, 1))
                    if review_delay > alert_sla:
                        sla_failure = True
                reviewed_date = self._add_business_days(created_date, review_delay)
            else:
                sla_failure = True

            ticket_ref = self._generate_ticket_reference(created_date.year)

            notes_missing: List[str] = []
            if enforce_note_failure and not note_failure_applied:
                missing_count = self.random.randint(1, min(3, len(REQUIRED_NOTE_ELEMENTS)))
                notes_missing = self.random.sample(REQUIRED_NOTE_ELEMENTS, missing_count)
                note_failure_applied = True

            note_failure = bool(notes_missing) or reviewed_flag != "Y"
            reviewer_notes = self._compose_reviewer_notes(
                policy,
                reason,
                similarity,
                ticket_ref,
                disposition,
                reviewed_date,
                notes_missing,
                pending_review=reviewed_flag != "Y",
            )

            escalation_date: Optional[dt.date] = None
            senior_reviewer_id = ""
            if disposition in {"escalated", "confirmed"} and reviewed_date:
                escalation_lag = self.random.randint(0, 3)
                escalation_date = self._add_business_days(reviewed_date, escalation_lag)
                senior_reviewer_id = self.random.choice(SENIOR_POOL)

            alert_record = {
                "alert_id": alert_id,
                "policy_id": policy["policy_id"],
                "alert_created_date": created_date.isoformat(),
                "alert_reason": reason,
                "similarity_score": similarity,
                "reviewed_flag": reviewed_flag,
                "reviewed_date": reviewed_date.isoformat() if reviewed_date else "",
                "reviewer_id": reviewer_id,
                "reviewer_notes": reviewer_notes,
                "disposition": disposition,
                "escalation_date": escalation_date.isoformat() if escalation_date else "",
                "senior_reviewer_id": senior_reviewer_id,
                "supporting_docs": ticket_ref,
                "sla_failure": sla_failure,
                "note_failure": note_failure,
                "ofac_failure": alert_ofac_failure,
            }

            alerts.append(alert_record)

        # Safeguard: ensure OFAC failure policies have a confirmed alert.
        if enforce_ofac_failure and not any(alert["ofac_failure"] for alert in alerts):
            # Promote the first alert to a confirmed OFAC failure path.
            primary_alert = alerts[0]
            primary_alert["disposition"] = "confirmed"
            primary_alert["reviewed_flag"] = "Y"
            if not primary_alert.get("reviewed_date"):
                reviewed_date = self._add_business_days(
                    dt.date.fromisoformat(primary_alert["alert_created_date"]),
                    self.random.randint(1, alert_sla + 2),
                )
                primary_alert["reviewed_date"] = reviewed_date.isoformat()
            if not primary_alert.get("escalation_date"):
                escalation_date = self._add_business_days(
                    dt.date.fromisoformat(primary_alert["reviewed_date"] or primary_alert["alert_created_date"]),
                    self.random.randint(0, 2),
                )
                primary_alert["escalation_date"] = escalation_date.isoformat()
            primary_alert["senior_reviewer_id"] = self.random.choice(SENIOR_POOL)
            primary_alert["ofac_failure"] = True
            primary_alert["sla_failure"] = False
            primary_alert["note_failure"] = False
            reviewed_date = dt.date.fromisoformat(primary_alert["reviewed_date"])
            primary_alert["reviewer_notes"] = self._compose_reviewer_notes(
                policy,
                primary_alert["alert_reason"],
                float(primary_alert["similarity_score"]),
                primary_alert["supporting_docs"],
                primary_alert["disposition"],
                reviewed_date,
                missing_elements=[],
                pending_review=False,
            )

        return alerts

    def _build_reporting_rows_for_alerts(
        self,
        alerts: List[Dict[str, Any]],
        ofac_sla: int,
    ) -> List[Dict[str, Any]]:
        reports: List[Dict[str, Any]] = []
        if not alerts:
            return reports

        for alert in alerts:
            if alert["disposition"] != "confirmed" or not alert.get("escalation_date"):
                continue

            escalation_date = dt.date.fromisoformat(alert["escalation_date"])
            delay = self.random.randint(0, ofac_sla)
            report_failure = False
            if alert.get("ofac_failure"):
                delay = ofac_sla + self.random.randint(1, 4)
                report_failure = True

            report_date = self._add_business_days(escalation_date, delay)
            report_reference = f"REP-{self.report_counter:05d}"
            self.report_counter += 1

            report_row = {
                "policy_id": alert["policy_id"],
                "report_required_flag": "Y",
                "report_type": "initial",
                "report_date": report_date.isoformat(),
                "report_reference": report_reference,
                "ack_received_flag": "Y" if self.random.random() < 0.9 else "N",
                "ack_date": "",
                "report_failure": report_failure,
                "escalation_date": escalation_date.isoformat(),
            }

            if report_row["ack_received_flag"] == "Y":
                ack_delay = self.random.randint(7, 14)
                report_row["ack_date"] = (report_date + dt.timedelta(days=ack_delay)).isoformat()

            reports.append(report_row)

        return reports

    def _ensure_follow_up_reports(self, report_rows: List[Dict[str, Any]]) -> None:
        if not report_rows:
            return

        unique_policies = sorted({row["policy_id"] for row in report_rows})
        if not unique_policies:
            return

        follow_up_target = min(2, len(unique_policies))
        if follow_up_target == 0:
            return

        selected_policies = self.random.sample(unique_policies, follow_up_target)
        for policy_id in selected_policies:
            initial_reports = [
                row for row in report_rows if row["policy_id"] == policy_id and row.get("report_type") == "initial"
            ]
            if not initial_reports:
                continue

            base_report = max(initial_reports, key=lambda row: row["report_date"])
            base_date = dt.date.fromisoformat(base_report["report_date"])

            follow_up_date = base_date + dt.timedelta(days=self.random.randint(7, 21))
            follow_up_reference = f"REP-{self.report_counter:05d}"
            self.report_counter += 1

            follow_up = {
                "policy_id": policy_id,
                "report_required_flag": "Y",
                "report_type": "follow-up",
                "report_date": follow_up_date.isoformat(),
                "report_reference": follow_up_reference,
                "ack_received_flag": "Y" if self.random.random() < 0.9 else "N",
                "ack_date": "",
            }
            if follow_up["ack_received_flag"] == "Y":
                ack_delay = self.random.randint(7, 14)
                follow_up["ack_date"] = (follow_up_date + dt.timedelta(days=ack_delay)).isoformat()

            report_rows.append(follow_up)

    def _build_validation_row(
        self,
        policy_id: str,
        compliant: bool,
        failure_category: str,
        details: str,
    ) -> Dict[str, Any]:
        return {
            "policy_id": policy_id,
            "compliance_status": "compliant" if compliant else "non-compliant",
            "failure_category": failure_category,
            "failure_details": details,
        }

    def _pull_disposition(self, pool: List[str], forced: Optional[str]) -> str:
        if forced:
            if forced in pool:
                pool.remove(forced)
            return forced
        if pool:
            return pool.pop()
        return "cleared"

    def _weighted_choice(self, options: List[tuple]) -> Any:
        values, weights = zip(*options)
        return self.random.choices(values, weights=weights, k=1)[0]

    def _generate_watchlist_version(self) -> str:
        year = self.random.choice([2023, 2024, 2025])
        quarter = self.random.randint(1, 4)
        return f"OFAC-{year}-Q{quarter}"

    def _generate_country(self) -> str:
        if self.random.random() < COUNTRY_DOMESTIC_RATE:
            return "United States"
        country = self.fake.country()
        while country == "United States":
            country = self.fake.country()
        return country

    def _maybe_passport(self) -> str:
        if self.random.random() >= PASSPORT_RATE:
            return ""
        letters = "".join(self.random.choices("ABCDEFGHIJKLMNOPQRSTUVWXYZ", k=2))
        digits = f"{self.random.randint(0, 999999):06d}"
        return f"{letters}{digits}"

    def _random_recent_business_day(self, reference: dt.date, min_days: int, max_days: int) -> dt.date:
        max_days = max(min_days, max_days)
        offset = self.random.randint(min_days, max_days) if max_days > 0 else 0
        return self._subtract_business_days(reference, offset)

    def _add_business_days(self, base_date: dt.date, days: int) -> dt.date:
        if days <= 0:
            return base_date
        timestamp = pd.Timestamp(base_date)
        return (timestamp + days * BUSINESS_DAY).date()

    def _subtract_business_days(self, base_date: dt.date, days: int) -> dt.date:
        if days <= 0:
            return base_date
        timestamp = pd.Timestamp(base_date)
        return (timestamp - days * BUSINESS_DAY).date()

    def _generate_ticket_reference(self, year: int) -> str:
        sequence = self.random.randint(1000, 9999)
        return f"TICK-{year}-{sequence}"

    def _compose_reviewer_notes(
        self,
        policy: Dict[str, Any],
        reason: str,
        similarity: float,
        ticket_ref: str,
        disposition: str,
        reviewed_date: Optional[dt.date],
        missing_elements: List[str],
        pending_review: bool = False,
    ) -> str:
        if pending_review:
            return "Alert pending review; documentation will be completed once analysis is finished."

        similarity_pct = int(round(similarity * 100))
        reason_map = {
            "name_match": "name similarity",
            "dob_match": "date of birth alignment",
            "address_match": "address overlap",
        }
        secondary_info = policy.get("passport_no") or "no passport on file"

        components = {
            "Alert reason explanation": f"Alert triggered on {reason_map.get(reason, reason)} ({similarity_pct}% match).",
            "Secondary identifier checked": (
                f"Verified SSN last 4 ({policy['ssn_last4']}) and passport ({secondary_info}) with no watchlist match."
            ),
            "Ticket/reference number": f"Documented in {ticket_ref}.",
            "Disposition rationale": {
                "cleared": "Cleared as false positive based on corroborating identifiers.",
                "escalated": "Escalated to senior review due to residual sanctions indicators.",
                "confirmed": "Confirmed as a likely sanctions match; OFAC reporting initiated.",
            }[disposition],
            "Reviewer signature/date": self._signature_line(reviewed_date),
        }

        text_parts = [
            components[element]
            for element in REQUIRED_NOTE_ELEMENTS
            if element not in missing_elements
        ]
        return " ".join(text_parts).strip()

    def _signature_line(self, reviewed_date: Optional[dt.date]) -> str:
        review_date = reviewed_date or self.today
        initials = self.fake.first_name()[0]
        last = self.fake.last_name()
        return f"- {initials}.{last}, {review_date.isoformat()}"


def save_datasets(datasets: Dict[str, pd.DataFrame], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    filenames = {
        "policyholders": "policyholders.csv",
        "alerts": "alerts.csv",
        "ofac_reporting": "ofac_reporting.csv",
        "compliance_validation": "compliance_validation.csv",
    }
    for key, filename in filenames.items():
        df = datasets.get(key, pd.DataFrame())
        df.to_csv(output_dir / filename, index=False)


def parse_key_value_pairs(pairs: Optional[List[str]]) -> Dict[str, float]:
    result: Dict[str, float] = {}
    if not pairs:
        return result
    for item in pairs:
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        try:
            result[key] = float(value)
        except ValueError:
            raise ValueError(f"Unable to parse float from '{item}'.")
    return result


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate OFAC sanctions audit demo dataset.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to configuration YAML.")
    parser.add_argument("--output-dir", type=str, default="output", help="Directory for generated CSV files.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducibility.")
    parser.add_argument("--total-policies", type=int, help="Override total policies to generate.")
    parser.add_argument("--compliance-ratio", type=float, help="Override compliance ratio (0-1).")
    parser.add_argument(
        "--failure-distribution",
        nargs="*",
        help="Override failure distribution as key=value pairs (e.g., screening_timeliness=0.4).",
    )
    parser.add_argument(
        "--alert-rates",
        nargs="*",
        help="Override alert rates as key=value pairs (alert_generation=0.15 escalation_rate=0.2).",
    )
    return parser.parse_args(args)


def main(cli_args: Optional[List[str]] = None) -> None:
    args = parse_args(cli_args)
    config_path = Path(args.config)
    config = load_config(config_path)

    overrides: Dict[str, Dict[str, Any]] = {}
    population_overrides: Dict[str, Any] = {}
    if args.total_policies is not None:
        population_overrides["total_policies"] = args.total_policies
    if args.compliance_ratio is not None:
        population_overrides["compliance_ratio"] = args.compliance_ratio
    if population_overrides:
        overrides["population"] = population_overrides

    failure_override = parse_key_value_pairs(args.failure_distribution)
    if failure_override:
        overrides["failure_distribution"] = failure_override

    alert_override = parse_key_value_pairs(args.alert_rates)
    if alert_override:
        overrides["alert_rates"] = alert_override

    config = merge_overrides(config, overrides)
    validate_config(config)

    generator = OFACDatasetGenerator(config, seed=args.seed)
    datasets = generator.generate()

    output_dir = Path(args.output_dir)
    save_datasets(datasets, output_dir)

    policy_count = len(datasets["policyholders"])
    alert_count = len(datasets["alerts"])
    confirmed_count = len(datasets["alerts"][datasets["alerts"]["disposition"] == "confirmed"])
    non_compliant = datasets["compliance_validation"][
        datasets["compliance_validation"]["compliance_status"] == "non-compliant"
    ]

    print(f"Generated {policy_count} policies with {alert_count} alerts.")
    print(
        f"Confirmed matches: {confirmed_count} | Non-compliant policies: {len(non_compliant)} "
        f"({non_compliant['failure_category'].value_counts().to_dict()})"
    )
    print(f"CSV files written to {output_dir.resolve()}")


if __name__ == "__main__":
    main()
