"""LLM Note Evaluator
Uses heuristics to assess reviewer note quality against SOP."""

from __future__ import annotations

import re
from typing import Dict, List


def evaluate_note_quality(note_text: str) -> Dict[str, object]:
    """Evaluate a reviewer note against the five-element SOP checklist."""
    text = note_text or ""

    elements_found = {
        "Alert reason explanation": False,
        "Secondary identifier checked": False,
        "Ticket/reference number": False,
        "Disposition rationale": False,
        "Reviewer signature/date": False,
    }
    snippets: List[str] = []

    lower_text = text.lower()
    sentences = [sentence.strip() for sentence in text.split(".") if sentence.strip()]

    reason_keywords = ["alert triggered", "match", "similarity", "flagged", "reason"]
    if any(keyword in lower_text for keyword in reason_keywords):
        elements_found["Alert reason explanation"] = True
        snippets.extend(
            sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in reason_keywords)
        )

    id_keywords = ["ssn", "passport", "dob", "date of birth", "identifier", "verified"]
    if any(keyword in lower_text for keyword in id_keywords):
        elements_found["Secondary identifier checked"] = True
        snippets.extend(
            sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in id_keywords)
        )

    ticket_pattern = r"(TICK-\d{4}-\d{4}|ticket|reference|case)"
    if re.search(ticket_pattern, text, re.IGNORECASE):
        elements_found["Ticket/reference number"] = True
        match = re.search(r"TICK-\d{4}-\d{4}", text)
        if match:
            snippets.append(match.group())

    disposition_keywords = ["cleared", "false positive", "confirmed", "escalated", "no further action", "rationale"]
    if any(keyword in lower_text for keyword in disposition_keywords):
        elements_found["Disposition rationale"] = True
        snippets.extend(
            sentence for sentence in sentences if any(keyword in sentence.lower() for keyword in disposition_keywords)
        )

    date_pattern = r"\d{4}-\d{2}-\d{2}"
    signature_pattern = r"-\s*[A-Z][A-Za-z.]+\s*,?\s*\d{4}-\d{2}-\d{2}"
    if re.search(date_pattern, text) or re.search(signature_pattern, text):
        elements_found["Reviewer signature/date"] = True
        date_match = re.search(date_pattern, text)
        if date_match:
            snippets.append(f"Date: {date_match.group()}")

    snippets = snippets[:5]
    score = sum(1 for found in elements_found.values() if found)
    missing = [name for name, found in elements_found.items() if not found]

    if score == 5:
        rationale = "✅ Note meets all 5 SOP requirements. Documentation is complete and audit-ready."
    elif score >= 3:
        rationale = (
            f"⚠️ Note is partially compliant ({score}/5). Missing elements: {', '.join(missing)}. Requires remediation."
        )
    else:
        rationale = (
            f"❌ Note is non-compliant ({score}/5). Critical gaps in documentation. Must be rewritten."
        )

    return {
        "score": score,
        "elements": elements_found,
        "missing_elements": missing,
        "snippets": snippets,
        "rationale": rationale,
    }
