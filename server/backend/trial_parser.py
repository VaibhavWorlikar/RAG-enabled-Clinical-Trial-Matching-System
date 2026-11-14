"""
backend/trial_parser.py
───────────────────────
Turn one ClinicalTrials.gov v2 study record into a flat row for the `trials`
table, plus the text that gets embedded for semantic search.
"""
import re
from typing import Any, Optional

_AGE_RE = re.compile(r"^\s*(\d+(?:\.\d+)?)\s*(year|month|week|day|hour|minute)s?\s*$", re.I)
_UNIT_TO_YEARS = {
    "year": 1.0,
    "month": 1 / 12,
    "week": 1 / 52.1775,
    "day": 1 / 365.25,
    "hour": 1 / 8766,
    "minute": 1 / 525960,
}

_MAX_CRITERIA_CHARS = 8000


def parse_age_years(value: Optional[str]) -> Optional[float]:
    """'18 Years' → 18.0, '6 Months' → 0.5, 'N/A' / None → None."""
    if not value:
        return None
    m = _AGE_RE.match(str(value))
    if not m:
        return None
    return round(float(m.group(1)) * _UNIT_TO_YEARS[m.group(2).lower()], 3)


def _parse_date(value: Optional[str]) -> Optional[str]:
    if not value:
        return None
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return value
    if re.fullmatch(r"\d{4}-\d{2}", value):
        return f"{value}-01"
    return None


def parse_study(study: dict[str, Any]) -> Optional[dict[str, Any]]:
    """Accepts either a v2 API study object or a full study JSON file."""
    protocol = study.get("protocolSection") or {}
    ident = protocol.get("identificationModule") or {}
    nct_id = ident.get("nctId")
    if not nct_id:
        return None

    status = protocol.get("statusModule") or {}
    desc = protocol.get("descriptionModule") or {}
    cond = protocol.get("conditionsModule") or {}
    design = protocol.get("designModule") or {}
    elig = protocol.get("eligibilityModule") or {}
    arms = protocol.get("armsInterventionsModule") or {}
    locs = protocol.get("contactsLocationsModule") or {}

    criteria = (elig.get("eligibilityCriteria") or "").strip()
    if not criteria:
        return None  # nothing to reason over

    countries = sorted({l.get("country") for l in locs.get("locations") or [] if l.get("country")})
    interventions = [
        i.get("name") for i in arms.get("interventions") or [] if isinstance(i, dict) and i.get("name")
    ]

    return {
        "nct_id": nct_id,
        "title": ident.get("briefTitle") or ident.get("officialTitle") or nct_id,
        "brief_summary": (desc.get("briefSummary") or "").strip(),
        "conditions": list(cond.get("conditions") or []),
        "keywords": list(cond.get("keywords") or []),
        "interventions": interventions[:20],
        "study_type": design.get("studyType"),
        "phase": ", ".join(design.get("phases") or []) or None,
        "status": status.get("overallStatus"),
        "sex": (elig.get("sex") or "ALL").upper(),
        "min_age_years": parse_age_years(elig.get("minimumAge")),
        "max_age_years": parse_age_years(elig.get("maximumAge")),
        "healthy_volunteers": bool(elig.get("healthyVolunteers", False)),
        "criteria": criteria[:_MAX_CRITERIA_CHARS],
        "countries": countries,
        "last_updated": _parse_date(
            (status.get("lastUpdatePostDateStruct") or {}).get("date")
        ),
    }


def embedding_text(row: dict[str, Any]) -> str:
    """
    What gets embedded. Conditions/title/summary carry the topical signal;
    only the head of the criteria is included because exclusion lists are
    long and mostly noise for "is this trial about my condition?".
    """
    parts = [
        f"Title: {row['title']}",
        f"Conditions: {', '.join(row['conditions'])}",
    ]
    if row["keywords"]:
        parts.append(f"Keywords: {', '.join(row['keywords'])}")
    if row["interventions"]:
        parts.append(f"Interventions: {', '.join(row['interventions'][:8])}")
    if row["brief_summary"]:
        parts.append(f"Summary: {row['brief_summary'][:1500]}")
    parts.append(f"Eligibility: {row['criteria'][:1500]}")
    return "\n".join(parts)
