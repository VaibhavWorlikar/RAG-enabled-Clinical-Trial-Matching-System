"""
backend/trial_gap_analyser.py
─────────────────────────────
"Ask before suggesting": decide whether to ask the patient follow-up questions
before showing results, based on the structured `missing_info` that the
eligibility reasoner returns for each trial.

  extract_gaps(results)              → missing fields, most frequent first
  should_ask_before_showing(...)     → True if a POSSIBLE verdict hinges on an
                                       unasked question
  gaps_to_questions(fields, sex)     → friendly question strings
  build_followup_message(questions)  → the chat message sent to the user
"""
from collections import Counter
from typing import Any, Optional

# Field key → question. Order = priority when frequencies tie.
_FIELD_QUESTIONS: dict[str, str] = {
    "pregnancy_status": "Are you currently pregnant, breastfeeding, or planning a pregnancy?",
    "diagnosis_confirmation": "Has your diagnosis been confirmed by a doctor or a test (e.g. biopsy, imaging, blood work)?",
    "disease_stage_severity": "Do you know the stage or severity of your condition?",
    "prior_treatments": "What treatments have you already tried for this condition, and how did they go?",
    "current_medications": "Are you taking any medications right now (including supplements)?",
    "heart_history": "Do you have any history of heart problems (heart failure, arrhythmia, heart attack)?",
    "liver_function": "Do you have any liver conditions or abnormal liver test results?",
    "kidney_function": "Do you have any kidney disease or abnormal kidney test results?",
    "lab_values": "Do you have any recent blood test or lab results related to your condition?",
    "bmi_weight": "What are your approximate height and weight (or BMI)?",
    "smoking_status": "Do you smoke, or have you smoked in the past?",
    "alcohol_substance_use": "Do you drink alcohol or use other substances regularly?",
    "infections": "Do you have any active infections, or a history of HIV, hepatitis B/C, or TB?",
    "other_conditions": "Do you have any other ongoing medical conditions?",
    "recent_surgery": "Have you had surgery or a major procedure in the last few months?",
    "biomarkers_genetics": "Has your condition been tested for specific biomarkers or genetic mutations (e.g. HER2, EGFR, BRCA)?",
    "performance_status": "How active are you day to day — fully active, limited, or needing help with daily tasks?",
    "mental_health": "Do you have any history of mental health conditions?",
}

GAP_FIELDS: tuple[str, ...] = tuple(_FIELD_QUESTIONS)

MAX_QUESTIONS = 4


def extract_gaps(trial_results: list[dict[str, Any]]) -> list[str]:
    """Missing fields from POSSIBLE trials, ordered by how many trials need them."""
    counts: Counter[str] = Counter()
    for r in trial_results:
        if r.get("eligibility") == "POSSIBLE":
            counts.update(set(r.get("missing_info") or []))
    priority = {f: i for i, f in enumerate(GAP_FIELDS)}
    return sorted(counts, key=lambda f: (-counts[f], priority.get(f, 99)))


def _filter_for_profile(
    fields: list[str], sex: Optional[str], profile: Optional[dict[str, Any]] = None
) -> list[str]:
    """Drop questions that don't apply or that the profile already answers."""
    profile = profile or {}
    skip = set()
    if (sex or "").lower() == "male":
        skip.add("pregnancy_status")
    if profile.get("medications"):
        skip.add("current_medications")
    return [f for f in fields if f not in skip]


def should_ask_before_showing(
    trial_results: list[dict[str, Any]],
    gap_fields: list[str],
    asked_gaps: list[str],
    sex: Optional[str] = None,
    profile: Optional[dict[str, Any]] = None,
) -> bool:
    """
    Ask only when at least one trial is POSSIBLE and there is a relevant
    question we haven't asked yet in this session. Never asks twice.
    """
    if not any(r.get("eligibility") == "POSSIBLE" for r in trial_results):
        return False
    asked = set(asked_gaps or [])
    return any(f not in asked for f in _filter_for_profile(gap_fields, sex, profile))


def gaps_to_questions(
    gap_fields: list[str],
    asked_gaps: Optional[list[str]] = None,
    sex: Optional[str] = None,
    profile: Optional[dict[str, Any]] = None,
    max_questions: int = MAX_QUESTIONS,
) -> tuple[list[str], list[str]]:
    """Returns (fields_asked, questions) for not-yet-asked fields."""
    asked = set(asked_gaps or [])
    fields = [f for f in _filter_for_profile(gap_fields, sex, profile) if f not in asked][:max_questions]
    return fields, [_FIELD_QUESTIONS[f] for f in fields]


def build_followup_message(questions: list[str], trials_found: int) -> str:
    noun = "trial" if trials_found == 1 else "trials"
    lines = [
        f"I found **{trials_found} {noun}** that could fit — but a few details decide "
        "whether you'd actually qualify. Could you tell me:",
        "",
    ]
    lines += [f"{i}. {q}" for i, q in enumerate(questions, 1)]
    lines += ["", "Answer whatever you can (or say **skip**) and I'll show you the results."]
    return "\n".join(lines)
