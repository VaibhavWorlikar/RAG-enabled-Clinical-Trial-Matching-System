"""Unit tests for the pure (no network / no DB) parts of the pipeline."""
import json
from pathlib import Path

import pytest

from backend.condition_normalizer import normalize_conditions
from backend.nlu_extractor import build_patient_text, merge_profiles
from backend.rag_pipeline import condition_patterns
from backend.trial_gap_analyser import (
    GAP_FIELDS,
    build_followup_message,
    extract_gaps,
    gaps_to_questions,
    should_ask_before_showing,
)
from backend.trial_parser import embedding_text, parse_age_years, parse_study

SAMPLES = Path(__file__).parent / "data_sample"


# ─── Trial parsing ────────────────────────────────────────────────────────────

@pytest.mark.parametrize(
    "raw,expected",
    [
        ("18 Years", 18.0),
        ("6 Months", 0.5),
        ("1 Year", 1.0),
        ("2 Weeks", round(2 / 52.1775, 3)),
        ("N/A", None),
        (None, None),
        ("", None),
    ],
)
def test_parse_age_years(raw, expected):
    assert parse_age_years(raw) == expected


def test_parse_all_samples():
    rows = [parse_study(json.loads(p.read_text())) for p in sorted(SAMPLES.glob("*.json"))]
    rows = [r for r in rows if r]
    assert rows, "expected at least one parsable sample"
    for r in rows:
        assert r["nct_id"].startswith("NCT")
        assert r["sex"] in ("ALL", "MALE", "FEMALE")
        assert r["criteria"]
        assert "Conditions:" in embedding_text(r)


def test_parse_sample_ages():
    row = parse_study(json.loads((SAMPLES / "NCT00000200.json").read_text()))
    assert (row["min_age_years"], row["max_age_years"]) == (18.0, 45.0)


def test_parse_study_without_id_or_criteria():
    assert parse_study({}) is None
    assert parse_study({"protocolSection": {"identificationModule": {"nctId": "NCT1"}}}) is None


# ─── Condition normalisation & lexical patterns ───────────────────────────────

def test_normalize_conditions():
    assert normalize_conditions("T2DM, HTN, asthma") == [
        "type 2 diabetes mellitus",
        "hypertension",
        "asthma",
    ]


def test_condition_patterns_handle_possessives_and_short_aliases():
    pats = condition_patterns(["crohn's disease"])
    assert "%crohn%" in pats  # "Crohn Disease" / "Crohn's disease" both match
    assert "%ibd%" not in pats and "%all%" not in pats  # <4 chars skipped


def test_condition_patterns_for_unmapped_condition():
    pats = condition_patterns(["polycystic ovary syndrome"])
    assert "%polycystic ovary%" in pats


# ─── NLU profile merging ──────────────────────────────────────────────────────

def test_merge_profiles_detects_missing_fields():
    merged = merge_profiles({}, {"age": 34, "sex": None, "conditions": ["crohn's disease"]})
    assert merged["missing_fields"] == ["sex"]
    assert merged["clarification_needed"]
    assert "sex" in merged["clarification_question"].lower()


def test_merge_profiles_unions_lists_and_completes():
    a = {"age": 34, "sex": "male", "conditions": ["asthma"], "other_facts": ["non-smoker"]}
    b = {"age": None, "sex": None, "conditions": ["asthma", "obesity"], "other_facts": ["not on steroids"]}
    merged = merge_profiles(a, b)
    assert merged["conditions"] == ["asthma", "obesity"]
    assert merged["other_facts"] == ["non-smoker", "not on steroids"]
    assert merged["age"] == 34 and not merged["clarification_needed"]
    assert "Other facts: non-smoker; not on steroids" in build_patient_text(merged)


# ─── Gap analysis ─────────────────────────────────────────────────────────────

RESULTS = [
    {"eligibility": "POSSIBLE", "missing_info": ["smoking_status", "pregnancy_status"]},
    {"eligibility": "POSSIBLE", "missing_info": ["smoking_status"]},
    {"eligibility": "NO", "missing_info": ["heart_history"]},  # ignored: already NO
]


def test_extract_gaps_orders_by_frequency_and_ignores_no():
    assert extract_gaps(RESULTS) == ["smoking_status", "pregnancy_status"]


def test_should_ask_only_once_and_respects_sex():
    gaps = extract_gaps(RESULTS)
    assert should_ask_before_showing(RESULTS, gaps, [], "female")
    assert not should_ask_before_showing(RESULTS, gaps, ["smoking_status"], "male")
    assert not should_ask_before_showing(RESULTS, gaps, gaps, "female")


def test_never_ask_when_nothing_is_possible():
    results = [{"eligibility": "YES", "missing_info": ["smoking_status"]}]
    assert not should_ask_before_showing(results, ["smoking_status"], [], None)


def test_gaps_to_questions():
    fields, questions = gaps_to_questions(["pregnancy_status", "smoking_status"], [], "male")
    assert fields == ["smoking_status"] and len(questions) == 1
    msg = build_followup_message(questions, 3)
    assert "3 trials" in msg and "1." in msg


def test_gap_fields_have_questions():
    assert len(GAP_FIELDS) == len(set(GAP_FIELDS)) >= 10


def test_skip_medication_question_when_known():
    fields, _ = gaps_to_questions(
        ["current_medications", "smoking_status"], [], "male", {"medications": ["metformin"]}
    )
    assert fields == ["smoking_status"]


def test_merge_replace_mode_drops_contradicted_facts_but_keeps_conditions():
    old = {"age": 34, "sex": "male", "conditions": ["crohn's disease"], "other_facts": ["not on medication"]}
    new = {"conditions": [], "medications": ["mesalamine"], "other_facts": ["non-smoker"]}
    merged = merge_profiles(old, new, replace_lists=True)
    assert merged["conditions"] == ["crohn's disease"]
    assert merged["other_facts"] == ["non-smoker"]


def test_normalize_strips_qualifiers():
    assert normalize_conditions("Crohn's disease (suspected), suspected T2DM") == [
        "crohn's disease",
        "type 2 diabetes mellitus",
    ]


# ─── Contacts & report upload ────────────────────────────────────────────────

from backend.report_reader import ReportError, detect_type  # noqa: E402
from backend.trial_contacts import _parse, rank_sites  # noqa: E402

SITES = [
    {"facility": "A", "city": "Boston", "country": "United States", "status": "RECRUITING"},
    {"facility": "B", "city": "Mumbai", "country": "India", "status": "NOT_YET_RECRUITING"},
    {"facility": "C", "city": "Pune", "country": "India", "status": "RECRUITING"},
]


def test_rank_sites_prefers_city_then_country():
    assert [s["facility"] for s in rank_sites(SITES, "Pune, India")] == ["C", "B", "A"]
    assert [s["facility"] for s in rank_sites(SITES, "India")] == ["C", "B", "A"]
    assert rank_sites(SITES, None)[0]["status"] == "RECRUITING"


def test_parse_contacts():
    study = {"protocolSection": {"contactsLocationsModule": {
        "centralContacts": [{"name": "Jane Doe", "phone": "555-0100", "phoneExt": "12", "email": "j@x.org"}],
        "locations": [{"facility": "Site 1", "city": "Pune", "country": "India",
                       "contacts": [{"name": "Dr X", "email": "x@site.org"}]}],
    }}}
    p = _parse(study)
    assert p["central_contacts"][0] == {"name": "Jane Doe", "phone": "555-0100 ext. 12", "email": "j@x.org"}
    assert p["all_sites"][0]["contact"]["email"] == "x@site.org"


def test_detect_type_uses_magic_bytes():
    assert detect_type(b"%PDF-1.7 ...", "x.bin") == "application/pdf"
    assert detect_type(b"\x89PNG\r\n\x1a\n...", "scan.pdf") == "image/png"
    assert detect_type(b"Hb 13.2 g/dL", "labs.txt") == "text/plain"
    with pytest.raises(ReportError):
        detect_type(b"MZ\x90\x00", "virus.pdf")
