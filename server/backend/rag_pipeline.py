"""
backend/rag_pipeline.py
───────────────────────
Hybrid retrieval over the `trials` table:

  1. Semantic   — pgvector cosine distance between the patient query embedding
                  and each trial's embedding.
  2. Structured — age / sex / recruiting-status filters applied *inside* the
                  SQL query, so they can never be bypassed by a fallback.
  3. Lexical    — trials whose listed conditions (or title) mention one of the
                  patient's normalised conditions or their known aliases are
                  ranked ahead of purely semantic matches.
"""
import logging
import re
from typing import Any, Optional

from backend import db
from backend.condition_normalizer import build_search_variants
from backend.config import CANDIDATE_POOL, MAX_COSINE_DISTANCE, RETRIEVAL_K
from backend.embeddings import embed_query

logger = logging.getLogger(__name__)

OPEN_STATUSES = ["RECRUITING", "NOT_YET_RECRUITING", "ENROLLING_BY_INVITATION"]

# Generic words stripped to get a condition's "core" term for lexical matching.
_GENERIC_SUFFIXES = re.compile(
    r"\b(disease|disorder|syndrome|mellitus|infection|condition)s?\b", re.I
)

_TRIAL_COLUMNS = """
    nct_id, title, brief_summary, conditions, interventions, phase, status,
    sex, min_age_years, max_age_years, criteria, countries
"""


def _normalise_term(term: str) -> str:
    term = term.lower().replace("’", "'")
    term = re.sub(r"'s\b", "", term).replace("'", "")
    return " ".join(term.split())


def condition_patterns(conditions: list[str]) -> list[str]:
    """
    Build SQL LIKE patterns from canonical conditions + their aliases.
    Aliases shorter than 4 chars ("ms", "all", "ad") are skipped because as
    substrings they would match almost anything.
    """
    terms: set[str] = set()
    for variant in build_search_variants([c.lower() for c in conditions]):
        norm = _normalise_term(variant)
        core = " ".join(_GENERIC_SUFFIXES.sub("", norm).split())
        for t in (norm, core):
            if len(t) >= 4:
                terms.add(t)
    return sorted(f"%{t}%" for t in terms)


def build_query_text(profile: dict[str, Any]) -> str:
    conditions = ", ".join(profile.get("conditions") or [])
    parts = [f"Clinical trial for patients with {conditions}."]
    if profile.get("symptoms"):
        parts.append(f"Symptoms: {', '.join(profile['symptoms'])}.")
    if profile.get("medical_history"):
        parts.append(f"History: {', '.join(profile['medical_history'])}.")
    if profile.get("medications"):
        parts.append(f"Current medications: {', '.join(profile['medications'])}.")
    return " ".join(parts)


async def retrieve_trials(
    profile: dict[str, Any],
    k: int = RETRIEVAL_K,
    exclude_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    conditions = profile.get("conditions") or []
    age = profile.get("age")
    sex = (profile.get("sex") or "").upper() or None

    query_vec = db.to_vector_literal(await embed_query(build_query_text(profile)))
    patterns = condition_patterns(conditions)

    rows = await db.fetch_all(
        f"""
        SELECT {_TRIAL_COLUMNS},
               embedding <=> %(vec)s::vector AS distance,
               EXISTS (
                   SELECT 1 FROM unnest(conditions || title) AS c
                    WHERE translate(lower(c), '''’', '') LIKE ANY(%(patterns)s::text[])
               ) AS condition_match
          FROM trials
         WHERE embedding IS NOT NULL
           AND status = ANY(%(statuses)s)
           AND NOT (nct_id = ANY(%(exclude)s))
           AND (%(age)s::real IS NULL
                OR ((min_age_years IS NULL OR min_age_years <= %(age)s)
                AND (max_age_years IS NULL OR max_age_years >= %(age)s)))
           AND (%(sex)s::text IS NULL OR sex = 'ALL' OR sex = %(sex)s)
         ORDER BY distance
         LIMIT %(pool)s
        """,
        {
            "vec": query_vec,
            "patterns": patterns,
            "statuses": OPEN_STATUSES,
            "exclude": exclude_ids or [],
            "age": age,
            "sex": sex,
            "pool": max(CANDIDATE_POOL, k * 4),
        },
    )

    candidates = [r for r in rows if r["distance"] <= MAX_COSINE_DISTANCE]
    matched = [r for r in candidates if r["condition_match"]]
    unmatched = [r for r in candidates if not r["condition_match"]]
    # Unmatched trials are only used to fill the list when lexical hits are scarce,
    # and only if they are semantically close.
    fill = [r for r in unmatched if r["distance"] <= MAX_COSINE_DISTANCE - 0.15]
    results = (matched + fill)[:k]

    logger.info(
        "Retrieval: conditions=%s pool=%d within_threshold=%d lexical=%d returned=%d",
        conditions, len(rows), len(candidates), len(matched), len(results),
    )
    return results


async def get_trials_by_ids(nct_ids: list[str]) -> list[dict[str, Any]]:
    if not nct_ids:
        return []
    rows = await db.fetch_all(
        f"SELECT {_TRIAL_COLUMNS}, NULL::float AS distance FROM trials WHERE nct_id = ANY(%s)",
        (nct_ids,),
    )
    order = {nid: i for i, nid in enumerate(nct_ids)}
    return sorted(rows, key=lambda r: order.get(r["nct_id"], 0))
