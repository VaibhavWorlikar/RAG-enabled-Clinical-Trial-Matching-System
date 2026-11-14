"""
backend/rag_reasoner.py
───────────────────────
LLM eligibility reasoning over retrieved trials.

• One call per trial, run concurrently (bounded by a semaphore).
• Structured output (JSON schema) — the model returns a verdict, short
  reasons, AND a machine-readable list of missing patient facts. The gap
  analyser consumes that list directly instead of regex-scanning prose.
• Grounded: the model sees only the patient profile and the trial's own
  eligibility criteria.
"""
import asyncio
import json
import logging
from typing import Any

from openai import APIConnectionError, AsyncOpenAI, RateLimitError

from backend.config import OPENAI_API_KEY, OPENAI_MODEL, sampling_params
from backend.trial_gap_analyser import GAP_FIELDS

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_CONCURRENCY = 10
_MAX_CRITERIA_CHARS = 6000

_SYSTEM_PROMPT = f"""You are a clinical trial eligibility screener.

Compare the patient profile against the trial's eligibility criteria and return JSON.

RULES
1. Use ONLY the eligibility criteria text and the patient profile. No outside knowledge.
2. eligibility:
   - "YES": the patient's stated condition fits the inclusion criteria and nothing stated
     in the profile violates an exclusion criterion.
   - "NO": a stated fact (condition, age, sex, medication, history) clearly violates an
     inclusion or exclusion criterion, or the trial is for a different condition.
   - "POSSIBLE": the condition fits, but deciding requires facts the profile does not state.
3. reasons: 2-4 short plain-English bullets about what matches or conflicts. Never present
   an assumption as a fact.
4. missing_info: the criteria-relevant facts the profile does NOT state, chosen only from
   this list: {", ".join(GAP_FIELDS)}. Include an item only if the criteria actually depend
   on it. Do NOT list a field the profile already addresses, even partially (e.g. severity
   is described but not the trial's specific score) — name that specific missing measure in
   reasons instead. Empty list if nothing important is missing.
5. summary: one sentence describing what the trial is studying."""

_RESPONSE_FORMAT = {
    "type": "json_schema",
    "json_schema": {
        "name": "eligibility_assessment",
        "strict": True,
        "schema": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "eligibility": {"type": "string", "enum": ["YES", "NO", "POSSIBLE"]},
                "summary": {"type": "string"},
                "reasons": {"type": "array", "items": {"type": "string"}},
                "missing_info": {
                    "type": "array",
                    "items": {"type": "string", "enum": list(GAP_FIELDS)},
                },
            },
            "required": ["eligibility", "summary", "reasons", "missing_info"],
        },
    },
}


def _truncate(text: str, max_chars: int = _MAX_CRITERIA_CHARS) -> str:
    text = text or ""
    return text if len(text) <= max_chars else text[:max_chars] + "\n[...criteria truncated...]"


def _base_result(trial: dict[str, Any]) -> dict[str, Any]:
    distance = trial.get("distance")
    return {
        "trial_id": trial["nct_id"],
        "title": trial.get("title") or trial["nct_id"],
        "phase": trial.get("phase"),
        "status": trial.get("status"),
        "countries": list(trial.get("countries") or [])[:6],
        "url": f"https://clinicaltrials.gov/study/{trial['nct_id']}",
        "distance": round(float(distance), 4) if distance is not None else None,
    }


async def _reason_one(
    patient_text: str, trial: dict[str, Any], sem: asyncio.Semaphore
) -> dict[str, Any]:
    result = _base_result(trial)
    user_prompt = (
        f"{patient_text.strip()}\n\n"
        f"Trial ID: {trial['nct_id']}\n"
        f"Trial title: {trial.get('title')}\n"
        f"Trial conditions: {', '.join(trial.get('conditions') or [])}\n\n"
        f"Eligibility criteria:\n{_truncate(trial.get('criteria') or '')}"
    )
    try:
        async with sem:
            resp = await _client.chat.completions.create(
                model=OPENAI_MODEL,
                **sampling_params(0),
                response_format=_RESPONSE_FORMAT,
                messages=[
                    {"role": "system", "content": _SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                timeout=30,
            )
        data = json.loads(resp.choices[0].message.content)
        reasons = [r.strip() for r in data.get("reasons", []) if r.strip()][:4]
        result.update(
            eligibility=data["eligibility"],
            summary=data.get("summary", ""),
            reasons=reasons,
            missing_info=[m for m in data.get("missing_info", []) if m in GAP_FIELDS],
            analysis=f"Eligibility: {data['eligibility']}\n\nReasoning:\n"
            + "\n".join(f"- {r}" for r in reasons),
        )
    except RateLimitError:
        logger.error("OpenAI rate limit for trial %s", trial["nct_id"])
        result.update(_error("The AI service is busy right now. Please try again shortly."))
    except APIConnectionError:
        logger.error("OpenAI connection error for trial %s", trial["nct_id"])
        result.update(_error("Could not reach the AI service."))
    except Exception as exc:  # noqa: BLE001 — one bad trial must not sink the batch
        logger.exception("Reasoning failed for trial %s: %s", trial["nct_id"], exc)
        result.update(_error("Eligibility analysis failed for this trial."))
    return result


def _error(message: str) -> dict[str, Any]:
    return {
        "eligibility": "ERROR",
        "summary": "",
        "reasons": [],
        "missing_info": [],
        "analysis": message,
    }


async def rag_reason(patient_text: str, trials: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Assess every trial concurrently. Order of results matches `trials`."""
    trials = [t for t in trials if (t.get("criteria") or "").strip()]
    if not trials:
        return []
    sem = asyncio.Semaphore(_CONCURRENCY)  # per call: bound to the running event loop
    return list(await asyncio.gather(*(_reason_one(patient_text, t, sem) for t in trials)))
