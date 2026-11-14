"""
backend/response_generator.py
─────────────────────────────
Turns the structured eligibility results into Aria's conversational reply.
Supports a full (non-streaming) reply and token streaming for SSE.
"""

import logging
from typing import Any, AsyncIterator

from openai import AsyncOpenAI

from backend.config import OPENAI_API_KEY, OPENAI_MODEL, fast_params

logger = logging.getLogger(__name__)

_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

# ─── System prompts ───────────────────────────────────────────────────────────

_RESPONSE_SYSTEM = """You are Aria, a warm, slightly playful clinical trial assistant who genuinely cares about the people you're helping.

TONE RULES (non-negotiable):
- Sound like a knowledgeable friend, not a medical report generator
- Be slightly playful but always reassuring — this is a healthcare context
- Vary your opening line every time (examples: "Alright, based on what you told me, here's what I found 👇", "Okay so I dug through the database and here's the honest picture 👀", "Hmm, this is actually interesting — I found a couple of options worth talking about 💙")
- Never start with "I"
- Never say "Trial NCTXXXX" as a header — that's robotic
- Never say "Eligibility Verdict: YES/NO/POSSIBLE" — that's robotic
- Always end with one warm, empathetic closing sentence

FORMAT RULES for each trial (follow this order, no exceptions):
1. START with a human meaning line using the right emoji:
   🟢 If eligibility is YES  → "This one actually looks like a solid match for you"
   🟡 If eligibility is POSSIBLE → "This one might work — there are a couple of things to double-check"
   🔴 If eligibility is NO   → "This one's probably not the right fit right now"
2. **Bold the trial name** on its own line (use the actual trial title, not the ID)
3. Bullet points — WHY it fits or doesn't fit, in plain English (max 4 bullets)
   - Reference specific things the patient told you ("You mentioned X, and this trial is looking for exactly that")
   - Reference what's confirmed vs uncertain
4. If a nearby site exists (same country or city as the patient), say so in one bullet.
5. Add a blank line between each trial block
6. Do NOT write phone numbers, emails or links — the app shows them on a card under your
   message. After the last trial, add one line: "Contact details and links for each study
   are in the cards below 👇"

WHAT TO AVOID:
❌ "Based on the provided eligibility criteria…" (too robotic)
❌ "The patient's age of X meets the inclusion criterion of…" (report language)
❌ Repeating the full eligibility criteria text verbatim
❌ Starting a trial block with the NCT ID
❌ Using the word "verdict"

DATA RULE:
Use ONLY the trial data and patient profile provided. Do not invent medical details.
Order: present 🟢 trials first, then 🟡, then 🔴.
Close by reminding them to discuss any trial with their doctor (one warm sentence)."""

_NO_RESULTS_SYSTEM = """You are Aria, a warm and caring clinical trial assistant.
No matching trials were found. Write a brief (3-4 sentence) empathetic message that:
- Acknowledges this without being dismissive
- Suggests 2-3 concrete next steps (broaden the search, speak to their doctor, check clinicaltrials.gov)
- Ends on an encouraging note
- Uses plain markdown (no excessive formatting)
- Never sounds like an error message"""


# ─── Public API ───────────────────────────────────────────────────────────────

def _messages(profile: dict[str, Any], results: list[dict[str, Any]]) -> list[dict]:
    return [
        {"role": "system", "content": _RESPONSE_SYSTEM},
        {"role": "user", "content": _build_response_prompt(profile, results)},
    ]


async def generate_response(
    patient_profile: dict[str, Any], trial_results: list[dict[str, Any]]
) -> str:
    """Complete reply as one string."""
    chunks = [c async for c in stream_response(patient_profile, trial_results)]
    return "".join(chunks)


async def stream_response(
    patient_profile: dict[str, Any], trial_results: list[dict[str, Any]]
) -> AsyncIterator[str]:
    """Yields reply text chunks (for SSE streaming)."""
    usable = [r for r in trial_results if r.get("eligibility") != "ERROR"]
    if not usable:
        if trial_results:  # every trial errored — don't pretend nothing exists
            yield _fallback_response(trial_results)
        else:
            yield await _generate_no_results(patient_profile)
        return

    try:
        stream = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            **fast_params(0.4),
            stream=True,
            messages=_messages(patient_profile, usable),
            timeout=45,
        )
        async for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as exc:  # noqa: BLE001
        logger.error("Response generation failed: %s", exc)
        yield _fallback_response(usable)


# ─── Internal helpers ─────────────────────────────────────────────────────────

def _build_response_prompt(profile: dict[str, Any], results: list[dict[str, Any]]) -> str:
    def join(key: str, default: str) -> str:
        return ", ".join(profile.get(key) or []) or default

    lines = [
        "=== PATIENT PROFILE ===",
        f"Age: {profile.get('age') or 'not given'}",
        f"Sex: {profile.get('sex') or 'not given'}",
        f"Conditions: {join('conditions', 'not specified')}",
        f"Symptoms: {join('symptoms', 'none mentioned')}",
        f"Medications: {join('medications', 'none mentioned')}",
        f"Medical history: {join('medical_history', 'none mentioned')}",
        f"Other facts: {'; '.join(profile.get('other_facts') or []) or 'none'}",
        "",
        f"=== TRIAL RESULTS ({len(results)}) ===",
        "",
    ]
    for i, r in enumerate(results, 1):
        lines += [
            f"--- Trial {i} ---",
            f"Title: {r.get('title')}",
            f"ID: {r.get('trial_id')}",
            f"URL: {r.get('url')}",
            f"Phase: {r.get('phase') or 'n/a'} | Status: {r.get('status') or 'n/a'}",
            f"What it studies: {r.get('summary') or 'n/a'}",
            f"Eligibility: {r.get('eligibility')}",
            "Screening notes:",
            *[f"- {x}" for x in r.get("reasons") or []],
            f"Contacts: {_fmt_contacts(r)}",
            f"Nearest sites: {_fmt_sites(r)}",
            "",
        ]
    lines += [
        f"Patient location: {profile.get('location') or 'not given'}",
        "=== YOUR TASK ===",
        "Write Aria's reply following the format rules exactly.",
    ]
    return "\n".join(lines)


def _fmt_contacts(r: dict[str, Any]) -> str:
    people = [
        " / ".join(x for x in (c.get("name"), c.get("phone"), c.get("email")) if x)
        for c in r.get("contacts") or []
    ]
    return "; ".join(p for p in people if p) or "none listed (use the ClinicalTrials.gov page)"


def _fmt_sites(r: dict[str, Any]) -> str:
    sites = [
        ", ".join(x for x in (s.get("facility"), s.get("city"), s.get("country")) if x)
        for s in r.get("sites") or []
    ]
    total = r.get("total_sites")
    more = f" (+{total - len(sites)} more sites)" if total and total > len(sites) else ""
    return ("; ".join(sites) + more) if sites else "not listed"


async def _generate_no_results(profile: dict[str, Any]) -> str:
    conditions = ", ".join(profile.get("conditions") or []) or "your condition"
    prompt = (
        f"No recruiting clinical trials matched: {conditions}. "
        f"Patient: age {profile.get('age', 'unknown')}, sex {profile.get('sex', 'unknown')}."
    )
    try:
        resp = await _client.chat.completions.create(
            model=OPENAI_MODEL,
            **fast_params(0.4),
            messages=[
                {"role": "system", "content": _NO_RESULTS_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            timeout=15,
        )
        return resp.choices[0].message.content.strip()
    except Exception:  # noqa: BLE001
        return (
            f"I couldn't find recruiting trials in my database that match **{conditions}** "
            "for your age and sex. That doesn't mean nothing exists:\n\n"
            "- Search [ClinicalTrials.gov](https://clinicaltrials.gov) with broader terms\n"
            "- Ask your doctor about trials they know of locally\n"
            "- Try describing your condition differently and I'll search again\n\n"
            "*Always talk to your physician before joining any clinical trial.*"
        )


def _fallback_response(results: list[dict[str, Any]]) -> str:
    """Deterministic reply used when the LLM is unavailable."""
    icon = {"YES": "🟢", "POSSIBLE": "🟡", "NO": "🔴"}
    meaning = {
        "YES": "This one looks like a solid match",
        "POSSIBLE": "This one might work — a few things to double-check",
        "NO": "This one's probably not the right fit",
    }
    lines = ["Here's what I found based on your profile:\n"]
    for r in results:
        e = r.get("eligibility", "POSSIBLE")
        lines.append(f"{icon.get(e, '⚪')} {meaning.get(e, 'Could not assess this one automatically')}")
        lines.append(f"**{r.get('title')}**")
        for reason in (r.get("reasons") or [])[:3]:
            lines.append(f"- {reason}")
        if e in ("YES", "POSSIBLE"):
            lines.append(f"📞 **How to reach them:** {_fmt_contacts(r)} — nearest site: {_fmt_sites(r)}")
        lines.append(f"[View {r['trial_id']} on ClinicalTrials.gov]({r.get('url')})\n")
    lines.append("*Please speak with your physician before pursuing any trial.*")
    return "\n".join(lines)
