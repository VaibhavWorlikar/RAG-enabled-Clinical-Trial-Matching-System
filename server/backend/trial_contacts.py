"""
backend/trial_contacts.py
─────────────────────────
Live contact details for the trials we show the patient.

Contacts and site lists change often and are bulky (some trials have hundreds
of sites), so instead of storing them for every trial we fetch them from the
ClinicalTrials.gov v2 API for the handful of trials actually shown, cache them
in-process for a few hours, and rank sites by closeness to the patient.
"""
import logging
import time
from typing import Any, Optional

import httpx

logger = logging.getLogger(__name__)

API_URL = "https://clinicaltrials.gov/api/v2/studies"
_TTL_SECONDS = 6 * 3600
_cache: dict[str, tuple[float, dict[str, Any]]] = {}

_SITE_STATUS_RANK = {"RECRUITING": 0, "NOT_YET_RECRUITING": 1, "ENROLLING_BY_INVITATION": 2}


def _person(c: dict[str, Any]) -> dict[str, Any]:
    phone = c.get("phone")
    if phone and c.get("phoneExt"):
        phone = f"{phone} ext. {c['phoneExt']}"
    return {k: v for k, v in {"name": c.get("name"), "role": c.get("role"),
                              "phone": phone, "email": c.get("email")}.items() if v}


def _parse(study: dict[str, Any]) -> dict[str, Any]:
    mod = (study.get("protocolSection") or {}).get("contactsLocationsModule") or {}
    sites = []
    for loc in mod.get("locations") or []:
        site = {
            "facility": loc.get("facility"),
            "city": loc.get("city"),
            "state": loc.get("state"),
            "country": loc.get("country"),
            "status": loc.get("status"),
        }
        contacts = [_person(c) for c in loc.get("contacts") or []]
        contacts = [c for c in contacts if c.get("phone") or c.get("email")]
        if contacts:
            site["contact"] = contacts[0]
        sites.append({k: v for k, v in site.items() if v})
    return {
        "central_contacts": [p for p in (_person(c) for c in mod.get("centralContacts") or []) if p][:2],
        "officials": [
            {k: v for k, v in {"name": o.get("name"), "affiliation": o.get("affiliation")}.items() if v}
            for o in (mod.get("overallOfficials") or [])[:1]
        ],
        "all_sites": sites,
    }


def rank_sites(sites: list[dict[str, Any]], location: Optional[str], limit: int = 3) -> list[dict[str, Any]]:
    """Order sites: same city > same state/region > same country, then recruiting status."""
    tokens = [t.strip().lower() for t in (location or "").split(",") if t.strip()]

    def score(site: dict[str, Any]) -> tuple:
        city = (site.get("city") or "").lower()
        state = (site.get("state") or "").lower()
        country = (site.get("country") or "").lower()
        proximity = 3
        if tokens:
            if city and city in tokens:
                proximity = 0
            elif state and state in tokens:
                proximity = 1
            elif country and country in tokens:
                proximity = 2
        return (proximity, _SITE_STATUS_RANK.get(site.get("status") or "", 9))

    return sorted(sites, key=score)[:limit]


async def fetch_contacts(nct_ids: list[str]) -> dict[str, dict[str, Any]]:
    """NCT ID → parsed contact info. Never raises; missing entries mean 'unavailable'."""
    now = time.time()
    out = {i: _cache[i][1] for i in nct_ids if i in _cache and now - _cache[i][0] < _TTL_SECONDS}
    missing = [i for i in nct_ids if i not in out]
    if not missing:
        return out
    try:
        async with httpx.AsyncClient(timeout=8, headers={"User-Agent": "clinical-trial-matcher/1.0"}) as client:
            resp = await client.get(API_URL, params={
                "filter.ids": ",".join(missing),
                "fields": "protocolSection.identificationModule.nctId,protocolSection.contactsLocationsModule",
                "pageSize": len(missing),
                "format": "json",
            })
            resp.raise_for_status()
        for study in resp.json().get("studies", []):
            nct = study["protocolSection"]["identificationModule"]["nctId"]
            parsed = _parse(study)
            _cache[nct] = (now, parsed)
            out[nct] = parsed
    except Exception as exc:  # noqa: BLE001 — contacts are an enhancement, never fatal
        logger.warning("Contact lookup failed for %s: %s", missing, exc)
    return out


async def attach_contacts(results: list[dict[str, Any]], location: Optional[str]) -> None:
    """Add `contacts`, `sites`, `total_sites` to each result in place (skips NO / ERROR verdicts)."""
    shown = [r for r in results if r.get("eligibility") in ("YES", "POSSIBLE")]
    if not shown:
        return
    info = await fetch_contacts([r["trial_id"] for r in shown])
    for r in shown:
        i = info.get(r["trial_id"])
        if not i:
            continue
        r["contacts"] = i["central_contacts"] or i["officials"]
        r["sites"] = rank_sites(i["all_sites"], location)
        r["total_sites"] = len(i["all_sites"])
