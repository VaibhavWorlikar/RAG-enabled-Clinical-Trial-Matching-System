"""
scripts/ingest_trials.py
────────────────────────
Load open clinical trials from the ClinicalTrials.gov v2 API into Postgres,
embed them with OpenAI, and store the vectors in pgvector.

Incremental: trials whose last-update date hasn't changed are not re-embedded.

Run from the server/ directory:
    python -m scripts.ingest_trials                 # default: 20k most recently updated
    python -m scripts.ingest_trials --max 5000      # smaller/faster
    python -m scripts.ingest_trials --prune         # also delete trials no longer open
    python -m scripts.ingest_trials --from-dir tests/data_sample   # offline, from JSON files
    python -m scripts.ingest_trials --from-csv ~/Downloads/ctg-studies.csv --max 20000
    python -m scripts.ingest_trials --from-json ~/Downloads/ctg-studies.json --max 0 --all-statuses

CSV mode: the ClinicalTrials.gov CSV export is used to SELECT trials (open,
interventional, most recently updated first). The export has no eligibility
criteria column and only coarse age groups, so the selected trials' full
records are then fetched from the v2 API by NCT ID.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any, Iterator, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import httpx  # noqa: E402
import psycopg  # noqa: E402

from backend.config import DATABASE_URL  # noqa: E402
from backend.db import SCHEMA_PATH, to_vector_literal  # noqa: E402
from backend.embeddings import embed_batch  # noqa: E402
from backend.trial_parser import embedding_text, parse_study  # noqa: E402

API_URL = "https://clinicaltrials.gov/api/v2/studies"
OPEN_STATUSES = "RECRUITING,NOT_YET_RECRUITING,ENROLLING_BY_INVITATION"
FIELDS = ",".join(
    f"protocolSection.{m}"
    for m in (
        "identificationModule",
        "statusModule",
        "descriptionModule",
        "conditionsModule",
        "designModule",
        "eligibilityModule",
        "armsInterventionsModule",
        "contactsLocationsModule",
    )
)
EMBED_BATCH = 100

UPSERT_SQL = """
INSERT INTO trials (nct_id, title, brief_summary, conditions, keywords, interventions,
                    study_type, phase, status, sex, min_age_years, max_age_years,
                    healthy_volunteers, criteria, countries, last_updated, embedding, ingested_at)
VALUES (%(nct_id)s, %(title)s, %(brief_summary)s, %(conditions)s, %(keywords)s, %(interventions)s,
        %(study_type)s, %(phase)s, %(status)s, %(sex)s, %(min_age_years)s, %(max_age_years)s,
        %(healthy_volunteers)s, %(criteria)s, %(countries)s, %(last_updated)s,
        %(embedding)s::vector, now())
ON CONFLICT (nct_id) DO UPDATE SET
    title = EXCLUDED.title, brief_summary = EXCLUDED.brief_summary,
    conditions = EXCLUDED.conditions, keywords = EXCLUDED.keywords,
    interventions = EXCLUDED.interventions, study_type = EXCLUDED.study_type,
    phase = EXCLUDED.phase, status = EXCLUDED.status, sex = EXCLUDED.sex,
    min_age_years = EXCLUDED.min_age_years, max_age_years = EXCLUDED.max_age_years,
    healthy_volunteers = EXCLUDED.healthy_volunteers, criteria = EXCLUDED.criteria,
    countries = EXCLUDED.countries, last_updated = EXCLUDED.last_updated,
    embedding = COALESCE(EXCLUDED.embedding, trials.embedding), ingested_at = now()
"""


def fetch_from_api(max_trials: int, interventional_only: bool) -> Iterator[dict[str, Any]]:
    params: dict[str, Any] = {
        "filter.overallStatus": OPEN_STATUSES,
        "sort": "LastUpdatePostDate:desc",
        "pageSize": 1000,
        "fields": FIELDS,
        "format": "json",
    }
    if interventional_only:
        params["filter.advanced"] = "AREA[StudyType]INTERVENTIONAL"
    seen = 0
    with httpx.Client(timeout=60, headers={"User-Agent": "clinical-trial-matcher/1.0"}) as client:
        token: Optional[str] = None
        while seen < max_trials:
            if token:
                params["pageToken"] = token
            for attempt in range(4):
                resp = client.get(API_URL, params=params)
                if resp.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            resp.raise_for_status()
            data = resp.json()
            for study in data.get("studies", []):
                yield study
                seen += 1
                if seen >= max_trials:
                    return
            token = data.get("nextPageToken")
            if not token:
                return


def select_from_csv(path: str, max_trials: int, interventional_only: bool) -> list[str]:
    """Stream the (multi-GB) CSV export and return the NCT IDs to ingest."""
    csv.field_size_limit(sys.maxsize)
    open_statuses = set(OPEN_STATUSES.split(","))
    picked: list[tuple[str, str]] = []
    total = 0
    with open(Path(path).expanduser(), newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            total += 1
            if row.get("Study Status") not in open_statuses:
                continue
            if interventional_only and row.get("Study Type") != "INTERVENTIONAL":
                continue
            if row.get("NCT Number", "").startswith("NCT"):
                picked.append((row.get("Last Update Posted") or "", row["NCT Number"]))
    picked.sort(reverse=True)  # most recently updated first
    print(f"CSV: {total:,} studies, {len(picked):,} match filters, taking {min(max_trials, len(picked)):,}.")
    return [nct for _, nct in picked[:max_trials]]


def fetch_by_ids(nct_ids: list[str], chunk: int = 200) -> Iterator[dict[str, Any]]:
    with httpx.Client(timeout=60, headers={"User-Agent": "clinical-trial-matcher/1.0"}) as client:
        for i in range(0, len(nct_ids), chunk):
            ids = nct_ids[i : i + chunk]
            params = {"filter.ids": ",".join(ids), "fields": FIELDS, "pageSize": chunk, "format": "json"}
            for attempt in range(4):
                resp = client.get(API_URL, params=params)
                if resp.status_code == 200:
                    break
                time.sleep(2 ** attempt)
            resp.raise_for_status()
            yield from resp.json().get("studies", [])


def _json_lines(path: str) -> Iterator[dict[str, Any]]:
    """Stream a ClinicalTrials.gov JSON export (one study per line, ~10 GB)."""
    with open(Path(path).expanduser(), encoding="utf-8") as f:
        for line in f:
            line = line.strip().rstrip(",")
            if line.startswith("{"):
                yield json.loads(line)


def fetch_from_json(
    path: str, max_trials: int, interventional_only: bool, open_only: bool
) -> Iterator[dict[str, Any]]:
    """
    Two streaming passes, constant memory:
      1. collect (last-update date, NCT ID) of every study passing the filters
      2. yield the `max_trials` most recently updated ones (0 = all)
    """
    open_statuses = set(OPEN_STATUSES.split(","))

    def keep(study: dict[str, Any]) -> bool:
        proto = study.get("protocolSection") or {}
        status = (proto.get("statusModule") or {}).get("overallStatus")
        stype = (proto.get("designModule") or {}).get("studyType")
        return (not open_only or status in open_statuses) and (
            not interventional_only or stype == "INTERVENTIONAL"
        )

    candidates: list[tuple[str, str]] = []
    for study in _json_lines(path):
        if keep(study):
            proto = study["protocolSection"]
            updated = ((proto.get("statusModule") or {}).get("lastUpdatePostDateStruct") or {}).get("date") or ""
            candidates.append((updated, proto["identificationModule"]["nctId"]))
    candidates.sort(reverse=True)
    chosen = {nct for _, nct in (candidates[:max_trials] if max_trials else candidates)}
    print(f"JSON: {len(candidates):,} studies match filters, ingesting {len(chosen):,}.", flush=True)

    for study in _json_lines(path):
        nct = ((study.get("protocolSection") or {}).get("identificationModule") or {}).get("nctId")
        if nct in chosen:
            yield study


def fetch_from_dir(path: str) -> Iterator[dict[str, Any]]:
    for f in sorted(Path(path).glob("*.json")):
        yield json.loads(f.read_text())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max", type=int, default=20000, help="max trials to ingest (0 = no limit, JSON mode)")
    ap.add_argument("--all-types", action="store_true", help="include observational studies")
    ap.add_argument("--prune", action="store_true", help="delete trials not returned by this run")
    ap.add_argument("--from-dir", help="ingest ClinicalTrials.gov JSON files from a directory")
    ap.add_argument("--from-csv", help="select trials from a ClinicalTrials.gov CSV export")
    ap.add_argument("--from-json", help="ingest a full ClinicalTrials.gov JSON export (streamed)")
    ap.add_argument("--all-statuses", action="store_true",
                    help="JSON mode: include completed/terminated/etc. trials, not just open ones")
    ap.add_argument("--reembed", action="store_true", help="re-embed even unchanged trials")
    args = ap.parse_args()

    if not DATABASE_URL:
        sys.exit("DATABASE_URL is not set (put it in .env or run `vercel env pull`).")

    with psycopg.connect(DATABASE_URL, prepare_threshold=None, autocommit=True) as conn:
        conn.execute(SCHEMA_PATH.read_text())
        known = {
            r[0]: r[1]
            for r in conn.execute(
                "SELECT nct_id, last_updated FROM trials WHERE embedding IS NOT NULL"
            ).fetchall()
        }
        print(f"DB has {len(known):,} embedded trials.")

        if args.from_dir:
            source = fetch_from_dir(args.from_dir)
        elif args.from_json:
            source = fetch_from_json(
                args.from_json, args.max, not args.all_types, not args.all_statuses
            )
        elif args.from_csv:
            source = fetch_by_ids(select_from_csv(args.from_csv, args.max, not args.all_types))
        else:
            source = fetch_from_api(args.max, interventional_only=not args.all_types)

        seen_ids: list[str] = []
        batch: list[dict[str, Any]] = []
        stats = {"fetched": 0, "skipped": 0, "embedded": 0, "unchanged": 0}
        started = time.time()

        def flush() -> None:
            if not batch:
                return
            to_embed = [r for r in batch if r.pop("_needs_embed")]
            if to_embed:
                vectors = embed_batch([embedding_text(r) for r in to_embed])
                for r, v in zip(to_embed, vectors):
                    r["embedding"] = to_vector_literal(v)
                stats["embedded"] += len(to_embed)
            with conn.cursor() as cur:
                cur.executemany(UPSERT_SQL, batch)
            batch.clear()
            rate = stats["fetched"] / max(time.time() - started, 1e-6)
            print(
                f"  fetched {stats['fetched']:,} | embedded {stats['embedded']:,} | "
                f"unchanged {stats['unchanged']:,} | skipped {stats['skipped']:,} | {rate:.0f}/s",
                flush=True,
            )

        for study in source:
            stats["fetched"] += 1
            row = parse_study(study)
            if not row:
                stats["skipped"] += 1
                continue
            seen_ids.append(row["nct_id"])
            last = row["last_updated"]
            unchanged = (
                not args.reembed
                and row["nct_id"] in known
                and last is not None
                and str(known[row["nct_id"]]) == last
            )
            stats["unchanged"] += unchanged
            row["embedding"] = None  # COALESCE keeps the stored vector when unchanged
            row["_needs_embed"] = not unchanged
            batch.append(row)
            if len(batch) >= EMBED_BATCH:
                flush()
        flush()

        if args.prune and not (args.from_dir or args.from_json) and seen_ids:
            deleted = conn.execute(
                "DELETE FROM trials WHERE NOT (nct_id = ANY(%s))", (seen_ids,)
            ).rowcount
            print(f"Pruned {deleted:,} trials that are no longer open.")

        total = conn.execute("SELECT count(*) FROM trials").fetchone()[0]
        print(f"Done in {time.time() - started:.0f}s. {stats}. Trials in DB: {total:,}")


if __name__ == "__main__":
    main()
