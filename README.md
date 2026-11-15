# Aria — AI Clinical Trial Matching

### 🔗 Live demo: **[aria-clinical-trials.vercel.app](https://aria-clinical-trials.vercel.app)**

[![Live demo](https://img.shields.io/badge/Live%20demo-aria--clinical--trials.vercel.app-0e7490?style=for-the-badge&logo=vercel&logoColor=white)](https://aria-clinical-trials.vercel.app)
[![CI](https://github.com/VaibhavWorlikar/RAG-enabled-Clinical-Trial-Matching-System/actions/workflows/ci.yml/badge.svg)](https://github.com/VaibhavWorlikar/RAG-enabled-Clinical-Trial-Matching-System/actions/workflows/ci.yml)

Aria is a conversational assistant that matches patients with **currently recruiting clinical trials** from ClinicalTrials.gov. Describe your situation in plain language — or upload a doctor's report — and Aria extracts a structured profile, retrieves candidate trials with hybrid vector + structured search, screens each trial's eligibility criteria with an LLM, asks targeted follow-up questions when a verdict depends on missing facts, and explains the results with **contact details and the nearest study sites** for every promising trial.

### Features

- 💬 **Plain-language chat** — no forms; age, sex, location, conditions, medications and other facts are extracted automatically
- 📎 **Medical report upload** — PDF, photo or text; the model reads it natively (no OCR step) and the file is never stored
- 🔎 **Hybrid retrieval** over 70,000 open trials (pgvector + SQL filters + condition matching)
- 🧠 **Parallel LLM eligibility screening** with structured verdicts: good / possible / probably not
- ❓ **"Ask before suggesting"** — targeted follow-up questions only when a verdict hinges on a missing fact
- 📞 **Trial contacts** — live contact names, tap-to-call phone, email and nearest sites from ClinicalTrials.gov
- ♿ **Patient-friendly UI** — readable typography, light/dark themes, mobile layout, progress feedback

> ⚠️ Research/educational project. Results are AI-assisted screening, not medical advice. Always consult a physician.

## How it works

```
 "I'm 34, male, my doctor thinks I have Crohn's disease"
        │
        ▼
 1. NLU extraction (gpt-6-luna, JSON) — from chat text or an uploaded report
    → {age: 34, sex: male, conditions: ["crohn's disease"], ...}
    → synonym normalisation (T2DM → type 2 diabetes mellitus, ...)
        │  missing age / sex / condition? → ask for it
        ▼
 2. Hybrid retrieval (Postgres + pgvector)
    • semantic: cosine distance on text-embedding-3-small (512-d) vectors
    • structured: age range, sex, recruiting status — filtered in SQL
    • lexical: condition/alias match ranks trials about *your* condition first
        ▼
 3. Eligibility screening — 8 candidates, one LLM call each, run in parallel,
    strict JSON schema: {eligibility: YES|NO|POSSIBLE, reasons[], missing_info[]}
        ▼
 4. "Ask before suggesting" — if POSSIBLE verdicts hinge on facts the
    patient hasn't given (pregnancy, prior treatment, labs, ...), Aria asks
    up to 4 questions once, then re-screens the same trials with the answers
        ▼
 5. Conversational summary streamed over SSE + trial cards with live
    contacts (phone / email), nearest sites and ClinicalTrials.gov links
```

## Tech stack

| Layer | Tech |
|---|---|
| Frontend | React 18 + Vite, SSE streaming |
| Backend | FastAPI (async), Pydantic v2 |
| Data | Neon Postgres + **pgvector** (trials, vectors, users, sessions, messages) |
| AI | OpenAI `text-embedding-3-small` (embeddings), `gpt-6-luna` (NLU, report reading, screening, replies) |
| Auth | JWT (PyJWT) + bcrypt, per-user session ownership, daily rate limit |
| Data pipeline | ClinicalTrials.gov JSON/CSV export or v2 API → parse → embed → upsert (incremental) |
| Infra | Vercel Services (frontend + Python backend in one deployment), GitHub Actions CI + weekly data refresh |

## Project structure

```
├── vercel.json              # Vercel Services: /api/* → backend, /* → frontend
├── frontend/                # React + Vite chat UI
├── server/
│   ├── main.py              # ASGI entrypoint (main:app)
│   ├── backend/
│   │   ├── api.py           # routes + chat orchestration
│   │   ├── nlu_extractor.py # free text → structured patient profile
│   │   ├── condition_normalizer.py
│   │   ├── rag_pipeline.py  # hybrid pgvector retrieval
│   │   ├── rag_reasoner.py  # parallel structured eligibility screening
│   │   ├── trial_gap_analyser.py  # follow-up question logic
│   │   ├── report_reader.py # uploaded report → patient facts
│   │   ├── trial_contacts.py # live contacts + nearest-site ranking
│   │   ├── response_generator.py  # Aria's streamed reply
│   │   ├── auth.py, session_manager.py, db.py, schema.sql
│   │   └── trial_parser.py, embeddings.py
│   ├── scripts/ingest_trials.py   # ClinicalTrials.gov → Postgres/pgvector
│   └── tests/
└── .github/workflows/       # CI + weekly trial refresh
```

## Running locally

Prerequisites: Python 3.12, Node 20+, a Postgres database with pgvector (a free Neon database works), an OpenAI API key.

```bash
# 1. Configure
cp .env.example .env          # fill in DATABASE_URL, OPENAI_API_KEY, JWT_SECRET_KEY

# 2. Backend
cd server
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt -r requirements-dev.txt
python -m scripts.ingest_trials --max 2000   # quick start from the live API
# or from a downloaded ClinicalTrials.gov export (most recently updated open trials):
python -m scripts.ingest_trials --from-json ~/Downloads/ctg-studies.json --max 70000 --all-types
uvicorn main:app --reload                    # http://127.0.0.1:8000/api/docs

# 3. Frontend (new terminal)
cd frontend
npm install
npm run dev                                   # http://localhost:3000 (proxies /api to :8000)
```

Run tests: `cd server && pytest -q`

## Deploying to Vercel

1. Import the repo in Vercel (or `vercel link`). `vercel.json` defines both services.
2. Add **Neon** from the Vercel Marketplace (Storage tab) — this sets `DATABASE_URL`.
3. Add env vars `OPENAI_API_KEY`, `JWT_SECRET_KEY`, and optionally `ADMIN_USERNAMES`.
4. Load data once: `vercel env pull .env && cd server && python -m scripts.ingest_trials`.
5. Deploy: `vercel --prod`.
6. (Optional) Add `DATABASE_URL` and `OPENAI_API_KEY` as GitHub repo secrets to enable the weekly refresh workflow.

## API

Interactive docs at `/api/docs`. Main endpoints:

| Method | Path | Description |
|---|---|---|
| POST | `/api/auth/register`, `/api/auth/login` | Create account / get JWT |
| POST | `/api/chat` | Conversational matching (`stream: true` for SSE) |
| POST | `/api/chat/upload` | Upload a medical report (multipart, ≤4 MB) and match |
| POST | `/api/match` | Structured matching (age, sex, conditions → screened trials) |
| GET/DELETE | `/api/chat/sessions[/{id}]` | Your conversation history |
| GET | `/api/health` | Health + number of indexed trials |
| GET | `/api/admin/stats` | Usage stats (admin only) |

## Design decisions

- **pgvector over a FAISS file** — serverless functions have no persistent disk; one Postgres database holds vectors, trials, users and sessions, and filters run in the same SQL query as the similarity search.
- **Exact vector scan, no HNSW index** — at ~20k trials an exact scan takes milliseconds and, unlike approximate search with filters, never drops valid candidates.
- **Hosted embeddings over Bio_ClinicalBERT** — removes a ~1.5 GB PyTorch bundle and 20–40 s cold starts.
- **Structured screening output** — the model returns `missing_info` as an enum, so follow-up questions are driven by data rather than regex over prose.
- **Live data** — trials come from the ClinicalTrials.gov API and are refreshed weekly; only changed trials are re-embedded.

## Limitations

- The hosted demo covers the 70,000 most recently updated open trials (fits Neon's free 0.5 GB tier); `--max` controls this.
- Responses take ~15–25 s because every candidate trial is screened by a reasoning model.
- Screening is only as good as the free-text eligibility criteria and the information the patient provides.
- No location-based ranking yet (countries are stored and shown).
