-- Clinical trial matching — database schema (Postgres 15+ with pgvector).
-- Idempotent: safe to run on every deploy / ingest.

CREATE EXTENSION IF NOT EXISTS vector;

-- ─── Trials (loaded from the ClinicalTrials.gov v2 API) ──────────────────────
CREATE TABLE IF NOT EXISTS trials (
    nct_id             TEXT PRIMARY KEY,
    title              TEXT NOT NULL,
    brief_summary      TEXT,
    conditions         TEXT[] NOT NULL DEFAULT '{}',
    keywords           TEXT[] NOT NULL DEFAULT '{}',
    interventions      TEXT[] NOT NULL DEFAULT '{}',
    study_type         TEXT,
    phase              TEXT,
    status             TEXT,
    sex                TEXT,               -- ALL | FEMALE | MALE
    min_age_years      REAL,               -- NULL = no lower bound
    max_age_years      REAL,               -- NULL = no upper bound
    healthy_volunteers BOOLEAN,
    criteria           TEXT,
    countries          TEXT[] NOT NULL DEFAULT '{}',
    last_updated       DATE,
    embedding          vector(512),
    ingested_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Vector search is an exact scan: at ~20k rows it is a few ms and, unlike an
-- approximate HNSW scan, never silently drops rows that pass the age/sex filters.
CREATE INDEX IF NOT EXISTS trials_status_idx ON trials (status);

-- ─── Users ───────────────────────────────────────────────────────────────────
CREATE TABLE IF NOT EXISTS users (
    id          BIGSERIAL PRIMARY KEY,
    username    TEXT NOT NULL UNIQUE,
    hashed_pw   TEXT NOT NULL,
    role        TEXT NOT NULL DEFAULT 'patient' CHECK (role IN ('patient', 'doctor', 'admin')),
    disabled    BOOLEAN NOT NULL DEFAULT FALSE,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- ─── Chat sessions (owned by a user) ─────────────────────────────────────────
CREATE TABLE IF NOT EXISTS chat_sessions (
    id              UUID PRIMARY KEY,
    user_id         BIGINT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    patient_profile JSONB NOT NULL DEFAULT '{}',
    asked_gaps      TEXT[] NOT NULL DEFAULT '{}',
    pending_trials  TEXT[] NOT NULL DEFAULT '{}',   -- NCT IDs held while asking follow-ups
    search_count    INT NOT NULL DEFAULT 0,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_sessions_user_idx ON chat_sessions (user_id, updated_at DESC);

CREATE TABLE IF NOT EXISTS chat_messages (
    id          BIGSERIAL PRIMARY KEY,
    session_id  UUID NOT NULL REFERENCES chat_sessions(id) ON DELETE CASCADE,
    role        TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
    content     TEXT NOT NULL,
    metadata    JSONB,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS chat_messages_session_idx ON chat_messages (session_id, id);
