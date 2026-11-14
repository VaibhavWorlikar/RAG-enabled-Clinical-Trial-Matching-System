"""
backend/config.py
─────────────────
Single source of truth for every configurable value.
Secrets come from environment variables (.env locally, Vercel env vars in prod).
"""
import os

from dotenv import load_dotenv

# Load server/.env first, then the repo-root .env (used by `vercel env pull`).
_HERE = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(_HERE, "..", ".env"))
load_dotenv(os.path.join(_HERE, "..", "..", ".env"))
load_dotenv(os.path.join(_HERE, "..", "..", ".env.local"))

IS_PRODUCTION = os.getenv("VERCEL_ENV") in ("production", "preview")

# ─── Database (Neon Postgres + pgvector) ──────────────────────────────────────
DATABASE_URL = os.getenv("DATABASE_URL", "")

# ─── JWT ──────────────────────────────────────────────────────────────────────
JWT_SECRET_KEY = os.getenv("JWT_SECRET_KEY", "")
if not JWT_SECRET_KEY:
    if IS_PRODUCTION:
        raise RuntimeError("JWT_SECRET_KEY must be set in production.")
    JWT_SECRET_KEY = "dev-only-insecure-secret"
JWT_ALGORITHM = "HS256"
JWT_ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "720"))

# Comma-separated usernames that are granted the admin role on registration.
ADMIN_USERNAMES = {
    u.strip().lower() for u in os.getenv("ADMIN_USERNAMES", "").split(",") if u.strip()
}

# ─── OpenAI ───────────────────────────────────────────────────────────────────
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-6-luna")


def sampling_params(temperature: float) -> dict:
    """
    Newer OpenAI reasoning models (gpt-5.x, gpt-6.x, o-series) accept only the
    default temperature and return HTTP 400 for any other value, so the
    parameter is sent only to models that support it.
    """
    return {} if is_reasoning_model() else {"temperature": temperature}


def is_reasoning_model() -> bool:
    return OPENAI_MODEL.startswith(("gpt-5", "gpt-6", "o1", "o3", "o4"))


def fast_params(temperature: float) -> dict:
    """For simple steps (profile extraction, writing the reply): low reasoning effort."""
    return {"reasoning_effort": "low"} if is_reasoning_model() else {"temperature": temperature}
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
# Must match the vector(N) column in schema.sql.
EMBEDDING_DIMENSIONS = 512

# ─── RAG knobs ────────────────────────────────────────────────────────────────
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "5"))       # trials shown to the patient
SCREEN_K = int(os.getenv("SCREEN_K", "8"))             # candidates screened by the LLM
CANDIDATE_POOL = int(os.getenv("CANDIDATE_POOL", "60"))
# Cosine distance (0 = identical, 2 = opposite). Candidates above this are dropped.
MAX_COSINE_DISTANCE = float(os.getenv("MAX_COSINE_DISTANCE", "0.55"))

# ─── Abuse / cost protection ──────────────────────────────────────────────────
DAILY_MESSAGE_LIMIT = int(os.getenv("DAILY_MESSAGE_LIMIT", "40"))

# ─── App ──────────────────────────────────────────────────────────────────────
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
CORS_ORIGINS = [
    o.strip()
    for o in os.getenv("CORS_ORIGINS", "http://localhost:3000,http://127.0.0.1:3000").split(",")
    if o.strip()
]
