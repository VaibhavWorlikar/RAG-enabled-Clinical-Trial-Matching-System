"""
backend/api.py
──────────────
FastAPI app. Every route lives under /api so the same paths work locally
(behind the Vite dev proxy) and on Vercel (where /api/* is routed to this
service).

/api/chat flow
  1. NLU      — extract a structured profile from the message, merge it into
                the session's profile.
  2. Gate 1   — age, sex or condition missing → ask for it.
  3. Gate 2   — trials held back last turn while asking follow-ups → re-assess
                them with the enriched profile and show them.
  4. Retrieve — pgvector semantic search + age/sex/status SQL filters +
                condition-match ranking.
  5. Reason   — LLM eligibility check per trial (parallel, structured JSON).
  6. Gate 3   — a POSSIBLE verdict hinges on facts we haven't asked about →
                hold the trials, ask up to 4 follow-up questions (once).
  7. Respond  — Aria's conversational summary, streamed over SSE.
"""
import json
import logging
from datetime import timedelta
from typing import Any, AsyncIterator, Optional

from fastapi import APIRouter, Depends, FastAPI, File, Form, HTTPException, Query, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel, Field, field_validator

from backend import db
from backend import session_manager as sessions
from backend.auth import (
    SELF_SERVICE_ROLES,
    User,
    authenticate_user,
    create_access_token,
    create_user,
    get_current_user,
    require_role,
)
from backend.config import (
    CORS_ORIGINS,
    DAILY_MESSAGE_LIMIT,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    RETRIEVAL_K,
    SCREEN_K,
)
from backend.logger import logging_middleware, setup_logging
from backend.nlu_extractor import build_patient_text, extract_patient_profile, merge_profiles
from backend.rag_pipeline import get_trials_by_ids, retrieve_trials
from backend.rag_reasoner import rag_reason
from backend.report_reader import ReportError, read_report
from backend.trial_contacts import attach_contacts
from backend.response_generator import generate_response, stream_response
from backend.trial_gap_analyser import (
    build_followup_message,
    extract_gaps,
    gaps_to_questions,
    should_ask_before_showing,
)

setup_logging()
logger = logging.getLogger(__name__)

API_VERSION = "5.0.0"

app = FastAPI(
    title="Aria — Clinical Trial Matching API",
    description="Conversational clinical trial matching with hybrid retrieval and LLM eligibility screening.",
    version=API_VERSION,
    docs_url="/api/docs",
    openapi_url="/api/openapi.json",
    redoc_url=None,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=CORS_ORIGINS,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.middleware("http")(logging_middleware)

router = APIRouter(prefix="/api")


@app.exception_handler(RuntimeError)
async def _runtime_error(_: Request, exc: RuntimeError) -> JSONResponse:
    logger.error("Service unavailable: %s", exc)
    return JSONResponse(status_code=503, content={"detail": "Service temporarily unavailable."})


# ─── Schemas ──────────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    username: str = Field(..., min_length=3, max_length=40, pattern=r"^[A-Za-z0-9_.-]+$")
    password: str = Field(..., min_length=8, max_length=128)
    role: str = Field(default="patient")

    @field_validator("role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        if v not in SELF_SERVICE_ROLES:
            raise ValueError("role must be 'patient' or 'doctor'")
        return v


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    username: str
    role: str


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    stream: bool = False


class TrialCard(BaseModel):
    trial_id: str
    title: str
    eligibility: str
    url: str
    phase: Optional[str] = None
    status: Optional[str] = None
    summary: str = ""
    reasons: list[str] = []
    countries: list[str] = []
    contacts: list[dict] = []
    sites: list[dict] = []
    total_sites: Optional[int] = None
    distance: Optional[float] = None


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    patient_profile: dict
    trials_found: int
    trials: list[TrialCard] = []
    search_performed: bool
    clarification_needed: bool
    awaiting_gap_answers: bool = False


class PatientQuery(BaseModel):
    age: int = Field(..., ge=0, le=120)
    sex: str
    conditions: list[str] = Field(..., min_length=1, max_length=10)
    medications: list[str] = []
    other_facts: list[str] = []
    k: int = Field(default=5, ge=1, le=10)

    @field_validator("sex")
    @classmethod
    def validate_sex(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in ("male", "female"):
            raise ValueError("sex must be 'male' or 'female'")
        return v


# ─── Auth & health ────────────────────────────────────────────────────────────

@router.post("/auth/register", status_code=201, response_model=TokenResponse)
async def register(req: RegisterRequest):
    try:
        user = await create_user(req.username, req.password, req.role)
    except ValueError as exc:
        raise HTTPException(400, detail=str(exc))
    return _token_for(user)


@router.post("/auth/login", response_model=TokenResponse)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = await authenticate_user(form.username, form.password)
    if not user:
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return _token_for(user)


@router.get("/auth/me")
async def me(user: User = Depends(get_current_user)):
    return {"username": user.username, "role": user.role}


@router.get("/health")
async def health():
    try:
        row = await db.fetch_one("SELECT count(*) AS n FROM trials")
        return {"status": "ok", "version": API_VERSION, "trials_indexed": row["n"]}
    except Exception as exc:  # noqa: BLE001
        logger.error("Health check DB error: %s", exc)
        return JSONResponse(
            status_code=503, content={"status": "degraded", "version": API_VERSION}
        )


def _token_for(user: User) -> TokenResponse:
    token = create_access_token(user, timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES))
    return TokenResponse(access_token=token, username=user.username, role=user.role)


# ─── Conversational endpoint ──────────────────────────────────────────────────

async def _check_daily_limit(user: User) -> None:
    if await sessions.count_user_messages_today(user.id) >= DAILY_MESSAGE_LIMIT:
        raise HTTPException(
            429, detail=f"Daily limit of {DAILY_MESSAGE_LIMIT} messages reached. Try again tomorrow."
        )


@router.post("/chat")
async def chat(req: ChatRequest, user: User = Depends(get_current_user)):
    await _check_daily_limit(user)
    session = await sessions.get_or_create_session(req.session_id, user.id)
    history = await sessions.get_chat_history(str(session["id"]), last_n=8)
    await sessions.add_message(str(session["id"]), "user", req.message)

    existing = session.get("patient_profile") or {}
    extracted = await extract_patient_profile(
        req.message,
        [{"role": m["role"], "content": m["content"]} for m in history],
        current_profile=existing,
    )
    profile = merge_profiles(existing, extracted, replace_lists=bool(existing))
    return await _run_pipeline(req.stream, session, existing, extracted, profile)


@router.post("/chat/upload")
async def upload_report(
    file: UploadFile = File(...),
    session_id: Optional[str] = Form(default=None),
    note: str = Form(default="", max_length=1000),
    stream: bool = Form(default=False),
    user: User = Depends(get_current_user),
):
    """
    Upload a medical report (PDF / image / text, max 4 MB). The AI extracts the
    eligibility-relevant facts, merges them into the patient profile, and runs
    the same matching pipeline as /chat. The file itself is never stored.
    """
    await _check_daily_limit(user)
    data = await file.read(4 * 1024 * 1024 + 1)
    try:
        report = await read_report(data, file.filename or "report")
    except ReportError as exc:
        raise HTTPException(400, detail=str(exc))

    session = await sessions.get_or_create_session(session_id, user.id)
    summary = "\n".join(f"- {line}" for line in report["report_summary"])
    message = f"📄 Uploaded {report.get('document_type') or 'medical report'} ({file.filename}):\n{summary}"
    if note.strip():
        message += f"\n\n{note.strip()}"
    await sessions.add_message(str(session["id"]), "user", message)

    existing = session.get("patient_profile") or {}
    extracted = {k: report.get(k) for k in _PROFILE_KEYS if k in report}
    if note.strip():
        from_note = await extract_patient_profile(note, current_profile=None)
        extracted = merge_profiles(extracted, from_note)
    # Report facts are added to (never replace) what the patient already told us.
    profile = merge_profiles(existing, extracted)
    return await _run_pipeline(stream, session, existing, extracted, profile)


async def _run_pipeline(
    stream: bool,
    session: dict[str, Any],
    existing: dict[str, Any],
    extracted: dict[str, Any],
    profile: dict[str, Any],
):
    session_id = str(session["id"])
    new_conditions = set(extracted.get("conditions") or []) - set(existing.get("conditions") or [])
    await sessions.update_session(session_id, patient_profile=_stored_profile(profile))

    # ── Gate 1: basics missing ───────────────────────────────────────────────
    if profile.get("clarification_needed"):
        reply = profile.get("clarification_question") or (
            "Could you tell me which condition you're looking for trials for?"
        )
        return await _reply(
            stream, session_id, profile, reply, [], search_performed=False, clarification=True
        )

    patient_text = build_patient_text(profile)
    asked_gaps = list(session.get("asked_gaps") or [])
    pending_ids = list(session.get("pending_trials") or [])

    # ── Gate 2: user just answered follow-up questions ───────────────────────
    if pending_ids and not new_conditions:
        await sessions.update_session(
            session_id, pending_trials=[], search_count=session["search_count"] + 1
        )
        trials = await get_trials_by_ids(pending_ids)
        results = await rag_reason(patient_text, trials)
        return await _emit_results(stream, session_id, profile, results)

    # ── Retrieve + reason ────────────────────────────────────────────────────
    trials = await retrieve_trials(profile, k=SCREEN_K)
    results = _best(await rag_reason(patient_text, trials))
    await sessions.update_session(
        session_id, pending_trials=[], search_count=session["search_count"] + 1
    )

    # ── Gate 3: ask before showing? ──────────────────────────────────────────
    gap_fields = extract_gaps(results)
    if should_ask_before_showing(results, gap_fields, asked_gaps, profile.get("sex"), profile):
        fields, questions = gaps_to_questions(gap_fields, asked_gaps, profile.get("sex"), profile)
        relevant = [r["trial_id"] for r in results if r["eligibility"] != "NO"]
        await sessions.update_session(
            session_id,
            pending_trials=relevant or [r["trial_id"] for r in results],
            asked_gaps=list(dict.fromkeys(asked_gaps + fields)),
        )
        reply = build_followup_message(questions, len(relevant) or len(results))
        return await _reply(
            stream, session_id, profile, reply, [], search_performed=True,
            awaiting_gap_answers=True, trials_found=len(results),
        )

    return await _emit_results(stream, session_id, profile, results)


# ─── Structured (non-conversational) endpoint ─────────────────────────────────

@router.post("/match", response_model=list[TrialCard])
async def match(patient: PatientQuery, user: User = Depends(get_current_user)):
    """Form-style matching: structured profile in, screened trials out."""
    from backend.condition_normalizer import normalize_conditions

    profile = {
        "age": patient.age,
        "sex": patient.sex,
        "conditions": normalize_conditions(", ".join(patient.conditions)),
        "medications": patient.medications,
        "other_facts": patient.other_facts,
    }
    trials = await retrieve_trials(profile, k=patient.k)
    results = await rag_reason(build_patient_text(profile), trials)
    await attach_contacts(results, None)
    return [TrialCard(**_card(r)) for r in results]


# ─── Session endpoints (always scoped to the current user) ───────────────────

@router.get("/chat/sessions")
async def list_sessions(user: User = Depends(get_current_user)):
    rows = await sessions.list_sessions(user.id)
    return [
        {
            "session_id": str(r["id"]),
            "title": (r["first_message"] or "New conversation")[:80],
            "conditions": r["conditions"] or [],
            "search_count": r["search_count"],
            "updated_at": r["updated_at"].isoformat(),
        }
        for r in rows
    ]


@router.get("/chat/sessions/{session_id}")
async def get_session_detail(
    session_id: str,
    last_n: int = Query(default=40, ge=1, le=100),
    user: User = Depends(get_current_user),
):
    session = await sessions.get_session(session_id, user.id)
    if not session:
        raise HTTPException(404, detail="Session not found.")
    messages = await sessions.get_chat_history(session_id, last_n=last_n)
    return {
        **sessions.session_summary(session),
        "patient_profile": _safe_profile(session.get("patient_profile") or {}),
        "awaiting_gap_answers": bool(session.get("pending_trials")),
        "messages": [
            {
                "role": m["role"],
                "content": m["content"],
                "trials": (m.get("metadata") or {}).get("trials", []),
                "timestamp": m["created_at"].isoformat(),
            }
            for m in messages
        ],
    }


@router.delete("/chat/sessions/{session_id}")
async def delete_session(session_id: str, user: User = Depends(get_current_user)):
    if not await sessions.delete_session(session_id, user.id):
        raise HTTPException(404, detail="Session not found.")
    return {"message": "Session deleted."}


# ─── Admin ────────────────────────────────────────────────────────────────────

@router.get("/admin/stats")
async def admin_stats(_: User = Depends(require_role("admin"))):
    row = await db.fetch_one(
        """
        SELECT (SELECT count(*) FROM trials)         AS trials,
               (SELECT max(ingested_at) FROM trials) AS last_ingest,
               (SELECT count(*) FROM users)          AS users,
               (SELECT count(*) FROM chat_sessions)  AS sessions,
               (SELECT count(*) FROM chat_messages
                 WHERE created_at > now() - interval '24 hours') AS messages_24h
        """
    )
    return {**row, "last_ingest": row["last_ingest"].isoformat() if row["last_ingest"] else None}


app.include_router(router)


# ─── Helpers ──────────────────────────────────────────────────────────────────

_PROFILE_KEYS = ("age", "sex", "location", "conditions", "symptoms", "medications", "medical_history", "other_facts")


def _stored_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: profile.get(k) for k in _PROFILE_KEYS}


def _safe_profile(profile: dict[str, Any]) -> dict[str, Any]:
    return {k: profile.get(k) or ([] if k not in ("age", "sex", "location") else None) for k in _PROFILE_KEYS}


def _card(r: dict[str, Any]) -> dict[str, Any]:
    return {k: r.get(k) for k in TrialCard.model_fields if r.get(k) is not None}


_ORDER = {"YES": 0, "POSSIBLE": 1, "NO": 2, "ERROR": 3}


def _best(results: list[dict[str, Any]], k: int = RETRIEVAL_K) -> list[dict[str, Any]]:
    """
    Screen more candidates than we show: keep the k best by verdict (then
    semantic rank), plus at most 2 'probably not' trials so the patient can see
    why close matches were ruled out.
    """
    ranked = sorted(results, key=lambda r: _ORDER.get(r.get("eligibility"), 9))  # stable
    fits = [r for r in ranked if r.get("eligibility") in ("YES", "POSSIBLE")][:k]
    rest = [r for r in ranked if r.get("eligibility") not in ("YES", "POSSIBLE")]
    return fits + rest[: max(0, min(2, k - len(fits)))]


async def _emit_results(
    stream: bool, session_id: str, profile: dict[str, Any], results: list[dict[str, Any]]
):
    results = sorted(results, key=lambda r: _ORDER.get(r.get("eligibility"), 9))
    await attach_contacts(results, profile.get("location"))
    cards = [_card(r) for r in results]
    if stream:
        return StreamingResponse(
            _stream_results(session_id, profile, results, cards),
            media_type="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )
    reply = await generate_response(profile, results)
    return await _reply(stream, session_id, profile, reply, cards, search_performed=True)


async def _stream_results(
    session_id: str,
    profile: dict[str, Any],
    results: list[dict[str, Any]],
    cards: list[dict[str, Any]],
) -> AsyncIterator[str]:
    parts: list[str] = []
    async for chunk in stream_response(profile, results):
        parts.append(chunk)
        yield _sse({"type": "token", "content": chunk})
    await sessions.add_message(session_id, "assistant", "".join(parts), {"trials": cards})
    yield _sse(
        _done_payload(
            session_id, profile, cards, search_performed=True,
            clarification=False, awaiting_gap_answers=False, trials_found=len(cards),
        )
    )


async def _reply(
    stream: bool,
    session_id: str,
    profile: dict[str, Any],
    reply: str,
    cards: list[dict[str, Any]],
    *,
    search_performed: bool,
    clarification: bool = False,
    awaiting_gap_answers: bool = False,
    trials_found: Optional[int] = None,
):
    """Persist a complete reply and return it as JSON or a one-shot SSE stream."""
    await sessions.add_message(session_id, "assistant", reply, {"trials": cards} if cards else None)
    payload = _done_payload(
        session_id, profile, cards, search_performed=search_performed,
        clarification=clarification, awaiting_gap_answers=awaiting_gap_answers,
        trials_found=len(cards) if trials_found is None else trials_found,
    )
    if stream:
        async def one_shot() -> AsyncIterator[str]:
            yield _sse({"type": "token", "content": reply})
            yield _sse(payload)

        return StreamingResponse(one_shot(), media_type="text/event-stream")
    return ChatResponse(reply=reply, **{k: v for k, v in payload.items() if k != "type"})


def _done_payload(
    session_id: str,
    profile: dict[str, Any],
    cards: list[dict[str, Any]],
    *,
    search_performed: bool,
    clarification: bool,
    awaiting_gap_answers: bool,
    trials_found: int,
) -> dict[str, Any]:
    return {
        "type": "done",
        "session_id": session_id,
        "patient_profile": _safe_profile(profile),
        "trials_found": trials_found,
        "trials": cards,
        "search_performed": search_performed,
        "clarification_needed": clarification,
        "awaiting_gap_answers": awaiting_gap_answers,
    }


def _sse(data: dict[str, Any]) -> str:
    return f"data: {json.dumps(data, default=str)}\n\n"
