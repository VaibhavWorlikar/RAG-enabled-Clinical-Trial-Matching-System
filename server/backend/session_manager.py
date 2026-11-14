"""
backend/session_manager.py
──────────────────────────
Chat sessions persisted in Postgres.

Every read/write is scoped to the owning user_id, so one user can never read
or modify another user's session by guessing its ID.
"""
import json
import logging
import uuid
from typing import Any, Optional

from psycopg.types.json import Jsonb

from backend import db

logger = logging.getLogger(__name__)

_MAX_HISTORY_MESSAGES = 40


async def get_session(session_id: Optional[str], user_id: int) -> Optional[dict[str, Any]]:
    if not session_id:
        return None
    try:
        uuid.UUID(session_id)
    except ValueError:
        return None
    return await db.fetch_one(
        "SELECT * FROM chat_sessions WHERE id = %s AND user_id = %s",
        (session_id, user_id),
    )


async def get_or_create_session(session_id: Optional[str], user_id: int) -> dict[str, Any]:
    session = await get_session(session_id, user_id)
    if session:
        return session
    return await db.fetch_one(
        "INSERT INTO chat_sessions (id, user_id) VALUES (%s, %s) RETURNING *",
        (str(uuid.uuid4()), user_id),
    )


async def add_message(
    session_id: str, role: str, content: str, metadata: Optional[dict] = None
) -> None:
    await db.execute(
        "INSERT INTO chat_messages (session_id, role, content, metadata) VALUES (%s, %s, %s, %s)",
        (session_id, role, content, Jsonb(metadata) if metadata else None),
    )
    await db.execute("UPDATE chat_sessions SET updated_at = now() WHERE id = %s", (session_id,))


async def get_chat_history(session_id: str, last_n: int = 10) -> list[dict[str, Any]]:
    rows = await db.fetch_all(
        "SELECT role, content, metadata, created_at FROM chat_messages "
        "WHERE session_id = %s ORDER BY id DESC LIMIT %s",
        (session_id, min(last_n, _MAX_HISTORY_MESSAGES)),
    )
    return list(reversed(rows))


async def update_session(session_id: str, **fields: Any) -> None:
    """Update any of: patient_profile, asked_gaps, pending_trials, search_count."""
    allowed = {"patient_profile", "asked_gaps", "pending_trials", "search_count"}
    sets, params = [], []
    for key, value in fields.items():
        if key not in allowed:
            raise ValueError(f"Unknown session field {key}")
        sets.append(f"{key} = %s")
        params.append(Jsonb(value) if key == "patient_profile" else value)
    if not sets:
        return
    params.append(session_id)
    await db.execute(
        f"UPDATE chat_sessions SET {', '.join(sets)}, updated_at = now() WHERE id = %s",
        params,
    )


async def list_sessions(user_id: int, limit: int = 20) -> list[dict[str, Any]]:
    return await db.fetch_all(
        """
        SELECT s.id, s.search_count, s.created_at, s.updated_at,
               s.patient_profile->'conditions' AS conditions,
               (SELECT content FROM chat_messages m
                 WHERE m.session_id = s.id AND m.role = 'user'
                 ORDER BY m.id LIMIT 1) AS first_message
          FROM chat_sessions s
         WHERE s.user_id = %s
         ORDER BY s.updated_at DESC
         LIMIT %s
        """,
        (user_id, limit),
    )


async def delete_session(session_id: str, user_id: int) -> bool:
    return (
        await db.execute(
            "DELETE FROM chat_sessions WHERE id = %s AND user_id = %s", (session_id, user_id)
        )
        > 0
    )


async def count_user_messages_today(user_id: int) -> int:
    row = await db.fetch_one(
        """
        SELECT count(*) AS n
          FROM chat_messages m JOIN chat_sessions s ON s.id = m.session_id
         WHERE s.user_id = %s AND m.role = 'user'
           AND m.created_at > now() - interval '24 hours'
        """,
        (user_id,),
    )
    return int(row["n"]) if row else 0


def session_summary(session: dict[str, Any]) -> dict[str, Any]:
    p = session.get("patient_profile") or {}
    if isinstance(p, str):
        p = json.loads(p)
    return {
        "session_id": str(session["id"]),
        "search_count": session["search_count"],
        "age": p.get("age"),
        "sex": p.get("sex"),
        "conditions": p.get("conditions", []),
        "symptoms": p.get("symptoms", []),
        "medications": p.get("medications", []),
        "asked_gaps": list(session.get("asked_gaps") or []),
        "has_pending": bool(session.get("pending_trials")),
        "created_at": session["created_at"].isoformat(),
    }
