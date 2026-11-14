"""
backend/db.py
─────────────
Async Postgres connection pool (psycopg 3).

• Opened lazily on first use, so serverless cold starts don't pay for it
  unless a request actually touches the database.
• prepare_threshold=None disables server-side prepared statements, which is
  required behind Neon's pooled (PgBouncer, transaction-mode) endpoint.
"""
import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool

from backend.config import DATABASE_URL

logger = logging.getLogger(__name__)

_pool: Optional[AsyncConnectionPool] = None
_lock = asyncio.Lock()

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


async def get_pool() -> AsyncConnectionPool:
    global _pool
    if _pool is None:
        async with _lock:
            if _pool is None:
                if not DATABASE_URL:
                    raise RuntimeError("DATABASE_URL is not configured.")
                pool = AsyncConnectionPool(
                    DATABASE_URL,
                    min_size=0,
                    max_size=5,
                    open=False,
                    kwargs={"prepare_threshold": None, "row_factory": dict_row},
                )
                await pool.open()
                _pool = pool
    return _pool


async def fetch_all(sql: str, params: Any = None) -> list[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchall()


async def fetch_one(sql: str, params: Any = None) -> Optional[dict]:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return await cur.fetchone()


async def execute(sql: str, params: Any = None) -> int:
    pool = await get_pool()
    async with pool.connection() as conn:
        cur = await conn.execute(sql, params)
        return cur.rowcount


async def apply_schema() -> None:
    pool = await get_pool()
    async with pool.connection() as conn:
        await conn.execute(SCHEMA_PATH.read_text())
    logger.info("Schema applied.")


def to_vector_literal(vec: list[float]) -> str:
    """pgvector text format: '[0.1,0.2,...]'."""
    return "[" + ",".join(f"{x:.6f}" for x in vec) + "]"
