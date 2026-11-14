"""
backend/embeddings.py
─────────────────────
Text embeddings via OpenAI (text-embedding-3-small, truncated to 512 dims).

Replaces the original local Bio_ClinicalBERT model: no 400 MB model download,
no PyTorch in the serverless bundle, and queries + documents share one model.
"""
import logging

from openai import AsyncOpenAI, OpenAI

from backend.config import EMBEDDING_DIMENSIONS, EMBEDDING_MODEL, OPENAI_API_KEY

logger = logging.getLogger(__name__)

_MAX_CHARS = 8000  # ~2k tokens; well under the 8191-token model limit

_async_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
_sync_client = OpenAI(api_key=OPENAI_API_KEY)


def _clean(text: str) -> str:
    text = " ".join((text or "").split())
    if not text:
        raise ValueError("Cannot embed empty text.")
    return text[:_MAX_CHARS]


async def embed_query(text: str) -> list[float]:
    resp = await _async_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=_clean(text),
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return resp.data[0].embedding


def embed_batch(texts: list[str]) -> list[list[float]]:
    """Synchronous batch embedding for the ingestion script."""
    resp = _sync_client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=[_clean(t) for t in texts],
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return [d.embedding for d in sorted(resp.data, key=lambda d: d.index)]
