"""Vercel / uvicorn entrypoint: `uvicorn main:app --reload` from server/."""
from backend.api import app

__all__ = ["app"]
