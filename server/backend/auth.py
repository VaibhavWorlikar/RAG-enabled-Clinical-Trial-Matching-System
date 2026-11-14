"""
backend/auth.py
───────────────
JWT authentication with users stored in Postgres.

• Passwords are bcrypt-hashed.
• Self-registration can only create "patient" or "doctor" accounts.
  The admin role is granted solely to usernames listed in ADMIN_USERNAMES.
"""
import logging
from datetime import datetime, timedelta, timezone
from typing import Optional

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from psycopg.errors import UniqueViolation
from pydantic import BaseModel

from backend import db
from backend.config import (
    ADMIN_USERNAMES,
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES,
    JWT_ALGORITHM,
    JWT_SECRET_KEY,
)

logger = logging.getLogger(__name__)

oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/auth/login")

SELF_SERVICE_ROLES = ("patient", "doctor")


class User(BaseModel):
    id: int
    username: str
    role: str
    disabled: bool = False


# ─── Passwords ────────────────────────────────────────────────────────────────

def hash_password(plain: str) -> str:
    return bcrypt.hashpw(plain.encode()[:72], bcrypt.gensalt()).decode()


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return bcrypt.checkpw(plain.encode()[:72], hashed.encode())
    except ValueError:
        return False


# ─── Users ────────────────────────────────────────────────────────────────────

async def create_user(username: str, password: str, role: str = "patient") -> User:
    username = username.strip().lower()
    if role not in SELF_SERVICE_ROLES:
        raise ValueError("Role must be 'patient' or 'doctor'.")
    if username in ADMIN_USERNAMES:
        role = "admin"
    try:
        row = await db.fetch_one(
            "INSERT INTO users (username, hashed_pw, role) VALUES (%s, %s, %s) "
            "RETURNING id, username, role, disabled",
            (username, hash_password(password), role),
        )
    except UniqueViolation:
        raise ValueError("That username is already taken.")
    logger.info("Created user '%s' with role '%s'.", username, role)
    return User(**row)


async def authenticate_user(username: str, password: str) -> Optional[User]:
    row = await db.fetch_one(
        "SELECT id, username, role, disabled, hashed_pw FROM users WHERE username = %s",
        (username.strip().lower(),),
    )
    if not row or row["disabled"] or not verify_password(password, row.pop("hashed_pw")):
        return None
    return User(**row)


# ─── Tokens ───────────────────────────────────────────────────────────────────

def create_access_token(user: User, expires_delta: Optional[timedelta] = None) -> str:
    expire = datetime.now(timezone.utc) + (
        expires_delta or timedelta(minutes=JWT_ACCESS_TOKEN_EXPIRE_MINUTES)
    )
    payload = {"sub": str(user.id), "username": user.username, "exp": expire}
    return jwt.encode(payload, JWT_SECRET_KEY, algorithm=JWT_ALGORITHM)


async def get_current_user(token: str = Depends(oauth2_scheme)) -> User:
    unauthorized = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid or expired token.",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, JWT_SECRET_KEY, algorithms=[JWT_ALGORITHM])
        user_id = int(payload["sub"])
    except (jwt.PyJWTError, KeyError, ValueError):
        raise unauthorized

    # Role is read from the DB, never trusted from the token.
    row = await db.fetch_one(
        "SELECT id, username, role, disabled FROM users WHERE id = %s", (user_id,)
    )
    if not row or row["disabled"]:
        raise unauthorized
    return User(**row)


def require_role(*allowed_roles: str):
    async def _check(current_user: User = Depends(get_current_user)) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(status.HTTP_403_FORBIDDEN, detail="Not permitted.")
        return current_user

    return _check
