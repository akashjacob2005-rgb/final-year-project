"""Password hashing and JWT issuing/verification."""

from __future__ import annotations

import hashlib
import secrets
from datetime import datetime, timedelta, timezone

import jwt
from argon2 import PasswordHasher
from argon2.exceptions import VerificationError, VerifyMismatchError

from ..config import settings

# Argon2id — the current password-hashing recommendation, and it manages its own
# per-password salt.
_hasher = PasswordHasher()

ACCESS = "access"
REFRESH = "refresh"


def hash_password(password: str) -> str:
    return _hasher.hash(password)


def verify_password(password: str, password_hash: str) -> bool:
    try:
        _hasher.verify(password_hash, password)
        return True
    except (VerifyMismatchError, VerificationError, Exception):
        return False


def needs_rehash(password_hash: str) -> bool:
    try:
        return _hasher.check_needs_rehash(password_hash)
    except Exception:
        return False


def _encode(subject: str, token_type: str, expires: timedelta) -> str:
    now = datetime.now(timezone.utc)
    payload = {
        "sub": str(subject),
        "type": token_type,
        "iat": now,
        "exp": now + expires,
        "jti": secrets.token_urlsafe(16),
    }
    return jwt.encode(payload, settings.secret_key, algorithm=settings.algorithm)


def create_access_token(user_id: int) -> str:
    return _encode(user_id, ACCESS, timedelta(minutes=settings.access_token_minutes))


def create_refresh_token(user_id: int) -> tuple[str, datetime]:
    expires = timedelta(days=settings.refresh_token_days)
    token = _encode(user_id, REFRESH, expires)
    return token, datetime.now(timezone.utc) + expires


def decode_token(token: str, expected_type: str) -> dict | None:
    """Return the payload, or None if the token is invalid, expired, or of the
    wrong type. Rejecting on type stops a refresh token being replayed as an
    access token."""
    try:
        payload = jwt.decode(
            token, settings.secret_key, algorithms=[settings.algorithm]
        )
    except jwt.PyJWTError:
        return None
    if payload.get("type") != expected_type:
        return None
    return payload


def hash_refresh_token(token: str) -> str:
    """Refresh tokens are stored hashed, never in plaintext."""
    return hashlib.sha256(token.encode()).hexdigest()


def create_reset_token() -> tuple[str, datetime]:
    """Issue a password-reset secret and its expiry.

    Deliberately NOT a JWT. A reset token has to be revocable the instant it is
    used, and a JWT is only revocable by keeping a server-side record anyway —
    so this is an opaque random string whose hash is the record. Nothing is
    encoded in it, so nothing leaks if it ends up in a browser history or a
    forwarded email.
    """
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(
        minutes=settings.reset_token_minutes
    )
    return token, expires


def hash_reset_token(token: str) -> str:
    """Reset tokens are stored hashed, never in plaintext."""
    return hashlib.sha256(token.encode()).hexdigest()
