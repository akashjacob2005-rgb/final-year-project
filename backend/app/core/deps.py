"""Shared FastAPI dependencies."""

from __future__ import annotations

from fastapi import Cookie, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from ..database import get_db
from ..models import User
from .security import ACCESS, decode_token

ACCESS_COOKIE = "neuroguard_access"
REFRESH_COOKIE = "neuroguard_refresh"

_UNAUTHORISED = HTTPException(
    status_code=status.HTTP_401_UNAUTHORIZED,
    detail="Not authenticated",
)


def _token_from(request: Request, cookie_token: str | None) -> str | None:
    """Prefer the httpOnly cookie; fall back to a bearer header.

    The cookie is what the React app uses (it cannot be read by JavaScript, so
    XSS cannot steal it). The header path exists so the API stays usable from
    curl, tests and the Swagger UI.
    """
    if cookie_token:
        return cookie_token
    auth = request.headers.get("Authorization", "")
    if auth.lower().startswith("bearer "):
        return auth[7:].strip()
    return None


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    neuroguard_access: str | None = Cookie(default=None),
) -> User:
    token = _token_from(request, neuroguard_access)
    if not token:
        raise _UNAUTHORISED

    payload = decode_token(token, ACCESS)
    if not payload:
        raise _UNAUTHORISED

    try:
        user_id = int(payload["sub"])
    except (KeyError, TypeError, ValueError):
        raise _UNAUTHORISED

    user = db.get(User, user_id)
    if user is None or not user.is_active:
        raise _UNAUTHORISED
    return user
