"""Signup, login, refresh, logout, profile."""

from __future__ import annotations

import time
from collections import defaultdict
from datetime import datetime, timezone

from fastapi import APIRouter, Cookie, Depends, HTTPException, Request, Response, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import settings
from ...core import mailer
from ...core.deps import ACCESS_COOKIE, REFRESH_COOKIE, get_current_user
from ...core.security import (
    REFRESH,
    create_access_token,
    create_refresh_token,
    create_reset_token,
    decode_token,
    hash_password,
    hash_refresh_token,
    hash_reset_token,
    needs_rehash,
    verify_password,
)
from ...database import get_db
from ...models import PasswordResetToken, RefreshToken, User
from ...schemas import (
    AuthResponse,
    ForgotPasswordRequest,
    MessageResponse,
    ResetPasswordRequest,
    UserCreate,
    UserLogin,
    UserOut,
    UserUpdate,
)

router = APIRouter(prefix="/auth", tags=["auth"])

# Simple in-process login throttle. Enough to blunt credential stuffing in a
# single-instance deployment; a multi-instance one would need Redis.
_MAX_ATTEMPTS = 8
_WINDOW_S = 300
_attempts: dict[str, list[float]] = defaultdict(list)


def _throttle(key: str) -> None:
    now = time.time()
    recent = [t for t in _attempts[key] if now - t < _WINDOW_S]
    _attempts[key] = recent
    if len(recent) >= _MAX_ATTEMPTS:
        raise HTTPException(
            status.HTTP_429_TOO_MANY_REQUESTS,
            "Too many failed attempts. Try again in a few minutes.",
        )


def _record_failure(key: str) -> None:
    _attempts[key].append(time.time())


def _set_auth_cookies(response: Response, user_id: int, db: Session) -> None:
    access = create_access_token(user_id)
    refresh, expires_at = create_refresh_token(user_id)

    db.add(
        RefreshToken(
            user_id=user_id,
            token_hash=hash_refresh_token(refresh),
            expires_at=expires_at,
        )
    )
    db.commit()

    common = dict(
        httponly=True,
        secure=settings.cookie_secure,
        samesite=settings.cookie_samesite,
        domain=settings.cookie_domain,
        path="/",
    )
    response.set_cookie(
        ACCESS_COOKIE, access, max_age=settings.access_token_minutes * 60, **common
    )
    response.set_cookie(
        REFRESH_COOKIE, refresh, max_age=settings.refresh_token_days * 86400, **common
    )


def _clear_auth_cookies(response: Response) -> None:
    for name in (ACCESS_COOKIE, REFRESH_COOKIE):
        response.delete_cookie(
            name,
            path="/",
            domain=settings.cookie_domain,
            httponly=True,
            secure=settings.cookie_secure,
            samesite=settings.cookie_samesite,
        )


@router.post("/signup", response_model=AuthResponse, status_code=status.HTTP_201_CREATED)
def signup(payload: UserCreate, response: Response, db: Session = Depends(get_db)):
    email = payload.email.lower()
    if db.scalar(select(User).where(User.email == email)):
        raise HTTPException(status.HTTP_409_CONFLICT, "An account with that email already exists")

    user = User(
        email=email,
        password_hash=hash_password(payload.password),
        full_name=payload.full_name,
        age=payload.age,
        education_years=payload.education_years,
    )
    db.add(user)
    db.commit()
    db.refresh(user)

    _set_auth_cookies(response, user.id, db)
    return AuthResponse(user=UserOut.model_validate(user), message="Account created")


@router.post("/login", response_model=AuthResponse)
def login(
    payload: UserLogin,
    request: Request,
    response: Response,
    db: Session = Depends(get_db),
):
    email = payload.email.lower()
    key = f"{request.client.host if request.client else 'unknown'}:{email}"
    _throttle(key)

    user = db.scalar(select(User).where(User.email == email))
    # Same message and code whether the email is unknown or the password is
    # wrong, so the endpoint cannot be used to enumerate registered accounts.
    if not user or not verify_password(payload.password, user.password_hash):
        _record_failure(key)
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Incorrect email or password")

    if not user.is_active:
        raise HTTPException(status.HTTP_403_FORBIDDEN, "This account is disabled")

    if needs_rehash(user.password_hash):
        user.password_hash = hash_password(payload.password)
        db.commit()

    _attempts.pop(key, None)
    _set_auth_cookies(response, user.id, db)
    return AuthResponse(user=UserOut.model_validate(user), message="Signed in")


@router.post("/refresh", response_model=AuthResponse)
def refresh_tokens(
    response: Response,
    db: Session = Depends(get_db),
    neuroguard_refresh: str | None = Cookie(default=None),
):
    """Rotate the refresh token: the presented one is revoked and a new pair
    issued, so a stolen token is usable at most once."""
    if not neuroguard_refresh:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "No refresh token")

    payload = decode_token(neuroguard_refresh, REFRESH)
    if not payload:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Invalid refresh token")

    stored = db.scalar(
        select(RefreshToken).where(
            RefreshToken.token_hash == hash_refresh_token(neuroguard_refresh)
        )
    )
    if not stored or stored.revoked:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token is no longer valid")

    expires_at = stored.expires_at
    if expires_at.tzinfo is None:  # SQLite loses tzinfo on round-trip
        expires_at = expires_at.replace(tzinfo=timezone.utc)
    if expires_at < datetime.now(timezone.utc):
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Refresh token expired")

    user = db.get(User, stored.user_id)
    if not user or not user.is_active:
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, "Account unavailable")

    stored.revoked = True
    db.commit()

    _set_auth_cookies(response, user.id, db)
    return AuthResponse(user=UserOut.model_validate(user), message="Session refreshed")


@router.post("/logout")
def logout(
    response: Response,
    db: Session = Depends(get_db),
    neuroguard_refresh: str | None = Cookie(default=None),
):
    if neuroguard_refresh:
        stored = db.scalar(
            select(RefreshToken).where(
                RefreshToken.token_hash == hash_refresh_token(neuroguard_refresh)
            )
        )
        if stored:
            stored.revoked = True
            db.commit()

    _clear_auth_cookies(response)
    return {"message": "Signed out"}


# --------------------------------------------------------------------------
# password reset
# --------------------------------------------------------------------------
# Same generic reply whether or not the address is registered. Anything that
# varies with account existence — wording, status code, or response time —
# turns this endpoint into an account-enumeration oracle, which would undo the
# care taken in login() above.
_RESET_REQUESTED = (
    "If an account exists for that address, a reset link has been sent. "
    "The link expires in 30 minutes."
)


@router.post("/forgot-password", response_model=MessageResponse)
def forgot_password(
    payload: ForgotPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    email = payload.email.lower()
    # Throttled on the same counter as login, so this cannot be used as an
    # unmetered way to probe for registered addresses or to spam an inbox.
    key = f"reset:{request.client.host if request.client else 'unknown'}:{email}"
    _throttle(key)
    _record_failure(key)

    user = db.scalar(select(User).where(User.email == email))
    if user and user.is_active:
        # Supersede any outstanding token: a user who clicks "forgot password"
        # twice should not leave the first link live in their inbox.
        for old in db.scalars(
            select(PasswordResetToken).where(
                PasswordResetToken.user_id == user.id,
                PasswordResetToken.used.is_(False),
            )
        ).all():
            old.used = True

        token, expires_at = create_reset_token()
        db.add(
            PasswordResetToken(
                user_id=user.id,
                token_hash=hash_reset_token(token),
                expires_at=expires_at,
            )
        )
        db.commit()
        # Return value ignored on purpose — see mailer module docstring.
        mailer.send_reset_email(user.email, token, user.full_name)

    return MessageResponse(message=_RESET_REQUESTED)


@router.post("/reset-password", response_model=MessageResponse)
def reset_password(
    payload: ResetPasswordRequest,
    request: Request,
    db: Session = Depends(get_db),
):
    key = f"reset-confirm:{request.client.host if request.client else 'unknown'}"
    _throttle(key)

    stored = db.scalar(
        select(PasswordResetToken).where(
            PasswordResetToken.token_hash == hash_reset_token(payload.token)
        )
    )

    expires_at = stored.expires_at if stored else None
    if expires_at is not None and expires_at.tzinfo is None:
        expires_at = expires_at.replace(tzinfo=timezone.utc)  # SQLite drops tzinfo

    # One message for every failure mode. Distinguishing "expired" from "already
    # used" from "never existed" tells an attacker holding a stale link whether
    # it was ever real.
    if (
        not stored
        or stored.used
        or expires_at is None
        or expires_at < datetime.now(timezone.utc)
    ):
        _record_failure(key)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This reset link is invalid or has expired. Please request a new one.",
        )

    user = db.get(User, stored.user_id)
    if not user or not user.is_active:
        _record_failure(key)
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "This reset link is invalid or has expired. Please request a new one.",
        )

    user.password_hash = hash_password(payload.new_password)
    stored.used = True

    # Anyone holding a session for this account loses it. If the reset was
    # prompted by a compromise, leaving existing refresh tokens alive would let
    # the attacker keep the account despite the password change.
    for rt in db.scalars(
        select(RefreshToken).where(
            RefreshToken.user_id == user.id, RefreshToken.revoked.is_(False)
        )
    ).all():
        rt.revoked = True

    db.commit()
    _attempts.pop(key, None)
    _attempts.pop(f"{request.client.host if request.client else 'unknown'}:{user.email}", None)

    return MessageResponse(
        message="Your password has been changed. You can now sign in with it."
    )


@router.get("/me", response_model=UserOut)
def me(user: User = Depends(get_current_user)):
    return UserOut.model_validate(user)


@router.patch("/me", response_model=UserOut)
def update_me(
    payload: UserUpdate,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    for field, value in payload.model_dump(exclude_unset=True).items():
        setattr(user, field, value)
    db.commit()
    db.refresh(user)
    return UserOut.model_validate(user)
