"""Application configuration, loaded from environment / .env."""

from __future__ import annotations

import secrets
from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = BACKEND_DIR.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=(PROJECT_ROOT / ".env", BACKEND_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "NeuroGuard"
    environment: str = "development"
    debug: bool = True

    # Default to SQLite so the project runs with zero setup; docker-compose
    # overrides this with the Postgres URL.
    database_url: str = f"sqlite:///{BACKEND_DIR / 'data' / 'neuroguard.db'}"

    # Generated per-process if unset. Fine for dev; MUST be set in production or
    # every restart invalidates all sessions.
    secret_key: str = secrets.token_urlsafe(48)
    algorithm: str = "HS256"
    access_token_minutes: int = 30
    refresh_token_days: int = 7

    cookie_secure: bool = False          # True behind HTTPS
    cookie_samesite: str = "lax"
    cookie_domain: str | None = None

    cors_origins: str = "http://localhost:5173,http://127.0.0.1:5173"

    # --- password reset ---------------------------------------------------
    # Short by design: a reset link is a bearer credential for the account, and
    # it commonly sits in an inbox for far longer than it is needed.
    reset_token_minutes: int = 30
    # Where the emailed link points. Must be the frontend origin, not the API.
    frontend_base_url: str = "http://localhost:5173"

    # SMTP is optional. With no host configured the mailer writes the reset link
    # to the application log instead of sending it, which keeps the flow usable
    # offline and in demos without inventing a fake "check your email" success.
    smtp_host: str | None = None
    smtp_port: int = 587
    smtp_user: str | None = None
    smtp_password: str | None = None
    smtp_from: str = "NeuroGuard <no-reply@neuroguard.local>"
    smtp_starttls: bool = True

    upload_dir: Path = BACKEND_DIR / "data" / "uploads"
    max_audio_mb: int = 25

    whisper_model: str = "base.en"

    @property
    def cors_origin_list(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]

    @property
    def is_production(self) -> bool:
        return self.environment.lower() in {"production", "prod"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    s = Settings()
    s.upload_dir.mkdir(parents=True, exist_ok=True)
    if s.database_url.startswith("sqlite"):
        (BACKEND_DIR / "data").mkdir(parents=True, exist_ok=True)
    return s


settings = get_settings()
