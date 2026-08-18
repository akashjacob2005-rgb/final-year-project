"""SQLAlchemy ORM models.

JSON columns store the raw response for every test (which items were clicked,
in what order, with what timings). Keeping the raw data — not just the score —
means the scoring rules can be revised later and old sessions rescored, which
matters because the normative tables here are explicitly provisional.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import (
    JSON,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .database import Base


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    password_hash: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(120))

    # Needed for age/education normative adjustment of the structured tests.
    age: Mapped[int] = mapped_column(Integer)
    education_years: Mapped[int | None] = mapped_column(Integer, nullable=True)

    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    sessions: Mapped[list["AssessmentSession"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )
    refresh_tokens: Mapped[list["RefreshToken"]] = relationship(
        back_populates="user", cascade="all, delete-orphan"
    )


class RefreshToken(Base):
    __tablename__ = "refresh_tokens"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    # Only the hash is stored, so a database leak does not hand over live sessions.
    token_hash: Mapped[str] = mapped_column(String(128), unique=True, index=True)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))
    revoked: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    user: Mapped[User] = relationship(back_populates="refresh_tokens")


class AssessmentSession(Base):
    __tablename__ = "assessment_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)

    status: Mapped[str] = mapped_column(String(20), default="in_progress")  # in_progress|completed|abandoned
    started_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)

    # Snapshot of the demographics used for norming, so a later profile edit
    # never silently changes a historical result.
    age_at_assessment: Mapped[int | None] = mapped_column(Integer, nullable=True)
    education_years_at_assessment: Mapped[int | None] = mapped_column(Integer, nullable=True)

    overall_risk: Mapped[float | None] = mapped_column(Float, nullable=True)
    risk_band: Mapped[str | None] = mapped_column(String(20), nullable=True)
    confidence: Mapped[str | None] = mapped_column(String(20), nullable=True)
    language_probability: Mapped[float | None] = mapped_column(Float, nullable=True)
    composite_z: Mapped[float | None] = mapped_column(Float, nullable=True)

    # The randomised stimuli actually shown in this session (which five objects,
    # which digit sequences, node layout). Scoring compares against this rather
    # than against anything the client sends back, so a tampered client cannot
    # redefine what "correct" means.
    stimuli_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    result_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    model_version: Mapped[str | None] = mapped_column(String(40), nullable=True)

    user: Mapped[User] = relationship(back_populates="sessions")
    results: Mapped[list["TestResult"]] = relationship(
        back_populates="session", cascade="all, delete-orphan"
    )


class TestResult(Base):
    __tablename__ = "test_results"

    id: Mapped[int] = mapped_column(primary_key=True)
    session_id: Mapped[int] = mapped_column(
        ForeignKey("assessment_sessions.id", ondelete="CASCADE"), index=True
    )

    test_name: Mapped[str] = mapped_column(String(40), index=True)
    domain: Mapped[str] = mapped_column(String(30))

    raw_response: Mapped[dict] = mapped_column(JSON)   # exactly what the user did
    raw_measure: Mapped[float | None] = mapped_column(Float, nullable=True)
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    max_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    z_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    detail_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    duration_ms: Mapped[int | None] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    session: Mapped[AssessmentSession] = relationship(back_populates="results")
    recording: Mapped["AudioRecording | None"] = relationship(
        back_populates="test_result", cascade="all, delete-orphan", uselist=False
    )


class AudioRecording(Base):
    __tablename__ = "audio_recordings"

    id: Mapped[int] = mapped_column(primary_key=True)
    test_result_id: Mapped[int] = mapped_column(
        ForeignKey("test_results.id", ondelete="CASCADE"), index=True
    )

    file_path: Mapped[str] = mapped_column(String(500))
    mime_type: Mapped[str | None] = mapped_column(String(80), nullable=True)
    size_bytes: Mapped[int | None] = mapped_column(Integer, nullable=True)
    duration_seconds: Mapped[float | None] = mapped_column(Float, nullable=True)

    raw_transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    transcript: Mapped[str | None] = mapped_column(Text, nullable=True)
    timing_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    features_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=utcnow)

    test_result: Mapped[TestResult] = relationship(back_populates="recording")
