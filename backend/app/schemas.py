"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, EmailStr, Field, field_validator

TestName = Literal[
    "memory_recognition",
    "picture_description",
    "trail_making_b",
    "serial_sevens",
    "digit_span_backward",
    "verbal_fluency",
]


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
class UserCreate(BaseModel):
    email: EmailStr
    password: str = Field(min_length=8, max_length=128)
    full_name: str = Field(min_length=1, max_length=120)
    age: int = Field(ge=18, le=120)
    education_years: int | None = Field(default=None, ge=0, le=30)

    @field_validator("password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if v.isdigit() or v.isalpha():
            raise ValueError("Password must contain both letters and numbers")
        return v

    @field_validator("full_name")
    @classmethod
    def strip_name(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Name cannot be empty")
        return v


class UserLogin(BaseModel):
    email: EmailStr
    password: str


class UserOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    email: EmailStr
    full_name: str
    age: int
    education_years: int | None
    created_at: datetime


class UserUpdate(BaseModel):
    full_name: str | None = Field(default=None, min_length=1, max_length=120)
    age: int | None = Field(default=None, ge=18, le=120)
    education_years: int | None = Field(default=None, ge=0, le=30)


class AuthResponse(BaseModel):
    user: UserOut
    message: str = "ok"


# --------------------------------------------------------------------------
# assessment
# --------------------------------------------------------------------------
class SessionOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    status: str
    started_at: datetime
    completed_at: datetime | None
    overall_risk: float | None
    risk_band: str | None
    confidence: str | None


class TestSubmission(BaseModel):
    """Raw response for one structured test.

    `raw_response` is deliberately free-form per test; each scorer validates its
    own shape. Storing it verbatim lets sessions be rescored if the scoring
    rules or normative tables change.
    """

    test_name: TestName
    raw_response: dict[str, Any]
    duration_ms: int | None = Field(default=None, ge=0)


class TestResultOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    test_name: str
    domain: str
    raw_measure: float | None
    score: float | None
    max_score: float | None
    z_score: float | None
    detail_json: dict | None
    duration_ms: int | None


class TranscriptSubmission(BaseModel):
    """Fallback path when the browser cannot record audio."""

    transcript: str = Field(min_length=1, max_length=20000)
    duration_ms: int | None = Field(default=None, ge=0)


class AssessmentResult(BaseModel):
    session_id: int
    risk: float
    risk_percent: float
    band: str
    confidence: str
    components: dict[str, Any]
    weights: dict[str, float]
    domain_z: dict[str, float]
    weakest_domain: str | None
    notes: list[str]
    method: str
    disclaimer: str
    tests: list[TestResultOut]
    language: dict[str, Any] | None = None


class HistoryPoint(BaseModel):
    session_id: int
    completed_at: datetime | None
    risk_percent: float | None
    band: str | None
    domain_z: dict[str, float] = {}
