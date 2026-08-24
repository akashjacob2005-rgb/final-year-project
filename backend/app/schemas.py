"""Pydantic request/response models."""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    EmailStr,
    Field,
    field_validator,
    model_validator,
)

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


class ForgotPasswordRequest(BaseModel):
    email: EmailStr


class ResetPasswordRequest(BaseModel):
    token: str = Field(min_length=16, max_length=256)
    # Same rules as signup — a reset must not be a way to set a weaker password
    # than the account could have been created with.
    new_password: str = Field(min_length=8, max_length=128)

    @field_validator("new_password")
    @classmethod
    def password_strength(cls, v: str) -> str:
        if v.isdigit() or v.isalpha():
            raise ValueError("Password must contain both letters and numbers")
        return v


class MessageResponse(BaseModel):
    message: str


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
    # The user-facing headline: 10 is best, 0 is worst. Inverted from `risk`.
    score_out_of_10: float
    band: str
    # True when the band was raised by the safety override in fusion.fuse()
    # rather than by the score alone, so the UI can explain why a mid score
    # can carry a high band.
    band_escalated: bool = False
    escalation_reason: str | None = None
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

    @model_validator(mode="before")
    @classmethod
    def _backfill_score(cls, data):
        """Tolerate result blobs written before `score_out_of_10` existed.

        GET /assessment/{id} rehydrates this model straight from the stored
        `result_json`, so any field added here must cope with rows written by
        older code — otherwise every historical assessment 500s on read.

        Derived from `risk` rather than defaulted to 0.0: the score is a pure
        function of the risk, which every stored blob has, so an old result can
        show its real value instead of a placeholder that would read as a
        catastrophic outcome. Derived rather than migrated because `result_json`
        is a historical record and should not be rewritten.

        Mirrors fusion.fuse()["score_out_of_10"]; the formula is duplicated here
        deliberately, because schemas.py must not import from ml/.
        """
        if (
            isinstance(data, dict)
            and data.get("score_out_of_10") is None
            and data.get("risk") is not None
        ):
            return {**data, "score_out_of_10": round((1 - data["risk"]) * 10, 1)}
        return data


class HistoryPoint(BaseModel):
    session_id: int
    completed_at: datetime | None
    risk_percent: float | None
    score_out_of_10: float | None
    band: str | None
    domain_z: dict[str, float] = {}
