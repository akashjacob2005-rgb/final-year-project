"""Run an assessment session: start, submit tests, complete, fetch result."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from pathlib import Path

from fastapi import (
    APIRouter,
    Depends,
    File,
    Form,
    HTTPException,
    UploadFile,
    status,
)
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...config import settings
from ...core.deps import get_current_user
from ...database import get_db
from ...ml import language_model, speech, stimuli as stimuli_mod
from ...ml.pipeline import (
    DOMAIN_OF_TEST,
    ScoringError,
    fuse,
    normative_z,
    score_submission,
)
from ...models import AssessmentSession, AudioRecording, TestResult, User
from ...schemas import (
    AssessmentResult,
    SessionOut,
    TestResultOut,
    TestSubmission,
    TranscriptSubmission,
)

router = APIRouter(prefix="/assessment", tags=["assessment"])

TOTAL_TESTS = 6
ALLOWED_AUDIO = {"audio/webm", "audio/ogg", "audio/wav", "audio/mpeg", "audio/mp4", "video/webm"}


# --------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------
def _get_session(session_id: int, user: User, db: Session) -> AssessmentSession:
    s = db.get(AssessmentSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Assessment session not found")
    return s

def _require_open(s: AssessmentSession) -> None:
    if s.status == "completed":
        raise HTTPException(
            status.HTTP_409_CONFLICT,
            "This assessment is already complete. Start a new one to test again.",
        )


def _upsert_result(db: Session, session_id: int, test_name: str, **fields) -> TestResult:
    """One row per test per session — resubmitting a test replaces it."""
    existing = db.scalar(
        select(TestResult).where(
            TestResult.session_id == session_id, TestResult.test_name == test_name
        )
    )
    if existing:
        for k, v in fields.items():
            setattr(existing, k, v)
        return existing

    row = TestResult(
        session_id=session_id,
        test_name=test_name,
        domain=DOMAIN_OF_TEST.get(test_name, "unknown"),
        **fields,
    )
    db.add(row)
    return row


# --------------------------------------------------------------------------
# routes
# --------------------------------------------------------------------------
@router.post("/start", status_code=status.HTTP_201_CREATED)
def start_assessment(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Create a session and return the randomised stimuli for all six tests."""
    # Difficulty is tiered by age so the tests keep headroom for younger users;
    # the matching normative bands live in ml/scoring.py.
    generated = stimuli_mod.generate(age=user.age)

    s = AssessmentSession(
        user_id=user.id,
        status="in_progress",
        age_at_assessment=user.age,
        education_years_at_assessment=user.education_years,
        stimuli_json=generated,
    )
    db.add(s)
    db.commit()
    db.refresh(s)

    return {
        "session": SessionOut.model_validate(s),
        "stimuli": stimuli_mod.client_safe(generated),
        "total_tests": TOTAL_TESTS,
    }


@router.post("/{session_id}/test")
def submit_test(
    session_id: int,
    payload: TestSubmission,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Submit one structured (non-speech) test."""
    s = _get_session(session_id, user, db)
    _require_open(s)

    if payload.test_name == "picture_description":
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Use /audio or /transcript for the picture description test",
        )

    try:
        scored = score_submission(
            payload.test_name, payload.raw_response, s.stimuli_json or {}, payload.duration_ms
        )
    except ScoringError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc)) from exc

    scored.z = normative_z(
        scored.test, scored.raw, s.age_at_assessment or user.age,
        s.education_years_at_assessment,
    )

    row = _upsert_result(
        db,
        s.id,
        payload.test_name,
        raw_response=payload.raw_response,
        raw_measure=scored.raw,
        score=scored.score,
        max_score=scored.max_score,
        z_score=scored.z,
        detail_json=scored.detail,
        duration_ms=payload.duration_ms,
    )
    db.commit()
    db.refresh(row)

    return {
        "saved": True,
        "result": TestResultOut.model_validate(row),
        "completed_tests": _completed_count(db, s.id),
        "total_tests": TOTAL_TESTS,
    }


def _completed_count(db: Session, session_id: int) -> int:
    return len(
        db.scalars(select(TestResult).where(TestResult.session_id == session_id)).all()
    )


def _store_language_result(
    db: Session,
    s: AssessmentSession,
    transcript: str,
    raw_transcript: str,
    timing: dict | None,
    duration_ms: int | None,
    audio: dict | None = None,
) -> dict:
    """Run the trained language model and persist the outcome."""
    prediction = language_model.predict(transcript)

    row = _upsert_result(
        db,
        s.id,
        "picture_description",
        raw_response={
            "transcript": transcript,
            "raw_transcript": raw_transcript,
            "timing": timing or {},
        },
        raw_measure=prediction.get("probability"),
        score=prediction.get("probability"),
        max_score=1.0,
        z_score=None,
        detail_json={
            "available": prediction["available"],
            "reason": prediction.get("reason"),
            "word_count": prediction.get("word_count"),
            "contributions": prediction.get("contributions", []),
            "timing": timing or {},
        },
        duration_ms=duration_ms,
    )
    db.flush()

    if audio:
        existing = db.scalar(
            select(AudioRecording).where(AudioRecording.test_result_id == row.id)
        )
        if existing:
            db.delete(existing)
            db.flush()
        db.add(
            AudioRecording(
                test_result_id=row.id,
                file_path=audio["path"],
                mime_type=audio.get("mime_type"),
                size_bytes=audio.get("size_bytes"),
                duration_seconds=(timing or {}).get("duration_seconds"),
                raw_transcript=raw_transcript,
                transcript=transcript,
                timing_json=timing or {},
                features_json=prediction.get("features"),
            )
        )

    db.commit()
    return prediction


@router.post("/{session_id}/audio")
def submit_audio(
    session_id: int,
    file: UploadFile = File(...),
    duration_ms: int | None = Form(default=None),
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Upload the picture-description recording, transcribe it, and score it."""
    s = _get_session(session_id, user, db)
    _require_open(s)

    if file.content_type and file.content_type.split(";")[0] not in ALLOWED_AUDIO:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"Unsupported audio type '{file.content_type}'",
        )

    data = file.file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The recording was empty")
    if len(data) > settings.max_audio_mb * 1024 * 1024:
        raise HTTPException(
            status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            f"Recording exceeds {settings.max_audio_mb}MB",
        )

    suffix = Path(file.filename or "recording.webm").suffix or ".webm"
    dest = settings.upload_dir / f"session{s.id}_{uuid.uuid4().hex}{suffix}"
    dest.write_bytes(data)

    try:
        stt = speech.transcribe(str(dest))
    except speech.TranscriptionError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    prediction = _store_language_result(
        db,
        s,
        transcript=stt["transcript"],
        raw_transcript=stt["raw_transcript"],
        timing=stt["timing"],
        duration_ms=duration_ms,
        audio={
            "path": str(dest),
            "mime_type": file.content_type,
            "size_bytes": len(data),
        },
    )

    return {
        "saved": True,
        "transcript": stt["raw_transcript"],
        "timing": stt["timing"],
        "language": {
            "available": prediction["available"],
            "reason": prediction.get("reason"),
            "word_count": prediction.get("word_count"),
        },
        "completed_tests": _completed_count(db, s.id),
        "total_tests": TOTAL_TESTS,
    }


@router.post("/{session_id}/transcript")
def submit_transcript(
    session_id: int,
    payload: TranscriptSubmission,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Typed fallback when the browser cannot record audio.

    No speech-timing features exist on this path, so the timing panel is simply
    absent rather than filled with zeros that would read as pathological.
    """
    s = _get_session(session_id, user, db)
    _require_open(s)

    normalised = speech.normalise_transcript(payload.transcript)
    prediction = _store_language_result(
        db,
        s,
        transcript=normalised,
        raw_transcript=payload.transcript,
        timing=None,
        duration_ms=payload.duration_ms,
    )

    return {
        "saved": True,
        "language": {
            "available": prediction["available"],
            "reason": prediction.get("reason"),
            "word_count": prediction.get("word_count"),
        },
        "completed_tests": _completed_count(db, s.id),
        "total_tests": TOTAL_TESTS,
    }


@router.post("/{session_id}/complete", response_model=AssessmentResult)
def complete_assessment(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    """Fuse every submitted test into the final screening result."""
    s = _get_session(session_id, user, db)
    rows = db.scalars(
        select(TestResult).where(TestResult.session_id == s.id)
    ).all()

    if not rows:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "No tests have been submitted for this session"
        )

    domain_z = {}
    for r in rows:
        if r.z_score is not None:
            domain_z.setdefault(r.domain, []).append(r.z_score)
    domain_z = {d: sum(v) / len(v) for d, v in domain_z.items() if v}

    language_row = next((r for r in rows if r.test_name == "picture_description"), None)
    language_prob = None
    language_payload = None
    if language_row and (language_row.detail_json or {}).get("available"):
        language_prob = language_row.score
        language_payload = {
            "probability": language_row.score,
            "word_count": (language_row.detail_json or {}).get("word_count"),
            "contributions": (language_row.detail_json or {}).get("contributions", []),
            "timing": (language_row.detail_json or {}).get("timing", {}),
            "transcript": (language_row.raw_response or {}).get("raw_transcript"),
        }
    elif language_row:
        language_payload = {
            "probability": None,
            "reason": (language_row.detail_json or {}).get("reason"),
        }

    result = fuse(language_prob, domain_z, len(rows), TOTAL_TESTS)

    s.status = "completed"
    s.completed_at = datetime.now(timezone.utc)
    s.overall_risk = result["risk"]
    s.risk_band = result["band"]
    s.confidence = result["confidence"]
    s.language_probability = language_prob
    s.composite_z = result["components"]["composite_z"]
    s.result_json = result
    s.model_version = (
        language_model.model_metrics().get("deployed_model") or "unknown"
    )
    db.commit()

    return AssessmentResult(
        session_id=s.id,
        tests=[TestResultOut.model_validate(r) for r in rows],
        language=language_payload,
        **result,
    )


@router.get("/{session_id}", response_model=AssessmentResult)
def get_result(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    s = _get_session(session_id, user, db)
    if not s.result_json:
        raise HTTPException(
            status.HTTP_409_CONFLICT, "This assessment has not been completed yet"
        )

    rows = db.scalars(select(TestResult).where(TestResult.session_id == s.id)).all()
    language_row = next((r for r in rows if r.test_name == "picture_description"), None)
    language_payload = None
    if language_row:
        d = language_row.detail_json or {}
        language_payload = {
            "probability": language_row.score if d.get("available") else None,
            "reason": d.get("reason"),
            "word_count": d.get("word_count"),
            "contributions": d.get("contributions", []),
            "timing": d.get("timing", {}),
            "transcript": (language_row.raw_response or {}).get("raw_transcript"),
        }

    return AssessmentResult(
        session_id=s.id,
        tests=[TestResultOut.model_validate(r) for r in rows],
        language=language_payload,
        **s.result_json,
    )
