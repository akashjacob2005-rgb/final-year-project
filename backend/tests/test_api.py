"""End-to-end API tests: auth, a full six-test assessment, history.

Runs against a throwaway SQLite file and never touches the dev database.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest

# Must be set before app modules import settings.
_TMP_DB = Path(tempfile.mkdtemp()) / "test.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402

USER = {
    "email": "ada@example.com",
    "password": "lovelace123",
    "full_name": "Ada Lovelace",
    "age": 68,
    "education_years": 16,
}


@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def auth_client(client):
    r = client.post("/api/auth/signup", json=USER)
    assert r.status_code == 201, r.text
    return client


# --------------------------------------------------------------------------
# auth
# --------------------------------------------------------------------------
def test_health(client):
    r = client.get("/api/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_signup_rejects_weak_password(client):
    r = client.post("/api/auth/signup", json={**USER, "email": "x@y.com", "password": "12345678"})
    assert r.status_code == 422


def test_signup_rejects_duplicate_email(auth_client):
    r = auth_client.post("/api/auth/signup", json=USER)
    assert r.status_code == 409


def test_me_requires_auth(client):
    fresh = TestClient(app)
    assert fresh.get("/api/auth/me").status_code == 401


def test_login_and_me(auth_client):
    r = auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    assert r.status_code == 200, r.text

    r = auth_client.get("/api/auth/me")
    assert r.status_code == 200
    assert r.json()["email"] == USER["email"]
    assert r.json()["age"] == 68


def test_login_wrong_password_is_generic(auth_client):
    r = auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": "wrongwrong1"}
    )
    assert r.status_code == 401
    # Must not reveal whether the account exists.
    assert r.json()["detail"] == "Incorrect email or password"

    r = auth_client.post(
        "/api/auth/login", json={"email": "nobody@example.com", "password": "wrongwrong1"}
    )
    assert r.status_code == 401
    assert r.json()["detail"] == "Incorrect email or password"


def test_refresh_rotates_session(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    assert auth_client.post("/api/auth/refresh").status_code == 200
    assert auth_client.get("/api/auth/me").status_code == 200


# --------------------------------------------------------------------------
# assessment
# --------------------------------------------------------------------------
def _perfect_submissions(stimuli: dict) -> list[dict]:
    """Responses a cognitively healthy user would produce."""
    trials = [
        {"response": list(reversed(t["sequence"]))}
        for t in stimuli["digit_span_backward"]["trials"]
    ]
    return [
        {
            "test_name": "memory_recognition",
            "raw_response": {"selected": stimuli["memory_recognition"]["targets"]},
            "duration_ms": 9000,
        },
        {
            "test_name": "trail_making_b",
            "raw_response": {
                "clicked": ["1", "A", "2", "B", "3", "C", "4", "D", "5", "E"],
                "duration_ms": 19000,
            },
            "duration_ms": 19000,
        },
        {
            "test_name": "serial_sevens",
            "raw_response": {"responses": [93, 86, 79, 72, 65]},
            "duration_ms": 30000,
        },
        {
            "test_name": "digit_span_backward",
            "raw_response": {"trials": trials},
            "duration_ms": 40000,
        },
        {
            "test_name": "verbal_fluency",
            "raw_response": {
                "words": [
                    stimuli["verbal_fluency"]["letter"].lower() + s
                    for s in ["ish", "ast", "ather", "riend", "lower", "orest",
                              "arm", "unny", "ence", "loor", "rame", "ruit"]
                ]
            },
            "duration_ms": 60000,
        },
    ]


CONTROL_LIKE = (
    "the mother is standing at the sink drying the dishes and the water is "
    "running over onto the floor . the little boy is up on a stool reaching "
    "into the cookie jar in the cupboard and the stool is tipping over . the "
    "girl is standing below him with her hand up asking for a cookie . the "
    "window is open and there are curtains blowing and you can see the garden "
    "and the grass outside . it seems to be a warm summer day in the kitchen ."
)


def test_full_assessment_flow(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )

    r = auth_client.post("/api/assessment/start")
    assert r.status_code == 201, r.text
    body = r.json()
    session_id = body["session"]["id"]
    stimuli = body["stimuli"]

    # The expected trail path must NOT be handed to the client.
    assert "expected" not in stimuli["trail_making_b"]
    assert len(stimuli["memory_recognition"]["targets"]) == 5
    assert len(stimuli["memory_recognition"]["grid"]) == 15

    # Re-read the server-side stimuli (with answers) for building responses.
    from app.database import SessionLocal
    from app.models import AssessmentSession

    with SessionLocal() as db:
        full_stimuli = db.get(AssessmentSession, session_id).stimuli_json

    for sub in _perfect_submissions(full_stimuli):
        r = auth_client.post(f"/api/assessment/{session_id}/test", json=sub)
        assert r.status_code == 200, f"{sub['test_name']}: {r.text}"

    # Language test via the typed fallback (no audio needed in tests).
    r = auth_client.post(
        f"/api/assessment/{session_id}/transcript",
        json={"transcript": CONTROL_LIKE, "duration_ms": 45000},
    )
    assert r.status_code == 200, r.text
    assert r.json()["language"]["available"] is True

    r = auth_client.post(f"/api/assessment/{session_id}/complete")
    assert r.status_code == 200, r.text
    result = r.json()

    assert 0.0 <= result["risk"] <= 1.0
    assert result["band"] in {"Low", "Borderline", "Elevated"}
    assert result["confidence"] == "high"
    assert len(result["tests"]) == 6
    assert result["components"]["language_probability"] is not None
    # A strong performance on every test should not read as elevated risk.
    assert result["band"] == "Low", result

    # Completed sessions are locked.
    r = auth_client.post(
        f"/api/assessment/{session_id}/test",
        json={"test_name": "serial_sevens", "raw_response": {"responses": [1, 2, 3]}},
    )
    assert r.status_code == 409


def test_impaired_pattern_scores_higher(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    r = auth_client.post("/api/assessment/start")
    session_id = r.json()["session"]["id"]

    from app.database import SessionLocal
    from app.models import AssessmentSession

    with SessionLocal() as db:
        stim = db.get(AssessmentSession, session_id).stimuli_json

    poor = [
        {
            "test_name": "memory_recognition",
            # two hits and two false alarms
            "raw_response": {
                "selected": stim["memory_recognition"]["targets"][:2]
                + [g for g in stim["memory_recognition"]["grid"]
                   if g not in stim["memory_recognition"]["targets"]][:2]
            },
        },
        {
            "test_name": "trail_making_b",
            "raw_response": {
                "clicked": ["1", "A", "2", "B", "3", "C", "4", "D", "5", "E"],
                "duration_ms": 110000,
            },
        },
        {"test_name": "serial_sevens", "raw_response": {"responses": [93, 80, None, None, None]}},
        {"test_name": "digit_span_backward", "raw_response": {"trials": [{"response": [1, 2, 3]}]}},
        {"test_name": "verbal_fluency", "raw_response": {"words": ["fish", "fish", "apple"]}},
    ]
    for sub in poor:
        assert auth_client.post(f"/api/assessment/{session_id}/test", json=sub).status_code == 200

    r = auth_client.post(f"/api/assessment/{session_id}/complete")
    assert r.status_code == 200
    result = r.json()

    # No language input was given, so the score must fall back to structured
    # only, say so, and lower its own confidence.
    assert result["components"]["language_probability"] is None
    assert result["weights"]["structured"] == 1.0
    assert result["confidence"] in {"moderate", "low"}
    assert result["band"] in {"Borderline", "Elevated"}


def test_cannot_read_another_users_session(client, auth_client):
    other = TestClient(app)
    other.post(
        "/api/auth/signup",
        json={**USER, "email": "mallory@example.com", "full_name": "Mallory"},
    )
    r = other.post("/api/assessment/start")
    victim_session = r.json()["session"]["id"]

    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    assert auth_client.get(f"/api/assessment/{victim_session}").status_code == 404


# --------------------------------------------------------------------------
# history
# --------------------------------------------------------------------------
def test_history_and_trend(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )

    r = auth_client.get("/api/history/sessions")
    assert r.status_code == 200
    assert len(r.json()) >= 2

    r = auth_client.get("/api/history/trend")
    assert r.status_code == 200
    assert r.json()["count"] >= 2

    r = auth_client.get("/api/history/summary")
    assert r.status_code == 200
    assert r.json()["total_assessments"] >= 2


def test_model_info_reports_real_metrics(client):
    r = client.get("/api/history/model-info")
    assert r.status_code == 200
    body = r.json()
    if body.get("available"):
        assert body["dataset"]["n_total"] > 0
        assert any(m["roc_auc"] > 0.5 for m in body["models"])
        assert body["caveats"]


def test_logout_clears_session(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    assert auth_client.post("/api/auth/logout").status_code == 200
    assert auth_client.get("/api/auth/me").status_code == 401
