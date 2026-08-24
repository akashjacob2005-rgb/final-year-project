"""End-to-end API tests: auth, a full six-test assessment, history.

Runs against a throwaway SQLite file and never touches the dev database.
"""

from __future__ import annotations

import os
import re
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


# --------------------------------------------------------------------------
# age-scaled difficulty
#
# Two mechanisms scale with age: the stimuli here, and the normative z-scores in
# ml/scoring.py. They must agree. If the tiers and the norm bands ever drift
# apart, a healthy young user gets a harder test scored against an easier norm
# and is flagged as declining — which is the failure these tests exist to catch.
# --------------------------------------------------------------------------
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "ml"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "app" / "ml"))

import scoring  # noqa: E402
import stimuli  # noqa: E402

TIER_AGES = {"demanding": 25, "challenging": 50, "moderate": 68, "standard": 80}


def test_tier_boundaries_are_exact():
    for age, expected in [
        (18, "demanding"), (39, "demanding"),
        (40, "challenging"), (59, "challenging"),
        (60, "moderate"), (74, "moderate"),
        (75, "standard"), (120, "standard"),
    ]:
        assert stimuli.tier_for_age(age) == expected, age
    # Unknown age must reproduce the app's original difficulty, not guess.
    assert stimuli.tier_for_age(None) == "moderate"


def test_norm_bands_match_tier_bands():
    """The single most important invariant in the difficulty feature."""
    tier_bands = {(lo, hi) for _, lo, hi in stimuli.TIER_BANDS}
    for test, spec in scoring.NORMS.items():
        assert set(spec["bands"]) == tier_bands, test


def test_difficulty_increases_as_age_decreases():
    shapes = {}
    for tier, age in TIER_AGES.items():
        s = stimuli.generate(seed=7, age=age)
        assert s["tier"] == tier
        shapes[tier] = (
            len(s["memory_recognition"]["targets"]),
            len(s["digit_span_backward"]["trials"]),
            len(s["trail_making_b"]["nodes"]),
        )
    order = ["standard", "moderate", "challenging", "demanding"]
    for a, b in zip(order, order[1:]):
        assert all(x <= y for x, y in zip(shapes[a], shapes[b])), (a, b, shapes)


def test_digit_span_ladder_always_starts_at_three():
    """A ladder starting above 3 would score a true span of 4 as 0.

    Scoring records only sequences reproduced correctly, so on a 5-8 ladder
    someone with a span of 4 fails every trial and their raw measure jumps
    straight from 0 to 5 — reading a mild deficit as a profound one.
    """
    for age in TIER_AGES.values():
        trials = stimuli.generate(seed=3, age=age)["digit_span_backward"]["trials"]
        assert trials[0]["length"] == 3, age
        lengths = [t["length"] for t in trials]
        assert lengths == sorted(lengths)
        assert lengths == list(range(3, 3 + len(lengths)))


def test_model_scored_test_is_never_tiered():
    """picture_description must be byte-identical across tiers.

    The language model was fitted on descriptions of this exact scene with a
    25-word floor; varying the stimulus by age would silently invalidate it.
    """
    specs = {
        repr(stimuli.generate(seed=1, age=a)["picture_description"])
        for a in TIER_AGES.values()
    }
    assert len(specs) == 1
    # Verbal fluency is also untiered: a 60s word count has no ceiling to raise.
    durations = {
        stimuli.generate(seed=1, age=a)["verbal_fluency"]["duration_seconds"]
        for a in TIER_AGES.values()
    }
    assert durations == {60}


def test_typical_performer_scores_z_near_zero_in_every_tier():
    """Guards the double-penalty bug: harder stimuli with unchanged norms."""
    for age in TIER_AGES.values():
        for test, spec in scoring.NORMS.items():
            band = next(b for b in spec["bands"] if b[0] <= age <= b[1])
            mean = spec["bands"][band][0]
            z = scoring.normative_z(test, mean, age, education_years=14)
            assert abs(z) < 0.01, (test, age, z)


def test_memory_encoding_time_is_feasible():
    """Study time must scale with the number of objects.

    Regression test. A tier once showed 7 objects for 6 seconds — 0.86s each,
    against 1.60s in the original 5-object test — and healthy users scored 0/7,
    which then dominated their whole assessment. Difficulty belongs in how much
    is remembered and for how long, never in how fast the list flashes past.
    """
    for tier, params in stimuli.TIER_PARAMS.items():
        per_object = params["study_seconds"] / params["memory_targets"]
        assert per_object >= stimuli.MIN_SECONDS_PER_OBJECT, (
            f"{tier}: {per_object:.2f}s per object is too fast to encode"
        )


def test_z_scores_stay_in_a_defensible_range():
    """Even a floor score must not produce an absurd z.

    A bottomed-out test used to return z = -6.4, which swamped the domain
    average. Beyond +/-4 the value means 'floor' or 'ceiling', not a finer
    clinical grade.
    """
    for age in TIER_AGES.values():
        for test in scoring.NORMS:
            for raw in (0.0, 1.0, 999.0):
                z = scoring.normative_z(test, raw, age, education_years=10)
                assert abs(z) <= scoring.Z_CLAMP, (test, age, raw, z)


def test_memory_grid_is_well_formed_in_every_tier():
    for age in TIER_AGES.values():
        mem = stimuli.generate(seed=5, age=age)["memory_recognition"]
        assert set(mem["targets"]) <= set(mem["grid"])
        assert len(set(mem["grid"])) == len(mem["grid"])       # no duplicates
        assert len(mem["grid"]) == 3 * len(mem["targets"])     # 1 target : 2 distractors


def test_trail_nodes_stay_on_screen_in_every_tier():
    for age in TIER_AGES.values():
        tr = stimuli.generate(seed=11, age=age)["trail_making_b"]
        assert len(tr["nodes"]) == len(tr["expected"])
        for n in tr["nodes"]:
            assert 0 < n["x"] < 100, n
            assert 0 < n["y"] < 100, n


def test_client_never_receives_the_trail_answer():
    safe = stimuli.client_safe(stimuli.generate(seed=2, age=25))
    assert "expected" not in safe["trail_making_b"]
    assert safe["tier"] == "demanding"  # non-dict values survive the copy


def test_session_records_the_tier_it_was_sat_at(auth_client):
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    r = auth_client.post("/api/assessment/start")
    assert r.status_code == 201
    # USER is 68 -> moderate, the app's original difficulty.
    assert r.json()["stimuli"]["tier"] == "moderate"
    assert len(r.json()["stimuli"]["memory_recognition"]["targets"]) == 5


# --------------------------------------------------------------------------
# password reset
# --------------------------------------------------------------------------
RESET_USER = {
    "email": "grace@example.com",
    "password": "hopper1906",
    "full_name": "Grace Hopper",
    "age": 45,
    "education_years": 18,
}


def _issue_reset_token(client, caplog, email):
    """Request a reset and recover the token from the log-only mailer."""
    with caplog.at_level("WARNING", logger="neuroguard.mailer"):
        r = client.post("/api/auth/forgot-password", json={"email": email})
    assert r.status_code == 200
    match = re.search(r"reset-password\?token=(\S+)", caplog.text)
    return r, (match.group(1) if match else None)


def test_forgot_password_does_not_reveal_whether_account_exists(client, caplog):
    assert client.post("/api/auth/signup", json=RESET_USER).status_code == 201

    known, token = _issue_reset_token(client, caplog, RESET_USER["email"])
    caplog.clear()
    unknown, no_token = _issue_reset_token(client, caplog, "ghost@example.com")

    assert known.status_code == unknown.status_code == 200
    assert known.json() == unknown.json()
    # The real one issued a token; the phantom one did not — but the caller
    # cannot tell, because the token never appears in the response.
    assert token and not no_token
    assert "token" not in known.text.lower()


def test_reset_password_rejects_bad_tokens(client):
    r = client.post(
        "/api/auth/reset-password",
        json={"token": "n" * 40, "new_password": "freshpass1"},
    )
    assert r.status_code == 400
    # Weak passwords are refused on reset exactly as they are on signup.
    r = client.post(
        "/api/auth/reset-password",
        json={"token": "n" * 40, "new_password": "12345678"},
    )
    assert r.status_code == 422


def test_reset_password_changes_login_and_is_single_use(client, caplog):
    _, token = _issue_reset_token(client, caplog, RESET_USER["email"])
    assert token

    r = client.post(
        "/api/auth/reset-password",
        json={"token": token, "new_password": "newsecret42"},
    )
    assert r.status_code == 200

    # Replaying a consumed link must fail, and fail indistinguishably from an
    # expired or fabricated one.
    assert client.post(
        "/api/auth/reset-password",
        json={"token": token, "new_password": "another42x"},
    ).status_code == 400

    client.cookies.clear()
    assert client.post(
        "/api/auth/login",
        json={"email": RESET_USER["email"], "password": RESET_USER["password"]},
    ).status_code == 401
    assert client.post(
        "/api/auth/login",
        json={"email": RESET_USER["email"], "password": "newsecret42"},
    ).status_code == 200


def test_reset_revokes_existing_sessions(client, caplog):
    """A reset prompted by a compromise must evict whoever is already in."""
    client.cookies.clear()
    assert client.post(
        "/api/auth/login",
        json={"email": RESET_USER["email"], "password": "newsecret42"},
    ).status_code == 200
    assert client.get("/api/auth/me").status_code == 200

    _, token = _issue_reset_token(client, caplog, RESET_USER["email"])
    assert client.post(
        "/api/auth/reset-password",
        json={"token": token, "new_password": "thirdpass77"},
    ).status_code == 200

    assert client.post("/api/auth/refresh").status_code == 401


# --------------------------------------------------------------------------
# fusion safety override
#
# Averaging two components lets one be quietly outvoted. With a healthy-sounding
# transcript, failing every structured test gave risk 0.63 against a 0.65
# Elevated threshold — Borderline, missing the band by 0.02. These tests pin the
# override that fixes it, and more importantly pin the boundary where it must
# stay silent.
# --------------------------------------------------------------------------
import fusion  # noqa: E402

_GOOD_Z = {"memory": 1.0, "executive": 1.5, "attention": 0.7,
           "working_memory": 2.3, "language": 0.6}
_BELOW_PAR_Z = {"memory": -1.5, "executive": -1.0, "attention": -1.2,
                "working_memory": -1.0, "language": -1.3}
_FAILED_Z = {k: -4.0 for k in _GOOD_Z}
_HEALTHY_SPEECH = 0.26


def test_structured_failure_escalates_despite_good_speech():
    """The exact case that used to fall 0.02 short of Elevated."""
    r = fusion.fuse(_HEALTHY_SPEECH, _FAILED_Z, 6, 6)
    assert r["risk"] < fusion.BAND_ELEVATED       # score alone says Borderline
    assert r["band"] == "Elevated"                 # override says otherwise
    assert r["band_escalated"] is True
    assert r["escalation_reason"]
    assert any("below expectation" in n for n in r["notes"])


def test_atypical_speech_escalates_despite_good_structured_tests():
    r = fusion.fuse(0.95, _GOOD_Z, 6, 6)
    assert r["band"] == "Elevated"
    assert r["band_escalated"] is True


def test_override_stays_silent_on_ordinary_poor_performance():
    """The guard that matters most.

    Someone clearly below par but not impaired must NOT be escalated — an
    override that fires too readily is worse than the gap it was added to close.
    """
    r = fusion.fuse(_HEALTHY_SPEECH, _BELOW_PAR_Z, 6, 6)
    assert r["components"]["composite_z"] > fusion.SEVERE_COMPOSITE_Z
    assert r["band"] == "Borderline"
    assert r["band_escalated"] is False
    assert r["escalation_reason"] is None


def test_healthy_result_is_untouched():
    r = fusion.fuse(0.13, _GOOD_Z, 6, 6)
    assert r["band"] == "Low"
    assert r["band_escalated"] is False


def test_already_elevated_is_not_double_flagged():
    """Escalate only. A top-band result must not be re-escalated or re-noted."""
    r = fusion.fuse(0.94, _FAILED_Z, 6, 6)
    assert r["band"] == "Elevated"
    assert r["band_escalated"] is False
    assert r["escalation_reason"] is None


def test_override_never_alters_the_score():
    """The override is presentation only — the arithmetic must be untouched.

    The results page shows the headline as the weighted mean of its two
    components. If escalation moved `risk`, that displayed sum would silently
    stop adding up.
    """
    for lang, dz in [(_HEALTHY_SPEECH, _FAILED_Z), (0.95, _GOOD_Z),
                     (_HEALTHY_SPEECH, _BELOW_PAR_Z)]:
        r = fusion.fuse(lang, dz, 6, 6)
        p_struct, _ = fusion.structured_probability(dz)
        expected = fusion.WEIGHT_LANGUAGE * lang + fusion.WEIGHT_STRUCTURED * p_struct
        assert abs(r["risk"] - round(expected, 4)) < 1e-9
        assert r["score_out_of_10"] == round((1 - expected) * 10, 1)


# --------------------------------------------------------------------------
# stored-result backward compatibility
#
# GET /assessment/{id} rehydrates AssessmentResult straight from the stored
# `result_json`, so every field added to that model must tolerate rows written by
# older code. Adding `score_out_of_10` without a default broke the "Full
# breakdown" button for every existing assessment. The frozen payload below is
# what fuse() produced before that change; it must keep constructing forever.
# --------------------------------------------------------------------------
from app.schemas import AssessmentResult  # noqa: E402

_LEGACY_RESULT_JSON = {
    "risk": 0.4594,
    "risk_percent": 45.9,
    "band": "Borderline",
    "confidence": "high",
    "components": {
        "language_probability": 0.1289,
        "structured_probability": 0.7899,
        "composite_z": -1.883,
    },
    "weights": {"language": 0.5, "structured": 0.5},
    "domain_z": {"memory": 0.778, "executive": -1.921, "attention": -2.583},
    "weakest_domain": "working_memory",
    "notes": [],
    "method": "transparent weighted combination (not a learned fusion model)",
    "disclaimer": "This is a screening indicator for research and educational use.",
}


def test_legacy_result_json_still_loads():
    r = AssessmentResult(session_id=1, tests=[], language=None, **_LEGACY_RESULT_JSON)
    # Derived from risk, not defaulted to 0.0 — an old healthy result must not
    # be rendered as a catastrophic one.
    assert r.score_out_of_10 == round((1 - _LEGACY_RESULT_JSON["risk"]) * 10, 1)
    assert r.band == "Borderline"
    # The safety override did not exist when this was scored; it must not be
    # applied retroactively.
    assert r.band_escalated is False
    assert r.escalation_reason is None


def test_current_result_json_score_is_not_recomputed():
    """The backfill must only fill a gap, never override a stored value."""
    payload = {**_LEGACY_RESULT_JSON, "score_out_of_10": 9.9}
    r = AssessmentResult(session_id=1, tests=[], language=None, **payload)
    assert r.score_out_of_10 == 9.9


def test_full_breakdown_endpoint_serves_a_completed_session(auth_client):
    """The actual regression: GET /assessment/{id} returned 500."""
    auth_client.post(
        "/api/auth/login", json={"email": USER["email"], "password": USER["password"]}
    )
    sessions = auth_client.get("/api/history/sessions").json()
    assert sessions, "expected at least one completed session from earlier tests"

    r = auth_client.get(f"/api/assessment/{sessions[0]['session_id']}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert 0.0 <= body["score_out_of_10"] <= 10.0
    assert body["band"] in {"Low", "Borderline", "Elevated"}


def test_digit_span_round_counts():
    """Explicit product requirement: 4 rounds under 40, 3 rounds at 40 and over.

    A backward span of 7-8 is well past typical adult ability (normal is about
    4-5), so the deep ladder measured almost nobody and just felt punishing.
    Three rounds rather than two keeps a middle grade — with only a 3-4 ladder
    the raw score could land on just 0, 3 or 4, which is too coarse to grade
    working memory in the older tiers that most need grading.
    """
    rounds = {
        tier: len(stimuli.TIER_PARAMS[tier]["digit_lengths"])
        for tier, _, _ in stimuli.TIER_BANDS
    }
    assert rounds["demanding"] == 4        # 18-39
    assert rounds["challenging"] == 3      # 40-59
    assert rounds["moderate"] == 3         # 60-74
    assert rounds["standard"] == 3         # 75+


def test_no_norm_sits_above_its_own_ceiling():
    """A perfect score must never produce a negative z.

    Shortening a ladder without re-referencing its norm silently punishes
    flawless performance: on a 2-rung ladder the best raw is 4, and against the
    old mean of 4.6 that scored z = -0.50. Any future change to
    digit_lengths must move the matching band in NORMS with it.
    """
    for tier, lo, hi in stimuli.TIER_BANDS:
        best = max(stimuli.TIER_PARAMS[tier]["digit_lengths"])
        age = lo + 1
        z = scoring.normative_z("digit_span_backward", best, age, education_years=16)
        assert z > 0, (
            f"{tier}: a perfect span of {best} scores z={z:+.2f}; "
            f"the norm mean is above the test's ceiling"
        )
