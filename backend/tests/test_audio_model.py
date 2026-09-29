"""The audio-trained model at inference: block layout, model selection, fallback.

No real artifact is needed. Each test builds a TINY but genuine one — real
TfidfVectorizer, real StandardScalers, real CalibratedClassifierCV — assembled in
exactly the block order ml/train_audio_model.py::build_matrix uses. Mocks would
not catch the failure these tests exist for.

That failure is column alignment. The design matrix is

    [ handcrafted (27) | tfidf (len(vocabulary_)) | timing (9) ]

so the timing block does NOT start at column 27; it starts at 27 + vocabulary
size. Nothing raises if that offset is computed wrongly — the model still returns
a probability, and the results page still renders. It just attributes pronoun
weights to pause counts. Only an explicit assertion catches it.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import joblib
import numpy as np
import pytest

# Must be set before app modules import settings.
_TMP_DB = Path(tempfile.mkdtemp()) / "test_audio_model.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP_DB}")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")

from sklearn.calibration import CalibratedClassifierCV  # noqa: E402
from sklearn.feature_extraction.text import TfidfVectorizer  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from app.api.routes.assessment import _score_transcript  # noqa: E402
from app.ml import language_model as lm  # noqa: E402

# Long enough to clear MIN_WORDS (25) so the floor is not what is being tested.
_SCENE = (
    "the mother is washing the dishes and the water is running over the sink "
    "while the boy is climbing on the stool to reach the cookie jar and the "
    "girl is holding her hand out to take one from him"
)
_VAGUE = (
    "well there is a thing there and she is doing that with it and he is up on "
    "the thing getting the other thing down and she wants it too i think that "
    "is all i can see in this one really"
)


def _timing(**over) -> dict:
    base = {
        "duration_seconds": 60.0,
        "word_count": 90,
        "speech_rate_wpm": 90.0,
        "pause_count": 6,
        "long_pause_count": 2,
        "total_pause_seconds": 8.0,
        "mean_pause_seconds": 1.3,
        "pause_ratio": 0.13,
        "articulation_rate_wpm": 104.0,
    }
    base.update(over)
    return base


def _make_artifact(mode: str, text_column: str = "transcript") -> dict:
    """A real fitted artifact in the schema train_audio_model.py writes."""
    texts = [_SCENE, _VAGUE] * 4
    y = np.array([0, 1] * 4)
    timings = [_timing(pause_count=2 + 6 * label) for label in y]

    feats = [lm.extract_features(t) for t in texts]
    has_hand, has_tfidf, has_timing = lm._blocks(mode)

    parts, vec, hs, ts = [], None, None, None
    if has_hand:
        H = np.array([[f.get(n, 0.0) for n in lm.FEATURE_NAMES] for f in feats])
        H = np.nan_to_num(H, posinf=0.0, neginf=0.0)
        hs = StandardScaler().fit(H)
        parts.append(hs.transform(H))
    if has_tfidf:
        # min_df=1 rather than the trainer's 3: eight synthetic documents, and the
        # point here is block WIDTH, not vocabulary quality.
        vec = TfidfVectorizer(ngram_range=(1, 2), min_df=1).fit(texts)
        parts.append(vec.transform(texts).toarray())
    if has_timing:
        T = np.array([[t[n] for n in lm.TIMING_LABELS] for t in timings], dtype=float)
        ts = StandardScaler().fit(T)
        parts.append(ts.transform(T))

    X = np.hstack(parts)
    model = CalibratedClassifierCV(
        LogisticRegression(max_iter=1000, class_weight="balanced"), cv=2, method="sigmoid"
    ).fit(X, y)

    return {
        "model": model,
        "vectorizer": vec,
        "scaler": hs,
        "timing_scaler": ts,
        "mode": mode,
        "feature_names": lm.FEATURE_NAMES,
        "timing_feature_names": list(lm.TIMING_LABELS),
        "text_column": text_column,
        "hand_features_from": "transcript",
        "version": "test-2.0.0",
    }


@pytest.fixture
def install_audio(tmp_path, monkeypatch):
    """Install a synthetic audio artifact and clear the module's caches."""

    def _install(mode="text+timing", text_column="transcript"):
        path = tmp_path / "audio_model.joblib"
        joblib.dump(_make_artifact(mode, text_column), path)
        monkeypatch.setattr(lm.paths, "AUDIO_MODEL_PATH", path)
        lm._audio_artifact.cache_clear()
        return path

    yield _install
    lm._audio_artifact.cache_clear()


@pytest.fixture
def no_audio(tmp_path, monkeypatch):
    monkeypatch.setattr(lm.paths, "AUDIO_MODEL_PATH", tmp_path / "absent.joblib")
    lm._audio_artifact.cache_clear()
    yield
    lm._audio_artifact.cache_clear()


# --------------------------------------------------------------------------
# block layout — the two trainers name their modes differently
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "mode,expected",
    [
        # ml/train_language_model.py
        ("hand", (True, False, False)),
        ("tfidf", (False, True, False)),
        ("combined", (True, True, False)),
        # ml/train_audio_model.py — "text" means handcrafted AND tfidf
        ("text", (True, True, False)),
        ("timing", (False, False, True)),
        ("text+timing", (True, True, True)),
    ],
)
def test_blocks_reconciles_both_trainers(mode, expected):
    assert lm._blocks(mode) == expected


def test_timing_columns_sit_after_the_tfidf_block(install_audio):
    """The offset bug this file exists for: timing is at 27 + vocab, not 27."""
    install_audio(mode="text+timing")
    art = lm._audio_artifact()
    vocab = len(art["vectorizer"].vocabulary_)
    assert vocab > 0, "a zero-width tfidf block would make this test vacuous"

    cols = lm._explainable_columns(art)
    by_name = {name: idx for name, _, idx in cols}

    # handcrafted block first, in FEATURE_NAMES order
    assert by_name[lm.FEATURE_NAMES[0]] == 0
    assert by_name[lm.FEATURE_NAMES[-1]] == len(lm.FEATURE_NAMES) - 1

    # timing block after handcrafted AND tfidf
    timing_names = list(lm.TIMING_LABELS)
    assert by_name[timing_names[0]] == len(lm.FEATURE_NAMES) + vocab
    assert by_name[timing_names[-1]] == len(lm.FEATURE_NAMES) + vocab + len(timing_names) - 1

    # tfidf columns are counted but never labelled for the user
    assert len(cols) == len(lm.FEATURE_NAMES) + len(timing_names)


def test_explainable_columns_are_within_the_matrix(install_audio):
    """Every advertised column index must exist in the assembled matrix."""
    install_audio(mode="text+timing")
    art = lm._audio_artifact()
    feats = lm.extract_features(_SCENE)
    X = lm._build_matrix(art, _SCENE, feats, timing=_timing())
    assert all(i < X.shape[1] for _, _, i in lm._explainable_columns(art))


def test_timing_only_mode_has_no_handcrafted_explanation(install_audio):
    install_audio(mode="timing")
    art = lm._audio_artifact()
    names = {name for name, _, _ in lm._explainable_columns(art)}
    assert names == set(lm.TIMING_LABELS)
    assert lm.FEATURE_NAMES[0] not in names


# --------------------------------------------------------------------------
# predict_audio
# --------------------------------------------------------------------------
def test_predict_audio_returns_calibrated_probability(install_audio):
    install_audio()
    out = lm.predict_audio(_SCENE, _SCENE, _timing())
    assert out["available"] is True
    assert 0.0 <= out["probability"] <= 1.0
    assert out["model"] == "audio"
    assert out["model_version"] == "test-2.0.0"


def test_predict_audio_surfaces_timing_markers(install_audio):
    """Timing is a learned predictor now, so it must be explainable to the user."""
    install_audio()
    out = lm.predict_audio(_VAGUE, _VAGUE, _timing(pause_count=20, long_pause_count=12))
    features = {c["feature"] for c in out["contributions"]}
    assert features, "expected some contributions"
    # at least one reported marker comes from the timing block
    assert features & set(lm.TIMING_LABELS), f"no timing marker in {features}"
    for c in out["contributions"]:
        assert c["direction"] in ("increases risk", "lowers risk")


def test_predict_audio_enforces_the_word_floor(install_audio):
    install_audio()
    out = lm.predict_audio("the boy fell", "the boy fell", _timing())
    assert out["available"] is False
    assert out["probability"] is None
    assert str(lm.MIN_WORDS) in out["reason"]


def test_pause_marked_column_is_used_only_when_the_artifact_asks(install_audio):
    """text_column decides which text the vectorizer sees.

    Feeding pause-marked text to a vectorizer fitted on clean text (or vice versa)
    misaligns the vocabulary, so the two configurations must differ.
    """
    marked = _SCENE.replace("the boy", "longpause the shortpause boy")

    install_audio(mode="text+timing", text_column="transcript")
    clean_art = lm.predict_audio(_SCENE, marked, _timing())["probability"]

    install_audio(mode="text+timing", text_column="transcript_pauses")
    marked_art = lm.predict_audio(_SCENE, marked, _timing())["probability"]

    assert clean_art != marked_art


def test_missing_pause_text_degrades_to_the_clean_transcript(install_audio):
    """A cached session from before the encoder existed must still score."""
    install_audio(mode="text+timing", text_column="transcript_pauses")
    out = lm.predict_audio(_SCENE, None, _timing())
    assert out["available"] is True
    assert out["probability"] is not None


# --------------------------------------------------------------------------
# model selection and fallback
# --------------------------------------------------------------------------
def test_recording_uses_the_audio_model(install_audio):
    install_audio()
    assert lm.audio_available() is True
    assert _score_transcript(_SCENE, _SCENE, _timing())["model"] == "audio"


def test_typed_transcript_uses_the_text_model(install_audio):
    """No timing means the audio model cannot be used, artifact present or not.

    Zero-filling instead would standardise to 'spoke 0 words in 0 seconds' and
    push every typed submission toward the pathological end.
    """
    install_audio()
    assert _score_transcript(_SCENE, None, None)["model"] == "text"
    assert _score_transcript(_SCENE, None, {})["model"] == "text"


def test_missing_artifact_falls_back_instead_of_failing(no_audio):
    """A deployment without the audio artifact must still score recordings."""
    assert lm.audio_available() is False
    out = _score_transcript(_SCENE, _SCENE, _timing())
    assert out["model"] == "text"
    assert out["available"] is True
    assert out["probability"] is not None


# --------------------------------------------------------------------------
# the real artifact, when one has been trained
# --------------------------------------------------------------------------
_REAL = lm.paths.AUDIO_MODEL_PATH
_needs_real = pytest.mark.skipif(
    not _REAL.exists(),
    reason=f"no trained audio model at {_REAL}; run ml/train_audio_model.py",
)


@_needs_real
def test_real_artifact_matrix_width_matches_the_fitted_model():
    """Inference must build exactly the matrix training fitted.

    This is the one test that can catch a train/serve skew in the deployed
    artifact rather than in a synthetic one. A mismatch here means the blocks were
    assembled in a different order or a block was omitted — sklearn raises on the
    column count, so the assertion is really about failing in CI rather than on a
    user's recording.
    """
    lm._audio_artifact.cache_clear()
    art = lm._audio_artifact()
    feats = lm.extract_features(_SCENE)
    X = lm._build_matrix(art, _SCENE, feats, timing=_timing())

    has_hand, has_tfidf, has_timing = lm._blocks(art["mode"])
    expected = (
        (len(art["feature_names"]) if has_hand else 0)
        + (len(art["vectorizer"].vocabulary_) if has_tfidf else 0)
        + (len(art["timing_feature_names"]) if has_timing else 0)
    )
    assert X.shape == (1, expected)

    # and the fitted estimator accepts it
    prob = float(art["model"].predict_proba(X)[0, 1])
    assert 0.0 <= prob <= 1.0


@_needs_real
def test_real_artifact_was_trained_on_asr_text():
    """Guards the reason this model exists: training text must be Whisper output.

    If this artifact were ever retrained on clinician transcripts, the domain-shift
    fix would be silently undone while every metric still looked fine.
    """
    lm._audio_artifact.cache_clear()
    art = lm._audio_artifact()
    assert "whisper" in str(art.get("text_source", "")).lower()
    assert art.get("text_column") in ("transcript", "transcript_pauses")
    assert art.get("hand_features_from") == "transcript"


@_needs_real
def test_real_artifact_scores_a_recorded_session_end_to_end():
    """Score a genuine cached Pitt transcription, if the corpus is present.

    Uses the ASR cache rather than re-running Whisper, so the test stays fast and
    needs no audio decoding.
    """
    import json

    cache_dir = lm.paths.ML_DIR / "data" / "pitt" / "asr"
    cached = sorted(cache_dir.glob("*.json")) if cache_dir.is_dir() else []
    if not cached:
        pytest.skip("no ASR cache; run ml/build_dataset.py")

    for path in cached:
        data = json.loads(path.read_text())
        if len(data.get("transcript", "").split()) >= lm.MIN_WORDS:
            break
    else:
        pytest.skip("no cached transcription long enough to score")

    out = lm.predict_audio(
        data["transcript"], data.get("transcript_pauses"), data.get("timing")
    )
    assert out["available"] is True
    assert 0.0 <= out["probability"] <= 1.0
    assert out["model"] == "audio"
