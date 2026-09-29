"""Pause-encoded transcripts: hesitation written where it happened.

These tests exercise pause_encoded_transcript() directly. No audio and no Whisper
model is involved — the encoder consumes the word-timestamp list that
speech.transcribe() already builds, so it can be tested on synthetic words.

The invariant that matters most is the last test: the markers and the timing
scalars must always agree about what counts as a pause. They share thresholds
today; this test fails loudly if anyone later gives them separate ones.
"""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

# Must be set before app modules import settings.
_TMP_DB = Path(tempfile.mkdtemp()) / "test_speech.db"
os.environ.setdefault("DATABASE_URL", f"sqlite:///{_TMP_DB}")
os.environ.setdefault("ENVIRONMENT", "test")
os.environ.setdefault("SECRET_KEY", "test-secret-key-not-for-production")

from app.ml import speech  # noqa: E402

# scikit-learn's default token_pattern. The markers are useless if the
# vectorizer does not keep them as standalone tokens.
SKLEARN_TOKEN_RE = re.compile(r"(?u)\b\w\w+\b")


def words_from(spec: list[tuple[str, float, float]]) -> list[dict]:
    return [{"word": w, "start": s, "end": e} for w, s, e in spec]


def test_long_gap_becomes_longpause():
    out = speech.pause_encoded_transcript(
        words_from([("the", 0.0, 0.2), ("cookie", 1.5, 1.9)])
    )
    assert out == "the longpause cookie"


def test_short_gap_becomes_shortpause():
    out = speech.pause_encoded_transcript(
        words_from([("the", 0.0, 0.2), ("cookie", 0.7, 1.0)])
    )
    assert out == "the shortpause cookie"


def test_fluent_speech_gets_no_marker():
    out = speech.pause_encoded_transcript(
        words_from([("the", 0.0, 0.2), ("boy", 0.25, 0.4), ("fell", 0.45, 0.7)])
    )
    assert out == "the boy fell"
    assert speech.SHORT_PAUSE_TOKEN not in out
    assert speech.LONG_PAUSE_TOKEN not in out


def test_thresholds_are_inclusive():
    """A gap exactly on the threshold counts, matching _timing_features' >=."""
    short = speech.pause_encoded_transcript(
        words_from([("a", 0.0, 0.0), ("b", speech.PAUSE_THRESHOLD_S, 1.0)])
    )
    long_ = speech.pause_encoded_transcript(
        words_from([("a", 0.0, 0.0), ("b", speech.LONG_PAUSE_THRESHOLD_S, 2.0)])
    )
    assert short == "a shortpause b"
    assert long_ == "a longpause b"


def test_no_words_returns_empty():
    assert speech.pause_encoded_transcript([]) == ""


def test_single_word_has_nothing_to_pause_between():
    assert speech.pause_encoded_transcript(words_from([("cookie", 0.0, 0.4)])) == "cookie"


def test_output_is_normalised_like_the_plain_transcript():
    """Whisper attaches punctuation to words; the encoder must still emit text in
    the CHAT-ish style the vectorizer was fitted on."""
    out = speech.pause_encoded_transcript(
        words_from([("Boy,", 0.0, 0.3), ("Cookies!", 1.6, 2.0)])
    )
    assert out == "boy longpause cookies ."


def test_markers_survive_the_sklearn_tokenizer():
    out = speech.pause_encoded_transcript(
        words_from([("the", 0.0, 0.2), ("uh", 0.6, 0.8), ("jar", 2.0, 2.4)])
    )
    tokens = SKLEARN_TOKEN_RE.findall(out)
    assert speech.SHORT_PAUSE_TOKEN in tokens
    assert speech.LONG_PAUSE_TOKEN in tokens
    # and the bigram that carries the actual signal is formable
    assert tokens[tokens.index(speech.LONG_PAUSE_TOKEN) + 1] == "jar"


def test_markers_agree_with_the_timing_features():
    """Same word list -> marker counts must equal the timing scalars.

    Guards the shared-threshold contract between the two code paths.
    """
    words = words_from(
        [
            ("the", 0.0, 0.2),
            ("boy", 0.25, 0.4),     # fluent
            ("is", 0.9, 1.0),       # short pause (0.5s)
            ("um", 2.4, 2.6),       # long pause (1.4s)
            ("falling", 2.65, 3.1),  # fluent
        ]
    )
    encoded = speech.pause_encoded_transcript(words)
    timing = speech._timing_features(words, duration=3.1)

    tokens = SKLEARN_TOKEN_RE.findall(encoded)
    n_long = tokens.count(speech.LONG_PAUSE_TOKEN)
    n_short = tokens.count(speech.SHORT_PAUSE_TOKEN)

    assert n_long == timing["long_pause_count"] == 1
    assert n_short + n_long == timing["pause_count"] == 2
