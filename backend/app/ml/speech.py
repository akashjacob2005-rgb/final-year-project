"""Speech-to-text and speech-timing features for the picture-description test.

Two distinct things happen here, and what becomes of each depends on which model
is deployed:

1. TRANSCRIPTION feeds the trained model. Metrics in ml/artifacts/metrics.json
   (text model) and ml/artifacts/metrics_audio.json (audio model).
2. TIMING FEATURES (pause rate, hesitation, speech rate) are computed from word
   timestamps. Where they go depends on the artifact:

   * audio_model.joblib deployed -> they are LEARNED PREDICTORS. Trained on real
     DementiaBank Pitt audio, so a pause-rate weight is as validated as a
     pronoun-ratio one, and they enter the risk calculation.
   * text model only -> DESCRIPTIVE ONLY, as before. That model's corpus is the
     text-only HuggingFace mirror with no audio, so nothing could be fitted
     against these features and they are shown as observations, never as
     predictors.

   language_model._score_transcript picks between the two, and the per-result
   `model` field records which one ran. Do not describe timing features to the
   user as predictive without checking that field.
"""

from __future__ import annotations

import os
import re
import threading
from functools import lru_cache

MODEL_SIZE = os.getenv("WHISPER_MODEL", "base.en")
COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE", "int8")

# A silence longer than this between consecutive words counts as a pause.
PAUSE_THRESHOLD_S = 0.3
LONG_PAUSE_THRESHOLD_S = 1.0

# Markers spliced into the transcript by pause_encoded_transcript(). See its
# docstring for why these are bare words and not bracketed markup.
SHORT_PAUSE_TOKEN = "shortpause"
LONG_PAUSE_TOKEN = "longpause"

_lock = threading.Lock()


class TranscriptionError(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _whisper():
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:  # pragma: no cover
        raise TranscriptionError("faster-whisper is not installed") from exc

    # Prefer the cached weights. Without local_files_only the library still
    # contacts Hugging Face on every startup to check the revision, which makes
    # boot depend on outbound network even though the weights are baked into
    # the image. Fall back to downloading for a fresh local dev machine.
    try:
        return WhisperModel(
            MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE,
            local_files_only=True,
        )
    except Exception:
        return WhisperModel(MODEL_SIZE, device="cpu", compute_type=COMPUTE_TYPE)


def normalise_transcript(text: str) -> str:
    """Rewrite ASR output into the style of the training transcripts.

    The model was trained on CHAT-style clinical transcripts: lowercase, no
    capitalisation, sentence periods separated by spaces, no commas. Whisper
    emits conventionally punctuated prose. Without this step the TF-IDF
    vocabulary sees systematically different tokens at inference than it did in
    training, which quietly degrades accuracy.
    """
    t = (text or "").lower().strip()
    t = t.replace("...", " ")
    t = re.sub(r"[,;:\"“”()\[\]]", " ", t)
    t = re.sub(r"[!?]", ".", t)
    t = re.sub(r"\.", " . ", t)          # space-separated periods, as in training
    t = re.sub(r"[^a-z'.\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def pause_encoded_transcript(words: list[dict]) -> str:
    """Rewrite the transcript with hesitation marked where it actually happened.

    The timing features below summarise pauses into rates and counts, which
    throws away *where* the silence fell. Someone pausing before every noun
    ("the ... uh ... thing") is a different clinical picture from someone who
    pauses once to gather a thought, yet both produce the same pause_ratio.
    Splicing a marker into the word sequence lets the n-gram model learn
    "hesitated before this word" directly: TFIDF_KWARGS already sets
    ngram_range=(1, 2), so `longpause cookie` becomes a feature in its own right.

    Markers are bare alphabetic tokens rather than <angle-bracket> markup because
    normalise_transcript keeps only [a-z'.\\s], and scikit-learn's default
    token_pattern is \\b\\w\\w+\\b — either would strip the brackets anyway.
    Neither word occurs in natural Cookie Theft descriptions.

    Thresholds are the module's existing ones, so the markers and the timing
    scalars can never disagree about what counts as a pause.
    """
    if not words:
        return ""
    parts = [words[0]["word"]]
    for prev, cur in zip(words, words[1:]):
        gap = cur["start"] - prev["end"]
        if gap >= LONG_PAUSE_THRESHOLD_S:
            parts.append(LONG_PAUSE_TOKEN)
        elif gap >= PAUSE_THRESHOLD_S:
            parts.append(SHORT_PAUSE_TOKEN)
        parts.append(cur["word"])
    return normalise_transcript(" ".join(parts))


def _timing_features(words: list[dict], duration: float) -> dict:
    """Pause and rate statistics from word-level timestamps."""
    out = {
        "duration_seconds": round(duration, 2),
        "word_count": len(words),
        "speech_rate_wpm": 0.0,
        "pause_count": 0,
        "long_pause_count": 0,
        "total_pause_seconds": 0.0,
        "mean_pause_seconds": 0.0,
        "pause_ratio": 0.0,
        "articulation_rate_wpm": 0.0,
    }
    if not words or duration <= 0:
        return out

    out["speech_rate_wpm"] = round(60.0 * len(words) / duration, 1)

    gaps = []
    for prev, nxt in zip(words, words[1:]):
        gap = nxt["start"] - prev["end"]
        if gap >= PAUSE_THRESHOLD_S:
            gaps.append(gap)

    if gaps:
        out["pause_count"] = len(gaps)
        out["long_pause_count"] = sum(1 for g in gaps if g >= LONG_PAUSE_THRESHOLD_S)
        out["total_pause_seconds"] = round(sum(gaps), 2)
        out["mean_pause_seconds"] = round(sum(gaps) / len(gaps), 2)
        out["pause_ratio"] = round(min(1.0, sum(gaps) / duration), 3)

    speaking_time = max(0.1, duration - out["total_pause_seconds"])
    out["articulation_rate_wpm"] = round(60.0 * len(words) / speaking_time, 1)
    return out


def transcribe(audio_path: str) -> dict:
    """Transcribe an audio file and derive speech-timing features.

    faster-whisper decodes via PyAV, so browser webm/opus is handled directly
    with no external ffmpeg step.
    """
    with _lock:  # the model object is not thread-safe
        try:
            segments, info = _whisper().transcribe(
                audio_path,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                condition_on_previous_text=False,
            )
            segments = list(segments)
        except Exception as exc:
            raise TranscriptionError(f"Could not transcribe audio: {exc}") from exc

    words: list[dict] = []
    parts: list[str] = []
    for seg in segments:
        parts.append(seg.text)
        for w in (seg.words or []):
            words.append({"word": w.word.strip(), "start": w.start, "end": w.end})

    raw = " ".join(p.strip() for p in parts).strip()
    duration = float(getattr(info, "duration", 0.0) or 0.0)
    if words:
        duration = max(duration, words[-1]["end"])

    normalised = normalise_transcript(raw)
    return {
        "raw_transcript": raw,
        "transcript": normalised,
        # Falls back to the plain transcript when word timestamps are missing, so
        # this column is never silently empty for a recording that has words.
        "transcript_pauses": pause_encoded_transcript(words) or normalised,
        "language": getattr(info, "language", "en"),
        "timing": _timing_features(words, duration),
    }


def warm_up() -> None:
    """Load Whisper at startup so the first recording is not slow."""
    try:
        _whisper()
    except Exception:  # pragma: no cover
        pass
