"""Inference wrapper for the trained cognitive-decline language classifier.

Mirrors ml/train_language_model.py exactly: same feature extraction (imported,
not copied), same matrix assembly, same column order.
"""

from __future__ import annotations

import json
import threading
from functools import lru_cache

import numpy as np

from . import paths  # noqa: F401  (side effect: puts ml/ on sys.path)
from features import FEATURE_NAMES, extract_features  # noqa: E402

_lock = threading.Lock()

# Human-readable descriptions used in the results explanation. Keyed by feature.
FEATURE_LABELS: dict[str, str] = {
    "pronoun_noun_ratio": "use of pronouns in place of specific nouns",
    "mattr_50": "vocabulary variety",
    "type_token_ratio": "vocabulary variety",
    "hapax_ratio": "proportion of words used only once",
    "honore_statistic": "lexical richness",
    "brunet_index": "lexical density",
    "noun_ratio": "proportion of nouns",
    "verb_ratio": "proportion of verbs",
    "adj_ratio": "use of descriptive detail",
    "content_function_ratio": "balance of meaning-carrying words",
    "idea_density": "density of ideas per word",
    # content_units and content_unit_ratio differ only by a constant divisor, so
    # they share a label and the explanation de-duplicates them into one row.
    "content_units": "how much of the scene was described",
    "content_unit_ratio": "how much of the scene was described",
    "content_units_per_100w": "detail mentioned per 100 words",
    "mean_parse_depth": "grammatical complexity",
    "subordinate_ratio": "use of subordinate clauses",
    "mean_sentence_length": "sentence length",
    "prep_phrase_ratio": "use of prepositional phrases",
    "filler_ratio": "filled pauses and hesitations",
    "repetition_ratio": "immediate word repetition",
    "unique_bigram_ratio": "variety of word pairings",
    "total_words": "overall amount said",
    "total_sentences": "number of utterances",
    "mean_word_length": "average word length",
    "adv_ratio": "proportion of adverbs",
    "pron_ratio": "proportion of pronouns",
    "det_ratio": "proportion of determiners",
}

# Speech-timing markers. Separate from FEATURE_LABELS because these only exist on
# the audio path, and because they are phrased as behaviour ("how they spoke")
# rather than as properties of the text.
TIMING_LABELS: dict[str, str] = {
    "duration_seconds": "how long they spoke",
    "word_count": "number of words spoken",
    "speech_rate_wpm": "overall speaking rate",
    "pause_count": "how often speech was interrupted by a pause",
    "long_pause_count": "number of long pauses",
    "total_pause_seconds": "total time spent pausing",
    "mean_pause_seconds": "average pause length",
    "pause_ratio": "proportion of the recording spent in silence",
    "articulation_rate_wpm": "speaking rate while actually talking",
}

MIN_WORDS = 25  # below this, a transcript cannot be scored meaningfully


class ModelUnavailable(RuntimeError):
    pass


@lru_cache(maxsize=1)
def _artifact() -> dict:
    if not paths.LANGUAGE_MODEL_PATH.exists():
        raise ModelUnavailable(
            f"Model artifact not found at {paths.LANGUAGE_MODEL_PATH}. "
            "Run: python ml/download_data.py && python ml/train_language_model.py"
        )
    import joblib

    return joblib.load(paths.LANGUAGE_MODEL_PATH)


@lru_cache(maxsize=1)
def _audio_artifact() -> dict:
    if not paths.AUDIO_MODEL_PATH.exists():
        raise ModelUnavailable(
            f"Audio model artifact not found at {paths.AUDIO_MODEL_PATH}. "
            "Run: python ml/download_pitt.py && python ml/build_dataset.py "
            "&& python ml/train_audio_model.py"
        )
    import joblib

    return joblib.load(paths.AUDIO_MODEL_PATH)


def audio_available() -> bool:
    """Whether the audio-trained model can be used for this deployment.

    Checked per request so a deployment without the artifact degrades to the
    text-only model instead of failing the assessment. Deliberately cheap: the
    artifact itself is cached by _audio_artifact.
    """
    try:
        _audio_artifact()
        return True
    except Exception:
        return False


@lru_cache(maxsize=1)
def model_metrics() -> dict:
    if paths.METRICS_PATH.exists():
        return json.loads(paths.METRICS_PATH.read_text())
    return {}


@lru_cache(maxsize=1)
def audio_model_metrics() -> dict:
    if paths.AUDIO_METRICS_PATH.exists():
        return json.loads(paths.AUDIO_METRICS_PATH.read_text())
    return {}


def _mean_coefficients(calibrated) -> np.ndarray | None:
    """Average the underlying linear coefficients across calibration folds.

    Returns None for non-linear estimators, in which case we simply omit the
    per-feature explanation rather than inventing one.
    """
    coefs = []
    for cc in getattr(calibrated, "calibrated_classifiers_", []):
        est = getattr(cc, "estimator", None) or getattr(cc, "base_estimator", None)
        if est is not None and hasattr(est, "coef_"):
            coefs.append(np.asarray(est.coef_).ravel())
    if not coefs:
        return None
    return np.mean(coefs, axis=0)


def _blocks(mode: str) -> tuple[bool, bool, bool]:
    """Which feature blocks a mode string implies -> (handcrafted, tfidf, timing).

    The two trainers name their modes differently and both artifacts are served
    here, so the vocabularies have to be reconciled in one place:

        ml/train_language_model.py  ->  hand | tfidf | combined
        ml/train_audio_model.py     ->  text | timing | text+timing

    'text' means handcrafted AND tfidf together (train_audio_model.build_matrix
    appends both under `if "text" in mode`). Getting this wrong does not raise —
    it silently assembles a matrix with the wrong number of columns.
    """
    has_hand = mode in ("hand", "combined") or "text" in mode
    has_tfidf = mode in ("tfidf", "combined") or "text" in mode
    has_timing = "timing" in mode
    return has_hand, has_tfidf, has_timing


def _build_matrix(
    art: dict,
    text: str,
    feats: dict[str, float],
    timing: dict[str, float] | None = None,
) -> np.ndarray:
    """Assemble the design matrix in the trainers' block order: hand, tfidf, timing."""
    has_hand, has_tfidf, has_timing = _blocks(art["mode"])
    parts = []
    if has_hand:
        H = np.array([[feats.get(n, 0.0) for n in art["feature_names"]]], dtype=float)
        H = np.nan_to_num(H, posinf=0.0, neginf=0.0)
        parts.append(art["scaler"].transform(H))
    if has_tfidf:
        parts.append(art["vectorizer"].transform([text]).toarray())
    if has_timing:
        names = art["timing_feature_names"]
        T = np.array([[float((timing or {}).get(n, 0.0)) for n in names]], dtype=float)
        T = np.nan_to_num(T, posinf=0.0, neginf=0.0)
        parts.append(art["timing_scaler"].transform(T))
    return np.hstack(parts)


def predict(text: str, top_k: int = 5) -> dict:
    """Score a picture-description transcript.

    Returns the calibrated probability of the 'decline-consistent' class, the
    extracted features, and the linguistic markers that drove the result.
    """
    text = (text or "").strip()
    feats = extract_features(text)
    word_count = int(feats.get("total_words", 0))

    if word_count < MIN_WORDS:
        return {
            "available": False,
            "reason": (
                f"Only {word_count} words were captured. At least {MIN_WORDS} are "
                "needed for a reliable language analysis."
            ),
            "word_count": word_count,
            "probability": None,
            "features": feats,
            "contributions": [],
            "model": "text",
        }

    with _lock:  # sklearn predict is not guaranteed reentrant across threads
        art = _artifact()
        X = _build_matrix(art, text, feats)
        prob = float(art["model"].predict_proba(X)[0, 1])

    return {
        "available": True,
        "probability": prob,
        "word_count": word_count,
        "features": feats,
        "contributions": _explain(art, X, top_k),
        "model": "text",
        "model_version": art.get("version", "unknown"),
    }


def predict_audio(
    transcript: str,
    transcript_pauses: str | None,
    timing: dict[str, float] | None,
    top_k: int = 5,
) -> dict:
    """Score a picture-description RECORDING with the audio-trained model.

    Differs from predict() in three ways, all of them dictated by how
    ml/train_audio_model.py fitted the artifact:

    1. The timing block is a real predictor, not a description. It is required —
       callers without audio must use predict() instead, because a zeroed timing
       vector standardises to 'spoke 0 words in 0 seconds' and would distort the
       score rather than merely weaken it.
    2. The vectorizer may have been fitted on the pause-marked transcript. The
       artifact records which under 'text_column'; feeding the wrong one puts the
       vocabulary out of alignment with training.
    3. The handcrafted block always comes from the CLEAN transcript, whatever the
       vectorizer saw — 'shortpause'/'longpause' are not words the speaker said,
       and letting spaCy parse them inflates total_words and skews every ratio
       built on top of it.
    """
    text = (transcript or "").strip()
    feats = extract_features(text)
    word_count = int(feats.get("total_words", 0))

    if word_count < MIN_WORDS:
        return {
            "available": False,
            "reason": (
                f"Only {word_count} words were captured. At least {MIN_WORDS} are "
                "needed for a reliable language analysis."
            ),
            "word_count": word_count,
            "probability": None,
            "features": feats,
            "contributions": [],
            "model": "audio",
        }

    with _lock:  # sklearn predict is not guaranteed reentrant across threads
        art = _audio_artifact()
        vec_text = text
        if art.get("text_column") == "transcript_pauses":
            vec_text = (transcript_pauses or "").strip() or text
        X = _build_matrix(art, vec_text, feats, timing=timing)
        prob = float(art["model"].predict_proba(X)[0, 1])

    return {
        "available": True,
        "probability": prob,
        "word_count": word_count,
        "features": feats,
        "timing": timing or {},
        "contributions": _explain(art, X, top_k),
        "model": "audio",
        "model_version": art.get("version", "unknown"),
    }


def _explainable_columns(art: dict) -> list[tuple[str, str, int]]:
    """-> (feature name, human label, column index) for blocks a user can read.

    TF-IDF columns are skipped rather than labelled: an individual n-gram weight
    means nothing on a results page. Its WIDTH still has to be counted, because
    the timing block sits after it — for the deployed audio model that puts
    timing at column 27 + len(vocabulary_), not at column 27.
    """
    has_hand, has_tfidf, has_timing = _blocks(art["mode"])
    cols: list[tuple[str, str, int]] = []
    offset = 0

    if has_hand:
        names = art["feature_names"]
        cols += [
            (n, FEATURE_LABELS.get(n, n.replace("_", " ")), offset + i)
            for i, n in enumerate(names)
        ]
        offset += len(names)

    if has_tfidf:
        offset += len(getattr(art["vectorizer"], "vocabulary_", None) or ())

    if has_timing:
        names = art.get("timing_feature_names") or []
        cols += [
            (n, TIMING_LABELS.get(n, n.replace("_", " ")), offset + i)
            for i, n in enumerate(names)
        ]

    return cols


def _explain(art: dict, X: np.ndarray, top_k: int) -> list[dict]:
    """Per-feature signed contributions for the interpretable blocks.

    contribution = coefficient x standardised feature value, so it reads as
    'this marker pushed the score up/down by this much'. Covers the handcrafted
    linguistic block and, for the audio model, the speech-timing block — those
    are learned predictors now, so a long-pause weight is as reportable as a
    pronoun-ratio one.
    """
    coef = _mean_coefficients(art["model"])
    if coef is None:
        return []

    scored = [
        (name, label, float(coef[i] * X[0, i]))
        for name, label, i in _explainable_columns(art)
        if i < len(coef) and i < X.shape[1]
    ]

    # Some features are exact linear transforms of each other (content_units and
    # content_unit_ratio differ only by a constant divisor) and so carry the
    # same explanation. Showing both would spend two of the top slots saying one
    # thing, so keep only the first occurrence of each human-readable label.
    out: list[dict] = []
    seen_labels: set[str] = set()
    for name, label, contribution in sorted(scored, key=lambda t: -abs(t[2])):
        if len(out) >= top_k:
            break
        if abs(contribution) <= 1e-9 or label in seen_labels:
            continue
        seen_labels.add(label)
        out.append(
            {
                "feature": name,
                "label": label,
                "contribution": round(contribution, 4),
                "direction": "increases risk" if contribution > 0 else "lowers risk",
            }
        )
    return out


def warm_up() -> None:
    """Load the models and spaCy at startup so the first user request is not slow."""
    try:
        _artifact()
        extract_features("the mother is washing the dishes while the water runs over")
    except Exception:  # pragma: no cover - warm-up must never break startup
        pass
    # Separate try: a missing audio artifact is a supported deployment state, and
    # it must not stop the text model from being warmed.
    try:
        _audio_artifact()
    except Exception:  # pragma: no cover
        pass
