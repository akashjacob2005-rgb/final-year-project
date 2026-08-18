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
def model_metrics() -> dict:
    if paths.METRICS_PATH.exists():
        return json.loads(paths.METRICS_PATH.read_text())
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


def _build_matrix(art: dict, text: str, feats: dict[str, float]) -> np.ndarray:
    parts = []
    if art["mode"] in ("hand", "combined"):
        H = np.array([[feats.get(n, 0.0) for n in art["feature_names"]]], dtype=float)
        H = np.nan_to_num(H, posinf=0.0, neginf=0.0)
        parts.append(art["scaler"].transform(H))
    if art["mode"] in ("tfidf", "combined"):
        parts.append(art["vectorizer"].transform([text]).toarray())
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
        "model_version": art.get("version", "unknown"),
    }


def _explain(art: dict, X: np.ndarray, top_k: int) -> list[dict]:
    """Per-feature signed contributions for the handcrafted block.

    contribution = coefficient x standardised feature value, so it reads as
    'this marker pushed the score up/down by this much'. We only explain the
    handcrafted features — TF-IDF terms are individually meaningless to a user.
    """
    coef = _mean_coefficients(art["model"])
    if coef is None or art["mode"] not in ("hand", "combined"):
        return []

    names = art["feature_names"]
    n = min(len(names), len(coef), X.shape[1])
    contributions = coef[:n] * X[0, :n]

    # Some features are exact linear transforms of each other (content_units and
    # content_unit_ratio differ only by a constant divisor) and so carry the
    # same explanation. Showing both would spend two of the top slots saying one
    # thing, so keep only the first occurrence of each human-readable label.
    out: list[dict] = []
    seen_labels: set[str] = set()
    for i in np.argsort(-np.abs(contributions)):
        if len(out) >= top_k:
            break
        if abs(contributions[i]) <= 1e-9:
            continue
        label = FEATURE_LABELS.get(names[i], names[i].replace("_", " "))
        if label in seen_labels:
            continue
        seen_labels.add(label)
        out.append(
            {
                "feature": names[i],
                "label": label,
                "contribution": round(float(contributions[i]), 4),
                "direction": "increases risk" if contributions[i] > 0 else "lowers risk",
            }
        )
    return out


def warm_up() -> None:
    """Load the model and spaCy at startup so the first user request is not slow."""
    try:
        _artifact()
        extract_features("the mother is washing the dishes while the water runs over")
    except Exception:  # pragma: no cover - warm-up must never break startup
        pass
