"""Linguistic feature extraction for cognitive-decline detection.

SINGLE SOURCE OF TRUTH. Both training (ml/train_language_model.py) and inference
(backend/app/ml/) import from here. Never duplicate this logic — if training and
inference compute features differently, the deployed model is silently wrong.

Feature families are drawn from the aphasia/dementia discourse literature:
lexical richness, part-of-speech distribution, syntactic complexity, disfluency,
and information content (semantic content units for the Cookie Theft scene).
"""

from __future__ import annotations

import math
import re
from collections import Counter
from functools import lru_cache

# Filled-pause / hesitation tokens common in spontaneous speech transcripts.
FILLERS = {"uh", "um", "er", "ah", "eh", "hmm", "mm", "mhm", "oh", "well", "like"}

# Canonical Cookie Theft semantic content units. Word-finding difficulty shows up
# as *fewer* of these being named, even when the person talks at length.
CONTENT_UNITS: dict[str, tuple[str, ...]] = {
    "boy": ("boy", "son", "brother", "lad"),
    "girl": ("girl", "daughter", "sister"),
    "woman": ("woman", "mother", "mom", "mum", "lady"),
    "cookie": ("cookie", "cookies", "biscuit"),
    "jar": ("jar", "cookiejar"),
    "stool": ("stool", "chair", "seat"),
    "falling": ("fall", "falling", "falls", "tipping", "tip", "toppling", "wobbl"),
    "sink": ("sink", "basin"),
    "water": ("water", "overflow", "overflowing", "spilling", "running"),
    "dishes": ("dish", "dishes", "plate", "plates", "cup", "saucer"),
    "drying": ("dry", "drying", "wiping", "wipe", "washing", "wash"),
    "window": ("window",),
    "curtain": ("curtain", "curtains", "drape"),
    "kitchen": ("kitchen",),
    "cupboard": ("cupboard", "cabinet", "counter"),
    "outside": ("outside", "garden", "grass", "yard", "tree", "shrub"),
}

_WORD_RE = re.compile(r"[a-zA-Z']+")

FEATURE_NAMES: list[str] = [
    # length
    "total_words", "total_sentences", "mean_sentence_length",
    # lexical richness
    "type_token_ratio", "mattr_50", "hapax_ratio", "brunet_index", "honore_statistic",
    "mean_word_length",
    # part of speech
    "noun_ratio", "verb_ratio", "adj_ratio", "adv_ratio", "pron_ratio", "det_ratio",
    "pronoun_noun_ratio", "content_function_ratio",
    # syntax
    "mean_parse_depth", "subordinate_ratio", "prep_phrase_ratio",
    # disfluency
    "filler_ratio", "repetition_ratio", "unique_bigram_ratio",
    # information content
    "content_units", "content_unit_ratio", "idea_density", "content_units_per_100w",
]


@lru_cache(maxsize=1)
def _nlp():
    """Load spaCy once. Parser is needed for dependency depth."""
    import spacy

    try:
        return spacy.load("en_core_web_sm", disable=["ner", "lemmatizer"])
    except OSError as exc:  # pragma: no cover - environment problem, not logic
        raise RuntimeError(
            "spaCy model 'en_core_web_sm' not installed. "
            "Run: python -m spacy download en_core_web_sm"
        ) from exc


def _tokens(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


def _mattr(words: list[str], window: int = 50) -> float:
    """Moving-average type-token ratio.

    Plain TTR falls as text gets longer, so it partly measures verbosity rather
    than vocabulary. MATTR is length-robust, which matters here because dementia
    transcripts differ systematically in length from control ones.
    """
    if len(words) < window:
        return len(set(words)) / len(words) if words else 0.0
    ratios = [
        len(set(words[i : i + window])) / window
        for i in range(len(words) - window + 1)
    ]
    return sum(ratios) / len(ratios)


def _parse_depth(token) -> int:
    """Depth of a token below its sentence root.

    Compare by index, not identity: spaCy builds a fresh Token proxy on every
    attribute access, so `token.head is token` is False even at the root and an
    identity check loops until the guard trips.
    """
    depth = 0
    while token.head.i != token.i:
        token = token.head
        depth += 1
        if depth > 50:  # malformed-parse guard
            break
    return depth


def _zeros() -> dict[str, float]:
    return dict.fromkeys(FEATURE_NAMES, 0.0)


def extract_features(text: str) -> dict[str, float]:
    """Extract the full linguistic feature vector from a transcript.

    Returns all-zero features for empty or near-empty input rather than raising,
    so a user who says nothing produces a valid (and predictably poor) result
    instead of a 500.
    """
    text = (text or "").strip()
    words = _tokens(text)
    if len(words) < 3:
        return _zeros()

    f = _zeros()
    n = len(words)
    counts = Counter(words)
    types = len(counts)

    doc = _nlp()(text)
    sents = [s for s in doc.sents if len(s) > 0]
    n_sents = max(1, len(sents))

    # --- length -----------------------------------------------------------
    f["total_words"] = float(n)
    f["total_sentences"] = float(n_sents)
    f["mean_sentence_length"] = n / n_sents

    # --- lexical richness -------------------------------------------------
    f["type_token_ratio"] = types / n
    f["mattr_50"] = _mattr(words)
    hapax = sum(1 for c in counts.values() if c == 1)
    f["hapax_ratio"] = hapax / n
    f["mean_word_length"] = sum(len(w) for w in words) / n
    # Brunet's index: lower = richer vocabulary. Honore: higher = richer.
    f["brunet_index"] = n ** (types ** -0.165)
    if hapax < types:
        f["honore_statistic"] = 100 * math.log(n) / (1 - hapax / types)
    else:
        f["honore_statistic"] = 0.0

    # --- part of speech ---------------------------------------------------
    pos = Counter(t.pos_ for t in doc if not t.is_punct and not t.is_space)
    total_pos = max(1, sum(pos.values()))
    nouns = pos["NOUN"] + pos["PROPN"]
    verbs = pos["VERB"] + pos["AUX"]
    f["noun_ratio"] = nouns / total_pos
    f["verb_ratio"] = verbs / total_pos
    f["adj_ratio"] = pos["ADJ"] / total_pos
    f["adv_ratio"] = pos["ADV"] / total_pos
    f["pron_ratio"] = pos["PRON"] / total_pos
    f["det_ratio"] = pos["DET"] / total_pos
    # Pronoun-over-noun substitution ("he", "it", "that thing") is one of the
    # most reported markers of word-finding difficulty.
    f["pronoun_noun_ratio"] = pos["PRON"] / nouns if nouns else float(pos["PRON"])
    content = nouns + verbs + pos["ADJ"] + pos["ADV"]
    function = total_pos - content
    f["content_function_ratio"] = content / function if function else float(content)

    # --- syntax -----------------------------------------------------------
    depths = [_parse_depth(t) for t in doc if not t.is_punct and not t.is_space]
    f["mean_parse_depth"] = sum(depths) / len(depths) if depths else 0.0
    deps = Counter(t.dep_ for t in doc)
    f["subordinate_ratio"] = (
        deps["advcl"] + deps["ccomp"] + deps["xcomp"] + deps["relcl"]
    ) / n_sents
    f["prep_phrase_ratio"] = deps["prep"] / total_pos

    # --- disfluency -------------------------------------------------------
    f["filler_ratio"] = sum(counts[w] for w in FILLERS) / n
    immediate_repeats = sum(1 for a, b in zip(words, words[1:]) if a == b)
    f["repetition_ratio"] = immediate_repeats / n
    bigrams = list(zip(words, words[1:]))
    f["unique_bigram_ratio"] = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    # --- information content ---------------------------------------------
    blob = " ".join(words)
    hits = sum(
        1 for variants in CONTENT_UNITS.values()
        if any(v in blob for v in variants)
    )
    f["content_units"] = float(hits)
    f["content_unit_ratio"] = hits / len(CONTENT_UNITS)
    f["content_units_per_100w"] = 100 * hits / n
    # Idea density: propositions (verbs, adjectives, adverbs, prepositions,
    # conjunctions) per word. Low density with high word count = empty speech.
    props = verbs + pos["ADJ"] + pos["ADV"] + pos["ADP"] + pos["CCONJ"] + pos["SCONJ"]
    f["idea_density"] = props / n

    return f


def features_to_vector(feats: dict[str, float]) -> list[float]:
    """Order a feature dict into the canonical FEATURE_NAMES order."""
    return [float(feats.get(name, 0.0)) for name in FEATURE_NAMES]


def extract_batch(texts: list[str]):
    """Extract features for many texts. Returns a pandas DataFrame."""
    import pandas as pd

    return pd.DataFrame(
        [extract_features(t) for t in texts], columns=FEATURE_NAMES
    )
