"""Is this string an English word, and is it a proper noun?

Verbal fluency used to accept anything matching ^[a-z]{2,}$, so "fnndty" scored
exactly like "flower". That is not a cosmetic problem: the whole point of the
task is that producing words is effortful, and a scorer that accepts letter
strings rewards the one strategy that costs nothing. It also breaks the norms —
published FAS norms are counts of admissible words as judged by an examiner who
rejects non-words, proper nouns and inflected repeats, so scoring leniently
against those norms flatters everybody.

Two independent questions, two independent sources.

WORDHOOD comes from the WordNet lemma tables that ship inside spaCy's
en_core_web_sm, which the backend already installs for the picture-description
features. Nothing new is vendored and nothing new is downloaded. The tables are
base forms plus WordNet's irregular-inflection exceptions and detachment rules,
so "foxes", "sang" and "fastest" all resolve even though only "fox", "sing" and
"fast" are listed. Only the ~1 MB lookups file is read, not the pipeline.

PROPER NOUNS come from a committed blocklist, because WordNet lists France and
Florida as nouns exactly like fox and flower and cannot tell them apart. See
ml/build_proper_nouns.py for how that file is derived and what it deliberately
leaves out.

DEGRADED MODE
If spaCy or its model is missing, every word is accepted and `available()`
returns False. Fail-open is the right direction — a screening tool that scores
someone zero because a dependency is absent is worse than one that scores them
leniently — but it must never be silent, so score_verbal_fluency records
`lexicon_available` in its detail and the results page can say so.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

PROPER_NOUNS_PATH = Path(__file__).resolve().parent / "artifacts" / "proper_nouns.txt"

# WordNet's four open word classes. A string is a word if it is any of them.
_POS = ("noun", "verb", "adj", "adv")


@lru_cache(maxsize=1)
def _tables() -> tuple[dict, dict, dict] | None:
    """Load the WordNet lemma tables, or None if spaCy is unavailable.

    Reads the lookups file directly rather than calling spacy.load(). The
    pipeline is ~50 MB of tok2vec/parser/NER weights that answer a question we
    are not asking; the tables alone are about 6 MB resident. ml/features.py
    loads the pipeline with the lemmatizer disabled, so there is nothing to
    share with anyway.
    """
    try:
        from spacy.lookups import Lookups
        from spacy.util import get_package_path

        package = get_package_path("en_core_web_sm")
        # The installed package nests a versioned data directory inside itself.
        inner = next(
            (p for p in package.iterdir() if p.is_dir() and (p / "lemmatizer").is_dir()),
            package,
        )
        lookups = Lookups().from_disk(inner / "lemmatizer" / "lookups")
        index = lookups.get_table("lemma_index")
        exceptions = lookups.get_table("lemma_exc")
        rules = lookups.get_table("lemma_rules")
    except Exception:  # noqa: BLE001 - absent model, spaCy version drift, bad install
        return None

    # spaCy's Table hashes its string keys, so a missing POS raises rather than
    # returning a default. Normalise to plain dicts once, here.
    def grab(table, pos):
        try:
            return table[pos]
        except KeyError:
            return {} if table is exceptions else ()

    return (
        {pos: set(grab(index, pos)) for pos in _POS},
        {pos: dict(grab(exceptions, pos)) for pos in _POS},
        {pos: list(grab(rules, pos)) for pos in _POS},
    )


@lru_cache(maxsize=1)
def _proper_nouns() -> frozenset[str]:
    try:
        text = PROPER_NOUNS_PATH.read_text(encoding="utf-8")
    except OSError:
        return frozenset()
    return frozenset(line.strip() for line in text.splitlines() if line.strip())


def available() -> bool:
    """True when wordhood can actually be checked."""
    return _tables() is not None


def _candidates(word: str, pos: str, exceptions: dict, rules: dict):
    """Base forms `word` could be an inflection of, within one word class."""
    yield word
    yield from exceptions[pos].get(word, ())
    for old, new in rules[pos]:
        if old and word.endswith(old) and len(word) > len(old):
            yield word[: len(word) - len(old)] + new


@lru_cache(maxsize=4096)
def lemma(word: str) -> str:
    """Collapse a word to the base form it inflects from, or itself.

    Used to catch inflected repeats: on the FAS task "fox" then "foxes" is one
    word given twice, not two words.

    Deliberately conservative in two ways. It only undoes WordNet's own
    detachment rules and irregular forms, so sun/sunny and fast/fasten are left
    alone. And a form that WordNet lists in its own right is returned unchanged
    even when a rule could reduce it — "found" stays "found" rather than
    collapsing into "find", because founding a company is not finding a coin.
    The cost is that "fishing", "running" and "fastest" are all listed too, so
    they count separately from fish, run and fast. Crediting a word twice is the
    safer error here; refusing one the participant really produced is not.
    """
    tables = _tables()
    if tables is None:
        return word
    index, exceptions, rules = tables
    for pos in _POS:
        if word in index[pos]:
            return word  # already a base form; nothing to undo
    for pos in _POS:
        for candidate in _candidates(word, pos, exceptions, rules):
            if candidate in index[pos]:
                return candidate
    return word


def is_english_word(word: str) -> bool:
    """True if `word` is an English word in any word class or inflected form.

    Returns True for everything when the tables are unavailable — see the
    module docstring on degraded mode.
    """
    tables = _tables()
    if tables is None:
        return True
    index, exceptions, rules = tables
    return any(
        candidate in index[pos]
        for pos in _POS
        for candidate in _candidates(word, pos, exceptions, rules)
    )


def is_proper_noun(word: str) -> bool:
    """True if `word` is known only as a name of a person or place.

    Best effort, and knowingly incomplete — a blocklist cannot enumerate every
    name. It errs toward accepting: a word is blocked only when no ordinary
    lowercase sense is attested for it, so "frank", "august", "turkey" and
    "china" all stay scoreable.
    """
    return word in _proper_nouns()
