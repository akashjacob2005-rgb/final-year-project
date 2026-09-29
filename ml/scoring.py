"""Objective scoring for the five structured cognitive tests, plus normative
age/education adjustment.

Scoring rules follow the Montreal Cognitive Assessment (MoCA) where the test is
a MoCA item, so the numbers on screen mean the same thing a clinician's would.

IMPORTANT — read before quoting any of this as clinical:
The normative means/SDs in NORMS are approximations drawn from published MoCA
and neuropsychological literature ranges. They are here so the app can express a
result as a z-score rather than a bare count. They are NOT institution-validated
norms and must be replaced with locally validated values before any clinical use.
This limitation is stated in the results UI and in the report.

That caveat is now STRONGER, not weaker, and the reason matters. Task difficulty
is tiered by age (see backend/app/ml/stimuli.py), so three of the five tests below
are normed against a harder variant for younger users: more memory targets, a
longer trail, faster digit presentation. No published norms exist for those
variants — the literature norms the standard forms. The means/SDs for the
(18,39) and (40,59) bands are therefore estimates extrapolated from the
standard-form values, not measurements.

verbal_fluency and serial_sevens are the exceptions: both tasks are now identical
in every tier, so verbal_fluency's young bands are a straight split of a published
range and serial_sevens is the MoCA item everywhere. They carry only the original
caveat. Treat the other three young-band figures as the least reliable numbers in
this file, and re-derive all of them from local pilot data before any real use.

RE-REFERENCED AFTER THE DIFFICULTY REDUCTION
Every tier in stimuli.py was made easier (fewer memory targets with more study
time, shorter delays, shorter trails, a ladder capped at 5, slower digits,
serial sevens from 100 for everyone). Each band below was re-estimated in the
same change. Where a tier's variant did not change, its numbers did not either —
those are called out per band so it is obvious which figures moved and why.

The derivation is not measurement. Memory means were held at roughly 85-90% of
the target count, the hit rate the previous bands implied. Trail means were
rescaled by node count at each band's previous per-node rate. Digit-span and
serial-sevens means were nudged up to reflect the easier presentation. These are
the same kind of estimates the rest of the file already contains, and they carry
the same caveat.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

# --------------------------------------------------------------------------
# Test definitions
# --------------------------------------------------------------------------
TESTS = (
    "memory_recognition",
    "picture_description",
    "trail_making_b",
    "serial_sevens",
    "digit_span_backward",
    "verbal_fluency",
)

DOMAIN_OF_TEST = {
    "memory_recognition": "memory",
    "picture_description": "language",
    "trail_making_b": "executive",
    "serial_sevens": "attention",
    "digit_span_backward": "working_memory",
    "verbal_fluency": "language",
}


@dataclass
class TestScore:
    test: str
    raw: float                      # primary raw measure
    score: float                    # normalised points earned
    max_score: float                # points available
    z: float = 0.0                  # age/education adjusted z-score
    detail: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "test": self.test,
            "domain": DOMAIN_OF_TEST[self.test],
            "raw": self.raw,
            "score": self.score,
            "max_score": self.max_score,
            "z": round(self.z, 3),
            "detail": self.detail,
        }


# --------------------------------------------------------------------------
# 1. Memory recognition (5 objects, 10s delay, recognition grid)
# --------------------------------------------------------------------------
def score_memory_recognition(selected: list[str], targets: list[str]) -> TestScore:
    """Hits minus false alarms, floored at zero.

    Subtracting false alarms matters: without it, a user who selects every item
    on the grid scores full marks. Indiscriminate selection is itself a decline
    signal, not a perfect score.
    """
    sel = {s.strip().lower() for s in selected}
    tgt = {t.strip().lower() for t in targets}
    hits = len(sel & tgt)
    false_alarms = len(sel - tgt)
    misses = len(tgt - sel)
    score = max(0.0, hits - false_alarms)

    return TestScore(
        test="memory_recognition",
        raw=float(hits),
        score=float(score),
        max_score=float(len(tgt)),
        detail={
            "hits": hits,
            "misses": misses,
            "false_alarms": false_alarms,
            "targets": sorted(tgt),
            "selected": sorted(sel),
        },
    )


# --------------------------------------------------------------------------
# 2. Trail Making B (MoCA alternating 1-A-2-B-3-C-4-D-5-E)
# --------------------------------------------------------------------------
def score_trail_making_b(
    clicked: list[str], expected: list[str], duration_ms: float
) -> TestScore:
    """MoCA awards 1 point only for a fully correct, unbroken sequence.

    We also keep completion time, which carries more graded information than the
    single MoCA point and is what the z-score is computed from.
    """
    errors = 0
    cursor = 0
    for c in clicked:
        if cursor < len(expected) and c == expected[cursor]:
            cursor += 1
        else:
            errors += 1

    completed = cursor == len(expected)
    score = 1.0 if (completed and errors == 0) else 0.0
    seconds = duration_ms / 1000.0

    return TestScore(
        test="trail_making_b",
        raw=seconds,
        score=score,
        max_score=1.0,
        detail={
            "errors": errors,
            "completed": completed,
            "nodes_reached": cursor,
            "nodes_total": len(expected),
            "duration_seconds": round(seconds, 2),
        },
    )


# --------------------------------------------------------------------------
# 3. Serial 7s (MoCA: subtract 7 from 100, five times)
# --------------------------------------------------------------------------
def score_serial_sevens(responses: list[int | None], start: int = 100) -> TestScore:
    """MoCA scoring, including the chained-error rule.

    The rule that trips people up: each subtraction is judged against the number
    the participant *actually said* previously, not the ideal sequence. One early
    slip followed by five correct subtractions is 4 correct answers, not 0.
    """
    correct = 0
    prev = start
    marks = []
    for r in responses[:5]:
        if r is not None and r == prev - 7:
            correct += 1
            marks.append(True)
        else:
            marks.append(False)
        # Chain from what they said, so a single slip doesn't cascade.
        prev = r if r is not None else prev - 7

    if correct == 0:
        points = 0.0
    elif correct == 1:
        points = 1.0
    elif correct in (2, 3):
        points = 2.0
    else:
        points = 3.0

    return TestScore(
        test="serial_sevens",
        raw=float(correct),
        score=points,
        max_score=3.0,
        detail={
            "correct_subtractions": correct,
            "responses": responses[:5],
            "per_item_correct": marks,
            "expected_ideal": [start - 7 * i for i in range(1, 6)],
        },
    )


# --------------------------------------------------------------------------
# 4. Digit span backward (adaptive)
# --------------------------------------------------------------------------
def score_digit_span_backward(trials: list[dict]) -> TestScore:
    """Adaptive span: longest sequence correctly reproduced in reverse.

    Each trial: {"sequence": [2,7,4], "response": [4,7,2]}
    """
    longest = 0
    n_correct = 0
    for t in trials:
        seq = [int(d) for d in t.get("sequence", [])]
        resp = [int(d) for d in t.get("response", []) if str(d).strip() != ""]
        if seq and resp == seq[::-1]:
            n_correct += 1
            longest = max(longest, len(seq))

    # MoCA awards 1 point for a 3-digit backward span.
    moca_point = 1.0 if longest >= 3 else 0.0

    return TestScore(
        test="digit_span_backward",
        raw=float(longest),
        score=moca_point,
        max_score=1.0,
        detail={
            "longest_span": longest,
            "trials_correct": n_correct,
            "trials_total": len(trials),
        },
    )


# --------------------------------------------------------------------------
# 5. Verbal fluency (letter F, 60 seconds)
# --------------------------------------------------------------------------
_VALID_WORD = re.compile(r"^[a-z]{2,}$")


def score_verbal_fluency(
    words: list[str], letter: str = "f", duration_s: float = 60.0
) -> TestScore:
    """MoCA: 1 point for 11 or more admissible words in 60 seconds.

    Admissibility filtering is the whole test. An examiner administering FAS
    rejects non-words, proper nouns and inflected repeats as they are spoken,
    and the published norms are counts of what survived that. Scoring anything
    that starts with the right letter — which is what this used to do, so
    "fnndty" counted as a word — does not just add noise: it inflates the raw
    count against norms that assume the rejections happened, and it rewards the
    one strategy that takes no cognitive effort at all.

    The five rejection buckets, applied in order, are reported in `detail` so a
    clinician can audit every decision rather than trust the count:

      invalid      not a plain 2+ letter alphabetic string
      wrong_letter does not begin with the target letter
      not_a_word   no English word, in any word class or inflected form
      proper_noun  a name of a person or place
      repeat       already given, including as a regular inflection of one
                   already given (fox then foxes); see lexicon.lemma for the
                   deliberate limits on that

    Wordhood is checked by ml/lexicon.py. When that is unavailable the check is
    skipped rather than failed, and `lexicon_available` in the detail records it
    so a degraded score is never mistaken for a clean one.
    """
    from lexicon import available, is_english_word, is_proper_noun, lemma

    letter = letter.lower()
    checked = available()
    seen: set[str] = set()      # base forms already credited
    valid: list[str] = []
    rejected: dict[str, list[str]] = {
        "invalid": [],
        "wrong_letter": [],
        "not_a_word": [],
        "proper_noun": [],
        "repeat": [],
    }

    for w in words:
        w = w.strip().lower()
        if not w:
            continue
        if not _VALID_WORD.match(w):
            rejected["invalid"].append(w)
            continue
        if not w.startswith(letter):
            rejected["wrong_letter"].append(w)
            continue
        if not is_english_word(w):
            rejected["not_a_word"].append(w)
            continue
        if is_proper_noun(w):
            rejected["proper_noun"].append(w)
            continue
        # Collapse to a base form before the repeat check, so fox/foxes counts
        # once. With no lexicon, lemma() is the identity and this degrades to
        # the exact-duplicate check it replaces.
        root = lemma(w)
        if root in seen or w in seen:
            rejected["repeat"].append(w)
            continue
        seen.add(root)
        valid.append(w)

    count = len(valid)
    return TestScore(
        test="verbal_fluency",
        raw=float(count),
        score=1.0 if count >= 11 else 0.0,
        max_score=1.0,
        detail={
            "valid_count": count,
            "valid_words": valid,
            "rejected": rejected,
            "duration_seconds": duration_s,
            "moca_threshold": 11,
            # False means wordhood and proper-noun checks were skipped, so the
            # count is an upper bound. The results page surfaces this.
            "lexicon_available": checked,
        },
    )


# --------------------------------------------------------------------------
# Normative adjustment
# --------------------------------------------------------------------------
# mean / sd of the RAW measure, by age band. See module docstring caveat.
# "higher_is_better" flips the sign so that a negative z always means worse.
#
# THE BANDS HERE ARE THE DIFFICULTY TIERS. They must stay identical to
# TIER_BANDS in backend/app/ml/stimuli.py:
#
#   (18, 39)  -> "demanding"     (60, 74)  -> "moderate"
#   (40, 59)  -> "challenging"   (75, 120) -> "standard"
#
# Each band's mean/sd describes performance on the stimuli THAT BAND ACTUALLY
# RECEIVES, not on some common version of the test. A younger user gets more
# memory targets, a longer trail and faster digits; their norm is referenced to
# that harder variant. Changing a tier's parameters in stimuli.py without
# re-estimating its band here will silently mis-score everyone in that band —
# in whichever direction the change went. Making the test harder without
# re-estimating flags healthy people as declining; making it EASIER without
# re-estimating is worse, because it quietly clears people who should have been
# referred. test_typical_performer_scores_z_near_zero_in_every_tier is the guard.
NORMS: dict[str, dict] = {
    "memory_recognition": {  # hits, out of a tier-dependent number of targets
        # Means sit at roughly 85-90% of the target count — the hit rate the
        # previous bands implied, carried across to the shorter lists. SDs are a
        # little under one item: tighter than that, against a bounded score,
        # manufactures implausible z-values at the floor.
        #
        # A mean must stay BELOW the target count or a perfect score reads as
        # below average. test_a_ceiling_score_is_never_penalised enforces it.
        "higher_is_better": True,
        "bands": {
            (18, 39): (5.3, 1.0),    # 6 targets, 12s study, 15s delay
            (40, 59): (4.4, 0.9),    # 5 targets, 10s study, 12s delay
            (60, 74): (3.5, 0.8),    # 4 targets, 8s study, 10s delay
            (75, 120): (3.4, 0.8),   # 4 targets, 10s study, 8s delay
        },
    },
    "trail_making_b": {  # seconds to complete
        # Rescaled by node count at each band's own previous per-node rate
        # (2.4 s/node at 18-39 rising to 4.5 s/node at 75+), so the expected
        # pace per node is unchanged and only the trail got shorter.
        "higher_is_better": False,
        "bands": {
            (18, 39): (29.0, 10.0),  # 12 nodes
            (40, 59): (28.0, 10.0),  # 10 nodes
            (60, 74): (26.0, 11.0),  # 8 nodes
            (75, 120): (36.0, 16.0),  # 8 nodes
        },
    },
    "serial_sevens": {  # number of correct subtractions, always out of 5
        # No longer tiered — every band now starts from 100. The two older bands
        # were already normed against that task and are unchanged; the two
        # younger ones moved up off the 193 start they used to sit.
        "higher_is_better": True,
        "bands": {
            (18, 39): (4.5, 0.9),    # from 100 (was 4.3 against a 193 start)
            (40, 59): (4.4, 1.0),    # from 100 (was 4.1 against a 193 start)
            (60, 74): (4.1, 1.2),    # from 100 — unchanged
            (75, 120): (3.6, 1.5),   # from 100 — unchanged
        },
    },
    "digit_span_backward": {  # longest span reproduced correctly
        # Every band now runs the same 3-4-5 ladder and differs only in how long
        # each digit is shown, so the means differ by age alone. All four sit
        # below the ladder's ceiling of 5 — a norm above the ceiling would
        # penalise a PERFECT score, which is how the previous 4-rung band came
        # to score a flawless run at z = -0.50.
        "higher_is_better": True,
        "bands": {
            (18, 39): (4.5, 0.9),    # ladder 3-5, digits at 1200ms
            (40, 59): (4.3, 0.9),    # ladder 3-5, digits at 1300ms
            (60, 74): (4.1, 0.9),    # ladder 3-5, digits at 1400ms
            (75, 120): (3.9, 1.0),   # ladder 3-5, digits at 1500ms
        },
    },
    "verbal_fluency": {  # admissible words in 60s
        # Not tiered — every band sits the identical 60-second task, so this is a
        # straight split of the old 18-59 band and involves no re-referencing.
        #
        # Unchanged by the admissibility fix, and that is the point. These are
        # published FAS figures, which count only what an examiner accepted.
        # Scoring used to credit non-words and proper nouns, so raw counts came
        # in ABOVE what the norms describe and everyone read slightly high;
        # rejecting them properly moves the measure onto the norm, not off it.
        "higher_is_better": True,
        "bands": {
            (18, 39): (17.0, 4.9),
            (40, 59): (15.5, 4.7),
            (60, 74): (14.0, 4.5),   # unchanged
            (75, 120): (12.0, 4.3),  # unchanged
        },
    },
}

# Lower education depresses raw scores independently of cognition. MoCA applies a
# +1 total-score adjustment at <=12 years; we apply a small per-test z offset.
EDUCATION_Z_ADJUSTMENT = {"low": 0.35, "medium": 0.0, "high": -0.15}

# Largest magnitude a z-score may take. See normative_z for the rationale.
Z_CLAMP = 4.0


def _education_band(years: int | None) -> str:
    if years is None:
        return "medium"
    if years <= 12:
        return "low"
    if years >= 17:
        return "high"
    return "medium"


def normative_z(test: str, raw: float, age: int, education_years: int | None) -> float:
    """Convert a raw measure to an age/education-adjusted z-score.

    Negative z = below expectation for that age and education level.
    """
    spec = NORMS.get(test)
    if not spec:
        return 0.0

    bands = spec["bands"]
    match = next((ms for (lo, hi), ms in bands.items() if lo <= age <= hi), None)
    if match is None:
        # Age outside every band. Clamp to the nearest band rather than falling
        # through to the last one: an under-18 is given the hardest stimuli by
        # stimuli.tier_for_age, so scoring them against the 75+ norm would read
        # a difficult test as an easy one and flatter them badly.
        keys = sorted(bands)
        match = bands[keys[0]] if age < keys[0][0] else bands[keys[-1]]
    mean, sd = match
    if sd <= 0:
        return 0.0

    z = (raw - mean) / sd
    if not spec["higher_is_better"]:
        z = -z
    z += EDUCATION_Z_ADJUSTMENT[_education_band(education_years)]

    # Clamp. Past about four standard deviations the number stops carrying
    # information: it means "floor" or "ceiling", and the difference between
    # z=-4 and z=-6.4 is an artefact of a small SD meeting a bounded score, not
    # a clinical distinction. Left unclamped, one bottomed-out test dominates the
    # domain average and drags the whole assessment with it.
    return max(-Z_CLAMP, min(Z_CLAMP, z))


def apply_norms(
    scores: list[TestScore], age: int, education_years: int | None
) -> list[TestScore]:
    for s in scores:
        if s.test in NORMS:
            s.z = normative_z(s.test, s.raw, age, education_years)
    return scores


def domain_summary(scores: list[TestScore]) -> dict[str, float]:
    """Mean z per cognitive domain, for the results radar chart."""
    buckets: dict[str, list[float]] = {}
    for s in scores:
        if s.test in NORMS:
            buckets.setdefault(DOMAIN_OF_TEST[s.test], []).append(s.z)
    return {d: sum(v) / len(v) for d, v in buckets.items() if v}
