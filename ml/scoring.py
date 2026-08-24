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
is tiered by age (see backend/app/ml/stimuli.py), so four of the five tests below
are normed against a harder variant for younger users: more memory targets, a
deeper digit-span ladder, a 14-node trail, a serial-sevens start that forces
borrows. No published norms exist for those variants — the literature norms the
standard forms. The means/SDs for the (18,39) and (40,59) bands are therefore
estimates extrapolated from the standard-form values, not measurements.

verbal_fluency is the one exception: its task is identical in every tier, so its
young bands are a straight split of a published range and carry only the original
caveat. Treat the other four young-band figures as the least reliable numbers in
this file, and re-derive all of them from local pilot data before any real use.
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

    Admissibility filtering matters — without it the count rewards repetition
    and inflected variants of one root, which is exactly the pattern a declining
    participant produces.
    """
    letter = letter.lower()
    seen: set[str] = set()
    valid: list[str] = []
    rejected: dict[str, list[str]] = {"wrong_letter": [], "repeat": [], "invalid": []}

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
        if w in seen:
            rejected["repeat"].append(w)
            continue
        seen.add(w)
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
# memory targets, a deeper digit-span ladder, a 14-node trail and a serial-sevens
# start that forces borrows; their norm is referenced to that harder variant.
# Changing a tier's parameters in stimuli.py without re-estimating its band here
# will silently mis-score everyone in that band — most likely by making healthy
# young users look impaired, because they would be measured against a norm set
# for an easier test.
NORMS: dict[str, dict] = {
    "memory_recognition": {  # hits, out of a tier-dependent number of targets
        # SDs on the 7-target tiers are wider than the 5-target one: recognition
        # hit counts spread out as the list lengthens, and a tight SD against a
        # score that can hit the floor manufactures implausible z-values.
        "higher_is_better": True,
        "bands": {
            (18, 39): (6.0, 1.3),    # 7 targets, 11s study, 25s delay
            (40, 59): (6.1, 1.3),    # 7 targets, 11s study, 20s delay
            (60, 74): (4.3, 0.9),    # 5 targets — unchanged
            (75, 120): (3.4, 0.8),   # 4 targets, 10s study
        },
    },
    "trail_making_b": {  # seconds to complete
        "higher_is_better": False,
        "bands": {
            (18, 39): (34.0, 11.0),  # 14 nodes
            (40, 59): (39.0, 13.0),  # 14 nodes
            (60, 74): (32.0, 13.0),  # 10 nodes — unchanged
            (75, 120): (45.0, 20.0),  # 10 nodes — unchanged
        },
    },
    "serial_sevens": {  # number of correct subtractions, always out of 5
        "higher_is_better": True,
        "bands": {
            (18, 39): (4.3, 1.0),    # from 193
            (40, 59): (4.1, 1.1),    # from 193
            (60, 74): (4.1, 1.2),    # from 100 — unchanged
            (75, 120): (3.6, 1.5),   # from 100 — unchanged
        },
    },
    "digit_span_backward": {  # longest span reproduced correctly
        # Re-referenced when the ladders were shortened. Leaving the old means in
        # place would have penalised a PERFECT score: on a 2-rung ladder the best
        # achievable raw is 4, against a mean of 4.6, so a flawless performance
        # would have scored z = -0.50. A norm must never sit above the ceiling of
        # the test it norms.
        "higher_is_better": True,
        "bands": {
            (18, 39): (4.8, 1.1),    # ladder 3-6, 4 rounds
            (40, 59): (4.2, 0.9),    # ladder 3-5, 3 rounds
            (60, 74): (4.0, 0.9),    # ladder 3-5, 3 rounds
            (75, 120): (3.7, 0.9),   # ladder 3-5, 3 rounds
        },
    },
    "verbal_fluency": {  # admissible words in 60s
        # Not tiered — every band sits the identical 60-second task, so this is a
        # straight split of the old 18-59 band and involves no re-referencing.
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
