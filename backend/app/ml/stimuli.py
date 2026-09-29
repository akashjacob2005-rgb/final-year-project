"""Randomised test stimuli, generated server-side.

Generated per session and persisted, for two reasons:
  * repeat testing — a user taking the assessment monthly must not be shown the
    same five objects every time, or later scores measure recall of the app
    rather than memory;
  * integrity — scoring compares against the stored stimuli, never against what
    the client claims the correct answer was.

DIFFICULTY TIERS
Task difficulty is scaled by age. The reason is ceiling effects, not fairness:
a healthy 30-year-old scores 5/5 on memory recognition, 5/5 on serial sevens
and tops out the digit span ladder, which leaves no headroom to detect *early*
decline in a younger user. Age fairness is already handled separately, by the
normative z-scores in ml/scoring.py.

Because both mechanisms are active, the two must agree: the tier bands here are
the same bands as NORMS in ml/scoring.py, and the norms are referenced to the
difficulty each band actually receives. Changing a tier's parameters without
re-estimating that band's norm will silently mis-score everyone in it.

CALIBRATION PASS — every tier was made easier
The earlier settings were too hard for healthy people, not just for declining
ones. Pilot users in the top tier were asked to hold 7 objects across a 25-second
filled delay, reverse a 6-digit span, trace a 14-node alternating trail and
subtract 7 repeatedly from 193. Those are above the level the source instruments
actually use, and a test that a healthy person fails measures anxiety and task
novelty rather than cognition — the floor effect destroys discrimination just as
surely as the ceiling effect the tiers were introduced to fix.

What changed, in every tier:
  * fewer memory targets, with MORE study time per object and a shorter delay;
  * the digit-span ladder capped at 5 (a backward span of 6 is beyond typical
    adult ability) and digits presented more slowly, older tiers slowest;
  * shorter trails;
  * serial sevens from 100 everywhere — the MoCA anchor — instead of a
    three-digit start invented to make the task harder.
Difficulty still increases as age decreases; the whole scale simply sits lower.

Every corresponding band in NORMS was re-referenced in the same commit. The
tests that hold the two in step are in backend/tests/test_api.py, under
"age-scaled difficulty".
"""

from __future__ import annotations

import random

# Concrete, high-imageability, everyday nouns. Deliberately avoids near-synonyms
# and category-mates within a set, which would make recognition ambiguous.
OBJECT_POOL = [
    "apple", "table", "coin", "rose", "truck", "candle", "hammer", "guitar",
    "rabbit", "mirror", "ladder", "pillow", "anchor", "kettle", "violin",
    "balloon", "compass", "envelope", "feather", "jacket", "lantern", "pencil",
    "saddle", "teapot", "window", "basket", "camera", "dolphin", "engine",
]

# Six pairs is the longest trail any tier now uses; each tier slices off the
# front. The list keeps a seventh pair so a tier can be lengthened again without
# editing this constant, but nothing currently reaches it.
TRAIL_NUMBERS = ["1", "2", "3", "4", "5", "6", "7"]
TRAIL_LETTERS = ["A", "B", "C", "D", "E", "F", "G"]
FLUENCY_LETTERS = ["F", "A", "S"]  # F is the MoCA letter; A/S are standard alternates

# --------------------------------------------------------------------------
# Tiers
# --------------------------------------------------------------------------
# Bands are identical to NORMS in ml/scoring.py. The 60 and 75 cut-points are
# the ones MoCA and the neuropsychological literature use; 40 splits what would
# otherwise be a very wide 18-59 band containing quite different populations.
TIER_BANDS = (
    ("demanding", 18, 39),
    ("challenging", 40, 59),
    ("moderate", 60, 74),
    ("standard", 75, 120),
)

# Age is optional on a session, and an unknown age must not silently produce an
# easier or harder test than the norms expect. "moderate" is the pre-existing
# difficulty, so an unknown age reproduces the app's historical behaviour.
DEFAULT_TIER = "moderate"

# Encoding time must scale with the number of objects. Difficulty belongs in how
# MUCH is remembered and for how LONG, never in how fast it flashes past: below
# roughly 1.5 s per object nobody encodes the list at all, and the test stops
# measuring memory and starts measuring reading speed. An earlier version of this
# table gave the 7-object tiers 6 seconds (0.86 s/object, versus 1.60 in the
# original 5-object test) and produced 0/7 scores from healthy users.
# test_memory_encoding_time_is_feasible enforces the floor.
#
# Every tier now sits at or above 2.0 s/object, comfortably clear of the floor.
MIN_SECONDS_PER_OBJECT = 1.5

# The digit-span ladder tops out at 5 in every tier. A backward span of 6 is
# already beyond typical adult ability (normal is about 4-5), so the sixth rung
# was failed by nearly everyone and carried no information — it only added a
# round that felt like failure before the test moved on.
MAX_DIGIT_SPAN = 5

TIER_PARAMS: dict[str, dict] = {
    "demanding": {
        "memory_targets": 6,
        "memory_distractors": 12,
        "study_seconds": 12,      # 2.0 s/object
        "delay_seconds": 15,
        # Ladders EXTEND upward, they never shift off 3 — see _digit_span_backward.
        "digit_lengths": [3, 4, 5],
        # How long each digit stays on screen. Slower is easier: the digits have
        # to be held AND reversed, and older tiers get the most time. This does
        # not touch the raw measure (longest span reproduced), but it does change
        # the norm — the digit_span_backward band was re-referenced to match.
        "digit_interval_ms": 1200,
        "trail_pairs": 6,         # 12 nodes
        "sevens_start": 100,
    },
    "challenging": {
        "memory_targets": 5,
        "memory_distractors": 10,
        "study_seconds": 10,      # 2.0 s/object
        "delay_seconds": 12,
        "digit_lengths": [3, 4, 5],
        "digit_interval_ms": 1300,
        "trail_pairs": 5,         # 10 nodes
        "sevens_start": 100,
    },
    "moderate": {
        "memory_targets": 4,
        "memory_distractors": 8,
        "study_seconds": 8,       # 2.0 s/object
        "delay_seconds": 10,
        "digit_lengths": [3, 4, 5],
        "digit_interval_ms": 1400,
        "trail_pairs": 4,         # 8 nodes
        "sevens_start": 100,
    },
    "standard": {
        # Target count held at 4 rather than cut to 3. Hits are the raw measure,
        # so a 3-item list leaves only four distinguishable scores and the test
        # stops grading. This tier gets its relief from time and from the
        # shorter trail instead: 2.5 s per object to study, the shortest delay,
        # and the slowest digits.
        "memory_targets": 4,
        "memory_distractors": 8,
        "study_seconds": 10,      # 2.5 s/object
        "delay_seconds": 8,
        "digit_lengths": [3, 4, 5],
        "digit_interval_ms": 1500,
        "trail_pairs": 4,         # 8 nodes
        "sevens_start": 100,
    },
}


def tier_for_age(age: int | None) -> str:
    """Map an age to a difficulty tier. Unknown age -> the historical default."""
    if age is None:
        return DEFAULT_TIER
    try:
        age = int(age)
    except (TypeError, ValueError):
        return DEFAULT_TIER
    for name, lo, hi in TIER_BANDS:
        if lo <= age <= hi:
            return name
    # Below the youngest band: use the hardest tier rather than falling through
    # to the default, so an out-of-range age is never accidentally made easier.
    return TIER_BANDS[0][0] if age < TIER_BANDS[0][1] else TIER_BANDS[-1][0]


def _memory_recognition(rng: random.Random, p: dict) -> dict:
    n_targets = p["memory_targets"]
    total = n_targets + p["memory_distractors"]
    pool = rng.sample(OBJECT_POOL, total)
    targets = pool[:n_targets]
    distractors = pool[n_targets:total]
    grid = targets + distractors
    rng.shuffle(grid)
    return {
        "targets": targets,
        "grid": grid,
        "study_seconds": p["study_seconds"],
        "delay_seconds": p["delay_seconds"],
    }


def _trail_making_b(rng: random.Random, p: dict) -> dict:
    """Alternating 1-A-2-B-... as in MoCA, with a non-overlapping node layout."""
    pairs = p["trail_pairs"]
    expected = [
        v
        for pair in zip(TRAIL_NUMBERS[:pairs], TRAIL_LETTERS[:pairs])
        for v in pair
    ]  # 1,A,2,B,3,C,...

    # Jittered grid cells: keeps nodes apart (so clicks are unambiguous) while
    # still randomising the path between sessions. Four columns; the number of
    # rows follows the node count. The 3-row spacing is the original layout, so
    # the 8-, 10- and 12-node tiers all render on the familiar geometry.
    cols = 4
    n_nodes = len(expected)
    rows = -(-n_nodes // cols)  # ceil
    y0, y_step = (18, 28) if rows <= 3 else (14, 22)

    cells = [(c, r) for r in range(rows) for c in range(cols)][:n_nodes]
    rng.shuffle(cells)
    nodes = []
    for label, (col, row) in zip(expected, cells):
        nodes.append(
            {
                "label": label,
                "x": round(12 + col * 25 + rng.uniform(-4, 4), 2),  # percent
                "y": round(y0 + row * y_step + rng.uniform(-5, 5), 2),
            }
        )
    rng.shuffle(nodes)
    return {"nodes": nodes, "expected": expected}


def _digit_span_backward(rng: random.Random, p: dict) -> dict:
    """Adaptive backward span.

    Every tier's ladder starts at 3. Harder tiers add rungs at the top rather
    than removing them from the bottom, because scoring records the longest
    sequence reproduced *correctly*: on a ladder starting at 5, someone whose
    true span is 4 fails every trial and scores 0, so the raw measure would jump
    from 0 straight to 5 and a mild deficit would read as a severe one.

    Every tier now runs the same 3-4-5 ladder; tiers differ only in how long
    each digit is shown. Keeping three rungs everywhere is deliberate — a
    two-rung ladder yields only the raw values 0, 3 and 4, which is too coarse
    to grade with.
    """
    trials = [
        {"length": n, "sequence": [rng.randint(0, 9) for _ in range(n)]}
        for n in p["digit_lengths"]
    ]
    return {"trials": trials, "digit_interval_ms": p["digit_interval_ms"]}


def _serial_sevens(p: dict) -> dict:
    # No longer tiered. Younger tiers used to start from an odd three-digit
    # number (193) to force borrows, but 100 already borrows on every step —
    # the three-digit start added arithmetic load, not attentional load, which
    # is what this item is supposed to measure. Everyone now sits the MoCA
    # version: 100, minus 7, five times.
    return {"start": p["sevens_start"], "steps": 5, "subtract": 7}


def _verbal_fluency(rng: random.Random) -> dict:
    # Not tiered: a 60-second word count has no ceiling to raise, and changing
    # the duration would break comparability with the published MoCA threshold.
    return {"letter": rng.choice(FLUENCY_LETTERS), "duration_seconds": 60}


def _picture_description() -> dict:
    # Not tiered, and must not become tiered: the trained language model was
    # fitted on descriptions of this exact scene, and its metrics assume the
    # 25-word floor. Varying the stimulus would invalidate the model.
    return {
        "image": "/assets/kitchen-scene.svg",
        "prompt": "Tell me everything you see going on in this picture.",
        "min_seconds": 20,
        "max_seconds": 120,
        # The trained model needs a reasonable amount of speech to be reliable.
        "min_words": 25,
    }


def generate(seed: int | None = None, age: int | None = None) -> dict:
    """Build one session's stimuli, with difficulty scaled to the user's age."""
    rng = random.Random(seed)
    tier = tier_for_age(age)
    p = TIER_PARAMS[tier]
    return {
        # Persisted so a later tier change cannot retroactively alter how an old
        # session is read, and so the history view can flag a tier crossover.
        "tier": tier,
        "memory_recognition": _memory_recognition(rng, p),
        "picture_description": _picture_description(),
        "trail_making_b": _trail_making_b(rng, p),
        "serial_sevens": _serial_sevens(p),
        "digit_span_backward": _digit_span_backward(rng, p),
        "verbal_fluency": _verbal_fluency(rng),
    }


def client_safe(stimuli: dict) -> dict:
    """Strip answers the client must not see before responding.

    The digit sequences and memory targets ARE sent — they have to be shown to
    be memorised. What is withheld is the Trail Making expected path, which the
    user is supposed to work out for themselves.
    """
    # Non-dict values (e.g. the "tier" string) are passed through untouched.
    out = {k: (dict(v) if isinstance(v, dict) else v) for k, v in stimuli.items()}
    if isinstance(out.get("trail_making_b"), dict):
        out["trail_making_b"].pop("expected", None)
    return out
