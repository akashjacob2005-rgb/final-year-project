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

# Extended to 7 pairs; each tier slices off the front.
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
MIN_SECONDS_PER_OBJECT = 1.5

TIER_PARAMS: dict[str, dict] = {
    "demanding": {
        "memory_targets": 7,
        "memory_distractors": 14,
        "study_seconds": 11,
        "delay_seconds": 25,
        # Ladders EXTEND upward, they never shift off 3 — see _digit_span_backward.
        # Capped at 6: a backward span of 7-8 is well beyond typical adult
        # ability (normal is about 4-5), so the top rungs measured almost nobody
        # and made the test feel punishing rather than discriminating.
        "digit_lengths": [3, 4, 5, 6],
        "trail_pairs": 7,
        "sevens_start": 193,
    },
    "challenging": {
        "memory_targets": 7,
        "memory_distractors": 14,
        "study_seconds": 11,
        "delay_seconds": 20,
        "digit_lengths": [3, 4, 5],
        "trail_pairs": 7,
        "sevens_start": 193,
    },
    "moderate": {  # the app's original difficulty, unchanged
        "memory_targets": 5,
        "memory_distractors": 10,
        "study_seconds": 8,
        "delay_seconds": 10,
        "digit_lengths": [3, 4, 5],
        "trail_pairs": 5,
        "sevens_start": 100,
    },
    "standard": {
        "memory_targets": 4,
        "memory_distractors": 8,
        "study_seconds": 10,
        "delay_seconds": 10,
        "digit_lengths": [3, 4, 5],
        "trail_pairs": 5,
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
    # rows follows the node count. The 3-row spacing is the original layout and
    # is preserved exactly, so the 10-node tiers render as they always have.
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
    """
    trials = [
        {"length": n, "sequence": [rng.randint(0, 9) for _ in range(n)]}
        for n in p["digit_lengths"]
    ]
    return {"trials": trials, "digit_interval_ms": 1000}


def _serial_sevens(p: dict) -> dict:
    # Harder tiers start from an odd three-digit number, which forces a borrow
    # on most steps. The subtrahend stays at 7 and the count stays at 5 so the
    # MoCA scoring rule and the raw measure remain comparable across tiers.
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
