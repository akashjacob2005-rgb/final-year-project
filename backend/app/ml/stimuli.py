"""Randomised test stimuli, generated server-side.

Generated per session and persisted, for two reasons:
  * repeat testing — a user taking the assessment monthly must not be shown the
    same five objects every time, or later scores measure recall of the app
    rather than memory;
  * integrity — scoring compares against the stored stimuli, never against what
    the client claims the correct answer was.
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

DIGIT_SPAN_LENGTHS = [3, 4, 5, 6]
TRAIL_NUMBERS = ["1", "2", "3", "4", "5"]
TRAIL_LETTERS = ["A", "B", "C", "D", "E"]
FLUENCY_LETTERS = ["F", "A", "S"]  # F is the MoCA letter; A/S are standard alternates


def _memory_recognition(rng: random.Random) -> dict:
    pool = rng.sample(OBJECT_POOL, 15)
    targets = pool[:5]
    distractors = pool[5:15]
    grid = targets + distractors
    rng.shuffle(grid)
    return {
        "targets": targets,
        "grid": grid,
        "study_seconds": 8,
        "delay_seconds": 10,
    }


def _trail_making_b(rng: random.Random) -> dict:
    """Alternating 1-A-2-B-... as in MoCA, with a non-overlapping node layout."""
    expected = [
        v for pair in zip(TRAIL_NUMBERS, TRAIL_LETTERS) for v in pair
    ]  # 1,A,2,B,3,C,4,D,5,E

    # Jittered grid cells: keeps nodes apart (so clicks are unambiguous) while
    # still randomising the path between sessions.
    cells = [(c, r) for r in range(3) for c in range(4)][:10]
    rng.shuffle(cells)
    nodes = []
    for label, (col, row) in zip(expected, cells):
        nodes.append(
            {
                "label": label,
                "x": round(12 + col * 25 + rng.uniform(-4, 4), 2),  # percent
                "y": round(18 + row * 28 + rng.uniform(-5, 5), 2),
            }
        )
    rng.shuffle(nodes)
    return {"nodes": nodes, "expected": expected}


def _digit_span_backward(rng: random.Random) -> dict:
    trials = [
        {"length": n, "sequence": [rng.randint(0, 9) for _ in range(n)]}
        for n in DIGIT_SPAN_LENGTHS
    ]
    return {"trials": trials, "digit_interval_ms": 1000}


def _serial_sevens() -> dict:
    return {"start": 100, "steps": 5, "subtract": 7}


def _verbal_fluency(rng: random.Random) -> dict:
    return {"letter": rng.choice(FLUENCY_LETTERS), "duration_seconds": 60}


def _picture_description() -> dict:
    return {
        "image": "/assets/kitchen-scene.svg",
        "prompt": "Tell me everything you see going on in this picture.",
        "min_seconds": 20,
        "max_seconds": 120,
        # The trained model needs a reasonable amount of speech to be reliable.
        "min_words": 25,
    }


def generate(seed: int | None = None) -> dict:
    rng = random.Random(seed)
    return {
        "memory_recognition": _memory_recognition(rng),
        "picture_description": _picture_description(),
        "trail_making_b": _trail_making_b(rng),
        "serial_sevens": _serial_sevens(),
        "digit_span_backward": _digit_span_backward(rng),
        "verbal_fluency": _verbal_fluency(rng),
    }


def client_safe(stimuli: dict) -> dict:
    """Strip answers the client must not see before responding.

    The digit sequences and memory targets ARE sent — they have to be shown to
    be memorised. What is withheld is the Trail Making expected path, which the
    user is supposed to work out for themselves.
    """
    out = {k: dict(v) for k, v in stimuli.items()}
    out["trail_making_b"].pop("expected", None)
    return out
