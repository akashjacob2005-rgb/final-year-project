"""Dispatch a submitted test response to the correct scorer.

Scoring always uses the stored stimuli for the session, never values echoed back
by the client.
"""

from __future__ import annotations

from . import paths  # noqa: F401  (side effect: puts ml/ on sys.path)
from fusion import fuse  # noqa: E402
from scoring import (  # noqa: E402
    DOMAIN_OF_TEST,
    TestScore,
    apply_norms,
    domain_summary,
    normative_z,
    score_digit_span_backward,
    score_memory_recognition,
    score_serial_sevens,
    score_trail_making_b,
    score_verbal_fluency,
)

# Re-exported so callers get the shared ml/ modules through one import that is
# guaranteed to have run the sys.path setup first.
__all__ = [
    "DOMAIN_OF_TEST",
    "ScoringError",
    "TestScore",
    "apply_norms",
    "domain_summary",
    "fuse",
    "normative_z",
    "score_submission",
]


class ScoringError(ValueError):
    pass


def _as_int_list(values) -> list[int]:
    out = []
    for v in values or []:
        try:
            out.append(int(str(v).strip()))
        except (TypeError, ValueError):
            continue
    return out


def score_submission(
    test_name: str,
    raw_response: dict,
    stimuli: dict,
    duration_ms: int | None = None,
) -> TestScore:
    spec = (stimuli or {}).get(test_name, {})

    if test_name == "memory_recognition":
        targets = spec.get("targets") or []
        if not targets:
            raise ScoringError("Session stimuli are missing the memory targets")
        return score_memory_recognition(
            selected=[str(s) for s in raw_response.get("selected", [])],
            targets=[str(t) for t in targets],
        )

    if test_name == "trail_making_b":
        expected = spec.get("expected") or []
        if not expected:
            raise ScoringError("Session stimuli are missing the trail sequence")
        return score_trail_making_b(
            clicked=[str(c) for c in raw_response.get("clicked", [])],
            expected=[str(e) for e in expected],
            duration_ms=float(
                raw_response.get("duration_ms") or duration_ms or 0.0
            ),
        )

    if test_name == "serial_sevens":
        responses: list[int | None] = []
        for v in (raw_response.get("responses") or [])[:5]:
            if v is None or str(v).strip() == "":
                responses.append(None)
            else:
                try:
                    responses.append(int(str(v).strip()))
                except ValueError:
                    responses.append(None)
        return score_serial_sevens(responses, start=int(spec.get("start", 100)))

    if test_name == "digit_span_backward":
        # Pair each submitted response with the sequence actually presented.
        presented = {
            i: t.get("sequence", []) for i, t in enumerate(spec.get("trials") or [])
        }
        trials = []
        for i, t in enumerate(raw_response.get("trials") or []):
            seq = presented.get(i)
            if seq is None:
                continue
            trials.append(
                {"sequence": _as_int_list(seq), "response": _as_int_list(t.get("response"))}
            )
        return score_digit_span_backward(trials)

    if test_name == "verbal_fluency":
        return score_verbal_fluency(
            words=[str(w) for w in raw_response.get("words", [])],
            letter=str(spec.get("letter", "F")),
            duration_s=float(spec.get("duration_seconds", 60)),
        )

    if test_name == "picture_description":
        raise ScoringError(
            "picture_description is scored by the language model, not this dispatcher"
        )

    raise ScoringError(f"Unknown test: {test_name}")
