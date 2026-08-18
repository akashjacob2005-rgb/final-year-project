"""Combine the trained language model with the structured-test z-scores into a
single screening risk score.

READ THIS BEFORE CITING THE OUTPUT
----------------------------------
This layer is NOT machine-learned, and the report must not claim it is. No
public dataset exists in which people complete *these six specific tests* and
carry a diagnostic label, so there is nothing to fit fusion weights against.

What it is instead: a transparent, documented weighting of two components —
  * p_language : output of the trained DementiaBank classifier (ROC-AUC 0.905,
                 cross-validated). This part IS empirically grounded.
  * p_structured : derived from age/education-adjusted z-scores using the
                 standard neuropsychological convention that z <= -1.5 marks
                 impairment. Norm-referenced, not learned.

Every constant below is a stated judgement, visible and adjustable, rather than
a fitted parameter presented as science.
"""

from __future__ import annotations

import math

# Equal weighting. Rationale: the language component is the only empirically
# validated part, which argues for weighting it heavily; the structured battery
# covers four cognitive domains the language test does not touch, which argues
# the other way. Equal weight is the defensible middle, and it is a judgement.
WEIGHT_LANGUAGE = 0.5
WEIGHT_STRUCTURED = 0.5

# Neuropsychological convention: z <= -1.0 borderline, z <= -1.5 impaired.
# Centring the logistic at z = -1.0 puts a typical healthy performer (z = 0)
# at low risk and a clearly impaired one (z = -2) at high risk.
Z_CENTRE = -1.0
Z_STEEPNESS = 1.5

# Screening bands, not diagnostic categories.
BAND_LOW = 0.35
BAND_ELEVATED = 0.65

DOMAIN_WEIGHTS = {
    "memory": 1.3,          # earliest and most sensitive domain in decline
    "executive": 1.0,
    "attention": 1.0,
    "working_memory": 1.0,
}


def _sigmoid(x: float) -> float:
    if x >= 0:
        return 1.0 / (1.0 + math.exp(-x))
    e = math.exp(x)
    return e / (1.0 + e)


def structured_probability(domain_z: dict[str, float]) -> tuple[float, float]:
    """Map domain z-scores to a risk-like value in [0, 1].

    Returns (probability, weighted_composite_z).
    """
    if not domain_z:
        return 0.5, 0.0

    num = sum(DOMAIN_WEIGHTS.get(d, 1.0) * z for d, z in domain_z.items())
    den = sum(DOMAIN_WEIGHTS.get(d, 1.0) for d in domain_z)
    composite = num / den if den else 0.0
    return _sigmoid(-(composite - Z_CENTRE) * Z_STEEPNESS), composite


def band_for(risk: float) -> str:
    if risk < BAND_LOW:
        return "Low"
    if risk < BAND_ELEVATED:
        return "Borderline"
    return "Elevated"


def fuse(
    language_probability: float | None,
    domain_z: dict[str, float],
    n_tests_completed: int,
    n_tests_total: int = 6,
) -> dict:
    """Produce the final screening result.

    If the language component is unavailable (no/too-short recording, or a
    transcription failure) the structured component carries the whole score and
    confidence is explicitly reduced rather than the failure being hidden.
    """
    p_struct, composite_z = structured_probability(domain_z)

    if language_probability is None:
        risk = p_struct
        weights = {"language": 0.0, "structured": 1.0}
        notes = [
            "Language analysis unavailable, so this score reflects the "
            "structured tests only. Treat it as less reliable."
        ]
    else:
        risk = (
            WEIGHT_LANGUAGE * language_probability
            + WEIGHT_STRUCTURED * p_struct
        )
        weights = {"language": WEIGHT_LANGUAGE, "structured": WEIGHT_STRUCTURED}
        notes = []

    completeness = n_tests_completed / max(1, n_tests_total)
    confidence = "high"
    if completeness < 0.5:
        confidence = "low"
        notes.append(
            f"Only {n_tests_completed} of {n_tests_total} tests were completed."
        )
    elif completeness < 1.0 or language_probability is None:
        confidence = "moderate"
        if completeness < 1.0:
            notes.append(
                f"{n_tests_completed} of {n_tests_total} tests completed."
            )

    weakest = min(domain_z.items(), key=lambda kv: kv[1]) if domain_z else None

    return {
        "risk": round(risk, 4),
        "risk_percent": round(risk * 100, 1),
        "band": band_for(risk),
        "confidence": confidence,
        "components": {
            "language_probability": (
                round(language_probability, 4)
                if language_probability is not None
                else None
            ),
            "structured_probability": round(p_struct, 4),
            "composite_z": round(composite_z, 3),
        },
        "weights": weights,
        "domain_z": {k: round(v, 3) for k, v in domain_z.items()},
        "weakest_domain": weakest[0] if weakest else None,
        "notes": notes,
        "method": "transparent weighted combination (not a learned fusion model)",
        "disclaimer": (
            "This is a screening indicator for research and educational use. "
            "It is not a diagnosis. Only a qualified clinician can assess "
            "cognitive impairment."
        ),
    }
