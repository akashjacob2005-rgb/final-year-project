"""Numpy inference for the logistic-regression connectivity model.

train_graph.py always trains this model as the baseline the GAT must beat.
When the GAT does not beat it in cross-validation, this is the model that is
deployed (graph_config.json: "model_type": "logreg"), because shipping the
weaker model would be dishonest.

Features: the Fisher-z upper triangle of the N x N functional connectivity
matrix, standardised with the training mean/scale, then a linear layer and
softmax (sklearn's multinomial / binary logistic regression, reproduced
exactly). Explanation: the region pairs whose term (coefficient x standardised
connection strength) pushed this subject's score most towards the predicted
class.
"""

from __future__ import annotations

import numpy as np


def fc_features(fc: np.ndarray) -> np.ndarray:
    """Fisher-z upper triangle (k=1) of one N x N matrix, or of a stack (S, N, N)."""
    n = fc.shape[-1]
    iu = np.triu_indices(n, k=1)
    tri = fc[..., iu[0], iu[1]]
    return np.arctanh(np.clip(tri, -0.999, 0.999))


def forward(fc: np.ndarray, weights: dict) -> dict:
    """Class probabilities and per-feature contributions for one FC matrix."""
    z = (fc_features(fc) - weights["scaler_mean"]) / weights["scaler_scale"]
    coef = np.atleast_2d(weights["coef"])            # (1, F) binary, (C, F) multiclass
    logits = coef @ z + weights["intercept"]
    if coef.shape[0] == 1:                           # sklearn binary: sigmoid of one logit
        p1 = 1.0 / (1.0 + np.exp(-logits[0]))
        probs = np.array([1.0 - p1, p1])
        contrib = coef[0] * z                        # towards class 1
        sign = 1.0 if p1 >= 0.5 else -1.0            # towards the predicted class
    else:
        e = np.exp(logits - logits.max())
        probs = e / e.sum()
        top = int(probs.argmax())
        contrib = coef[top] * z
        sign = 1.0
    return {"probabilities": probs, "contributions": sign * contrib}


def top_edges(fc: np.ndarray, contributions: np.ndarray, region_names: list[str], k: int = 10) -> list[dict]:
    """Region pairs that pushed the prediction most towards the predicted class."""
    n = fc.shape[0]
    iu = np.triu_indices(n, k=1)
    order = np.argsort(-contributions)[:k]
    return [
        {"region_a": region_names[iu[0][i]], "region_b": region_names[iu[1][i]],
         "importance": round(float(contributions[i]), 4), "fc": round(float(fc[iu[0][i], iu[1][i]]), 4)}
        for i in order if contributions[i] > 0
    ]
