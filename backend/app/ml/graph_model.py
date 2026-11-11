"""Graph pathway: connectivity graph (rs-fMRI [+ DTI]) -> GAT -> Normal / MCI / Dementia.

A THIRD, separate pathway, like MRI: no dataset pairs connectivity data with
the app's six tests, so the result is never fused into the behavioural risk
score (ml/fusion.py keeps a disabled hook for when paired data exists).

Inference is pure numpy (ml/graph/gat_numpy.py) on weights exported by
ml/graph/train_graph.py — no torch on the server. Until that script has run on
real data there are no artifacts: the pathway then reports "not trained" and
the endpoint returns 503. It never fabricates probabilities.

Gated by GRAPH_ENABLED, failing closed in production exactly like MRI_ENABLED.
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache

from . import paths

GRAPH_MODEL_PATH = paths.ARTIFACTS_DIR / "graph_model.npz"
GRAPH_CONFIG_PATH = paths.ARTIFACTS_DIR / "graph_config.json"
GRAPH_METRICS_PATH = paths.ARTIFACTS_DIR / "graph_metrics.json"

TOP_EDGES = 10
NOT_TRAINED = (
    "The graph model has not been trained yet: no resting-state fMRI / DTI "
    "training data has been processed, so there is no model to run."
)

_lock = threading.Lock()
_cache: dict = {}


class GraphError(Exception):
    """The uploaded data cannot be analysed (client's fault) -> 422."""


class GraphNotTrained(Exception):
    """No trained artifact exists -> 503 with NOT_TRAINED."""


def enabled() -> bool:
    flag = os.getenv("GRAPH_ENABLED")
    if flag is not None:
        return flag.lower() in {"1", "true", "yes"}
    # Fail closed in production: only an explicit env var turns it on there,
    # so a stray .env value can't load nilearn next to Whisper on a 512MB box.
    if os.getenv("ENVIRONMENT", "development").lower() == "production":
        return False
    from ..config import settings

    return bool(settings.graph_enabled)


def trained() -> bool:
    return GRAPH_MODEL_PATH.exists() and GRAPH_CONFIG_PATH.exists()


@lru_cache(maxsize=1)
def model_metrics() -> dict | None:
    if not GRAPH_METRICS_PATH.exists():
        return None
    return json.loads(GRAPH_METRICS_PATH.read_text())


def reset() -> None:
    with _lock:
        _cache.clear()
    model_metrics.cache_clear()


def _load() -> tuple[dict, dict]:
    with _lock:
        if "weights" not in _cache:
            from graph.gat_numpy import load_weights  # shared ml/ code

            _cache["config"] = json.loads(GRAPH_CONFIG_PATH.read_text())
            _cache["weights"] = load_weights(GRAPH_MODEL_PATH)
    return _cache["weights"], _cache["config"]


def _atlas(name: str):
    """Labels image for the atlas the model was trained on (patched in tests)."""
    from graph.atlas import load_atlas

    return load_atlas(name)[0]


def parse_sc(raw: bytes | None, suffix: str | None, n_regions: int):
    """Validate an uploaded SC matrix against the model's region count."""
    if raw is None:
        return None
    from graph.graph_build import MatrixError, load_matrix_bytes, validate_matrix

    try:
        return validate_matrix(load_matrix_bytes(raw, suffix or ""), n_regions)
    except MatrixError as exc:
        raise GraphError(str(exc)) from exc


def analyze(fmri_bytes: bytes, sc_raw: bytes | None = None, sc_suffix: str | None = None) -> dict:
    if not trained():
        raise GraphNotTrained(NOT_TRAINED)
    from graph import gat_numpy
    from graph.connectivity import BoldError, fc_from_bold, load_bold_bytes
    from graph.graph_build import MatrixError, build_graph

    weights, config = _load()
    n = int(config["n_regions"])
    # Cheap checks first: a bad matrix should fail before minutes of fMRI work.
    sc = parse_sc(sc_raw, sc_suffix, n)
    if config.get("with_sc") and sc is None:
        raise GraphError("this model was trained with DTI structural connectivity; "
                         "upload the SC matrix (.csv or .npy) as well")
    try:
        img = load_bold_bytes(fmri_bytes)
        fc = fc_from_bold(img, _atlas(config["atlas"]))
    except BoldError as exc:
        raise GraphError(str(exc)) from exc
    except Exception as exc:  # nilearn: affine/grid mismatches, resampling failures
        raise GraphError(f"could not extract region time series from this scan: {exc}") from exc
    if fc.shape != (n, n):
        raise GraphError(f"the atlas produced {fc.shape[0]} regions but the model expects {n}")
    names = config.get("class_short") or ["Normal", "MCI", "Dementia"]
    if config.get("model_type") == "logreg":
        # The baseline won in cross-validation, so it is the deployed model.
        from graph import linear_numpy

        out = linear_numpy.forward(fc, weights)
        probs = out["probabilities"]
        top_edges = linear_numpy.top_edges(fc, out["contributions"], config["region_names"], k=TOP_EDGES)
    else:
        try:
            graph = build_graph(fc, sc if config.get("with_sc") else None,
                                top_k=config.get("top_k"), threshold=config.get("threshold"),
                                fisher_z=bool(config.get("fisher_z")))
        except MatrixError as exc:
            raise GraphError(str(exc)) from exc
        out = gat_numpy.forward(graph, weights, config)
        probs = out["probabilities"]
        top_edges = gat_numpy.top_edges(graph, out["attention"], config["region_names"], k=TOP_EDGES)
    top = int(probs.argmax())
    caveats = list((model_metrics() or {}).get("caveats", []))
    if sc is not None and not config.get("with_sc"):
        caveats.insert(0, "This model was trained without DTI, so the uploaded structural "
                          "connectivity matrix was validated but not used.")
    return {
        "available": True,
        "probabilities": {k: round(float(p), 4) for k, p in zip(names, probs)},
        "predicted_class": names[top],
        "predicted_index": top,
        "top_edges": top_edges,
        "model_type": config.get("model_type", "gat"),
        "n_regions": n,
        "used_sc": bool(config.get("with_sc")),
        "caveats": caveats,
        "model_version": config.get("version"),
        "note": ("Research prototype trained on OASIS-3 connectivity. Reported separately "
                 "from the behavioural risk score; not a diagnosis."),
    }
