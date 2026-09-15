"""MRI pathway: ONNX inference over the OASIS-3-trained slice classifier.

This is deliberately a SEPARATE pathway from the behavioural risk score.
No dataset pairs brain MRI with the app's six tests, so there is no honest way
to learn a combined weight — the MRI result is reported on its own.

The heavy dependencies (onnxruntime, nibabel) are imported lazily and the
whole pathway is gated by MRI_ENABLED so that a memory-constrained deployment
(e.g. a 512MB free tier already carrying Whisper) can switch it off.
"""

from __future__ import annotations

import json
import os
import threading
from functools import lru_cache

import numpy as np

from . import paths

MRI_MODEL_PATH = paths.ARTIFACTS_DIR / "mri_model.onnx"
MRI_METRICS_PATH = paths.ARTIFACTS_DIR / "mri_metrics.json"

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

_FALLBACK_CLASSES = [
    "No impairment signs (CDR 0)",
    "Signs of impairment (CDR >= 0.5)",
]

_lock = threading.Lock()
_session = None


def class_names() -> list[str]:
    """Class labels travel with the trained artifact, not the code."""
    m = model_metrics()
    return (m or {}).get("classes", _FALLBACK_CLASSES)


class MriError(Exception):
    """Raised when an uploaded volume cannot be analysed (client's fault)."""


class MriUnavailable(Exception):
    """Raised when the model artifact cannot be served (our fault)."""


def _sample_dir() -> str | None:
    # Real env var wins (tests/deploys); .env-loaded settings as fallback.
    d = os.getenv("MRI_SAMPLE_DIR")
    if d:
        return d
    from ..config import settings

    return settings.mri_sample_dir


def samples_available() -> bool:
    d = _sample_dir()
    if not d:
        return False
    from pathlib import Path

    p = Path(d)
    return (p / "normal.nii.gz").exists() and (p / "impaired.nii.gz").exists()


def sample_bytes(case: str) -> bytes:
    """Read one of the bundled anonymized research scans (normal|impaired)."""
    from pathlib import Path

    d = _sample_dir()
    if not d:
        raise MriUnavailable("sample scans are not configured on this deployment")
    path = Path(d) / f"{case}.nii.gz"
    if not path.exists():
        raise MriUnavailable(f"sample scan '{case}' is missing")
    return path.read_bytes()


def enabled() -> bool:
    flag = os.getenv("MRI_ENABLED")
    if flag is not None and flag.lower() in {"0", "false", "no"}:
        return False
    # Fail closed on memory-constrained production hosts: unless MRI_ENABLED
    # is set explicitly, production keeps the pathway off. A wiped env var
    # must not silently re-enable onnxruntime next to Whisper on a 512MB box.
    if flag is None and os.getenv("ENVIRONMENT", "development").lower() == "production":
        return False
    return MRI_MODEL_PATH.exists()


def _get_session():
    global _session
    with _lock:
        if _session is None:
            import onnxruntime as ort

            try:
                _session = ort.InferenceSession(
                    str(MRI_MODEL_PATH), providers=["CPUExecutionProvider"]
                )
            except Exception as exc:  # missing .onnx.data sidecar, corrupt file
                raise MriUnavailable(
                    f"MRI model artifact failed to load: {exc}"
                ) from exc
    return _session


# Inferno-like colormap anchors (r, g, b) — avoids a matplotlib dependency.
_CMAP_ANCHORS = np.array(
    [
        (0, 0, 4), (40, 11, 84), (101, 21, 110), (159, 42, 99),
        (212, 72, 66), (245, 125, 21), (250, 193, 39), (252, 255, 164),
    ],
    dtype=np.float32,
)


def _colormap(heat: np.ndarray) -> np.ndarray:
    """heat in [0,1] (H,W) -> uint8 RGB via piecewise-linear inferno."""
    pos = np.clip(heat, 0.0, 1.0) * (len(_CMAP_ANCHORS) - 1)
    lo = np.floor(pos).astype(int)
    hi = np.minimum(lo + 1, len(_CMAP_ANCHORS) - 1)
    frac = (pos - lo)[..., None]
    rgb = _CMAP_ANCHORS[lo] * (1 - frac) + _CMAP_ANCHORS[hi] * frac
    return rgb.astype(np.uint8)


def _png_b64(arr: np.ndarray) -> str:
    import base64
    import io as _io

    from PIL import Image

    mode = "L" if arr.ndim == 2 else "RGB"
    buf = _io.BytesIO()
    Image.fromarray(arr, mode=mode).save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _overlay(gray: np.ndarray, heat01: np.ndarray) -> np.ndarray:
    """Blend a [0,1] heatmap over a uint8 grayscale slice; alpha scales with heat
    so cool regions keep showing anatomy."""
    from PIL import Image

    h, w = gray.shape
    heat = np.asarray(
        Image.fromarray((heat01 * 255).astype(np.uint8), mode="L").resize(
            (w, h), Image.Resampling.BILINEAR
        ),
        dtype=np.float32,
    ) / 255.0
    color = _colormap(heat).astype(np.float32)
    base = np.repeat(gray[..., None], 3, axis=2).astype(np.float32)
    alpha = (0.55 * heat)[..., None]
    return np.clip(base * (1 - alpha) + color * alpha, 0, 255).astype(np.uint8)


def _to_model_input(slices: list[np.ndarray]) -> np.ndarray:
    """uint8 slices -> normalised NCHW batch, mirroring the training transforms."""
    from PIL import Image

    batch = []
    for sl in slices:
        img = Image.fromarray(sl, mode="L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        x = np.asarray(img, dtype=np.float32) / 255.0          # HW
        x = np.repeat(x[None, ...], 3, axis=0)                 # CHW, grey->RGB
        x = (x - IMAGENET_MEAN[:, None, None]) / IMAGENET_STD[:, None, None]
        batch.append(x)
    return np.stack(batch).astype(np.float32)


def predict(volume_bytes: bytes, *, is_dicom_zip: bool = False) -> dict:
    """Analyse an uploaded volume (.nii/.nii.gz, or a zipped DICOM series)."""
    from mri.preprocess import extract_slices, load_nifti_bytes  # shared ml/ code

    try:
        if is_dicom_zip:
            from mri.dicom_support import DicomError, load_dicom_zip_bytes

            try:
                volume = load_dicom_zip_bytes(volume_bytes)
            except DicomError as exc:
                raise MriError(str(exc)) from exc
        else:
            volume = load_nifti_bytes(volume_bytes)
    except MriError:
        raise
    except Exception as exc:  # nibabel raises many concrete types
        raise MriError(f"Could not read the file as a NIfTI volume: {exc}") from exc
    try:
        slices = extract_slices(volume)
    except ValueError as exc:
        raise MriError(str(exc)) from exc

    x = _to_model_input(slices)
    run = _get_session().run
    out_a = run(None, {"input": x})
    out_b = run(None, {"input": x[:, :, :, ::-1].copy()})

    def _softmax(logits: np.ndarray) -> np.ndarray:
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # Average PROBABILITIES across the TTA pair — the same aggregation
    # session_eval used during validation. Summing logits first would sharpen
    # every distribution and show confidences the metrics never measured.
    slice_probs = (_softmax(out_a[0]) + _softmax(out_b[0])) / 2
    probs = slice_probs.mean(axis=0)  # mean over slices

    names = class_names()
    if len(names) != probs.shape[0]:
        raise MriUnavailable(
            f"artifact/metrics mismatch: model outputs {probs.shape[0]} classes "
            f"but mri_metrics.json names {len(names)}"
        )
    top = int(probs.argmax())

    # CAM output exists on artifacts exported by ml/mri/export_onnx.py; older
    # logits-only artifacts still work, just without the visual explanation.
    slice_views = []
    if len(out_a) > 1:
        cam = (out_a[1] + out_b[1][:, :, :, ::-1]) / 2  # un-flip the TTA pass
        heat = np.maximum(cam[:, top], 0.0)  # ReLU, predicted class only
        peak = heat.max()
        if peak > 0:
            heat = heat / peak  # joint normalisation: hot slices stay hotter
        for i, sl in enumerate(slices):
            slice_views.append(
                {
                    "index": i,
                    "class_probability": round(float(slice_probs[i, top]), 4),
                    "image": _png_b64(sl),
                    "overlay": _png_b64(_overlay(sl, heat[i])),
                }
            )

    return {
        "available": True,
        "probabilities": {names[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "predicted_class": names[top],
        "predicted_index": top,
        "confidence": round(float(probs[top]), 4),
        "slices_analysed": len(slices),
        "volume_shape": list(volume.shape),
        "slices": slice_views,
        "explanation_note": (
            "Heatmaps show the regions that most influenced the model's decision "
            "(class activation mapping). They indicate model attention, not a "
            "clinical annotation of pathology."
        ),
        "note": (
            "Research prototype trained on OASIS-3. Reported separately from the "
            "behavioural risk score; not a diagnosis."
        ),
    }


@lru_cache(maxsize=1)
def model_metrics() -> dict | None:
    if not MRI_METRICS_PATH.exists():
        return None
    return json.loads(MRI_METRICS_PATH.read_text())
