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
MRI_ENSEMBLE_METRICS_PATH = paths.ARTIFACTS_DIR / "mri_ensemble_metrics.json"

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
_HALF_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
_HALF_STD = np.array([0.5, 0.5, 0.5], dtype=np.float32)

# Ensemble member registry. The ResNet is mandatory (it also carries the CAM
# output the heatmaps come from); transformer members load only when their
# ONNX exists next to it — FAIL-SOFT, so a deployment that only ships the
# ResNet artifact (e.g. the memory-constrained free tier) behaves exactly as
# before. Normalisation must match each member's pretraining (see ARCHS in
# ml/mri/train_mri.py).
_MEMBERS = [
    ("resnet18", "mri_model.onnx", IMAGENET_MEAN, IMAGENET_STD),
    ("dinov2_s", "mri_model_dinov2_s.onnx", IMAGENET_MEAN, IMAGENET_STD),
    ("vit_b16", "mri_model_vit_b16.onnx", _HALF_MEAN, _HALF_STD),
]

_FALLBACK_CLASSES = [
    "No impairment signs (CDR 0)",
    "Signs of impairment (CDR >= 0.5)",
]

_lock = threading.Lock()
_sessions: dict[str, object] = {}


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


def _model_dir():
    """Directory holding the member ONNX files: the artifacts dir, or a
    candidate bundle when promote.py / external_validate.py --model point
    MRI_MODEL_PATH elsewhere."""
    return MRI_MODEL_PATH.parent


def member_files_present() -> list[str]:
    """Arch names whose ONNX artifact exists — cheap, no model load."""
    return [arch for arch, fname, _, _ in _MEMBERS
            if (_model_dir() / fname).exists()]


def _get_sessions() -> list[tuple[str, object, np.ndarray, np.ndarray]]:
    """Loaded ensemble members as (arch, session, mean, std).

    The ResNet must load or the pathway is unavailable; a transformer member
    that is missing or corrupt is skipped with the rest of the ensemble intact.
    """
    with _lock:
        if not _sessions:
            import onnxruntime as ort

            for arch, fname, _, _ in _MEMBERS:
                path = _model_dir() / fname
                if not path.exists():
                    if arch == "resnet18":
                        raise MriUnavailable("MRI model artifact is missing")
                    continue
                try:
                    _sessions[arch] = ort.InferenceSession(
                        str(path), providers=["CPUExecutionProvider"]
                    )
                except Exception as exc:  # missing .onnx.data sidecar, corrupt file
                    if arch == "resnet18":
                        raise MriUnavailable(
                            f"MRI model artifact failed to load: {exc}"
                        ) from exc
                    # fail-soft: serve without this optional member
    return [(arch, _sessions[arch], mean, std)
            for arch, _, mean, std in _MEMBERS if arch in _sessions]


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


def _to_raw_batch(slices: list[np.ndarray]) -> np.ndarray:
    """uint8 slices -> un-normalised [0,1] NCHW batch (resize + grey->RGB)."""
    from PIL import Image

    batch = []
    for sl in slices:
        img = Image.fromarray(sl, mode="L").resize(
            (IMG_SIZE, IMG_SIZE), Image.Resampling.BILINEAR
        )
        x = np.asarray(img, dtype=np.float32) / 255.0          # HW
        batch.append(np.repeat(x[None, ...], 3, axis=0))       # CHW, grey->RGB
    return np.stack(batch)


def _to_model_input(slices: list[np.ndarray],
                    mean: np.ndarray = IMAGENET_MEAN,
                    std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """uint8 slices -> normalised NCHW batch, mirroring the training transforms."""
    raw = _to_raw_batch(slices)
    return ((raw - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)


def _softmax(logits: np.ndarray) -> np.ndarray:
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def preprocessing() -> dict:
    """How the deployed artifact's training slices were made. Travels with
    the artifact (mri_metrics.json "preprocessing", written by train_mri.py)
    so inference can never silently use a different pipeline. Older
    artifacts have no block -> the original percentage-slice pipeline."""
    prep = (model_metrics() or {}).get("preprocessing") or {}
    return {"mode": prep.get("mode", "percentile"),
            "planes": tuple(prep.get("planes", ("axial",)))}


def _registered(raw: bytes | None, volume: np.ndarray, planes: tuple[str, ...]) -> list[np.ndarray]:
    try:
        from mri.preprocess import registered_slices
        return registered_slices(raw, volume, planes)
    except ImportError as exc:
        raise MriUnavailable(
            "this model needs brain-template registration (ANTsPy + skull-stripping), "
            f"which is not installed on this deployment: {exc}"
        ) from exc
    except ValueError:
        raise
    except Exception as exc:  # registration failed on this scan
        raise MriError(f"Could not align the scan to the brain template: {exc}") from exc


def calibration(members: list[str] | None = None) -> dict | None:
    """Optional {temperature, threshold} fitted by compare_experiments.py on
    out-of-fold predictions and installed by ml/mri/promote.py.

    Temperature rescales the session probability so displayed percentages
    match observed frequencies; threshold is the operating point that
    balances normal and impaired recall (argmax at 0.5 favoured 'impaired'
    because training is 2:1 impaired). A calibration is only valid for the
    exact member set it was fitted on: the ensemble's lives in
    mri_ensemble_metrics.json, the single ResNet's in mri_metrics.json.
    Absent or mismatched -> None -> plain argmax, as before.
    """
    if members is None:
        members = member_files_present()
    source = ensemble_metrics() if len(members) > 1 else model_metrics()
    cal = (source or {}).get("calibration")
    if not cal or "temperature" not in cal or "threshold" not in cal:
        return None
    if sorted(cal.get("members", ["resnet18"])) != sorted(members):
        return None
    return {"temperature": float(cal["temperature"]), "threshold": float(cal["threshold"])}


def score_slices(slices: list[np.ndarray]) -> dict:
    """Run every loaded member on preprocessed slices.

    Returns hflip-TTA slice probabilities averaged over members, the session
    probabilities (calibrated when a matching calibration block exists), the
    decision, and the ResNet's raw outputs for CAM. Shared by predict() and
    ml/mri/external_validate.py so the API and the external test can never
    disagree.
    """
    # Average PROBABILITIES across the TTA pair and across ensemble members —
    # the same aggregation session_eval / ensemble_eval used during validation.
    # Summing logits first would sharpen every distribution and show
    # confidences the metrics never measured.
    raw = _to_raw_batch(slices)
    members = _get_sessions()
    slice_probs = None
    resnet_out = None  # the (a, b) TTA outputs of the CAM-carrying member
    for arch, sess, mean, std in members:
        x = ((raw - mean[None, :, None, None]) / std[None, :, None, None]).astype(np.float32)
        out_a = sess.run(None, {"input": x})
        out_b = sess.run(None, {"input": x[:, :, :, ::-1].copy()})
        member_probs = (_softmax(out_a[0]) + _softmax(out_b[0])) / 2
        slice_probs = member_probs if slice_probs is None else slice_probs + member_probs
        if arch == "resnet18":
            resnet_out = (out_a, out_b)
    slice_probs = slice_probs / len(members)
    raw_probs = slice_probs.mean(axis=0)  # mean over slices
    probs, top = raw_probs, int(raw_probs.argmax())

    member_names = [arch for arch, _, _, _ in members]
    cal = calibration(member_names)
    if cal is not None and raw_probs.shape[0] == 2:
        # Same monotone transform compare_experiments.py fitted on OOF session
        # probabilities, so the ranking (AUC) is untouched.
        p1 = float(np.clip(raw_probs[1], 1e-7, 1 - 1e-7))
        p1 = float(1.0 / (1.0 + np.exp(-np.log(p1 / (1 - p1)) / cal["temperature"])))
        probs = np.array([1.0 - p1, p1])
        top = int(p1 >= cal["threshold"])
    return {"probs": probs, "raw_probs": raw_probs, "top": top, "slice_probs": slice_probs,
            "resnet_out": resnet_out, "members": member_names, "calibration": cal}


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
        extract_slices(volume)  # shape/thin-slab validation for every mode
        prep = preprocessing()
        if prep["mode"] == "registered":
            slices = _registered(volume_bytes if not is_dicom_zip else None, volume, prep["planes"])
        else:
            slices = extract_slices(volume)
    except ValueError as exc:
        raise MriError(str(exc)) from exc

    scored = score_slices(slices)
    probs, top, slice_probs = scored["probs"], scored["top"], scored["slice_probs"]
    resnet_out, member_names = scored["resnet_out"], scored["members"]

    names = class_names()
    if len(names) != probs.shape[0]:
        raise MriUnavailable(
            f"artifact/metrics mismatch: model outputs {probs.shape[0]} classes "
            f"but mri_metrics.json names {len(names)}"
        )

    # CAM output exists on artifacts exported by ml/mri/export_onnx.py; older
    # logits-only artifacts still work, just without the visual explanation.
    # In ensemble mode the heatmap comes from the ResNet member — exact CAM
    # only exists for the GAP+linear CNN head, not for the transformers.
    slice_views = []
    if resnet_out is not None and len(resnet_out[0]) > 1:
        out_a, out_b = resnet_out
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

    explanation = (
        "Heatmaps show the regions that most influenced the model's decision "
        "(class activation mapping). They indicate model attention, not a "
        "clinical annotation of pathology."
    )
    if len(member_names) > 1:
        explanation += (
            " The heatmap comes from the CNN member of the ensemble; the "
            "verdict averages all members."
        )

    return {
        "available": True,
        "probabilities": {names[i]: round(float(p), 4) for i, p in enumerate(probs)},
        "predicted_class": names[top],
        "predicted_index": top,
        "confidence": round(float(probs[top]), 4),
        "calibrated": scored["calibration"] is not None,
        "decision_threshold": (round(scored["calibration"]["threshold"], 4)
                               if scored["calibration"] else 0.5),
        "slices_analysed": len(slices),
        "volume_shape": list(volume.shape),
        "model_members": member_names,
        "slices": slice_views,
        "explanation_note": explanation,
        "note": (
            "Research prototype trained on OASIS-3. Reported separately from the "
            "behavioural risk score; not a diagnosis."
        ),
    }


def reset() -> None:
    """Drop cached ONNX sessions and metrics (after promote.py swaps the
    artifacts, or when external_validate.py --model points at a candidate)."""
    with _lock:
        _sessions.clear()
    model_metrics.cache_clear()
    ensemble_metrics.cache_clear()


@lru_cache(maxsize=1)
def model_metrics() -> dict | None:
    if not MRI_METRICS_PATH.exists():
        return None
    return json.loads(MRI_METRICS_PATH.read_text())


@lru_cache(maxsize=1)
def ensemble_metrics() -> dict | None:
    """CV metrics of the deployed ensemble (docs/ARCH_COMPARISON.md study)."""
    if not MRI_ENSEMBLE_METRICS_PATH.exists():
        return None
    return json.loads(MRI_ENSEMBLE_METRICS_PATH.read_text())
