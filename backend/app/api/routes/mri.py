"""MRI analysis: upload a T1w NIfTI volume, get the CDR-based prediction.

Separate pathway from the assessment flow — see app/ml/mri_model.py for why
the MRI result is never folded into the behavioural risk score.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status

from ...core.deps import get_current_user
from ...ml import mri_model
from ...models import User

router = APIRouter(prefix="/mri", tags=["mri"])

MAX_MRI_MB = 80
ALLOWED_SUFFIXES = (".nii", ".nii.gz")


@router.get("/info")
def mri_info():
    """Public model card data for the MRI pathway (mirrors /history/model-info)."""
    metrics = mri_model.model_metrics()
    if not metrics or not mri_model.enabled():
        return {"available": False, "message": "MRI model not available on this deployment"}
    return {
        "available": True,
        "model": metrics.get("model"),
        "task": metrics.get("task"),
        "label_mode": metrics.get("label_mode"),
        "classes": metrics.get("classes"),
        "dataset": metrics.get("dataset"),
        "aggregation": metrics.get("aggregation"),
        "cv": metrics.get("cv"),
        # kept for older artifacts trained before cross-validation existed
        "test": metrics.get("test"),
        "validation": metrics.get("validation"),
        "caveats": metrics.get("caveats", []),
    }


@router.post("/analyze")
def analyze(
    file: UploadFile = File(...),
    user: User = Depends(get_current_user),
):
    if not mri_model.enabled():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "MRI analysis is not enabled on this deployment",
        )

    name = (file.filename or "").lower()
    if not name.endswith(ALLOWED_SUFFIXES):
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            "Please upload a T1-weighted MRI volume as .nii or .nii.gz",
        )

    # Read in chunks and stop at the cap: never buffer more than the limit,
    # regardless of what the client claims or sends.
    limit = MAX_MRI_MB * 1024 * 1024
    chunks, total = [], 0
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"Volume exceeds {MAX_MRI_MB}MB",
            )
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "The uploaded file was empty")

    try:
        return mri_model.predict(data)
    except mri_model.MriError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except mri_model.MriUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
