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
ALLOWED_SUFFIXES = (".nii", ".nii.gz", ".zip")


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
        "samples_available": mri_model.samples_available(),
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
            "Please upload an MRI as .nii / .nii.gz, or your scan folder "
            "(DICOM) zipped as a .zip",
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

    is_dicom_zip = name.endswith(".zip")
    if is_dicom_zip and data[:4] != b"PK\x03\x04":
        raise HTTPException(
            status.HTTP_422_UNPROCESSABLE_ENTITY,
            "The file has a .zip name but is not a readable zip archive",
        )

    try:
        return mri_model.predict(data, is_dicom_zip=is_dicom_zip)
    except mri_model.MriError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except mri_model.MriUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc


@router.post("/analyze-sample")
def analyze_sample(
    case: str = "impaired",
    user: User = Depends(get_current_user),
):
    """Run the analysis on a bundled anonymized research scan — lets anyone
    experience the feature without owning an MRI file."""
    if not mri_model.enabled():
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "MRI analysis is not enabled on this deployment",
        )
    if case not in ("normal", "impaired"):
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "case must be normal or impaired")
    try:
        result = mri_model.predict(mri_model.sample_bytes(case))
    except mri_model.MriError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc
    except mri_model.MriUnavailable as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    result["sample"] = case
    return result
