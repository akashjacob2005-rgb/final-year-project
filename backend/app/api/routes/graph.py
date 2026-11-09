"""Graph analysis: upload rs-fMRI (+ DTI connectome), get Normal / MCI / Dementia.

Separate pathway from the assessment flow and from MRI — see
app/ml/graph_model.py for why it is never folded into the risk score.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, HTTPException, UploadFile, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...core.deps import get_current_user
from ...database import get_db
from ...ml import graph_model
from ...models import GraphResult, User

router = APIRouter(prefix="/graph", tags=["graph"])

MAX_FMRI_MB = 400   # a 4-D BOLD series is much larger than a T1
MAX_SC_MB = 20
FMRI_SUFFIXES = (".nii", ".nii.gz")
SC_SUFFIXES = (".csv", ".npy")


def _read_capped(file: UploadFile, limit_mb: int, what: str) -> bytes:
    # Stop at the cap: never buffer more than the limit, whatever is sent.
    limit = limit_mb * 1024 * 1024
    chunks, total = [], 0
    while True:
        chunk = file.file.read(1024 * 1024)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, f"{what} exceeds {limit_mb}MB")
        chunks.append(chunk)
    data = b"".join(chunks)
    if not data:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, f"The uploaded {what} was empty")
    return data


def _result_out(r: GraphResult) -> dict:
    return {"id": r.id, "created_at": r.created_at, "fmri_filename": r.fmri_filename,
            "probabilities": r.probabilities, "predicted_class": r.predicted_class,
            "top_edges": r.top_edges, "used_sc": r.used_sc, "model_version": r.model_version}


@router.get("/info")
def graph_info():
    """Model card data, or an honest 'not trained' — never placeholder metrics."""
    if not graph_model.enabled():
        return {"available": False, "enabled": False, "trained": graph_model.trained(),
                "message": "Graph analysis is not enabled on this deployment"}
    metrics = graph_model.model_metrics()
    if not graph_model.trained() or not metrics:
        return {"available": False, "enabled": True, "trained": False, "message": graph_model.NOT_TRAINED}
    return {
        "available": True, "enabled": True, "trained": True,
        "model": metrics.get("model"),
        "task": metrics.get("task"),
        "classes": metrics.get("classes"),
        "dataset": metrics.get("dataset"),
        "graph": metrics.get("graph"),
        "cv": metrics.get("cv"),
        "baseline": metrics.get("baseline"),
        "caveats": metrics.get("caveats", []),
        "version": metrics.get("version"),
    }


@router.post("/analyze")
def analyze(
    fmri: UploadFile = File(...),
    sc: UploadFile | None = File(None),
    user: User = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    if not graph_model.enabled():
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE,
                            "Graph analysis is not enabled on this deployment")

    fname = (fmri.filename or "").lower()
    if not fname.endswith(FMRI_SUFFIXES):
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                            "Upload the resting-state fMRI as a 4-D .nii or .nii.gz file")
    sc_suffix = None
    if sc is not None:
        sname = (sc.filename or "").lower()
        if not sname.endswith(SC_SUFFIXES):
            raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY,
                                "Upload the structural connectivity matrix as .csv or .npy")
        sc_suffix = ".npy" if sname.endswith(".npy") else ".csv"
    fmri_bytes = _read_capped(fmri, MAX_FMRI_MB, "fMRI file")
    sc_bytes = _read_capped(sc, MAX_SC_MB, "connectivity matrix") if sc is not None else None

    try:
        result = graph_model.analyze(fmri_bytes, sc_bytes, sc_suffix)
    except graph_model.GraphNotTrained as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except graph_model.GraphError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_ENTITY, str(exc)) from exc

    row = GraphResult(user_id=user.id, fmri_filename=fmri.filename, used_sc=result["used_sc"],
                      probabilities=result["probabilities"], predicted_class=result["predicted_class"],
                      top_edges=result["top_edges"], model_version=result["model_version"])
    db.add(row)
    db.commit()
    db.refresh(row)
    result["id"] = row.id
    return result


@router.get("/results")
def list_results(user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    rows = db.scalars(select(GraphResult).where(GraphResult.user_id == user.id)
                      .order_by(GraphResult.created_at.desc())).all()
    return [_result_out(r) for r in rows]


@router.get("/results/{result_id}")
def get_result(result_id: int, user: User = Depends(get_current_user), db: Session = Depends(get_db)):
    row = db.get(GraphResult, result_id)
    # Same 404 whether the row is missing or someone else's: don't leak existence.
    if row is None or row.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Result not found")
    return _result_out(row)
