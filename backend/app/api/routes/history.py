"""Longitudinal history and dashboard summary.

Tracking change over time is the point of repeat testing: a single score is a
weak signal, whereas a consistent downward trend in the same person is a much
stronger one.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import desc, select
from sqlalchemy.orm import Session

from ...core.deps import get_current_user
from ...database import get_db
from ...ml import language_model
from ...models import AssessmentSession, TestResult, User

router = APIRouter(prefix="/history", tags=["history"])


@router.get("/sessions")
def list_sessions(
    limit: int = 50,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    rows = db.scalars(
        select(AssessmentSession)
        .where(
            AssessmentSession.user_id == user.id,
            AssessmentSession.status == "completed",
        )
        .order_by(desc(AssessmentSession.completed_at))
        .limit(min(limit, 200))
    ).all()

    return [
        {
            "session_id": s.id,
            "completed_at": s.completed_at,
            "risk_percent": round((s.overall_risk or 0) * 100, 1),
            "band": s.risk_band,
            "confidence": s.confidence,
            "domain_z": (s.result_json or {}).get("domain_z", {}),
        }
        for s in rows
    ]


@router.get("/trend")
def trend(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    """Oldest-first series for the monitoring chart."""
    rows = db.scalars(
        select(AssessmentSession)
        .where(
            AssessmentSession.user_id == user.id,
            AssessmentSession.status == "completed",
        )
        .order_by(AssessmentSession.completed_at)
    ).all()

    points = [
        {
            "session_id": s.id,
            "completed_at": s.completed_at,
            "risk_percent": round((s.overall_risk or 0) * 100, 1),
            "band": s.risk_band,
            "composite_z": s.composite_z,
            "language_probability": s.language_probability,
            "domain_z": (s.result_json or {}).get("domain_z", {}),
        }
        for s in rows
    ]

    direction, change = "insufficient_data", None
    if len(points) >= 2:
        change = points[-1]["risk_percent"] - points[0]["risk_percent"]
        # A few points of movement is noise, not a trend worth flagging.
        if change > 10:
            direction = "worsening"
        elif change < -10:
            direction = "improving"
        else:
            direction = "stable"

    return {
        "points": points,
        "count": len(points),
        "direction": direction,
        "change_since_first": round(change, 1) if change is not None else None,
    }


@router.get("/summary")
def summary(
    db: Session = Depends(get_db), user: User = Depends(get_current_user)
):
    sessions = db.scalars(
        select(AssessmentSession)
        .where(
            AssessmentSession.user_id == user.id,
            AssessmentSession.status == "completed",
        )
        .order_by(desc(AssessmentSession.completed_at))
    ).all()

    latest = sessions[0] if sessions else None
    domain_avg: dict[str, list[float]] = {}
    if latest:
        for d, z in (latest.result_json or {}).get("domain_z", {}).items():
            domain_avg.setdefault(d, []).append(z)

    return {
        "total_assessments": len(sessions),
        "latest": (
            {
                "session_id": latest.id,
                "completed_at": latest.completed_at,
                "risk_percent": round((latest.overall_risk or 0) * 100, 1),
                "band": latest.risk_band,
                "confidence": latest.confidence,
                "domain_z": (latest.result_json or {}).get("domain_z", {}),
                "weakest_domain": (latest.result_json or {}).get("weakest_domain"),
            }
            if latest
            else None
        ),
        "user": {
            "full_name": user.full_name,
            "age": user.age,
            "education_years": user.education_years,
        },
    }


@router.delete("/sessions/{session_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_session(
    session_id: int,
    db: Session = Depends(get_db),
    user: User = Depends(get_current_user),
):
    s = db.get(AssessmentSession, session_id)
    if not s or s.user_id != user.id:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Assessment session not found")
    db.delete(s)
    db.commit()


@router.get("/model-info")
def model_info():
    """Expose the model's real evaluation metrics.

    Deliberately public and linked from the results page: a screening tool that
    hides how well it actually performs is not one anybody should trust.
    """
    metrics = language_model.model_metrics()
    if not metrics:
        return {"available": False, "message": "Model has not been trained yet"}

    return {
        "available": True,
        "deployed_model": metrics.get("deployed_model"),
        "deployed_because": metrics.get("deployed_because"),
        "evaluation": metrics.get("evaluation"),
        "dataset": metrics.get("dataset"),
        "caveats": metrics.get("caveats", []),
        "models": [
            {
                "name": m["name"],
                "roc_auc": round(m["roc_auc_mean"], 3),
                "roc_auc_ci95": [round(v, 3) for v in m["roc_auc_ci95"]],
                "accuracy": round(m["accuracy_mean"], 3),
                "sensitivity": round(m["sensitivity_mean"], 3),
                "specificity": round(m["specificity_mean"], 3),
            }
            for m in metrics.get("all_models", [])
        ],
    }
