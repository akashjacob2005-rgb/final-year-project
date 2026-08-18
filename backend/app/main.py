"""NeuroGuard API entrypoint."""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api.routes import assessment, auth, history
from .config import settings
from .database import Base, engine
from .ml import language_model, speech

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("neuroguard")


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Alembic owns the schema in production; create_all keeps first-run local
    # setup to a single command.
    Base.metadata.create_all(bind=engine)

    # Load the model and Whisper now so the first user request is not slow.
    language_model.warm_up()
    if settings.environment != "test":
        speech.warm_up()
    log.info("NeuroGuard started (env=%s)", settings.environment)
    yield


app = FastAPI(
    title="NeuroGuard API",
    description=(
        "Early cognitive decline screening. Research and educational use only — "
        "not a diagnostic device."
    ),
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origin_list,
    allow_credentials=True,  # required for the httpOnly auth cookies
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(auth.router, prefix="/api")
app.include_router(assessment.router, prefix="/api")
app.include_router(history.router, prefix="/api")


@app.get("/api/health", tags=["meta"])
def health():
    return {
        "status": "ok",
        "service": "NeuroGuard",
        "environment": settings.environment,
        "model_loaded": language_model.paths.LANGUAGE_MODEL_PATH.exists(),
    }


# --------------------------------------------------------------------------
# Serve the built React app from the same origin in production, so there is one
# URL, no CORS in play, and the auth cookie is first-party.
# --------------------------------------------------------------------------
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend" / "dist"

if FRONTEND_DIST.is_dir():
    app.mount(
        "/assets", StaticFiles(directory=FRONTEND_DIST / "assets"), name="assets"
    )

    @app.get("/{full_path:path}", include_in_schema=False)
    def spa(full_path: str):
        """Serve index.html for client-side routes so a hard refresh on
        /dashboard does not 404."""
        candidate = FRONTEND_DIST / full_path
        if full_path and candidate.is_file():
            return FileResponse(candidate)
        return FileResponse(FRONTEND_DIST / "index.html")
