# syntax=docker/dockerfile:1

# ============================================================================
# Stage 1 — build the React app
# ============================================================================
FROM node:22-slim AS frontend

WORKDIR /build

# Copy manifests first so the dependency layer is cached across source edits.
COPY frontend/package.json frontend/package-lock.json* ./
RUN npm ci --no-audit --no-fund

COPY frontend/ ./
RUN npm run build


# ============================================================================
# Stage 2 — Python runtime, serving the API and the built frontend
# ============================================================================
FROM python:3.13-slim AS runtime

# base.en is the accuracy/size compromise for a free-tier container (~140MB,
# ~1GB RAM at inference). Drop to tiny.en if the host is memory constrained.
ARG WHISPER_MODEL=base.en
ARG SPACY_MODEL=en_core_web_sm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    WHISPER_MODEL=${WHISPER_MODEL} \
    NEUROGUARD_ML_DIR=/app/ml \
    HF_HOME=/opt/models

# ffmpeg is a safety net for browser audio: faster-whisper decodes webm/opus via
# PyAV, but codec edge cases across browsers are common enough that shipping a
# real decoder is cheaper than debugging a failed demo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt
RUN pip install --no-cache-dir -r backend/requirements.txt \
    && python -m spacy download ${SPACY_MODEL}

# Bake the Whisper weights into the image. Downloading them on first request
# would make the very first user wait minutes, and would fail entirely on a
# host without outbound network access.
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL}', device='cpu', compute_type='int8')"

# ml/ holds the shared feature code AND the trained artifact (~123KB), so the
# image is self-contained — no model download or training at runtime.
COPY ml/ ./ml/
COPY backend/ ./backend/
COPY --from=frontend /build/dist ./frontend/dist

# Drop privileges. The upload directory must be writable by this user.
RUN useradd --create-home --uid 10001 neuroguard \
    && mkdir -p /app/backend/data/uploads \
    && chown -R neuroguard:neuroguard /app /opt/models
USER neuroguard

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD curl -fsS http://127.0.0.1:8000/api/health || exit 1

WORKDIR /app/backend

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
