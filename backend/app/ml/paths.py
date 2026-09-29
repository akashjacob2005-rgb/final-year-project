"""Locate the shared ml/ package and put it on the import path.

Training and inference MUST use the same feature-extraction code. Rather than
copying ml/features.py into the backend (where the two would silently drift),
we add the real directory to sys.path and import from it.

Override with NEUROGUARD_ML_DIR when the layout differs, e.g. in Docker.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_env = os.getenv("NEUROGUARD_ML_DIR")
ML_DIR = Path(_env) if _env else Path(__file__).resolve().parents[3] / "ml"
ARTIFACTS_DIR = ML_DIR / "artifacts"
LANGUAGE_MODEL_PATH = ARTIFACTS_DIR / "language_model.joblib"
METRICS_PATH = ARTIFACTS_DIR / "metrics.json"

# The audio-trained model (ml/train_audio_model.py). Scores the picture-description
# recording using ASR text AND speech timing, so it only applies to requests that
# carry audio. The text-only artifact above stays the fallback for typed
# transcripts, which have no timing to give it.
AUDIO_MODEL_PATH = ARTIFACTS_DIR / "audio_model.joblib"
AUDIO_METRICS_PATH = ARTIFACTS_DIR / "metrics_audio.json"

if str(ML_DIR) not in sys.path:
    sys.path.insert(0, str(ML_DIR))
