"""MRI pathway tests: preprocessing edge cases and the /api/mri endpoints.

Uses a small synthetic volume generated in-memory (no real OASIS data needed)
for every preprocessing test. The happy-path route test runs real ONNX
inference against the committed artifact and is skipped if the artifact is
absent (e.g. a fresh clone before training).
"""

from __future__ import annotations

import base64
import gzip
import io
import os
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Must be set before app modules import settings (same pattern as test_api.py).
_TMP_DB = Path(tempfile.mkdtemp()) / "test_mri.db"
os.environ["DATABASE_URL"] = f"sqlite:///{_TMP_DB}"
os.environ["ENVIRONMENT"] = "test"
os.environ["SECRET_KEY"] = "test-secret-key-not-for-production"

from fastapi.testclient import TestClient  # noqa: E402

from app.database import Base, engine  # noqa: E402
from app.main import app  # noqa: E402
from app.ml import mri_model  # noqa: E402
from mri import preprocess  # noqa: E402  (ml/ dir via app.ml.paths)

USER = {
    "email": "mri-tester@example.com",
    "password": "testpass123",
    "full_name": "Mri Tester",
    "age": 60,
    "education_years": 12,
}


def synthetic_nifti_bytes(
    shape=(64, 64, 64), gzipped: bool = True, dtype=np.float32
) -> bytes:
    """A brain-sized random volume as .nii(.gz) bytes."""
    import nibabel as nib

    rng = np.random.default_rng(0)
    data = (rng.random(shape) * 1000).astype(dtype)
    img = nib.Nifti1Image(data, affine=np.eye(4))
    raw = img.to_bytes()
    return gzip.compress(raw) if gzipped else raw


# --------------------------------------------------------------------------
# preprocessing unit tests
# --------------------------------------------------------------------------
def test_slice_indices_shape_and_bounds():
    idx = preprocess.slice_indices(256)
    assert len(idx) == preprocess.N_SLICES
    assert idx[0] == int(256 * preprocess.LO_FRAC)
    assert idx[-1] == int(256 * preprocess.HI_FRAC) - 1
    assert all(a <= b for a, b in zip(idx, idx[1:]))


def test_extract_slices_normal_volume():
    vol = np.random.default_rng(1).random((64, 64, 64)).astype(np.float32)
    slices = preprocess.extract_slices(vol)
    assert len(slices) == preprocess.N_SLICES
    assert all(s.dtype == np.uint8 for s in slices)
    assert slices[0].shape == (64, 64)  # transposed HxW of a 64x64 slice


def test_extract_slices_trailing_singleton_ok():
    vol = np.zeros((64, 64, 64, 1), dtype=np.float32)
    vol[8:56, 8:56, 8:56, 0] = 100.0
    assert len(preprocess.extract_slices(vol)) == preprocess.N_SLICES


def test_extract_slices_rejects_4d_series():
    vol = np.zeros((32, 32, 32, 5), dtype=np.float32)
    with pytest.raises(ValueError, match="4D"):
        preprocess.extract_slices(vol)


def test_extract_slices_rejects_thin_slab():
    vol = np.zeros((256, 256, 20), dtype=np.float32)
    with pytest.raises(ValueError, match="axial slices"):
        preprocess.extract_slices(vol)


def test_load_nifti_gz_and_plain_roundtrip():
    for gz in (True, False):
        data = preprocess.load_nifti_bytes(synthetic_nifti_bytes(gzipped=gz))
        assert data.shape == (64, 64, 64)
        assert data.dtype == np.float32


def test_load_nifti_int16_cast():
    data = preprocess.load_nifti_bytes(synthetic_nifti_bytes(dtype=np.int16))
    assert data.dtype == np.float32


def test_load_nifti_corrupt_bytes_raise():
    with pytest.raises(Exception):
        preprocess.load_nifti_bytes(b"definitely not a nifti file")


def test_decompression_bomb_capped(monkeypatch):
    # 1GB of zeros compresses to ~1MB; with the cap lowered it must abort
    # instead of materialising the payload.
    monkeypatch.setattr(preprocess, "MAX_DECOMPRESSED_BYTES", 32 * 1024 * 1024)
    bomb = gzip.compress(b"\x00" * (64 * 1024 * 1024))
    with pytest.raises(ValueError, match="expands"):
        preprocess.load_nifti_bytes(bomb)


def test_orientation_canonicalisation():
    """A volume stored with flipped axes must decode to the same canonical data."""
    import nibabel as nib

    rng = np.random.default_rng(2)
    data = (rng.random((48, 56, 64)) * 100).astype(np.float32)
    ras = nib.Nifti1Image(data, affine=np.eye(4))
    # Same anatomy stored LAS (x axis flipped): flip the array and the affine.
    las_affine = np.diag([-1.0, 1.0, 1.0, 1.0])
    las = nib.Nifti1Image(data[::-1].copy(), affine=las_affine)
    a = preprocess.load_nifti_bytes(ras.to_bytes())
    b = preprocess.load_nifti_bytes(las.to_bytes())
    assert np.allclose(a, b), "canonicalisation should undo the storage flip"


# --------------------------------------------------------------------------
# route tests
# --------------------------------------------------------------------------
@pytest.fixture(scope="module")
def client():
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c
    Base.metadata.drop_all(bind=engine)


@pytest.fixture(scope="module")
def auth_client(client):
    r = client.post("/api/auth/signup", json=USER)
    assert r.status_code == 201, r.text
    return client


ARTIFACT = mri_model.MRI_MODEL_PATH.exists()
needs_artifact = pytest.mark.skipif(not ARTIFACT, reason="mri_model.onnx not present")


def test_analyze_requires_auth(client):
    r = client.post(
        "/api/mri/analyze", files={"file": ("x.nii.gz", b"12345", "application/gzip")}
    )
    assert r.status_code == 401


def test_analyze_rejects_wrong_suffix(auth_client):
    r = auth_client.post(
        "/api/mri/analyze", files={"file": ("scan.pdf", b"12345", "application/pdf")}
    )
    assert r.status_code == 400


def test_analyze_rejects_empty_file(auth_client):
    r = auth_client.post(
        "/api/mri/analyze", files={"file": ("x.nii", b"", "application/octet-stream")}
    )
    assert r.status_code == 400


def test_analyze_rejects_oversize(auth_client, monkeypatch):
    from app.api.routes import mri as mri_route

    monkeypatch.setattr(mri_route, "MAX_MRI_MB", 1)
    big = b"\x00" * (2 * 1024 * 1024)
    r = auth_client.post(
        "/api/mri/analyze", files={"file": ("x.nii", big, "application/octet-stream")}
    )
    assert r.status_code == 413


def test_analyze_disabled_via_env(auth_client, monkeypatch):
    monkeypatch.setenv("MRI_ENABLED", "false")
    r = auth_client.post(
        "/api/mri/analyze", files={"file": ("x.nii.gz", b"123", "application/gzip")}
    )
    assert r.status_code == 503


def test_enabled_fails_closed_in_production(monkeypatch):
    monkeypatch.delenv("MRI_ENABLED", raising=False)
    monkeypatch.setenv("ENVIRONMENT", "production")
    assert mri_model.enabled() is False
    monkeypatch.setenv("MRI_ENABLED", "true")
    assert mri_model.enabled() is ARTIFACT


@needs_artifact
def test_analyze_corrupt_volume_is_422(auth_client):
    r = auth_client.post(
        "/api/mri/analyze",
        files={"file": ("x.nii.gz", gzip.compress(b"garbage"), "application/gzip")},
    )
    assert r.status_code == 422


@needs_artifact
def test_analyze_happy_path(auth_client):
    r = auth_client.post(
        "/api/mri/analyze",
        files={"file": ("brain.nii.gz", synthetic_nifti_bytes(), "application/gzip")},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    probs = list(body["probabilities"].values())
    assert abs(sum(probs) - 1.0) < 1e-3
    assert body["predicted_class"] in body["probabilities"]
    assert body["slices_analysed"] == preprocess.N_SLICES
    assert len(body["slices"]) == preprocess.N_SLICES
    # base64 images decode to PNGs
    png = base64.b64decode(body["slices"][0]["image"])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    overlay = base64.b64decode(body["slices"][0]["overlay"])
    assert overlay[:8] == b"\x89PNG\r\n\x1a\n"


@needs_artifact
def test_info_reports_cv_metrics(client):
    r = client.get("/api/mri/info")
    assert r.status_code == 200
    body = r.json()
    if body["available"]:
        assert body["classes"]
        assert "cv" in body
