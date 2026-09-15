"""DICOM ingestion: convert a zipped .dcm series into the pipeline's volume.

Patients don't have NIfTI files — scanning centres hand out CDs/downloads of
DICOM series. This module lets the API accept "the whole folder from your MRI
CD, zipped": bounded unzip → dicom2nifti conversion → the largest produced
volume → the same canonicalised loader the .nii path uses, so everything
downstream (slicing, model, CAM) is identical.

Bounds mirror the decompression discipline in preprocess._bounded_gunzip:
a zip may not expand beyond MAX_EXTRACTED_BYTES or MAX_MEMBER_COUNT.
"""

from __future__ import annotations

import tempfile
import zipfile
from pathlib import Path

import numpy as np

MAX_EXTRACTED_BYTES = 600 * 1024 * 1024
MAX_MEMBER_COUNT = 4000


class DicomError(Exception):
    """The zip could not be read as a convertible DICOM series."""


def _safe_extract(zf: zipfile.ZipFile, dest: Path) -> None:
    total = 0
    members = zf.infolist()
    if len(members) > MAX_MEMBER_COUNT:
        raise DicomError(f"zip contains too many files ({len(members)})")
    for m in members:
        total += m.file_size
        if total > MAX_EXTRACTED_BYTES:
            raise DicomError("zip expands beyond any plausible MRI series size")
        # zip-slip guard: resolved target must stay inside dest
        target = (dest / m.filename).resolve()
        if not str(target).startswith(str(dest.resolve())):
            raise DicomError("zip contains unsafe paths")
    zf.extractall(dest)


def load_dicom_zip_bytes(raw: bytes) -> np.ndarray:
    """Zipped DICOM series -> float32 canonical volume (via preprocess loader)."""
    import io

    import dicom2nifti

    from .preprocess import load_nifti_bytes  # package-relative (mri.preprocess)

    with tempfile.TemporaryDirectory(prefix="ng-dicom-") as tmp:
        tmp_path = Path(tmp)
        src = tmp_path / "src"
        out = tmp_path / "out"
        src.mkdir()
        out.mkdir()

        try:
            with zipfile.ZipFile(io.BytesIO(raw)) as zf:
                _safe_extract(zf, src)
        except zipfile.BadZipFile as exc:
            raise DicomError("the file is not a readable zip archive") from exc

        try:
            dicom2nifti.convert_directory(
                str(src), str(out), compression=True, reorient=True
            )
        except Exception as exc:
            raise DicomError(
                f"could not convert the DICOM series: {exc}"
            ) from exc

        produced = sorted(out.glob("*.nii*"))
        if not produced:
            raise DicomError(
                "no MRI series found in the zip — upload the folder your "
                "scanning centre gave you, zipped as-is"
            )

        # Several series on one CD (localizers, T2, T1): take the largest
        # volume by voxel count — in practice the structural T1.
        def voxels(p: Path) -> int:
            import nibabel as nib

            try:
                return int(np.prod(nib.load(str(p)).shape))
            except Exception:
                return 0

        best = max(produced, key=voxels)
        if voxels(best) == 0:
            raise DicomError("the converted series could not be read")
        return load_nifti_bytes(best.read_bytes())
