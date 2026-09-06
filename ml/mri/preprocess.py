"""NIfTI -> axial slice extraction, shared by training export and inference.

Current export (v2): N_SLICES axial slices spanning LO_FRAC..HI_FRAC of the
canonical (RAS) volume, each transposed then flipped vertically
(``slice.T[::-1]``) and intensity-normalised to its own 1st..99th percentile.
Volumes are reoriented to RAS with ``nib.as_closest_canonical`` before slicing,
so files stored in any axis order produce anatomically axial slices. (OASIS-3
files are already RAS, so this is a no-op for the training data.)

Historical note: the v1 export (8 slices over 40-60%, no canonicalisation) was
reverse-engineered from the original PNGs to <0.5/255 mean pixel error; v2
kept the same per-slice transform and widened the coverage.

Inference MUST reuse this module so uploaded volumes are processed exactly as
the training data was. Training reads the pre-exported PNGs directly.
"""

from __future__ import annotations

import io

import numpy as np

# v2: 16 slices over 35-65% of the volume (the original export used 8 over
# 40-60%). Wider coverage reaches more temporal-lobe anatomy; training and
# inference share these constants so they can never drift apart.
N_SLICES = 16
LO_FRAC = 0.35
HI_FRAC = 0.65

# A volume must have a reasonable number of real axial slices for the sampled
# indices to be distinct anatomy rather than near-duplicates (thin-slab
# clinical acquisitions can have ~20 slices; 35-65% of that is 6 indices
# stretched to 16).
MIN_AXIAL_SLICES = 2 * N_SLICES

# Decompression guard: an 80MB upload can legally expand ~1000x under gzip.
# Anything bigger than this decompressed is not a plausible single T1w volume.
MAX_DECOMPRESSED_BYTES = 600 * 1024 * 1024


def slice_indices(n: int) -> np.ndarray:
    lo = int(n * LO_FRAC)
    hi = int(n * HI_FRAC) - 1
    return np.linspace(lo, hi, N_SLICES).astype(int)


def extract_slices(volume: np.ndarray) -> list[np.ndarray]:
    """Return the N_SLICES normalised uint8 axial slices of a 3D volume."""
    if volume.ndim == 4 and volume.shape[3] == 1:
        volume = volume[..., 0]  # trailing singleton (common in exports)
    if volume.ndim != 3:
        raise ValueError(
            f"expected a single 3D structural volume, got shape {tuple(volume.shape)} — "
            "4D series (fMRI/ASL) are not T1w scans this model can read"
        )
    if volume.shape[2] < MIN_AXIAL_SLICES:
        raise ValueError(
            f"volume has only {volume.shape[2]} axial slices; at least "
            f"{MIN_AXIAL_SLICES} are needed for a reliable analysis"
        )
    out = []
    for idx in slice_indices(volume.shape[2]):
        sl = np.take(volume, int(idx), axis=2).astype(np.float32).T[::-1]
        lo, hi = np.percentile(sl, 1), np.percentile(sl, 99)
        sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
        out.append(sl.astype(np.uint8))
    return out


def _bounded_gunzip(raw: bytes) -> bytes:
    """Decompress with a hard output cap so a crafted file can't exhaust RAM."""
    import gzip

    out = io.BytesIO()
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
        while True:
            chunk = gz.read(8 * 1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
            if out.tell() > MAX_DECOMPRESSED_BYTES:
                raise ValueError(
                    "compressed file expands beyond the size of any plausible "
                    "single MRI volume"
                )
    return out.getvalue()


def load_nifti_bytes(raw: bytes) -> np.ndarray:
    """Decode an uploaded .nii or .nii.gz file into a canonical float32 array.

    - Bounded decompression (decompression-bomb guard).
    - Reoriented to RAS so slicing is anatomically axial for any source.
    - RGB/RGBA volumes (structured "void" records numpy cannot cast directly)
      are converted to luminance.
    """
    import nibabel as nib

    if raw[:2] == b"\x1f\x8b":
        raw = _bounded_gunzip(raw)
    holder = nib.FileHolder(fileobj=io.BytesIO(raw))
    img = nib.Nifti1Image.from_file_map({"header": holder, "image": holder})
    try:
        img = nib.as_closest_canonical(img)
    except Exception:
        pass  # malformed affine: fall back to stored orientation

    data = np.asanyarray(img.dataobj)
    if data.dtype.kind == "V":  # RGB(A) structured voxels -> mean of channels
        data = np.stack(
            [data[name].astype(np.float32) for name in data.dtype.names], axis=-1
        ).mean(axis=-1)
    return np.ascontiguousarray(data, dtype=np.float32)
