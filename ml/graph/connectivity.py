"""MNI-space rs-fMRI -> cleaned region time series -> functional connectivity.

Shared by training (ml/graph/preprocess.py) and inference
(backend/app/ml/graph_model.py) so the two can never drift apart. The input
contract is the same in both: a 4-D BOLD series already motion-corrected and
in MNI152 space (preprocess.py does both for OASIS-3; for uploads the
expected source is an fMRIPrep `space-MNI152...` output).

Cleaning (nilearn.signal.clean via NiftiLabelsMasker):
    linear detrend -> band-pass 0.01-0.1 Hz -> z-score per region,
    with the mean white-matter and CSF signals regressed out as confounds
    (template tissue masks, so the same confounds exist at inference).
"""

from __future__ import annotations

import numpy as np

LOW_PASS = 0.1
HIGH_PASS = 0.01
DEFAULT_TR = 2.2         # OASIS-3 resting-state BOLD TR; the NIfTI header wins when set
MIN_TIMEPOINTS = 40      # fewer volumes cannot support a 0.01 Hz high-pass or a stable r
MAX_DECOMPRESSED_BYTES = 1500 * 1024 * 1024  # a 4-D BOLD is far bigger than a T1


class BoldError(ValueError):
    """The upload is not a usable 4-D BOLD series (the client's fault)."""


def load_bold_bytes(raw: bytes):
    """Decode an uploaded .nii/.nii.gz into a nibabel 4-D image (bounded gunzip)."""
    import gzip
    import io

    import nibabel as nib

    if raw[:2] == b"\x1f\x8b":
        out = io.BytesIO()
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as gz:
            while True:
                chunk = gz.read(8 * 1024 * 1024)
                if not chunk:
                    break
                out.write(chunk)
                if out.tell() > MAX_DECOMPRESSED_BYTES:
                    raise BoldError("compressed file expands beyond any plausible BOLD series")
        raw = out.getvalue()
    try:
        holder = nib.FileHolder(fileobj=io.BytesIO(raw))
        img = nib.Nifti1Image.from_file_map({"header": holder, "image": holder})
        img = nib.Nifti1Image(np.asanyarray(img.dataobj), img.affine, img.header)
    except Exception as exc:  # nibabel raises many concrete types
        raise BoldError(f"Could not read the file as a NIfTI volume: {exc}") from exc
    return check_bold(img)


def check_bold(img):
    shape = img.shape
    if len(shape) != 4:
        raise BoldError(
            f"expected a 4-D resting-state BOLD series (x, y, z, time), got shape {tuple(shape)}"
            " — a 3-D structural scan belongs on the MRI page"
        )
    if shape[3] < MIN_TIMEPOINTS:
        raise BoldError(f"only {shape[3]} time points; at least {MIN_TIMEPOINTS} are needed")
    return img


def repetition_time(img) -> float:
    zooms = img.header.get_zooms()
    tr = float(zooms[3]) if len(zooms) > 3 else 0.0
    # Headers sometimes store ms, or 0/1 placeholders.
    if tr > 100:
        tr /= 1000.0
    return tr if 0.3 <= tr <= 10 else DEFAULT_TR


def tissue_confounds(img) -> np.ndarray:
    """Mean white-matter and CSF signal (T x 2) from MNI template masks."""
    from nilearn import datasets, image, masking

    wm = datasets.load_mni152_wm_mask()
    gm = datasets.load_mni152_gm_mask()
    brain = datasets.load_mni152_brain_mask()
    ref = image.index_img(img, 0)
    wm = image.resample_to_img(wm, ref, interpolation="nearest")
    gm = image.resample_to_img(gm, ref, interpolation="nearest")
    brain = image.resample_to_img(brain, ref, interpolation="nearest")
    csf_data = (brain.get_fdata() > 0) & (gm.get_fdata() == 0) & (wm.get_fdata() == 0)
    csf = image.new_img_like(ref, csf_data.astype(np.int8))
    cols = []
    for mask in (wm, csf):
        if mask.get_fdata().sum() < 10:
            continue  # field of view misses the tissue: skip rather than regress noise
        cols.append(masking.apply_mask(img, mask).mean(axis=1))
    return np.column_stack(cols) if cols else np.zeros((img.shape[3], 0))


def region_timeseries(img, labels_img, confounds: np.ndarray | None = None,
                      use_tissue_confounds: bool = True) -> np.ndarray:
    """(T x N) cleaned region time series; N = labels in the atlas."""
    from nilearn.maskers import NiftiLabelsMasker

    conf = tissue_confounds(img) if use_tissue_confounds else None
    if confounds is not None:
        conf = confounds if conf is None else np.column_stack([conf, confounds])
    masker = NiftiLabelsMasker(
        labels_img=labels_img, detrend=True, standardize="zscore_sample",
        low_pass=LOW_PASS, high_pass=HIGH_PASS, t_r=repetition_time(img),
        keep_masked_labels=True,
    )
    import warnings

    with warnings.catch_warnings():
        # keep_masked_labels=True keeps one column per atlas region even when a
        # region is outside the field of view (node i must always mean region
        # i); nilearn warns that the option will change in 0.15.
        warnings.simplefilter("ignore", FutureWarning)
        ts = masker.fit_transform(img, confounds=conf if conf is not None and conf.size else None)
    return np.nan_to_num(np.asarray(ts, dtype=np.float64))


def functional_connectivity(ts: np.ndarray) -> np.ndarray:
    """N x N Pearson correlation (nilearn ConnectivityMeasure), diagonal = 1.

    Regions that fall outside the field of view have a flat series; their
    correlations are 0, not NaN, so the graph stays finite.
    """
    from nilearn.connectome import ConnectivityMeasure

    fc = ConnectivityMeasure(kind="correlation", standardize="zscore_sample").fit_transform([ts])[0]
    fc = np.nan_to_num(fc)
    np.fill_diagonal(fc, 1.0)
    return (fc + fc.T) / 2


def fc_from_bold(img, labels_img, confounds: np.ndarray | None = None,
                 use_tissue_confounds: bool = True) -> np.ndarray:
    return functional_connectivity(region_timeseries(img, labels_img, confounds, use_tissue_confounds))
