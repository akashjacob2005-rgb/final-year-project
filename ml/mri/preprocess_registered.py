"""E1: registered + skull-stripped slice extraction (experiment pipeline).

The deployed preprocessing slices raw volumes at fixed *percentages* of the
array, so "slice 8" is different anatomy in different heads. This pipeline
makes slice indices anatomical:

    N4 bias correction (ANTsPy)
      -> skull-strip (SynthStrip via Docker: robust across 1.5T/3T)
      -> 12-dof affine registration to the MNI152 1mm template (ANTsPy)
      -> slices at FIXED MNI z-coordinates (default 16 axial, z = -30..+45mm,
         hippocampus level up to the ventricles)

Outputs the same slices/<session>/<session>_slNN.png layout the training
manifest expects. Offline/experiment tooling only — the serving backend keeps
the deployed (unregistered) pipeline; see docs/ARCH_COMPARISON.md.

Pilot (5 volumes, timed, writes to a scratch dir):
    python ml/mri/preprocess_registered.py --pilot 5 \
        --data-root ~/Downloads/7th_sem/PP-2/data \
        --oasis1-dir ~/Downloads/7th_sem/PP-2/external/oasis1 \
        --out-dir /tmp/e1_pilot
"""

from __future__ import annotations

import argparse
import glob
import subprocess
import tempfile
import time
from pathlib import Path

import numpy as np

N_SLICES = 16
Z_MM_LO, Z_MM_HI = -30.0, 45.0  # MNI axial band: hippocampus -> ventricles
SYNTHSTRIP_IMAGE = "freesurfer/synthstrip:latest"


def _template():
    """MNI152 1mm brain (template masked to brain, matching stripped inputs)."""
    import ants
    from nilearn.datasets import load_mni152_brain_mask, load_mni152_template

    t = load_mni152_template(resolution=1)
    m = load_mni152_brain_mask(resolution=1)
    import nibabel as nib

    data = np.asarray(t.dataobj, dtype=np.float32) * np.asarray(m.dataobj)
    brain = nib.Nifti1Image(data, t.affine)
    with tempfile.NamedTemporaryFile(suffix=".nii.gz", delete=False) as f:
        brain.to_filename(f.name)
        img = ants.image_read(f.name)
    return img, t.affine


def _docker_up() -> bool:
    try:
        return subprocess.run(["docker", "info"], capture_output=True,
                              timeout=10).returncode == 0
    except Exception:
        return False


_USE_SYNTHSTRIP: bool | None = None


def _skull_strip(in_path: Path, out_path: Path) -> str:
    """SynthStrip (Docker) when available, else deepbet (pip, PyTorch).

    Both are deep-learning strippers robust across field strengths; deepbet is
    the no-Docker fallback so the pipeline runs on a bare Mac.
    """
    global _USE_SYNTHSTRIP
    if _USE_SYNTHSTRIP is None:
        _USE_SYNTHSTRIP = _docker_up()
        print(f"skull-strip backend: {'synthstrip (docker)' if _USE_SYNTHSTRIP else 'deepbet'}")
    if _USE_SYNTHSTRIP:
        workdir = in_path.parent
        res = subprocess.run(
            ["docker", "run", "--rm", "-v", f"{workdir}:/data", SYNTHSTRIP_IMAGE,
             "-i", f"/data/{in_path.name}", "-o", f"/data/{out_path.name}"],
            capture_output=True, text=True)
        if res.returncode != 0 or not out_path.exists():
            raise RuntimeError(f"synthstrip failed: {res.stderr[-400:]}")
        return "synthstrip"
    from deepbet import run_bet

    run_bet([str(in_path)], [str(out_path)], threshold=0.5, n_dilate=0, no_gpu=True)
    if not out_path.exists():
        raise RuntimeError("deepbet produced no output")
    return "deepbet"


def process_volume(vol_path: Path, session: str, out_slices: Path,
                   template, template_affine) -> dict:
    """One volume -> N_SLICES registered axial PNGs. Returns stage timings."""
    import ants
    from PIL import Image

    t: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        if vol_path.suffix == ".img":
            # OASIS-1 ANALYZE headers misreport orientation (verified visually
            # in external_validate.py: the true superior axis is axis 1 of the
            # "canonical" volume, anterior at low axis-0 indices). Rebuild a
            # NIfTI with axes relabeled to approximate RAS so the affine
            # registration starts near the right orientation.
            import nibabel as nib

            data = np.asarray(nib.as_closest_canonical(nib.load(str(vol_path))).dataobj,
                              dtype=np.float32)
            while data.ndim > 3:
                data = data[..., 0]
            data = np.transpose(data, (2, 0, 1))[:, ::-1, :]  # z=true axial, +y=anterior
            fixed_path = tmp / "fixed_orient.nii.gz"
            nib.Nifti1Image(data, np.eye(4)).to_filename(str(fixed_path))
            img = ants.image_read(str(fixed_path))
        else:
            img = ants.image_read(str(vol_path))
        n4 = ants.n4_bias_field_correction(img)
        t["n4"] = time.time() - t0

        t0 = time.time()
        n4_path, stripped_path = tmp / "n4.nii.gz", tmp / "stripped.nii.gz"
        ants.image_write(n4, str(n4_path))
        _skull_strip(n4_path, stripped_path)
        stripped = ants.image_read(str(stripped_path))
        t["strip"] = time.time() - t0

        t0 = time.time()
        reg = ants.registration(fixed=template, moving=stripped,
                                type_of_transform="Affine")
        warped = reg["warpedmovout"]
        t["register"] = time.time() - t0

    data = warped.numpy()  # on the template grid (RAS, 1mm)
    inv = np.linalg.inv(template_affine)
    sdir = out_slices / session
    sdir.mkdir(parents=True, exist_ok=True)
    for i, z_mm in enumerate(np.linspace(Z_MM_LO, Z_MM_HI, N_SLICES)):
        k = int(round((inv @ np.array([0.0, 0.0, z_mm, 1.0]))[2]))
        k = int(np.clip(k, 0, data.shape[2] - 1))
        sl = data[:, :, k].T[::-1]  # anterior-up, radiological
        nz = sl[sl > 0]
        lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if nz.size else (0, 1)
        sl = np.clip((sl - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0
        Image.fromarray(sl.astype(np.uint8), mode="L").save(
            sdir / f"{session}_sl{i:02d}.png")
    return t


def pilot_volumes(data_root: Path, oasis1_dir: Path, n: int) -> list[tuple[str, Path]]:
    """n volumes mixed from both sources (3 OASIS-3 : 2 OASIS-1 for n=5)."""
    vols: list[tuple[str, Path]] = []
    sessions = sorted((data_root / "oasis3_nii").iterdir())
    for sess in sessions:
        hits = sorted(glob.glob(str(sess / "**" / "*T1w.nii.gz"), recursive=True))
        if hits:
            vols.append((sess.name, Path(hits[-1])))
        if len(vols) >= (n + 1) // 2 + 1:
            break
    for sess in sorted(oasis1_dir.glob("disc3/OAS1_*_MR1")):
        hits = sorted(sess.glob("PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.img"))
        if hits:
            vols.append((sess.name, hits[0]))
        if len(vols) >= n:
            break
    return vols[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True)
    ap.add_argument("--oasis1-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--pilot", type=int, default=0, help="process only N volumes, timed")
    args = ap.parse_args()

    out_slices = Path(args.out_dir).expanduser() / "slices"
    template, template_affine = _template()

    vols = pilot_volumes(Path(args.data_root).expanduser(),
                         Path(args.oasis1_dir).expanduser(),
                         args.pilot or 5)
    print(f"processing {len(vols)} volumes...")
    totals = []
    for session, path in vols:
        t0 = time.time()
        stages = process_volume(path, session, out_slices, template, template_affine)
        wall = time.time() - t0
        totals.append(wall)
        print(f"{session}: {wall:.1f}s  "
              + "  ".join(f"{k}={v:.1f}s" for k, v in stages.items()))
    print(f"\nmean {np.mean(totals):.1f}s/volume -> "
          f"369 volumes ≈ {369 * np.mean(totals) / 3600:.1f}h single-core "
          f"(~{369 * np.mean(totals) / 3600 / 4:.1f}h with 4 workers)")


if __name__ == "__main__":
    main()
