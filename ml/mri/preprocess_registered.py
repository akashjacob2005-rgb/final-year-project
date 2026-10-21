"""E1: registered + skull-stripped slice extraction (experiment pipeline).

The deployed preprocessing slices raw volumes at fixed *percentages* of the
array, so "slice 8" is different anatomy in different heads. This pipeline
makes slice indices anatomical:

    N4 bias correction (ANTsPy)
      -> skull-strip (SynthStrip via Docker: robust across 1.5T/3T)
      -> 12-dof affine registration to the MNI152 1mm template (ANTsPy)
      -> slices at FIXED MNI z-coordinates (default 16 axial, z = -30..+45mm,
         hippocampus level up to the ventricles)

Two planes, both at fixed template coordinates:
    axial    16 slices, MNI z = -30..+45 mm (hippocampus -> ventricles)
    coronal  16 slices, MNI y = -8..-40 mm  (hippocampal head -> tail: the
             structure that atrophies first in Alzheimer's, which unregistered
             percentage-slices only reach by luck)

Full export (Step 2 of docs/MRI_EXPERIMENTS.md) — every session of an
existing export (e.g. processed_v3), resumable, parallel:
    python ml/mri/preprocess_registered.py --export \
        --manifest-from ~/Downloads/7th_sem/PP-2/data/processed_v3 \
        --data-root ~/Downloads/7th_sem/PP-2/data \
        --oasis1-dir ~/Downloads/7th_sem/PP-2/external/oasis1 \
        --out-dir ~/Downloads/7th_sem/PP-2/data/processed_v4 --workers 4
writes slices/<session>/<session>_{ax,co}NN.png, the registered brains in
registered/<session>.nii.gz (so new slice choices never need re-registration),
and three manifests: manifest_axial.csv, manifest_coronal.csv,
manifest_both.csv (train with train_mri.py --manifest <file>).

The backend serves the same function (registered_slices_from_path) when
MRI_PREPROCESS=registered, so training and inference stay identical.

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
Y_MM_LO, Y_MM_HI = -8.0, -40.0  # MNI coronal band: hippocampal head -> tail
PLANES = ("axial", "coronal")
PLANE_TAG = {"axial": "ax", "coronal": "co"}
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


def _load_ants(vol_path: Path, tmp: Path):
    import ants

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
        return ants.image_read(str(fixed_path))
    return ants.image_read(str(vol_path))


def register_volume(vol_path: Path, template) -> tuple[object, dict]:
    """N4 -> skull-strip -> affine to the template. Returns (warped ANTs image
    on the template grid, stage timings)."""
    import ants

    t: dict = {}
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        t0 = time.time()
        n4 = ants.n4_bias_field_correction(_load_ants(vol_path, tmp))
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
        t["register"] = time.time() - t0
    return reg["warpedmovout"], t


def _norm_uint8(sl: np.ndarray) -> np.ndarray:
    nz = sl[sl > 0]
    lo, hi = (np.percentile(nz, 1), np.percentile(nz, 99)) if nz.size else (0, 1)
    return (np.clip((sl - lo) / (hi - lo + 1e-8), 0.0, 1.0) * 255.0).astype(np.uint8)


def slices_from_registered(data: np.ndarray, template_affine: np.ndarray,
                           planes: tuple[str, ...] = ("axial",)) -> list[np.ndarray]:
    """Fixed-MNI-coordinate slices from a volume on the template grid (RAS 1mm).
    Axial first, then coronal — the order the manifests and backend use."""
    inv = np.linalg.inv(template_affine)
    out: list[np.ndarray] = []
    if "axial" in planes:
        for z_mm in np.linspace(Z_MM_LO, Z_MM_HI, N_SLICES):
            k = int(np.clip(round((inv @ np.array([0.0, 0.0, z_mm, 1.0]))[2]), 0, data.shape[2] - 1))
            out.append(_norm_uint8(data[:, :, k].T[::-1]))  # anterior-up, radiological
    if "coronal" in planes:
        for y_mm in np.linspace(Y_MM_LO, Y_MM_HI, N_SLICES):
            j = int(np.clip(round((inv @ np.array([0.0, y_mm, 0.0, 1.0]))[1]), 0, data.shape[1] - 1))
            out.append(_norm_uint8(data[:, j, :].T[::-1]))  # superior-up
    return out


_TEMPLATE = None


def template():
    """Cached (ANTs template image, affine)."""
    global _TEMPLATE
    if _TEMPLATE is None:
        _TEMPLATE = _template()
    return _TEMPLATE


def registered_slices_from_path(vol_path: Path, planes: tuple[str, ...] = ("axial",)) -> list[np.ndarray]:
    """The serving entry point: one volume file -> registered uint8 slices."""
    tpl, aff = template()
    warped, _ = register_volume(vol_path, tpl)
    return slices_from_registered(warped.numpy(), aff, planes)


def process_volume(vol_path: Path, session: str, out_dir: Path,
                   template_img, template_affine, keep_volume: bool = True) -> dict:
    """One volume -> registered axial + coronal PNGs (+ registered brain)."""
    from PIL import Image

    warped, t = register_volume(vol_path, template_img)
    if keep_volume:
        (out_dir / "registered").mkdir(parents=True, exist_ok=True)
        warped.to_file(str(out_dir / "registered" / f"{session}.nii.gz"))
    data = warped.numpy()
    sdir = out_dir / "slices" / session
    sdir.mkdir(parents=True, exist_ok=True)
    sl = slices_from_registered(data, template_affine, PLANES)
    for plane_i, plane in enumerate(PLANES):
        for i in range(N_SLICES):
            Image.fromarray(sl[plane_i * N_SLICES + i], mode="L").save(
                sdir / f"{session}_{PLANE_TAG[plane]}{i:02d}.png")
    return t


def source_volume(session: str, data_root: Path, oasis1_dir: Path) -> Path | None:
    """Raw volume behind a processed_v3 session (same choice as the v3 export:
    last T1w run for OASIS-3, SUBJ_111 for OASIS-1, holdout discs never)."""
    if session.upper().startswith("OAS1"):
        for disc in sorted(oasis1_dir.glob("disc*")):
            if disc.name in {"disc1", "disc2"}:
                continue
            hits = sorted((disc / session).glob("PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.img"))
            if hits:
                return hits[0]
        return None
    hits = sorted(glob.glob(str(data_root / "oasis3_nii" / session / "**" / "*T1w.nii.gz"), recursive=True))
    return Path(hits[-1]) if hits else None


def _export_one(job):
    session, vol, out_dir = job
    try:
        tpl, aff = template()
        t = process_volume(vol, session, out_dir, tpl, aff)
        return session, None, sum(t.values())
    except Exception as exc:  # one bad scan must not kill a 2-hour batch
        return session, f"{type(exc).__name__}: {exc}", 0.0


def export_all(manifest_from: Path, data_root: Path, oasis1_dir: Path, out_dir: Path,
               workers: int, limit: int = 0) -> None:
    import csv
    from concurrent.futures import ProcessPoolExecutor

    manifest = manifest_from / "manifest_v2.csv"
    with open(manifest) as f:
        rows = list(csv.DictReader(f))
    sessions: dict[str, dict] = {}
    for r in rows:
        sess = r["path"].replace("\\", "/").split("slices/")[-1].split("/")[0]
        sessions.setdefault(sess, {"label": r["label"], "subject_id": r["subject_id"],
                                   "source": r.get("source") or ("oasis1" if sess.upper().startswith("OAS1") else "oasis3")})
    out_dir.mkdir(parents=True, exist_ok=True)

    def done(sess: str) -> bool:
        return len(list((out_dir / "slices" / sess).glob("*.png"))) == N_SLICES * len(PLANES)

    jobs, missing = [], []
    for sess in sorted(sessions):
        if done(sess):
            continue
        vol = source_volume(sess, data_root, oasis1_dir)
        if vol is None:
            missing.append(sess)
            continue
        jobs.append((sess, vol, out_dir))
    if limit:
        jobs = jobs[:limit]
    print(f"{len(sessions)} sessions in {manifest}; {len(jobs)} to register, "
          f"{len(sessions) - len(jobs) - len(missing)} already done, {len(missing)} without a source volume")
    if missing:
        print("  no source volume for:", ", ".join(missing[:10]), "..." if len(missing) > 10 else "")

    t0, failed = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, workers)) as pool:
        for n, (sess, err, secs) in enumerate(pool.map(_export_one, jobs), 1):
            if err:
                failed.append((sess, err))
                print(f"[{n}/{len(jobs)}] FAILED {sess}: {err}")
            else:
                eta = (time.time() - t0) / n * (len(jobs) - n) / 60
                print(f"[{n}/{len(jobs)}] {sess} {secs:.0f}s  (eta {eta:.0f} min)", flush=True)

    # manifests over every session that finished (this run or earlier ones)
    fields = ["path", "label", "subject_id", "source"]
    planesets = {"axial": ("axial",), "coronal": ("coronal",), "both": PLANES}
    for name, planes in planesets.items():
        out_rows = []
        for sess, meta in sorted(sessions.items()):
            if not done(sess):
                continue
            for plane in planes:
                for i in range(N_SLICES):
                    out_rows.append({"path": f"slices/{sess}/{sess}_{PLANE_TAG[plane]}{i:02d}.png", **meta})
        with open(out_dir / f"manifest_{name}.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(out_rows)
    n_done = sum(done(s) for s in sessions)
    print(f"\nregistered export: {n_done}/{len(sessions)} sessions, {len(failed)} failed this run")
    print(f"manifests: {out_dir}/manifest_{{axial,coronal,both}}.csv")
    if failed:
        print("re-run the same command to retry the failures (finished sessions are skipped)")


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
    ap.add_argument("--export", action="store_true",
                    help="register EVERY session of --manifest-from into --out-dir (resumable)")
    ap.add_argument("--manifest-from", default=None, help="existing export dir (e.g. processed_v3)")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0, help="export at most N new sessions (testing)")
    args = ap.parse_args()

    if args.export:
        if not args.manifest_from:
            ap.error("--export needs --manifest-from")
        export_all(Path(args.manifest_from).expanduser(), Path(args.data_root).expanduser(),
                   Path(args.oasis1_dir).expanduser(), Path(args.out_dir).expanduser(),
                   args.workers, args.limit)
        return

    out_root = Path(args.out_dir).expanduser()
    template_img, template_affine = template()

    vols = pilot_volumes(Path(args.data_root).expanduser(),
                         Path(args.oasis1_dir).expanduser(),
                         args.pilot or 5)
    print(f"processing {len(vols)} volumes...")
    totals = []
    for session, path in vols:
        t0 = time.time()
        stages = process_volume(path, session, out_root, template_img, template_affine)
        wall = time.time() - t0
        totals.append(wall)
        print(f"{session}: {wall:.1f}s  "
              + "  ".join(f"{k}={v:.1f}s" for k, v in stages.items()))
    print(f"\nmean {np.mean(totals):.1f}s/volume -> "
          f"369 volumes ≈ {369 * np.mean(totals) / 3600:.1f}h single-core "
          f"(~{369 * np.mean(totals) / 3600 / 4:.1f}h with 4 workers)")


if __name__ == "__main__":
    main()
