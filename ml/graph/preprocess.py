"""OASIS-3 rs-fMRI (+ optional precomputed DTI connectomes) -> per-session FC/SC matrices.

Labels: processed/graph_session_labels.csv (build_session_list.py) if present,
else processed/session_labels.csv.

    python ml/graph/preprocess.py --data-root ~/Downloads/7th_sem/PP-2/data \
        [--sc-dir /path/to/connectomes] [--atlas harvard_oxford_cort_2mm] [--workers 4]

Per session (resting-state BOLD from download_oasis_func.py):
  1. ANTs rigid motion correction; mean framewise displacement (FD) recorded.
     Sessions with mean FD > MAX_MEAN_FD_MM are excluded and listed — motion
     correlates with age and impairment and would otherwise leak into FC.
  2. Mean BOLD affinely registered to the MNI152 template (the same template
     ml/mri/preprocess_registered.py uses) and the transform applied to every
     volume, on a 2 mm grid.
  3. ml/graph/connectivity.fc_from_bold: atlas region time series with
     detrend, 0.01-0.1 Hz band-pass, z-scoring, WM/CSF confound regression,
     then Pearson FC. This is the SAME function the backend runs on uploads
     (which must already be motion-corrected and in MNI space), so training
     and inference share every step after (2).

Outputs <data-root>/processed_graph/<session>/fc.npy (+ sc.npy) and
processed_graph/manifest.csv (session, subject_id, label, fc, sc, mean_fd,
atlas, n_regions). Labels come from processed/session_labels.csv — the CDR
nearest the scan date, already mapped to 0 = CDR 0 (Normal), 1 = CDR 0.5 (MCI),
2 = CDR >= 1 (Dementia).

Structural connectivity (DTI) is NOT computed here: tractography is hours per
subject and lives outside Python. v1 takes precomputed matrices from --sc-dir
(<session>.npy or <session>.csv, N x N in this atlas's region order). Recipe
with MRtrix3, per session, after registering the atlas to the subject's DWI:

    dwi2response dhollander dwi.mif wm.txt gm.txt csf.txt
    dwi2fod msmt_csd dwi.mif wm.txt wmfod.mif gm.txt gm.mif csf.txt csf.mif
    tckgen wmfod.mif tracks.tck -seed_dynamic wmfod.mif -select 5M
    tcksift2 tracks.tck wmfod.mif weights.txt
    tck2connectome tracks.tck atlas_in_dwi.nii.gz sc.csv \
        -tck_weights_in weights.txt -symmetric -zero_diagonal

Without --sc-dir the manifest has an empty sc column and the model trains on
FC only (graph_config.json records it).

Resumable: sessions whose fc.npy exists are skipped.
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # ml/ on the path

MAX_MEAN_FD_MM = 0.5
GRID_MM = 2.0


def find_bold(session_dir: Path) -> Path | None:
    hits = sorted(glob.glob(str(session_dir / "**" / "*bold.nii.gz"), recursive=True))
    rest = [h for h in hits if "rest" in Path(h).name.lower()]
    return Path((rest or hits)[0]) if hits else None


def load_sc(sc_dir: Path | None, session: str, n: int) -> np.ndarray | None:
    from graph.graph_build import load_matrix_bytes, validate_matrix

    if sc_dir is None:
        return None
    for suffix in (".npy", ".csv"):
        p = sc_dir / f"{session}{suffix}"
        if p.exists():
            return validate_matrix(load_matrix_bytes(p.read_bytes(), suffix), n)
    return None


def ants_to_nib(img):
    """In-memory ANTs -> nibabel (a NIfTI write/read of a 4-D series costs ~30 s)."""
    import nibabel as nib

    d = img.dimension
    aff = np.eye(4)
    aff[:3, :3] = np.array(img.direction).reshape(d, d)[:3, :3] * np.array(img.spacing[:3])
    aff[:3, 3] = np.array(img.origin[:3])
    aff = np.diag([-1, -1, 1, 1]) @ aff  # ITK physical space is LPS, NIfTI is RAS
    return nib.Nifti1Image(img.numpy().astype(np.float32), aff)


def mni_bold(bold_path: Path, timings: dict | None = None):
    """Motion-correct and register a BOLD series to MNI. Returns (nibabel img, mean FD)."""
    import ants
    import nibabel as nib
    from mri.preprocess_registered import template

    t = timings if timings is not None else {}
    tpl, _ = template()
    tpl2 = ants.resample_image(tpl, (GRID_MM, GRID_MM, GRID_MM), use_voxels=False, interp_type=0)
    bold = ants.image_read(str(bold_path))
    tr = float(nib.load(str(bold_path)).header.get_zooms()[3])

    t0 = time.time()
    # BOLDRigid is ANTs' mode for fMRI realignment; plain "Rigid" gave
    # implausible FD on a series with known sub-voxel shifts.
    mc = ants.motion_correction(bold, type_of_transform="BOLDRigid")
    fd = np.asarray(mc.get("FD", [0.0]), dtype=float)
    t["motion"] = time.time() - t0

    t0 = time.time()
    mean_bold = ants.get_average_of_timeseries(mc["motion_corrected"])
    reg = ants.registration(fixed=tpl2, moving=mean_bold, type_of_transform="Affine")
    warped = ants.apply_transforms(fixed=tpl2, moving=mc["motion_corrected"],
                                   transformlist=reg["fwdtransforms"], imagetype=3)
    img = ants_to_nib(warped)
    t["register"] = time.time() - t0
    img.header.set_zooms(img.header.get_zooms()[:3] + (tr,))
    return img, float(fd.mean())


def _one(job):
    session, bold_path, out_dir, atlas_name, sc_dir = job
    from graph.atlas import load_atlas
    from graph.connectivity import fc_from_bold

    try:
        labels_img, names = load_atlas(atlas_name)
        timings: dict = {}
        img, mean_fd = mni_bold(bold_path, timings)
        if mean_fd > MAX_MEAN_FD_MM:
            return session, None, mean_fd, f"excluded: mean FD {mean_fd:.2f} mm > {MAX_MEAN_FD_MM}"
        t0 = time.time()
        fc = fc_from_bold(img, labels_img)
        timings["fc"] = time.time() - t0
        sc = load_sc(sc_dir, session, fc.shape[0])
        d = out_dir / session
        d.mkdir(parents=True, exist_ok=True)
        np.save(d / "fc.npy", fc.astype(np.float32))
        if sc is not None:
            np.save(d / "sc.npy", sc.astype(np.float32))
        (d / "meta.txt").write_text(f"mean_fd={mean_fd:.4f}\natlas={atlas_name}\nregions={len(names)}\n"
                                    + "".join(f"t_{k}={v:.1f}\n" for k, v in timings.items()))
        return session, fc.shape[0], mean_fd, None
    except Exception as exc:  # one bad scan must not stop a long batch
        return session, None, None, f"{type(exc).__name__}: {exc}"


def main() -> None:
    from concurrent.futures import ProcessPoolExecutor

    from graph.atlas import DEFAULT_ATLAS

    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir with oasis3_func/ and processed/session_labels.csv")
    ap.add_argument("--sc-dir", default=None, help="precomputed DTI connectomes <session>.npy|.csv")
    ap.add_argument("--atlas", default=DEFAULT_ATLAS)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    root = Path(args.data_root).expanduser()
    func_root = root / "oasis3_func"
    out_dir = root / "processed_graph"
    out_dir.mkdir(exist_ok=True)
    sc_dir = Path(args.sc_dir).expanduser() if args.sc_dir else None

    labels, label_name = load_labels(root)
    print(f"labels from {label_name}: {len(labels)} sessions")

    jobs, no_label = [], 0
    for sess_dir in sorted(p for p in func_root.iterdir() if p.is_dir()):
        sess = sess_dir.name
        if sess not in labels:
            no_label += 1
            continue
        if (out_dir / sess / "fc.npy").exists():
            continue
        bold = find_bold(sess_dir)
        if bold is not None:
            jobs.append((sess, bold, out_dir, args.atlas, sc_dir))
    if args.limit:
        jobs = jobs[: args.limit]
    print(f"{len(jobs)} sessions to process ({no_label} without a CDR label skipped)")
    from graph.atlas import load_atlas
    import json
    # Region order = graph node order; train_graph.py copies these names into
    # graph_config.json so explanations can name the regions.
    (out_dir / "regions.json").write_text(json.dumps(load_atlas(args.atlas)[1], indent=1))

    t0, problems = time.time(), []
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as pool:
        for n, (sess, regions, fd, err) in enumerate(pool.map(_one, jobs), 1):
            if err:
                problems.append((sess, err))
                print(f"[{n}/{len(jobs)}] {sess}: {err}")
            else:
                print(f"[{n}/{len(jobs)}] {sess}: {regions} regions, mean FD {fd:.2f} mm "
                      f"({(time.time() - t0) / n:.0f}s/session)", flush=True)

    write_manifest(out_dir, labels)
    print(f"{len(problems)} problems this run")


def load_labels(root: Path) -> tuple[dict[str, int], str]:
    """graph_session_labels.csv (build_session_list.py) if present, else session_labels.csv."""
    label_file = root / "processed" / "graph_session_labels.csv"
    if not label_file.exists():
        label_file = root / "processed" / "session_labels.csv"
    with open(label_file) as f:
        return {row["mr_session"]: int(row["label"]) for row in csv.DictReader(f)}, label_file.name


def write_manifest(out_dir: Path, labels: dict[str, int]) -> list[dict]:
    """manifest.csv over every processed session (this run or earlier ones)."""
    rows = []
    for d in sorted(p for p in out_dir.iterdir() if (p / "fc.npy").exists() and p.name in labels):
        meta = dict(line.split("=", 1) for line in (d / "meta.txt").read_text().split())
        rows.append({
            "session": d.name, "subject_id": d.name.split("_")[0], "label": labels[d.name],
            "fc": f"{d.name}/fc.npy", "sc": f"{d.name}/sc.npy" if (d / "sc.npy").exists() else "",
            "mean_fd": meta.get("mean_fd", ""), "atlas": meta.get("atlas", ""),
            "n_regions": meta.get("regions", ""),
        })
    with open(out_dir / "manifest.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]) if rows else ["session"])
        w.writeheader()
        w.writerows(rows)
    counts = {c: sum(r["label"] == c for r in rows) for c in (0, 1, 2)}
    print(f"\n{len(rows)} sessions in {out_dir / 'manifest.csv'} "
          f"(Normal {counts[0]}, MCI {counts[1]}, Dementia {counts[2]}; "
          f"{sum(bool(r['sc']) for r in rows)} with SC)")
    return rows


if __name__ == "__main__":
    main()
