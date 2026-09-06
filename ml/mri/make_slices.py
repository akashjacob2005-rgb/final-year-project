"""Regenerate the training slice export from raw OASIS-3 NIfTI volumes.

Reads session_labels.csv and every downloaded session under oasis3_nii/,
extracts preprocess.N_SLICES axial slices per session (last T1w run, matching
the original export's convention), and writes:

    <data-root>/processed_v2/slices/<session>/<session>_slNN.png
    <data-root>/processed_v2/manifest_v2.csv   (path,label,subject_id)

`label` stays the 3-class CDR label (0/1/2); binary grouping happens at
training time so one export serves both label modes.

Idempotent: sessions whose slice directory already holds N_SLICES files are
skipped, so re-running after downloading more sessions only processes the new
ones.

Usage:
    python ml/mri/make_slices.py --data-root /path/to/data
"""

from __future__ import annotations

import argparse
import csv
import glob
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from preprocess import N_SLICES, extract_slices  # noqa: E402


def last_t1w(session_dir: Path) -> Path | None:
    """The original export used the last run when a session has several."""
    hits = sorted(glob.glob(str(session_dir / "**" / "*T1w.nii.gz"), recursive=True))
    return Path(hits[-1]) if hits else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-root", required=True, help="dir containing oasis3_nii/ and processed/session_labels.csv")
    args = ap.parse_args()

    import nibabel as nib
    from PIL import Image

    root = Path(args.data_root)
    nii_root = root / "oasis3_nii"
    out_root = root / "processed_v2"
    slices_root = out_root / "slices"
    slices_root.mkdir(parents=True, exist_ok=True)

    labels: dict[str, int] = {}
    with open(root / "processed" / "session_labels.csv") as f:
        for row in csv.DictReader(f):
            labels[row["mr_session"]] = int(row["label"])

    rows, skipped, done, failed = [], 0, 0, 0
    sessions = sorted(p.name for p in nii_root.iterdir() if p.is_dir())
    for sess in sessions:
        label = labels.get(sess)
        if label is None:
            skipped += 1
            continue
        subject = sess.split("_")[0]
        sess_dir = slices_root / sess
        existing = sorted(sess_dir.glob("*.png"))
        if len(existing) != N_SLICES:
            nii = last_t1w(nii_root / sess)
            if nii is None:
                skipped += 1
                continue
            try:
                img = nib.as_closest_canonical(nib.load(nii))  # RAS, like inference
                volume = np.asarray(img.dataobj, dtype=np.float32)
                slices = extract_slices(volume)
            except Exception as exc:
                print(f"FAILED {sess}: {exc}")
                failed += 1
                continue
            sess_dir.mkdir(parents=True, exist_ok=True)
            for old in existing:
                old.unlink()
            for i, sl in enumerate(slices):
                Image.fromarray(sl, mode="L").save(sess_dir / f"{sess}_sl{i:02d}.png")
        for i in range(N_SLICES):
            rows.append((f"slices/{sess}/{sess}_sl{i:02d}.png", label, subject))
        done += 1

    with open(out_root / "manifest_v2.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "subject_id"])
        w.writerows(rows)

    print(f"sessions exported: {done}  (skipped {skipped}, failed {failed})")
    print(f"slices in manifest: {len(rows)}")
    print(f"manifest: {out_root / 'manifest_v2.csv'}")


if __name__ == "__main__":
    main()
