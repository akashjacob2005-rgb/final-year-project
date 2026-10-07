"""Export OASIS-1 training slices and merge them with the OASIS-3 export.

Phase-B-lite data expansion: OASIS-1 cross-sectional subjects (1.5T, ANALYZE
format, direct public download) are added to the OASIS-3 training pool.
Discs 1-2 are RESERVED for external validation (ml/mri/external_validate.py
used disc1) and are never exported here — training on them would destroy the
held-out cross-scanner validation story.

Only CDR-assessed subjects are exported (binary screening needs labels; the
young controls without a CDR are skipped). Labels map onto the manifest's
3-class scheme exactly like the OASIS-3 export: CDR 0 -> 0, 0.5 -> 1, >=1 -> 2
(train_mri.py --label-mode binary then folds 1 and 2 together).

Slice geometry reuses the orientation verified visually for OASIS-1 in
external_validate.py: the true axial plane is axis 1 of the canonicalised
SUBJ_111 volume, anterior-up as stored, same LO/HI fractions and 1-99
percentile normalisation as the OASIS-3 pipeline.

Usage (after external/oasis1/fetch_discs.sh has populated disc3..disc12):
    ml/mri/.venv/bin/python ml/mri/make_slices_oasis1.py \
        --oasis1-dir ~/Downloads/7th_sem/PP-2/external/oasis1 \
        --oasis3-export ~/Downloads/7th_sem/PP-2/data/processed_v2 \
        --out-dir ~/Downloads/7th_sem/PP-2/data/processed_v3
"""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np

HOLDOUT_DISCS = {"disc1", "disc2"}  # external-validation reserve, never train


def main() -> None:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from external_validate import load_oasis1_axial_slices, read_cdr
    from PIL import Image

    ap = argparse.ArgumentParser()
    ap.add_argument("--oasis1-dir", required=True)
    ap.add_argument("--oasis3-export", required=True)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    oasis1 = Path(args.oasis1_dir).expanduser()
    src3 = Path(args.oasis3_export).expanduser()
    out = Path(args.out_dir).expanduser()
    (out / "slices").mkdir(parents=True, exist_ok=True)

    # 1. carry the OASIS-3 export over unchanged (slices + manifest rows)
    rows: list[dict] = []
    with open(src3 / "manifest_v2.csv") as f:
        for row in csv.DictReader(f):
            rows.append(row)
    fieldnames = list(rows[0].keys())
    if "source" not in fieldnames:
        fieldnames.append("source")
    for row in rows:
        row.setdefault("source", "oasis3")
    if not (out / "slices" / rows[0]["path"].split("slices/")[-1].split("/")[0]).exists():
        print("copying OASIS-3 slices...")
        shutil.copytree(src3 / "slices", out / "slices", dirs_exist_ok=True)

    # 2. export OASIS-1 training discs
    n_subj, counts = 0, {0: 0, 1: 0, 2: 0}
    discs = sorted(d for d in oasis1.glob("disc*") if d.is_dir()
                   and d.name not in HOLDOUT_DISCS)
    print(f"OASIS-1 training discs: {[d.name for d in discs]} "
          f"(holdout, untouched: {sorted(HOLDOUT_DISCS)})")
    for disc in discs:
        for sess_dir in sorted(disc.glob("OAS1_*_MR1")):
            txts = list(sess_dir.glob("OAS1_*.txt"))
            imgs = list(sess_dir.glob("PROCESSED/MPRAGE/SUBJ_111/*_sbj_111.img"))
            if not txts or not imgs:
                continue
            cdr = read_cdr(txts[0])
            if cdr is None:
                continue  # no CDR assessed (young control) -> unlabeled, skip
            label = 0 if cdr == 0 else (1 if cdr == 0.5 else 2)
            session = sess_dir.name                      # OAS1_0042_MR1
            subject = "_".join(session.split("_")[:2])   # OAS1_0042
            try:
                slices = load_oasis1_axial_slices(imgs[0])
            except Exception as exc:
                print(f"skip {session}: {exc}")
                continue
            sdir = out / "slices" / session
            sdir.mkdir(parents=True, exist_ok=True)
            for i, sl in enumerate(slices):
                Image.fromarray(np.asarray(sl), mode="L").save(
                    sdir / f"slice_{i:02d}.png")
                rows.append({**{k: "" for k in fieldnames},
                             "path": f"slices/{session}/slice_{i:02d}.png",
                             "label": str(label),
                             "subject_id": subject,
                             "source": "oasis1"})
            n_subj += 1
            counts[label] += 1

    with open(out / "manifest_v2.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)

    sessions = {r["path"].split("/")[1] for r in rows}
    print(f"\nOASIS-1 added: {n_subj} subjects "
          f"(CDR0={counts[0]}, CDR0.5={counts[1]}, CDR>=1={counts[2]})")
    print(f"combined export: {len(sessions)} sessions, {len(rows)} slices")
    print(f"manifest: {out / 'manifest_v2.csv'}")


if __name__ == "__main__":
    main()
